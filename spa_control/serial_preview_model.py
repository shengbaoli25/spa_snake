"""Trainable, offline SPA predictor. No robot connections or pretrained weights.

The image input must be real 576-D MobileNetV3-Small features supplied by the
caller, with provenance and host receipt times. This module does not download or
claim a pretrained visual extractor. Timestamps must use one host monotonic
clock; camera exposure clocks require independent calibration. Historical
records are accepted only for supervised response pretraining, not PPO.

Four-drive recent inputs and two-drive historical inputs are normalized by a
caller-owned, training-only frozen normalization. Historical input columns 0:2,
2:4, 4:6 represent current, speed, and commanded values for the two front
drives; remaining dimensions may carry validity/age information. Source must
record actual sample validity and complete-packet receipt times.
Each candidate contains four signed projected commands divided by 500 followed
by four applied signed residuals divided by 5 in verified structural order.
The quality vector must contain feature validity, age, previous residual,
geometry, nominal/target schedule, hold time and remaining episode time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    recent_dim: int
    front_dim: int
    quality_dim: int
    candidate_dim: int = 8
    windows: int = 8
    width_s: float = .5
    recent_s: float = .5

    def validate(self):
        if self.recent_dim < 1 or self.front_dim < 6 or self.quality_dim < 1 or self.candidate_dim != 8 or self.windows != 8 or self.width_s <= 0 or self.recent_s <= 0:
            raise ValueError("invalid SPA feature dimensions or window timing")


def _mlp(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, outputs))


def _gather_sequences(gru, values, valid):
    """Compacted causal samples; zero output for empty input, no NaN padding use."""
    outputs = []
    for row, mask in zip(values, valid):
        chosen = row[mask]
        if chosen.shape[0] == 0:
            outputs.append(row.new_zeros(gru.hidden_size))
        else:
            if not bool(torch.isfinite(chosen).all()):
                raise ValueError("valid input contains nonfinite values")
            outputs.append(gru(chosen.unsqueeze(0))[1][0, 0])
    return torch.stack(outputs)


def _causal_mask(received, valid, cutoff, low=None, high=None):
    if received.shape != valid.shape or received.shape[0] != cutoff.shape[0]:
        raise ValueError("receipt and validity dimensions mismatch")
    cutoff = cutoff.unsqueeze(-1)
    result = valid.bool() & torch.isfinite(received) & (received <= cutoff)
    if low is not None:
        result &= received > low.unsqueeze(-1)
    if high is not None:
        result &= received <= high.unsqueeze(-1)
    return result


def _ordered(received, mask):
    for times, valid in zip(received, mask):
        values = times[valid]
        if values.numel() > 1 and bool((values[1:] < values[:-1]).any()):
            raise ValueError("samples must be ordered by host receipt time")


def ordered_response(raw):
    """Three ordered quantiles for displacement, load, occupancy; P(stop)."""
    if raw.shape[-1] != 10:
        raise ValueError("response has ten raw features")
    output = []
    for i in range(3):
        mu, lower, upper = raw[..., 3*i:3*i+3].unbind(-1)
        quantiles = torch.stack((mu - F.softplus(lower), mu, mu + F.softplus(upper)), dim=-1)
        if i == 1:
            quantiles = F.softplus(quantiles)
        elif i == 2:
            quantiles = torch.sigmoid(quantiles)
        output.append(quantiles)
    return torch.cat(output + [torch.sigmoid(raw[..., 9:10])], dim=-1)


class SerialPreviewModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.visual = _mlp(576, 32, 32)
        self.recent_gru = nn.GRU(config.recent_dim, 32, batch_first=True)
        self.front_gru = nn.GRU(config.front_dim, 32, batch_first=True)
        # Includes b, front history, 8 command descriptors, window age/coverage.
        self.window_map = _mlp(32 + 32 + 8 + 2, 64, 32)
        self.selector = _mlp(32 + 32 + config.quality_dim + 32 + 2, 64, 1)
        self.null_selector = _mlp(32 + 32 + config.quality_dim, 64, 1)
        self.reference = _mlp(32 + 32 + config.quality_dim + 8, 64, 10)
        self.correction = _mlp(32 + 32 + config.quality_dim + 8, 64, 10)
        self.phase = "untrained"

    def set_phase(self, phase: str):
        """Stage 1: train current reference; stage 2: freeze it, train history."""
        if phase not in ("reference", "history", "frozen"):
            raise ValueError("phase must be reference, history or frozen")
        reference_modules = (self.visual, self.recent_gru, self.reference)
        history_modules = (self.front_gru, self.window_map, self.selector,
                           self.null_selector, self.correction)
        for module in reference_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(phase == "reference")
        for module in history_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(phase == "history")
        self.phase = phase

    def forward(self, *, visual, visual_times, visual_valid, recent, recent_times,
                recent_valid, front, front_times, front_valid, cutoff, quality,
                candidates, support, front_coverage=None):
        cfg = self.config
        b = cutoff.shape[0]
        if cutoff.ndim != 1 or not bool(torch.isfinite(cutoff).all()):
            raise ValueError("invalid cutoff")
        if visual.ndim != 3 or visual.shape[0] != b or visual.shape[-1] != 576 or visual.shape[1] > 8 or visual_times.shape != visual.shape[:2] or visual_valid.shape != visual.shape[:2]:
            raise ValueError("expected up to eight 576-dimensional visual features")
        if recent.ndim != 3 or recent.shape[0] != b or recent.shape[1] < 1 or recent.shape[-1] != cfg.recent_dim or recent_times.shape != recent.shape[:2] or recent_valid.shape != recent.shape[:2]:
            raise ValueError("recent history dimensions mismatch")
        if front.ndim != 4 or front.shape[:2] != (b, cfg.windows) or front.shape[2] < 1 or front.shape[-1] != cfg.front_dim or front_times.shape != front.shape[:3] or front_valid.shape != front.shape[:3]:
            raise ValueError("older history dimensions mismatch")
        if quality.shape != (b, cfg.quality_dim) or candidates.shape != (b, 3, 8) or support.shape != (b, 3):
            raise ValueError("quality or projected candidate dimensions mismatch")
        if front_coverage is None:
            # No provenance for coverage: explicitly conservative, never infer
            # availability from sequence padding or sample counts.
            front_coverage = quality.new_zeros((b, cfg.windows))
        elif front_coverage.shape != (b, cfg.windows) or not bool(torch.isfinite(front_coverage).all()) or not bool(((front_coverage >= 0) & (front_coverage <= 1)).all()):
            raise ValueError("front_coverage must be finite [batch,8] within [0,1]")
        else:
            front_coverage = front_coverage.to(dtype=quality.dtype, device=quality.device)
        if not bool(torch.isfinite(quality).all() and torch.isfinite(candidates).all()) or not bool(support.bool().any(dim=1).all()):
            raise ValueError("invalid context or empty candidate support")
        if bool((candidates.abs() > 1.000001).any()):
            raise ValueError("candidate commands / 500 and residuals / 5 must be in [-1,1]")
        # visual pooling only complete packets available at decision cutoff.
        vm = _causal_mask(visual_times, visual_valid, cutoff)
        _ordered(visual_times, vm)
        pooled = []
        for row, mask in zip(visual, vm):
            selected = row[mask]
            if selected.shape[0] and not bool(torch.isfinite(selected).all()):
                raise ValueError("invalid visual feature")
            pooled.append(selected.mean(0) if selected.shape[0] else row.new_zeros(576))
        visual_embedding = self.visual(torch.stack(pooled))
        rm = _causal_mask(recent_times, recent_valid, cutoff, low=cutoff-cfg.recent_s, high=cutoff)
        _ordered(recent_times, rm)
        recent_embedding = _gather_sequences(self.recent_gru, recent, rm)
        context = torch.cat((visual_embedding, recent_embedding, quality), dim=-1)

        valid_windows, window_z, descriptors = [], [], []
        for k in range(1, cfg.windows + 1):
            upper = cutoff - cfg.recent_s - (k - 1) * cfg.width_s
            lower = cutoff - cfg.recent_s - k * cfg.width_s
            mask = _causal_mask(front_times[:, k-1], front_valid[:, k-1], cutoff, lower, upper)
            _ordered(front_times[:, k-1], mask)
            encoded = _gather_sequences(self.front_gru, front[:, k-1], mask)
            descriptor = []
            coverage = front_coverage[:, k-1:k]
            for sequence, accepted, row_coverage in zip(front[:, k-1], mask, coverage):
                commands = sequence[accepted, 4:6]
                if len(commands):
                    features = torch.stack((commands.mean(0), torch.sqrt(torch.mean(commands**2, dim=0)),
                                            commands[-1] - commands[0], row_coverage.expand(2)), dim=1).reshape(-1)
                else:
                    features = sequence.new_zeros(8)
                descriptor.append(features)
            descriptor = torch.stack(descriptor)
            age = cutoff.new_full((b, 1), cfg.recent_s + (k - .5) * cfg.width_s)
            metadata = torch.cat((age, coverage), dim=-1)
            z = self.window_map(torch.cat((visual_embedding, encoded, descriptor, metadata), -1))
            window_z.append(z)
            descriptors.append(metadata)
            valid_windows.append(mask.any(-1))
        z = torch.stack(window_z, dim=1)
        metadata = torch.stack(descriptors, dim=1)
        mask = torch.stack(valid_windows, dim=1)
        score_context = context.unsqueeze(1).expand(-1, cfg.windows, -1)
        logits = self.selector(torch.cat((score_context, z, metadata), -1)).squeeze(-1)
        logits = logits.masked_fill(~mask, -torch.inf)
        null_logit = self.null_selector(context)
        weights = torch.softmax(torch.cat((null_logit, logits), dim=1), dim=1)
        alpha0, alpha = weights[:, :1], weights[:, 1:]
        delta_z = z - visual_embedding.unsqueeze(1)
        zbar = torch.sum(alpha.unsqueeze(-1) * delta_z, dim=1)
        ctx = context.unsqueeze(1).expand(-1, 3, -1)
        raw_base = self.reference(torch.cat((ctx, candidates), dim=-1))
        gq = torch.cat((recent_embedding, quality), dim=-1)
        anchor_input = torch.cat((visual_embedding.unsqueeze(1).expand(-1, 3, -1),
                                  gq.unsqueeze(1).expand(-1, 3, -1), candidates), -1)
        anchor = self.correction(anchor_input)
        full_z = z.unsqueeze(2).expand(-1, -1, 3, -1)
        full_gq = gq[:, None, None, :].expand(-1, cfg.windows, 3, -1)
        full_candidates = candidates[:, None, :, :].expand(-1, cfg.windows, -1, -1)
        response = self.correction(torch.cat((full_z, full_gq, full_candidates), dim=-1))
        correction = torch.sum(alpha[:, :, None, None] * (response - anchor[:, None, :, :]), dim=1)
        raw = raw_base + correction
        prediction = ordered_response(raw)
        augmentation = torch.cat((context, zbar, alpha0,
            torch.cat((prediction, candidates, support.float().unsqueeze(-1)), dim=-1).flatten(start_dim=1)), dim=-1)
        return {"raw": raw, "base_raw": raw_base, "prediction": prediction,
                "correction": correction, "zbar": zbar, "null_weight": alpha0,
                "history_weights": alpha, "valid_windows": mask,
                "front_coverage": front_coverage,
                "visual_available": vm.any(-1), "recent_available": rm.any(-1),
                "augmentation": augmentation}

    def export_checkpoint(self, path):
        """Weights and config only; the caller must separately version data/normalization."""
        torch.save({"format": 1, "config": vars(self.config), "state_dict": self.state_dict(),
                    "phase": self.phase, "training_claim": "unknown; verify dataset manifest"}, Path(path))

    @classmethod
    def load_checkpoint(cls, path):
        record = torch.load(Path(path), map_location="cpu", weights_only=True)
        if record.get("format") != 1:
            raise ValueError("unknown checkpoint format")
        model = cls(ModelConfig(**record["config"]))
        model.load_state_dict(record["state_dict"], strict=True)
        if record["phase"] in ("reference", "history", "frozen"):
            model.set_phase(record["phase"])
        return model


def supervised_loss(raw, executed, targets, target_valid, scales):
    """Pinball loss for executed action labels plus BCE-with-logits for stop.

    `targets` [B,4] may contain NaN only where mask is false; scales for the
    three continuous targets must be fixed from the training split only.
    """
    if raw.ndim != 3 or raw.shape[1:] != (3, 10) or targets.shape != (len(raw), 4) or target_valid.shape != targets.shape or executed.shape != (len(raw),):
        raise ValueError("invalid supervised target dimensions")
    if not bool(((executed >= 0) & (executed < 3)).all()):
        raise ValueError("executed action index invalid")
    scales = torch.as_tensor(scales, dtype=raw.dtype, device=raw.device)
    if scales.shape != (3,) or not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all()):
        raise ValueError("invalid frozen training scales")
    mask = target_valid.bool()
    if not bool(mask.any()) or not bool(torch.isfinite(targets[mask]).all()):
        raise ValueError("no valid labels or nonfinite valid labels")
    chosen = raw[torch.arange(len(raw), device=raw.device), executed.long()]
    prediction = ordered_response(chosen)
    losses = []
    qs = raw.new_tensor([.1, .5, .9])
    for i in range(3):
        valid = mask[:, i]
        if bool(valid.any()):
            error = (targets[valid, i, None] - prediction[valid, 3*i:3*i+3]) / scales[i]
            losses.append(torch.maximum(qs * error, (qs - 1) * error).mean())
    valid = mask[:, 3]
    if bool(valid.any()):
        labels = targets[valid, 3]
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise ValueError("intervention labels must be binary")
        losses.append(F.binary_cross_entropy_with_logits(chosen[valid, 9], labels))
    return torch.stack(losses).mean()


def supervised_step(model, optimizer, batch, executed, targets, target_valid, scales):
    """One supervised step. Instantiate a fresh optimizer after set_phase()."""
    if model.phase not in ("reference", "history"):
        raise ValueError("select a supervised training phase first")
    trained = {id(p) for p in model.parameters() if p.requires_grad}
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    if trained != optimized:
        raise ValueError("optimizer must contain exactly the current phase's trainable parameters")
    if executed.ndim != 1 or len(executed) != len(batch['support']) or not bool(((executed >= 0) & (executed < 3)).all()):
        raise ValueError("invalid executed action indices")
    if not bool(batch['support'].bool()[torch.arange(len(executed), device=executed.device), executed.long()].all()):
        raise ValueError("executed action is outside stored candidate support")
    optimizer.zero_grad(set_to_none=True)
    output = model(**batch)
    raw = output["base_raw"] if model.phase == "reference" else output["raw"]
    loss = supervised_loss(raw, executed, targets, target_valid, scales)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
