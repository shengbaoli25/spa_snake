"""Two-drive offline serial-preview inference and synthetic demonstration.

No hardware access or actuator commands. Structural front/rear mapping must be
explicitly supplied; it is not established by local encoder direction.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch import nn
from .serial_preview_model import (_mlp, _causal_mask, _ordered, _gather_sequences,
                                   ordered_response, SerialPreviewModel)
from .serial_preview import ACTIONS, CategoricalPPO, measured_load_costs, measured_reward
from .serial_preview_pipeline import DrivePacket, PipelineScales, _snapshot, _sequence


@dataclass(frozen=True)
class TwoMotorConfig:
    quality_dim: int = 5
    recent_dim: int = 12
    front_dim: int = 6
    candidate_dim: int = 4
    windows: int = 8
    width_s: float = .5
    recent_s: float = .5

    def validate(self):
        if (self.quality_dim < 1 or self.recent_dim != 12 or self.front_dim != 6
            or self.candidate_dim != 4 or self.windows != 8 or self.width_s != .5 or self.recent_s != .5):
            raise ValueError("invalid two-drive dimensions or timing")


class TwoMotorModel(nn.Module):
    """Untrained variant of the same architecture, sized to two real drives."""
    def __init__(self, config: TwoMotorConfig):
        super().__init__()
        config.validate()
        self.config = config
        self.visual = _mlp(576, 32, 32)
        self.recent_gru = nn.GRU(12, 32, batch_first=True)
        self.front_gru = nn.GRU(6, 32, batch_first=True)
        self.window_map = _mlp(32 + 32 + 4 + 2, 64, 32)
        self.selector = _mlp(32 + 32 + config.quality_dim + 32 + 2, 64, 1)
        self.null_selector = _mlp(32 + 32 + config.quality_dim, 64, 1)
        self.reference = _mlp(32 + 32 + config.quality_dim + 4, 64, 10)
        self.correction = _mlp(32 + 32 + config.quality_dim + 4, 64, 10)
        self.phase = "untrained"

    set_phase = SerialPreviewModel.set_phase

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
        if quality.shape != (b, cfg.quality_dim) or candidates.shape != (b, 3, 4) or support.shape != (b, 3):
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
                commands = sequence[accepted, 2:3]
                if len(commands):
                    features = torch.stack((commands.mean(0), torch.sqrt(torch.mean(commands**2, dim=0)),
                                            commands[-1] - commands[0], row_coverage.expand(1)), dim=1).reshape(-1)
                else:
                    features = sequence.new_zeros(4)
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



def project_two(nominal, front_motor):
    """Candidate vectors use true M1..M4 slots; only rear gets a residual.

    Both tested electrical directions are -1. M3/M4 order is user-supplied;
    it cannot be established from encoder signs or a random model.
    """
    if type(front_motor) is not int or front_motor not in (3, 4):
        raise ValueError("front_motor must be 3 or 4")
    q = np.asarray(nominal, dtype=float)
    if q.shape != (2,) or not np.isfinite(q).all() or np.any(q < 0) or np.any(q > 500) or np.any(q != np.round(q)):
        raise ValueError("two nonnegative integer magnitudes in [0,500] required")
    rear_motor = 7 - front_motor
    vectors, features = [], []
    for action in ACTIONS:
        rear_q = int(np.clip(q[1] + action, 0, 500))
        values = np.zeros(4, dtype=int)
        values[front_motor - 1], values[rear_motor - 1] = -int(q[0]), -rear_q
        vectors.append(values)
        # True two-channel structural order, including only the actual applied residual.
        features.append([-q[0] / 500., -rear_q / 500., 0., -(rear_q-q[1]) / 5.])
    vectors = np.stack(vectors)
    support = np.array([not any(np.array_equal(vectors[i], vectors[j]) for j in range(i))
                        for i in range(3)], dtype=bool)
    return vectors, np.asarray(features, dtype=float), support


class TwoMotorPipeline:
    """Build causal observations from independently timestamped M3 and M4.

    Vision encoder must follow the existing encode_window interface. This is an
    offline research interface and never opens a port or sends a motor command.
    """
    def __init__(self, model, policy, visual_encoder, scales):
        scales.validate()
        if not isinstance(model.config, TwoMotorConfig) or policy.obs_dim != 142 + model.config.quality_dim:
            raise ValueError("two-motor model/policy dimension mismatch")
        self.model, self.policy, self.visual_encoder, self.scales = model, policy, visual_encoder, scales

    def assemble(self, *, frames, drives, cutoff_ns, quality, nominal, front_motor, require_visual=True):
        if type(cutoff_ns) is not int or cutoff_ns < 0:
            raise ValueError("invalid monotonic cutoff")
        q = np.asarray(quality, dtype=float)
        if q.shape != (self.model.config.quality_dim,) or not np.isfinite(q).all():
            raise ValueError("quality must be finite and complete")
        vectors, features, support = project_two(nominal, front_motor)
        front = front_motor - 1
        rear = 6 - front_motor
        eligible, future = [], 0
        for p in drives:
            if not isinstance(p, DrivePacket) or type(p.receive_ns) is not int or p.receive_ns < 0:
                raise ValueError("invalid drive packet")
            if p.receive_ns > cutoff_ns:
                future += 1
                continue
            if (type(p.channel) is not int or p.channel not in (2, 3) or
                type(p.command_time_ns) is not int or not 0 <= p.command_time_ns <= p.receive_ns):
                raise ValueError("invalid channel or command time")
            if p.valid and (not math.isfinite(p.current_raw) or not math.isfinite(p.speed_raw) or
                            type(p.sent_command) is not int or abs(p.sent_command) > 500):
                raise ValueError("invalid current, speed or command")
            eligible.append(p)
        eligible.sort(key=lambda p: p.receive_ns)
        state = [None] * 4
        recent, old = [], [[] for _ in range(8)]
        pos = 0
        while pos < len(eligible):
            now = eligible[pos].receive_ns
            while pos < len(eligible) and eligible[pos].receive_ns == now:
                p = eligible[pos]
                state[p.channel] = p
                pos += 1
            relative = (now-cutoff_ns)/1e9
            row = _snapshot(state, now, (front,rear), self.scales)
            if -.5 < relative <= 0 and row is not None:
                recent.append((relative, row))
        if _snapshot(state, cutoff_ns, (front,rear), self.scales) is None or not recent:
            raise ValueError("M3 or M4 feedback/command is missing or stale")
        grid_state = [None] * 4
        index = 0
        grid = sorted((cutoff_ns-500_000_000-k*500_000_000+j*100_000_000,k-1)
                      for k in range(1,9) for j in range(1,6))
        for stamp, k in grid:
            while index < len(eligible) and eligible[index].receive_ns <= stamp:
                p = eligible[index]
                grid_state[p.channel] = p
                index += 1
            row = _snapshot(grid_state, stamp, (front,), self.scales)
            if row is not None:
                old[k].append(((stamp-cutoff_ns)/1e9, row))
        coverage = [len(w)/5 for w in old]
        vision = self.visual_encoder.encode_window(frames, cutoff_ns, max_frames=8)
        valid = [x for x in vision['frames'] if x['valid'] and x['receive_monotonic_ns'] <= cutoff_ns]
        if require_visual and not valid:
            raise ValueError("no valid causal camera features")
        visual = torch.zeros((1,max(1,len(valid)),576), dtype=torch.float32)
        vt = torch.zeros((1,max(1,len(valid))), dtype=torch.float64)
        vm = torch.zeros((1,max(1,len(valid))), dtype=torch.bool)
        for i,item in enumerate(valid):
            value = np.asarray(item['features'], dtype=float)
            if value.shape != (576,) or not np.isfinite(value).all():
                raise ValueError("invalid 576-dimensional visual feature")
            visual[0,i] = torch.tensor(value, dtype=torch.float32)
            vt[0,i] = (item['receive_monotonic_ns']-cutoff_ns)/1e9
            vm[0,i] = True
        rv, rt, rm = _sequence(recent, 12, 2)
        n = max(1, *(len(w) for w in old))
        fv = torch.zeros((1,8,n,6), dtype=torch.float32)
        ft = torch.zeros((1,8,n), dtype=torch.float64)
        fm = torch.zeros((1,8,n), dtype=torch.bool)
        for k,window in enumerate(old):
            for i,(relative,row) in enumerate(window):
                fv[0,k,i] = torch.tensor(row, dtype=torch.float32)
                ft[0,k,i] = relative
                fm[0,k,i] = True
        batch = dict(visual=visual,visual_times=vt,visual_valid=vm,recent=rv,recent_times=rt,recent_valid=rm,
            front=fv,front_times=ft,front_valid=fm,cutoff=torch.zeros(1),
            quality=torch.tensor(q,dtype=torch.float32).unsqueeze(0),
            front_coverage=torch.tensor([coverage],dtype=torch.float32),
            candidates=torch.tensor(features,dtype=torch.float32).unsqueeze(0),
            support=torch.tensor(support,dtype=torch.bool).unsqueeze(0))
        metadata = {'front_motor':front_motor,'rear_motor':7-front_motor,'candidate_vectors':vectors.tolist(),
            'support':support.tolist(),'history_coverage':coverage,'history_rows':[len(w) for w in old],
            'future_drive_packets_excluded':future,'future_frames_excluded':vision.get('future_frames_excluded',0),
            'frame_count':len(valid),'recent_rows':len(recent),'clock_sync':'not_calibrated',
            'latest_feedback_age_s':[(cutoff_ns-state[i].receive_ns)/1e9 for i in (front,rear)]}
        return batch, metadata

    def infer(self, *, deterministic=True, **kwargs):
        batch, metadata = self.assemble(**kwargs)
        self.model.eval()
        with torch.no_grad():
            result = self.model(**batch)
        observation = result['augmentation'][0].cpu().numpy().astype(float)
        support = np.asarray(metadata['support'],dtype=bool)
        probabilities, _ = self.policy._probabilities(observation[None,:],support[None,:])
        sample = self.policy.act(observation,support,deterministic=deterministic)
        return {'observation':observation.tolist(),'action':ACTIONS[sample.action_index],
            'action_index':sample.action_index,'candidate_probabilities':probabilities[0].tolist(),
            'prediction_features':result['prediction'][0].tolist(),
            'history_weights':result['history_weights'][0].tolist(),
            'null_weight':float(result['null_weight'][0,0]),
            'history_available':result['valid_windows'][0].tolist(),'timing':metadata,
            'model_trained_on_robot':False,'physical_policy_allowed':False,
            'notice':'仅离线演示，前后顺序为假设；模型未训练，不能控制电机。'}


def run_two_motor_demo(front_motor=3):
    """JSON-friendly synthetic demonstration. No serial, camera, or hardware I/O."""
    class SyntheticVision:
        def encode_window(self, frames, cutoff_ns, max_frames=8):
            return {'frames':[{'receive_monotonic_ns':cutoff_ns-10_000_000,
                               'valid':True,'features':np.zeros(576)}],
                    'future_frames_excluded':0}
    cutoff = 10_000_000_000
    drives = [DrivePacket(ch, cutoff-age*1_000_000, 100+ch, -20-ch, -60,
                          cutoff-age*1_000_000)
              for age in range(4400,0,-100) for ch in (2,3)]
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = TwoMotorModel(TwoMotorConfig())
    model.set_phase('frozen')
    policy = CategoricalPPO(147,seed=0)
    pipeline = TwoMotorPipeline(model,policy,SyntheticVision(),PipelineScales(7000,500,.25,.5))
    out = pipeline.infer(frames=[],drives=drives,cutoff_ns=cutoff,quality=[0,0,0,0,1],
                         nominal=[60,60],front_motor=front_motor)
    out.update({'drive_source':'synthetic','image_source':'synthetic_feature_vector',
                'mapping_confirmed':False,'model_trained_on_robot':False})
    return out
