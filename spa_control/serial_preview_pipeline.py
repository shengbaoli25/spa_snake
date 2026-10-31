"""Offline-only JPEG + timestamped four-drive SPA inference pipeline.

This module opens no serial port, network socket, camera or motor. It cannot
arm a policy or send commands. Every time is host monotonic receipt time;
camera exposure time and physical command-application time are not implied.
The caller must supply frozen normalization scales and a complete quality
vector with ages, validity, geometry, task schedule, previous action and time
remaining, as specified in the manuscript. All inputs are caller-owned data.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np
import torch

from .serial_preview import ACTIONS, CategoricalPPO, project_candidates
from .serial_preview_model import SerialPreviewModel


@dataclass(frozen=True)
class DrivePacket:
    """One complete per-channel feedback packet and last known sent command.

    channel is structural 0..3, after an independently verified mapping.
    `command_time_ns` is the host time of the command record; it is not an
    acknowledgment that the actuator physically applied that command.
    """
    channel: int
    receive_ns: int
    current_raw: float
    speed_raw: float
    sent_command: int
    command_time_ns: int
    valid: bool = True


@dataclass(frozen=True)
class PipelineScales:
    """Values must be established from and frozen on the training split."""
    current: float
    speed: float
    max_feedback_age_s: float
    max_command_age_s: float

    def validate(self):
        if not all(math.isfinite(v) and v > 0 for v in
                   (self.current, self.speed, self.max_feedback_age_s, self.max_command_age_s)):
            raise ValueError("normalization and maximum ages must be finite and positive")


def _snapshot(state, at_ns: int, channels, scales):
    values = []
    for channel in channels:
        p = state[channel]
        if p is None or not p.valid:
            return None
        age = (at_ns - p.receive_ns) / 1e9
        command_age = (at_ns - p.command_time_ns) / 1e9
        if age < 0 or command_age < 0 or age > scales.max_feedback_age_s or command_age > scales.max_command_age_s:
            return None
        values.append((p.current_raw / scales.current, p.speed_raw / scales.speed,
                       p.sent_command / 500., age, command_age, 1.))
    # Group features by type: current, speed, command, feedback age, command age,
    # valid flag. Thus first six front features agree with SerialPreviewModel.
    return [values[i][j] for j in range(6) for i in range(len(values))]


def _sequence(rows, width, channels):
    count = max(1, len(rows))
    features = torch.zeros((1, count, width), dtype=torch.float32)
    times = torch.zeros((1, count), dtype=torch.float64)
    valid = torch.zeros((1, count), dtype=torch.bool)
    for i, (relative_s, row) in enumerate(rows):
        features[0, i] = torch.tensor(row, dtype=torch.float32)
        times[0, i] = relative_s
        valid[0, i] = True
    return features, times, valid


class SerialPreviewPipeline:
    """Build one causal observation, run pretrained vision and offline policy."""
    def __init__(self, model: SerialPreviewModel, policy: CategoricalPPO,
                 visual_encoder, scales: PipelineScales):
        scales.validate()
        if model.config.recent_dim != 24 or model.config.front_dim != 12:
            raise ValueError("pipeline expects recent_dim=24 and front_dim=12")
        if policy.obs_dim != 154 + model.config.quality_dim:
            raise ValueError("policy observation dimension does not match SPA model")
        self.model, self.policy, self.visual_encoder, self.scales = model, policy, visual_encoder, scales

    def assemble(self, *, frames, drives: Sequence[DrivePacket], cutoff_ns: int,
                 quality, nominal, signs, rear, active, require_visual=True):
        if isinstance(cutoff_ns, bool) or not isinstance(cutoff_ns, int) or cutoff_ns < 0:
            raise ValueError("cutoff must be nonnegative host monotonic nanoseconds")
        q = np.asarray(quality, dtype=float)
        if q.shape != (self.model.config.quality_dim,) or not np.isfinite(q).all():
            raise ValueError("quality must be complete, finite and match model config")
        if tuple(rear) != (False, False, True, True) or tuple(active) != (True, True, True, True):
            raise ValueError("structural channels 0,1 must be front and 2,3 rear; all four drives must be active")
        vectors, support = project_candidates(nominal, signs, rear, active)
        base = np.asarray(nominal, dtype=float) * np.asarray(signs, dtype=float) * np.asarray(active, dtype=bool)
        candidate_features = np.concatenate((vectors / 500., (vectors - base) / 5.), axis=1)
        if np.any(np.abs(candidate_features) > 1.000001):
            raise ValueError("invalid projected motor candidate or residual")
        eligible, future_count = [], 0
        for p in drives:
            if not isinstance(p, DrivePacket):
                raise ValueError("drive input must be DrivePacket")
            if isinstance(p.receive_ns, bool) or not isinstance(p.receive_ns, int) or p.receive_ns < 0:
                raise ValueError("invalid feedback receipt time")
            if p.receive_ns > cutoff_ns:
                future_count += 1
                continue
            if (type(p.channel) is not int or p.channel not in range(4) or
                isinstance(p.command_time_ns, bool) or not isinstance(p.command_time_ns, int) or
                p.command_time_ns < 0 or p.command_time_ns > p.receive_ns):
                raise ValueError("invalid channel or command time")
            if p.valid and (not math.isfinite(p.current_raw) or not math.isfinite(p.speed_raw) or
                            not isinstance(p.sent_command, int) or isinstance(p.sent_command, bool) or abs(p.sent_command) > 500):
                raise ValueError("valid feedback must contain finite values and a bounded integer command")
            eligible.append(p)
        eligible.sort(key=lambda p: p.receive_ns)
        state = [None] * 4
        recent, old = [], [[] for _ in range(8)]
        pos = 0
        # Apply simultaneous arrivals together so there is only one as-of row
        # at each timestamp; never use a packet that arrives later.
        while pos < len(eligible):
            now = eligible[pos].receive_ns
            while pos < len(eligible) and eligible[pos].receive_ns == now:
                p = eligible[pos]
                state[p.channel] = p
                pos += 1
            relative = (now - cutoff_ns) / 1e9
            recent_row = _snapshot(state, now, range(4), self.scales)
            if -.5 < relative <= 0. and recent_row is not None:
                recent.append((relative, recent_row))
        latest = _snapshot(state, cutoff_ns, range(4), self.scales)
        if latest is None or not recent:
            raise ValueError("missing or stale current, speed, command or feedback from one of four channels")
        # Fixed 0.1 s as-of grid: five endpoints per 0.5 s lag window.
        # Walk the grid chronologically so a sample contributes only after its
        # receipt; count a grid point only when both front channels are fresh.
        coverage_counts = [0] * 8
        grid = sorted((cutoff_ns - 500_000_000 - k*500_000_000 + j*100_000_000, k-1)
                      for k in range(1, 9) for j in range(1, 6))
        grid_state = [None] * 4
        index = 0
        for grid_ns, window_index in grid:
            while index < len(eligible) and eligible[index].receive_ns <= grid_ns:
                p = eligible[index]
                grid_state[p.channel] = p
                index += 1
            front_row = _snapshot(grid_state, grid_ns, range(2), self.scales)
            if front_row is not None:
                coverage_counts[window_index] += 1
                old[window_index].append(((grid_ns - cutoff_ns) / 1e9, front_row))
        coverage = [count / 5. for count in coverage_counts]
        vision = self.visual_encoder.encode_window(frames, cutoff_ns, max_frames=8)
        valid_frames = [item for item in vision['frames'] if item['valid'] and item['receive_monotonic_ns'] <= cutoff_ns]
        if require_visual and not valid_frames:
            raise ValueError("no valid causal head-camera frames")
        vis_count = max(1, len(valid_frames))
        visual = torch.zeros((1, vis_count, 576), dtype=torch.float32)
        visual_times = torch.zeros((1, vis_count), dtype=torch.float64)
        visual_valid = torch.zeros((1, vis_count), dtype=torch.bool)
        for i, item in enumerate(valid_frames):
            features = np.asarray(item['features'], dtype=float)
            if features.shape != (576,) or not np.isfinite(features).all():
                raise ValueError("visual encoder must output finite 576-dimensional features")
            visual[0, i] = torch.tensor(features, dtype=torch.float32)
            visual_times[0, i] = (item['receive_monotonic_ns'] - cutoff_ns) / 1e9
            visual_valid[0, i] = True
        rv, rt, rm = _sequence(recent, 24, 4)
        max_front = max(1, *(len(window) for window in old))
        fv = torch.zeros((1, 8, max_front, 12), dtype=torch.float32)
        ft = torch.zeros((1, 8, max_front), dtype=torch.float64)
        fm = torch.zeros((1, 8, max_front), dtype=torch.bool)
        for k, window in enumerate(old):
            for i, (relative, features) in enumerate(window):
                fv[0, k, i] = torch.tensor(features, dtype=torch.float32)
                ft[0, k, i] = relative
                fm[0, k, i] = True
        batch = dict(visual=visual, visual_times=visual_times, visual_valid=visual_valid,
                     recent=rv, recent_times=rt, recent_valid=rm,
                     front=fv, front_times=ft, front_valid=fm,
                     cutoff=torch.zeros(1, dtype=torch.float32),
                     quality=torch.tensor(q, dtype=torch.float32).unsqueeze(0),
                     front_coverage=torch.tensor(coverage, dtype=torch.float32).unsqueeze(0),
                     candidates=torch.tensor(candidate_features, dtype=torch.float32).unsqueeze(0),
                     support=torch.tensor(support, dtype=torch.bool).unsqueeze(0))
        metadata = {'cutoff_monotonic_ns': cutoff_ns, 'relative_clock_origin_ns': cutoff_ns,
                    'clock_sync': 'not_calibrated', 'future_drive_packets_excluded': future_count,
                    'future_frames_excluded': vision.get('future_frames_excluded', 0),
                    'frame_count': len(valid_frames), 'recent_rows': len(recent),
                    'history_rows': [len(w) for w in old],
                    'history_coverage': coverage,
                    'history_grid_period_s': .1,
                    'latest_feedback_age_s': [(cutoff_ns - state[i].receive_ns)/1e9 for i in range(4)],
                    'latest_command_age_s': [(cutoff_ns - state[i].command_time_ns)/1e9 for i in range(4)],
                    'candidate_vectors': vectors.tolist(), 'support': support.tolist()}
        return batch, metadata

    def infer(self, *, deterministic=True, **kwargs):
        """One diagnostic policy decision, never sends it to a physical device."""
        batch, metadata = self.assemble(**kwargs)
        self.model.eval()
        with torch.no_grad():
            result = self.model(**batch)
        obs = result['augmentation'][0].detach().cpu().numpy().astype(float)
        support = np.asarray(metadata['support'], dtype=bool)
        probabilities, _ = self.policy._probabilities(obs[None,:], support[None,:])
        sample = self.policy.act(obs, support, deterministic=deterministic)
        return {'observation': obs.tolist(), 'candidate_probabilities': probabilities[0].tolist(),
                'action': ACTIONS[sample.action_index], 'action_index': sample.action_index,
                'policy_version': sample.version, 'log_probability': sample.log_prob, 'value': sample.value,
                'prediction_features': result['prediction'][0].tolist(),
                'history_weights': result['history_weights'][0].tolist(),
                'null_weight': float(result['null_weight'][0, 0]),
                'history_available': result['valid_windows'][0].tolist(),
                'timing': metadata, 'untrained': True, 'physical_policy_allowed': False,
                'notice': '仅离线：SPA/PPO 权重未经实机验证；头部图像不提供地面速度或奖励'}
