"""Physical units, command projection and trial data; independent of the GUI."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Sequence

import numpy as np


@dataclass
class ExperimentConfig:
    host: str = "192.168.10.2"
    port: int = 9001
    channels: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    signs: list[int] = field(default_factory=lambda: [-1, 1, -1, 1])
    nominal_command: float = 300.0
    command_limit: float = 500.0
    residual_limit: float = 5.0
    duration_s: float = 12.0
    control_hz: float = 10.0
    diameter_mm: float = 80.0
    lead_angle_deg: float = 17.0
    lead_mm: float | None = None
    rotor_to_helix_ratio: float | None = None
    rpm_per_command: float | None = None
    calibration_commands: list[float] = field(default_factory=list)
    calibration_rpm: list[float] = field(default_factory=list)
    parameters_confirmed: bool = False
    target_mps: float | None = None
    # Allow substantial loaded-current peaks seen in the two-section pilot
    # while retaining a hard stop below the motor feedback raw range limit.
    current_limit_raw: float = 16000.0
    speed_scale_rpm: float = 1000.0
    sensor_timeout_s: float = 0.8
    reward_scale_mps: float = 0.05
    action_penalty: float = 0.0
    smoothness_penalty: float = 0.0

    def validate(self, physical: bool = False):
        if len(self.channels) != 4 or len(set(self.channels)) != 4 or any(type(c) is not int or not 1 <= c <= 6 for c in self.channels):
            raise ValueError("请选择四个不同的 M1–M6 通道，按结构顺序排列")
        if len(self.signs) != 4 or any(s not in (-1, 1) for s in self.signs):
            raise ValueError("每个电机方向必须为 +1 或 -1")
        ranges = {
            "nominal_command": (0, 500), "command_limit": (1, 500),
            "residual_limit": (0, 5), "duration_s": (0.5, 120),
            "control_hz": (2, 20), "diameter_mm": (1, 1000),
            "lead_angle_deg": (0.01, 89), "current_limit_raw": (1, 32767),
            "speed_scale_rpm": (1, 100000), "sensor_timeout_s": (0.2, 3),
            "reward_scale_mps": (0.0001, 10), "action_penalty": (0, 100),
            "smoothness_penalty": (0, 100),
        }
        for key, (low, high) in ranges.items():
            value = getattr(self, key)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{key} 必须在 {low}–{high} 内")
        if self.nominal_command > self.command_limit:
            raise ValueError("标称指令不能超过指令上限")
        if self.nominal_command != int(self.nominal_command):
            raise ValueError("标称指令必须是整数，与桥接速度协议一致")
        for key in ("lead_mm", "rotor_to_helix_ratio", "rpm_per_command", "target_mps"):
            value = getattr(self, key)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} 必须为正的有限数")
        xs, ys = self.calibration_commands, self.calibration_rpm
        if xs or ys:
            if len(xs) < 2 or len(xs) != len(ys):
                raise ValueError("指令/rpm 标定至少两组且一一对应")
            if not all(math.isfinite(v) and v >= 0 for v in xs + ys):
                raise ValueError("标定值必须为非负有限数")
            if any(b <= a for a, b in zip(xs, xs[1:])) or any(b <= a for a, b in zip(ys, ys[1:])):
                raise ValueError("指令与 rpm 标定值均须严格递增")
        if physical:
            if not self.parameters_confirmed:
                raise ValueError("请填写并确认实机参数、通道/方向、指令单位与电流限值")
            if self.rotor_to_helix_ratio is None or (self.rpm_per_command is None and not xs):
                raise ValueError("请给出转子/螺旋减速比与指令→rpm 标定；不能把控制器指令直接当 rpm")
            self.theory_speed(self.nominal_command)

    def lead_m(self) -> float:
        if self.lead_mm is not None:
            return self.lead_mm / 1000.0
        return math.pi * self.diameter_mm / 1000.0 * math.tan(math.radians(self.lead_angle_deg))

    def motor_rpm(self, command: float) -> float:
        q = abs(float(command))
        if self.calibration_commands:
            if not self.calibration_commands[0] <= q <= self.calibration_commands[-1]:
                raise ValueError("指令超出 rpm 标定区间，不进行外推")
            return float(np.interp(q, self.calibration_commands, self.calibration_rpm))
        if self.rpm_per_command is None:
            raise ValueError("未填写指令→转子 rpm 标定")
        return q * self.rpm_per_command

    def theory_speed(self, command: float) -> float:
        if self.rotor_to_helix_ratio is None:
            raise ValueError("未填写转子/螺旋减速比 G")
        return self.motor_rpm(command) * self.lead_m() / (60.0 * self.rotor_to_helix_ratio)

    def target_speed(self) -> float:
        return self.theory_speed(self.nominal_command) if self.target_mps is None else self.target_mps

    def project(self, normalized_action: Sequence[float]) -> tuple[list[int], list[float]]:
        if not math.isfinite(self.nominal_command) or self.nominal_command != int(self.nominal_command):
            raise ValueError("标称指令必须是整数，与桥接速度协议一致")
        action = np.asarray(normalized_action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("残差必须是四个有限数")
        requested = np.clip(action, -1, 1) * self.residual_limit
        # Apply physical limits before integer quantization for the bridge protocol.
        low = math.ceil(max(0, self.nominal_command - self.residual_limit))
        high = math.floor(min(self.command_limit, self.nominal_command + self.residual_limit))
        magnitude = np.clip(np.rint(self.nominal_command + requested), low, high).astype(int)
        commands = [0] * 6
        for channel, sign, value in zip(self.channels, self.signs, magnitude):
            commands[channel - 1] = int(sign * value)
        applied = [float(v - self.nominal_command) for v in magnitude]
        return commands, applied

    def as_dict(self):
        return asdict(self)


def command_line(speeds: Sequence[int]) -> bytes:
    if len(speeds) != 6 or any(type(v) is not int or abs(v) > 500 for v in speeds):
        raise ValueError("协议要求六路 [-500,500] 整数指令")
    return ("CTRL SPEED " + " ".join(f"M{i}={q}" for i, q in enumerate(speeds, 1)) + "\n").encode("ascii")


def observation(config: ExperimentConfig, telemetry: dict, previous_residual) -> np.ndarray:
    """17 values; camera measured speed is supervision only, never actor input."""
    values = []
    for j, (channel, sign) in enumerate(zip(config.channels, config.signs)):
        motor = telemetry[channel]
        values.extend((sign * motor["speed"] / config.speed_scale_rpm,
                       motor["current_raw"] / config.current_limit_raw,
                       config.nominal_command / config.command_limit,
                       previous_residual[j] / max(config.residual_limit, 1)))
    values.append(config.target_speed() / config.reward_scale_mps)
    return np.clip(np.asarray(values, dtype=float), -10, 10)


def speed_reward(config: ExperimentConfig, measured_mps: float, action, previous_action):
    if not math.isfinite(measured_mps):
        raise ValueError("相机速度无效，不能形成奖励")
    error = config.target_speed() - measured_mps
    raw = -abs(error)
    reward = raw / config.reward_scale_mps
    reward -= config.action_penalty * float(np.mean(np.square(action)))
    reward -= config.smoothness_penalty * float(np.mean(np.square(np.asarray(action) - previous_action)))
    return error, raw, float(reward)
