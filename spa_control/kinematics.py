"""Ideal screw reference and the measured speed tracking reward component.

This module never controls hardware. A lead and a rotor/helix ratio must be
confirmed for the installed mechanism; motor feedback is relative to the stator.
The ideal reference assumes the chassis does not roll and no ground slip.
"""
from __future__ import annotations

import math
from numbers import Real


def _finite(name: str, value: Real) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} 必须是有限数值")
    return float(value)


def calculate_reference(motor_rpm: float, lead_mm: float,
                        motor_to_helix_ratio: float, forward_sign: int = 1,
                        distance_m: float | None = None) -> dict:
    """Signed no-slip reference for two matching, cooperating screw sections.

    motor_rpm is the common normalized rotor target (rpm), not measured ground
    speed. Ratio is rotor revolutions per helix revolution. forward_sign maps a
    positive normalized rotor target to the camera's positive track direction;
    it must be calibrated. distance_m is a positive path length, not a stop
    command. Never add the two section speeds together.
    """
    rpm = _finite("电机目标转速", motor_rpm)
    lead = _finite("导程", lead_mm)
    ratio = _finite("传动比", motor_to_helix_ratio)
    if abs(rpm) > 500:
        raise ValueError("电机目标转速必须在 ±500 rpm 范围")
    if lead <= 0 or ratio <= 0:
        raise ValueError("导程和转子/螺旋传动比必须大于零")
    if isinstance(forward_sign, bool) or forward_sign not in (-1, 1):
        raise ValueError("方向映射必须为 +1 或 -1")
    if distance_m is not None:
        distance_m = _finite("目标距离", distance_m)
        if distance_m <= 0:
            raise ValueError("目标距离必须大于零")
    speed = forward_sign * (rpm / ratio) * (lead / 1000.0) / 60.0
    if not math.isfinite(speed):
        raise ValueError("计算速度超出数值范围")
    duration = distance_m / abs(speed) if distance_m is not None and speed != 0 else None
    if duration is not None and not math.isfinite(duration):
        raise ValueError("计算时间超出数值范围")
    return {"motor_rpm": rpm, "lead_mm": lead, "motor_to_helix_ratio": ratio,
            "forward_sign": forward_sign, "theory_speed_mps": speed,
            "distance_m": distance_m, "ideal_duration_s": duration}


def speed_tracking_reward(target_mps: float, measured_mps: float,
                          sigma: float = 0.5, min_speed_mps: float = 0.01) -> dict:
    """Paper's Gaussian tracking term, using valid overhead-camera speed only.

    This excludes load, occupancy, action, smoothness and stop penalties, and is
    not a full PPO reward. Callers must verify freshness and calibration first.
    """
    target = _finite("目标速度", target_mps)
    measured = _finite("全局相机速度", measured_mps)
    sigma = _finite("sigma", sigma)
    min_speed = _finite("最低归一化速度", min_speed_mps)
    if sigma <= 0 or min_speed <= 0:
        raise ValueError("sigma 和最低归一化速度必须大于零")
    error = measured - target
    normalized = error / max(abs(target), min_speed)
    # sqrt keeps squaring safely in a range where the exponential is meaningful.
    z = normalized / sigma
    tracking = math.exp(-0.5 * z * z) if abs(z) < 40 else 0.0
    return {"target_mps": target, "measured_mps": measured, "error_mps": error,
            "normalized_error": normalized, "tracking_reward": tracking,
            "sigma": sigma, "min_speed_mps": min_speed}
