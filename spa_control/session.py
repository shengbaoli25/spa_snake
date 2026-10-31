"""Finite real-robot trials; all network writes happen outside Tk callbacks."""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict, replace
import json
import math
from numbers import Integral
from pathlib import Path
import threading
import time
from datetime import datetime

import numpy as np

from .core import ExperimentConfig, observation, speed_reward
from .ppo import PPOAgent, Transition
from .trial_log import AsyncCsvWriter


def _baseline_joint_speeds(config, joint_speeds=None, only_joint=None):
    config.validate(physical=False)
    if only_joint is not None and (isinstance(only_joint, bool) or not isinstance(only_joint, Integral)
                                   or not 0 <= only_joint < 4):
        raise ValueError("单电机索引必须是结构顺序 0–3")
    if joint_speeds is None:
        return [int(config.nominal_command)] * 4
    try:
        speeds = list(joint_speeds)
    except TypeError as exc:
        raise ValueError("请按结构顺序提供四个整数指令幅值") from exc
    if len(speeds) != 4 or any(isinstance(speed, bool) or not isinstance(speed, Integral)
                               or not 0 <= speed <= config.command_limit for speed in speeds):
        raise ValueError(f"四路指令幅值必须是 0–{int(config.command_limit)} 的整数")
    return [int(speed) for speed in speeds]


def baseline_commands(config, joint_speeds=None, only_joint=None):
    """Preview six physical channel commands without hardware or policy access.

    joint_speeds contains four nonnegative integer bridge magnitudes in
    structural order. only_joint selects one structural index; every other
    physical channel receives zero, including unconfigured channels.
    """
    speeds = _baseline_joint_speeds(config, joint_speeds, only_joint)
    commands = [0] * 6
    for index, (channel, sign, speed) in enumerate(zip(config.channels, config.signs, speeds)):
        if only_joint is None or index == only_joint:
            commands[channel - 1] = int(sign * speed)
    return commands


class TelemetryOvercurrentError(ValueError):
    """Carry the triggering measurement through immediate trial shutdown."""

    def __init__(self, channel, current_raw, limit_raw, telemetry, observed_at):
        super().__init__(f"M{channel} 电流原始值 {current_raw:g} 超过设置限值 {limit_raw:g}")
        self.channel = channel
        self.current_raw = float(current_raw)
        self.limit_raw = float(limit_raw)
        self.telemetry = copy.deepcopy(telemetry)
        self.observed_at = float(observed_at)


class TrialRunner:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.agent = PPOAgent(17, 4, seed=42)
        self.link = None
        self.camera = None
        self.busy = False
        self.message = "等待连接与参数配置"
        self.latest = {}
        self.history = []
        self.completed = {"baseline": 0, "train": 0, "eval": 0, "baseline_no_camera": 0, "single_motor": 0}
        self.cancel = threading.Event()
        self.worker = None
        self.gui_heartbeat = time.monotonic()
        self.policy_config = None
        self.abort_reason = "操作者停止"
        self.motion_lock = threading.Lock()
        self._csv_queue_size = 256
        self._csv_finalize_timeout_s = 1.0

    def ping(self):
        self.gui_heartbeat = time.monotonic()

    @staticmethod
    def policy_signature(config):
        # These fields determine action meaning and fixed observation scales.
        keys = ("channels", "signs", "command_limit", "residual_limit", "current_limit_raw",
                "speed_scale_rpm", "reward_scale_mps")
        return {key: config.as_dict()[key] for key in keys}

    def reset_policy(self):
        if self.busy:
            raise ValueError("试验运行中不能重置策略")
        self.agent = PPOAgent(17, 4, seed=42)
        self.policy_config = None
        self.message = "初始策略已重置，确定性残差为 0"

    def start(self, config: ExperimentConfig, mode: str, demo=False):
        if self.busy:
            raise ValueError("上一条试验或策略更新尚未结束")
        if mode not in ("baseline", "train", "eval"):
            raise ValueError("未知试验模式")
        config.validate(physical=True)
        if config.nominal_command <= 0:
            raise ValueError("前进验证需使用非零标称指令")
        if mode == "train" and config.duration_s * config.control_hz < 2:
            raise ValueError("PPO 训练每条至少需要两步，请增加时长或控制频率")
        if not self.link or not self.link.connected:
            raise ValueError("请先连接机器人")
        if not self.camera:
            raise ValueError("请先连接全局测速并选择跟踪 ROI")
        if mode != "baseline":
            if self.policy_config is not None and self.policy_config != self.policy_signature(config):
                raise ValueError("通道/尺度已改变，请重置策略或恢复原设置")
            self.policy_config = self.policy_signature(config)
        self._sensors(config)
        self.cancel.clear()
        self.abort_reason = "操作者停止"
        self.busy = True
        self.latest = {}
        self.history = []
        self.ping()
        self.worker = threading.Thread(target=self._run, args=(copy.deepcopy(config), mode, demo), daemon=True)
        self.worker.start()

    def start_baseline(self, config: ExperimentConfig, demo=False, joint_speeds=None, only_joint=None):
        """Run the fixed forward command using motor telemetry only.

        The configured number is a bridge command, not a calibrated target rpm.
        This path neither requires camera/kinematics nor forms PPO transitions.
        """
        if self.busy:
            raise ValueError("上一条试验或策略更新尚未结束")
        speeds = _baseline_joint_speeds(config, joint_speeds, only_joint)
        if only_joint is not None:
            only_joint = int(only_joint)
        required_channels = config.channels if only_joint is None else [config.channels[only_joint]]
        if not self.link or not self.link.connected:
            raise ValueError("请先连接机器人")
        self._telemetry(config, required_channels)
        with self.motion_lock:
            self.cancel.clear()
            self.abort_reason = "操作者停止"
            self.busy = True
            self.latest = {}
            self.history = []
            self.ping()
        self.worker = threading.Thread(target=self._run_baseline,
                                       args=(copy.deepcopy(config), bool(demo), speeds, only_joint), daemon=True)
        self.worker.start()

    def abort(self, reason="操作者停止"):
        with self.motion_lock:
            self.abort_reason = reason
            self.cancel.set()
            if self.link:
                self.link.stop()

    def _telemetry(self, config, required_channels=None):
        if not self.link or not self.link.connected:
            raise ValueError("机器人通信已断开")
        telemetry = self.link.snapshot()
        now = time.monotonic()
        for channel in config.channels if required_channels is None else required_channels:
            motor = telemetry.get(channel)
            if not motor or not all(isinstance(motor.get(key), (float, int)) and math.isfinite(motor[key])
                                    for key in ("speed", "current_raw", "last_seen")):
                raise ValueError(f"M{channel} 遥测缺失或不是有限数")
            if not 0 <= now - motor["last_seen"] <= config.sensor_timeout_s:
                raise ValueError(f"M{channel} 遥测缺失/过时")
            if abs(motor["current_raw"]) >= config.current_limit_raw:
                raise TelemetryOvercurrentError(channel, motor["current_raw"], config.current_limit_raw,
                                                telemetry, now)
        return telemetry

    def _sensors(self, config):
        telemetry = self._telemetry(config)
        sample = self.camera.latest(max_age_s=config.sensor_timeout_s)
        if sample is None or not sample.tracking_ok or sample.velocity_mps is None:
            raise ValueError("全局相机速度缺失、跟踪丢失或已过时")
        if not math.isfinite(sample.velocity_mps):
            raise ValueError("全局相机速度无效")
        return telemetry, sample

    def _transport_evidence(self):
        """Best-effort diagnostics must never interrupt an emergency stop."""
        evidence = {}
        try:
            if self.link is not None:
                evidence["status"] = self.link.status
        except Exception as exc:
            evidence["status_error"] = str(exc)
        try:
            if self.link is not None and hasattr(self.link, "diagnostics"):
                evidence["diagnostics"] = self.link.diagnostics()
        except Exception as exc:
            evidence["diagnostics_error"] = str(exc)
        return evidence

    def _run_baseline(self, config, demo, joint_speeds=None, only_joint=None):
        mode = "baseline_no_camera" if only_joint is None else "single_motor"
        joint_speeds = _baseline_joint_speeds(config, joint_speeds, only_joint)
        commands = baseline_commands(config, joint_speeds, only_joint)
        required_channels = config.channels if only_joint is None else [config.channels[only_joint]]
        started = time.monotonic()
        run_dir = self.output_dir / ("demo" if demo else "real") / (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + mode)
        rows = []
        safety_fault = None
        logger = None
        csv_complete = False
        transport_before_stop = None
        outcome, error, stop_sent = "aborted", "", False
        self.message = ("四电机运行中" if only_joint is None else f"单电机 M{config.channels[only_joint]} 运行中")
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "config.json").write_text(
                json.dumps(config.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            metadata = {"demo": demo, "mode": mode, "schema_version": 1,
                        "command_units": "bridge protocol integer; not calibrated rpm",
                        "telemetry_source": "DEMO" if demo else "robot CAN telemetry",
                        "camera_used": False, "ppo_used": False,
                        "joint_speeds": joint_speeds, "only_joint": only_joint, "full_commands": commands}
            (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            # Manual baseline settings do not add a learned residual.
            applied = [0.0] * 4
            fieldnames = ["elapsed_s", "tx_monotonic_s", "sample_monotonic_s", "interval_s",
                          "mode", "demo", "telemetry_source", "joint_speeds", "only_joint", "full_commands"]
            for j in range(4):
                fieldnames.extend(f"j{j}_{key}" for key in (
                    "channel", "sign", "active", "rpm", "current_raw", "command", "residual_applied", "last_seen_s"))
            logger = AsyncCsvWriter(run_dir / "steps.csv", fieldnames, capacity=self._csv_queue_size)
            logger.start(timeout_s=self._csv_finalize_timeout_s)
            motion_start = time.monotonic()
            end = motion_start + config.duration_s
            period = 1.0 / config.control_hz
            step_index = 0
            while motion_start + step_index * period < end:
                scheduled_at = motion_start + step_index * period
                if self.cancel.wait(max(0, scheduled_at - time.monotonic())):
                    raise RuntimeError(self.abort_reason)
                logger.check()
                if time.monotonic() - self.gui_heartbeat > 1.0:
                    raise RuntimeError("界面心跳超时")
                self._telemetry(config, required_channels)
                with self.motion_lock:
                    if self.cancel.is_set():
                        raise RuntimeError(self.abort_reason)
                    now = time.monotonic()
                    if now - self.gui_heartbeat > 1.0:
                        raise RuntimeError("界面心跳超时")
                    if now - scheduled_at >= period:
                        raise RuntimeError("控制周期超时，已中止且不会补发过期指令")
                    if now >= end:
                        break
                    tx = self.link.send_speeds(commands, lease_s=max(0.3, 2.5 / config.control_hz))
                boundary = min(motion_start + (step_index + 1) * period, end)
                if self.cancel.wait(max(0, boundary - time.monotonic())):
                    raise RuntimeError(self.abort_reason)
                logger.check()
                telemetry = self._telemetry(config, required_channels)
                if self.cancel.is_set():
                    raise RuntimeError(self.abort_reason)
                now = time.monotonic()
                row = {"elapsed_s": tx - motion_start, "tx_monotonic_s": tx,
                       "sample_monotonic_s": now, "interval_s": now - tx,
                       "mode": mode, "demo": demo, "telemetry_source": metadata["telemetry_source"],
                       "joint_speeds": json.dumps(joint_speeds), "only_joint": only_joint,
                       "full_commands": json.dumps(commands)}
                for j, (channel, sign) in enumerate(zip(config.channels, config.signs)):
                    motor = telemetry.get(channel) or {}
                    fresh = (isinstance(motor.get("last_seen"), (float, int))
                             and math.isfinite(motor["last_seen"])
                             and 0 <= now - motor["last_seen"] <= config.sensor_timeout_s)
                    def measured(key):
                        value = motor.get(key)
                        return value if fresh and isinstance(value, (float, int)) and math.isfinite(value) else None
                    values = {"channel": channel, "sign": sign, "active": channel in required_channels,
                              "rpm": measured("speed"), "current_raw": measured("current_raw"),
                              "command": commands[channel - 1], "residual_applied": applied[j],
                              "last_seen_s": measured("last_seen")}
                    row.update({f"j{j}_{key}": value for key, value in values.items()})
                logger.submit(row)
                rows.append(row)
                self.latest = row
                self.history = rows[-1200:].copy()
                step_index += 1
            transport_before_stop = self._transport_evidence()
            with self.motion_lock:
                stop_sent = bool(self.link.stop())
                if not stop_sent:
                    raise RuntimeError("停止命令未能写入通信链路，试验中止")
                if self.cancel.is_set():
                    raise RuntimeError(self.abort_reason)
                if not rows:
                    raise RuntimeError("没有有效电机遥测数据")
                outcome = "completed"
        except Exception as exc:
            error = str(exc)
            if transport_before_stop is None:
                transport_before_stop = self._transport_evidence()
            if isinstance(exc, TelemetryOvercurrentError):
                # Fault samples are diagnostic data, not completed control
                # intervals. Preserve them without adding an artificial CSV step.
                snapshot = {}
                for channel in config.channels:
                    motor = exc.telemetry.get(channel) or {}
                    snapshot[str(channel)] = {
                        key: float(value) if isinstance(value, (float, int)) and math.isfinite(value) else None
                        for key in ("speed", "current_raw", "last_seen")
                        for value in (motor.get(key),)
                    }
                safety_fault = {"type": "overcurrent", "channel": exc.channel,
                                "current_raw": exc.current_raw, "limit_raw": exc.limit_raw,
                                "observed_monotonic_s": exc.observed_at, "telemetry": snapshot}
        finally:
            if transport_before_stop is None:
                transport_before_stop = self._transport_evidence()
            if not stop_sent:
                try:
                    with self.motion_lock:
                        stop_sent = bool(self.link and self.link.stop())
                except Exception as exc:
                    error += f"；停止发送失败: {exc}"
            # Disk waits occur only after a stop attempt, never in the motion
            # loop. A bounded join cannot make an incomplete trial successful.
            if logger is not None:
                try:
                    logger.finish(timeout_s=self._csv_finalize_timeout_s)
                    csv_complete = True
                except Exception as exc:
                    error += f"；{exc}"
                    outcome = "aborted"
            if self.cancel.is_set() and outcome == "completed":
                error = self.abort_reason
                outcome = "aborted"
            current_samples = [abs(row[f"j{j}_current_raw"]) for row in rows for j in range(4)
                               if row[f"j{j}_current_raw"] is not None]
            if safety_fault is not None:
                current_samples.extend(abs(motor["current_raw"]) for motor in safety_fault["telemetry"].values()
                                       if motor["current_raw"] is not None)
            summary = {"outcome": outcome, "reason": error, "mode": mode, "demo": demo,
                       "stop_write_succeeded": stop_sent, "stop_execution_confirmed": False,
                       "duration_wall_s": time.monotonic() - started, "steps": len(rows),
                       "nominal_command": config.nominal_command,
                       "joint_speeds": joint_speeds, "only_joint": only_joint, "full_commands": commands,
                       "command_units": "bridge protocol integer; not calibrated rpm",
                       "peak_abs_current_raw": max(current_samples, default=None),
                       "safety_fault": safety_fault, "csv_complete": csv_complete,
                       "csv_log": logger.diagnostics() if logger is not None else None,
                       "transport_before_stop": transport_before_stop,
                       "transport_diagnostics": self._transport_evidence()}
            try:
                temporary_summary = run_dir / "summary.json.tmp"
                temporary_summary.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary_summary.replace(run_dir / "summary.json")
            except Exception as exc:
                error += f"；保存汇总失败: {exc}"
                outcome = "aborted"
            else:
                if outcome == "completed":
                    self.completed[mode] += 1
            self.message = ("完成" if outcome == "completed" else f"中止: {error}") + f" | {run_dir.name}"
            self.busy = False

    def _run(self, config, mode, demo):
        started = time.monotonic()
        run_dir = self.output_dir / ("demo" if demo else "real") / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + mode)
        rows, transitions, rewards = [], [], []
        outcome, error, metrics = "aborted", "", {}
        self.message = f"{mode} 试验运行中"
        stop_sent = False
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "config.json").write_text(json.dumps(config.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
            (run_dir / "policy_config.json").write_text(json.dumps(self.policy_signature(config), indent=2), encoding="utf-8")
            (run_dir / "metadata.json").write_text(json.dumps({"demo": demo, "mode": mode, "schema_version": 1}, indent=2), encoding="utf-8")
            camera_info = {"source_type": type(self.camera).__name__, "clock": "local time.monotonic",
                           "velocity_filter": "source causal trailing estimator; includes filter delay"}
            if hasattr(self.camera, "calibration"):
                camera_info["calibration"] = {"points": [asdict(p) for p in self.camera.calibration.points],
                                               "metadata": self.camera.calibration.metadata}
                camera_info["direction"] = self.camera.processor.direction
                camera_info["camera_index"] = self.camera.camera_index
            if hasattr(self.camera, "path"):
                camera_info["recording_csv"] = str(self.camera.path)
            (run_dir / "camera_metadata.json").write_text(json.dumps(camera_info, ensure_ascii=False, indent=2), encoding="utf-8")
            self.agent.save(run_dir / "policy_before.npz")
            previous_action = np.zeros(4)
            previous_residual = np.zeros(4)
            telemetry, camera = self._sensors(config)
            obs = observation(config, telemetry, previous_residual)
            fieldnames = ["elapsed_s", "tx_monotonic_s", "sample_monotonic_s", "sample_received_s",
                          "interval_s", "camera_source", "position_m", "target_mps", "theory_mps", "actual_mps",
                          "error_mps", "reward_raw", "reward_scaled", "policy_version", "mode", "demo"]
            for j in range(4):
                fieldnames.extend(f"j{j}_{key}" for key in ("channel", "sign", "rpm", "current_raw", "residual_requested", "residual_applied", "command"))
            with (run_dir / "steps.csv").open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                motion_start = time.monotonic()
                end = motion_start + config.duration_s
                while time.monotonic() < end:
                    if self.cancel.is_set():
                        raise RuntimeError(self.abort_reason)
                    if time.monotonic() - self.gui_heartbeat > 1.0:
                        raise RuntimeError("界面心跳超时")
                    self._sensors(config)
                    sample = self.agent.act(obs, deterministic=(mode != "train"))
                    action = np.zeros(4) if mode == "baseline" else sample.action
                    commands, applied = config.project(action)
                    with self.motion_lock:
                        if self.cancel.is_set():
                            raise RuntimeError(self.abort_reason)
                        tx = self.link.send_speeds(commands, lease_s=max(0.3, 2.5 / config.control_hz))
                    boundary = min(tx + 1.0 / config.control_hz, end)
                    if self.cancel.wait(max(0, boundary - time.monotonic())):
                        raise RuntimeError(self.abort_reason)
                    telemetry, camera = self._sensors(config)
                    # No pre-action camera sample is reused as a new transition.
                    if camera.received_monotonic_s <= tx or camera.timestamp_s <= tx:
                        raise RuntimeError("动作区间内无新相机速度帧；降低控制频率或检查测速输入")
                    if self.cancel.is_set():
                        raise RuntimeError(self.abort_reason)
                    next_obs = observation(config, telemetry, applied)
                    done = time.monotonic() >= end
                    error_mps, reward_raw, reward = speed_reward(config, camera.velocity_mps, action, previous_action)
                    row = {"elapsed_s": tx - motion_start, "tx_monotonic_s": tx,
                           "sample_monotonic_s": camera.timestamp_s, "sample_received_s": camera.received_monotonic_s,
                           "interval_s": time.monotonic() - tx, "camera_source": camera.source,
                           "position_m": camera.position_m, "target_mps": config.target_speed(),
                           "theory_mps": config.theory_speed(config.nominal_command), "actual_mps": camera.velocity_mps,
                           "error_mps": error_mps, "reward_raw": reward_raw, "reward_scaled": reward,
                           "policy_version": sample.policy_version, "mode": mode, "demo": demo}
                    for j, (channel, sign) in enumerate(zip(config.channels, config.signs)):
                        for key, value in {"channel": channel, "sign": sign, "rpm": telemetry[channel]["speed"],
                                           "current_raw": telemetry[channel]["current_raw"], "residual_requested": float(action[j] * config.residual_limit),
                                           "residual_applied": applied[j], "command": commands[channel - 1]}.items():
                            row[f"j{j}_{key}"] = value
                    writer.writerow(row)
                    file.flush()
                    rows.append(row)
                    rewards.append(reward)
                    self.latest = row
                    self.history = rows[-1200:].copy()
                    if mode == "train":
                        transitions.append(Transition(obs=obs.copy(), raw_action=sample.raw_action.copy(), log_prob=sample.log_prob,
                                                      value=sample.value, reward=reward, next_value=0.0 if done else self.agent.value(next_obs),
                                                      done=done, policy_version=sample.policy_version))
                    obs, previous_action, previous_residual = next_obs, np.asarray(action).copy(), applied
                stop_sent = self.link.stop()
                if not stop_sent:
                    raise RuntimeError("停止命令未能写入通信链路，试验中止")
                if self.cancel.is_set():
                    raise RuntimeError(self.abort_reason)
                if not rows:
                    raise RuntimeError("没有有效试验数据")
                if mode == "train":
                    # Duration is a finite trial task: its last valid transition is terminal.
                    if not transitions[-1].done:
                        transitions[-1] = replace(transitions[-1], done=True, next_value=0.0)
                    self.message = "电机已发送停止；正在更新 PPO"
                    # Train a copy so cancellation or an optimizer failure cannot
                    # partly replace the active policy used for the next trial.
                    candidate = copy.deepcopy(self.agent)
                    metrics = candidate.update(transitions)
                    with self.motion_lock:
                        if self.cancel.is_set():
                            raise RuntimeError(self.abort_reason)
                        self.agent = candidate
                    self.agent.save(run_dir / "policy_after.npz")
                outcome = "completed"
                self.completed[mode] += 1
        except Exception as exc:
            error = str(exc)
        finally:
            if not stop_sent:
                stop_sent = bool(self.link and self.link.stop())
            summary = {"outcome": outcome, "reason": error, "mode": mode, "demo": demo,
                       "stop_write_succeeded": stop_sent, "stop_execution_confirmed": False,
                       "duration_wall_s": time.monotonic() - started, "steps": len(rows),
                       "mean_absolute_error_mps": float(np.mean([abs(r["error_mps"]) for r in rows])) if rows else None,
                       "mean_reward": float(np.mean(rewards)) if rewards else None,
                       "peak_abs_current_raw": max((abs(r[f"j{j}_current_raw"]) for r in rows for j in range(4)), default=None),
                       "policy_version": self.agent.policy_version, "ppo": metrics}
            try:
                if transitions:
                    np.savez_compressed(run_dir / "rollout.npz", observations=np.stack([t.obs for t in transitions]),
                                        raw_actions=np.stack([t.raw_action for t in transitions]),
                                        log_probs=np.asarray([t.log_prob for t in transitions]),
                                        values=np.asarray([t.value for t in transitions]),
                                        rewards=np.asarray([t.reward for t in transitions]),
                                        next_values=np.asarray([t.next_value for t in transitions]),
                                        dones=np.asarray([t.done for t in transitions]),
                                        policy_versions=np.asarray([t.policy_version for t in transitions]),
                                        used_for_update=np.asarray(outcome == "completed" and mode == "train"))
                (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as exc:
                error += f"；保存汇总失败: {exc}"
            self.message = ("完成" if outcome == "completed" else f"中止: {error}") + f" | {run_dir.name}"
            self.busy = False


class DemoRobot:
    """Explicit offline plumbing demo, never used as experimental evidence."""
    def __init__(self, config):
        self.config = config
        self.connected = True
        self.speeds = [0] * 6
        self.deadline = 0.0
        self.status = "离线演示（无机器人通信）"
        self.last_feedback = "演示数据"
        self.last = time.monotonic()
        self.position = 0.0
        self.velocity = 0.0

    def send_speeds(self, speeds, lease_s=0.4):
        self.speeds = speeds.copy()
        self.deadline = time.monotonic() + lease_s
        return time.monotonic()

    def stop(self):
        self.speeds = [0] * 6
        return True

    def close(self):
        self.stop()
        self.connected = False

    def snapshot(self):
        if time.monotonic() >= self.deadline:
            self.stop()
        # A coefficient of one supplies explicit DEMO data when physical rpm
        # calibration is absent; it never calibrates the real bridge command.
        demo_rpm_scale = self.config.rpm_per_command if self.config.rpm_per_command is not None else 1.0
        return {i: {"speed": q * demo_rpm_scale, "current_raw": abs(q) * 12,
                    "last_seen": time.monotonic(), "can_id": hex(0x204 + i), "temp": 25}
                for i, q in enumerate(self.speeds, 1)}

    def latest(self, max_age_s=0.8):
        from .camera import CameraSample
        now = time.monotonic()
        dt = now - self.last
        self.last = now
        forward = [self.speeds[c - 1] * s for c, s in zip(self.config.channels, self.config.signs)]
        desired = self.config.theory_speed(max(0, float(np.mean(forward)))) * 0.93
        self.velocity += (desired - self.velocity) * (1 - math.exp(-dt / 0.2))
        self.position += self.velocity * dt
        return CameraSample(now, now, self.position, self.velocity, True, "DEMO")
