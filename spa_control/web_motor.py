"""C-board telemetry and bounded, explicitly requested movement.

One worker owns every serial read/write. Motion is guarded independently of the
policy and never clears a fault. BENCH commands use signed encoder rpm.
"""
from __future__ import annotations

import copy
import math
import re
import threading
import time


BOARD_SERIAL = "206B377B5948"
PREFIX = "RM_C MOTOR_USB v4-full "
FRAME_MAX_AGE_S = 0.5
FEEDBACK_MAX_AGE_S = 0.25
CLEAR_BASELINE_S = 1.2
CLEAR_TIMEOUT_S = 5.0
CLEAR_FRAME_GAP_S = 0.25
CLEARABLE_FAULTS = frozenset({"USB_CLOSED", "COMMAND_TIMEOUT", "FEEDBACK_STALE",
                              "OVERCURRENT", "OVERTEMP", "CAN_ERROR"})
MOTION_BASELINE_S = 1.2
MOTION_PREPARE_TIMEOUT_S = 5.0
MOTION_FRAME_GAP_S = 0.25
MOTION_ACK_TIMEOUT_S = 0.22
MOTION_RENEW_S = 0.1
MOTION_EXTERNAL_LEASE_S = 0.5
MOTION_STOP_TIMEOUT_S = 2.0
_INTEGER = re.compile(r"-?\d+|0x[0-9a-fA-F]+\Z")


class MotorError(RuntimeError):
    pass


def _fields(text):
    result = {}
    for word in text.split():
        if word.count("=") != 1:
            raise ValueError("遥测字段格式错误")
        key, value = word.split("=", 1)
        if not key or not value or key in result:
            raise ValueError("遥测字段缺失或重复")
        result[key] = value
    return result


def _int(value):
    if not _INTEGER.fullmatch(value):
        raise ValueError("遥测数字格式错误")
    return int(value, 16 if value.startswith("0x") else 10)


def parse_frame(line):
    """Parse complete v4-full records. Other lines are not telemetry."""
    if not line.startswith(PREFIX):
        if line.startswith("RM_C MOTOR_USB "):
            raise ValueError("固件版本不匹配，需要 v4-full")
        return None
    parts = line[len(PREFIX):].split(" | ")
    if len(parts) != 5:
        raise ValueError("遥测电机段不完整")
    h = _fields(parts[0])
    required = {"can_ready", "can_healthy", "ESR", "fault", "active", "mode",
                "dir3", "dir4", "roll", "lease_ms", "limit", "output_limit", "current_trip",
                "trip_mask", "trip_current", "trip_ms"}
    if not required <= h.keys():
        raise ValueError("遥测头缺少字段")
    for key in ("can_ready", "can_healthy", "ESR", "active", "dir3", "dir4",
                "lease_ms", "limit", "output_limit", "current_trip", "trip_mask", "trip_current", "trip_ms"):
        _int(h[key])
    if not (0 <= _int(h["trip_mask"]) <= 0x0f and -32768 <= _int(h["trip_current"]) <= 32767
            and 0 <= _int(h["trip_ms"]) <= 0xffffffff):
        raise ValueError("故障采样值无效")
    if h["can_ready"] not in ("0", "1") or h["can_healthy"] not in ("0", "1"):
        raise ValueError("CAN 状态无效")
    if _int(h["dir3"]) not in (-1, 1) or _int(h["dir4"]) not in (-1, 1):
        raise ValueError("电机映射无效")
    motors = {}
    names = {"id", "angle", "spd", "torque", "setI", "setV", "target", "temp", "seen", "rx", "age_ms"}
    for part in parts[1:]:
        match = re.fullmatch(r"M([1-4]) (.*)", part)
        if not match or match[1] in motors:
            raise ValueError("电机通道无效或重复")
        channel = int(match[1])
        values = _fields(match[2])
        if not names <= values.keys():
            raise ValueError("电机遥测缺少字段")
        m = {key: _int(values[key]) for key in names}
        if (m["id"] != 0x204 + channel or not 0 <= m["angle"] < 8192
                or m["seen"] not in (0, 1) or not 0 <= m["rx"] <= 0xffffffff
                or m["age_ms"] < 0):
            raise ValueError("电机遥测值无效")
        motors[match[1]] = m
    if set(motors) != {"1", "2", "3", "4"}:
        raise ValueError("遥测电机通道不完整")
    return {"header": h, "motors": motors}


def _ports():
    from serial.tools import list_ports
    return list(list_ports.comports())


def _open_port(port):
    import serial
    return serial.Serial(port, 115200, timeout=0.05, write_timeout=0.1)


class WebMotor:
    """One worker owns the port; disconnect requests shutdown asynchronously."""

    def __init__(self, port_lister=None, serial_factory=None, clock=None):
        self._port_lister = port_lister or _ports
        self._serial_factory = serial_factory or _open_port
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._worker = None
        self._stop = None
        self._clear = None
        self._motion = None
        self._last_motion = {"state": "idle", "phase": "idle", "message": "尚未启动运动", "owner": None,
                             "targets": {"3": 0, "4": 0}, "elapsed_s": 0.0, "remaining_s": 0.0}
        self._state = {"state": "disconnected", "message": "未连接 C 板",
                       "port": None, "frames_received": 0, "stop_sent": False,
                       "stop_acknowledged": False, "fault": None, "motors": None,
                       "frame_at": None, "motor_sample_at": {}, "motor_counts": {},
                       "clear_state": "idle", "clear_message": "尚未请求清除"}

    def connect(self):
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise MotorError("C 板连接或断开正在进行")
            self._stop = threading.Event()
            self._clear = None
            self._motion = None
            self._last_motion = {"state": "idle", "phase": "idle", "message": "连接后保持停止", "owner": None,
                                 "targets": {"3": 0, "4": 0}, "elapsed_s": 0.0, "remaining_s": 0.0}
            self._state = {"state": "connecting", "message": "正在连接；成功打开后发送 CTRL STOP",
                           "port": None, "frames_received": 0, "stop_sent": False,
                           "stop_acknowledged": False, "fault": None, "motors": None,
                           "frame_at": None, "motor_sample_at": {}, "motor_counts": {},
                           "clear_state": "idle", "clear_message": "尚未请求清除"}
            self._worker = threading.Thread(target=self._run, args=(self._stop,), daemon=True,
                                            name="c-board-telemetry")
            self._worker.start()
        return self.status()

    def clear_usb_closed(self):
        """Compatibility endpoint: retain its original USB-only restriction."""
        return self._request_clear(only_fault="USB_CLOSED")

    def clear_fault(self):
        """Explicitly acknowledge one recoverable fault, always staying stopped."""
        return self._request_clear()

    def _request_clear(self, only_fault=None):
        with self._lock:
            if (self._state["state"] != "connected" or not self._worker
                    or not self._worker.is_alive() or self._stop.is_set()):
                raise MotorError("请先连接 C 板并等待遥测")
            if not self._state["stop_acknowledged"]:
                raise MotorError("等待连接时的 STOP 确认后再清除")
            fault = self._state.get("fault")
            if only_fault is not None and fault != only_fault:
                raise MotorError("仅允许清除 USB_CLOSED；当前故障不是 USB_CLOSED")
            if fault not in CLEARABLE_FAULTS:
                raise MotorError("当前故障不支持网页清除：" + str(fault))
            if self._clear is not None:
                raise MotorError("故障清除正在进行")
            if self._motion is not None:
                raise MotorError("请先停止运动，再清除故障")
            if self._state["frame_at"] is None or self._clock() - self._state["frame_at"] > CLEAR_FRAME_GAP_S:
                raise MotorError("C 板遥测已过期，请等待新鲜反馈")
            trip = tuple(self._state[key] for key in ("trip_mask", "trip_current", "trip_ms"))
            self._clear = {"phase": "requested", "deadline": time.monotonic() + CLEAR_TIMEOUT_S,
                           "fault": fault, "trip": trip, "legacy_usb": only_fault is not None}
            self._state.update(clear_state="checking", clear_message="正在发送 STOP 并检查连续静止反馈")
        return self.status()

    @staticmethod
    def _targets(targets):
        if not isinstance(targets, dict) or not targets or any(str(k) not in ("3", "4") for k in targets):
            raise MotorError("只允许指定 M3、M4 的目标转速")
        result = {"3": 0, "4": 0}
        for key, value in targets.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value or abs(value) > 500:
                raise MotorError("目标转速必须为 −500 至 500 之间的整数 rpm")
            result[str(key)] = int(value)
        return result

    def start_test(self, motor, rpm=60, duration_s=3):
        """Queue a single-channel test. The other channel remains zero."""
        if isinstance(motor, bool) or str(motor) not in ("3", "4"):
            raise MotorError("单电机测试只支持 M3 或 M4")
        return self.start_motion({str(motor): rpm}, duration_s, owner="manual")

    def start_motion(self, targets, duration_s, owner="manual"):
        """Queue motion; paper-owned motion additionally needs an external lease.

        Preparation never drives. Call update_motion at <0.5 s intervals even
        during preparation when owner='paper'; duration starts at the first
        SPEED write. Only the serial owner actually sends any commands.
        """
        targets = self._targets(targets)
        if owner not in ("manual", "paper"):
            raise MotorError("未知运动来源")
        maximum = 10.0 if owner == "manual" else 300.0
        if isinstance(duration_s, bool) or not isinstance(duration_s, (int, float)) or not math.isfinite(duration_s) or not 0.2 <= duration_s <= maximum:
            raise MotorError(f"本次运动时长须在 0.2 至 {maximum:g} 秒之间")
        if owner == "manual" and not any(targets.values()):
            raise MotorError("目标全为零；请使用停止按钮")
        with self._lock:
            now = self._clock()
            if self._state["state"] != "connected" or not self._worker or not self._worker.is_alive() or self._stop.is_set():
                raise MotorError("请先连接 C 板并等待遥测")
            if self._clear is not None or self._motion is not None:
                raise MotorError("已有清除或运动操作正在进行")
            if not self._state["stop_acknowledged"]:
                raise MotorError("尚未收到 STOP 确认")
            if self._state["fault"] != "NONE":
                raise MotorError("板端存在故障：" + str(self._state["fault"]))
            if self._state["frame_at"] is None or now - self._state["frame_at"] > MOTION_FRAME_GAP_S:
                raise MotorError("C 板遥测已过期")
            self._motion = {"phase": "requested", "owner": owner, "targets": targets, "duration_s": float(duration_s),
                            "requested_at": now, "prepare_deadline": now + MOTION_PREPARE_TIMEOUT_S,
                            "external_until": now + MOTION_EXTERNAL_LEASE_S, "started_at": None,
                            "message": "等待 STOP、连续静止反馈和 BENCH 模式确认", "failed": False}
        return self.status()

    def update_motion(self, targets, owner="paper"):
        targets = self._targets(targets)
        with self._lock:
            op = self._motion
            if op is None or op["owner"] != owner or owner != "paper" or op["phase"] in ("stop_requested", "stopping", "stop_verifying"):
                raise MotorError("没有可更新的论文控制运动")
            now = self._clock()
            # A late heartbeat cannot revive an expired operation.
            if now > op["external_until"]:
                self._motion_stop_locked("策略续租超时", failed=True)
                raise MotorError("策略续租已超时，正在停止")
            op.update(targets=targets, external_until=now + MOTION_EXTERNAL_LEASE_S)
        return self.status()

    def stop_motion(self, reason="网页停止"):
        """Stop is always available, including during preparation or CLEAR."""
        with self._lock:
            if self._clear is not None:
                self._clear = None
                self._state.update(clear_state="failed", clear_message="清除已被停止请求取消")
            if self._state["state"] != "connected" or not self._worker or not self._worker.is_alive():
                raise MotorError("C 板未连接，无法确认 STOP")
            if self._motion is None:
                self._motion = {"phase": "stop_requested", "owner": "manual", "targets": {"3": 0, "4": 0},
                                "duration_s": 0.0, "started_at": None, "message": str(reason), "failed": False}
            else:
                self._motion_stop_locked(str(reason))
        return self.status()

    def disconnect(self):
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                self._stop.set()
                self._state.update(state="disconnecting", message="正在断开；关闭 USB 可能使板端锁存 USB_CLOSED")
            else:
                self._state.update(state="disconnected", message="已断开；关闭 USB 可能使板端锁存 USB_CLOSED")
        return self.status()

    def close(self):
        self.disconnect()
        with self._lock:
            worker = self._worker
        if worker and worker is not threading.current_thread():
            worker.join(timeout=1.0)

    def status(self):
        with self._lock:
            result = copy.deepcopy(self._state)
            op = copy.deepcopy(self._motion)
            motion = copy.deepcopy(self._last_motion)
        now = self._clock()
        if op is not None:
            phase = op["phase"]
            running = op["started_at"] is not None
            elapsed = max(0.0, now - op["started_at"]) if running else 0.0
            motion = {"state": "stopping" if phase in ("stop_requested", "stopping", "stop_verifying") else "running" if running else "preparing",
                      "phase": phase, "owner": op["owner"], "targets": op["targets"], "message": op["message"],
                      "duration_s": op["duration_s"], "elapsed_s": elapsed, "remaining_s": max(0.0, op["duration_s"] - elapsed),
                      "started_monotonic_s": op["started_at"],
                      "last_speed_sent_monotonic_s": op.get("last_speed_sent_monotonic_s"),
                      "command_sent_monotonic_s": op.get("last_speed_sent_monotonic_s"),
                      "last_speed_targets": op.get("last_speed_targets"),
                      "last_speed_ack_monotonic_s": op.get("last_speed_ack_monotonic_s"),
                      "last_speed_ack_targets": op.get("last_speed_ack_targets")}
        result["motion"] = motion
        frame_at = result.pop("frame_at")
        sample_times = result.pop("motor_sample_at")
        result.pop("motor_counts")
        frame_age = None if frame_at is None else max(0.0, now - frame_at)
        result["snapshot_monotonic_s"] = now
        result["frame_receive_monotonic_s"] = frame_at
        result["frame_age_s"] = frame_age
        result["frame_fresh"] = result["state"] == "connected" and frame_age is not None and frame_age <= FRAME_MAX_AGE_S
        for channel, m in (result["motors"] or {}).items():
            sample_at = sample_times.get(channel)
            age = None if sample_at is None else max(0.0, now - sample_at)
            m["feedback_estimated_monotonic_s"] = sample_at
            m["feedback_age_s"] = age
            m["fresh"] = bool(result["frame_fresh"] and m["seen"] == 1 and age is not None and age <= FEEDBACK_MAX_AGE_S)
        result["active"] = result["state"] == "connected"
        result["clear_supported"] = result.get("fault") in CLEARABLE_FAULTS
        result["notice"] = "运动仅由测试/实验按钮启动；超时或故障自动 STOP。断开可能锁存 USB_CLOSED，仅手动清除。转速为转子相对定子。"
        return result

    def _motion_stop_locked(self, message, failed=False):
        op = self._motion
        if op is None:
            return
        if failed:
            op["failed"] = True
        if op["phase"] not in ("stop_requested", "stopping", "stop_verifying"):
            op.update(phase="stop_requested", message=message)

    def _motion_finish_locked(self, confirmed, message=None):
        op = self._motion
        if op is None:
            return
        elapsed = 0.0 if op["started_at"] is None else max(0.0, self._clock() - op["started_at"])
        self._last_motion = {"state": "failed" if op["failed"] or not confirmed else "completed", "phase": "idle",
                             "owner": op["owner"], "targets": {"3": 0, "4": 0}, "elapsed_s": elapsed,
                             "duration_s": op["duration_s"], "started_monotonic_s": op["started_at"],
                             "remaining_s": 0.0, "stop_confirmed": confirmed,
                             "message": message or (op["message"] + "；STOP 已确认，目标、输出和转速归零"),
                             "last_speed_sent_monotonic_s": op.get("last_speed_sent_monotonic_s"),
                             "command_sent_monotonic_s": op.get("last_speed_sent_monotonic_s"),
                             "last_speed_targets": op.get("last_speed_targets"),
                             "last_speed_ack_monotonic_s": op.get("last_speed_ack_monotonic_s"),
                             "last_speed_ack_targets": op.get("last_speed_ack_targets")}
        self._motion = None

    @staticmethod
    def _motion_guard(frame, stopped=False):
        h, motors = frame["header"], frame["motors"]
        if h["fault"] != "NONE":
            raise MotorError("板端故障：" + h["fault"])
        if h["can_ready"] != "1" or h["can_healthy"] != "1" or _int(h["ESR"]) != 0:
            raise MotorError("CAN 未就绪或存在错误")
        if (_int(h["active"]) != 0x0c or _int(h["current_trip"]) != 7000 or _int(h["limit"]) != 500
                or _int(h["output_limit"]) != 25000 or _int(h["lease_ms"]) != 300
                or _int(h["trip_mask"]) != 0 or _int(h["trip_current"]) != 0):
            raise MotorError("固件限制或故障采样不符")
        for ch, m in motors.items():
            if ch in ("1", "2") or stopped:
                if any(m[key] != 0 for key in ("target", "setI", "setV")):
                    raise MotorError("M" + ch + " 目标或输出未归零")
            if ch in ("3", "4"):
                if m["seen"] != 1 or m["rx"] == 0 or m["age_ms"] >= 100:
                    raise MotorError("M" + ch + " 反馈缺失或过期")
                if abs(m["torque"]) >= 7000 or not 0 <= m["temp"] < 80:
                    raise MotorError("M" + ch + " 电流或温度超限")
                if stopped and (abs(m["spd"]) > 2 or abs(m["torque"]) >= 1000 or m["temp"] >= 50):
                    raise MotorError("M" + ch + " 未静止或基线电流、温度异常")

    def _motion_frame_locked(self, frame, now):
        op = self._motion
        if op is None or op["phase"] in ("requested", "waiting_stop", "stop_requested", "stopping"):
            return
        phase = op["phase"]
        motors = frame["motors"]
        counts = tuple(motors[ch]["rx"] for ch in ("3", "4"))
        if phase == "stop_verifying":
            # Zero output alone is insufficient to claim the mechanism stopped.
            fresh = all(motors[ch]["seen"] == 1 and motors[ch]["age_ms"] < 100 and motors[ch]["rx"] > 0 for ch in ("3", "4"))
            newer = all(0 < ((b-a) & 0xffffffff) < 0x80000000 for a, b in zip(op["stop_counts"], counts))
            zero = all(all(m[key] == 0 for key in ("target", "setI", "setV")) for m in motors.values())
            still = all(abs(motors[ch]["spd"]) <= 2 for ch in ("3", "4"))
            if fresh and newer and zero and still:
                self._motion_finish_locked(True)
            return
        try:
            preparing = phase in ("baseline", "send_mode", "waiting_mode", "verify_mode", "send_speed") and op["started_at"] is None
            self._motion_guard(frame, stopped=preparing)
            if "last_frame_at" in op:
                if now - op["last_frame_at"] > MOTION_FRAME_GAP_S:
                    raise MotorError("连续遥测中断")
                if not all(0 < ((b-a) & 0xffffffff) < 0x80000000 for a, b in zip(op["counts"], counts)):
                    raise MotorError("M3/M4 反馈计数未推进")
            op.update(last_frame_at=now, counts=counts)
            if phase == "baseline":
                if "baseline_at" not in op:
                    op.update(baseline_at=now, samples=0)
                op["samples"] += 1
                if op["samples"] >= 8 and now - op["baseline_at"] >= MOTION_BASELINE_S:
                    op["phase"] = "send_mode"
            elif phase == "verify_mode":
                if frame["header"]["mode"] != "BENCH":
                    raise MotorError("BENCH 模式遥测确认失败")
                op["phase"] = "send_speed"
            elif op["started_at"] is not None:
                if frame["header"]["mode"] != "BENCH":
                    raise MotorError("运动期间模式改变")
                # A frame already queued before SPEED/ACK can reflect the
                # previous command. Once a full report period has elapsed,
                # exact target echo is required; inactive channels always zero.
                sent = op.get("last_speed_targets", {})
                for ch in ("3", "4"):
                    if sent.get(ch) == 0 and op.get("last_speed_ack_targets", {}).get(ch) == 0 and motors[ch]["target"] == 0 and (motors[ch]["setI"] != 0 or motors[ch]["setV"] != 0):
                        raise MotorError("非激活 M" + ch + " 输出不为零")
                if op.get("target_changed_at") is not None and now - op["target_changed_at"] > 0.22:
                    if any(motors[ch]["target"] != sent[ch] for ch in ("3", "4")):
                        raise MotorError("目标转速回显不符")
        except (MotorError, ValueError, KeyError) as exc:
            self._motion_stop_locked(str(exc), failed=True)

    def _motion_tick(self, port):
        # All writes take this lock so a simultaneous stop/update cannot race
        # between the final guard and issuing a nonzero SPEED command.
        with self._lock:
            op = self._motion
            if op is None:
                return
            now = self._clock()
            phase = op["phase"]
            stopping = phase in ("stop_requested", "stopping", "stop_verifying")
            if not stopping:
                if self._stop.is_set():
                    self._motion_stop_locked("连接关闭")
                elif op["owner"] == "paper" and now > op["external_until"]:
                    self._motion_stop_locked("策略续租超时", failed=True)
                elif op["started_at"] is not None and now - op["started_at"] >= op["duration_s"]:
                    self._motion_stop_locked("设定时长已结束")
                elif op["started_at"] is None and now > op["prepare_deadline"]:
                    self._motion_stop_locked("运动准备超时", failed=True)
                elif now - self._state["frame_at"] > MOTION_FRAME_GAP_S:
                    self._motion_stop_locked("C 板遥测超时", failed=True)
                elif phase in ("waiting_stop", "waiting_mode", "waiting_speed") and now > op["ack_deadline"]:
                    self._motion_stop_locked("命令确认超时", failed=True)
            phase = op["phase"]
            if phase == "stop_requested":
                op.update(phase="stopping", stop_deadline=now + MOTION_STOP_TIMEOUT_S, last_stop_at=now,
                          stop_counts=tuple(self._state["motor_counts"].get(ch, 0) for ch in ("3", "4")))
                self._state["stop_acknowledged"] = False
                self._write_command(port, "CTRL STOP")
            elif phase in ("stopping", "stop_verifying"):
                if now > op["stop_deadline"]:
                    self._write_command(port, "CTRL STOP")
                    self._motion_finish_locked(False, op["message"] + "；已发送 STOP，但停止状态未确认，请检查现场")
                elif now - op["last_stop_at"] >= MOTION_RENEW_S:
                    op["last_stop_at"] = now
                    self._write_command(port, "CTRL STOP")
            elif phase == "requested":
                op.update(phase="waiting_stop", ack_deadline=now + MOTION_ACK_TIMEOUT_S)
                self._state["stop_acknowledged"] = False
                self._write_command(port, "CTRL STOP")
            elif phase == "send_mode":
                op.update(phase="waiting_mode", ack_deadline=now + MOTION_ACK_TIMEOUT_S)
                self._write_command(port, "CTRL MODE BENCH")
            elif phase == "send_speed" or (phase == "running" and now - op["last_speed_sent_monotonic_s"] >= MOTION_RENEW_S):
                targets = dict(op["targets"])
                command = f"CTRL SPEED M1=0 M2=0 M3={targets['3']} M4={targets['4']}"
                if targets != op.get("last_speed_targets"):
                    op["target_changed_at"] = now
                op.update(phase="waiting_speed", ack_deadline=now + MOTION_ACK_TIMEOUT_S,
                          last_speed_sent_monotonic_s=now, last_speed_targets=targets,
                          expected_ack=f"CTRL OK M1=0 M2=0 M3={targets['3']} M4={targets['4']}",
                          message="正在执行；定时、遥测和通信保护独立工作")
                if op["started_at"] is None:
                    op["started_at"] = now
                self._state["stop_acknowledged"] = False
                self._write_command(port, command)

    @staticmethod
    def _write_command(port, command):
        payload = (command + "\n").encode("ascii")
        if port.write(payload) != len(payload):
            raise MotorError(command.split()[1] + " 写入不完整")

    def _clear_finish(self, state, message, operation=None):
        with self._lock:
            if operation is not None and self._clear is not operation:
                return
            self._clear = None
            self._state.update(clear_state=state, clear_message=message)

    def _clear_guard(self, frame, fault, trip=None):
        h, motors = frame["header"], frame["motors"]
        if h["fault"] != fault:
            raise MotorError("故障状态已变化：" + h["fault"])
        if h["can_ready"] != "1" or h["can_healthy"] != "1" or _int(h["ESR"]) != 0:
            raise MotorError("CAN 未就绪或存在错误")
        if (_int(h["active"]) != 0x0c or _int(h["current_trip"]) != 7000
                or _int(h["limit"]) != 500 or _int(h["output_limit"]) != 25000
                or _int(h["lease_ms"]) != 300):
            raise MotorError("固件配置不符")
        current_trip = tuple(_int(h[key]) for key in ("trip_mask", "trip_current", "trip_ms"))
        if trip is not None and fault != "NONE" and current_trip != trip:
            raise MotorError("故障采样已变化，请检查最新故障后重新操作")
        if (trip is None or fault == "NONE") and (current_trip[0] != 0 or current_trip[1] != 0):
            raise MotorError("存在电机故障采样，拒绝清除")
        if fault == "NONE" and current_trip[2] != 0:
            raise MotorError("清除后故障时间未归零，结果未确认")
        channels = []
        for ch, m in motors.items():
            if m["target"] != 0 or m["setI"] != 0 or m["setV"] != 0:
                raise MotorError("电机目标或输出不为零")
            if ch in ("3", "4") or m["seen"] == 1 or (current_trip[0] & (1 << (int(ch)-1))):
                channels.append(ch)
                if m["seen"] != 1 or m["rx"] == 0 or m["age_ms"] >= 50:
                    raise MotorError("M" + ch + " 反馈缺失或过期")
                if abs(m["spd"]) > 2 or abs(m["torque"]) >= 1000 or not 0 <= m["temp"] < 50:
                    raise MotorError("M" + ch + " 未静止或电流、温度异常")
        return tuple(channels)

    def _clear_advance_frame(self, frame):
        # Called only by the serial owner thread.
        with self._lock:
            operation = self._clear
        if operation is None or operation["phase"] not in ("checking", "send_clear", "waiting_ack", "verifying"):
            return
        try:
            phase = operation["phase"]
            if phase in ("checking", "send_clear"):
                channels = self._clear_guard(frame, operation["fault"],
                                             None if operation["legacy_usb"] else operation["trip"])
                now = time.monotonic()
                counts = tuple(frame["motors"][ch]["rx"] for ch in channels)
                if "last_at" in operation:
                    if channels != operation["channels"]:
                        raise MotorError("电机连接状态变化，请重新检查")
                    if now - operation["last_at"] > CLEAR_FRAME_GAP_S:
                        raise MotorError("连续遥测中断")
                    if not all(0 < ((b-a) & 0xffffffff) < 0x80000000 for a, b in zip(operation["counts"], counts)):
                        raise MotorError("电机反馈计数未更新")
                else:
                    operation["first_at"] = now
                    operation["samples"] = 0
                operation.update(last_at=now, counts=counts, channels=channels, samples=operation["samples"] + 1)
                if operation["samples"] >= 8 and now - operation["first_at"] >= CLEAR_BASELINE_S:
                    operation["phase"] = "send_clear"
            elif phase == "waiting_ack":
                # Earlier telemetry may still be in flight when the command is sent.
                observed_fault = frame["header"]["fault"]
                if observed_fault not in (operation["fault"], "NONE"):
                    raise MotorError("出现其他故障，清除结果无法确认")
                self._clear_guard(frame, observed_fault,
                                  None if operation["legacy_usb"] else operation["trip"])
            else:
                channels = self._clear_guard(frame, "NONE")
                if channels != operation["channels"]:
                    raise MotorError("清除后电机连接状态变化，结果未确认")
                counts = tuple(frame["motors"][ch]["rx"] for ch in channels)
                if not all(0 < ((b-a) & 0xffffffff) < 0x80000000 for a, b in zip(operation["counts"], counts)):
                    raise MotorError("清除后电机反馈未更新，结果未确认")
                self._clear_finish("succeeded", operation["fault"] + " 已清除；保持停止，目标与输出均为零", operation)
        except (MotorError, ValueError, KeyError) as exc:
            self._clear_finish("failed", str(exc), operation)

    def _clear_tick(self, port):
        with self._lock:
            op = self._clear
        if op is None:
            return
        if time.monotonic() > op["deadline"]:
            self._clear_finish("failed", "清除超时；请查看最新故障状态，若命令已发送则结果未确认", op)
            return
        if op["phase"] == "requested":
            with self._lock:
                if self._stop.is_set() or self._clear is not op:
                    return
                op["phase"] = "waiting_stop"
                self._state["stop_acknowledged"] = False
                if port.write(b"CTRL STOP\n") != len(b"CTRL STOP\n"):
                    raise MotorError("CTRL STOP 写入不完整")
        elif op["phase"] == "send_clear":
            if time.monotonic() - op["last_at"] > CLEAR_FRAME_GAP_S:
                self._clear_finish("failed", "清除前遥测已过期", op)
                return
            with self._lock:
                if self._stop.is_set() or self._clear is not op:
                    return
                op["phase"] = "waiting_ack"
                self._state.update(clear_state="waiting_ack", clear_message="已发送 CLEAR，等待板端确认")
                if port.write(b"CTRL CLEAR\n") != len(b"CTRL CLEAR\n"):
                    raise MotorError("CTRL CLEAR 写入不完整；结果未确认")

    def _accept(self, line):
        if line == "CTRL OK STOP":
            with self._lock:
                self._state["stop_acknowledged"] = True
                if self._clear and self._clear["phase"] == "waiting_stop":
                    self._clear["phase"] = "checking"
                if self._motion:
                    if self._motion["phase"] == "waiting_stop":
                        self._motion["phase"] = "baseline"
                    elif self._motion["phase"] == "stopping":
                        self._motion["phase"] = "stop_verifying"
                    elif self._motion["started_at"] is not None and self._motion["phase"] not in ("stop_requested", "stop_verifying"):
                        self._motion_stop_locked("运行中收到额外 STOP，保持停止", failed=True)
            return
        if line == "CTRL OK CLEAR":
            with self._lock:
                if self._clear and self._clear["phase"] == "waiting_ack":
                    self._clear["phase"] = "verifying"
                    self._state.update(clear_state="verifying", clear_message="已收到 CLEAR 确认，等待新鲜状态")
            return
        if line.startswith("CTRL OK MODE") or line.startswith("CTRL OK M1="):
            with self._lock:
                op = self._motion
                if op and op["phase"] == "waiting_mode":
                    if line == "CTRL OK MODE BENCH":
                        op["phase"] = "verify_mode"
                    else:
                        self._motion_stop_locked("模式确认不符：" + line, failed=True)
                elif op and op["phase"] == "waiting_speed":
                    if line == op["expected_ack"]:
                        op.update(phase="running", last_speed_ack_monotonic_s=self._clock(),
                                  last_speed_ack_targets=dict(op["last_speed_targets"]))
                    else:
                        self._motion_stop_locked("速度命令确认不符：" + line, failed=True)
            return
        if line.startswith("CTRL ERR"):
            with self._lock:
                pending = self._clear
                if self._motion:
                    self._motion_stop_locked("板端拒绝命令：" + line, failed=True)
            if pending is not None:
                self._clear_finish("failed", "板端拒绝命令：" + line, pending)
            return
        frame = parse_frame(line)
        if frame is None:
            return
        now = self._clock()
        h = frame["header"]
        with self._lock:
            state = self._state
            for ch in ("3", "4"):
                m = frame["motors"][ch]
                previous = state["motor_counts"].get(ch)
                count = m["rx"]
                fresh_count = previous is None or 0 < ((count - previous) & 0xffffffff) < 0x80000000
                if m["seen"] != 1 or count == 0:
                    state["motor_sample_at"].pop(ch, None)
                elif fresh_count:
                    state["motor_counts"][ch] = count
                    state["motor_sample_at"][ch] = now - m["age_ms"] / 1000
                elif ch in state["motor_sample_at"]:
                    state["motor_sample_at"][ch] = min(state["motor_sample_at"][ch], now - m["age_ms"] / 1000)
            state.update(state="connected", message="已接收 C 板遥测", frame_at=now,
                         frames_received=state["frames_received"] + 1,
                         fault=h["fault"], can_ready=h["can_ready"] == "1",
                         can_healthy=h["can_healthy"] == "1", can_esr=h["ESR"],
                         mode=h["mode"], active_mask=h["active"], roll=h["roll"],
                         speed_limit=_int(h["limit"]), output_limit=_int(h["output_limit"]),
                         current_trip=_int(h["current_trip"]), lease_ms=_int(h["lease_ms"]),
                         trip_mask=_int(h["trip_mask"]), trip_current=_int(h["trip_current"]), trip_ms=_int(h["trip_ms"]),
                         directions={"3": _int(h["dir3"]), "4": _int(h["dir4"])},
                         motors={ch: frame["motors"][ch] for ch in ("3", "4")})
            self._motion_frame_locked(frame, now)
        self._clear_advance_frame(frame)

    def _run(self, stop):
        port = None
        error = None
        try:
            matches = [p for p in self._port_lister() if (getattr(p, "vid", None), getattr(p, "pid", None)) == (0x0483, 0x5740)
                       and str(getattr(p, "serial_number", "")).upper() == BOARD_SERIAL]
            if len(matches) != 1:
                raise MotorError("未唯一识别指定 C 板（0483:5740，序列 206B377B5948）")
            with self._lock:
                self._state["port"] = matches[0].device
            if stop.is_set():
                return
            port = self._serial_factory(matches[0].device)
            if stop.is_set():
                return
            written = port.write(b"CTRL STOP\n")
            if written != len(b"CTRL STOP\n"):
                raise MotorError("CTRL STOP 写入不完整")
            with self._lock:
                self._state.update(stop_sent=True, message="已发送 CTRL STOP，等待确认及遥测")
            pending = b""
            while not stop.is_set():
                self._motion_tick(port)
                self._clear_tick(port)
                chunk = port.read(4096)
                if stop.is_set():
                    break
                if not chunk:
                    continue
                pending += chunk
                if len(pending) > 65536:
                    raise MotorError("串口遥测超过长度限制")
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    self._accept(line.decode("ascii", errors="strict").strip())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if port is not None:
                # Disconnect/error must issue STOP before releasing the sole
                # serial owner. Never trust a previous speed renewal to stop.
                with self._lock:
                    had_motion = self._motion is not None
                    if had_motion:
                        self._motion_stop_locked("串口关闭，停止运动", failed=bool(error))
                if had_motion:
                    try:
                        self._write_command(port, "CTRL STOP")
                        with self._lock:
                            op = self._motion
                            op.update(phase="stopping", stop_counts=tuple(self._state["motor_counts"].get(ch, 0) for ch in ("3", "4")))
                            self._state["stop_acknowledged"] = False
                        # Bound by real wall time even when tests inject a clock.
                        stop_deadline = time.monotonic() + 0.7
                        pending = b""
                        while time.monotonic() < stop_deadline:
                            with self._lock:
                                if self._motion is None:
                                    break
                            pending += port.read(4096)
                            if len(pending) > 65536:
                                break
                            while b"\n" in pending:
                                line, pending = pending.split(b"\n", 1)
                                self._accept(line.decode("ascii", errors="strict").strip())
                    except Exception as exc:
                        error = error or f"停止确认失败：{type(exc).__name__}: {exc}"
                    finally:
                        with self._lock:
                            if self._motion is not None:
                                self._motion_finish_locked(False, "已尝试 STOP，串口关闭前未能确认停止")
                try:
                    port.close()
                except Exception as exc:
                    error = error or f"关闭串口失败：{type(exc).__name__}: {exc}"
            with self._lock:
                if self._clear is not None:
                    self._clear = None
                    self._state.update(clear_state="failed", clear_message="串口已断开，清除结果未确认")
                if error:
                    self._state.update(state="error", message=f"C 板连接中断：{error}；关闭 USB 可能锁存 USB_CLOSED")
                else:
                    self._state.update(state="disconnected", message="已断开；关闭 USB 可能使板端锁存 USB_CLOSED")
