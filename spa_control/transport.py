"""Direct robot TCP transport and a host-side command lease watchdog.

Only the caller can issue motion; the watchdog may only stop it.  This module
uses ordinary TCP sockets, so HTTP proxy settings are never consulted.
"""
from __future__ import annotations

from collections import deque
import math
import re
import socket
import threading
import time

from .core import command_line


_MOTOR = re.compile(r"^M([1-6])\b", re.IGNORECASE)
_FIELD = re.compile(r"\b([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([^\s|,]+)")
_SOCKET_TIMEOUT_S = 0.25


def motor_feedback_state(motors, channel, max_age_s, current_limit_raw, now=None) -> tuple[bool, str]:
    """Explain readiness using complete measured telemetry, without network I/O."""
    now = time.monotonic() if now is None else now
    def finite_number(value):
        return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
    if type(channel) is not int or not 1 <= channel <= 6:
        return False, "电机通道编号无效"
    prefix = f"M{channel} "
    if (not finite_number(now) or not finite_number(max_age_s) or max_age_s <= 0
            or not finite_number(current_limit_raw) or current_limit_raw <= 0):
        return False, prefix + "遥测检查参数无效"
    motor = motors.get(channel)
    if not isinstance(motor, dict) or not motor:
        return False, prefix + "未收到该通道遥测"
    if motor.get("device_fault"):
        return False, prefix + f"设备故障锁定（{motor['device_fault']}），请检查后重新连接"
    if motor.get("feedback_valid") is False:
        return False, prefix + "设备未提供有效 CAN 反馈"
    if not finite_number(motor.get("speed")):
        return False, prefix + "速度缺失或无效"
    if not finite_number(motor.get("current_raw")):
        return False, prefix + "电流缺失或无效"
    if not finite_number(motor.get("last_seen")):
        return False, prefix + "时间戳缺失或无效"
    age = now - motor["last_seen"]
    if age < 0:
        return False, prefix + "遥测时间戳不属于当前时基"
    if age > max_age_s:
        return False, prefix + f"遥测已过时（{age:.2f} 秒）"
    if abs(motor["current_raw"]) >= current_limit_raw:
        return False, prefix + f"电流超限（{motor['current_raw']:g}，限值 {current_limit_raw:g}）"
    return True, prefix + "遥测就绪"


class RobotLink:
    """One telemetry reader, one socket, serialized sends, and no action replay."""

    def __init__(self, host: str, port: int):
        self.host, self.port = str(host), int(port)
        self._state_lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._shutdown = threading.Event()
        self._threads: list[threading.Thread] = []
        self._generation = 0
        self._stopping = False
        self._closing = False
        self._lease_fault = False
        self._device_fault: str | None = None
        self._lease_deadline: float | None = None
        self._motors: dict[int, dict] = {}
        self._feedback_rx: dict[int, int] = {}
        self._extended_feedback_channels: set[int] = set()
        self._status = "未连接"
        self._last_feedback = ""
        self._connected_at: float | None = None
        self._received_bytes = 0
        self._received_lines = 0
        self._last_receive_at: float | None = None
        self._reset_send_diagnostics()
        self.lines: deque[tuple[float, str]] = deque(maxlen=200)

    def _reset_send_diagnostics(self):
        self._sent_motion_count = 0
        self._sent_stop_count = 0
        self._last_motion_send_at = None
        self._previous_motion_send_at = None
        self._last_motion_gap_s = None
        self._max_motion_gap_s = None
        self._last_send_duration_s = None
        self._last_tx_line = ""
        self._last_transport_error = None

    def _record_error(self, operation, exc=None, message=None):
        with self._state_lock:
            self._last_transport_error = {
                "operation": operation, "type": type(exc).__name__ if exc is not None else "PeerClosed",
                "message": str(exc) if exc is not None else message,
                "errno": getattr(exc, "errno", None), "winerror": getattr(exc, "winerror", None),
                "monotonic_s": time.monotonic(),
            }

    @property
    def connected(self) -> bool:
        with self._state_lock:
            return self._socket is not None

    @property
    def status(self) -> str:
        with self._state_lock:
            return self._status

    @property
    def last_feedback(self) -> str:
        with self._state_lock:
            return self._last_feedback

    @property
    def lease_fault(self) -> bool:
        with self._state_lock:
            return self._lease_fault

    def snapshot(self) -> dict[int, dict]:
        with self._state_lock:
            result = {channel: dict(values) for channel, values in self._motors.items()}
            if self._device_fault:
                for values in result.values():
                    values["device_fault"] = self._device_fault
                    # Session consumers that predate device_fault must also
                    # fail closed instead of using otherwise fresh samples.
                    values["last_seen"] = None
            return result

    def diagnostics(self) -> dict:
        """Return reception evidence only; a live socket need not have telemetry."""
        with self._state_lock:
            now = time.monotonic()
            connected = self._socket is not None
            return {
                "connected": connected,
                "connected_for_s": max(0.0, now - self._connected_at)
                if connected and self._connected_at is not None else 0.0,
                "received_bytes": self._received_bytes,
                "received_lines": self._received_lines,
                "last_receive_age_s": max(0.0, now - self._last_receive_at)
                if self._last_receive_at is not None else None,
                "last_line": self._last_feedback,
                "recent_lines": [line for _, line in self.lines],
                "status": self._status,
                "lease_fault": self._lease_fault,
                "device_fault": self._device_fault,
                "sent_motion_count": self._sent_motion_count,
                "sent_stop_count": self._sent_stop_count,
                "last_motion_send_age_s": max(0.0, now - self._last_motion_send_at)
                if self._last_motion_send_at is not None else None,
                "last_motion_gap_s": self._last_motion_gap_s,
                "max_motion_gap_s": self._max_motion_gap_s,
                "last_send_duration_s": self._last_send_duration_s,
                "last_tx_line": self._last_tx_line,
                "last_transport_error": dict(self._last_transport_error) if self._last_transport_error else None,
            }

    def connect(self) -> None:
        """Connect directly; never arm or issue a speed command on connection."""
        with self._lifecycle_lock:
            if self.connected:
                return
            with self._state_lock:
                self._status = "连接中"
            try:
                sock = socket.create_connection((self.host, self.port), timeout=0.75)
                sock.settimeout(_SOCKET_TIMEOUT_S)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError as exc:
                with self._state_lock:
                    self._status = f"连接失败: {exc}"
                raise
            shutdown = threading.Event()
            with self._state_lock:
                self._generation += 1
                self._socket = sock
                self._shutdown = shutdown
                self._lease_deadline = None
                self._stopping = False
                self._closing = False
                self._lease_fault = False
                self._device_fault = None
                self._motors.clear()
                self._feedback_rx.clear()
                self._extended_feedback_channels.clear()
                self.lines.clear()
                self._last_feedback = ""
                self._connected_at = time.monotonic()
                self._received_bytes = 0
                self._received_lines = 0
                self._last_receive_at = None
                self._reset_send_diagnostics()
                self._status = f"已直连 {self.host}:{self.port}"
            self._threads = [
                threading.Thread(target=self._read_loop, args=(sock, shutdown), daemon=True, name="spa-robot-reader"),
                threading.Thread(target=self._watchdog_loop, args=(sock, shutdown), daemon=True, name="spa-command-lease"),
            ]
            for worker in self._threads:
                worker.start()

    def send_speeds(self, speeds: list[int], lease_s: float = 0.4) -> float:
        """Issue one command; reject commands queued before a stop request.

        A successful nonzero send must be renewed by a *new* caller command
        before its lease ends. A zero command clears the lease.
        """
        payload = command_line(speeds)
        is_motion = any(speeds)
        if not math.isfinite(lease_s) or not 0.05 <= lease_s <= 2.0:
            raise ValueError("控制租约必须在 0.05–2 秒之间")
        with self._state_lock:
            generation = self._generation
            expected_socket = self._socket
            if expected_socket is None:
                raise ConnectionError("机器人未连接")
            if self._stopping or self._closing:
                raise InterruptedError("停机处理中，命令已取消")
            if is_motion and self._lease_fault:
                raise InterruptedError("控制租约已超时锁定，请断开后重新连接")
            if is_motion and self._device_fault:
                raise InterruptedError(f"设备故障锁定（{self._device_fault}），请检查后重新连接")
        with self._send_lock:
            with self._state_lock:
                if generation != self._generation or self._stopping or self._closing:
                    raise InterruptedError("命令已被停机取消")
                if self._socket is not expected_socket:
                    raise ConnectionError("机器人连接已变化")
                if is_motion and self._lease_fault:
                    raise InterruptedError("控制租约已超时锁定，请断开后重新连接")
                if is_motion and self._device_fault:
                    raise InterruptedError(f"设备故障锁定（{self._device_fault}），请检查后重新连接")
            try:
                sent_at = time.monotonic()
                expected_socket.sendall(payload)
            except OSError as exc:
                self._record_error("send", exc)
                self._disconnect(expected_socket, f"发送失败: {exc}")
                raise ConnectionError(f"机器人指令发送失败: {type(exc).__name__}: {exc}") from exc
            with self._state_lock:
                if self._socket is expected_socket:
                    self._last_send_duration_s = time.monotonic() - sent_at
                    self._last_tx_line = payload.decode("ascii").strip()
                    if is_motion:
                        self._sent_motion_count += 1
                        if self._previous_motion_send_at is None:
                            self._last_motion_gap_s = None
                            self._max_motion_gap_s = None
                        else:
                            self._last_motion_gap_s = sent_at - self._previous_motion_send_at
                            self._max_motion_gap_s = max(self._max_motion_gap_s or 0.0, self._last_motion_gap_s)
                        self._last_motion_send_at = sent_at
                        self._previous_motion_send_at = sent_at
                    else:
                        self._previous_motion_send_at = None
                # A concurrent stop has already invalidated this command and
                # will send STOP after the in-flight socket write completes.
                if generation != self._generation or self._socket is not expected_socket:
                    raise InterruptedError("发送期间已触发停机或断开")
                self._lease_deadline = sent_at + lease_s if is_motion else None
            return sent_at

    def stop(self) -> bool:
        """Invalidate pending sends immediately, then transmit CTRL STOP."""
        return self._stop()

    def _stop(self, expected_socket=None, only_if_expired: bool = False) -> bool:
        with self._state_lock:
            sock = self._socket
            if expected_socket is not None and sock is not expected_socket:
                return False
            if only_if_expired and (self._lease_deadline is None or time.monotonic() < self._lease_deadline):
                return False
            if only_if_expired:
                self._lease_fault = True
            self._generation += 1
            generation = self._generation
            self._lease_deadline = None
            self._previous_motion_send_at = None
            self._stopping = True
            if sock is None:
                self._stopping = False
                return False
        try:
            with self._send_lock:
                with self._state_lock:
                    if self._socket is not sock:
                        return False
                started = time.monotonic()
                sock.sendall(b"CTRL STOP\n")
                with self._state_lock:
                    if self._socket is sock:
                        self._sent_stop_count += 1
                        self._previous_motion_send_at = None
                        self._last_tx_line = "CTRL STOP"
                        self._last_send_duration_s = time.monotonic() - started
            if only_if_expired:
                with self._state_lock:
                    if self._socket is sock:
                        self._status = "控制超时已发送停止，需重新连接"
            return True
        except OSError as exc:
            self._record_error("stop", exc)
            self._disconnect(sock, f"停止发送失败: {exc}")
            return False
        finally:
            with self._state_lock:
                if self._generation == generation:
                    self._stopping = False

    def close(self) -> None:
        with self._lifecycle_lock:
            with self._state_lock:
                self._closing = True
            self.stop()
            with self._state_lock:
                sock = self._socket
                workers = list(self._threads)
            if sock is not None:
                self._disconnect(sock, "已断开")
            for worker in workers:
                if worker is not threading.current_thread():
                    worker.join(timeout=0.4)

    def _disconnect(self, sock: socket.socket, reason: str) -> None:
        with self._state_lock:
            if self._socket is not sock:
                return
            self._socket = None
            self._generation += 1
            self._lease_deadline = None
            self._stopping = False
            self._motors.clear()
            self._feedback_rx.clear()
            self._extended_feedback_channels.clear()
            self._status = reason
            self._shutdown.set()
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        sock.close()

    def _read_loop(self, sock: socket.socket, shutdown: threading.Event) -> None:
        pending = b""
        while not shutdown.is_set():
            try:
                data = sock.recv(8192)
                if not data:
                    if not shutdown.is_set():
                        self._record_error("receive", message="对端关闭 TCP 连接")
                        self._disconnect(sock, "机器人连接已关闭")
                    return
                with self._state_lock:
                    if self._socket is not sock:
                        return
                    self._received_bytes += len(data)
                    self._last_receive_at = time.monotonic()
                pending += data
                if len(pending) > 65536:
                    self._disconnect(sock, "遥测行超过长度限制")
                    return
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    self._consume_line(line.decode("utf-8", errors="replace").strip(), sock)
            except socket.timeout:
                continue
            except OSError as exc:
                if not shutdown.is_set():
                    self._record_error("receive", exc)
                    self._disconnect(sock, f"接收失败: {exc}")
                return

    def _watchdog_loop(self, sock: socket.socket, shutdown: threading.Event) -> None:
        while not shutdown.wait(0.02):
            self._stop(expected_socket=sock, only_if_expired=True)

    def _consume_line(self, line: str, sock: socket.socket) -> None:
        now = time.monotonic()
        updates = []
        for segment in line.split("|"):
            match = _MOTOR.match(segment.strip())
            if match is None:
                continue
            fields = {name.lower(): value for name, value in _FIELD.findall(segment)}
            values = {}
            for source, dest in (("angle", "angle"), ("temp", "temp"), ("seti", "set_current"), ("setv", "set_voltage")):
                number = self._number(fields.get(source))
                if number is not None:
                    values[dest] = number
            if "id" in fields:
                try:
                    values["can_id"] = int(fields["id"], 16 if fields["id"].lower().startswith("0x") else 10)
                except ValueError:
                    pass
            speed = self._first_number(fields, ("spd", "speed", "rotor_speed"))
            current = self._first_number(fields, ("current_raw", "torque_current", "torque", "current", "cur"))
            updates.append((int(match.group(1)), values, fields, speed, current))
        fault = self._line_fault(line)
        stop_for_fault = False
        with self._state_lock:
            if self._socket is not sock:
                return
            self._last_feedback = line
            self._received_lines += 1
            self.lines.append((now, line))
            if fault and not self._device_fault:
                # A later healthy header/OK reply cannot rearm a command
                # rejected by the device. Reconnection is an explicit action.
                self._device_fault = fault
                self._generation += 1
                self._lease_deadline = None
                self._status = f"设备故障锁定（{fault}），请检查后重新连接"
                stop_for_fault = True
            for channel, values, fields, speed, current in updates:
                motor = self._motors.setdefault(channel, {"last_seen": 0.0})
                if any(name in fields for name in ("seen", "rx", "age_ms")):
                    self._extended_feedback_channels.add(channel)
                extended = channel in self._extended_feedback_channels
                if extended:
                    seen = self._number(fields.get("seen"))
                    rx = self._number(fields.get("rx"))
                    age_ms = self._number(fields.get("age_ms"))
                    valid = (seen == 1 and rx is not None and rx.is_integer() and 0 < rx <= 0xFFFFFFFF
                             and age_ms is not None and age_ms >= 0)
                    values.update(seen=seen, rx=rx, age_ms=age_ms)
                    if not valid:
                        values.update(feedback_valid=False, last_seen=None)
                    elif speed is not None and current is not None:
                        count = int(rx)
                        previous = self._feedback_rx.get(channel)
                        progressed = previous is None or 0 < ((count - previous) & 0xFFFFFFFF) < 0x80000000
                        sample_time = now - age_ms / 1000.0
                        if progressed:
                            self._feedback_rx[channel] = count
                            values.update(speed=speed, current_raw=current, last_seen=sample_time,
                                          feedback_valid=True)
                        else:
                            # Repeated reports of one CAN sample cannot renew
                            # freshness, even if age_ms is incorrectly zero.
                            old_time = motor.get("last_seen")
                            if isinstance(old_time, (int, float)) and math.isfinite(old_time):
                                values["last_seen"] = min(old_time, sample_time)
                    # An extended partial frame cannot borrow old measured
                    # values and label that pair as a fresh sample.
                elif speed is not None and current is not None:
                    # Legacy TCP telemetry has no device sample metadata.
                    values.update(speed=speed, current_raw=current, last_seen=now, feedback_valid=True)
                motor.update(values)
        if stop_for_fault:
            self._stop(expected_socket=sock)

    @staticmethod
    def _line_fault(line: str) -> str | None:
        header = line.split("|", 1)[0].strip()
        if header.upper().startswith("CTRL ERR"):
            return header[len("CTRL ERR"):].strip() or "COMMAND_REJECTED"
        fields = {name.lower(): value for name, value in _FIELD.findall(header)}
        value = fields.get("fault")
        # v2 uses the stop reason directly in its telemetry fault field,
        # even though STOP does not latch the firmware fault flag. Numeric
        # fault=1 remains a fault regardless of the accompanying reason.
        if value is not None and value.upper() not in ("0", "NONE", "OK", "FALSE", "STOP"):
            return fields.get("reason") or value
        return None

    @staticmethod
    def _number(value) -> float | None:
        if value is None:
            return None
        try:
            result = float(value)
            return result if math.isfinite(result) else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _first_number(cls, fields, names) -> float | None:
        for name in names:
            if name in fields:
                return cls._number(fields[name])
        return None
