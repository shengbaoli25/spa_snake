"""Explicitly controlled SPA-CAM serial adapter for a local web server.

No serial port is opened at import or enumeration time. Only ``connect`` starts
the existing receiver worker; the receiver has no robot motor functionality.
"""
from __future__ import annotations

import threading
import time
import re

from .bluetooth_camera import CameraReceiver


class CameraError(RuntimeError):
    """An actionable camera setup or connection error."""


_KNOWN_SPA_CAM_ADDRESS = "20500D8D471A"
_BTH_ADDRESS = re.compile(r"&0&([0-9A-F]{12})_", re.IGNORECASE)


def _bluetooth_address(hwid):
    """Extract the remote device address from an outgoing Windows SPP ID."""
    if not hwid.upper().startswith("BTHENUM\\"):
        return None
    match = _BTH_ADDRESS.search(hwid)
    if not match or match.group(1) == "000000000000":
        return None
    return match.group(1).upper()


def _bluetooth_name(address):
    """Read Windows' cached device name; unavailable platforms fail closed."""
    try:
        import winreg
        path = r"SYSTEM\CurrentControlSet\Services\BTHPORT\Parameters\Devices" + "\\" + address
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            for field in ("FriendlyName", "Name"):
                try:
                    value, _ = winreg.QueryValueEx(key, field)
                except OSError:
                    continue
                if isinstance(value, bytes):
                    # Device Name is typically null-terminated UTF-8; some
                    # FriendlyName values are stored as UTF-16LE.
                    if len(value) > 2 and value[1] == 0:
                        value = value.decode("utf-16-le", "replace").rstrip("\x00")
                    else:
                        value = value.split(b"\x00", 1)[0].decode("utf-8", "replace")
                if isinstance(value, str) and value.strip("\x00 "):
                    return value.strip("\x00 ")
    except (ImportError, OSError):
        pass
    return None


def list_camera_ports():
    """Enumerate serial ports without opening them, marking the C board unsafe."""
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise CameraError("缺少 pyserial，请安装 spa_control 的依赖") from exc
    try:
        entries = list_ports.comports()
    except Exception as exc:
        raise CameraError(f"无法枚举串口：{exc}") from exc
    ports = []
    for item in entries:
        hwid = str(getattr(item, "hwid", "") or "")
        vid = getattr(item, "vid", None)
        pid = getattr(item, "pid", None)
        motor_board = ((vid, pid) == (0x0483, 0x5740)
                       or "VID:PID=0483:5740" in hwid.upper()
                       or ("VID_0483" in hwid.upper() and "PID_5740" in hwid.upper()))
        description = str(getattr(item, "description", "") or "")
        address = _bluetooth_address(hwid)
        name = _bluetooth_name(address) if address else None
        preferred = not motor_board and (address == _KNOWN_SPA_CAM_ADDRESS
                                         or (name is not None and name.casefold() == "spa-cam"))
        friendly_name = name or ("SPA-CAM" if address == _KNOWN_SPA_CAM_ADDRESS else description)
        ports.append({"device": str(item.device), "description": description,
                      "friendly_name": friendly_name, "preferred": preferred,
                      "bluetooth_address": address, "hwid": hwid,
                      "available_for_camera": not motor_board})
    return ports


class WebCamera:
    """Thread-safe façade over the existing bounded JPEG receiver.

    A successful ``connect`` only starts the worker; check ``status`` for an
    eventual first frame or error. Frames have unsynchronized device and host
    clocks and should not be treated as robot-state timestamps.
    """

    def __init__(self, receiver=None, port_lister=None):
        self._receiver = receiver if receiver is not None else CameraReceiver()
        self._port_lister = port_lister if port_lister is not None else list_camera_ports
        self._operation_lock = threading.Lock()
        self._port = None

    def ports(self):
        return self._port_lister()

    def connect(self, port):
        if not isinstance(port, str) or not port.strip():
            raise CameraError("请选择相机串口")
        port = port.strip()
        with self._operation_lock:
            if self._receiver.snapshot()[0]:
                raise CameraError("相机正在连接或运行，请先断开")
            selected = next((p for p in self.ports()
                             if p["device"].casefold() == port.casefold()), None)
            if selected is None:
                raise CameraError("串口不存在或已断开，请刷新列表")
            if not selected["available_for_camera"]:
                raise CameraError("该串口属于 C 板，不能用于相机")
            try:
                self._receiver.connect(selected["device"])
            except Exception as exc:
                raise CameraError(f"无法启动相机：{exc}") from exc
            self._port = selected["device"]
        return self.status()

    def disconnect(self):
        with self._operation_lock:
            self._receiver.disconnect()
        return self.status()

    def set_light(self, enabled):
        if not isinstance(enabled, bool):
            raise CameraError("补光状态必须为 true 或 false")
        with self._receiver.lock:
            self._receiver.light = enabled
        return self.status()

    def status(self):
        active, message, latest, count = self._receiver.snapshot()
        with self._receiver.lock:
            light = self._receiver.light
        age = None if latest is None else max(0.0, (time.monotonic_ns() - latest[1]) / 1e9)
        return {"active": active, "message": message, "port": self._port,
                "light": light, "frames_received": count,
                "frame_age_s": age, "frame_stale": age is None or age > 2.0,
                "clock_sync": "not_calibrated"}

    def latest_frame(self):
        """Return bytes and metadata, or None until a frame is available."""
        _, _, latest, _ = self._receiver.snapshot()
        if latest is None:
            return None
        packet, received_ns, _ = latest
        return {"jpeg": packet.jpeg, "boot_id": packet.boot_id,
                "sequence": packet.sequence, "capture_us": packet.capture_us,
                "receive_monotonic_ns": received_ns,
                "age_s": max(0.0, (time.monotonic_ns() - received_ns) / 1e9),
                "clock_sync": "not_calibrated"}
