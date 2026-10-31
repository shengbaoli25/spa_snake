"""Bounded, passive recording of the two connected motor IDs and SPA-CAM.

This module does not connect devices or send any robot command. Host receipt
timestamps are not synchronized sensor exposure or motor acquisition times.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parent / "runs" / "two_motor"
PERIOD_S = 0.1
CAMERA_MAX_AGE_NS = 2_000_000_000


def _safe(value):
    """Guarantee ordinary JSON; missing/non-finite sensor values stay null."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return None


def _write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_safe(value), ensure_ascii=False, allow_nan=False,
                               indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


class TwoMotorRecorder:
    def __init__(self, camera, motor, global_camera=None):
        self.camera, self.motor = camera, motor
        self.global_camera = global_camera
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._closed = False
        self._status = {"state": "idle", "active": False, "samples": 0,
                        "images": 0, "directory": None, "error": None}

    def status(self):
        with self._lock:
            return dict(self._status)

    def start(self, duration_s):
        if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
                or not math.isfinite(duration_s) or not 2 <= duration_s <= 30):
            raise ValueError("采集时长须为 2 到 30 秒")
        with self._lock:
            if self._closed:
                raise RuntimeError("采集器已关闭")
            if self._status["active"]:
                raise ValueError("采集正在进行")
            folder = ROOT / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:12])
            folder.mkdir(parents=True, exist_ok=False)
            (folder / "images").mkdir()
            self._stop = threading.Event()
            self._status = {"state": "recording", "active": True, "samples": 0,
                            "images": 0, "directory": str(folder), "error": None,
                            "duration_s": float(duration_s)}
            self._thread = threading.Thread(target=self._run, args=(folder, float(duration_s), self._stop),
                                            daemon=True, name="two-motor-recorder")
            self._thread.start()
            return dict(self._status)

    def stop(self):
        with self._lock:
            if self._status["active"]:
                self._stop.set()
                self._status["state"] = "stopping"
            return dict(self._status)

    def close(self):
        with self._lock:
            self._closed = True
            self._stop.set()
            thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=3)

    def _run(self, folder, duration_s, stop):
        start_ns = time.monotonic_ns()
        deadline_ns = start_ns + int(duration_s * 1e9)
        manifest = {"schema": "two_motor_passive_record_v1",
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "start_monotonic_ns": start_ns, "duration_s": duration_s,
                    "sampling_period_s": PERIOD_S, "clock_sync": "not_calibrated",
                    "motor_clock": "host receive time; age-derived time is an estimate",
                    "camera_clock": "host receive time; capture_us is device clock",
                    "channels": ["M3", "M4"], "front_rear_mapping": "unknown",
                    "profile": "passive engineering recording; not training or motor control",
                    "motor_commands_sent": False, "ground_speed": "global_camera_if_valid" if self.global_camera is not None else "unavailable",
                    "global_camera_clock": "host monotonic frame read; exposure clock not synchronized",
                    "reward": "unavailable", "state": "recording", "samples": 0, "images": 0}
        last_frame_key, last_image = None, None
        next_ns = start_ns
        try:
            _write_json(folder / "manifest.json", manifest)
            with (folder / "events.jsonl").open("w", encoding="utf-8", buffering=1) as events:
                while not stop.is_set():
                    now = time.monotonic_ns()
                    if now >= deadline_ns:
                        break
                    if now < next_ns:
                        stop.wait(min((next_ns - now) / 1e9, (deadline_ns - now) / 1e9))
                        continue
                    # Snapshot APIs only copy in-memory state; serial I/O belongs to workers.
                    try:
                        motor = self.motor.status()
                        motor_error = None
                    except Exception as exc:
                        motor, motor_error = {}, str(exc)
                    try:
                        camera_status, frame = self.camera.status(), self.camera.latest_frame()
                        camera_error = None
                    except Exception as exc:
                        camera_status, frame, camera_error = {}, None, str(exc)
                    try:
                        global_status = self.global_camera.status() if self.global_camera is not None else None
                        global_error = None
                    except Exception as exc:
                        global_status, global_error = None, str(exc)
                    sample_ns = time.monotonic_ns()
                    motors = {}
                    channels = motor.get("motors", {}) if isinstance(motor, dict) else {}
                    if not isinstance(channels, dict):
                        channels = {}
                    for channel in ("3", "4"):
                        values = channels.get(channel)
                        valid = bool(isinstance(values, dict) and values.get("fresh") and values.get("seen"))
                        motors["M" + channel] = {"available": valid,
                            "reason": None if valid else "missing_or_stale",
                            "feedback": _safe(values) if isinstance(values, dict) else None}
                    camera = {"available": False, "reason": "missing_or_stale", "image": None}
                    if isinstance(frame, dict) and camera_status.get("active"):
                        received = frame.get("receive_monotonic_ns")
                        jpeg = frame.get("jpeg")
                        if (type(received) is int and 0 <= sample_ns - received <= CAMERA_MAX_AGE_NS
                                and isinstance(jpeg, bytes) and jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
                            key = (frame.get("boot_id"), frame.get("sequence"), received)
                            if key != last_frame_key:
                                last_image = f"images/{manifest['images']:06d}.jpg"
                                (folder / last_image).write_bytes(jpeg)
                                manifest["images"] += 1
                                last_frame_key = key
                            camera = {"available": True, "reason": None, "image": last_image,
                                      "boot_id": _safe(frame.get("boot_id")),
                                      "sequence": _safe(frame.get("sequence")),
                                      "capture_us": _safe(frame.get("capture_us")),
                                      "receive_monotonic_ns": received, "clock_sync": "not_calibrated"}
                    global_measurement = {"valid": False, "reason": "missing_or_stale",
                                          "position_m": None, "velocity_mps": None,
                                          "sample_timestamp_monotonic_s": None}
                    if isinstance(global_status, dict):
                        stamp = global_status.get("sample_timestamp_monotonic_s")
                        position, velocity = global_status.get("position_m"), global_status.get("velocity_mps")
                        numeric = lambda x: type(x) in (float, int) and math.isfinite(x)
                        if (global_status.get("valid") is True and all(map(numeric, (stamp, position, velocity)))
                                and 0 <= sample_ns / 1e9 - stamp <= .8):
                            global_measurement.update(valid=True, reason=None, position_m=position,
                                                      velocity_mps=velocity, sample_timestamp_monotonic_s=stamp)
                        global_measurement.update(state=_safe(global_status.get("state")),
                                                  revision=_safe(global_status.get("revision")),
                                                  sample_revision=_safe(global_status.get("sample_revision")),
                                                  direction=_safe(global_status.get("direction")),
                                                  calibration_points=_safe(global_status.get("points")),
                                                  sample_age_s=_safe(global_status.get("sample_age_s")),
                                                  note=_safe(global_status.get("note")))
                    record = {"type": "sample", "host_monotonic_ns": sample_ns,
                              "motor_snapshot_monotonic_s": _safe(motor.get("snapshot_monotonic_s")),
                              "motor_frame_receive_monotonic_s": _safe(motor.get("frame_receive_monotonic_s")),
                              "motor_state": _safe({k: motor.get(k) for k in ("state", "active", "frame_fresh", "fault", "can_ready", "can_healthy", "directions")}),
                              "motors": motors, "camera": camera, "global_camera": global_measurement,
                              "global_camera_error": global_error,
                              "motor_error": motor_error, "camera_error": camera_error}
                    events.write(json.dumps(_safe(record), ensure_ascii=False, allow_nan=False) + "\n")
                    manifest["samples"] += 1
                    with self._lock:
                        self._status.update(samples=manifest["samples"], images=manifest["images"])
                    # Skip overdue cycles rather than accumulating a burst.
                    next_ns = max(next_ns + int(PERIOD_S * 1e9), time.monotonic_ns())
            manifest["state"] = "stopped" if stop.is_set() else "completed"
        except Exception as exc:
            manifest["state"], manifest["error"] = "error", str(exc)
        finally:
            manifest["end_monotonic_ns"] = time.monotonic_ns()
            try:
                _write_json(folder / "manifest.json", manifest)
            except Exception as exc:
                manifest["state"], manifest["error"] = "error", f"写入清单失败：{exc}"
            with self._lock:
                self._status.update(state=manifest["state"], active=False,
                                    samples=manifest["samples"], images=manifest["images"],
                                    error=manifest.get("error"))
