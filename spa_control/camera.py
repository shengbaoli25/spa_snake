"""Camera measurements for SPA; importing this module never opens a camera.

The RGB geometry, leading-edge reference point and causal speed filter follow
Desktop/snakerobot_rl/pc_tools/global_camera_speed_{core,gui}.py.  The CSV
receiver consumes that GUI's actual, flushed recording format unchanged.

The optional UDP adapter is a NEW localhost interface, not an interface the old
GUI already implements. Each JSON datagram requires timestamp_monotonic_s,
position_m, velocity_mps and tracking_ok; timestamps must use THIS PC's
time.monotonic clock. Do not replay recordings as live training measurements.
"""
from __future__ import annotations

import csv
import io
import json
import math
import socket
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TrackCalibrationPoint:
    image_x_px: float
    image_y_px: float
    position_m: float
    depth_m: float | None = None
    camera_xyz_m: list[float] | None = None


@dataclass(frozen=True)
class TrackProjection:
    position_m: float
    distance_from_track_px: float
    segment_index: int
    segment_fraction: float


class StraightTrackCalibration:
    """Same piecewise image-to-track projection used by the existing GUI."""

    def __init__(self, points: list[TrackCalibrationPoint], metadata: dict | None = None):
        self.points = sorted(points, key=lambda p: p.position_m)
        self.metadata = dict(metadata or {})
        positions: set[float] = set()
        for point in self.points:
            if not all(math.isfinite(float(v)) for v in
                       (point.image_x_px, point.image_y_px, point.position_m)):
                raise ValueError("标定坐标必须是有限数值")
            if point.position_m in positions:
                raise ValueError("标定实际位置不能重复")
            positions.add(point.position_m)
        for a, b in zip(self.points, self.points[1:]):
            if math.hypot(b.image_x_px-a.image_x_px, b.image_y_px-a.image_y_px) < 2:
                raise ValueError("相邻标定像素点过近")

    @property
    def ready(self) -> bool:
        return len(self.points) >= 2

    @property
    def expected_frame_size(self) -> tuple[int, int] | None:
        width, height = self.metadata.get("frame_width"), self.metadata.get("frame_height")
        return (int(width), int(height)) if width and height else None

    @classmethod
    def load(cls, path: str | Path) -> "StraightTrackCalibration":
        with Path(path).open(encoding="utf-8-sig") as file:
            data = json.load(file)
        if data.get("schema_version") != 1 or data.get("track_type", "straight") != "straight":
            raise ValueError("需要 schema_version=1 的 straight 轨道标定")
        result = cls([TrackCalibrationPoint(**p) for p in data["points"]], data.get("metadata"))
        if not result.ready:
            raise ValueError("至少需要两个轨道标定点")
        return result

    def project(self, image_x_px: float, image_y_px: float) -> TrackProjection:
        if not self.ready:
            raise ValueError("至少需要两个轨道标定点")
        query = np.array([image_x_px, image_y_px], dtype=float)
        if not np.all(np.isfinite(query)):
            raise ValueError("图像坐标必须是有限数值")
        best: TrackProjection | None = None
        for i, (a, b) in enumerate(zip(self.points, self.points[1:])):
            start = np.array([a.image_x_px, a.image_y_px])
            delta = np.array([b.image_x_px-a.image_x_px, b.image_y_px-a.image_y_px])
            fraction = float((query-start) @ delta / (delta @ delta))
            clipped = min(1., max(0., fraction))
            distance = float(np.linalg.norm(query-start-clipped*delta))
            candidate = TrackProjection(a.position_m+clipped*(b.position_m-a.position_m),
                                        distance, i, clipped)
            if best is None or distance < best.distance_from_track_px:
                best = candidate
        assert best is not None
        # Preserve the existing application's endpoint extrapolation convention.
        if ((best.segment_index == 0 and best.segment_fraction == 0.) or
                (best.segment_index == len(self.points)-2 and best.segment_fraction == 1.)):
            a, b = self.points[best.segment_index:best.segment_index+2]
            start = np.array([a.image_x_px, a.image_y_px])
            delta = np.array([b.image_x_px-a.image_x_px, b.image_y_px-a.image_y_px])
            fraction = float((query-start) @ delta / (delta @ delta))
            return TrackProjection(a.position_m+fraction*(b.position_m-a.position_m),
                                   float(np.linalg.norm(query-start-fraction*delta)),
                                   best.segment_index, fraction)
        return best


@dataclass(frozen=True)
class SpeedEstimate:
    raw_position_m: float
    filtered_position_m: float
    velocity_mps: float
    sample_count: int
    span_s: float


class CausalSpeedEstimator:
    """Position low-pass followed by trailing-window least-squares slope."""

    def __init__(self, window_s: float = .30, position_time_constant_s: float = .06):
        if not math.isfinite(window_s) or window_s <= 0:
            raise ValueError("速度滤波窗口必须为有限正数")
        if not math.isfinite(position_time_constant_s) or position_time_constant_s < 0:
            raise ValueError("位置滤波时间常数必须为有限非负数")
        self.window_s = float(window_s)
        self.position_time_constant_s = float(position_time_constant_s)
        self.reset()

    def reset(self) -> None:
        self._samples: deque[tuple[float, float]] = deque()
        self._last_time: float | None = None
        self._filtered: float | None = None

    def add(self, timestamp_s: float, position_m: float) -> SpeedEstimate:
        if not math.isfinite(timestamp_s) or not math.isfinite(position_m):
            raise ValueError("测速时间和位置必须是有限数值")
        if self._last_time is not None and timestamp_s <= self._last_time:
            raise ValueError("测速时间戳必须严格递增")
        if self._filtered is None:
            filtered = float(position_m)
        else:
            dt = timestamp_s-float(self._last_time)
            alpha = 1. if self.position_time_constant_s == 0 else 1.-math.exp(-dt/self.position_time_constant_s)
            filtered = self._filtered+alpha*(position_m-self._filtered)
        self._last_time, self._filtered = timestamp_s, filtered
        self._samples.append((timestamp_s, filtered))
        while len(self._samples) > 2 and self._samples[1][0] < timestamp_s-self.window_s:
            self._samples.popleft()
        times = np.array([p[0]-timestamp_s for p in self._samples])
        positions = np.array([p[1] for p in self._samples])
        centered = times-times.mean()
        denominator = float(centered @ centered)
        velocity = float(centered @ (positions-positions.mean())/denominator) if denominator > 1e-12 else 0.
        return SpeedEstimate(position_m, filtered, velocity, len(self._samples),
                             timestamp_s-self._samples[0][0])


def bbox_reference_point(bbox: tuple[float, float, float, float], direction: str) -> tuple[float, float]:
    x, y, width, height = bbox
    if direction not in ("left", "right"):
        raise ValueError("图像前进方向必须为 left 或 right")
    return (x if direction == "left" else x+width), y+height*.5


@dataclass(frozen=True)
class CameraSample:
    timestamp_s: float
    received_monotonic_s: float
    position_m: float | None
    velocity_mps: float | None
    tracking_ok: bool
    source: str
    note: str = ""
    bbox: tuple[float, float, float, float] | None = None

    def is_fresh(self, max_age_s: float = .5, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        # Both clocks are this PC's monotonic clock. Checking acquisition time
        # prevents buffered CSV or UDP packets from being treated as live data.
        return (0 <= now-self.timestamp_s <= max_age_s and
                0 <= now-self.received_monotonic_s <= max_age_s)


class _SampleSource:
    def __init__(self):
        self._lock = threading.Lock()
        self._sample: CameraSample | None = None
        self._status = "未启动"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    def latest(self, max_age_s: float = .5) -> CameraSample | None:
        if self._stop.is_set():
            return None
        with self._lock:
            sample = self._sample
        return sample if sample is not None and sample.is_fresh(max_age_s) else None

    def _publish(self, sample: CameraSample, status: str | None = None) -> None:
        with self._lock:
            self._sample = sample
            if status is not None:
                self._status = status

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._sample = None
            self._status = "已停止"
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.5)
        with self._lock:
            self._sample = None
            self._status = "已停止"


def parse_speed_payload(payload: dict[str, Any], source: str = "udp",
                        received_monotonic_s: float | None = None) -> CameraSample:
    """Validate one localhost UDP packet or existing-camera CSV row."""
    if "schema_version" in payload and int(payload["schema_version"]) != 1:
        raise ValueError("测速消息 schema_version 必须为 1")
    timestamp = float(payload["timestamp_monotonic_s"])
    received = time.monotonic() if received_monotonic_s is None else float(received_monotonic_s)
    if not math.isfinite(timestamp) or not math.isfinite(received):
        raise ValueError("测速时间戳必须是有限数值")
    tracking = payload["tracking_ok"]
    if tracking not in (True, False, 0, 1, "0", "1", "true", "false"):
        raise ValueError("tracking_ok 必须为布尔值或 0/1")
    ok = tracking in (True, 1, "1", "true")
    if not ok:
        return CameraSample(timestamp, received, None, None, False, source, "相机跟踪丢失或未就绪")
    # Existing GUI writes filtered_position_m; new UDP uses position_m.
    position = float(payload.get("position_m", payload.get("filtered_position_m")))
    velocity = float(payload["velocity_mps"])
    if not math.isfinite(position) or not math.isfinite(velocity):
        raise ValueError("有效测速消息必须包含有限的位置和速度")
    return CameraSample(timestamp, received, position, velocity, True, source)


class UdpSpeedReceiver(_SampleSource):
    """Optional NEW localhost JSON protocol; no robot network connection."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9870):
        super().__init__()
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("UDP 测速只允许本机，远程时间戳需另行做时钟同步")
        self.host, self.port = host, int(port)
        self._socket: socket.socket | None = None
        self._last_timestamp: float | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        sock = socket.socket(socket.AF_INET6 if self.host == "::1" else socket.AF_INET,
                             socket.SOCK_DGRAM)
        try:
            sock.bind((self.host, self.port))
            sock.settimeout(.2)
        except Exception:
            sock.close()
            raise
        self._socket = sock
        self.port = int(sock.getsockname()[1])
        self._stop.clear()
        self._last_timestamp = None
        self._set_status(f"等待本机 UDP 测速 {self.host}:{self.port}")
        self._thread = threading.Thread(target=self._run, daemon=True, name="spa-speed-udp")
        self._thread.start()

    def _run(self) -> None:
        assert self._socket is not None
        try:
            while not self._stop.is_set():
                try:
                    raw, _ = self._socket.recvfrom(8192)
                except socket.timeout:
                    continue
                try:
                    payload = json.loads(raw.decode("utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("测速消息必须是 JSON 对象")
                    sample = parse_speed_payload(payload)
                    if not sample.is_fresh():
                        raise ValueError("UDP 测速已过时或时间戳不属于本机 monotonic 时钟")
                    if self._last_timestamp is not None and sample.timestamp_s <= self._last_timestamp:
                        raise ValueError("忽略乱序或重复测速包")
                    self._last_timestamp = sample.timestamp_s
                    self._publish(sample, "UDP 测速正常" if sample.tracking_ok else sample.note)
                except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                    self._set_status(f"拒绝测速包：{exc}")
        except OSError as exc:
            if not self._stop.is_set():
                self._set_status(f"UDP 测速错误：{exc}")
        finally:
            self._socket.close()


class CsvSpeedReceiver(_SampleSource):
    """Read-only tail of the original global_camera_speed_gui recording CSV.

    Requires the old GUI and this app on the same PC. Old records retain their
    acquisition timestamps and fail latest() freshness checks; they are never
    silently treated as current camera measurements.
    """

    def __init__(self, path: str | Path):
        super().__init__()
        self.path = Path(path)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.path.is_file():
            raise FileNotFoundError(f"测速 CSV 不存在：{self.path}")
        self._stop.clear()
        self._set_status(f"读取测速 CSV：{self.path.name}")
        self._thread = threading.Thread(target=self._run, daemon=True, name="spa-speed-csv")
        self._thread.start()

    def _run(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8-sig", newline="") as file:
                header = file.readline()
                if not header.endswith("\n"):
                    raise ValueError("CSV 表头尚未写完整，请先在测速程序开始记录")
                fields = next(csv.reader([header]))
                required = {"timestamp_monotonic_s", "filtered_position_m", "velocity_mps", "tracking_ok"}
                if not required.issubset(fields):
                    raise ValueError("不是原全局相机测速程序的 CSV 格式")
                pending = ""
                last_timestamp: float | None = None
                while not self._stop.is_set():
                    chunk = file.read(65536)
                    if not chunk:
                        if self.path.stat().st_size < file.tell():
                            raise ValueError("CSV 文件被截断，请重新连接新的记录文件")
                        self._stop.wait(.03)
                        continue
                    pending += chunk
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        if not line.strip():
                            continue
                        try:
                            values = next(csv.reader(io.StringIO(line)))
                            if len(values) != len(fields):
                                raise ValueError("CSV 行列数不完整")
                            sample = parse_speed_payload(dict(zip(fields, values)), "camera_csv")
                            if last_timestamp is not None and sample.timestamp_s <= last_timestamp:
                                raise ValueError("CSV 时间戳不递增")
                            last_timestamp = sample.timestamp_s
                            if sample.timestamp_s > time.monotonic()+.01:
                                raise ValueError("CSV 不是本机当前启动周期的 monotonic 时间")
                            self._publish(sample, "CSV 测速正常" if sample.tracking_ok else sample.note)
                        except (ValueError, TypeError, KeyError) as exc:
                            self._set_status(f"拒绝测速记录：{exc}")
        except (OSError, ValueError) as exc:
            self._set_status(f"CSV 测速错误：{exc}")


class RgbPositionProcessor:
    """Hardware-free measured-pixel input, also used by the CSRT camera source."""

    def __init__(self, calibration: StraightTrackCalibration, direction: str = "left",
                 window_s: float = .30, position_time_constant_s: float = .06,
                 warmup_s: float = .20):
        bbox_reference_point((0, 0, 1, 1), direction)
        if not math.isfinite(warmup_s) or warmup_s < 0 or warmup_s > window_s:
            raise ValueError("测速预热时长应处于 0 到滤波窗口之间")
        self.calibration, self.direction = calibration, direction
        self.estimator = CausalSpeedEstimator(window_s, position_time_constant_s)
        self.warmup_s = warmup_s
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        self.estimator.reset()
        self._last_timestamp = None

    def add_bbox(self, timestamp_s: float, bbox: tuple[float, float, float, float],
                 frame_size: tuple[int, int] | None = None) -> CameraSample:
        if (frame_size is not None and self.calibration.expected_frame_size is not None
                and tuple(frame_size) != self.calibration.expected_frame_size):
            raise ValueError(f"相机分辨率 {frame_size} 与标定 {self.calibration.expected_frame_size} 不一致")
        if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox) or bbox[2] <= 0 or bbox[3] <= 0:
            raise ValueError("无效的头部跟踪框")
        if self._last_timestamp is not None and timestamp_s-self._last_timestamp > .5:
            self.reset()
        projection = self.calibration.project(*bbox_reference_point(bbox, self.direction))
        estimate = self.estimator.add(timestamp_s, projection.position_m)
        self._last_timestamp = timestamp_s
        valid = estimate.sample_count >= 3 and estimate.span_s >= self.warmup_s
        return CameraSample(timestamp_s, time.monotonic(), estimate.filtered_position_m,
                            estimate.velocity_mps if valid else None, valid, "rgb_camera",
                            f"距轨道 {projection.distance_from_track_px:.1f}px" if valid else "测速预热中",
                            bbox)


class RgbCameraSpeedSource(_SampleSource):
    """Direct RGB camera, user-selected CSRT head ROI, original RGB calibration.

    start() is the only operation that opens hardware, in a background thread.
    latest_frame() returns a BGR copy for a Tk/PIL preview. set_roi() takes image
    pixel coordinates, not scaled canvas coordinates. close the old camera GUI
    before using this direct source to avoid competing for the same camera.
    """

    def __init__(self, calibration_path: str | Path, camera_index: int = 0,
                 width: int = 1280, height: int = 720, fps: int = 30,
                 direction: str = "left"):
        super().__init__()
        self.calibration = StraightTrackCalibration.load(calibration_path)
        self.processor = RgbPositionProcessor(self.calibration, direction)
        self.camera_index = int(camera_index)
        self.width, self.height, self.fps = int(width), int(height), int(fps)
        self._frame: np.ndarray | None = None
        self._requested_roi: tuple[float, float, float, float] | None = None
        self._clear_roi = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("直接相机模式需要 opencv-contrib-python；或使用原测速程序 CSV 接入") from exc
        self._make_tracker(cv2)  # Fail before opening hardware if CSRT is absent.
        self.processor.reset()
        self._stop.clear()
        self._set_status("正在连接 RGB 相机")
        self._thread = threading.Thread(target=self._run, args=(cv2,), daemon=True, name="spa-rgb-camera")
        self._thread.start()

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def set_roi(self, bbox: tuple[float, float, float, float]) -> None:
        if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in bbox) or bbox[2] < 4 or bbox[3] < 4:
            raise ValueError("请框选至少 4×4 像素的头部 ROI")
        with self._lock:
            if self._frame is None:
                raise RuntimeError("请先等待相机画面")
            self._requested_roi = tuple(float(v) for v in bbox)
            self._sample = None
            self._status = "头部 ROI 已选，正在初始化跟踪"

    def clear_roi(self) -> None:
        with self._lock:
            self._requested_roi = None
            self._clear_roi = True
            self._sample = None

    @staticmethod
    def _make_tracker(cv2: Any) -> Any:
        if hasattr(cv2, "TrackerCSRT_create"):
            return cv2.TrackerCSRT_create()
        if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        raise RuntimeError("CSRT 跟踪器不可用，请安装 opencv-contrib-python")

    def _run(self, cv2: Any) -> None:
        capture = None
        tracker = None
        try:
            import os
            backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
            capture = cv2.VideoCapture(self.camera_index, backend)
            if not capture.isOpened():
                raise RuntimeError(f"无法打开 RGB 相机 {self.camera_index}")
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            capture.set(cv2.CAP_PROP_FPS, self.fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._set_status("RGB 相机已连接，请在画面框选机器人头部")
            while not self._stop.is_set():
                ok, frame = capture.read()
                timestamp = time.monotonic()
                if not ok:
                    raise RuntimeError("RGB 相机读取失败")
                frame_size = (int(frame.shape[1]), int(frame.shape[0]))
                expected = self.calibration.expected_frame_size
                if expected is not None and frame_size != expected:
                    raise RuntimeError(f"相机实际分辨率 {frame_size} 与标定 {expected} 不一致")
                with self._lock:
                    self._frame = frame
                    request, clear = self._requested_roi, self._clear_roi
                    self._requested_roi, self._clear_roi = None, False
                if clear:
                    tracker = None
                    self.processor.reset()
                    self._set_status("跟踪已停止，请重新框选头部")
                if request is not None:
                    x, y, w, h = request
                    fw, fh = frame_size
                    x1, y1 = max(0, min(fw-1, round(x))), max(0, min(fh-1, round(y)))
                    x2, y2 = max(x1+1, min(fw, round(x+w))), max(y1+1, min(fh, round(y+h)))
                    tracker = self._make_tracker(cv2)
                    if tracker.init(frame, (x1, y1, x2-x1, y2-y1)) is False:
                        raise RuntimeError("CSRT 无法初始化该 ROI，请重新框选")
                    self.processor.reset()
                    self._set_status("头部跟踪已开始，测速预热中")
                    continue
                if tracker is not None:
                    tracked, bbox = tracker.update(frame)
                    if not tracked:
                        tracker = None
                        self.processor.reset()
                        self._publish(CameraSample(timestamp, timestamp, None, None, False,
                                                   "rgb_camera", "头部跟踪丢失，请重新框选"),
                                      "头部跟踪丢失，请重新框选")
                    else:
                        sample = self.processor.add_bbox(timestamp, tuple(float(v) for v in bbox), frame_size)
                        self._publish(sample, "RGB 测速正常；"+sample.note)
        except Exception as exc:
            now = time.monotonic()
            self._publish(CameraSample(now, now, None, None, False, "rgb_camera", str(exc)),
                          f"相机错误：{exc}")
        finally:
            if capture is not None:
                capture.release()
            if self._stop.is_set():
                with self._lock:
                    self._sample = None
                    self._frame = None
                    self._status = "已停止"
