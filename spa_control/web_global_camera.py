"""Web adapter for a separately connected global RGB tracking camera.

Calibration and ROI use source-image pixels, not displayed CSS coordinates. No
camera opens until connect() is called. Samples are invalidated on lost frames,
tracking failures, calibration changes, and disconnect; they are never zero-filled.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import threading
import time

from .camera import (CameraSample, RgbCameraSpeedSource, RgbPositionProcessor,
                     StraightTrackCalibration, TrackCalibrationPoint)


class GlobalCameraError(ValueError):
    pass


class WebGlobalCamera:
    def __init__(self):
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._frame = None
        self._frame_time = None
        self._sample = None
        self._sample_revision = None
        self._calibration = None
        self._direction = 'left'
        self._pending_roi = None
        self._revision = 0
        self._message = '未连接全局相机'
        self._state = 'disconnected'
        self._index = None

    def connect(self, camera_index=0, width=1280, height=720, fps=30):
        try:
            index, width, height, fps = int(camera_index), int(width), int(height), int(fps)
        except (ValueError, TypeError, OverflowError) as exc:
            raise GlobalCameraError('相机编号、宽高和帧率必须为整数') from exc
        if not (0 <= index <= 20 and 160 <= width <= 3840 and 120 <= height <= 2160 and 1 <= fps <= 120):
            raise GlobalCameraError('相机参数超出范围')
        try:
            import cv2
            RgbCameraSpeedSource._make_tracker(cv2)
        except Exception as exc:
            raise GlobalCameraError(f'需要安装支持 CSRT 的 OpenCV：{exc}') from exc
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise GlobalCameraError('相机正在连接、运行或关闭，请稍后重试')
            self._stop.clear()
            self._frame = self._sample = self._calibration = self._frame_time = None
            self._sample_revision = None
            self._pending_roi = None
            self._revision += 1
            self._index = index
            self._state, self._message = 'connecting', '正在连接全局相机'
            self._thread = threading.Thread(target=self._run, args=(cv2, index, width, height, fps),
                                            name='web-global-camera', daemon=True)
            self._thread.start()
        return self.status()

    def disconnect(self):
        with self._lock:
            self._stop.set()
            self._revision += 1
            self._sample = self._frame = self._frame_time = None
            self._sample_revision = None
            thread = self._thread
            self._state, self._message = 'disconnecting', '正在关闭全局相机'
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.5)
        with self._lock:
            if thread is None or not thread.is_alive():
                self._state, self._message = 'disconnected', '全局相机已断开'
        return self.status()

    close = disconnect

    def _require_frame(self):
        if self._state != 'connected' or self._frame is None or self._frame_time is None or time.monotonic()-self._frame_time > 1:
            raise GlobalCameraError('请先连接全局相机并等待新鲜画面')
        return self._frame.shape[1], self._frame.shape[0]

    def set_calibration(self, points, direction='left'):
        if direction not in ('left', 'right'):
            raise GlobalCameraError('头部前沿方向必须为 left 或 right')
        if not isinstance(points, list) or not (2 <= len(points) <= 30):
            raise GlobalCameraError('需要 2 至 30 个轨道标定点')
        with self._lock:
            width, height = self._require_frame()
            try:
                parsed = [TrackCalibrationPoint(float(p['image_x_px']), float(p['image_y_px']),
                                                float(p['position_m'])) for p in points]
                if any(not (0 <= p.image_x_px < width and 0 <= p.image_y_px < height) for p in parsed):
                    raise ValueError('轨道标定点超出画面')
                calibration = StraightTrackCalibration(parsed, {'frame_width': width, 'frame_height': height})
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                raise GlobalCameraError(f'无效的标定点：{exc}') from exc
            self._calibration, self._direction = calibration, direction
            self._revision += 1
            self._pending_roi = self._sample = None
            self._sample_revision = None
            self._message = '轨道标定已设置，请框选机器人头部'
        return self.status()

    def load_calibration(self, path=None, direction='left'):
        if direction not in ('left', 'right'):
            raise GlobalCameraError('头部前沿方向必须为 left 或 right')
        # Only the known, fixed local calibration; never accept an arbitrary web path.
        if path is not None:
            raise GlobalCameraError('不接受自定义文件路径')
        try:
            calibration = StraightTrackCalibration.load(Path.home() / 'Desktop' / 'global_camera_calibration.json')
        except Exception as exc:
            raise GlobalCameraError(f'无法加载桌面旧标定：{exc}') from exc
        with self._lock:
            size = self._require_frame()
            if calibration.expected_frame_size != size:
                raise GlobalCameraError(f'实际分辨率 {size} 与旧标定 {calibration.expected_frame_size} 不一致')
            self._calibration, self._direction = calibration, direction
            self._revision += 1
            self._pending_roi = self._sample = None
            self._sample_revision = None
            self._message = '已加载旧标定，请确认相机位置及焦距未改变，再框选头部'
        return self.status()

    def set_roi(self, bbox):
        try:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError('应提供 x、y、宽、高')
            x, y, w, h = (float(v) for v in bbox)
            if not all(math.isfinite(v) for v in (x, y, w, h)):
                raise ValueError('坐标必须是有限数值')
        except (ValueError, TypeError, OverflowError) as exc:
            raise GlobalCameraError(f'无效跟踪框：{exc}') from exc
        with self._lock:
            width, height = self._require_frame()
            if self._calibration is None:
                raise GlobalCameraError('请先标定轨道')
            if w < 4 or h < 4 or x < 0 or y < 0 or x+w > width or y+h > height:
                raise GlobalCameraError('跟踪框必须在图像内且至少 4×4 像素')
            x1, y1 = round(x), round(y)
            x2, y2 = min(width, round(x+w)), min(height, round(y+h))
            if x2-x1 < 4 or y2-y1 < 4:
                raise GlobalCameraError('跟踪框取整后必须至少 4×4 像素')
            # Invalidate an in-flight tracker update before accepting a new ROI.
            self._revision += 1
            self._pending_roi = (x1, y1, x2-x1, y2-y1)
            self._sample = None
            self._sample_revision = None
            self._message = '正在初始化跟踪'
        return self.status()

    def clear_roi(self):
        with self._lock:
            self._revision += 1
            self._pending_roi = self._sample = None
            self._sample_revision = None
            self._message = '跟踪已停止'
        return self.status()

    def status(self):
        with self._lock:
            now = time.monotonic()
            sample = self._sample
            age = None if sample is None else max(0., now-sample.timestamp_s)
            valid = bool(self._state == 'connected' and sample is not None and self._sample_revision == self._revision and sample.tracking_ok and sample.velocity_mps is not None and sample.is_fresh(.8, now))
            frame_age = None if self._frame_time is None else max(0., now-self._frame_time)
            if frame_age is None or frame_age > .8:
                valid = False
            points = [] if self._calibration is None else [dict(image_x_px=p.image_x_px, image_y_px=p.image_y_px, position_m=p.position_m) for p in self._calibration.points]
            return dict(state=self._state, message=self._message, camera_index=self._index,
                        revision=self._revision, sample_revision=self._sample_revision if valid else None,
                        frame_size=None if self._frame is None else [self._frame.shape[1], self._frame.shape[0]],
                        frame_age_s=frame_age, calibrated=bool(points), points=points, direction=self._direction,
                        valid=valid, sample_age_s=age, position_m=sample.position_m if valid else None,
                        velocity_mps=sample.velocity_mps if valid else None,
                        bbox=list(sample.bbox) if sample is not None and sample.bbox is not None and age is not None and age <= .8 else None,
                        sample_timestamp_monotonic_s=sample.timestamp_s if valid else None,
                        note=sample.note if sample is not None else '', clock='host_monotonic_frame_read')

    def latest_frame(self):
        with self._lock:
            if self._frame is None or self._frame_time is None or time.monotonic()-self._frame_time > 1:
                return None
            frame = self._frame.copy()
        import cv2
        ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return bytes(jpeg) if ok else None

    def _run(self, cv2, index, width, height, fps):
        capture = tracker = processor = None
        revision = -1
        try:
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if not capture.isOpened():
                raise RuntimeError(f'无法打开全局相机 {index}，请确认未被其他程序占用')
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            with self._lock:
                self._state, self._message = 'connected', '全局相机已连接，请标定轨道'
            while not self._stop.is_set():
                ok, frame = capture.read()
                now = time.monotonic()
                if not ok:
                    raise RuntimeError('全局相机读取失败')
                with self._lock:
                    if self._stop.is_set():
                        break
                    self._frame, self._frame_time = frame, now
                    calibration, direction, current_revision = self._calibration, self._direction, self._revision
                    roi, self._pending_roi = self._pending_roi, None
                if calibration is not None and calibration.expected_frame_size != (frame.shape[1], frame.shape[0]):
                    raise RuntimeError('全局相机分辨率变化，请重新标定')
                if current_revision != revision:
                    tracker = None
                    processor = RgbPositionProcessor(calibration, direction) if calibration else None
                    revision = current_revision
                if roi is not None and processor is not None:
                    tracker = RgbCameraSpeedSource._make_tracker(cv2)
                    if tracker.init(frame, roi) is False:
                        raise RuntimeError('头部跟踪初始化失败，请重新框选')
                    processor.reset()
                    continue
                if tracker is None or processor is None:
                    continue
                tracked, bbox = tracker.update(frame)
                if tracked:
                    sample = processor.add_bbox(now, tuple(float(v) for v in bbox), (frame.shape[1], frame.shape[0]))
                    message = '全局相机测速正常' if sample.tracking_ok else '跟踪预热中'
                else:
                    tracker = None
                    processor.reset()
                    sample = CameraSample(now, time.monotonic(), None, None, False, 'rgb_camera', '跟踪丢失，请重新框选')
                    message = '头部跟踪丢失，请重新框选'
                with self._lock:
                    if revision == self._revision and not self._stop.is_set():
                        self._sample, self._sample_revision, self._message = sample, revision, message
        except Exception as exc:
            with self._lock:
                if not self._stop.is_set():
                    self._state, self._message = 'error', f'全局相机错误：{exc}'
                    self._sample = self._frame = self._frame_time = None
                    self._sample_revision = None
        finally:
            if capture is not None:
                capture.release()
            with self._lock:
                if self._stop.is_set():
                    self._state, self._message = 'disconnected', '全局相机已断开'
                    self._sample = self._frame = self._frame_time = None
                    self._sample_revision = None
