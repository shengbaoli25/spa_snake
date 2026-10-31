"""Offline-only MobileNetV3-Small embeddings for received JPEG frames.

Receipt times and cutoff must share the host monotonic clock; they are not
camera exposure times. This module performs no camera or motor I/O.
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
import threading
from urllib.parse import urlparse


class VisionModelError(RuntimeError):
    """Weights or the local inference runtime could not be loaded."""


class MobileNetWindowEncoder:
    """Frozen ImageNet MobileNetV3-Small avgpool embeddings (576 values)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._transform = None
        self._torch = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
        except (ImportError, OSError, RuntimeError) as exc:
            raise VisionModelError(f"无法加载 PyTorch/torchvision：{exc}") from exc
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        filename = Path(urlparse(weights.url).path).name
        path = Path(torch.hub.get_dir()) / "checkpoints" / filename
        if not path.is_file():
            raise VisionModelError(
                f"缺少官方预训练权重 {path}；请手动从 {weights.url} 下载到该位置。"
            )
        try:
            # Verify the hash embedded in the official filename before loading.
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if not digest.startswith("047dcff4"):
                raise VisionModelError(f"预训练权重校验失败：{path}；请重新从 {weights.url} 下载。")
            state = torch.load(path, map_location="cpu", weights_only=True)
            model = mobilenet_v3_small(weights=None)
            model.load_state_dict(state, strict=True)
            model.eval()
            model.requires_grad_(False)
            self._torch = torch
            self._transform = weights.transforms()
            self._model = model
        except VisionModelError:
            raise
        except Exception as exc:
            raise VisionModelError(f"无法加载本地预训练权重：{exc}") from exc

    def encode_window(self, frames, cutoff_ns, max_frames=8):
        """Encode up to eight newest frames received by ``cutoff_ns``.

        Each input mapping requires ``jpeg`` bytes and integer
        ``receive_monotonic_ns``. Results are ordered oldest-to-newest. A
        broken JPEG stays in the result with ``valid=False`` and no features;
        frames from the future and invalid timestamps are excluded and counted.
        No black or synthetic frame is inserted for missing input.
        """
        if isinstance(cutoff_ns, bool) or not isinstance(cutoff_ns, int) or cutoff_ns < 0:
            raise ValueError("cutoff_ns 必须是非负整数单调时钟纳秒")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 1 <= max_frames <= 8:
            raise ValueError("max_frames 必须在 1 到 8 之间")
        from PIL import Image, UnidentifiedImageError

        eligible = []
        future = invalid_time = 0
        for frame in frames or ():
            if not isinstance(frame, dict):
                invalid_time += 1
                continue
            timestamp = frame.get("receive_monotonic_ns")
            if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
                invalid_time += 1
            elif timestamp > cutoff_ns:
                future += 1
            else:
                eligible.append((timestamp, frame))
        eligible.sort(key=lambda item: item[0])
        eligible = eligible[-max_frames:]
        output = []
        decoded = []
        for timestamp, frame in eligible:
            item = {"receive_monotonic_ns": timestamp, "age_s": (cutoff_ns - timestamp) / 1e9,
                    "valid": False, "features": None, "error": None}
            jpeg = frame.get("jpeg")
            if not isinstance(jpeg, bytes) or not jpeg:
                item["error"] = "missing_jpeg"
            elif len(jpeg) > 8 * 1024 * 1024:
                item["error"] = "jpeg_too_large"
            else:
                try:
                    with Image.open(io.BytesIO(jpeg)) as image:
                        if image.format != "JPEG":
                            raise ValueError("not_jpeg")
                        image.load()
                        decoded.append((len(output), image.convert("RGB")))
                except (OSError, ValueError, UnidentifiedImageError):
                    item["error"] = "invalid_jpeg"
            output.append(item)
        if decoded:
            with self._lock:
                self._load()
                torch = self._torch
                batch = torch.stack([self._transform(image) for _, image in decoded])
                with torch.no_grad():
                    embedding = self._model.avgpool(self._model.features(batch)).flatten(1)
                if tuple(embedding.shape) != (len(decoded), 576):
                    raise VisionModelError(f"视觉特征维度错误：{tuple(embedding.shape)}")
                for (index, _), vector in zip(decoded, embedding.tolist()):
                    if not all(math.isfinite(value) for value in vector):
                        raise VisionModelError("视觉特征包含非有限数值")
                    output[index].update(valid=True, features=vector)
        return {"frames": output, "valid_count": sum(item["valid"] for item in output),
                "future_frames_excluded": future, "invalid_timestamps_excluded": invalid_time,
                "cutoff_monotonic_ns": cutoff_ns, "clock_sync": "not_calibrated"}
