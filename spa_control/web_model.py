"""Explicit, synthetic offline check of the full serial-preview inference path.

The current machine can use its existing local CPU PyTorch cache; a portable
installation should install the optional dependencies in requirements_ml.txt.
Nothing in this module transmits commands to a robot.
"""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import sys
import time


def _ensure_torch():
    if importlib.util.find_spec("torch") is None:
        cache = Path.home() / ".cache" / "ocp-model-validation"
        if (cache / "torch" / "__init__.py").is_file():
            sys.path.append(str(cache))
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("缺少 PyTorch；请按 README_web.md 安装可选机器学习依赖")


def run_offline_check(camera):
    """Run pretrained vision + untrained SPA/PPO on synthetic drive samples.

    If a fresh user-connected camera frame exists it is used for the picture;
    otherwise the test image is generated locally. Drive values and quality
    data are always synthetic, and no result is eligible for real control.
    """
    _ensure_torch()
    try:
        import torch
        from PIL import Image
        from .serial_preview import CategoricalPPO
        from .serial_preview_model import ModelConfig, SerialPreviewModel
        from .serial_preview_pipeline import DrivePacket, PipelineScales, SerialPreviewPipeline
        from .serial_preview_vision import MobileNetWindowEncoder

        torch.set_num_threads(1)
        camera_status, frame = camera.status(), camera.latest_frame()
        # Snapshot first: the camera thread may publish a new frame between
        # calls, so take the as-of cutoff only after choosing that frame.
        cutoff = time.monotonic_ns()
        if camera_status.get("active") and frame and frame.get("age_s", float("inf")) <= 2:
            jpeg = frame["jpeg"]
            timestamp = frame["receive_monotonic_ns"]
            source = "live_camera"
        else:
            image = Image.new("RGB", (320, 240), (88, 128, 96))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            jpeg, timestamp, source = buffer.getvalue(), cutoff - 50_000_000, "synthetic"
        frames = [{"jpeg": jpeg, "receive_monotonic_ns": timestamp}]
        drives = []
        # These are illustrative inputs, deliberately never real CAN data.
        for age_ms in range(4400, 0, -100):
            stamp = cutoff - age_ms * 1_000_000
            for channel in range(4):
                drives.append(DrivePacket(channel, stamp, 100. + channel, 20. + channel,
                                          -300 if channel % 2 == 0 else 300, stamp))
        with torch.random.fork_rng():
            torch.manual_seed(0)
            model = SerialPreviewModel(ModelConfig(recent_dim=24, front_dim=12, quality_dim=5))
        model.set_phase("frozen")
        policy = CategoricalPPO(obs_dim=159, seed=0)
        pipeline = SerialPreviewPipeline(model, policy, MobileNetWindowEncoder(),
                                         PipelineScales(7000., 500., .25, .5))
        result = pipeline.infer(frames=frames, drives=drives, cutoff_ns=cutoff,
                                quality=[0., 0., 0., 0., 1.], nominal=[300]*4,
                                signs=[-1, 1, -1, 1], rear=[False, False, True, True],
                                active=[True]*4)
        return {"image_source": source, "drive_source": "synthetic",
                "vision_weights": "MobileNetV3-Small ImageNet1K V1",
                "model_trained_on_robot": False, "physical_policy_allowed": False,
                "action": result["action"], "probabilities": result["candidate_probabilities"],
                "null_weight": result["null_weight"],
                "history_available_count": sum(result["history_available"]),
                "notice": "仅整链检查：四路电机反馈为合成数据，SPA/PPO 模型未经真实数据训练。"}
    except Exception as exc:
        raise RuntimeError(f"模型整链检查失败：{exc}") from exc
