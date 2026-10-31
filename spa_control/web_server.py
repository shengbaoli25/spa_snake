"""Loopback workbench with explicit bounded motor tests and two-section SPA.

Startup opens no devices. Motion requires an explicit same-origin POST; all
commands use the serial owner's independent guards and expiring motion lease.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .web_camera import CameraError, WebCamera
from .web_motor import WebMotor
from .two_motor_record import TwoMotorRecorder
from .web_global_camera import WebGlobalCamera


STATIC = Path(__file__).with_name("web_static")
ALLOWED_STATIC = {"/": "index.html", "/index.html": "index.html",
                  "/style.css": "style.css", "/app.js": "app.js"}
MAX_BODY = 16384


class Workbench:
    def __init__(self, camera=None, simulation=None, motor=None, recorder=None, global_camera=None, experiment=None):
        self.camera = camera if camera is not None else WebCamera()
        self.global_camera = global_camera if global_camera is not None else WebGlobalCamera()
        self.motor = motor if motor is not None else WebMotor()
        self.recorder = recorder if recorder is not None else TwoMotorRecorder(self.camera, self.motor, self.global_camera)
        if experiment is None:
            from .web_experiment import WebExperiment
            experiment = WebExperiment(self.motor, self.camera, self.global_camera)
        self.experiment = experiment
        self._control_lock = threading.RLock()
        self._control_at = None
        self._control_watchdog = None
        self._closing = threading.Event()
        if simulation is None:
            from .serial_preview import DemoExperiment
            simulation = DemoExperiment()
        self.simulation = simulation
        self._simulation_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._model_result = None

    def status(self):
        camera = self.camera.status()
        frame = self.camera.latest_frame()
        camera.update(status=camera.get("message", ""),
                      received=camera.get("frames_received", 0),
                      age_s=camera.get("frame_age_s"),
                      sequence=frame["sequence"] if frame else None)
        with self._simulation_lock:
            simulation = self.simulation.status()
        experiment = self.experiment.status()
        return {"camera": camera,
                "algorithm": {"status": "两节 SPA · 实机开发",
                              "description": "真实区间奖励、监督预训练与新轨迹 PPO；合成检查独立保存。",
                              "source": "Serial Preview Augmentation，2026-10-05 本地论文快照",
                              "real_control": True},
                "simulation": simulation, "model_check": self._model_result,
                "motor": self.motor.status(), "recorder": self.recorder.status(),
                "global_camera": self.global_camera.status(), "experiment": experiment}

    def _motion_busy(self):
        return self.motor.status().get("motion", {}).get("state") in ("preparing", "running", "stopping")

    def _clear_busy(self):
        return self.motor.status().get("clear_state") in ("checking", "waiting_ack", "verifying")

    def _renew_control(self):
        self._control_at = time.monotonic()
        if self._control_watchdog is None or not self._control_watchdog.is_alive():
            self._control_watchdog = threading.Thread(target=self._watch_control, name="web-control-watchdog", daemon=True)
            self._control_watchdog.start()

    def heartbeat(self):
        with self._control_lock:
            self._control_at = time.monotonic()
        return {"ok": True, "timeout_s": 2.0}

    def _watch_control(self):
        while not self._closing.wait(.1):
            with self._control_lock:
                if self._control_at is None:
                    continue
                if not self._motion_busy() and not self.experiment.status().get("active"):
                    self._control_at = None
                    continue
                if time.monotonic() - self._control_at > 2.0:
                    try:
                        self._stop_control("网页心跳超时")
                    except RuntimeError:
                        # Motor status reports a disconnected/failed owner;
                        # no stale producer is allowed to continue renewing.
                        pass

    def _require_open(self):
        if self._closing.is_set():
            raise ValueError("服务正在关闭，不能开始新操作")

    def start_test(self, data):
        with self._control_lock:
            self._require_open()
            if self.experiment.status().get("active"):
                raise ValueError("实验或训练正在进行，请先停止")
            if self._clear_busy():
                raise ValueError("故障清除正在进行，请等待结果")
            result = self.motor.start_test(data.get("motor"), data.get("rpm", -60), data.get("duration_s", 3))
            self._renew_control()
            return result

    def configure_experiment(self, data):
        with self._control_lock:
            self._require_open()
            if self._motion_busy() or self._clear_busy():
                raise ValueError("请先停止电机再保存配置")
            return self.experiment.configure(data)

    def clear_motor_fault(self):
        with self._control_lock:
            self._require_open()
            if self.experiment.status().get("active") or self._motion_busy():
                raise ValueError("请先停止实验和电机，再清除故障")
            self._control_at = None
            return self.motor.clear_fault()

    def start_experiment(self, data):
        with self._control_lock:
            self._require_open()
            if self._motion_busy() or self._clear_busy():
                raise ValueError("电机测试或故障清除正在进行，请等待停止或清除结果")
            result = self.experiment.start(data.get("mode", "baseline"), data)
            self._renew_control()
            return result

    def pretrain_experiment(self, data):
        with self._control_lock:
            self._require_open()
            if self._motion_busy() or self._clear_busy():
                raise ValueError("请先停止电机再训练预测器")
            self._control_at = None
            return self.experiment.pretrain(data)

    def _stop_control(self, reason):
        # Cancel the producer first, then the serial owner. No producer can
        # re-arm a stopped motion without a new explicit start request.
        try:
            self.experiment.stop()
        finally:
            try:
                result = self.motor.stop_motion(reason=reason)
            finally:
                self._control_at = None
        return result

    def stop_control(self):
        with self._control_lock:
            return self._stop_control("网页停止")

    def kinematics_preview(self, data):
        from .kinematics import calculate_reference, speed_tracking_reward
        result = calculate_reference(data.get("motor_rpm"), data.get("lead_mm", 30),
                                     data.get("motor_to_helix_ratio", 1),
                                     data.get("forward_sign", 1), data.get("distance_m"))
        camera = self.global_camera.status()
        result["tracking"] = (speed_tracking_reward(result["theory_speed_mps"], camera["velocity_mps"])
                              if camera.get("valid") and camera.get("velocity_mps") is not None else None)
        result["camera_position_m"] = camera.get("position_m") if camera.get("valid") else None
        result["notice"] = "理想无滑移参考；速度奖励仅为论文跟踪项，非完整区间奖励。未发送运动命令。"
        return result

    def reset(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("种子必须是 0 到 4294967295 的整数")
        with self._simulation_lock:
            return self.simulation.reset(seed)

    def step(self):
        with self._simulation_lock:
            return self.simulation.step()

    def train(self):
        with self._simulation_lock:
            return self.simulation.train_batch()

    def check_model(self):
        if not self._model_lock.acquire(blocking=False):
            raise ValueError("模型整链检查正在进行")
        try:
            from .web_model import run_offline_check
            self._model_result = run_offline_check(self.camera)
            return self._model_result
        finally:
            self._model_lock.release()

    def check_two_motor_model(self, front_motor):
        if type(front_motor) is not int or front_motor not in (3, 4):
            raise ValueError("请选择假设在前的电机 M3 或 M4")
        if not self._model_lock.acquire(blocking=False):
            raise ValueError("模型检查正在进行")
        try:
            from .web_model import _ensure_torch
            _ensure_torch()
            from .two_motor_preview import run_two_motor_demo
            return run_two_motor_demo(front_motor)
        finally:
            self._model_lock.release()

    def close(self):
        with self._control_lock:
            self._closing.set()
            self.experiment.close()
            self.recorder.close()
            self.motor.close()
            self.global_camera.close()
            self.camera.disconnect()


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, workbench=None):
        if address[0] not in ("127.0.0.1", "localhost"):
            raise ValueError("The workbench must bind to loopback")
        self.workbench = workbench if workbench is not None else Workbench()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server_version = "SPAWorkbench/1"

    def log_message(self, format, *args):
        # No query strings, image content or device metadata in access logs.
        return

    def _valid_host(self):
        host = self.headers.get("Host", "").lower()
        port = self.server.server_address[1]
        return host in (f"127.0.0.1:{port}", f"localhost:{port}")

    def _send(self, status, payload, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, status, data):
        payload = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(status, payload, "application/json; charset=utf-8")

    def _host_check(self):
        if not self._valid_host():
            self._json(403, {"error": "仅允许从本机访问"})
            return False
        return True

    def do_GET(self):
        if not self._host_check():
            return
        path = urlsplit(self.path).path
        try:
            if path == "/api/status":
                self._json(200, self.server.workbench.status())
            elif path == "/api/ports":
                self._json(200, {"ports": self.server.workbench.camera.ports()})
            elif path == "/api/camera/frame":
                camera = self.server.workbench.camera
                status, frame = camera.status(), camera.latest_frame()
                if not status["active"] or not frame or frame["age_s"] > 2.0:
                    self._json(503, {"error": "没有新鲜的相机画面"})
                else:
                    self._send(200, frame["jpeg"], "image/jpeg")
            elif path == "/api/global-camera/frame":
                frame = self.server.workbench.global_camera.latest_frame()
                if frame is None:
                    self._json(503, {"error": "没有新鲜的全局相机画面"})
                else:
                    self._send(200, frame, "image/jpeg")
            elif path in ALLOWED_STATIC:
                target = STATIC / ALLOWED_STATIC[path]
                if not target.is_file():
                    self._json(404, {"error": "页面资源不存在"})
                    return
                mime = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                if mime.startswith("text/") or mime == "application/javascript":
                    mime += "; charset=utf-8"
                self._send(200, target.read_bytes(), mime)
            else:
                self._json(404, {"error": "接口不存在"})
        except (CameraError, ValueError, RuntimeError) as exc:
            self._json(400, {"error": str(exc)})

    def do_POST(self):
        if not self._host_check():
            return
        origin = self.headers.get("Origin")
        if origin and origin.lower() != "http://" + self.headers["Host"].lower():
            self._json(403, {"error": "请求来源不匹配"})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            self._json(415, {"error": "请求必须是 JSON"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= length <= MAX_BODY:
                raise ValueError("请求长度无效")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("请求应为 JSON 对象")
            path = urlsplit(self.path).path
            camera = self.server.workbench.camera
            if path == "/api/camera/connect":
                result = camera.connect(data.get("port"))
            elif path == "/api/camera/disconnect":
                result = camera.disconnect()
            elif path == "/api/camera/light":
                result = camera.set_light(data.get("enabled"))
            elif path == "/api/simulation/reset":
                result = self.server.workbench.reset(data.get("seed", 0))
            elif path == "/api/simulation/step":
                result = self.server.workbench.step()
            elif path == "/api/simulation/train":
                result = self.server.workbench.train()
            elif path == "/api/model/check":
                result = self.server.workbench.check_model()
            elif path == "/api/model/two":
                result = self.server.workbench.check_two_motor_model(data.get("front_motor"))
            elif path == "/api/motor/connect":
                result = self.server.workbench.motor.connect()
            elif path == "/api/motor/disconnect":
                result = self.server.workbench.motor.disconnect()
            elif path == "/api/motor/clear-usb-closed":
                result = self.server.workbench.motor.clear_usb_closed()
            elif path == "/api/motor/clear-fault":
                result = self.server.workbench.clear_motor_fault()
            elif path == "/api/motor/test":
                result = self.server.workbench.start_test(data)
            elif path == "/api/motor/stop":
                result = self.server.workbench.stop_control()
            elif path == "/api/control/heartbeat":
                result = self.server.workbench.heartbeat()
            elif path == "/api/experiment/configure":
                result = self.server.workbench.configure_experiment(data)
            elif path == "/api/experiment/start":
                result = self.server.workbench.start_experiment(data)
            elif path == "/api/experiment/pretrain":
                result = self.server.workbench.pretrain_experiment(data)
            elif path == "/api/experiment/stop":
                self.server.workbench.stop_control()
                result = self.server.workbench.experiment.status()
            elif path == "/api/record/start":
                result = self.server.workbench.recorder.start(data.get("duration_s"))
            elif path == "/api/record/stop":
                result = self.server.workbench.recorder.stop()
            elif path == "/api/kinematics/preview":
                result = self.server.workbench.kinematics_preview(data)
            elif path == "/api/global-camera/connect":
                result = self.server.workbench.global_camera.connect(data.get("camera_index", 0), data.get("width", 1280), data.get("height", 720), data.get("fps", 30))
            elif path == "/api/global-camera/disconnect":
                result = self.server.workbench.global_camera.disconnect()
            elif path == "/api/global-camera/calibration":
                result = self.server.workbench.global_camera.set_calibration(data.get("points"), data.get("direction", "left"))
            elif path == "/api/global-camera/load-calibration":
                result = self.server.workbench.global_camera.load_calibration(direction=data.get("direction", "left"))
            elif path == "/api/global-camera/roi":
                result = self.server.workbench.global_camera.set_roi(data.get("bbox"))
            elif path == "/api/global-camera/clear-roi":
                result = self.server.workbench.global_camera.clear_roi()
            else:
                self._json(404, {"error": "接口不存在"})
                return
            self._json(200, result)
        except (CameraError, ValueError, RuntimeError, UnicodeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口必须在 1..65535")
    server = LocalServer(("127.0.0.1", args.port))
    print(f"SPA 本地工作台：http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.workbench.close()
        server.server_close()


if __name__ == "__main__":
    main()
