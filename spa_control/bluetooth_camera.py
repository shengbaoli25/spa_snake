"""Bounded JPEG transport for the independent ESP32-CAM over SPP or USB UART."""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import struct
import threading
import time
import zlib
import tkinter as tk
from tkinter import ttk

MAGIC = b"SPACAM01"
HEADER = struct.Struct("<8sIIQII")
MAX_JPEG = 128 * 1024
USB_BAUD = 921600
REQUEST_INTERVAL_S = 0.05
PREVIEW_INTERVAL_MS = 33


@dataclass(frozen=True)
class CameraPacket:
    boot_id: int
    sequence: int
    capture_us: int
    jpeg: bytes


class FrameParser:
    def __init__(self):
        self.buffer = bytearray()
        self.rejected = 0

    def feed(self, data):
        self.buffer.extend(data)
        packets = []
        while True:
            offset = self.buffer.find(MAGIC)
            if offset < 0:
                self.buffer[:] = self.buffer[-(len(MAGIC) - 1):]
                break
            if offset:
                del self.buffer[:offset]
            if len(self.buffer) < HEADER.size:
                break
            _, boot, sequence, capture_us, length, crc = HEADER.unpack_from(self.buffer)
            if not 4 <= length <= MAX_JPEG:
                self.rejected += 1
                del self.buffer[0]
                continue
            if len(self.buffer) < HEADER.size + length:
                break
            jpeg = bytes(self.buffer[HEADER.size:HEADER.size + length])
            del self.buffer[:HEADER.size + length]
            if (zlib.crc32(jpeg) != crc or not jpeg.startswith(b"\xff\xd8")
                    or not jpeg.endswith(b"\xff\xd9")):
                self.rejected += 1
                continue
            packets.append(CameraPacket(boot, sequence, capture_us, jpeg))
        return packets


class CameraReceiver:
    def __init__(self, state_provider=None):
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.port = None
        self.state_provider = state_provider
        self.latest = None
        self.status = "未连接相机"
        self.received = 0
        self.record_dir = None
        self.light = False
        self.active = False

    def connect(self, port, baud=USB_BAUD):
        if self.active:
            return
        if not port:
            raise ValueError("请选择相机串口")
        self.stop_event = threading.Event()
        with self.lock:
            self.active = True
            self.status = f"正在打开 {port}"
            self.latest = None
            self.received = 0
        self.thread = threading.Thread(target=self._run, args=(port, baud), daemon=True,
                                       name="head-camera-receiver")
        self.thread.start()

    def disconnect(self):
        self.stop_event.set()
        # The worker owns serial I/O; its reads/writes have bounded timeouts.

    def snapshot(self):
        with self.lock:
            return self.active, self.status, self.latest, self.received

    def set_recording(self, directory):
        with self.lock:
            self.record_dir = directory

    def _run(self, port_name, baud):
        port = None
        try:
            import serial
            from PIL import Image
            port = serial.Serial(port=None, baudrate=baud, timeout=0.01, write_timeout=1)
            port.dtr = False
            port.rts = False
            port.port = port_name
            port.open()
            self.port = port
            port.write(b"?")
            parser = FrameParser()
            pending = None
            next_request = 0.0
            sent_light = None
            previous = None
            tail = bytearray()
            while not self.stop_event.is_set():
                now = time.monotonic()
                with self.lock:
                    light = self.light
                if pending is None and now >= next_request:
                    if sent_light != light:
                        port.write(b"L" if light else b"l")
                        sent_light = light
                    port.write(b"F")
                    pending = now
                data = port.read(min(max(port.in_waiting, 1), 8192))
                if data:
                    tail.extend(data)
                    tail[:] = tail[-256:]
                for packet in parser.feed(data):
                    receive_ns = time.monotonic_ns()
                    with Image.open(io.BytesIO(packet.jpeg)) as image:
                        if image.width > 1280 or image.height > 960:
                            raise ValueError("相机图像尺寸超过接收上限")
                        preview = image.convert("RGB")
                        preview.thumbnail((640, 320))
                    elapsed = (receive_ns - previous) / 1e9 if previous else None
                    previous = receive_ns
                    latest = (packet, receive_ns, preview)
                    fps = f"{1 / elapsed:.1f} fps" if elapsed and elapsed > 0 else "首帧"
                    with self.lock:
                        self.latest = latest
                        self.received += 1
                        self.status = (f"{port_name} · {fps} · 帧 {packet.sequence} · "
                                       f"JPEG {len(packet.jpeg) / 1024:.1f} KB · 校验失败 {parser.rejected}")
                        record_dir = self.record_dir
                    if record_dir:
                        self._record(record_dir, packet, receive_ns)
                    pending = None
                    next_request = time.monotonic() + REQUEST_INTERVAL_S
                if pending is not None and time.monotonic() - pending > 8:
                    detail = tail.decode("ascii", "ignore").strip()
                    raise TimeoutError(f"8秒内未收到完整图像；请检查配对串口、IO0和相机。{detail[:100]}")
        except Exception as exc:
            with self.lock:
                self.status = f"相机连接失败: {exc}"
        finally:
            if port:
                try:
                    port.close()
                except Exception:
                    pass
            with self.lock:
                self.active = False
                if self.stop_event.is_set():
                    self.status = "相机已断开"

    def _record(self, directory, packet, receive_ns):
        try:
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            stem = f"{packet.boot_id:08x}_{packet.sequence:010d}_{receive_ns}"
            (directory / f"{stem}.jpg").write_bytes(packet.jpeg)
            state = self.state_provider() if self.state_provider else None
            metadata = {"image": f"{stem}.jpg", "camera_boot_id": packet.boot_id,
                        "frame_sequence": packet.sequence, "camera_capture_us": packet.capture_us,
                        "host_receive_monotonic_ns": receive_ns,
                        "clock_sync": "not_calibrated", "robot_state_at_receive": state}
            with (directory / "frames.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        except Exception as exc:
            with self.lock:
                self.record_dir = None
                self.status = f"图像接收正常，保存失败: {exc}"


class CameraPanel(ttk.Frame):
    def __init__(self, parent, output_dir, state_provider=None):
        super().__init__(parent, padding=8)
        self.output_dir = Path(output_dir)
        self.receiver = CameraReceiver(state_provider)
        self.closed = False
        self.last_frame = None
        self.last_view_size = None
        self.photo = None
        self.recording = False
        self.port_var = tk.StringVar()
        self.status_var = tk.StringVar(value="未连接相机")
        bar = ttk.Frame(self)
        bar.pack(fill="x", pady=(0, 6))
        ttk.Label(bar, text="相机串口").pack(side="left")
        self.selector = ttk.Combobox(bar, textvariable=self.port_var, state="readonly", width=42)
        self.selector.pack(side="left", padx=6)
        ttk.Button(bar, text="刷新", command=self.refresh_ports).pack(side="left", padx=3)
        self.connect_button = ttk.Button(bar, text="连接相机", command=self.toggle_connection)
        self.connect_button.pack(side="left", padx=3)
        self.record_button = ttk.Button(bar, text="开始录制", command=self.toggle_recording)
        self.record_button.pack(side="left", padx=3)
        self.light_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="补光", variable=self.light_var,
                        command=self.set_light).pack(side="left", padx=6)
        ttk.Label(self, textvariable=self.status_var, wraplength=1040).pack(fill="x")
        self.view = tk.Canvas(self, background="#151515", highlightthickness=0, height=240)
        self.view.pack(fill="both", expand=True, pady=(6, 0))
        self.refresh_ports()
        self.after_id = self.after(PREVIEW_INTERVAL_MS, self.tick)

    def refresh_ports(self):
        self.ports = {}
        try:
            from serial.tools import list_ports
            for port in list_ports.comports():
                self.ports[f"{port.device} · {port.description}"] = port.device
        except ImportError:
            self.status_var.set("缺少pyserial，请安装requirements.txt")
        self.selector.configure(values=tuple(self.ports))
        if self.port_var.get() not in self.ports:
            self.port_var.set("")

    def toggle_connection(self):
        if self.receiver.snapshot()[0]:
            self.receiver.disconnect()
        else:
            try:
                self.receiver.connect(self.ports.get(self.port_var.get()))
            except ValueError as exc:
                self.status_var.set(str(exc))

    def set_light(self):
        with self.receiver.lock:
            self.receiver.light = self.light_var.get()

    def toggle_recording(self):
        self.recording = not self.recording
        directory = self.output_dir / time.strftime("%Y%m%d_%H%M%S") if self.recording else None
        self.receiver.set_recording(directory)
        self.record_button.configure(text="停止录制" if self.recording else "开始录制")

    def tick(self):
        if self.closed:
            return
        active, status, latest, count = self.receiver.snapshot()
        self.status_var.set(status)
        self.connect_button.configure(text="断开相机" if active else "连接相机")
        self.selector.configure(state="disabled" if active else "readonly")
        with self.receiver.lock:
            if self.recording and self.receiver.record_dir is None:
                self.recording = False
                self.record_button.configure(text="开始录制")
        view_size = (self.view.winfo_width(), self.view.winfo_height())
        if (self.winfo_viewable() and latest
                and (self.last_frame != latest[1] or self.last_view_size != view_size)):
            from PIL import ImageTk
            packet, timestamp, image = latest
            fitted = image.copy()
            fitted.thumbnail((max(1, view_size[0] - 16), max(1, view_size[1] - 16)))
            self.photo = ImageTk.PhotoImage(fitted, master=self)
            self.view.delete("all")
            self.view.create_image(view_size[0] / 2, view_size[1] / 2,
                                   image=self.photo, anchor="center")
            self.last_frame = timestamp
            self.last_view_size = view_size
        if latest and time.monotonic_ns() - latest[1] > 2_000_000_000:
            self.status_var.set(status + " · 画面已过时")
        self.after_id = self.after(PREVIEW_INTERVAL_MS, self.tick)

    def close(self):
        self.closed = True
        self.after_cancel(self.after_id)
        self.receiver.disconnect()


def main():
    root = tk.Tk()
    root.title("SPA · 头部蓝牙相机")
    root.geometry("960x620")
    panel = CameraPanel(root, Path(__file__).resolve().parent / "runs" / "head_camera")
    panel.pack(fill="both", expand=True)
    def close():
        panel.close()
        root.destroy()
    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


if __name__ == "__main__":
    main()
