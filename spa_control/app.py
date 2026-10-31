"""Four-motor baseline UI, without camera or gamepad inputs."""
from __future__ import annotations
import json
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .core import ExperimentConfig
from .session import DemoRobot, TrialRunner, baseline_commands
from .transport import RobotLink, motor_feedback_state
from .bluetooth_camera import CameraPanel

APP_DIR = Path(__file__).resolve().parent
BG, INK, BLUE = "#edf2f7", "#162a43", "#176de5"
COLORS = ("#176de5", "#0b9778", "#dc8530", "#8a54c0")
DIRECTIONS = {"正转 (+)": 1, "反转 (−)": -1}


class Workbench(tk.Tk):
    def __init__(self, demo=False):
        super().__init__()
        self.title("SPA · 电机调试与头部相机")
        self.geometry("1220x920")
        self.minsize(1080, 880)
        self.configure(bg=BG)
        self.demo, self.pending, self.closing = demo, False, False
        self.runner = TrialRunner(APP_DIR / "runs")
        self.events = queue.Queue()
        self.latest_gui_error = None
        self.vars, self.controls = {}, []
        self.motor_speeds, self.motor_directions, self.motor_readouts = [], [], []
        self.single_stop_buttons = []
        self.single_start_buttons = []
        self.last_diagnostic_text = None
        self.last_diagnostic_refresh = 0.0
        self.active_joint = None
        self.single_duration = tk.StringVar(value="3")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=INK, font=("Microsoft YaHei UI", 10))
        style.configure("TButton", padding=(12, 9), font=("Microsoft YaHei UI", 10))
        style.configure("Accent.TButton", background=BLUE, foreground="white")
        style.configure("Compact.TButton", padding=(10, 5))
        style.configure("Treeview", rowheight=33, font=("Consolas", 12))
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<Escape>", lambda _: self.stop_trial("键盘急停"))
        self.bind("<FocusOut>", lambda _: self.after(150, self.check_focus))
        self.after(100, self.tick)
        if demo:
            self.runner.link = DemoRobot(self.get_config())
            self.runner.message = "离线演示就绪；仅模拟电机遥测，不连接机器人"

    def _build(self):
        header = tk.Frame(self, bg=INK)
        header.pack(fill="x")
        tk.Label(header, text="SPA  /  MOTOR BASELINE", bg=INK, fg="white", font=("Segoe UI", 21, "bold")).pack(side="left", padx=22, pady=18)
        tk.Label(header, text="单路调试 / 初始策略", bg=INK, fg="#c4d7ec", font=("Microsoft YaHei UI", 12)).pack(side="left", padx=15)
        tk.Button(header, text="急停  ESC", command=lambda: self.stop_trial("急停按钮"), bg="#c63043", fg="white", font=("Microsoft YaHei UI", 14, "bold"), relief="flat", padx=24, pady=8).pack(side="right", padx=20)
        outer = ttk.Frame(self, padding=18)
        outer.pack(fill="both", expand=True)
        connection = ttk.Frame(outer)
        connection.pack(fill="x", pady=(0, 14))
        ttk.Label(connection, text="以太网直连").pack(side="left")
        for key, value, width in (("host", "192.168.10.2", 17), ("port", "9001", 6)):
            self.vars[key] = tk.StringVar(value=value)
            entry = ttk.Entry(connection, textvariable=self.vars[key], width=width)
            entry.pack(side="left", padx=5)
            self.controls.append(entry)
        self.connect_button = ttk.Button(connection, text="连接机器人", command=self.connect_robot)
        self.connect_button.pack(side="left", padx=8)
        self.link_status = tk.StringVar(value="未连接 · 不使用 HTTP 代理")
        ttk.Label(connection, textvariable=self.link_status, foreground=BLUE).pack(side="left", padx=10)
        ttk.Label(connection, text="离线演示 · 无实机通信" if self.demo else "标准初始策略").pack(side="right")
        parameters = ttk.LabelFrame(outer, text="运行参数", padding=10)
        parameters.pack(fill="x")
        defaults = ExperimentConfig()
        fields = [("共同速度（点应用生效）", "nominal_command", defaults.nominal_command),
                  ("四路运行时长 / s", "duration_s", defaults.duration_s),
                  ("刷新频率 / Hz", "control_hz", defaults.control_hz),
                  ("指令上限", "command_limit", defaults.command_limit),
                  ("四路通道（结构顺序）", "channels", "1,2,3,4"),
                  ("电流限值 / raw", "current_limit_raw", defaults.current_limit_raw),
                  ("遥测超时 / s", "sensor_timeout_s", defaults.sensor_timeout_s)]
        for index, (label, key, value) in enumerate(fields):
            row, col = divmod(index, 4)
            group = ttk.Frame(parameters)
            group.grid(row=row, column=col, sticky="ew", padx=10, pady=6)
            parameters.columnconfigure(col, weight=1)
            ttk.Label(group, text=label).pack(anchor="w", pady=(0, 5))
            self.vars[key] = tk.StringVar(value=str(int(value)) if isinstance(value, float) and value.is_integer() else str(value))
            entry = ttk.Entry(group, textvariable=self.vars[key], width=22)
            entry.pack(fill="x")
            self.controls.append(entry)
            self.vars[key].trace_add("write", lambda *_: self.preview())
        common_button = ttk.Button(parameters, text="共同速度应用到四路", command=self.apply_common_speed)
        common_button.grid(row=1, column=3, padx=10, pady=6, sticky="ew")
        self.controls.append(common_button)
        self.command_preview = tk.StringVar()
        ttk.Label(parameters, textvariable=self.command_preview, foreground=BLUE, font=("Consolas", 13), wraplength=1120).grid(row=2, column=0, columnspan=4, sticky="w", padx=10, pady=(12, 3))
        ttk.Label(parameters, text="按下方每路方向和速度运行。速度为协议设定值，实际转子 rpm 看反馈；残差为 0。", foreground="#64778d", wraplength=1110).grid(row=3, column=0, columnspan=4, sticky="w", padx=10, pady=5)
        presets = ttk.Frame(parameters)
        presets.grid(row=4, column=0, columnspan=4, sticky="ew", padx=10, pady=(5, 0))
        for label, reference in (("四路交替（当前要求）", False), ("附件旧接线（M1–M4）", True)):
            button = ttk.Button(presets, text=label, command=lambda ref=reference: self.apply_forward_preset(ref))
            button.pack(side="left", padx=(0, 8))
            self.controls.append(button)
        original_cadence = ttk.Button(presets, text="原版重发 2 Hz", command=self.use_original_cadence)
        original_cadence.pack(side="left", padx=(0, 8))
        self.controls.append(original_cadence)
        self.preset_status = tk.StringVar()
        ttk.Label(presets, textvariable=self.preset_status, foreground="#64778d").pack(side="left", padx=4)
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=12)
        self.start_button = ttk.Button(toolbar, text="四路一起运行", style="Accent.TButton", command=self.start_trial)
        self.start_button.pack(side="left", padx=(0, 10))
        ttk.Button(toolbar, text="停止全部", command=lambda: self.stop_trial("操作者停止")).pack(side="left", padx=(0, 10))
        for label, command in (("保存初始策略", self.save_config), ("读取初始策略", self.load_config)):
            button = ttk.Button(toolbar, text=label, command=command)
            button.pack(side="left", padx=(0, 10))
            self.controls.append(button)
        ttk.Label(toolbar, text="PPO 未启用", foreground="#64778d").pack(side="right")
        self.display_tabs = ttk.Notebook(outer)
        motors = ttk.LabelFrame(self.display_tabs, text="逐个确认电机，再保存初始策略", padding=10)
        self.display_tabs.add(motors, text="电机独立控制")
        note = ttk.Frame(motors)
        note.grid(row=0, column=0, columnspan=9, sticky="ew", pady=(0, 8))
        ttk.Label(note, text="单路运行时长 / s").pack(side="left")
        duration_entry = ttk.Entry(note, textvariable=self.single_duration, width=6)
        duration_entry.pack(side="left", padx=8)
        self.controls.append(duration_entry)
        ttk.Label(note, text="单独运行：其余五路为 0；停下后再切换电机或方向。", foreground="#64778d").pack(side="left", padx=12)
        headings = ("结构 / 通道", "方向", "速度设定", "", "", "反馈 rpm", "电流 raw", "最近下发", "状态")
        for col, label in enumerate(headings):
            ttk.Label(motors, text=label).grid(row=1, column=col, padx=8, pady=4)
            motors.columnconfigure(col, weight=1)
        for j in range(4):
            readouts = {key: tk.StringVar(value="—") for key in ("channel", "rpm", "current", "command", "state")}
            self.motor_readouts.append(readouts)
            row = j + 2
            ttk.Label(motors, textvariable=readouts["channel"], foreground=COLORS[j]).grid(row=row, column=0, padx=6)
            direction = tk.StringVar(value="反转 (−)" if defaults.signs[j] < 0 else "正转 (+)")
            self.motor_directions.append(direction)
            selector = ttk.Combobox(motors, textvariable=direction, values=tuple(DIRECTIONS), state="readonly", width=10)
            selector.grid(row=row, column=1, padx=6, pady=4)
            self.controls.append(selector)
            speed = tk.StringVar(value="300")
            self.motor_speeds.append(speed)
            entry = ttk.Spinbox(motors, from_=0, to=500, increment=10, textvariable=speed, width=8)
            entry.grid(row=row, column=2, padx=6)
            self.controls.append(entry)
            button = ttk.Button(motors, text="单独运行", style="Compact.TButton", command=lambda joint=j: self.start_single(joint))
            button.grid(row=row, column=3, padx=4, pady=2)
            self.controls.append(button)
            self.single_start_buttons.append(button)
            stop = ttk.Button(motors, text="停止本路", style="Compact.TButton", command=lambda joint=j: self.stop_single(joint), state="disabled")
            stop.grid(row=row, column=4, padx=4)
            self.single_stop_buttons.append(stop)
            for col, key in enumerate(("rpm", "current", "command", "state"), 5):
                ttk.Label(motors, textvariable=readouts[key]).grid(row=row, column=col, padx=8)
            for variable in (direction, speed):
                variable.trace_add("write", lambda *_: self.preview())
        footer = ttk.Frame(outer)
        footer.pack(side="bottom", fill="x")
        self.status = tk.StringVar(value="连接机器人后点击开始；无需相机、传动比或螺旋几何参数")
        ttk.Label(footer, textvariable=self.status, wraplength=1140).pack(fill="x", pady=3)
        self.transmission_status = tk.StringVar(value="尚未发送运动指令")
        ttk.Label(footer, textvariable=self.transmission_status, foreground=BLUE, wraplength=1140).pack(fill="x", pady=2)
        self.counts = tk.StringVar()
        ttk.Label(footer, textvariable=self.counts, foreground="#64778d", wraplength=1140).pack(anchor="w")
        self.display_tabs.pack(fill="both", expand=True, pady=(0, 8))
        plot_frame = ttk.Frame(self.display_tabs)
        self.display_tabs.add(plot_frame, text="转速曲线")
        self.plot = tk.Canvas(plot_frame, bg="white", highlightthickness=0, height=190)
        self.plot.pack(fill="both", expand=True, pady=(12, 8))
        diagnostics = ttk.Frame(self.display_tabs, padding=10)
        self.display_tabs.add(diagnostics, text="通信诊断")
        self.diagnostic_status = tk.StringVar(value="未连接")
        ttk.Label(diagnostics, textvariable=self.diagnostic_status, wraplength=1080).pack(fill="x", pady=(0, 8))
        ttk.Label(diagnostics, text="TCP 连接成功后仍需收到电机转速与电流。无数据时检查树莓派转发服务及 C 板数据链路。", foreground="#64778d").pack(anchor="w", pady=(0, 8))
        self.raw_feedback = tk.Text(diagnostics, height=8, wrap="none", font=("Consolas", 10), state="disabled")
        scrollbar = ttk.Scrollbar(diagnostics, orient="vertical", command=self.raw_feedback.yview)
        scrollbar.pack(side="right", fill="y")
        self.raw_feedback.configure(yscrollcommand=scrollbar.set)
        self.raw_feedback.pack(fill="both", expand=True)
        self.head_camera = CameraPanel(self.display_tabs, APP_DIR / "runs" / "head_camera",
                                       self.camera_robot_state)
        self.display_tabs.add(self.head_camera, text="头部相机")
        self.preview()

    def camera_robot_state(self):
        link = self.runner.link
        return {"host_snapshot_monotonic_ns": time.monotonic_ns(),
                "motors": link.snapshot() if link else {},
                "latest_control_interval": self.runner.latest.copy()}

    def get_config(self):
        config = ExperimentConfig()
        for key, var in self.vars.items():
            text = var.get().strip()
            if key in ("channels", "signs"):
                value = [int(v.strip()) for v in text.replace("，", ",").split(",")]
            elif key == "host":
                value = text
            elif key == "port":
                value = int(text)
            else:
                value = float(text)
            setattr(config, key, value)
        config.signs = [DIRECTIONS[var.get()] for var in self.motor_directions]
        config.residual_limit = 0.0
        config.validate()
        return config

    def get_joint_speeds(self):
        try:
            return [int(var.get().strip()) for var in self.motor_speeds]
        except ValueError as exc:
            raise ValueError("每路速度需填写非负整数，方向使用下拉框选择") from exc

    def preview(self):
        if not hasattr(self, "command_preview") or len(self.motor_speeds) != 4:
            return
        try:
            config = self.get_config()
            commands = baseline_commands(config, self.get_joint_speeds())
            self.command_preview.set("初始指令   " + "   ".join(f"M{i}={q:+d}" for i,q in enumerate(commands,1)))
            if hasattr(self, "preset_status"):
                if config.signs == [-1, 1, -1, 1]:
                    label = "当前：结构顺序 − / + / − / +"
                elif config.channels == [1, 2, 3, 4] and config.signs == [-1, 1, 1, 1]:
                    label = "当前：附件 M1–M4  − / + / + / +"
                else:
                    label = "当前：自定义通道 / 方向"
                self.preset_status.set(label)
        except (ValueError, TypeError, KeyError) as exc:
            self.command_preview.set(str(exc))
            if hasattr(self, "preset_status"):
                self.preset_status.set("参数待修正")

    def apply_forward_preset(self, reference=False):
        if self.runner.busy:
            return
        # Reference signs describe old physical channels, not arbitrary joints.
        if reference:
            self.vars["channels"].set("1,2,3,4")
        for variable, sign in zip(self.motor_directions, [-1, 1, 1, 1] if reference else [-1, 1, -1, 1]):
            variable.set("正转 (+)" if sign > 0 else "反转 (−)")
        self.preview()

    def apply_common_speed(self):
        if self.runner.busy or self.pending:
            return
        try:
            config = self.get_config()
            for variable in self.motor_speeds:
                variable.set(str(int(config.nominal_command)))
            self.preview()
        except (ValueError, TypeError) as exc:
            messagebox.showerror("共同速度", str(exc))

    def use_original_cadence(self):
        if self.runner.busy or self.pending:
            return
        self.vars["control_hz"].set("2")
        self.runner.message = "已选原上位机重发节奏：每 500 ms 发一次；尚未启动"

    def connect_robot(self):
        if self.pending or self.runner.busy or self.demo:
            return
        if self.runner.link and self.runner.link.connected:
            self.runner.link.close()
            self.runner.link = None
            return
        try:
            host, port = self.vars["host"].get().strip(), int(self.vars["port"].get())
            if not host or not 1 <= port <= 65535:
                raise ValueError("请检查 IP 与端口")
        except ValueError as exc:
            messagebox.showerror("连接", str(exc))
            return
        self.pending = True
        def worker():
            try:
                link = RobotLink(host, port)
                link.connect()
                self.events.put(("link", link))
            except Exception as exc:
                self.events.put(("error", str(exc)))
            finally:
                self.events.put(("pending", False))
        threading.Thread(target=worker, daemon=True).start()

    def start_trial(self):
        self._start_motion()

    def start_single(self, joint):
        self._start_motion(only_joint=joint)

    def _start_motion(self, only_joint=None):
        if self.pending or self.closing or self.runner.busy:
            return
        try:
            config = self.get_config()
            speeds = self.get_joint_speeds()
            if only_joint is not None:
                config.duration_s = float(self.single_duration.get())
                config.validate()
            if self.demo:
                self.runner.link.config = config
            self.runner.start_baseline(config, demo=self.demo, joint_speeds=speeds, only_joint=only_joint)
            self.active_joint = only_joint
        except Exception as exc:
            messagebox.showerror("无法开始", str(exc))

    def stop_single(self, joint):
        if self.runner.busy and self.active_joint == joint:
            self.stop_trial("单电机测试停止")

    def stop_trial(self, reason):
        self.runner.abort(reason)

    def save_config(self):
        if self.runner.busy:
            return
        try:
            config = self.get_config()
            speeds = self.get_joint_speeds()
            baseline_commands(config, speeds)
            data = config.as_dict()
            data["baseline_joint_speeds"] = speeds
            path = filedialog.asksaveasfilename(initialdir=APP_DIR, initialfile="initial_motor_policy.json", defaultextension=".json")
            if path:
                Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                self.runner.message = f"初始策略已保存：{Path(path).name}"
        except Exception as exc:
            messagebox.showerror("参数", str(exc))

    def load_config(self):
        if self.runner.busy:
            return
        path = filedialog.askopenfilename(initialdir=APP_DIR, filetypes=[("JSON", "*.json")])
        if path:
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
                speeds = data.pop("baseline_joint_speeds", None)
                config = ExperimentConfig(**data)
                baseline_commands(config, speeds)
                speeds = speeds if speeds is not None else [config.nominal_command] * 4
                for key,var in self.vars.items():
                    value = getattr(config,key)
                    var.set(",".join(map(str,value)) if isinstance(value,list) else str(value))
                for j in range(4):
                    self.motor_directions[j].set("正转 (+)" if config.signs[j] > 0 else "反转 (−)")
                    self.motor_speeds[j].set(str(int(speeds[j])))
                self.preview()
                self.runner.message = f"初始策略已读取：{Path(path).name}；尚未运行"
            except Exception as exc:
                messagebox.showerror("参数", str(exc))

    def tick(self):
        self.runner.ping()
        while not self.events.empty():
            kind,value = self.events.get_nowait()
            if kind == "link":
                self.runner.link = value
            elif kind == "pending":
                self.pending = value
            else:
                self.runner.message = value
        busy = self.runner.busy
        for widget in self.controls:
            idle_state = "readonly" if isinstance(widget, ttk.Combobox) else "normal"
            widget.configure(state="disabled" if busy or self.pending else idle_state)
        self.start_button.configure(state="disabled" if busy or self.pending else "normal")
        for j, button in enumerate(self.single_stop_buttons):
            button.configure(state="normal" if busy and self.active_joint == j else "disabled")
        self.connect_button.configure(state="disabled" if busy or self.pending or self.demo else "normal")
        link = self.runner.link
        self.connect_button.configure(text="断开机器人" if link and link.connected else "连接机器人")
        self.link_status.set("正在连接…" if self.pending else link.status if link else "未连接 · 以太网直连")
        motors, row = link.snapshot() if link else {}, self.runner.latest
        ready = [False] * 4
        try:
            config = self.get_config()
            baseline_commands(config, self.get_joint_speeds())
            now = time.monotonic()
            for j,c in enumerate(config.channels):
                m = motors.get(c,{})
                valid, feedback_reason = motor_feedback_state(motors, c, config.sensor_timeout_s, config.current_limit_raw, now)
                feedback_reason = feedback_reason.removeprefix(f"M{c} ")
                ready[j] = bool(link and link.connected and valid and not getattr(link, "lease_fault", False))
                live = valid or feedback_reason.startswith("电流超限")
                same_channel = row.get(f"j{j}_channel") == c
                values = {"channel": f"{j+1} / M{c}", "command": row.get(f"j{j}_command", "—") if same_channel else "—",
                          "rpm": f"{m['speed']:.1f}" if live and "speed" in m else "—",
                          "current": f"{m['current_raw']:.0f}" if live and "current_raw" in m else "—",
                          "state": feedback_reason.split("（")[0] if link and link.connected else "未连接"}
                for key, value in values.items():
                    self.motor_readouts[j][key].set(value)
                if link and getattr(link, "lease_fault", False):
                    self.motor_readouts[j]["state"].set("控制已锁定")
        except (ValueError, TypeError, KeyError):
            pass
        can_start = not busy and not self.pending and not self.closing
        self.start_button.configure(state="normal" if can_start and all(ready) else "disabled")
        for j, button in enumerate(self.single_start_buttons):
            button.configure(state="normal" if can_start and ready[j] else "disabled")
        if link and link.connected and not self.pending:
            if getattr(link, "lease_fault", False):
                self.link_status.set("控制发送超时锁定 · 请断开后重新连接")
            else:
                self.link_status.set(f"{'演示' if self.demo else 'TCP 已连接'} · 电机反馈就绪 {sum(ready)}/4")
        self.refresh_diagnostics(link)
        self.status.set(self.runner.message)
        mode = f"单路 {self.active_joint+1} 运行中" if busy and self.active_joint is not None else "四路运行中" if busy else "待机"
        self.counts.set(f"四路试验 {self.runner.completed.get('baseline_no_camera',0)} 条 · 单路调试 {self.runner.completed.get('single_motor',0)} 条 · {mode} · 最近下发是历史记录，停止状态以反馈为准")
        self.draw_plot()
        if not self.closing:
            self.after(100,self.tick)

    def refresh_diagnostics(self, link):
        transmission = "离线演示（无网络发送）" if self.demo else "尚未发送运动指令"
        if self.demo:
            status, raw = "离线演示：数据不来自真实机器人", "无实机通信"
        elif link is None:
            status, raw = "未连接机器人", "尚未接收数据"
        else:
            data = link.diagnostics()
            age = data["last_receive_age_s"]
            received = f"接收 {data['received_bytes']} 字节 / {data['received_lines']} 行"
            if data["connected"] and data["received_bytes"] == 0:
                status = f"TCP 已连接 {data['connected_for_s']:.1f} 秒，接收 0 字节；等待树莓派转发数据。电机尚未就绪。"
            elif data["connected"]:
                status = f"{received}；最近数据 {age:.1f} 秒前。电机就绪状态还需完整、及时的转速和电流反馈。"
            else:
                status = f"{link.status}；本次连接{received}"
            if data.get("lease_fault"):
                status = "控制租约超时已锁定；需要断开后重新连接。" + status
            def ms(value):
                return "—" if value is None else f"{1000*value:.1f}"
            transmission = (f"本连接运动指令 {data.get('sent_motion_count',0)} 帧 · "
                            f"最近间隔 {ms(data.get('last_motion_gap_s'))} ms · "
                            f"本段最大 {ms(data.get('max_motion_gap_s'))} ms · "
                            f"发送耗时 {ms(data.get('last_send_duration_s'))} ms · "
                            f"STOP {data.get('sent_stop_count',0)} 次（链路写入，不代表执行确认）")
            if data.get("last_transport_error"):
                fault = data["last_transport_error"]
                status += f" 最近链路错误：{fault['operation']} / {fault['type']}: {fault['message']}"
            raw = "\n".join(data["recent_lines"]) or "尚未收到完整数据行"
        self.diagnostic_status.set(status)
        self.transmission_status.set(transmission)
        now = time.monotonic()
        show_raw = self.display_tabs.index(self.display_tabs.select()) == 2
        if show_raw and raw != self.last_diagnostic_text and now - self.last_diagnostic_refresh >= 0.5:
            self.raw_feedback.configure(state="normal")
            self.raw_feedback.delete("1.0", "end")
            self.raw_feedback.insert("1.0", raw)
            self.raw_feedback.configure(state="disabled")
            self.raw_feedback.see("end")
            self.last_diagnostic_text = raw
            self.last_diagnostic_refresh = now

    def draw_plot(self):
        p = self.plot
        p.delete("all")
        width,height=max(300,p.winfo_width()),max(100,p.winfo_height())
        left,right,top,bottom=70,width-22,40,height-32
        p.create_text(20,18,text="四路实际转子转速 / rpm",anchor="w",fill=INK,font=("Microsoft YaHei UI",12,"bold"))
        for j,color in enumerate(COLORS):
            p.create_text(width-300+j*75,18,text=f"结构 {j+1}",fill=color)
        rows=self.runner.history
        limit=max([350]+[abs(row[f"j{j}_rpm"])*1.15 for row in rows for j in range(4) if row.get(f"j{j}_rpm") is not None])
        xmax=max([12]+[r["elapsed_s"] for r in rows])
        for value in (-limit,-limit/2,0,limit/2,limit):
            y=bottom-(value+limit)/(2*limit)*(bottom-top)
            p.create_line(left,y,right,y,fill="#e1e8f0")
            p.create_text(left-8,y,text=f"{value:.0f}",anchor="e",fill="#65768b")
        p.create_text(right,bottom+18,text=f"{xmax:.1f} s",anchor="e",fill="#65768b")
        for j,color in enumerate(COLORS):
            points=[]
            for row in rows:
                if row.get(f"j{j}_rpm") is None:
                    if len(points) >= 4:
                        p.create_line(*points,fill=color,width=2)
                    points=[]
                    continue
                points.extend((left+row["elapsed_s"]/xmax*(right-left),bottom-(row[f"j{j}_rpm"]+limit)/(2*limit)*(bottom-top)))
            if len(points)>=4:
                p.create_line(*points,fill=color,width=2)
        if not rows:
            p.create_text((left+right)/2,(top+bottom)/2,text="开始后记录四路转速、电流和下发指令",fill="#65768b")

    def check_focus(self):
        if self.runner.busy and self.focus_displayof() is None:
            self.stop_trial("窗口失去焦点")

    def report_callback_exception(self, exc, value, traceback):
        self.latest_gui_error=str(value)
        self.runner.abort("界面异常")
        super().report_callback_exception(exc,value,traceback)

    def on_close(self):
        if self.closing:
            return
        self.closing=True
        self.head_camera.close()
        self.stop_trial("关闭程序")
        def finish():
            if self.pending:
                self.tick()
                self.after(100,finish)
                return
            if self.runner.busy:
                self.after(100,finish)
                return
            if self.runner.link:
                self.runner.link.close()
            self.destroy()
        finish()


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--demo",action="store_true",help="离线模拟电机遥测，无实机通信")
    parser.add_argument("--head-camera", action="store_true", help="打开头部相机页面")
    args = parser.parse_args()
    app = Workbench(demo=args.demo)
    if args.head_camera:
        app.display_tabs.select(app.head_camera)
    app.mainloop()


if __name__ == "__main__":
    main()
