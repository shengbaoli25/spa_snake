"""Previous camera-based workbench retained for research compatibility.

The current launcher uses app.py: four-motor baseline without a camera.
"""
from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image, ImageTk

from .camera import CsvSpeedReceiver, RgbCameraSpeedSource
from .core import ExperimentConfig
from .ppo import PPOAgent
from .session import TrialRunner, DemoRobot
from .transport import RobotLink

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION = Path.home() / "Desktop" / "global_camera_calibration.json"
BG, PANEL, INK, MUTED, BLUE, GREEN = "#edf2f7", "#ffffff", "#162a43", "#65768b", "#176de5", "#0b9778"


class Workbench(tk.Tk):
    def __init__(self, demo=False):
        super().__init__()
        self.title("SPA · 四电机残差强化学习")
        self.geometry("1420x920")
        self.minsize(1120, 780)
        self.configure(bg=BG)
        self.runner = TrialRunner(APP_DIR / "runs")
        self.events = queue.Queue()
        self.demo = demo
        self.link_pending = False
        self.camera_pending = False
        self.roi_start = None
        self.photo = None
        self.frame_shape = None
        self.frame_rect = None
        self.roi_rectangle = None
        self.closing = False
        self.roi_dragging = False
        self.vars = {}
        self.config_widgets = []
        self.last_busy = False
        self.latest_gui_error = None
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=INK, font=("Microsoft YaHei UI", 10))
        style.configure("TButton", padding=(10, 7), font=("Microsoft YaHei UI", 10))
        style.configure("Accent.TButton", background=BLUE, foreground="white")
        style.configure("TNotebook.Tab", padding=(15, 9))
        self._build()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.bind("<Escape>", lambda _: self.stop_trial("键盘急停"))
        self.bind("<FocusOut>", lambda _: self.after(150, self.check_focus))
        self.after(100, self.tick)
        if demo:
            self.after(300, self.setup_demo)

    def label(self, parent, text, **kwargs):
        return tk.Label(parent, text=text, bg=kwargs.pop("bg", BG), fg=kwargs.pop("fg", INK), **kwargs)

    def _build(self):
        header = tk.Frame(self, bg=INK, height=74)
        header.pack(fill="x")
        self.label(header, "SPA  /  RESIDUAL CONTROL", bg=INK, fg="white", font=("Segoe UI", 20, "bold")).pack(side="left", padx=24, pady=18)
        self.label(header, "四电机 · 单地形验证", bg=INK, fg="#bbd1eb", font=("Microsoft YaHei UI", 11)).pack(side="left", padx=12)
        tk.Button(header, text="急停  ESC", command=lambda: self.stop_trial("急停按钮"), bg="#c63043", fg="white", font=("Microsoft YaHei UI", 13, "bold"), relief="flat", padx=28, pady=8).pack(side="right", padx=24)
        connection = ttk.Frame(self, padding=(18, 10))
        connection.pack(fill="x")
        self.vars["host"] = tk.StringVar(value="192.168.10.2")
        self.vars["port"] = tk.StringVar(value="9001")
        ttk.Label(connection, text="机器人直连").pack(side="left")
        ttk.Entry(connection, textvariable=self.vars["host"], width=17).pack(side="left", padx=6)
        ttk.Entry(connection, textvariable=self.vars["port"], width=6).pack(side="left")
        self.connect_button = ttk.Button(connection, text="连接机器人", command=self.connect_robot)
        self.connect_button.pack(side="left", padx=8)
        self.link_text = tk.StringVar(value="未连接 · 使用系统已配置以太网，不走 HTTP 代理")
        ttk.Label(connection, textvariable=self.link_text).pack(side="left", padx=5)
        self.demo_text = tk.StringVar(value="真机模式 · 每次只运行一条试验")
        ttk.Label(connection, textvariable=self.demo_text, foreground=BLUE).pack(side="right")
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=18, pady=(0, 8))
        self.dashboard = ttk.Frame(self.notebook, padding=12)
        self.settings = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(self.dashboard, text="试验控制与测速")
        self.notebook.add(self.settings, text="模型参数与通道")
        self._settings()
        self._dashboard()
        self.status = tk.StringVar(value="请先填写模型参数，连接机器人与全局相机")
        ttk.Label(self, textvariable=self.status, wraplength=1340, padding=(22, 8)).pack(fill="x")

    def _entry(self, parent, row, label, key, value, width=17):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=6, padx=(4, 14))
        self.vars[key] = tk.StringVar(value=str(value) if value is not None else "")
        entry = ttk.Entry(parent, textvariable=self.vars[key], width=width)
        entry.grid(row=row, column=1, sticky="ew", pady=6)
        self.config_widgets.append(entry)
        return entry

    def _settings(self):
        body = ttk.Frame(self.settings)
        body.pack(fill="x")
        body.columnconfigure((0, 1, 2), weight=1)
        motor = ttk.LabelFrame(body, text="01  结构顺序与基准控制", padding=16)
        motor.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self._entry(motor, 0, "四路通道（结构顺序）", "channels", "1,2,3,4")
        self._entry(motor, 1, "前进方向（±1）", "signs", "-1,1,-1,1")
        defaults = ExperimentConfig()
        fields = [("标称指令 q_nom", "nominal_command"), ("指令幅值上限", "command_limit"),
                  ("残差幅值上限（≤5）", "residual_limit"), ("轨迹时长 / s", "duration_s"),
                  ("控制频率 / Hz", "control_hz"), ("电流限值 / raw", "current_limit_raw"),
                  ("转速归一化 / rpm", "speed_scale_rpm")]
        for row, (label, key) in enumerate(fields, 2):
            self._entry(motor, row, label, key, getattr(defaults, key))
        ttk.Label(motor, text="方向沿用现有上位机前四路。\n通道可选 M1–M6 中任意四路；其余恒为 0。", wraplength=330, foreground=MUTED).grid(row=11, column=0, columnspan=2, sticky="w", pady=12)
        geometry = ttk.LabelFrame(body, text="02  螺旋运动学", padding=16)
        geometry.grid(row=0, column=1, sticky="nsew", padx=(0, 10))
        self._entry(geometry, 0, "外径 D / mm（旧方案值）", "diameter_mm", 80)
        self._entry(geometry, 1, "导程角 β / °（相对圆周）", "lead_angle_deg", 17)
        self._entry(geometry, 2, "实测导程 / mm（可选）", "lead_mm", None)
        self._entry(geometry, 3, "传动比 G（转子/螺旋）", "rotor_to_helix_ratio", None)
        self._entry(geometry, 4, "每指令单位的转子 rpm", "rpm_per_command", None)
        self._entry(geometry, 5, "或标定指令列表", "calibration_commands", "")
        self._entry(geometry, 6, "对应转子 rpm 列表", "calibration_rpm", "")
        ttk.Label(geometry, text="L = π D tan β\nv_theory = g_cal(q_nom) × L / (60 G)\n四个同构螺旋提供共同前进速度，不把速度相加。\n实测导程优先于 D/β，标定表优先于比例系数。", wraplength=350, foreground=MUTED).grid(row=9, column=0, columnspan=2, sticky="w", pady=16)
        reward = ttk.LabelFrame(body, text="03  目标与奖励", padding=16)
        reward.grid(row=0, column=2, sticky="nsew")
        self._entry(reward, 0, "目标速度 / m/s（空=理论）", "target_mps", None)
        self._entry(reward, 1, "奖励固定尺度 / m/s", "reward_scale_mps", .05)
        self._entry(reward, 2, "动作幅值惩罚（可选）", "action_penalty", 0)
        self._entry(reward, 3, "动作变化惩罚（可选）", "smoothness_penalty", 0)
        self._entry(reward, 4, "传感器超时 / s", "sensor_timeout_s", .8)
        ttk.Label(reward, text="原始奖励 = −|v_target − v_camera|\nPPO 奖励 = 原始奖励 / 固定尺度\n目标来自标称指令，与残差输出无关。\n初始确定性策略：Δq = [0, 0, 0, 0]。", foreground=MUTED, wraplength=340).grid(row=7, column=0, columnspan=2, sticky="w", pady=16)
        bar = ttk.Frame(self.settings)
        bar.pack(fill="x", pady=18)
        self.confirmed = tk.BooleanVar(value=False)
        check = ttk.Checkbutton(bar, text="已核对本次实机几何、传动比、指令/rpm 单位、通道方向与电流限值", variable=self.confirmed)
        check.pack(side="left")
        self.config_widgets.append(check)
        ttk.Button(bar, text="更新预览", command=self.preview).pack(side="right", padx=4)
        ttk.Button(bar, text="保存参数", command=self.save_config).pack(side="right", padx=4)
        ttk.Button(bar, text="读取参数", command=self.load_config).pack(side="right", padx=4)
        self.formula = tk.StringVar(value="填写 G 与指令→rpm 标定后可计算理论速度")
        ttk.Label(self.settings, textvariable=self.formula, font=("Microsoft YaHei UI", 12), foreground=BLUE, wraplength=1240).pack(anchor="w", pady=8)
        ttk.Label(self.settings, text="现有标定仅适用于相同相机位置、焦距、分辨率与轨道。更改安装后，应先在原测速软件重新标定并保存 JSON。", foreground=MUTED, wraplength=1240).pack(anchor="w", pady=10)

    def _dashboard(self):
        self.dashboard.columnconfigure(0, weight=3)
        self.dashboard.columnconfigure(1, weight=2)
        self.dashboard.rowconfigure(2, weight=1)
        actions = ttk.Frame(self.dashboard)
        actions.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        self.trial_buttons = []
        for label, mode in [("标准前进 · 基线", "baseline"), ("残差 PPO · 训练", "train"), ("冻结策略 · 测试", "eval")]:
            button = ttk.Button(actions, text=label, command=lambda m=mode: self.start_trial(m), style="Accent.TButton")
            button.pack(side="left", padx=(0, 10))
            self.trial_buttons.append(button)
        ttk.Button(actions, text="载入策略", command=self.load_policy).pack(side="left", padx=4)
        ttk.Button(actions, text="重置策略", command=self.reset_policy).pack(side="left", padx=4)
        self.counts = tk.StringVar(value="基线 0/50    训练 0/100    测试 0/30")
        ttk.Label(actions, textvariable=self.counts, foreground=GREEN).pack(side="right")
        self.meter_vars = {}
        meters = tk.Frame(self.dashboard, bg=BG)
        meters.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for i, (key, label, color) in enumerate([( "target", "目标速度 / m·s⁻¹", BLUE), ("theory", "理论无滑移速度 / m·s⁻¹", INK), ("actual", "全局相机实测 / m·s⁻¹", GREEN), ("error", "速度误差 / m·s⁻¹", "#ba5b1e"), ("reward", "原始奖励", INK)]):
            card = tk.Frame(meters, bg=PANEL, padx=16, pady=12)
            card.grid(row=0, column=i, sticky="nsew", padx=(0, 8))
            meters.columnconfigure(i, weight=1)
            self.label(card, label, bg=PANEL, fg=MUTED, font=("Microsoft YaHei UI", 9)).pack(anchor="w")
            var = tk.StringVar(value="—")
            self.meter_vars[key] = var
            tk.Label(card, textvariable=var, bg=PANEL, fg=color, font=("Consolas", 23, "bold")).pack(anchor="w", pady=(5, 0))
        left = ttk.Frame(self.dashboard)
        left.grid(row=2, column=0, sticky="nsew", padx=(0, 12))
        self.plot = tk.Canvas(left, bg=PANEL, highlightthickness=0, height=250)
        self.plot.pack(fill="both", expand=True)
        self.table = ttk.Treeview(left, columns=("channel", "sign", "nom", "res", "cmd", "rpm", "current"), show="headings", height=4)
        for key, title in zip(self.table["columns"], ("结构/通道", "方向", "标称", "残差", "下发", "转子rpm", "电流raw")):
            self.table.heading(key, text=title)
            self.table.column(key, width=85, anchor="center")
        for j in range(4):
            self.table.insert("", "end", iid=str(j), values=(f"{j+1} / M{j+1}", "−" if j % 2 == 0 else "+", 40, 0, 0, "—", "—"))
        self.table.pack(fill="x", pady=(12, 0))
        self.trial_info = tk.StringVar(value="每条轨迹自动记录参数、动作、相机速度、误差、奖励和策略版本")
        ttk.Label(left, textvariable=self.trial_info, wraplength=670, foreground=MUTED).pack(fill="x", pady=8)
        right = ttk.Frame(self.dashboard)
        right.grid(row=2, column=1, sticky="nsew")
        options = ttk.LabelFrame(right, text="全局相机测速", padding=10)
        options.pack(fill="x")
        self.camera_mode = tk.StringVar(value="直接 RGB 相机")
        ttk.Combobox(options, textvariable=self.camera_mode, values=("直接 RGB 相机", "读取原测速实时 CSV"), state="readonly", width=23).grid(row=0, column=0, sticky="ew", pady=4)
        self.camera_button = ttk.Button(options, text="连接测速", command=self.connect_camera)
        self.camera_button.grid(row=0, column=1, padx=6)
        self.camera_index = tk.StringVar(value="0")
        self.camera_direction = tk.StringVar(value="left")
        details = ttk.Frame(options)
        details.grid(row=1, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Label(details, text="相机编号").pack(side="left")
        ttk.Entry(details, textvariable=self.camera_index, width=4).pack(side="left", padx=4)
        ttk.Label(details, text="头部前沿").pack(side="left", padx=(10, 4))
        ttk.Combobox(details, textvariable=self.camera_direction, values=("left", "right"), state="readonly", width=6).pack(side="left")
        self.camera_path = tk.StringVar(value=str(DEFAULT_CALIBRATION))
        ttk.Entry(options, textvariable=self.camera_path, width=42).grid(row=2, column=0, sticky="ew", pady=4)
        ttk.Button(options, text="选择文件", command=self.pick_camera_file).grid(row=2, column=1, padx=6)
        options.columnconfigure(0, weight=1)
        self.camera_status = tk.StringVar(value="直接相机：选择标定 JSON 后连接，在画面拖框选头部。")
        ttk.Label(right, textvariable=self.camera_status, wraplength=500, foreground=MUTED).pack(fill="x", pady=8)
        self.video = tk.Canvas(right, bg="#192c43", highlightthickness=0, height=265)
        self.video.pack(fill="both", expand=True)
        self.video.create_text(230, 100, text="全局相机画面\n连接后拖动框选头部", fill="#bccfe4", font=("Microsoft YaHei UI", 12), justify="center")
        self.video.bind("<ButtonPress-1>", self.roi_press)
        self.video.bind("<B1-Motion>", self.roi_move)
        self.video.bind("<ButtonRelease-1>", self.roi_release)
        ttk.Label(right, text="沿用原测速算法：轨道投影 → 位置低通 → 0.30 s 尾窗拟合。", foreground=MUTED, wraplength=500).pack(fill="x", pady=8)

    def get_config(self):
        config = ExperimentConfig()
        for key, var in self.vars.items():
            value = var.get().strip()
            if key in ("channels", "signs"):
                parsed = [int(x.strip()) for x in value.replace("，", ",").split(",")]
            elif key in ("calibration_commands", "calibration_rpm"):
                parsed = [float(x.strip()) for x in value.replace("，", ",").split(",")] if value else []
            elif key == "host":
                parsed = value
            elif key == "port":
                parsed = int(value)
            else:
                parsed = float(value) if value else None
            setattr(config, key, parsed)
        config.parameters_confirmed = self.confirmed.get()
        config.validate()
        return config

    def use_config(self, config):
        for key, var in self.vars.items():
            value = getattr(config, key)
            var.set(",".join(map(str, value)) if isinstance(value, list) else "" if value is None else str(value))
        self.confirmed.set(config.parameters_confirmed)

    def preview(self):
        try:
            config = self.get_config()
            theory = config.theory_speed(config.nominal_command)
            commands, _ = config.project(np.zeros(4))
            self.formula.set(f"导程 {config.lead_m()*1000:.3f} mm/rev  ·  转子 {config.motor_rpm(config.nominal_command):.2f} rpm  ·  理论速度 {theory:.5f} m/s  ·  M1…M6 = {commands}")
            self.meter_vars["target"].set(f"{config.target_speed():.4f}")
            self.meter_vars["theory"].set(f"{theory:.4f}")
        except Exception as exc:
            self.formula.set(str(exc))

    def save_config(self):
        try:
            config = self.get_config()
            path = filedialog.asksaveasfilename(initialdir=APP_DIR, defaultextension=".json", initialfile="robot_parameters.json")
            if path:
                Path(path).write_text(json.dumps(config.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            messagebox.showerror("参数", str(exc))

    def load_config(self):
        if self.runner.busy:
            return
        path = filedialog.askopenfilename(initialdir=APP_DIR, filetypes=[("JSON", "*.json")])
        if path:
            try:
                config = ExperimentConfig(**json.loads(Path(path).read_text(encoding="utf-8-sig")))
                config.validate()
                self.use_config(config)
                self.confirmed.set(False)
                self.preview()
            except Exception as exc:
                messagebox.showerror("参数", str(exc))

    def connect_robot(self):
        if self.runner.busy or self.link_pending or self.demo:
            return
        if self.runner.link and self.runner.link.connected:
            self.runner.link.close()
            self.runner.link = None
            return
        try:
            host, port = self.vars["host"].get().strip(), int(self.vars["port"].get())
            if not host or not 1 <= port <= 65535:
                raise ValueError("主机/端口无效")
        except Exception as exc:
            messagebox.showerror("连接", str(exc))
            return
        self.link_pending = True
        self.link_text.set("正在直连…")
        def worker():
            try:
                link = RobotLink(host, port)
                link.connect()
                self.events.put(("link", link))
            except Exception as exc:
                self.events.put(("error", str(exc)))
            finally:
                self.events.put(("link_pending", False))
        threading.Thread(target=worker, daemon=True).start()

    def pick_camera_file(self):
        if self.runner.busy:
            return
        csv_mode = "CSV" in self.camera_mode.get()
        path = filedialog.askopenfilename(initialdir=Path.home()/"Desktop", filetypes=[("实时 CSV" if csv_mode else "标定 JSON", "*.csv" if csv_mode else "*.json")])
        if path:
            self.camera_path.set(path)

    def connect_camera(self):
        if self.runner.busy or self.camera_pending or self.demo:
            return
        if self.runner.camera:
            self.runner.camera.stop()
            self.runner.camera = None
            self.camera_status.set("测速已断开")
            self.meter_vars["actual"].set("—")
            self.frame_rect = self.frame_shape = None
            self.photo = None
            self.video.delete("all")
            self.video.create_text(220,100,text="测速已断开",fill="#bccfe4",font=("Microsoft YaHei UI",12))
            return
        try:
            if "CSV" in self.camera_mode.get():
                source = CsvSpeedReceiver(self.camera_path.get())
            else:
                source = RgbCameraSpeedSource(self.camera_path.get(), camera_index=int(self.camera_index.get()), direction=self.camera_direction.get())
            source.start()
            self.runner.camera = source
        except Exception as exc:
            messagebox.showerror("全局测速", str(exc))

    def start_trial(self, mode):
        try:
            config = self.get_config()
            self.runner.start(config, mode, demo=self.demo)
            self.preview()
        except Exception as exc:
            messagebox.showerror("尚不能开始试验", str(exc))

    def stop_trial(self, reason):
        self.runner.abort(reason)

    def reset_policy(self):
        try:
            self.runner.reset_policy()
        except Exception as exc:
            messagebox.showerror("策略", str(exc))

    def load_policy(self):
        if self.runner.busy:
            return
        path = filedialog.askopenfilename(initialdir=APP_DIR/"runs", filetypes=[("策略", "*.npz")])
        if path:
            try:
                signature = Path(path).with_name("policy_config.json")
                if not signature.exists():
                    raise ValueError("此策略缺少同目录 policy_config.json，无法核对通道/量程")
                saved = json.loads(signature.read_text(encoding="utf-8"))
                provenance_path = Path(path).with_name("metadata.json")
                if not provenance_path.exists():
                    raise ValueError("策略缺少来源 metadata.json")
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                if bool(provenance.get("demo")) != self.demo:
                    raise ValueError("离线演示策略与真机策略不能混用")
                if saved != self.runner.policy_signature(self.get_config()):
                    raise ValueError("当前参数与策略通道/量程不一致")
                agent = PPOAgent.load(path)
                if agent.obs_dim != 17 or agent.action_dim != 4:
                    raise ValueError("策略输入输出维度与本程序不匹配")
                self.runner.agent = agent
                self.runner.policy_config = saved
                self.runner.message = f"已载入策略 v{agent.policy_version}"
            except Exception as exc:
                messagebox.showerror("策略", str(exc))

    def setup_demo(self):
        config = ExperimentConfig(signs=[-1,1,-1,1], rotor_to_helix_ratio=19, rpm_per_command=20, parameters_confirmed=True)
        self.use_config(config)
        robot = DemoRobot(config)
        self.runner.link = robot
        self.runner.camera = robot
        self.demo_text.set("离线演示 · 所有数据为模拟 · 无实机连接")
        self.runner.message = "离线演示就绪，可检查三种试验流程；示例参数不能用于实机"
        self.preview()

    def roi_press(self, event):
        if self.runner.busy or not isinstance(self.runner.camera, RgbCameraSpeedSource) or not self.frame_rect:
            return
        self.roi_start = (event.x, event.y)
        self.roi_dragging = True
        self.video.delete("selection")
        self.roi_rectangle = self.video.create_rectangle(event.x, event.y, event.x, event.y, outline="#ffce56", width=2, tags="selection")

    def roi_move(self, event):
        if self.roi_start:
            self.video.coords(self.roi_rectangle, *self.roi_start, event.x, event.y)

    def roi_release(self, event):
        if self.roi_start and self.frame_rect and self.frame_shape:
            x0, y0, scale = self.frame_rect
            height, width = self.frame_shape[:2]
            xa, ya = self.roi_start
            x, y = max(0, (min(xa, event.x)-x0)/scale), max(0, (min(ya, event.y)-y0)/scale)
            right, bottom = min(width, (max(xa, event.x)-x0)/scale), min(height, (max(ya, event.y)-y0)/scale)
            try:
                if right-x >= 8 and bottom-y >= 8:
                    self.runner.camera.set_roi((x,y,right-x,bottom-y))
            except Exception as exc:
                self.runner.message = str(exc)
        self.roi_start = None
        self.roi_dragging = False
        self.video.delete("selection")

    def render_camera(self):
        source = self.runner.camera
        if not isinstance(source, RgbCameraSpeedSource) or self.roi_dragging:
            return
        frame = source.latest_frame()
        if frame is None:
            return
        self.frame_shape = frame.shape
        height, width = frame.shape[:2]
        cw, ch = max(1,self.video.winfo_width()), max(1,self.video.winfo_height())
        scale = min(cw/width, ch/height)
        w, h = max(1,int(width*scale)), max(1,int(height*scale))
        x, y = (cw-w)/2, (ch-h)/2
        self.frame_rect = (x,y,scale)
        image = Image.fromarray(frame[:,:,::-1]).resize((w,h))
        self.photo = ImageTk.PhotoImage(image)
        self.video.delete("all")
        self.video.create_image(x,y,anchor="nw",image=self.photo)
        sample = source.latest()
        if sample and sample.bbox:
            bx,by,bw,bh = sample.bbox
            self.video.create_rectangle(x+bx*scale,y+by*scale,x+(bx+bw)*scale,y+(by+bh)*scale,outline="#65ebbd",width=2)

    def draw_plot(self):
        canvas = self.plot
        canvas.delete("all")
        w,h = max(200,canvas.winfo_width()), max(180,canvas.winfo_height())
        left, top, right, bottom = 64, 46, w-22, h-35
        canvas.create_text(18,18,text="速度跟踪",anchor="w",fill=INK,font=("Microsoft YaHei UI",12,"bold"))
        canvas.create_text(w-18,18,text="目标  蓝色      实测  绿色",anchor="e",fill=MUTED,font=("Microsoft YaHei UI",9))
        data = self.runner.history
        values = [r[k] for r in data for k in ("target_mps", "actual_mps")]
        low, high = min([0]+values), max([.05]+values)*1.15
        if high-low < 1e-6:
            high = low+.05
        xmax = max([12]+[r["elapsed_s"] for r in data])
        for i in range(5):
            v = low+(high-low)*i/4
            y = bottom-(bottom-top)*i/4
            canvas.create_line(left,y,right,y,fill="#e4eaf1")
            canvas.create_text(left-8,y,text=f"{v:.3f}",anchor="e",fill=MUTED,font=("Consolas",9))
        canvas.create_text(right,bottom+20,text=f"{xmax:.1f} s",anchor="e",fill=MUTED)
        canvas.create_text(left,bottom+20,text="0",anchor="w",fill=MUTED)
        for key,color in (("target_mps",BLUE),("actual_mps",GREEN)):
            points=[]
            for row in data:
                points.extend((left+row["elapsed_s"]/xmax*(right-left),bottom-(row[key]-low)/(high-low)*(bottom-top)))
            if len(points)>=4:
                canvas.create_line(*points,fill=color,width=2)
        if not data:
            canvas.create_text((left+right)/2,(top+bottom)/2,text="开始试验后显示真实速度对比",fill=MUTED,font=("Microsoft YaHei UI",11))

    def tick(self):
        self.runner.ping()
        while not self.events.empty():
            kind, value = self.events.get_nowait()
            if kind == "link":
                self.runner.link = value
            elif kind == "link_pending":
                self.link_pending = value
            elif kind == "error":
                self.runner.message = value
        busy = self.runner.busy
        if busy != self.last_busy:
            for widget in self.config_widgets + self.trial_buttons:
                widget.configure(state="disabled" if busy else "normal")
            self.last_busy = busy
        link = self.runner.link
        if not self.link_pending:
            self.link_text.set(link.status if link else "未连接 · 以太网直连")
            self.connect_button.configure(text="断开机器人" if link and link.connected else "连接机器人")
        camera = self.runner.camera
        self.camera_button.configure(text="断开测速" if camera else "连接测速")
        if camera:
            self.camera_status.set(camera.status)
            sample = camera.latest()
            self.meter_vars["actual"].set(f"{sample.velocity_mps:.4f}" if sample and sample.tracking_ok and sample.velocity_mps is not None else "无有效帧")
        row = self.runner.latest
        if row:
            for key, field in (("target","target_mps"),("theory","theory_mps"),("error","error_mps"),("reward","reward_raw")):
                self.meter_vars[key].set(f"{row[field]:.4f}")
        motors = link.snapshot() if link else {}
        try:
            config = self.get_config()
            for j,(channel,sign) in enumerate(zip(config.channels,config.signs)):
                motor = motors.get(channel,{})
                fresh_row = row if row and busy else {}
                self.table.item(str(j), values=(f"{j+1} / M{channel}","+" if sign>0 else "−",config.nominal_command,
                                               fresh_row.get(f"j{j}_residual_applied",0),fresh_row.get(f"j{j}_command",0),
                                               motor.get("speed","—"),motor.get("current_raw","—")))
        except (ValueError,TypeError):
            pass
        count=self.runner.completed
        self.counts.set(f"本次会话：基线 {count['baseline']}/50    训练 {count['train']}/100    测试 {count['eval']}/30")
        self.trial_info.set(f"策略 v{self.runner.agent.policy_version}  ·  {len(self.runner.history)} 步  ·  {'运行中' if busy else '待机'}")
        self.status.set(self.runner.message)
        self.draw_plot()
        self.render_camera()
        self.after(100,self.tick)

    def check_focus(self):
        if self.runner.busy and self.focus_displayof() is None:
            self.stop_trial("窗口失去焦点")

    def report_callback_exception(self, exc, value, traceback):
        self.latest_gui_error = str(value)
        self.runner.abort("界面回调异常")
        self.runner.message = f"界面异常，已请求停止: {value}"
        super().report_callback_exception(exc, value, traceback)

    def on_close(self):
        if self.closing:
            return
        self.closing=True
        self.stop_trial("关闭程序")
        def finish():
            if self.runner.busy:
                self.after(100,finish)
                return
            if self.runner.camera and self.runner.camera is not self.runner.link:
                self.runner.camera.stop()
            if self.runner.link:
                self.runner.link.close()
            self.destroy()
        finish()


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--demo",action="store_true",help="仅离线演示，不连接机器人/相机")
    args=parser.parse_args()
    Workbench(demo=args.demo).mainloop()


if __name__=="__main__":
    main()
