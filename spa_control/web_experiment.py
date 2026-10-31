"""Bounded two-section commissioning and real SPA learning for the web workbench.

Only an explicit start() may acquire motor ownership. The independent monitor
collects measured data and renews a bounded command lease; inference never owns
the serial port. Global-camera measurements are labels/rewards, never model
inputs. Saved supervised records cannot enter the fresh PPO buffer.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import random
import threading
import time
import uuid

from .serial_preview import (ACTIONS, CategoricalPPO, RewardWeights, Transition,
                             failure_tail, measured_load_costs, measured_reward)


class ExperimentError(RuntimeError):
    pass


DEFAULTS = dict(front_motor=None, base_rpm=60, direction_m3=-1, direction_m4=-1,
                forward_sign=1, distance_m=.3, duration_s=12., hold_s=.5,
                lead_mm=30., motor_to_helix_ratio=1., target_speed_mps=.03,
                preparation_group="development-1")
WEIGHTS = RewardWeights(current=.2, occupancy=.1, action=.02, smoothness=.02,
                        stop=1., sigma=.5, min_speed=.005)
QUALITY_FIELDS = ("front_feedback_age", "rear_feedback_age", "head_age", "head_valid",
                  "lead_m", "ratio", "previous_residual", "nominal", "target_speed",
                  "hold_s", "remaining_fraction", "front_direction", "rear_direction",
                  "front_feedback_valid", "rear_feedback_valid")


def validate_config(data, previous=None):
    config = dict(DEFAULTS if previous is None else previous)
    aliases = {"m3_sign": "direction_m3", "m4_sign": "direction_m4"}
    for key, value in data.items():
        key = aliases.get(key, key)
        if key in config:
            config[key] = value
    if config["front_motor"] is not None and (type(config["front_motor"]) is not int or config["front_motor"] not in (3, 4)):
        raise ExperimentError("前节必须选择 M3 或 M4")
    for name in ("direction_m3", "direction_m4", "forward_sign"):
        if type(config[name]) is not int or config[name] not in (-1, 1):
            raise ExperimentError(f"{name} 必须为 -1 或 +1")
    for name, low, high in (("base_rpm", 1, 500), ("duration_s", 2, 60),
                            ("hold_s", .25, 2), ("distance_m", .001, 10),
                            ("lead_mm", .001, 1000), ("motor_to_helix_ratio", .001, 1000)):
        try:
            value = float(config[name])
        except (TypeError, ValueError):
            raise ExperimentError(f"{name} 必须是有效数值")
        if isinstance(config[name], bool) or not math.isfinite(value) or not low <= value <= high:
            raise ExperimentError(f"{name} 必须在 {low} 到 {high} 之间")
        config[name] = value
    if config["base_rpm"] != round(config["base_rpm"]):
        raise ExperimentError("电机转速使用整数 rpm")
    config["base_rpm"] = int(config["base_rpm"])
    if "target_speed_mps" not in data:
        config["target_speed_mps"] = config["forward_sign"] * config["base_rpm"] * config["lead_mm"] / 60000 / config["motor_to_helix_ratio"]
    try:
        target = float(config["target_speed_mps"])
    except (TypeError, ValueError):
        raise ExperimentError("目标速度无效")
    if isinstance(config["target_speed_mps"],bool) or not math.isfinite(target) or not .00001 <= abs(target) <= 1:
        raise ExperimentError("目标速度绝对值须在 0.00001 到 1 m/s")
    theoretical=config["forward_sign"]*config["base_rpm"]*config["lead_mm"]/60000/config["motor_to_helix_ratio"]
    if not math.isclose(target,theoretical,rel_tol=1e-6,abs_tol=1e-7):
        raise ExperimentError("目标速度与转速、导程和传动比不一致；30 mm/转、1:1 时 100 rpm = 0.05 m/s")
    config["target_speed_mps"] = target
    group = str(config["preparation_group"]).strip()
    if not group or len(group) > 100:
        raise ExperimentError("请填写本次独立场地准备的组名（最多 100 字）")
    config["preparation_group"] = group
    return config


def project(config):
    """Structural front/rear projection, retaining true four-channel slots."""
    import numpy as np
    front, rear = config["front_motor"], 7 - config["front_motor"]
    q = config["base_rpm"]
    vectors, features, residuals = [], [], []
    for action in ACTIONS:
        rear_q = max(0, min(500, q + action))
        vector = [0, 0, 0, 0]
        vector[front-1] = config[f"direction_m{front}"] * q
        vector[rear-1] = config[f"direction_m{rear}"] * rear_q
        vectors.append(vector)
        features.append([vector[front-1]/500., vector[rear-1]/500., 0.,
                         config[f"direction_m{rear}"]*(rear_q-q)/5.])
        residuals.append(rear_q-q)
    support = [vector not in vectors[:i] for i, vector in enumerate(vectors)]
    return vectors, np.asarray(features), support, residuals


def interval_metrics(drives, cameras, start, end, config, residual, previous,
                     stopped=False, weights=WEIGHTS):
    """Measured 0.1 s causal as-of grid; held values are never new packets."""
    import numpy as np
    if not start < end:
        raise ExperimentError("奖励区间为空")
    grid = np.arange(start+.1, end+1e-7, .1).tolist()
    if not grid:
        grid = [end]
    selected = {ch: sorted((p for p in drives if p["channel"] == ch), key=lambda p:p["time"])
                for ch in (3, 4)}
    cams = sorted(cameras, key=lambda p:p["time"])
    def asof(values, at, maximum_age):
        accepted = [p for p in values if p["time"] <= at]
        if not accepted or at-accepted[-1]["time"] > maximum_age:
            return None
        return accepted[-1]
    rows, velocity = [], []
    for stamp in grid:
        pair = [asof(selected[ch], stamp, .25) for ch in (3, 4)]
        camera = asof(cams, stamp, .25)
        if any(p is None for p in pair) or camera is None:
            raise ExperimentError("奖励区间缺少新鲜电流或相机测速；整轮禁止 PPO 更新")
        rows.append([p["current"] for p in pair])
        velocity.append(camera["velocity"])
    begin, finish = asof(cams, start, .25), asof(cams, end, .25)
    if begin is None or finish is None or finish["time"] <= begin["time"]:
        raise ExperimentError("位移标签缺少有效区间端点")
    ci, occupancy, coverage = measured_load_costs(np.asarray(rows), np.ones((len(rows),2),bool), [7000,7000], .7)
    rear = 7-config["front_motor"]
    original = [p for p in selected[rear] if start < p["time"] <= end]
    if not original:
        raise ExperimentError("区间没有后节原始反馈包")
    rear_column = rear-3
    labels = [finish["position"]-begin["position"], max(abs(p["current"])/7000 for p in original),
              float(np.mean(np.abs(np.asarray(rows)[:,rear_column])/7000 > .7)), float(stopped)]
    mean_velocity = float(np.mean(velocity))
    reward = measured_reward(mean_velocity,config["target_speed_mps"],ci,occupancy,
                             [residual],[previous],stopped,weights)
    return dict(reward=reward,velocity_mps=mean_velocity,target_speed_mps=config["target_speed_mps"],
                current_cost=ci,occupancy=occupancy,grid_coverage=coverage,grid_points=len(grid),
                packet_counts={str(ch):sum(start<p["time"]<=end for p in selected[ch]) for ch in (3,4)},
                camera_packets=sum(start<p["time"]<=end for p in cams),labels=labels,
                action_cost=(residual/5)**2,smoothness_cost=((residual-previous)/10)**2,
                stop=bool(stopped),start_monotonic_s=start,end_monotonic_s=end)


class _FeatureCache:
    def __init__(self, encoder):
        self.encoder, self.cache = encoder, {}

    def encode_window(self, frames, cutoff_ns, max_frames=8):
        selected = sorted((f for f in frames if f["receive_monotonic_ns"] <= cutoff_ns),
                          key=lambda f:f["receive_monotonic_ns"])[-max_frames:]
        missing = [f for f in selected if f["receive_monotonic_ns"] not in self.cache]
        if missing:
            for result in self.encoder.encode_window(missing,cutoff_ns,max_frames)["frames"]:
                self.cache[result["receive_monotonic_ns"]] = result
        self.cache = {stamp:item for stamp,item in self.cache.items() if cutoff_ns-stamp < 10_000_000_000}
        return {"frames":[self.cache[f["receive_monotonic_ns"]] for f in selected],"future_frames_excluded":0}


class WebExperiment:
    def __init__(self,motor,camera,global_camera,runs_root=None):
        self.motor,self.camera,self.global_camera=motor,camera,global_camera
        self.root=Path(runs_root) if runs_root else Path(__file__).parent/"runs"/"paper_two_motor"
        self._lock=threading.RLock()
        self._cancel=threading.Event()
        self._monitor_stop=threading.Event()
        self._thread=None
        self._config=dict(DEFAULTS)
        self._state="idle"
        self._message="先单测 M3/M4，选择前节及方向；30 mm/转、1:1 已确认"
        self._model=self._policy=self._pipeline=None
        self._pretrained=False
        self._training_signature=None
        self._run=None
        self._reward=None
        self._progress=0.
        self._decisions=0
        self._last_update=None
        self._error=None
        self._targets={3:0,4:0}
        self._drives=deque(maxlen=20000)
        self._cameras=deque(maxlen=10000)
        self._frames=deque(maxlen=64)
        self._events_lock=threading.Lock()
        self._catalog=[]
        self._collection_fault=False
        self._mode=None
        self._origin=None
        self._checkpoint_available=False
        self._checkpoint_version=0
        self._decision_deadline=None
        settings=self.root/"settings.json"
        if settings.is_file():
            try:self._config=validate_config(json.loads(settings.read_text(encoding="utf-8")))
            except (ValueError,OSError,ExperimentError):pass
        checkpoint_meta=self.root/"checkpoint.json"
        if checkpoint_meta.is_file() and (self.root/"latest.pt").is_file():
            try:
                metadata=json.loads(checkpoint_meta.read_text(encoding="utf-8"))
                self._training_signature=metadata["signature"]
                self._checkpoint_available=bool(metadata.get("pretrained"))
                self._checkpoint_version=int(metadata.get("policy_version",0))
                self._pretrained=self._checkpoint_available and self._training_signature==self._signature()
            except (KeyError,ValueError,OSError):pass
        self._load_catalog()

    def _load_catalog(self):
        self._catalog=[]
        if not self.root.exists(): return
        for path in sorted(self.root.glob("*/manifest.json")):
            try:
                record=json.loads(path.read_text(encoding="utf-8"))
                if record.get("format")==1 and record.get("supervised_samples",0)>0:
                    self._catalog.append((path.parent,record))
            except (OSError,ValueError): pass

    def _signature(self,config=None):
        c=self._config if config is None else config
        keys=("front_motor","direction_m3","direction_m4","forward_sign","base_rpm","target_speed_mps",
              "lead_mm","motor_to_helix_ratio","hold_s","duration_s")
        return {k:c[k] for k in keys}

    def configure(self,data):
        with self._lock:
            if self._thread and self._thread.is_alive(): raise ExperimentError("实验运行中不能修改配置")
            self._config=validate_config(data,self._config)
            self._pretrained=(self._checkpoint_available or self._pretrained) and self._training_signature==self._signature()
            self.root.mkdir(parents=True,exist_ok=True)
            (self.root/"settings.json").write_text(json.dumps(self._config,ensure_ascii=False,indent=2),encoding="utf-8")
        return self.status()

    def _blockers(self,mode):
        result=[]
        c=self._config
        if c["front_motor"] not in (3,4): result.append("请选择实物前节 M3 或 M4")
        m=self.motor.status()
        if not m.get("frame_fresh") or m.get("fault")!="NONE" or not m.get("can_healthy"):
            result.append("C 板需要新鲜、无故障的两路 CAN 反馈")
        if any(not (m.get("motors") or {}).get(str(ch),{}).get("fresh") for ch in (3,4)):
            result.append("M3/M4 反馈不完整")
        if not self.global_camera.status().get("valid"):
            result.append("全局相机需要连接、标定并跟踪机器人")
        if mode in ("collect","train"):
            head=self.camera.status()
            if not head.get("active") or head.get("frame_age_s") is None or head["frame_age_s"]>.8:
                result.append("SPA-CAM 头部图像缺失或过期")
        if mode=="train" and not self._pretrained:
            result.append("请先分组采集真实数据并完成两阶段预训练")
        return result

    def status(self):
        with self._lock:
            groups={}
            for _,record in self._catalog:
                group=record["config"]["preparation_group"]
                groups[group]=groups.get(group,0)+record["supervised_samples"]
            baseline=self._blockers("baseline")
            collect=self._blockers("collect")
            train=self._blockers("train")
            active=bool(self._thread and self._thread.is_alive())
            matching_groups={r["config"]["preparation_group"] for _,r in self._catalog if r.get("signature")==self._signature()}
            return dict(state=self._state,mode=self._mode,active=active,message=self._message,config=dict(self._config),
                        ready_baseline=not baseline and not active,ready_collect=not collect and not active,
                        ready_train=not train and not active,ready_pretrain=len(matching_groups)>=2 and not active,
                        blockers=train,blockers_by_mode={"baseline":baseline,"collect":collect,"train":train},
                        samples=sum(groups.values()),samples_per_group=groups,reward=self._reward,
                        policy_version=self._checkpoint_version if self._policy is None else self._policy.version,pretrained=self._pretrained,
                        directory=None if self._run is None else str(self._run),progress_m=self._progress,
                        decisions=self._decisions,last_update=self._last_update,error=self._error,
                        reward_weights=asdict(WEIGHTS),quality_fields=QUALITY_FIELDS,
                        timing_note="开发调试：按主机接收时钟对齐；曝光/接收与命令实际应用偏差尚未标定，不能据此声称论文效果已验证")

    def start(self,kind="baseline",options=None):
        if kind not in ("baseline","collect","train"): raise ExperimentError("实验模式须为 baseline、collect 或 train")
        with self._lock:
            if self._thread and self._thread.is_alive(): raise ExperimentError("已有实验运行")
            if options:
                self._config=validate_config(options,self._config)
                if self._signature()!=self._training_signature:self._pretrained=False
            blockers=self._blockers(kind)
            if blockers: raise ExperimentError("；".join(blockers))
            motion=self.motor.status().get("motion",{})
            if motion.get("state") in ("preparing","running","stopping"):
                raise ExperimentError("请等待当前电机测试停止")
            self._cancel.clear()
            self._state="preparing"
            self._message="检查真实数据与控制时序，启动后自动限时/限距离停车"
            self._error=None
            self._collection_fault=False
            self._mode=kind
            self._thread=threading.Thread(target=self._run_episode,args=(kind,dict(self._config)),daemon=True,name="SPA-real-episode")
            self._thread.start()
        return self.status()

    def stop(self):
        with self._lock:
            self._cancel.set()
            active=bool(self._thread and self._thread.is_alive())
            preparing=self._state=="pretraining"
            if active and not preparing:self.motor.stop_motion("用户停止实验")
        return self.status()

    def close(self):
        self.stop()
        thread=self._thread
        if thread and thread is not threading.current_thread():thread.join(timeout=3)

    def _event(self,record):
        if self._run is None:return
        with self._events_lock:
            with (self._run/"events.jsonl").open("a",encoding="utf-8") as f:
                f.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")

    def _ensure_model(self):
        if self._model is not None:return
        from .web_model import _ensure_torch
        _ensure_torch()
        import torch
        from .two_motor_preview import TwoMotorConfig,TwoMotorModel,TwoMotorPipeline
        from .serial_preview_pipeline import PipelineScales
        from .serial_preview_vision import MobileNetWindowEncoder
        torch.set_num_threads(1)
        self._model=TwoMotorModel(TwoMotorConfig(quality_dim=len(QUALITY_FIELDS)))
        self._policy=CategoricalPPO(142+len(QUALITY_FIELDS),seed=0)
        self._pipeline=TwoMotorPipeline(self._model,self._policy,_FeatureCache(MobileNetWindowEncoder()),PipelineScales(7000,500,.25,.5))
        if self._checkpoint_available:
            record=torch.load(self.root/"latest.pt",map_location="cpu",weights_only=True)
            if record.get("format")!=1 or record.get("signature")!=self._training_signature:
                self._pretrained=False
                raise ExperimentError("检查点配置不一致，请重新预训练")
            self._model.load_state_dict(record["model"])
            self._model.set_phase("frozen");self._model.eval()
            self._policy.params={k:v.numpy().copy() for k,v in record["policy"].items()}
            self._policy.m={k:v.numpy().copy() for k,v in record["optimizer_m"].items()}
            self._policy.v={k:v.numpy().copy() for k,v in record["optimizer_v"].items()}
            self._policy.steps=record["optimizer_steps"]
            self._policy.version=record["policy_version"]
            if "rng_state" in record:self._policy.rng.bit_generator.state=record["rng_state"]
            self._last_update=record["last_update"]

    def _monitor(self,kind,revision,deadline):
        counters={}
        last_camera=last_frame=None
        commands=deque(maxlen=100)
        while not self._monitor_stop.wait(.02):
            collection_fault=True
            try:
                if time.monotonic()>deadline:raise ExperimentError("实验绝对时限到达")
                if self._decision_deadline is not None and time.monotonic()>self._decision_deadline:
                    raise ExperimentError("推理截止时间已过，独立监控停车")
                m=self.motor.status()
                if m.get("frame_fresh") and m.get("fault") not in (None,"NONE"):
                    collection_fault=False
                    raise ExperimentError("C 板保护停止："+str(m.get("fault")))
                if not m.get("frame_fresh") or not m.get("can_healthy"):
                    raise ExperimentError("C 板反馈或保护状态异常")
                gc=self.global_camera.status()
                if not gc.get("valid") or gc.get("revision")!=revision or gc.get("sample_age_s",99)>.25:
                    raise ExperimentError("全局相机失效、数据过期或标定发生变化")
                if self._origin is not None:
                    self._progress=self._config["forward_sign"]*(gc["position_m"]-self._origin)
                    if self._progress>=self._config["distance_m"]:
                        self._message="达到目标距离"
                        self._cancel.set()
                        self.motor.stop_motion("达到目标距离")
                        return
                if kind!="baseline":
                    cs=self.camera.status()
                    if not cs.get("active") or cs.get("frame_age_s") is None or cs["frame_age_s"]>.8:
                        raise ExperimentError("头部相机断帧")
                motion=m.get("motion",{})
                if motion.get("state") in ("failed","completed","idle"):
                    collection_fault=False
                    raise ExperimentError("电机控制提前停止："+motion.get("message",""))
                if self._cancel.is_set():return
                with self._lock:
                    if self._monitor_stop.is_set() or self._cancel.is_set():return
                    targets=dict(self._targets)
                    self.motor.update_motion(targets,owner="paper")
                sent=motion.get("last_speed_sent_monotonic_s")
                sent_targets=motion.get("last_speed_targets") or {}
                if sent is not None and (not commands or commands[-1][0]!=sent):commands.append((sent,sent_targets))
                frame_stamp=m.get("frame_receive_monotonic_s",time.monotonic())
                new=[]
                for ch in (3,4):
                    p=(m.get("motors") or {}).get(str(ch),{})
                    if not p.get("fresh"):raise ExperimentError(f"M{ch} 反馈过期")
                    if p.get("rx")!=counters.get(ch):
                        counters[ch]=p.get("rx")
                        # A packet is available to the host only when the USB
                        # telemetry arrived. Do not backdate receipt by CAN age.
                        stamp=frame_stamp
                        sent_before=[pair for pair in commands if pair[0]<=stamp]
                        command_time,vector=sent_before[-1] if sent_before else (stamp,{3:0,4:0})
                        command=vector.get(str(ch),vector.get(ch,0))
                        new.append(dict(channel=ch,time=stamp,feedback_estimated_time=stamp-max(0,p.get("age_ms",0))/1000,
                                        current=float(p["torque"]),speed=float(p["spd"]),command=int(command),command_time=command_time,rx=p["rx"]))
                with self._lock:self._drives.extend(new)
                for p in new:self._event(dict(type="drive",**p))
                ct=gc["sample_timestamp_monotonic_s"]
                if ct!=last_camera:
                    last_camera=ct
                    p=dict(time=ct,position=float(gc["position_m"]),velocity=float(gc["velocity_mps"]),revision=revision)
                    with self._lock:self._cameras.append(p)
                    self._event(dict(type="global_camera",**p))
                frame=self.camera.latest_frame() if kind!="baseline" else None
                if frame and frame["receive_monotonic_ns"]!=last_frame:
                    last_frame=frame["receive_monotonic_ns"]
                    with self._lock:self._frames.append(frame)
                    (self._run/"images"/f"{last_frame}.jpg").write_bytes(frame["jpeg"])
                    self._event(dict(type="head_frame",receive_monotonic_ns=last_frame,path=f"images/{last_frame}.jpg"))
            except Exception as exc:
                if self._monitor_stop.is_set() or self._cancel.is_set():return
                with self._lock:self._error=str(exc);self._collection_fault=collection_fault
                self._cancel.set()
                self.motor.stop_motion(str(exc))
                return

    def _assemble(self,config,previous,remaining,cutoff):
        import torch
        from .serial_preview_pipeline import DrivePacket
        with self._lock:
            records=list(self._drives);frames=list(self._frames)
        by_channel={ch:[p for p in records if p["channel"]==ch and p["time"]<=cutoff] for ch in (3,4)}
        if any(not values for values in by_channel.values()):raise ExperimentError("没有两路因果反馈")
        front,rear=config["front_motor"],7-config["front_motor"]
        frames=[f for f in frames if f["receive_monotonic_ns"]<=int(cutoff*1e9)]
        if not frames:raise ExperimentError("没有决策截止时间之前的头部图像")
        age=(int(cutoff*1e9)-frames[-1]["receive_monotonic_ns"])/1e9
        quality=[cutoff-by_channel[front][-1]["time"],cutoff-by_channel[rear][-1]["time"],age,1.,
                 config["lead_mm"]/1000,config["motor_to_helix_ratio"],previous/5,config["base_rpm"]/500,
                 config["target_speed_mps"],config["hold_s"],max(0.,remaining/config["duration_s"]),
                 config[f"direction_m{front}"],config[f"direction_m{rear}"],1.,1.]
        packets=[DrivePacket(p["channel"]-1,int(p["time"]*1e9),p["current"],p["speed"],p["command"],int(p["command_time"]*1e9)) for p in records]
        batch,meta=self._pipeline.assemble(frames=frames,drives=packets,cutoff_ns=int(cutoff*1e9),quality=quality,
                                          nominal=[config["base_rpm"]]*2,front_motor=front)
        vectors,features,support,residuals=project(config)
        batch["candidates"]=torch.tensor(features,dtype=torch.float32).unsqueeze(0)
        batch["support"]=torch.tensor([support],dtype=torch.bool)
        meta.update(candidate_vectors=vectors,support=support)
        return batch,meta,residuals

    def _wait_stop(self):
        deadline=time.monotonic()+2.3
        while time.monotonic()<deadline:
            m=self.motor.status()
            if m.get("motion",{}).get("state") not in ("preparing","running","stopping"):
                motors=(m.get("motors") or {})
                if (m.get("frame_fresh") and m.get("stop_acknowledged") and m.get("motion",{}).get("stop_confirmed")
                    and all(motors.get(str(ch),{}).get("fresh") and abs(motors.get(str(ch),{}).get("spd",999))<=3
                            and motors.get(str(ch),{}).get("target")==0 and motors.get(str(ch),{}).get("setV")==0 for ch in (3,4))):
                    return True
            time.sleep(.02)
        return False

    def _wait_applied(self,targets,requested):
        """Bound request-to-ACK and require fresh target echo before labeling."""
        deadline=requested+.25
        first_ack=None
        while time.monotonic()<deadline:
            if self._cancel.is_set():raise ExperimentError(self._error or "动作确认前已停止；该动作没有可用标签")
            state=self.motor.status();motion=state.get("motion",{})
            ack=motion.get("last_speed_ack_monotonic_s")
            ack_targets=motion.get("last_speed_ack_targets") or {}
            echo=state.get("frame_receive_monotonic_s",0)
            if (first_ack is None and ack is not None and ack>=requested
                and all(ack_targets.get(str(ch),ack_targets.get(ch))==targets[ch] for ch in (3,4))):
                first_ack=ack
            if (first_ack is not None and echo>=first_ack
                and all((state.get("motors") or {}).get(str(ch),{}).get("target")==targets[ch]
                        and (state.get("motors") or {}).get(str(ch),{}).get("fresh") for ch in (3,4))):
                return max(first_ack,echo),dict(ack_monotonic_s=first_ack,echo_monotonic_s=echo,request_to_ack_s=first_ack-requested)
            self._cancel.wait(.005)
        self._collection_fault=True
        raise ExperimentError("动作 ACK 或新鲜目标回显未在 0.25 秒内确认")

    def _run_episode(self,kind,config):
        monitor=None
        samples=[]
        intervals=[]
        manifest=dict(format=1,mode=kind,config=config,signature=self._signature(config),
                      reward_weights=asdict(WEIGHTS),warning_ratio=.7,current_safe_raw=7000,
                      quality_fields=QUALITY_FIELDS,source="real_hardware",supervised_samples=0,
                      clock="host_receipt",timing_calibrated=False,publication_ready=False,
                      prepared_model=self._pretrained,policy_version=0 if self._policy is None else self._policy.version)
        started=None
        owned=False
        try:
            self._run=self.root/(time.strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:8])
            (self._run/"images").mkdir(parents=True)
            if kind!="baseline":
                self._ensure_model()
                frame=self.camera.latest_frame()
                self._pipeline.visual_encoder.encode_window([frame],time.monotonic_ns())
            with self._lock:
                self._drives.clear();self._cameras.clear();self._frames.clear()
                self._targets={3:0,4:0};self._progress=0.;self._decisions=0;self._reward=None;self._origin=None
            revision=self.global_camera.status()["revision"]
            with self._lock:
                if self._cancel.is_set():raise ExperimentError("已取消")
                self.motor.start_motion({3:0,4:0},config["duration_s"]+5,owner="paper")
                owned=True
            self._monitor_stop.clear()
            monitor=threading.Thread(target=self._monitor,args=(kind,revision,time.monotonic()+config["duration_s"]+8),daemon=True,name="SPA-independent-supervisor")
            monitor.start()
            wait_deadline=time.monotonic()+5
            while self.motor.status().get("motion",{}).get("state")!="running":
                if self._cancel.wait(.02) or time.monotonic()>wait_deadline:raise ExperimentError(self._error or "电机准备超时或已停止")
            if self._cancel.wait(.3):raise ExperimentError(self._error or "已停止")
            if kind!="baseline":
                # Populate all eight cached features while targets are still
                # zero; the first multi-image CPU batch can exceed 0.2 s.
                with self._lock:warm_frames=list(self._frames)
                self._pipeline.visual_encoder.encode_window(warm_frames,time.monotonic_ns())
                if self._cancel.is_set():raise ExperimentError(self._error or "预热时已停止")
            started=time.monotonic()
            self._origin=self.global_camera.status()["position_m"]
            self._state="running"
            self._message={"baseline":"执行双电机标称基线","collect":"均衡采集后节 -5/0/+5 的真实区间标签","train":"冻结 SPA 表征，采集新鲜随机策略轨迹"}[kind]
            self._event(dict(type="episode_start",**manifest))
            previous=0
            deadline=started+config["duration_s"]
            next_dispatch=started+.2
            action_order=[]
            pending=None
            def finish_interval(item,end,stopped):
                with self._lock:drives,cameras=list(self._drives),list(self._cameras)
                metric=interval_metrics(drives,cameras,item["start"],end,config,item["residual"],item["previous"],stopped)
                if stopped:
                    remaining=max(0,int(math.ceil((deadline-end)/config["hold_s"])))
                    metric["failure_tail"]=failure_tail(remaining,.99,WEIGHTS)
                    metric["reward"]+=metric["failure_tail"]
                metric["command_timing"]=item["command_timing"]
                self._reward=metric["reward"]
                self._decisions+=1
                self._event(dict(type="interval",**metric))
                intervals.append(metric)
                if item["batch"] is not None:
                    samples.append(dict(batch={key:value.detach().cpu() for key,value in item["batch"].items()},
                        action_index=item["action"],targets=metric["labels"],signature=self._signature(config),
                        source="real_supervised",preparation_group=config["preparation_group"]))
                if item["sample"] is not None:
                    import numpy as np
                    sample=item["sample"]
                    metric["transition"]=Transition(observation=item["observation"],support=np.asarray(item["support"],bool),
                        action_index=item["action"],old_log_prob=sample.log_prob,value=sample.value,reward=metric["reward"],
                        next_value=0.,done=False,version=sample.version,source="real_on_policy")
            while time.monotonic()<deadline and not self._cancel.is_set():
                # Infer the NEXT command while the present command is still
                # held. Receipt/ACK delays are retained as timing metadata.
                if self._cancel.wait(max(0.,next_dispatch-.2-time.monotonic())):break
                cutoff=time.monotonic()
                self._decision_deadline=next_dispatch
                batch=meta=sample=observation=None
                prediction_features=None
                vectors,_,support,residuals=project(config)
                if kind!="baseline":
                    batch,meta,residuals=self._assemble(config,previous,deadline-cutoff,cutoff)
                if kind=="train":
                    import torch
                    with torch.no_grad():output=self._model(**batch)
                    observation=output["augmentation"][0].numpy().astype(float)
                    prediction_features=output["prediction"][0].tolist()
                    sample=self._policy.act(observation,support,deterministic=False)
                    action=sample.action_index
                elif kind=="collect":
                    if not action_order:
                        action_order=[i for i,accepted in enumerate(support) if accepted]
                        random.shuffle(action_order)
                    action=action_order.pop()
                else:action=next(i for i,r in enumerate(residuals) if r==0)
                self._decision_deadline=None
                if time.monotonic()>next_dispatch:
                    self._collection_fault=True
                    raise ExperimentError("推理超出固定 0.2 秒预算；请在停机状态检查处理速度")
                if self._cancel.wait(max(0.,next_dispatch-time.monotonic())):break
                dispatch=time.monotonic()
                if pending:
                    finish_interval(pending,dispatch,False)
                    pending=None
                if deadline-dispatch<.2:break
                with self._lock:self._targets={3:vectors[action][2],4:vectors[action][3]}
                self.motor.update_motion(self._targets,owner="paper")
                verified,command_timing=self._wait_applied(self._targets,dispatch)
                command_timing.update(request_monotonic_s=dispatch,verified_monotonic_s=verified,
                    actual_hold_s=config["hold_s"],application_time="unknown; host ACK and fresh target echo only")
                self._event(dict(type="decision",cutoff_monotonic_s=cutoff,dispatch_request_monotonic_s=dispatch,
                                 verified_monotonic_s=verified,command_timing=command_timing,
                                 action_index=action,residual=residuals[action],candidate_vectors=vectors,support=support,
                                 old_log_probability=None if sample is None else sample.log_prob,
                                 value=None if sample is None else sample.value,observation=None if observation is None else observation.tolist(),
                                 policy_version=None if sample is None else sample.version,prediction_features=prediction_features,timing=meta))
                pending=dict(start=verified,previous=previous,residual=residuals[action],batch=batch,
                             sample=sample,observation=observation,support=support,action=action,command_timing=command_timing)
                previous=residuals[action]
                next_dispatch=min(deadline,verified+config["hold_s"])
                if next_dispatch==deadline:
                    self._cancel.wait(max(0.,deadline-time.monotonic()))
                    break
            if pending:
                finish_interval(pending,min(time.monotonic(),deadline),
                                self._cancel.is_set() and self._progress<config["distance_m"])
            self._monitor_stop.set()
            if monitor:monitor.join(timeout=1)
            self.motor.stop_motion("实验结束")
            if not self._wait_stop():raise ExperimentError("停止命令尚未确认零输出，禁止更新策略")
            owned=False
            if self._collection_fault:raise ExperimentError(self._error or "采集数据不完整")
            if kind=="train":
                transitions=[item["transition"] for item in intervals]
                if len(transitions)<2:raise ExperimentError("有效新轨迹少于两步，未更新 PPO")
                transitions=[replace(t,next_value=transitions[i+1].value if i+1<len(transitions) else 0.,done=i+1==len(transitions)) for i,t in enumerate(transitions)]
                self._state="updating"
                self._last_update=self._policy.update(transitions)
                self._save_checkpoint()
                manifest["ppo_update"]=self._last_update
            manifest["completed"]=True
            manifest["terminal_reason"]=self._error or ("distance_reached" if self._progress>=config["distance_m"] else "operator_stop" if self._cancel.is_set() else "time_limit")
            self._state="completed"
            self._message=f"已停机：{len(intervals)} 个真实区间，{self._progress:.3f} m"+("，PPO 已更新" if kind=="train" else "")
        except Exception as exc:
            self._error=str(exc)
            self._state="failed"
            self._message=str(exc)
            manifest.update(completed=False,collection_fault=str(exc),ppo_excluded=True)
        finally:
            self._monitor_stop.set()
            self._decision_deadline=None
            if owned:
                try:self.motor.stop_motion("实验清理停止")
                except Exception as exc:manifest["stop_error"]=str(exc)
            if monitor:monitor.join(timeout=1)
            if self._run:
                if samples:
                    import torch
                    torch.save(samples,self._run/"supervised.pt")
                manifest.update(supervised_samples=len(samples),decisions=len(intervals),progress_m=self._progress,
                                elapsed_s=0 if started is None else time.monotonic()-started,error=self._error)
                (self._run/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
                self._load_catalog()

    def pretrain(self,options=None):
        options=options or {}
        epochs=int(options.get("epochs",10))
        if not 1<=epochs<=100:raise ExperimentError("预训练轮数必须在 1 到 100 之间")
        with self._lock:
            if self._thread and self._thread.is_alive():raise ExperimentError("请先停止当前实验")
            if self.motor.status().get("motion",{}).get("state") in ("preparing","running","stopping"):
                raise ExperimentError("电机未停机，不能训练")
            eligible=[(path,r) for path,r in self._catalog if r.get("signature")==self._signature()]
            groups={r["config"]["preparation_group"] for _,r in eligible}
            if len(groups)<2:raise ExperimentError("至少采集两个独立场地准备组，用整组划分训练/验证；请修改准备组名后采集")
            self._cancel.clear();self._state="pretraining";self._error=None
            self._message="用真实执行动作标签进行两阶段预训练；电机保持停止"
            self._thread=threading.Thread(target=self._pretrain_worker,args=(eligible,epochs),daemon=True,name="SPA-supervised")
            self._thread.start()
        return self.status()

    def _pretrain_worker(self,eligible,epochs):
        try:
            self._ensure_model()
            import torch
            from .two_motor_preview import TwoMotorConfig,TwoMotorModel
            from .serial_preview_model import supervised_step,supervised_loss
            # Restart supervised fitting so previous fits cannot leak a future
            # held-out preparation group into this training/validation split.
            with torch.random.fork_rng():
                torch.manual_seed(0)
                self._model=TwoMotorModel(TwoMotorConfig(quality_dim=len(QUALITY_FIELDS)))
            self._pipeline.model=self._model
            groups=sorted({r["config"]["preparation_group"] for _,r in eligible})
            heldout=groups[-1]
            train=[];validation=[]
            for path,record in eligible:
                samples=torch.load(path/"supervised.pt",map_location="cpu",weights_only=True)
                if record["config"]["preparation_group"]==heldout:validation.extend(samples)
                else:train.append(samples)
            if sum(map(len,train))<6 or len(validation)<2:raise ExperimentError("训练至少需要 6 个区间，整组验证至少需要 2 个区间")
            all_labels=torch.tensor([s["targets"][:3] for run in train for s in run],dtype=torch.float32)
            scales=all_labels.std(dim=0).clamp(min=.01)
            losses={}
            for phase in ("reference","history"):
                self._model.set_phase(phase)
                optimizer=torch.optim.Adam([p for p in self._model.parameters() if p.requires_grad],lr=3e-4)
                for epoch in range(epochs):
                    # Each independent trajectory contributes exactly one sampled
                    # interval per repetition, irrespective of its frame count.
                    epoch_losses=[]
                    repetitions=max(len(run) for run in train)
                    for _ in range(repetitions):
                        for run in train:
                            if self._cancel.is_set():raise ExperimentError("预训练已取消")
                            s=random.choice(run)
                            epoch_losses.append(supervised_step(self._model,optimizer,s["batch"],torch.tensor([s["action_index"]]),
                                torch.tensor([s["targets"]],dtype=torch.float32),torch.ones((1,4),dtype=torch.bool),scales))
                    losses[phase]=sum(epoch_losses)/len(epoch_losses)
                    self._message=f"预训练 {phase}：{epoch+1}/{epochs}，loss={losses[phase]:.4f}"
            self._model.set_phase("frozen");self._model.eval()
            with torch.no_grad():
                validation_loss=sum(float(supervised_loss(self._model(**s["batch"])["raw"],torch.tensor([s["action_index"]]),
                    torch.tensor([s["targets"]],dtype=torch.float32),torch.ones((1,4),dtype=torch.bool),scales)) for s in validation)/len(validation)
            self._policy=CategoricalPPO(142+len(QUALITY_FIELDS),seed=0)
            self._pipeline.policy=self._policy
            self._training_signature=self._signature()
            self._pretrained=True
            self._checkpoint_available=True
            self._last_update=dict(stage="supervised",losses=losses,validation_loss=validation_loss,heldout_group=heldout,
                                   train_groups=[g for g in groups if g!=heldout],target_scales=scales.tolist(),epochs=epochs,
                                   trained_on_robot=True,performance_validated=False)
            self._save_checkpoint()
            self._state="ready"
            self._message="两阶段预训练完成，表征已冻结；可点击真实 PPO，尚未证明控制收益"
        except Exception as exc:
            self._pretrained=False;self._state="failed";self._error=str(exc);self._message=str(exc)

    def _save_checkpoint(self):
        import torch
        self.root.mkdir(parents=True,exist_ok=True)
        torch.save(dict(format=1,model_config=vars(self._model.config),model=self._model.state_dict(),
                        signature=self._training_signature,pretrained=self._pretrained,last_update=self._last_update,
                        policy_version=self._policy.version,policy={k:torch.tensor(v) for k,v in self._policy.params.items()},
                        optimizer_m={k:torch.tensor(v) for k,v in self._policy.m.items()},
                        optimizer_v={k:torch.tensor(v) for k,v in self._policy.v.items()},optimizer_steps=self._policy.steps,
                        rng_state=self._policy.rng.bit_generator.state,
                        source="real_supervised_then_fresh_on_policy",timing_calibrated=False),self.root/"latest.tmp.pt")
        (self.root/"latest.tmp.pt").replace(self.root/"latest.pt")
        self._checkpoint_version=self._policy.version
        (self.root/"checkpoint.json").write_text(json.dumps(dict(signature=self._training_signature,
            pretrained=self._pretrained,policy_version=self._policy.version),indent=2),encoding="utf-8")
