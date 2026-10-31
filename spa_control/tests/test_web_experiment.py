"""Hardware-free end-to-end tests; all generated samples live in temp dirs."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from spa_control.web_experiment import (ExperimentError,WebExperiment,DEFAULTS,
                                       QUALITY_FIELDS,interval_metrics,project,validate_config)


class FakeMotor:
    def __init__(self):
        self.targets={3:0,4:0}
        self.state="idle"
        self.sent=self.ack=0.
        self.calls=[]
        self.fault="NONE"
        self.stale=False
        self.stop_confirmed=True

    def status(self):
        now=time.monotonic()
        return dict(frame_fresh=not self.stale,fault=self.fault,can_healthy=True,
            frame_receive_monotonic_s=now,stop_acknowledged=self.stop_confirmed,
            motors={str(ch):dict(fresh=True,torque=100+ch,spd=0,target=self.targets[ch],setV=0,
                                 age_ms=1,rx=int(now*10000)) for ch in (3,4)},
            motion=dict(state=self.state,owner="paper",targets=self.targets,
                last_speed_sent_monotonic_s=self.sent,last_speed_targets=dict(self.targets),
                last_speed_ack_monotonic_s=self.ack,last_speed_ack_targets=dict(self.targets),
                stop_confirmed=self.stop_confirmed,message="fake only"))

    def start_motion(self,targets,duration_s,owner):
        self.state="running";self.targets=dict(targets)
        self.stop_confirmed=False
        self.calls.append(("start",dict(targets)))
        self.sent=self.ack=time.monotonic()

    def update_motion(self,targets,owner):
        if self.state!="running":raise RuntimeError("no active fake motion")
        self.targets=dict(targets)
        self.sent=self.ack=time.monotonic()
        self.calls.append(("update",dict(targets)))

    def stop_motion(self,reason):
        self.state="completed";self.targets={3:0,4:0}
        self.stop_confirmed=True
        self.calls.append(("stop",reason))


class FakeCamera:
    def __init__(self):self.active=True
    def status(self):return dict(active=self.active,frame_age_s=0.01)
    def latest_frame(self):return dict(jpeg=b"test-only-fake-jpeg",receive_monotonic_ns=time.monotonic_ns(),age_s=0.)


class FakeGlobal:
    def __init__(self):
        self.revision=1;self.valid=True;self.start=time.monotonic()
    def status(self):
        now=time.monotonic()
        return dict(valid=self.valid,revision=self.revision,sample_age_s=0.,
            sample_timestamp_monotonic_s=now,position_m=(now-self.start)*.03,velocity_mps=.03)


class FakeEncoder:
    def encode_window(self,frames,cutoff_ns,max_frames=8):
        return dict(frames=[dict(receive_monotonic_ns=f["receive_monotonic_ns"],valid=True,features=[0.]*576)
                           for f in frames if f["receive_monotonic_ns"]<=cutoff_ns][-max_frames:])


def finish(experiment,timeout=15):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if not experiment.status()["active"]:return experiment.status()
        time.sleep(.02)
    raise AssertionError("fake experiment did not finish")


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory()
        self.motor=FakeMotor();self.camera=FakeCamera();self.global_camera=FakeGlobal()
        self.experiment=WebExperiment(self.motor,self.camera,self.global_camera,self.folder.name)
    def tearDown(self):
        self.experiment.close();self.folder.cleanup()

    def configured(self):
        self.experiment.configure(dict(front_motor=3,duration_s=2.4,hold_s=.5,distance_m=1))

    def inject_model(self):
        from spa_control.web_model import _ensure_torch
        _ensure_torch()
        import torch
        from spa_control.two_motor_preview import TwoMotorModel,TwoMotorConfig,TwoMotorPipeline
        from spa_control.serial_preview import CategoricalPPO
        from spa_control.serial_preview_pipeline import PipelineScales
        torch.set_num_threads(1)
        model=TwoMotorModel(TwoMotorConfig(quality_dim=len(QUALITY_FIELDS)))
        policy=CategoricalPPO(142+len(QUALITY_FIELDS))
        self.experiment._model=model;self.experiment._policy=policy
        self.experiment._pipeline=TwoMotorPipeline(model,policy,FakeEncoder(),PipelineScales(7000,500,.25,.5))

    def test_idle_is_lazy_and_never_commands(self):
        self.experiment.status();self.experiment.close()
        self.assertEqual(self.motor.calls,[])
        self.assertIsNone(self.experiment._model)
        self.assertFalse(self.experiment.status()["ready_baseline"])

    def test_disconnected_motor_none_mapping(self):
        self.motor.status=lambda:dict(motors=None)
        self.assertFalse(self.experiment.status()["ready_baseline"])

    def test_config_defaults_conversion_persistence_and_limits(self):
        config=validate_config(dict(base_rpm=100,front_motor=4))
        self.assertEqual(config["target_speed_mps"],.05)
        self.experiment.configure(config)
        restored=WebExperiment(self.motor,self.camera,self.global_camera,self.folder.name)
        self.assertEqual(restored.status()["config"],config)
        for invalid in (dict(base_rpm=501),dict(direction_m3=0),dict(target_speed_mps=True),dict(target_speed_mps=.5)):
            with self.assertRaises(ExperimentError):validate_config(invalid)

    def test_projection_structural_sign_and_inactive_zero(self):
        config=validate_config(dict(front_motor=4,base_rpm=499,direction_m4=1))
        vectors,features,support,residuals=project(config)
        self.assertEqual(vectors,[[0,0,-494,499],[0,0,-499,499],[0,0,-500,499]])
        self.assertEqual(residuals,[-5,0,1])
        self.assertEqual(support,[True]*3)

    def test_missing_head_blocks_collect_but_not_baseline(self):
        self.configured();self.camera.active=False
        self.assertTrue(self.experiment.status()["ready_baseline"])
        self.assertFalse(self.experiment.status()["ready_collect"])
        with self.assertRaises(ExperimentError):self.experiment.start("collect")
        self.assertEqual(self.motor.calls,[])

    def test_baseline_at_least_three_intervals_and_stop_confirmed(self):
        self.configured();self.experiment.start("baseline")
        status=finish(self.experiment)
        self.assertEqual(status["state"],"completed",status)
        self.assertGreaterEqual(status["decisions"],3)
        self.assertEqual(self.motor.targets,{3:0,4:0})
        self.assertEqual(status["policy_version"],0)
        records=[json.loads(line) for line in (Path(status["directory"])/"events.jsonl").read_text(encoding="utf-8").splitlines()]
        labels=[r for r in records if r["type"]=="interval"]
        self.assertTrue(all(r["grid_coverage"]==1 for r in labels))
        self.assertTrue(all(r["start_monotonic_s"]>=r["command_timing"]["ack_monotonic_s"] for r in labels))
        self.assertTrue(all(r["time"]>=r["feedback_estimated_time"] for r in records if r["type"]=="drive"))

    def test_revision_change_stops_and_excludes_ppo(self):
        self.configured();self.experiment.start("baseline")
        time.sleep(.8);self.global_camera.revision+=1
        status=finish(self.experiment)
        self.assertEqual(status["state"],"failed")
        manifest=json.loads((Path(status["directory"])/"manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["ppo_excluded"])
        self.assertEqual(self.motor.targets,{3:0,4:0})

    def test_collect_real_model_assembly_then_fresh_ppo_three_intervals(self):
        self.configured();self.inject_model()
        self.experiment.start("collect")
        collected=finish(self.experiment)
        self.assertEqual(collected["state"],"completed",collected)
        self.assertGreaterEqual(collected["samples"],3)
        self.assertEqual(collected["policy_version"],0)
        # Tests mark their fake model frozen explicitly; production requires
        # actual supervised pretraining. Existing records must NOT enter PPO.
        self.experiment._model.set_phase("frozen")
        self.experiment._pretrained=True
        self.experiment._training_signature=self.experiment._signature()
        self.experiment.start("train")
        trained=finish(self.experiment)
        self.assertEqual(trained["state"],"completed",trained)
        self.assertGreaterEqual(trained["decisions"],3)
        self.assertEqual(trained["policy_version"],1)
        self.assertEqual(trained["last_update"]["transitions"],trained["decisions"])
        self.assertTrue(all(not p.requires_grad for p in self.experiment._model.parameters()))

    def test_train_cannot_use_untrained_model(self):
        self.configured()
        with self.assertRaises(ExperimentError):self.experiment.start("train")
        with self.assertRaises(ExperimentError):self.experiment.pretrain()
        self.assertEqual(self.motor.calls,[])

    def test_protective_stop_retains_measured_penalty_and_fresh_update(self):
        self.configured();self.inject_model()
        self.experiment._model.set_phase("frozen")
        self.experiment._pretrained=True
        self.experiment._training_signature=self.experiment._signature()
        self.experiment.start("train")
        # Synchronize to an actual recorded third command, then interrupt in
        # the middle of its measured hold rather than racing a boundary.
        deadline=time.monotonic()+8
        while time.monotonic()<deadline:
            directory=self.experiment.status()["directory"]
            path=None if directory is None else Path(directory)/"events.jsonl"
            if path is not None and path.exists():
                text=path.read_text(encoding="utf-8")
                lines=text.split("\n")[:-1]
                decisions=[r for r in map(json.loads,lines) if r["type"]=="decision"]
                if len(decisions)>=3:
                    phase=time.monotonic()-decisions[-1]["verified_monotonic_s"]
                    if .15<phase<.35:break
            time.sleep(.005)
        else:self.fail("no third active command to interrupt")
        self.motor.fault="OVERCURRENT"
        status=finish(self.experiment)
        self.assertEqual(status["state"],"completed",status)
        self.assertEqual(status["policy_version"],1)
        events=[json.loads(line) for line in (Path(status["directory"])/"events.jsonl").read_text(encoding="utf-8").splitlines()]
        stopped=[r for r in events if r["type"]=="interval" and r["stop"]]
        self.assertEqual(len(stopped),1)
        self.assertLess(stopped[0]["failure_tail"],0)
        self.assertEqual(stopped[0]["labels"][3],1)

    def test_supervised_whole_group_split_freeze_and_checkpoint_restore(self):
        self.configured();self.inject_model()
        import torch
        from spa_control.serial_preview_pipeline import DrivePacket
        cutoff=10_000_000_000
        drives=[DrivePacket(ch,cutoff-age*1_000_000,100,-60,-60,cutoff-age*1_000_000)
                for age in range(4400,0,-100) for ch in (2,3)]
        batch,_=self.experiment._pipeline.assemble(frames=[dict(receive_monotonic_ns=cutoff-1)],
            drives=drives,cutoff_ns=cutoff,quality=[0.]*len(QUALITY_FIELDS),nominal=[60,60],front_motor=3)
        for group,count in (("a-train",8),("z-holdout",3)):
            folder=Path(self.folder.name)/group;folder.mkdir()
            config=dict(self.experiment._config,preparation_group=group)
            records=[dict(batch=batch,action_index=i%3,targets=[.01 if group=="a-train" else 10.,.05,.1,0.],source="real_supervised") for i in range(count)]
            torch.save(records,folder/"supervised.pt")
            (folder/"manifest.json").write_text(json.dumps(dict(format=1,signature=self.experiment._signature(),
                config=config,supervised_samples=count)),encoding="utf-8")
        self.experiment._load_catalog()
        self.experiment.pretrain(dict(epochs=1))
        status=finish(self.experiment)
        self.assertEqual(status["state"],"ready",status)
        self.assertEqual(status["last_update"]["heldout_group"],"z-holdout")
        self.assertEqual(status["last_update"]["train_groups"],["a-train"])
        self.assertTrue(all(abs(value-.01)<1e-6 for value in status["last_update"]["target_scales"]))
        self.assertTrue(all(not p.requires_grad for p in self.experiment._model.parameters()))
        self.assertEqual(self.motor.calls,[])
        restored=WebExperiment(self.motor,self.camera,self.global_camera,self.folder.name)
        self.assertTrue(restored.status()["pretrained"])
        self.assertIsNone(restored._model)
        restored._ensure_model()
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(restored._model.parameters(),self.experiment._model.parameters())))
        self.assertEqual(self.motor.calls,[])


class RewardTests(unittest.TestCase):
    def fixture(self):
        drives=[dict(channel=ch,time=t/10,current=3500,speed=-60) for t in range(11) for ch in (3,4)]
        camera=[dict(time=t/10,position=t/10*.03,velocity=.03) for t in range(11)]
        return drives,camera,validate_config(dict(front_motor=3))

    def test_complete_interval_reward_and_original_packet_counts(self):
        drives,camera,config=self.fixture()
        result=interval_metrics(drives,camera,0.,.5,config,5,0)
        self.assertAlmostEqual(result["reward"],1-.02-.02*.25)
        self.assertEqual(result["packet_counts"],{"3":5,"4":5})
        self.assertAlmostEqual(result["labels"][0],.015)
        self.assertEqual(result["labels"][1],.5)

    def test_missing_real_measurements_cannot_be_zero_filled(self):
        drives,camera,config=self.fixture()
        with self.assertRaises(ExperimentError):interval_metrics(drives,[],0,.5,config,0,0)
        with self.assertRaises(ExperimentError):interval_metrics([p for p in drives if p["channel"]==3],camera,0,.5,config,0,0)

    def test_protective_stop_has_penalty_and_all_active_current_cost(self):
        drives,camera,config=self.fixture()
        for p in drives:
            if p["channel"]==3:p["current"]=7000
        result=interval_metrics(drives,camera,0,.5,config,0,0,True)
        self.assertEqual(result["labels"][3],1)
        self.assertAlmostEqual(result["current_cost"],.045)
        self.assertEqual(result["occupancy"],1)
        self.assertAlmostEqual(result["reward"],-.109)


if __name__=="__main__":unittest.main()
