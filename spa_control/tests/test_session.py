"""Threaded trial integration using fake robot and camera only.

Demo measurements test software plumbing; they are not robot experiments.
"""
from contextlib import contextmanager
from dataclasses import replace
import csv
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from spa_control.camera import CameraSample
from spa_control.core import ExperimentConfig
from spa_control.ppo import PPOAgent
from spa_control.session import DemoRobot, TrialRunner, baseline_commands


class RecordingRobot(DemoRobot):
    def __init__(self, config):
        super().__init__(config)
        self.sends = []
        self.stops = []

    def send_speeds(self, speeds, lease_s=0.4):
        self.sends.append((time.monotonic(), list(speeds)))
        return super().send_speeds(speeds, lease_s)

    def stop(self):
        self.stops.append(time.monotonic())
        return super().stop()


class DelayedCamera:
    def __init__(self, offset_s=0.0, frozen_received=False):
        self.timestamp = time.monotonic() - offset_s
        self.frozen_received = frozen_received

    def latest(self, max_age_s=0.8):
        return CameraSample(self.timestamp, self.timestamp if self.frozen_received else time.monotonic(),
                            0.0, 0.0, True, "UNIT_TEST_ONLY")


class InterceptCsvStream:
    """Inject disk behavior after the header, using a genuine temporary CSV."""
    def __init__(self, stream, first_row_flush):
        self.stream = stream
        self.first_row_flush = first_row_flush
        self.flush_count = 0

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, *args):
        return self.stream.__exit__(*args)

    def write(self, *args):
        return self.stream.write(*args)

    def flush(self):
        self.flush_count += 1
        if self.flush_count == 2:
            self.first_row_flush()
        return self.stream.flush()


@contextmanager
def intercept_csv_flush(callback):
    original_open = Path.open
    def open_csv(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        if path.name == "steps.csv" and args and args[0] == "w":
            return InterceptCsvStream(stream, callback)
        return stream
    with patch.object(Path, "open", new=open_csv):
        yield


@contextmanager
def heartbeat(runner):
    stop = threading.Event()
    def run():
        while not stop.wait(0.02):
            runner.ping()
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = ExperimentConfig(parameters_confirmed=True, rotor_to_helix_ratio=20,
                                       rpm_per_command=10, lead_mm=60, duration_s=0.5, control_hz=20,
                                       signs=[1, -1, 1, -1], nominal_command=40, command_limit=120)
        self.runner = TrialRunner(self.root)
        self.robot = RecordingRobot(self.config)
        self.runner.link = self.robot
        self.runner.camera = self.robot

    def tearDown(self):
        if self.runner.busy:
            self.runner.abort("测试清理")
        if self.runner.worker is not None:
            self.runner.worker.join(timeout=2)
        self.temp.cleanup()

    def finish(self):
        self.runner.worker.join(timeout=4)
        self.assertFalse(self.runner.worker.is_alive(), "worker failed to finish")
        self.assertFalse(self.runner.busy)
        paths = sorted((self.root / "demo").glob("*/summary.json"))
        self.assertTrue(paths)
        return json.loads(paths[-1].read_text(encoding="utf-8"))

    def run_trial(self, mode, setup=None):
        with heartbeat(self.runner):
            self.runner.start(setup or self.config, mode, demo=True)
            return self.finish()

    def test_baseline_train_eval_and_fresh_policy_versions(self):
        captured = []
        original_update = PPOAgent.update
        def capture(agent, transitions):
            captured.append(list(transitions))
            return original_update(agent, transitions)
        with patch.object(PPOAgent, "update", new=capture):
            baseline = self.run_trial("baseline")
            self.assertEqual(baseline["outcome"], "completed", baseline)
            self.assertEqual(self.runner.agent.total_updates, 0)
            self.assertTrue(all(commands == [40, -40, 40, -40, 0, 0] for _, commands in self.robot.sends))
            train = self.run_trial("train")
            self.assertEqual(train["outcome"], "completed", train)
            self.assertEqual(self.runner.agent.total_updates, 1)
            train_again = self.run_trial("train")
            self.assertEqual(train_again["outcome"], "completed", train_again)
            self.assertEqual(self.runner.agent.total_updates, 2)
            evaluation = self.run_trial("eval")
            self.assertEqual(evaluation["outcome"], "completed", evaluation)
            self.assertEqual(self.runner.agent.total_updates, 2)
        self.assertEqual(len(captured), 2)
        self.assertEqual({t.policy_version for t in captured[0]}, {0})
        self.assertEqual({t.policy_version for t in captured[1]}, {1})
        for data in captured:
            self.assertTrue(data[-1].done)
            self.assertEqual(data[-1].next_value, 0)
            self.assertTrue(all(not t.done for t in data[:-1]))
        self.assertEqual(self.runner.completed, {"baseline": 1, "train": 2, "eval": 1,
                                                 "baseline_no_camera": 0, "single_motor": 0})
        self.assertEqual(self.robot.speeds, [0] * 6)
        self.assertFalse(evaluation["stop_execution_confirmed"])
        self.assertTrue(evaluation["stop_write_succeeded"])
        self.assertTrue(all(commands[4:] == [0, 0] for _, commands in self.robot.sends))

    def test_abort_discards_training_rollout_and_stops(self):
        with heartbeat(self.runner), patch.object(self.runner.agent, "update", wraps=self.runner.agent.update) as update:
            self.runner.start(self.config, "train", demo=True)
            deadline = time.monotonic() + 2
            while not self.runner.history and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(self.runner.history)
            self.runner.abort("测试人工急停")
            result = self.finish()
        self.assertEqual(result["outcome"], "aborted")
        self.assertIn("测试人工急停", result["reason"])
        update.assert_not_called()
        self.assertEqual(self.runner.agent.policy_version, 0)
        self.assertEqual(self.robot.speeds, [0] * 6)

    def test_abort_between_policy_sampling_and_send_never_restarts_motion(self):
        entered = threading.Event()
        release = threading.Event()
        original_act = self.runner.agent.act
        def blocked_act(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(timeout=2))
            return original_act(*args, **kwargs)
        with heartbeat(self.runner), patch.object(self.runner.agent, "act", side_effect=blocked_act):
            self.runner.start(self.config, "train", demo=True)
            self.assertTrue(entered.wait(timeout=2))
            try:
                self.runner.abort("测试采样期间急停")
            finally:
                release.set()
            result = self.finish()
        self.assertEqual(result["outcome"], "aborted")
        self.assertEqual(self.robot.sends, [], "A nonzero command was issued after emergency stop")
        self.assertEqual(self.runner.agent.total_updates, 0)

    def test_newly_received_but_pre_action_camera_measurement_is_rejected(self):
        self.runner.camera = DelayedCamera()
        result = self.run_trial("train")
        self.assertEqual(result["outcome"], "aborted", "Buffered pre-action frames must not form PPO rewards")
        self.assertEqual(result["steps"], 0)
        self.assertEqual(self.runner.agent.total_updates, 0)

    def test_camera_without_new_received_frame_aborts(self):
        self.runner.camera = DelayedCamera(frozen_received=True)
        result = self.run_trial("train")
        self.assertEqual(result["outcome"], "aborted")
        self.assertEqual(result["steps"], 0)
        self.assertEqual(self.runner.agent.total_updates, 0)

    def test_final_complete_transition_terminal_when_bookkeeping_crosses_deadline(self):
        original_value = self.runner.agent.value
        original_update = PPOAgent.update
        calls = 0
        captured = []
        def slow_value(obs):
            nonlocal calls
            calls += 1
            if calls == 2:
                time.sleep(0.5)
            return original_value(obs)
        def capture(agent, transitions):
            captured.extend(transitions)
            return original_update(agent, transitions)
        with patch.object(self.runner.agent, "value", side_effect=slow_value), \
                patch.object(PPOAgent, "update", new=capture):
            result = self.run_trial("train")
        self.assertEqual(result["outcome"], "completed", result)
        self.assertTrue(captured[-1].done)
        self.assertEqual(captured[-1].next_value, 0.0)

    def test_missing_stale_and_overcurrent_telemetry_fail_before_motion(self):
        original_snapshot = self.robot.snapshot
        cases = [({}, "缺失"),
                 ({"last_seen": time.monotonic() - 10}, "过时"),
                 ({"current_raw": self.config.current_limit_raw}, "电流")]
        for values, reason in cases:
            data = original_snapshot()
            if values:
                data[1].update(values)
            else:
                data.pop(1)
            with self.subTest(reason=reason), patch.object(self.robot, "snapshot", return_value=data):
                with self.assertRaises(ValueError):
                    self.runner.start(self.config, "train", demo=True)
        self.assertEqual(self.robot.sends, [])

    def test_nonfinite_and_future_telemetry_fail_before_motion(self):
        original_snapshot = self.robot.snapshot
        for field, value in (("speed", float("nan")), ("current_raw", float("nan")),
                             ("last_seen", float("nan")), ("last_seen", time.monotonic() + 20)):
            data = original_snapshot()
            data[1][field] = value
            with self.subTest(field=field, value=value), patch.object(self.robot, "snapshot", return_value=data):
                with self.assertRaises(ValueError):
                    self.runner._sensors(self.config)

    def test_calibration_and_policy_identity_fail_before_motion(self):
        with self.assertRaises(ValueError):
            self.runner.start(replace(self.config, parameters_confirmed=False), "baseline", demo=True)
        self.runner.policy_config = self.runner.policy_signature(self.config)
        with self.assertRaises(ValueError):
            self.runner.start(replace(self.config, channels=[2, 1, 3, 4]), "train", demo=True)
        self.assertEqual(self.robot.sends, [])

    def test_failed_stop_write_prevents_policy_update(self):
        def failed_stop():
            self.robot.speeds = [0] * 6
            return False
        with patch.object(self.robot, "stop", side_effect=failed_stop):
            result = self.run_trial("train")
        self.assertEqual(result["outcome"], "aborted")
        self.assertFalse(result["stop_write_succeeded"])
        self.assertEqual(self.runner.agent.total_updates, 0)

    def test_gui_heartbeat_loss_stops_without_training(self):
        self.runner.start(replace(self.config, duration_s=1.3), "train", demo=True)
        result = self.finish()
        self.assertEqual(result["outcome"], "aborted")
        self.assertIn("心跳", result["reason"])
        self.assertEqual(self.runner.agent.total_updates, 0)
        self.assertEqual(self.robot.speeds, [0] * 6)

    def test_abort_during_candidate_update_preserves_active_policy(self):
        entered = threading.Event()
        release = threading.Event()
        original_update = PPOAgent.update
        active = self.runner.agent
        initial_parameters = {name: value.copy() for name, value in active.params.items()}
        candidates = []
        def blocked_update(candidate, transitions):
            candidates.append(candidate)
            result = original_update(candidate, transitions)
            entered.set()
            self.assertTrue(release.wait(timeout=2))
            return result
        with heartbeat(self.runner), patch.object(PPOAgent, "update", new=blocked_update):
            self.runner.start(self.config, "train", demo=True)
            self.assertTrue(entered.wait(timeout=2))
            try:
                self.runner.abort("测试更新期间取消")
            finally:
                release.set()
            result = self.finish()
        self.assertEqual(result["outcome"], "aborted")
        self.assertIn("测试更新期间取消", result["reason"])
        self.assertEqual(len(candidates), 1)
        self.assertIsNot(candidates[0], active)
        self.assertEqual(candidates[0].total_updates, 1)
        self.assertIs(self.runner.agent, active)
        self.assertEqual(active.total_updates, 0)
        self.assertEqual(active.policy_version, 0)
        self.assertEqual(self.runner.completed["train"], 0)
        for name, value in active.params.items():
            np.testing.assert_array_equal(value, initial_parameters[name])
        self.assertEqual(self.robot.speeds, [0] * 6)
        self.assertEqual(list(self.root.glob("demo/*/policy_after.npz")), [])


class CameraFreeBaselineTests(unittest.TestCase):
    """Additional controls test only the new baseline entry point."""

    def setUp(self):
        SessionTests.setUp(self)

    tearDown = SessionTests.tearDown
    finish = SessionTests.finish

    def prepare_baseline(self, **changes):
        setup = replace(ExperimentConfig(duration_s=0.5, control_hz=20), **changes)
        self.robot = RecordingRobot(setup)
        self.runner.link = self.robot
        self.runner.camera = None
        return setup

    def run_baseline(self, setup, **kwargs):
        with heartbeat(self.runner):
            self.runner.start_baseline(setup, demo=True, **kwargs)
            return self.finish()

    def test_no_camera_or_calibration_runs_300_command_without_policy_access(self):
        setup = self.prepare_baseline()
        self.assertFalse(setup.parameters_confirmed)
        self.assertIsNone(setup.rpm_per_command)
        self.assertIsNone(setup.rotor_to_helix_ratio)
        initial_version = self.runner.agent.policy_version
        with patch.object(self.robot, "latest", side_effect=AssertionError("Camera must not be read")), \
                patch.object(self.runner.agent, "act", side_effect=AssertionError("Actor must not run")), \
                patch.object(self.runner.agent, "update", side_effect=AssertionError("PPO must not update")), \
                patch.object(self.runner.agent, "save", side_effect=AssertionError("No policy checkpoint")), \
                patch.object(ExperimentConfig, "theory_speed", side_effect=AssertionError("No theory required")):
            result = self.run_baseline(setup)
        self.assertEqual(result["outcome"], "completed", result)
        self.assertEqual(result["mode"], "baseline_no_camera")
        self.assertEqual(self.runner.completed["baseline_no_camera"], 1)
        self.assertEqual(self.runner.agent.policy_version, initial_version)
        self.assertTrue(self.robot.sends)
        self.assertTrue(all(commands == [-300, 300, -300, 300, 0, 0] for _, commands in self.robot.sends))
        self.assertEqual(self.robot.speeds, [0] * 6)
        self.assertEqual(self.runner.latest["j0_rpm"], -300)
        self.assertEqual(self.runner.latest["telemetry_source"], "DEMO")
        for index in range(4):
            self.assertEqual(self.runner.latest[f"j{index}_residual_applied"], 0)
        for key in result:
            self.assertNotIn("reward", key)
            self.assertNotIn("efficiency", key)
            self.assertNotIn("actual_mps", key)
        run_dir = next((self.root / "demo").iterdir())
        with (run_dir / "steps.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fields = reader.fieldnames
        self.assertEqual(len(rows), result["steps"])
        self.assertFalse(any("reward" in key or "mps" in key for key in fields))
        self.assertFalse((run_dir / "rollout.npz").exists())
        self.assertFalse((run_dir / "policy_before.npz").exists())
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertFalse(metadata["camera_used"])
        self.assertFalse(metadata["ppo_used"])
        self.assertIn("not calibrated rpm", metadata["command_units"])

    def test_camera_free_arbitrary_mapping_preserves_inactive_zero(self):
        result = self.run_baseline(self.prepare_baseline(channels=[6, 2, 5, 1]))
        self.assertEqual(result["outcome"], "completed", result)
        self.assertTrue(all(commands == [300, 300, 0, 0, -300, -300] for _, commands in self.robot.sends))

    def test_camera_free_abort_stops_and_does_not_count_completion(self):
        setup = self.prepare_baseline()
        with heartbeat(self.runner):
            self.runner.start_baseline(setup, demo=True)
            deadline = time.monotonic() + 2
            while not self.runner.history and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(self.runner.history)
            self.runner.abort("无相机急停测试")
            stopped_at = time.monotonic()
            result = self.finish()
        self.assertEqual(result["outcome"], "aborted")
        self.assertEqual(self.runner.completed["baseline_no_camera"], 0)
        self.assertEqual(self.robot.speeds, [0] * 6)
        self.assertTrue(all(sent_at <= stopped_at for sent_at, _ in self.robot.sends))

    def test_camera_free_sensor_faults_fail_before_send(self):
        setup = self.prepare_baseline()
        original = self.robot.snapshot
        for key, value in (("current_raw", setup.current_limit_raw), ("speed", float("nan")),
                           ("last_seen", time.monotonic() - 10)):
            data = original()
            data[1][key] = value
            with self.subTest(key=key), patch.object(self.robot, "snapshot", return_value=data):
                with self.assertRaises(ValueError):
                    self.runner.start_baseline(setup, demo=True)
        self.assertEqual(self.robot.sends, [])

    def test_camera_free_lost_heartbeat_stops(self):
        setup = self.prepare_baseline(duration_s=1.3)
        self.runner.start_baseline(setup, demo=True)
        result = self.finish()
        self.assertEqual(result["outcome"], "aborted")
        self.assertIn("心跳", result["reason"])
        self.assertEqual(self.robot.speeds, [0] * 6)

    def test_camera_free_overcurrent_during_motion_stops(self):
        setup = self.prepare_baseline()
        original = self.robot.snapshot
        def overcurrent():
            data = original()
            if self.robot.sends:
                data[1]["current_raw"] = setup.current_limit_raw
            return data
        with patch.object(self.robot, "snapshot", side_effect=overcurrent):
            result = self.run_baseline(setup)
        self.assertEqual(result["outcome"], "aborted")
        self.assertIn("电流", result["reason"])
        self.assertEqual(self.runner.completed["baseline_no_camera"], 0)
        self.assertEqual(self.robot.speeds, [0] * 6)
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["peak_abs_current_raw"], setup.current_limit_raw)
        self.assertEqual(result["safety_fault"]["channel"], 1)
        self.assertEqual(result["safety_fault"]["limit_raw"], setup.current_limit_raw)
        self.assertEqual(result["safety_fault"]["telemetry"]["1"]["current_raw"], setup.current_limit_raw)

    def test_camera_free_overcurrent_peak_includes_triggering_snapshot_after_valid_steps(self):
        setup = self.prepare_baseline()
        original = self.robot.snapshot
        fault_current = -(setup.current_limit_raw + 123)
        def overcurrent():
            data = original()
            if len(self.robot.sends) >= 2:
                data[2]["current_raw"] = fault_current
                data[4]["current_raw"] = fault_current - 77
            return data
        with patch.object(self.robot, "snapshot", side_effect=overcurrent):
            result = self.run_baseline(setup)
        self.assertEqual(result["outcome"], "aborted")
        self.assertGreater(result["steps"], 0)
        self.assertEqual(result["peak_abs_current_raw"], abs(fault_current - 77))
        self.assertEqual(result["safety_fault"]["type"], "overcurrent")
        self.assertEqual(result["safety_fault"]["channel"], 2)
        self.assertEqual(result["safety_fault"]["current_raw"], fault_current)
        self.assertEqual(result["safety_fault"]["telemetry"]["4"]["current_raw"], fault_current - 77)
        self.assertGreaterEqual(result["safety_fault"]["observed_monotonic_s"], self.robot.sends[-1][0])
        self.assertEqual(self.robot.speeds, [0] * 6)

    def test_camera_free_summary_write_failure_is_not_a_completed_trial(self):
        setup = self.prepare_baseline()
        original_write_text = Path.write_text
        def fail_summary(path, *args, **kwargs):
            if path.name in ("summary.json", "summary.json.tmp"):
                raise OSError("injected disk full")
            return original_write_text(path, *args, **kwargs)
        with heartbeat(self.runner), patch.object(Path, "write_text", new=fail_summary):
            self.runner.start_baseline(setup, demo=True)
            self.runner.worker.join(timeout=3)
        self.assertFalse(self.runner.worker.is_alive())
        self.assertFalse(self.runner.busy)
        self.assertTrue(self.runner.history)
        self.assertEqual(self.runner.completed["baseline_no_camera"], 0)
        self.assertIn("保存汇总失败", self.runner.message)
        self.assertIn("injected disk full", self.runner.message)
        self.assertTrue(self.runner.message.startswith("中止"))
        self.assertEqual(self.robot.speeds, [0] * 6)
        self.assertEqual(list(self.root.glob("demo/*/summary.json")), [])

    def test_camera_free_summary_commit_failure_is_not_a_completed_trial(self):
        setup = self.prepare_baseline()
        original_replace = Path.replace
        def fail_summary_commit(path, target):
            if path.name == "summary.json.tmp":
                raise OSError("injected rename failure")
            return original_replace(path, target)
        with heartbeat(self.runner), patch.object(Path, "replace", new=fail_summary_commit):
            self.runner.start_baseline(setup, demo=True)
            self.runner.worker.join(timeout=3)
        self.assertFalse(self.runner.worker.is_alive())
        self.assertFalse(self.runner.busy)
        self.assertEqual(self.runner.completed["baseline_no_camera"], 0)
        self.assertIn("保存汇总失败", self.runner.message)
        self.assertTrue(self.runner.message.startswith("中止"))
        self.assertEqual(self.robot.speeds, [0] * 6)
        self.assertEqual(list(self.root.glob("demo/*/summary.json")), [])

    def test_camera_free_failed_stop_write_is_not_completion(self):
        setup = self.prepare_baseline()
        def failed_stop():
            self.robot.speeds = [0] * 6
            return False
        with patch.object(self.robot, "stop", side_effect=failed_stop):
            result = self.run_baseline(setup)
        self.assertEqual(result["outcome"], "aborted")
        self.assertFalse(result["stop_write_succeeded"])
        self.assertEqual(self.runner.completed["baseline_no_camera"], 0)

    def test_baseline_command_preview_varied_speeds_nonconsecutive_mapping(self):
        setup = self.prepare_baseline(channels=[6, 2, 5, 1], signs=[-1, 1, 1, -1])
        speeds = [40, 80, 120, 160]
        self.assertEqual(baseline_commands(setup, speeds), [-160, 80, 0, 0, 120, -40])
        self.assertEqual(baseline_commands(setup, speeds, only_joint=2), [0, 0, 0, 0, 120, 0])
        self.assertEqual(baseline_commands(setup, [0, 0, 0, 0]), [0] * 6)
        self.assertEqual(self.robot.sends, [])

    def test_group_varied_speeds_persist_commands_and_count_as_group(self):
        setup = self.prepare_baseline(channels=[6, 2, 5, 1], signs=[-1, 1, 1, -1])
        speeds = [40, 80, 120, 160]
        expected = [-160, 80, 0, 0, 120, -40]
        result = self.run_baseline(setup, joint_speeds=speeds)
        self.assertEqual(result["outcome"], "completed", result)
        self.assertEqual(result["mode"], "baseline_no_camera")
        self.assertEqual(result["joint_speeds"], speeds)
        self.assertIsNone(result["only_joint"])
        self.assertEqual(result["full_commands"], expected)
        self.assertTrue(all(commands == expected for _, commands in self.robot.sends))
        self.assertEqual(self.runner.completed["baseline_no_camera"], 1)
        self.assertEqual(self.runner.completed["single_motor"], 0)

    def test_single_motor_only_one_online_logs_missing_measurements_and_separate_mode(self):
        setup = self.prepare_baseline(channels=[6, 2, 5, 1])
        speeds = [40, 80, 120, 160]
        original = self.robot.snapshot
        def only_selected_online():
            return {5: original()[5]}
        with patch.object(self.robot, "snapshot", side_effect=only_selected_online):
            result = self.run_baseline(setup, joint_speeds=speeds, only_joint=2)
        expected = [0, 0, 0, 0, -120, 0]
        self.assertEqual(result["outcome"], "completed", result)
        self.assertEqual(result["mode"], "single_motor")
        self.assertEqual(result["only_joint"], 2)
        self.assertEqual(result["full_commands"], expected)
        self.assertEqual(result["joint_speeds"], speeds)
        self.assertEqual(self.runner.completed["single_motor"], 1)
        self.assertEqual(self.runner.completed["baseline_no_camera"], 0)
        self.assertTrue(all(commands == expected for _, commands in self.robot.sends))
        self.assertEqual(self.runner.latest["j2_rpm"], -120)
        for j in (0, 1, 3):
            self.assertIsNone(self.runner.latest[f"j{j}_rpm"])
            self.assertIsNone(self.runner.latest[f"j{j}_current_raw"])
            self.assertIsNone(self.runner.latest[f"j{j}_last_seen_s"])
            self.assertFalse(self.runner.latest[f"j{j}_active"])
            self.assertEqual(self.runner.latest[f"j{j}_command"], 0)
        run_dir = next((self.root / "demo").iterdir())
        self.assertTrue(run_dir.name.endswith("_single_motor"))
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["full_commands"], expected)
        self.assertEqual(metadata["joint_speeds"], speeds)
        self.assertEqual(metadata["only_joint"], 2)
        with (run_dir / "steps.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["j0_rpm"], "")
        self.assertEqual(rows[0]["j0_current_raw"], "")
        self.assertEqual(json.loads(rows[0]["full_commands"]), expected)
        self.assertEqual(rows[0]["only_joint"], "2")
        self.assertEqual(self.robot.speeds, [0] * 6)
        self.assertEqual(self.runner.agent.total_updates, 0)

    def test_group_still_requires_all_four_channels_even_when_some_commands_zero(self):
        setup = self.prepare_baseline()
        data = self.robot.snapshot()
        with patch.object(self.robot, "snapshot", return_value={1: data[1]}):
            with self.assertRaises(ValueError):
                self.runner.start_baseline(setup, demo=True, joint_speeds=[80, 0, 0, 0])
        self.assertEqual(self.robot.sends, [])

    def test_single_motor_invalid_parameters_never_send(self):
        setup = self.prepare_baseline()
        cases = [([40, 50, 60], None), ([40, 50, 60, -1], None),
                 ([40, 50, 60, 501], None), ([40, 50, 60, 30.5], None),
                 ([40, 50, 60, True], None), (20, None),
                 ([40, 50, 60, 70], -1), ([40, 50, 60, 70], 4),
                 ([40, 50, 60, 70], 1.0), ([40, 50, 60, 70], True)]
        for speeds, only_joint in cases:
            with self.subTest(speeds=speeds, only_joint=only_joint):
                with self.assertRaises(ValueError):
                    baseline_commands(setup, speeds, only_joint)
                with self.assertRaises(ValueError):
                    self.runner.start_baseline(setup, demo=True, joint_speeds=speeds, only_joint=only_joint)
        self.assertFalse(self.runner.busy)
        self.assertEqual(self.robot.sends, [])

    def test_single_motor_overcurrent_with_other_channels_absent_retains_fault(self):
        setup = self.prepare_baseline(channels=[6, 2, 5, 1])
        original = self.robot.snapshot
        def only_selected_online():
            motor = original()[5]
            if self.robot.sends:
                motor["current_raw"] = -setup.current_limit_raw
            return {5: motor, 6: None}
        with patch.object(self.robot, "snapshot", side_effect=only_selected_online):
            result = self.run_baseline(setup, joint_speeds=[40, 50, 60, 70], only_joint=2)
        self.assertEqual(result["outcome"], "aborted")
        self.assertEqual(result["mode"], "single_motor")
        self.assertEqual(result["safety_fault"]["channel"], 5)
        self.assertEqual(result["peak_abs_current_raw"], setup.current_limit_raw)
        self.assertIsNone(result["safety_fault"]["telemetry"]["6"]["speed"])
        self.assertEqual(self.runner.completed["single_motor"], 0)
        self.assertEqual(self.robot.speeds, [0] * 6)

    def test_single_motor_stop_rejects_queued_route_change(self):
        setup = self.prepare_baseline()
        with heartbeat(self.runner):
            self.runner.start_baseline(setup, demo=True, joint_speeds=[40, 50, 60, 70], only_joint=1)
            deadline = time.monotonic() + 2
            while not self.runner.history and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertTrue(self.runner.history)
            with self.assertRaises(ValueError):
                self.runner.start_baseline(setup, demo=True, joint_speeds=[40, 50, 60, 70], only_joint=3)
            self.runner.abort("停止单路调试")
            stopped_at = time.monotonic()
            result = self.finish()
        self.assertEqual(result["outcome"], "aborted")
        self.assertTrue(all(commands == [0, 50, 0, 0, 0, 0] for _, commands in self.robot.sends))
        self.assertTrue(all(sent_at <= stopped_at for sent_at, _ in self.robot.sends))
        self.assertEqual(self.runner.completed["single_motor"], 0)
        self.assertEqual(self.runner.completed["baseline_no_camera"], 0)
        self.assertEqual(self.robot.speeds, [0] * 6)

    def test_slow_csv_flush_does_not_delay_absolute_control_periods(self):
        setup = self.prepare_baseline(duration_s=1.2, control_hz=10)
        stalled = threading.Event()
        def slow_flush():
            stalled.set()
            time.sleep(0.35)
        with intercept_csv_flush(slow_flush):
            result = self.run_baseline(setup, only_joint=0)
        self.assertTrue(stalled.is_set())
        self.assertEqual(result["outcome"], "completed", result)
        self.assertTrue(result["csv_complete"])
        self.assertEqual(result["csv_log"]["written_rows"], result["steps"])
        times = [sent for sent, _ in self.robot.sends]
        self.assertGreaterEqual(len(times), 10)
        self.assertLess(max(b - a for a, b in zip(times, times[1:])), 0.3)
        self.assertLess(abs((times[-1] - times[0]) - (len(times) - 1) / setup.control_hz), 0.15)
        self.assertEqual(self.robot.speeds, [0] * 6)

    def test_async_csv_disk_error_stops_and_never_counts_completion(self):
        setup = self.prepare_baseline(duration_s=0.8, control_hz=10)
        def failing_flush():
            raise OSError("injected asynchronous disk failure")
        with intercept_csv_flush(failing_flush):
            result = self.run_baseline(setup, only_joint=0)
        self.assertEqual(result["outcome"], "aborted")
        self.assertIn("injected asynchronous disk failure", result["reason"])
        self.assertFalse(result["csv_complete"])
        self.assertEqual(self.runner.completed["single_motor"], 0)
        self.assertEqual(self.robot.speeds, [0] * 6)

    def test_async_csv_queue_full_stops_before_waiting_for_writer(self):
        setup = self.prepare_baseline(duration_s=1.0, control_hz=10)
        self.runner._csv_queue_size = 1
        self.runner._csv_finalize_timeout_s = 0.1
        release = threading.Event()
        exited = threading.Event()
        def blocked_flush():
            try:
                release.wait(timeout=3)
            finally:
                exited.set()
        try:
            with intercept_csv_flush(blocked_flush):
                result = self.run_baseline(setup, only_joint=0)
            self.assertEqual(result["outcome"], "aborted")
            self.assertIn("队列已满", result["reason"])
            self.assertFalse(result["csv_complete"])
            self.assertEqual(self.runner.completed["single_motor"], 0)
            self.assertEqual(self.robot.speeds, [0] * 6)
            self.assertFalse(exited.is_set(), "STOP/final result waited indefinitely for disk")
        finally:
            release.set()
            self.assertTrue(exited.wait(timeout=1))
            # Allow the logging thread to drain its one queued row and close.
            time.sleep(0.03)

    def test_async_csv_finalize_timeout_stops_and_reports_incomplete_trial(self):
        setup = self.prepare_baseline(duration_s=0.5, control_hz=10)
        self.runner._csv_finalize_timeout_s = 0.1
        release = threading.Event()
        exited = threading.Event()
        def blocked_flush():
            try:
                release.wait(timeout=3)
            finally:
                exited.set()
        try:
            with intercept_csv_flush(blocked_flush):
                result = self.run_baseline(setup, only_joint=0)
            self.assertEqual(result["outcome"], "aborted")
            self.assertIn("收尾超时", result["reason"])
            self.assertFalse(result["csv_complete"])
            self.assertEqual(self.runner.completed["single_motor"], 0)
            self.assertEqual(self.robot.speeds, [0] * 6)
            self.assertFalse(exited.is_set())
            self.assertLess(result["duration_wall_s"], 1.0)
        finally:
            release.set()
            self.assertTrue(exited.wait(timeout=1))
            time.sleep(0.03)

    def test_abort_during_log_finalize_keeps_stop_and_discards_completion(self):
        setup = self.prepare_baseline(duration_s=0.5, control_hz=10)
        release = threading.Event()
        first_flush = threading.Event()
        def blocked_flush():
            first_flush.set()
            release.wait(timeout=2)
        try:
            with heartbeat(self.runner), intercept_csv_flush(blocked_flush):
                self.runner.start_baseline(setup, demo=True, only_joint=0)
                self.assertTrue(first_flush.wait(timeout=1))
                deadline = time.monotonic() + 1
                while self.robot.speeds != [0] * 6 and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.assertEqual(self.robot.speeds, [0] * 6)
                self.assertTrue(self.runner.busy)
                self.runner.abort("日志收尾期间取消")
                stopped_at = time.monotonic()
                release.set()
                result = self.finish()
            self.assertEqual(result["outcome"], "aborted")
            self.assertIn("日志收尾期间取消", result["reason"])
            self.assertEqual(self.runner.completed["single_motor"], 0)
            self.assertTrue(all(sent <= stopped_at for sent, _ in self.robot.sends))
        finally:
            release.set()

    def test_missed_full_control_period_aborts_without_catchup_send(self):
        setup = self.prepare_baseline(duration_s=0.8, control_hz=10)
        original_snapshot = self.robot.snapshot
        delayed = False
        def slow_telemetry():
            nonlocal delayed
            if self.robot.sends and not delayed:
                delayed = True
                time.sleep(0.15)
            return original_snapshot()
        with patch.object(self.robot, "snapshot", side_effect=slow_telemetry):
            result = self.run_baseline(setup, only_joint=0)
        self.assertEqual(result["outcome"], "aborted")
        self.assertIn("控制周期超时", result["reason"])
        self.assertEqual(len(self.robot.sends), 1)
        self.assertEqual(self.runner.completed["single_motor"], 0)
        self.assertEqual(self.robot.speeds, [0] * 6)

    def test_transport_evidence_preserved_before_stop_and_diagnostic_errors_do_not_block_stop(self):
        setup = self.prepare_baseline()
        original_stop = self.robot.stop
        def stop():
            self.robot.status = "stopped status"
            return original_stop()
        self.robot.status = "live status"
        with patch.object(self.robot, "diagnostics", create=True, return_value={"lease_fault": False}), \
                patch.object(self.robot, "stop", side_effect=stop):
            result = self.run_baseline(setup, only_joint=0)
        self.assertEqual(result["outcome"], "completed", result)
        self.assertEqual(result["transport_before_stop"]["diagnostics"], {"lease_fault": False})
        self.assertEqual(result["transport_diagnostics"]["status"], "stopped status")
        with patch.object(self.robot, "diagnostics", create=True, side_effect=RuntimeError("diagnostic unavailable")):
            result = self.run_baseline(setup, only_joint=0)
        self.assertEqual(result["outcome"], "completed", result)
        self.assertIn("diagnostic unavailable", result["transport_diagnostics"]["diagnostics_error"])
        self.assertEqual(self.robot.speeds, [0] * 6)


if __name__ == "__main__":
    unittest.main()
