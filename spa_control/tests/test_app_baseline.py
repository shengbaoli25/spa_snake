"""Headless baseline-form tests: no Tk window, network, camera or motor writes."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

from spa_control.app import DIRECTIONS, Workbench
from spa_control.core import ExperimentConfig
from spa_control.session import baseline_commands


class Value:
    """Only the StringVar get/set contract used by form callbacks."""

    def __init__(self, value):
        self.value = str(value)

    def get(self):
        return self.value

    def set(self, value):
        self.value = str(value)


class BaselineFormTests(unittest.TestCase):
    def setUp(self):
        self.network_guard = patch("spa_control.app.RobotLink.connect",
                                   side_effect=AssertionError("No hardware connection is allowed"))
        self.network_guard.start()
        self.addCleanup(self.network_guard.stop)
        self.dialog_patch = patch("spa_control.app.messagebox.showerror")
        self.error = self.dialog_patch.start()
        self.addCleanup(self.dialog_patch.stop)

    def make_form(self):
        config = ExperimentConfig()
        fields = ("host", "port", "nominal_command", "duration_s", "control_hz",
                  "command_limit", "channels", "current_limit_raw", "sensor_timeout_s")
        variables = {}
        for key in fields:
            value = getattr(config, key)
            variables[key] = Value(",".join(map(str, value)) if isinstance(value, list) else value)
        app = SimpleNamespace(
            vars=variables, motor_speeds=[Value(300) for _ in range(4)],
            motor_directions=[Value("正转 (+)" if sign > 0 else "反转 (−)") for sign in config.signs],
            single_duration=Value(3), demo=True, pending=False, closing=False,
            active_joint=None, command_preview=Value(""), preset_status=Value(""),
            runner=SimpleNamespace(busy=False, message="", link=SimpleNamespace(config=config)),
        )
        # Validate exactly as the session entry point does, but record only a
        # completed request. No real runner, thread, socket or command is used.
        app.started = []

        def mock_start(candidate, demo=False, joint_speeds=None, only_joint=None):
            commands = baseline_commands(candidate, joint_speeds, only_joint)
            app.started.append({"config": copy.deepcopy(candidate), "demo": demo,
                                "speeds": list(joint_speeds), "only_joint": only_joint,
                                "commands": commands})

        app.runner.start_baseline = Mock(side_effect=mock_start)
        app.runner.abort = Mock()
        for name in ("get_config", "get_joint_speeds", "preview", "apply_forward_preset",
                     "apply_common_speed", "save_config", "load_config", "_start_motion",
                     "start_single", "start_trial", "stop_trial", "stop_single"):
            setattr(app, name, MethodType(getattr(Workbench, name), app))
        return app

    @staticmethod
    def snapshot(app):
        return ({key: value.get() for key, value in app.vars.items()},
                [value.get() for value in app.motor_speeds],
                [value.get() for value in app.motor_directions])

    def write_load(self, app, payload):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            with patch("spa_control.app.filedialog.askopenfilename", return_value=str(path)):
                app.load_config()

    def configure_custom(self, app):
        app.vars["channels"].set("6,2,5,1")
        for variable, speed in zip(app.motor_speeds, [101, 202, 303, 404]):
            variable.set(speed)
        for variable, direction in zip(app.motor_directions,
                                       ["正转 (+)", "反转 (−)", "反转 (−)", "正转 (+)"]):
            variable.set(direction)

    def test_save_reload_preserves_four_independent_speeds_signs_and_mapping(self):
        app = self.make_form()
        self.configure_custom(app)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom.json"
            with patch("spa_control.app.filedialog.asksaveasfilename", return_value=str(path)):
                app.save_config()
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["baseline_joint_speeds"], [101, 202, 303, 404])
            self.assertEqual(data["channels"], [6, 2, 5, 1])
            self.assertEqual(data["signs"], [1, -1, -1, 1])
            self.assertEqual(data["residual_limit"], 0)
            restored = self.make_form()
            with patch("spa_control.app.filedialog.askopenfilename", return_value=str(path)):
                restored.load_config()
            self.assertEqual(restored.get_joint_speeds(), [101, 202, 303, 404])
            self.assertEqual(restored.get_config().channels, [6, 2, 5, 1])
            self.assertEqual(restored.get_config().signs, [1, -1, -1, 1])
            self.assertEqual(baseline_commands(restored.get_config(), restored.get_joint_speeds()),
                             [404, -202, 0, 0, -303, 101])
            restored.runner.start_baseline.assert_not_called()
        app.runner.start_baseline.assert_not_called()
        self.error.assert_not_called()

    def test_legacy_plain_experiment_config_uses_its_common_speed(self):
        app = self.make_form()
        self.configure_custom(app)
        config = ExperimentConfig(nominal_command=137, channels=[5, 3, 1, 6], signs=[1, 1, -1, -1])
        self.write_load(app, config.as_dict())
        self.assertEqual(app.get_joint_speeds(), [137] * 4)
        self.assertEqual(app.get_config().channels, [5, 3, 1, 6])
        self.assertEqual(app.get_config().signs, [1, 1, -1, -1])
        self.assertEqual(app.get_config().residual_limit, 0)
        app.runner.start_baseline.assert_not_called()
        self.error.assert_not_called()

    def test_invalid_file_never_partially_overwrites_current_form(self):
        app = self.make_form()
        self.configure_custom(app)
        before = self.snapshot(app)
        cases = [
            {"channels": [1, 1, 2, 3]}, {"signs": [1, 0, -1, 1]},
            {"baseline_joint_speeds": [10, 20, 30]},
            {"baseline_joint_speeds": [10, 20, 30, 501]},
            {"baseline_joint_speeds": [10, 20, 30, -1]},
            {"baseline_joint_speeds": [10, 20, 30, 40.5]},
            {"baseline_joint_speeds": [10, 20, 30, True]},
            {"duration_s": 0}, {"unexpected_field": "invalid"},
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                payload = ExperimentConfig().as_dict()
                payload.update(fields)
                self.error.reset_mock()
                self.write_load(app, payload)
                self.assertEqual(self.snapshot(app), before)
                self.error.assert_called_once()
                app.runner.start_baseline.assert_not_called()

    def test_malformed_json_load_is_atomic_and_does_not_start(self):
        app = self.make_form()
        before = self.snapshot(app)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_text('{"channels":', encoding="utf-8")
            with patch("spa_control.app.filedialog.askopenfilename", return_value=str(path)):
                app.load_config()
        self.assertEqual(self.snapshot(app), before)
        self.error.assert_called_once()
        app.runner.start_baseline.assert_not_called()

    def test_invalid_form_cannot_be_saved(self):
        app = self.make_form()
        app.motor_speeds[3].set("501")
        with patch("spa_control.app.filedialog.asksaveasfilename") as dialog:
            app.save_config()
        dialog.assert_not_called()
        self.error.assert_called_once()

    def test_single_run_uses_structural_index_mapping_and_independent_duration(self):
        app = self.make_form()
        self.configure_custom(app)
        app.single_duration.set("2.5")
        app.start_single(2)
        self.assertEqual(len(app.started), 1)
        request = app.started[0]
        self.assertEqual(request["only_joint"], 2)
        self.assertEqual(request["commands"], [0, 0, 0, 0, -303, 0])
        self.assertEqual(request["config"].duration_s, 2.5)
        self.assertEqual(app.vars["duration_s"].get(), "12.0")
        self.assertEqual(app.active_joint, 2)
        self.assertTrue(request["demo"])

    def test_four_motor_run_passes_all_independent_magnitudes_and_signs(self):
        app = self.make_form()
        self.configure_custom(app)
        app.start_trial()
        self.assertEqual(len(app.started), 1)
        self.assertIsNone(app.started[0]["only_joint"])
        self.assertEqual(app.started[0]["commands"], [404, -202, 0, 0, -303, 101])
        self.assertEqual(app.started[0]["config"].duration_s, 12)

    def test_invalid_input_produces_no_motion_request(self):
        cases = [("speed", "-1"), ("speed", "501"), ("speed", "1.5"), ("speed", "nan"),
                 ("duration", "0"), ("channels", "1,2,2,4"), ("direction", "invalid")]
        for kind, value in cases:
            with self.subTest(kind=kind, value=value):
                app = self.make_form()
                self.error.reset_mock()
                if kind == "speed":
                    app.motor_speeds[1].set(value)
                elif kind == "duration":
                    app.single_duration.set(value)
                elif kind == "channels":
                    app.vars["channels"].set(value)
                else:
                    app.motor_directions[1].set(value)
                app.start_single(1)
                self.assertEqual(app.started, [])
                self.assertIsNone(app.active_joint)
                self.error.assert_called_once()

    def test_busy_pending_and_closing_block_start_callbacks(self):
        for guard in ("busy", "pending", "closing"):
            with self.subTest(guard=guard):
                app = self.make_form()
                setattr(app.runner if guard == "busy" else app, guard, True)
                app.start_single(1)
                app.start_trial()
                app.runner.start_baseline.assert_not_called()

    def test_direction_presets_preserve_speeds_and_handle_mapping_explicitly(self):
        app = self.make_form()
        self.configure_custom(app)
        app.apply_forward_preset(False)
        self.assertEqual(app.get_config().channels, [6, 2, 5, 1])
        self.assertEqual(app.get_config().signs, [-1, 1, -1, 1])
        self.assertEqual(app.get_joint_speeds(), [101, 202, 303, 404])
        app.apply_forward_preset(True)
        self.assertEqual(app.get_config().channels, [1, 2, 3, 4])
        self.assertEqual(app.get_config().signs, [-1, 1, 1, 1])
        self.assertEqual(app.get_joint_speeds(), [101, 202, 303, 404])
        app.runner.start_baseline.assert_not_called()

    def test_common_speed_application_validates_before_mutation(self):
        app = self.make_form()
        self.configure_custom(app)
        app.vars["nominal_command"].set("175")
        app.apply_common_speed()
        self.assertEqual(app.get_joint_speeds(), [175] * 4)
        self.assertEqual(app.get_config().signs, [1, -1, -1, 1])
        for invalid in ("501", "-1", "17.5", "nan"):
            app.vars["nominal_command"].set(invalid)
            before = self.snapshot(app)
            self.error.reset_mock()
            app.apply_common_speed()
            self.assertEqual(self.snapshot(app), before)
            self.error.assert_called_once()
        app.runner.start_baseline.assert_not_called()

    def test_stop_single_only_aborts_matching_active_joint(self):
        app = self.make_form()
        app.runner.busy = True
        app.active_joint = 2
        app.stop_single(1)
        app.runner.abort.assert_not_called()
        app.stop_single(2)
        app.runner.abort.assert_called_once_with("单电机测试停止")


if __name__ == "__main__":
    unittest.main()
