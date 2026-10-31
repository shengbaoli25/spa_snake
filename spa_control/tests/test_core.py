"""Physical-unit and command-projection contracts, without hardware."""
from dataclasses import replace
import math
import unittest

import numpy as np

from spa_control.core import ExperimentConfig, command_line, observation, speed_reward


def config(**overrides):
    return replace(ExperimentConfig(parameters_confirmed=True, rotor_to_helix_ratio=20,
                                    rpm_per_command=10, lead_mm=60, signs=[1, -1, 1, -1],
                                    nominal_command=40, command_limit=120), **overrides)


class CoreTests(unittest.TestCase):
    def test_default_forward_signs_match_existing_monitor(self):
        # Verified against snake_robot_monitor.py MOTOR_FORWARD_DIR for M1-M4;
        # keep this portable without importing a GUI from the user's Desktop.
        monitor_forward_direction = {1: -1, 2: 1, 3: -1, 4: 1}
        setup = ExperimentConfig()
        self.assertEqual(setup.signs, [monitor_forward_direction[channel] for channel in setup.channels])
        self.assertEqual(setup.nominal_command, 300)
        self.assertEqual(setup.command_limit, 500)
        self.assertEqual(setup.project([0] * 4)[0], [-300, 300, -300, 300, 0, 0])

    def test_real_motion_requires_explicit_physical_calibration(self):
        with self.assertRaises(ValueError):
            ExperimentConfig().validate(physical=True)
        with self.assertRaises(ValueError):
            replace(config(), rpm_per_command=None).validate(physical=True)
        config().validate(physical=True)

    def test_alternating_forward_and_arbitrary_structural_mapping(self):
        setup = config(channels=[6, 1, 4, 2])
        setup.validate(physical=True)
        commands, residuals = setup.project([0, 0, 0, 0])
        self.assertEqual(commands, [-40, -40, 0, 40, 0, 40])
        self.assertEqual(residuals, [0, 0, 0, 0])
        self.assertEqual(command_line(commands), b"CTRL SPEED M1=-40 M2=-40 M3=0 M4=40 M5=0 M6=40\n")

    def test_residual_clipping_and_direction_are_independent(self):
        commands, applied = config().project([5, -5, 0.2, -0.2])
        self.assertEqual(commands, [45, -35, 41, -39, 0, 0])
        self.assertEqual(applied, [5, -5, 1, -1])
        commands, applied = config(nominal_command=2).project([-1, -1, 1, 1])
        self.assertEqual(commands, [0, 0, 7, -7, 0, 0])
        self.assertEqual(applied, [-2, -2, 5, 5])
        commands, applied = config(nominal_command=118, command_limit=120).project([1] * 4)
        self.assertEqual(commands, [120, -120, 120, -120, 0, 0])
        self.assertEqual(applied, [2] * 4)

    def test_quantization_cannot_expand_declared_residual_bound(self):
        setup = config(nominal_command=40.4)
        # A fixed integer bridge can either reject fractional nominal commands
        # or quantize explicitly without exceeding its reported residual bound.
        try:
            setup.validate(physical=True)
        except ValueError:
            return
        for action in ([1] * 4, [-1] * 4):
            _, applied = setup.project(action)
            self.assertTrue(all(abs(residual) <= setup.residual_limit for residual in applied))

    def test_fractional_residual_bound_is_preserved_after_integer_rounding(self):
        setup = config(residual_limit=0.6)
        try:
            setup.validate(physical=True)
        except ValueError:
            return
        for action in ([1] * 4, [-1] * 4):
            _, applied = setup.project(action)
            self.assertTrue(all(abs(residual) <= setup.residual_limit for residual in applied))

    def test_command_and_mapping_validation(self):
        for channels in ([1, 1, 2, 3], [0, 2, 3, 4], [1, 2, 3, 7], [1, 2, 3]):
            with self.assertRaises(ValueError):
                config(channels=channels).validate()
        for speeds in ([1, 2], [501, 0, 0, 0, 0, 0], [1.5, 0, 0, 0, 0, 0]):
            with self.assertRaises(ValueError):
                command_line(speeds)
        for action in ([0, 0], [0, 0, 0, np.nan], [0, 0, 0, np.inf]):
            with self.assertRaises(ValueError):
                config().project(action)

    def test_helix_geometry_and_rotor_gear_ratio_units(self):
        setup = config()
        self.assertAlmostEqual(setup.motor_rpm(40), 400)
        self.assertAlmostEqual(setup.theory_speed(40), 0.02)
        self.assertAlmostEqual(setup.target_speed(), 0.02)
        geometry = config(lead_mm=None, diameter_mm=100, lead_angle_deg=45)
        self.assertAlmostEqual(geometry.lead_m(), math.pi * 0.1)
        self.assertAlmostEqual(geometry.theory_speed(40), 400 / 20 / 60 * math.pi * 0.1)
        self.assertEqual(config(target_mps=0.1).target_speed(), 0.1)

    def test_empirical_rpm_calibration_interpolates_without_extrapolation(self):
        setup = config(rpm_per_command=None, calibration_commands=[0, 20, 60], calibration_rpm=[0, 150, 550])
        setup.validate(physical=True)
        self.assertEqual(setup.motor_rpm(40), 350)
        self.assertEqual(setup.motor_rpm(-40), 350)
        with self.assertRaises(ValueError):
            setup.motor_rpm(61)
        for xs, ys in (([0, 0], [0, 1]), ([0, 1], [1, 0]), ([0], [1]), ([0, 1], [0, np.nan])):
            with self.assertRaises(ValueError):
                config(calibration_commands=xs, calibration_rpm=ys).validate()

    def test_observation_structural_order_and_no_actual_camera_input(self):
        setup = config(channels=[4, 1, 6, 2], speed_scale_rpm=100, current_limit_raw=1000,
                       nominal_command=50, command_limit=100)
        telemetry = {channel: {"speed": channel * 10, "current_raw": channel * 20} for channel in range(1, 7)}
        result = observation(setup, telemetry, [1, 2, 3, 4])
        self.assertEqual(result.shape, (17,))
        np.testing.assert_allclose(result[:8], [.4, .08, .5, .2, -.1, .02, .5, .4])
        self.assertAlmostEqual(result[-1], setup.target_speed() / setup.reward_scale_mps)

    def test_reward_penalizes_underspeed_and_overspeed_symmetrically(self):
        setup = config(target_mps=0.1, reward_scale_mps=0.05)
        self.assertEqual(speed_reward(setup, 0.1, [0] * 4, [0] * 4), (0, 0, 0))
        low = speed_reward(setup, 0.08, [0] * 4, [0] * 4)
        high = speed_reward(setup, 0.12, [0] * 4, [0] * 4)
        self.assertAlmostEqual(low[0], 0.02)
        self.assertAlmostEqual(high[0], -0.02)
        self.assertAlmostEqual(low[1], high[1])
        self.assertAlmostEqual(low[2], -0.4)
        with self.assertRaises(ValueError):
            speed_reward(setup, np.nan, [0] * 4, [0] * 4)


if __name__ == "__main__":
    unittest.main()
