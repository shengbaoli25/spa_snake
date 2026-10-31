import math
import unittest

from spa_control.kinematics import calculate_reference, speed_tracking_reward


class KinematicsTest(unittest.TestCase):
    def test_known_ideal_reference(self):
        r = calculate_reference(100, 30, 1, distance_m=1)
        self.assertAlmostEqual(r['theory_speed_mps'], .05)
        self.assertAlmostEqual(r['ideal_duration_s'], 20)

    def test_direction_ratio_and_zero_speed(self):
        r = calculate_reference(-100, 30, 2, forward_sign=-1, distance_m=1)
        self.assertAlmostEqual(r['theory_speed_mps'], .025)
        self.assertAlmostEqual(r['ideal_duration_s'], 40)
        self.assertIsNone(calculate_reference(0, 30, 1, distance_m=1)['ideal_duration_s'])

    def test_invalid_inputs(self):
        for args in ((501, 30, 1), (100, 0, 1), (100, 30, 0), (math.nan, 30, 1)):
            with self.assertRaises(ValueError):
                calculate_reference(*args)
        with self.assertRaises(ValueError):
            calculate_reference(100, 30, 1, distance_m=-1)
        with self.assertRaises(ValueError):
            calculate_reference(100, 30, 1, forward_sign=True)

    def test_tracking_symmetry_and_missing(self):
        self.assertEqual(speed_tracking_reward(.05, .05)['tracking_reward'], 1)
        self.assertAlmostEqual(speed_tracking_reward(.05, .04)['tracking_reward'],
                               speed_tracking_reward(.05, .06)['tracking_reward'])
        for measured in (None, math.nan, math.inf):
            with self.assertRaises(ValueError):
                speed_tracking_reward(.05, measured)


if __name__ == '__main__':
    unittest.main()
