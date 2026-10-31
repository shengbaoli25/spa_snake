"""No physical camera or robot access: geometry, validity, loopback and CSV."""
import csv
import json
import socket
import tempfile
import time
import unittest
from pathlib import Path

from spa_control.camera import (
    CameraSample, CausalSpeedEstimator, CsvSpeedReceiver, RgbPositionProcessor,
    StraightTrackCalibration, TrackCalibrationPoint, UdpSpeedReceiver,
    bbox_reference_point, parse_speed_payload,
)


class CameraTests(unittest.TestCase):
    def test_piecewise_track_retains_existing_geometry(self):
        calibration = StraightTrackCalibration([
            TrackCalibrationPoint(100, 300, 0),
            TrackCalibrationPoint(220, 280, .1),
            TrackCalibrationPoint(400, 250, .2),
        ])
        self.assertAlmostEqual(calibration.project(310, 265).position_m, .15)
        self.assertAlmostEqual(calibration.project(310, 265).distance_from_track_px, 0)
        self.assertAlmostEqual(calibration.project(580, 220).position_m, .3)

    def test_constant_velocity_and_nonmonotonic_rejection(self):
        estimator = CausalSpeedEstimator(.3, 0)
        for i in range(31):
            estimate = estimator.add(i/30, .15+.08*i/30)
        self.assertAlmostEqual(estimate.velocity_mps, .08, places=10)
        with self.assertRaises(ValueError):
            estimator.add(1, 1)

    def test_rgb_resolution_warmup_direction_and_gap(self):
        calibration = StraightTrackCalibration([
            TrackCalibrationPoint(100, 0, 0), TrackCalibrationPoint(0, 0, 1),
        ], {"frame_width": 100, "frame_height": 50})
        processor = RgbPositionProcessor(calibration, position_time_constant_s=0)
        first = processor.add_bbox(0, (100, 0, 10, 10), (100, 50))
        self.assertFalse(first.tracking_ok)
        self.assertIsNone(first.velocity_mps)
        for i in range(1, 11):
            sample = processor.add_bbox(i*.05, (100-i, 0, 10, 10), (100, 50))
        self.assertTrue(sample.tracking_ok)
        self.assertAlmostEqual(sample.velocity_mps, .2)
        self.assertEqual(bbox_reference_point((10, 20, 30, 40), "right"), (40, 40))
        with self.assertRaises(ValueError):
            processor.add_bbox(.6, (90, 0, 10, 10), (200, 100))
        self.assertFalse(processor.add_bbox(2, (80, 0, 10, 10), (100, 50)).tracking_ok)

    def test_fresh_arrival_cannot_make_old_measurement_live(self):
        now = time.monotonic()
        sample = CameraSample(now-2, now, 1, .1, True, "test")
        self.assertFalse(sample.is_fresh(now=now))
        sample = CameraSample(now+.1, now, 1, .1, True, "test")
        self.assertFalse(sample.is_fresh(now=now))

    def test_lost_data_never_becomes_zero_speed_reward(self):
        sample = parse_speed_payload({"timestamp_monotonic_s": 1, "tracking_ok": "0"})
        self.assertFalse(sample.tracking_ok)
        self.assertIsNone(sample.velocity_mps)
        with self.assertRaises(ValueError):
            parse_speed_payload({"timestamp_monotonic_s": 1, "tracking_ok": True,
                                 "position_m": 0, "velocity_mps": float("nan")})

    @staticmethod
    def wait_until(predicate, seconds=1):
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            time.sleep(.01)
        raise AssertionError("receiver did not produce expected result")

    def test_udp_rejects_old_packets_and_receives_local_measurement(self):
        source = UdpSpeedReceiver(port=0)
        source.start()
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            packet = {"timestamp_monotonic_s": time.monotonic()-10,
                      "position_m": .3, "velocity_mps": .06, "tracking_ok": True}
            sender.sendto(json.dumps(packet).encode(), ("127.0.0.1", source.port))
            self.wait_until(lambda: "拒绝" in source.status)
            self.assertIsNone(source.latest())
            packet["timestamp_monotonic_s"] = time.monotonic()
            sender.sendto(json.dumps(packet).encode(), ("127.0.0.1", source.port))
            sample = self.wait_until(source.latest)
            self.assertAlmostEqual(sample.velocity_mps, .06)
            # Windows monotonic clock may return the same tick for consecutive
            # calls. A genuinely new frame must have a strictly newer timestamp.
            self.wait_until(lambda: time.monotonic() > sample.timestamp_s)
            packet["timestamp_monotonic_s"] = time.monotonic()
            packet["tracking_ok"] = False
            sender.sendto(json.dumps(packet).encode(), ("127.0.0.1", source.port))
            self.wait_until(lambda: (current := source.latest()) is not None and not current.tracking_ok)
            self.assertIsNone(source.latest().velocity_mps)
        finally:
            sender.close()
            source.stop()
        self.assertIsNone(source.latest())

    def test_existing_gui_csv_tail_does_not_replay_old_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"camera.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["timestamp_monotonic_s", "filtered_position_m", "velocity_mps", "tracking_ok"])
                writer.writerow([time.monotonic()-10, .1, .5, 1])
                file.flush()
                source = CsvSpeedReceiver(path)
                source.start()
                try:
                    self.wait_until(lambda: "正常" in source.status)
                    self.assertIsNone(source.latest())
                    writer.writerow([time.monotonic(), .2, .07, 1])
                    file.flush()
                    sample = self.wait_until(source.latest)
                    self.assertAlmostEqual(sample.velocity_mps, .07)
                    writer.writerow([time.monotonic(), "", "", 0])
                    file.flush()
                    self.wait_until(lambda: source.latest() and not source.latest().tracking_ok)
                    self.assertIsNone(source.latest().velocity_mps)
                finally:
                    source.stop()


if __name__ == "__main__":
    unittest.main()
