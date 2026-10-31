"""Passive recorder tests: fake sources only; no serial or motor writes."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from spa_control.two_motor_record import TwoMotorRecorder


class FakeMotor:
    def __init__(self):
        self.calls = 0

    def status(self):
        self.calls += 1
        return {"state": "connected", "active": True, "frame_fresh": True,
                "fault": "NONE", "can_ready": True, "can_healthy": True,
                "snapshot_monotonic_s": time.monotonic(),
                "frame_receive_monotonic_s": time.monotonic(),
                "directions": {"3": -1, "4": -1},
                "motors": {"3": {"id": 0x207, "seen": 1, "fresh": True,
                                  "spd": 3, "torque": 21, "age_ms": 5,
                                  "feedback_estimated_monotonic_s": time.monotonic()},
                           "4": {"id": 0x208, "seen": 1, "fresh": False,
                                  "spd": float("nan"), "age_ms": 200}}}


class FakeCamera:
    def __init__(self, available=True):
        self.available = available
        self.received = time.monotonic_ns()

    def status(self):
        return {"active": self.available}

    def latest_frame(self):
        return {"jpeg": b"\xff\xd8jpeg\xff\xd9", "boot_id": 4, "sequence": 7,
                "capture_us": 123, "receive_monotonic_ns": self.received}


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.patch = patch("spa_control.two_motor_record.ROOT", Path(self.temp.name))
        self.patch.start()
        self.motor, self.camera = FakeMotor(), FakeCamera()
        self.recorder = TwoMotorRecorder(self.camera, self.motor)

    def tearDown(self):
        self.recorder.close()
        self.patch.stop()
        self.temp.cleanup()

    def wait(self, condition, timeout=4):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(condition())

    def read(self, folder):
        path = Path(folder)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf8"))
        lines = (path / "events.jsonl").read_text(encoding="utf8").splitlines()
        events = [json.loads(line, parse_constant=lambda v: self.fail("non-JSON constant " + v)) for line in lines]
        return path, manifest, events

    def test_passive_record_stop_and_single_image_per_frame(self):
        started = self.recorder.start(10)
        self.wait(lambda: self.recorder.status()["samples"] >= 2)
        with self.assertRaises(ValueError):
            self.recorder.start(2)
        self.recorder.stop()
        self.wait(lambda: not self.recorder.status()["active"])
        path, manifest, events = self.read(started["directory"])
        self.assertEqual(manifest["state"], "stopped")
        self.assertEqual(manifest["images"], 1)
        self.assertEqual(len(list((path / "images").glob("*.jpg"))), 1)
        self.assertGreaterEqual(len(events), 2)
        sample = events[0]
        self.assertGreater(sample["host_monotonic_ns"], 0)
        self.assertEqual(sample["motors"]["M3"]["feedback"]["id"], 0x207)
        self.assertTrue(sample["motors"]["M3"]["available"])
        self.assertFalse(sample["motors"]["M4"]["available"])
        self.assertIsNone(sample["motors"]["M4"]["feedback"]["spd"])
        self.assertEqual(sample["camera"]["sequence"], 7)
        self.assertEqual((path / sample["camera"]["image"]).read_bytes(), b"\xff\xd8jpeg\xff\xd9")
        self.assertEqual(manifest["front_rear_mapping"], "unknown")
        self.assertEqual(manifest["clock_sync"], "not_calibrated")
        self.assertFalse(manifest["motor_commands_sent"])
        self.assertEqual(manifest["ground_speed"], "unavailable")
        self.assertEqual(manifest["reward"], "unavailable")
        self.assertNotIn("reward", sample)

    def test_timeout_and_missing_camera(self):
        self.camera.available = False
        started = self.recorder.start(2)
        self.wait(lambda: not self.recorder.status()["active"], timeout=3.5)
        _, manifest, events = self.read(started["directory"])
        self.assertEqual(manifest["state"], "completed")
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(manifest["images"], 0)
        self.assertFalse(events[0]["camera"]["available"])
        self.assertEqual(events[0]["camera"]["reason"], "missing_or_stale")

    def test_input_validation_and_close(self):
        for invalid in (True, None, -1, 1.99, 30.01, float("nan")):
            with self.assertRaises(ValueError):
                self.recorder.start(invalid)
        started = self.recorder.start(2)
        self.recorder.close()
        self.assertFalse(self.recorder.status()["active"])
        self.assertTrue(Path(started["directory"]).exists())
        with self.assertRaises(RuntimeError):
            self.recorder.start(2)

    def test_optional_global_speed_recorded_only_if_fresh_and_valid(self):
        class FakeGlobal:
            def __init__(self):
                self.valid = True
                self.stale = False
            def status(self):
                stamp = time.monotonic() - (2 if self.stale else 0)
                return {"state": "connected", "valid": self.valid,
                        "position_m": .35, "velocity_mps": .12,
                        "sample_timestamp_monotonic_s": stamp,
                        "direction": "left", "points": [{"position_m": 0}], "note": "ok"}
        source = FakeGlobal()
        self.recorder = TwoMotorRecorder(self.camera, self.motor, source)
        started = self.recorder.start(10)
        self.wait(lambda: self.recorder.status()["samples"] >= 2)
        source.stale = True
        self.wait(lambda: self.recorder.status()["samples"] >= 4)
        self.recorder.stop()
        self.wait(lambda: not self.recorder.status()["active"])
        _, manifest, events = self.read(started["directory"])
        good = next(e["global_camera"] for e in events if e["global_camera"]["valid"])
        bad = next(e["global_camera"] for e in events if not e["global_camera"]["valid"])
        self.assertEqual((good["position_m"], good["velocity_mps"]), (.35, .12))
        self.assertIsNotNone(good["sample_timestamp_monotonic_s"])
        self.assertIsNone(bad["position_m"])
        self.assertIsNone(bad["velocity_mps"])
        self.assertEqual(manifest["ground_speed"], "global_camera_if_valid")
        self.assertEqual(manifest["reward"], "unavailable")
        self.assertNotIn("reward", events[0])


if __name__ == "__main__":
    unittest.main()
