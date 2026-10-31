"""Guarded fault recovery: fake telemetry and ephemeral HTTP, never hardware."""
import http.client
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from spa_control.web_motor import MotorError, WebMotor
from spa_control.web_server import LocalServer, Workbench
from spa_control.tests.test_web_motor import FakePort, fixture
from spa_control.tests.test_web_control_api import Motor, Experiment
from spa_control.tests.test_web_server import FakeCamera
from spa_control.tests.test_web_two_motor_api import FakeRecorder
from spa_control.tests.test_web_global_api import FakeGlobal


class FaultClearTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.clock_patch = patch("spa_control.web_motor.time.monotonic", lambda: self.now)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.reset()

    def reset(self, fault="OVERCURRENT", trip=(4, 7340, 71234)):
        self.count = 100
        self.fault, self.trip = fault, trip
        self.port = FakePort()
        self.motor = WebMotor(clock=lambda: self.now)
        self.motor._worker = SimpleNamespace(is_alive=lambda: True)
        self.motor._stop = threading.Event()
        self.motor._accept("CTRL OK STOP")
        self.feed()

    def feed(self, *, dt=.1, fault=None, trip=None, count=None, change=None):
        self.now += dt
        self.count = self.count + 1 if count is None else count
        fault = self.fault if fault is None else fault
        trip = self.trip if trip is None else trip
        line = fixture(self.count, fault=fault).replace("spd=-10", "spd=0")
        line = line.replace("trip_mask=0x00", f"trip_mask=0x{trip[0]:02X}")
        line = line.replace("trip_current=0", f"trip_current={trip[1]}")
        line = line.replace("trip_ms=0", f"trip_ms={trip[2]}")
        if change:
            line = change(line)
        self.motor._accept(line)

    def tick(self, dt=0):
        self.now += dt
        self.motor._clear_tick(self.port)

    def begin(self):
        self.motor.clear_fault()
        self.tick()
        self.assertEqual(self.port.writes, [b"CTRL STOP\n"])
        self.motor._accept("CTRL OK STOP")

    def baseline(self, change=None):
        for _ in range(14):
            self.feed(change=change)
            self.tick()
        self.assertEqual(self.port.writes, [b"CTRL STOP\n", b"CTRL CLEAR\n"])
        self.assertEqual(self.motor.status()["clear_state"], "waiting_ack")

    def test_latched_overcurrent_sample_clears_only_after_ack_and_new_zero_state(self):
        state = self.motor.status()
        self.assertTrue(state["clear_supported"])
        for key, expected in zip(("trip_mask", "trip_current", "trip_ms"), self.trip):
            self.assertEqual(state[key], expected)
            self.assertIs(type(state[key]), int)
        self.begin()
        self.baseline()
        self.motor._accept("CTRL OK CLEAR")
        self.assertEqual(self.motor.status()["clear_state"], "verifying")
        self.feed(fault="NONE", trip=(0, 0, 0))
        self.assertEqual(self.motor.status()["clear_state"], "succeeded")
        self.assertEqual(self.motor.status()["fault"], "NONE")
        self.assertFalse(self.motor.status()["clear_supported"])
        self.tick(20)
        self.assertEqual(self.port.writes, [b"CTRL STOP\n", b"CTRL CLEAR\n"])
        self.assertFalse(any(b"SPEED" in command for command in self.port.writes))

    def test_supported_latches_recover_with_zero_current_and_temperature(self):
        for fault in ("USB_CLOSED", "COMMAND_TIMEOUT", "FEEDBACK_STALE", "OVERCURRENT", "OVERTEMP", "CAN_ERROR"):
            with self.subTest(fault=fault):
                self.reset(fault=fault, trip=(0, 0, 0))
                self.assertTrue(self.motor.status()["clear_supported"])
                self.begin()
                self.baseline()
                self.motor._accept("CTRL OK CLEAR")
                self.feed(fault="NONE", trip=(0, 0, 0))
                self.assertEqual(self.motor.status()["clear_state"], "succeeded")

    def test_old_usb_endpoint_does_not_clear_other_latches(self):
        with self.assertRaises(MotorError):
            self.motor.clear_usb_closed()
        self.assertEqual(self.port.writes, [])

    def test_rejects_unknown_fault_stale_state_or_missing_stop_ack(self):
        for condition in ("unknown", "stale", "no_ack"):
            with self.subTest(condition=condition):
                self.reset()
                if condition == "unknown":
                    self.feed(fault="UNKNOWN_FAULT")
                    self.assertFalse(self.motor.status()["clear_supported"])
                elif condition == "stale":
                    self.now += 1
                else:
                    self.motor._state["stop_acknowledged"] = False
                with self.assertRaises(MotorError):
                    self.motor.clear_fault()
                self.assertEqual(self.port.writes, [])

    def test_new_stop_ack_and_full_baseline_duration_are_required(self):
        self.motor.clear_fault()
        self.tick()
        for _ in range(14):
            self.feed()
            self.tick()
        self.assertEqual(self.port.writes, [b"CTRL STOP\n"])
        self.motor._accept("CTRL OK STOP")
        for _ in range(8):
            self.feed()
            self.tick()
        self.assertEqual(self.port.writes, [b"CTRL STOP\n"])
        for _ in range(6):
            self.feed()
            self.tick()
        self.assertEqual(self.port.writes, [b"CTRL STOP\n", b"CTRL CLEAR\n"])

    def test_dangerous_feedback_or_changed_latch_aborts_without_clear(self):
        changes = {
            "speed": lambda s: s.replace("spd=0", "spd=3"),
            "current": lambda s: s.replace("torque=20", "torque=1000"),
            "temperature": lambda s: s.replace("temp=25", "temp=50"),
            "can": lambda s: s.replace("can_healthy=1", "can_healthy=0"),
            "esr": lambda s: s.replace("ESR=0x00000000", "ESR=0x00000001"),
            "age": lambda s: s.replace("age_ms=1", "age_ms=50"),
            "target": lambda s: s.replace("target=0", "target=1"),
            "output": lambda s: s.replace("setV=0", "setV=1"),
            "fault": lambda s: s.replace("fault=OVERCURRENT", "fault=OVERTEMP"),
            "trip_mask": lambda s: s.replace("trip_mask=0x04", "trip_mask=0x08"),
            "trip_current": lambda s: s.replace("trip_current=7340", "trip_current=7350"),
            "trip_ms": lambda s: s.replace("trip_ms=71234", "trip_ms=71235"),
        }
        for name, change in changes.items():
            with self.subTest(condition=name):
                self.reset()
                self.begin()
                self.feed(change=change)
                self.tick()
                self.assertEqual(self.motor.status()["clear_state"], "failed")
                self.assertNotIn(b"CTRL CLEAR\n", self.port.writes)

    def test_all_seen_or_triggered_motors_need_fresh_advancing_feedback(self):
        def add_seen_m1(line):
            parts = line.split(" | ")
            parts[1] = parts[1].replace("seen=0 rx=0 age_ms=999999", f"seen=1 rx={self.count} age_ms=1")
            return " | ".join(parts)

        for triggered in (False, True):
            with self.subTest(triggered=triggered):
                self.reset(trip=(5 if triggered else 4, 7340, 71234))
                self.feed(change=add_seen_m1)
                self.begin()
                self.feed(change=add_seen_m1)
                self.tick()
                # A newly received frame cannot refresh a repeated CAN count.
                frozen_count = self.count
                def stale_m1(line):
                    parts = add_seen_m1(line).split(" | ")
                    parts[1] = parts[1].replace(f"rx={self.count}", f"rx={frozen_count}")
                    return " | ".join(parts)
                self.feed(change=stale_m1)
                self.tick()
                self.assertEqual(self.motor.status()["clear_state"], "failed")
                self.assertNotIn(b"CTRL CLEAR\n", self.port.writes)

    def test_clear_without_ack_times_out_once_and_never_restarts(self):
        self.begin()
        self.baseline()
        self.feed(fault="NONE", trip=(0, 0, 0))
        self.assertNotEqual(self.motor.status()["clear_state"], "succeeded")
        self.tick(6)
        self.assertEqual(self.motor.status()["clear_state"], "failed")
        self.tick(6)
        self.assertEqual(self.port.writes.count(b"CTRL CLEAR\n"), 1)

    def test_clear_ack_requires_reset_trip_and_new_feedback_count(self):
        for condition in ("trip_not_reset", "counter_not_new"):
            with self.subTest(condition=condition):
                self.reset()
                self.begin()
                self.baseline()
                self.motor._accept("CTRL OK CLEAR")
                self.feed(fault="NONE", trip=self.trip if condition == "trip_not_reset" else (0, 0, 0),
                          count=self.count if condition == "counter_not_new" else None)
                self.assertEqual(self.motor.status()["clear_state"], "failed")
                self.assertEqual(self.port.writes.count(b"CTRL CLEAR\n"), 1)

    def test_board_rejection_aborts_and_motion_cannot_start_during_clear(self):
        self.begin()
        with self.assertRaises(MotorError):
            self.motor.start_test(3, -60, 3)
        self.baseline()
        self.motor._accept("CTRL ERR CAN")
        self.assertEqual(self.motor.status()["clear_state"], "failed")
        self.tick(6)
        self.assertEqual(self.port.writes.count(b"CTRL CLEAR\n"), 1)


class ClearingMotor(Motor):
    def clear_fault(self):
        self.events.append(("clear_fault",))
        return {**self.status(), "clear_state": "checking"}


class FaultClearAPITests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.motor = ClearingMotor(self.events)
        self.experiment = Experiment(self.events)
        self.workbench = Workbench(camera=FakeCamera(), motor=self.motor,
            recorder=FakeRecorder(), global_camera=FakeGlobal(), experiment=self.experiment)
        self.server = LocalServer(("127.0.0.1", 0), self.workbench)
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=.01), daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.workbench.close()
        self.thread.join(2)

    def request(self, origin=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_address[1], timeout=2)
        headers = {"Content-Type": "application/json"}
        if origin:
            headers["Origin"] = origin
        try:
            conn.request("POST", "/api/motor/clear-fault", "{}", headers)
            response = conn.getresponse()
            return response.status, json.loads(response.read())
        finally:
            conn.close()

    def test_same_origin_requests_clear_without_starting_motion(self):
        origin = f"http://127.0.0.1:{self.server.server_address[1]}"
        code, result = self.request(origin)
        self.assertEqual(code, 200)
        self.assertEqual(result["clear_state"], "checking")
        self.assertEqual(self.events, [("clear_fault",)])

    def test_cross_origin_cannot_clear(self):
        self.assertEqual(self.request("https://other.example")[0], 403)
        self.assertEqual(self.events, [])

    def test_active_experiment_motion_or_shutdown_rejects_clear(self):
        self.experiment.active = True
        self.assertEqual(self.request()[0], 400)
        self.experiment.active = False
        for phase in ("preparing", "running", "stopping"):
            self.motor.motion = phase
            self.assertEqual(self.request()[0], 400)
        self.motor.motion = "idle"
        self.workbench._closing.set()
        self.assertEqual(self.request()[0], 400)
        self.assertEqual(self.events, [])

    def test_clear_waits_for_control_lock(self):
        entered, finished = threading.Event(), threading.Event()
        outcomes = []
        def call():
            entered.set()
            outcomes.append(self.workbench.clear_motor_fault())
            finished.set()
        with self.workbench._control_lock:
            worker = threading.Thread(target=call, daemon=True)
            worker.start()
            self.assertTrue(entered.wait(.5))
            self.assertFalse(finished.wait(.05))
            self.assertEqual(self.events, [])
        worker.join(1)
        self.assertTrue(finished.is_set())
        self.assertEqual(outcomes[0]["clear_state"], "checking")


if __name__ == "__main__":
    unittest.main()
