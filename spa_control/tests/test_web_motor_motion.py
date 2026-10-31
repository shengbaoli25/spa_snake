"""Deterministic protocol tests; these never enumerate or open hardware."""
import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from spa_control.web_motor import MotorError, WebMotor
from spa_control.tests.test_web_motor import FakePort, fixture, wait_until


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.count = 100
        self.port = FakePort()
        self.motor = WebMotor(clock=lambda: self.now)
        self.motor._worker = SimpleNamespace(is_alive=lambda: True)
        self.motor._stop = threading.Event()
        self.motor._accept("CTRL OK STOP")
        self.feed()

    def feed(self, *, dt=0, mode="PAIR", targets=None, speed=0, change=None, advance=True, fault="NONE"):
        self.now += dt
        if advance:
            self.count += 1
        line = fixture(self.count, fault=fault).replace("spd=-10", f"spd={speed}").replace("mode=PAIR", "mode=" + mode)
        if targets:
            segments = line.split(" | ")
            for ch, rpm in targets.items():
                segments[int(ch)] = segments[int(ch)].replace("target=0", f"target={rpm}")
            line = " | ".join(segments)
        if change:
            line = change(line)
        self.motor._accept(line)

    def tick(self, dt=0):
        self.now += dt
        self.motor._motion_tick(self.port)

    def begin(self, channel=3, rpm=60, duration=3, owner="manual"):
        if owner == "manual":
            self.motor.start_test(channel, rpm, duration)
        else:
            self.motor.start_motion({"3": rpm, "4": rpm}, duration, owner=owner)
        self.tick()
        self.motor._accept("CTRL OK STOP")
        for i in range(14):
            if owner == "paper":
                self.motor.update_motion({"3": rpm, "4": rpm})
            self.feed(dt=.1)
            self.tick()
        self.assertEqual(self.port.writes[-1], b"CTRL MODE BENCH\n")
        self.motor._accept("CTRL OK MODE BENCH")
        self.feed(dt=.05, mode="BENCH")
        self.tick()
        self.assertIn(b"CTRL SPEED", self.port.writes[-1])
        self.motor._accept(self.port.writes[-1].decode().strip().replace("CTRL SPEED", "CTRL OK"))
        self.assertEqual(self.motor.status()["motion"]["state"], "running")

    def stop_confirm(self):
        self.tick()
        self.assertEqual(self.port.writes[-1], b"CTRL STOP\n")
        self.motor._accept("CTRL OK STOP")
        self.feed(dt=.01, mode="BENCH")

    def test_individual_test_sends_only_selected_motor_after_barrier(self):
        self.begin(channel=4, rpm=-60)
        self.assertEqual(self.port.writes[:3], [b"CTRL STOP\n", b"CTRL MODE BENCH\n", b"CTRL SPEED M1=0 M2=0 M3=0 M4=-60\n"])
        state = self.motor.status()["motion"]
        self.assertEqual(state["last_speed_targets"], {"3": 0, "4": -60})
        self.assertEqual(state["last_speed_ack_targets"], state["last_speed_targets"])
        self.assertIsNotNone(state["last_speed_ack_monotonic_s"])
        json.dumps(self.motor.status(), allow_nan=False)

    def test_no_speed_before_mode_ack_and_new_frame(self):
        self.motor.start_test(3)
        self.tick()
        self.motor._accept("CTRL OK STOP")
        for i in range(14):
            self.feed(dt=.1)
            self.tick()
        self.assertEqual(len(self.port.writes), 2)
        self.motor._accept("CTRL OK MODE BENCH")
        self.tick()
        self.assertEqual(len(self.port.writes), 2)
        self.feed(dt=.05, mode="BENCH")
        self.tick()
        self.assertEqual(len(self.port.writes), 3)

    def test_duration_stops_and_waits_for_ack_and_new_zero_frame(self):
        self.begin(duration=.2)
        self.feed(dt=.21, mode="BENCH", targets={"3":60}, speed=10)
        self.tick()
        self.assertEqual(self.motor.status()["motion"]["state"], "stopping")
        self.motor._accept("CTRL OK STOP")
        self.feed(dt=.01, mode="BENCH", speed=10)
        self.assertEqual(self.motor.status()["motion"]["state"], "stopping")
        self.feed(dt=.1, mode="BENCH")
        result = self.motor.status()["motion"]
        self.assertEqual(result["state"], "completed")
        self.assertTrue(result["stop_confirmed"])

    def test_stop_without_motion_is_always_sent(self):
        self.motor.stop_motion()
        self.stop_confirm()
        self.assertEqual(self.motor.status()["motion"]["state"], "completed")

    def test_stop_during_prepare_never_sends_speed(self):
        self.motor.start_test(3)
        self.tick()
        self.motor.stop_motion()
        self.stop_confirm()
        self.assertFalse(any(b"SPEED" in item for item in self.port.writes))
        self.assertEqual(self.motor.status()["motion"]["state"], "completed")

    def test_manual_stop_during_running_remains_stopped(self):
        self.begin()
        self.motor.stop_motion()
        self.stop_confirm()
        n = len(self.port.writes)
        self.tick(.1)
        self.assertEqual(len(self.port.writes), n)

    def test_ack_timeout_stops_before_firmware_lease_and_does_not_retry_speed(self):
        self.begin()
        self.feed(dt=.11, mode="BENCH", targets={"3":60})
        self.tick()
        n = sum(b"SPEED" in item for item in self.port.writes)
        self.feed(dt=.23, mode="BENCH", targets={"3":60})
        self.tick()
        self.assertEqual(self.port.writes[-1], b"CTRL STOP\n")
        self.assertEqual(sum(b"SPEED" in item for item in self.port.writes), n)
        self.assertIn("确认超时", self.motor.status()["motion"]["message"])

    def test_duplicate_counter_faults_even_with_new_usb_reports(self):
        self.begin()
        self.feed(dt=.1, mode="BENCH", targets={"3":60}, advance=False)
        self.stop_confirm()
        self.assertEqual(self.motor.status()["motion"]["state"], "failed")
        self.assertIn("计数未推进", self.motor.status()["motion"]["message"])

    def test_no_telemetry_for_quarter_second_stops(self):
        self.begin()
        self.tick(.251)
        self.assertEqual(self.port.writes[-1], b"CTRL STOP\n")
        self.assertIn("遥测超时", self.motor.status()["motion"]["message"])

    def test_board_fault_never_auto_clears_and_stops(self):
        self.begin()
        self.feed(dt=.1, mode="BENCH", fault="OVER_CURRENT")
        self.stop_confirm()
        self.assertEqual(self.motor.status()["motion"]["state"], "failed")
        self.assertFalse(any(b"CLEAR" in item for item in self.port.writes))

    def test_motor_limits_temperature_and_inactive_outputs_stop(self):
        changes = [lambda s:s.replace("current_trip=7000", "current_trip=9000"),
                   lambda s:s.replace("limit=500 ", "limit=800 "),
                   lambda s:s.replace("output_limit=25000", "output_limit=30000"),
                   lambda s:s.replace("lease_ms=300", "lease_ms=999"),
                   lambda s:s.replace("temp=25", "temp=80"),
                   lambda s:s.replace("torque=20", "torque=7000"),
                   lambda s:s.replace("age_ms=1", "age_ms=100"),
                   lambda s:s.replace("setV=0", "setV=1", 1)]
        for change in changes:
            with self.subTest(change=change):
                self.setUp()
                self.begin()
                self.feed(dt=.1, mode="BENCH", change=change)
                self.tick()
                self.assertEqual(self.port.writes[-1], b"CTRL STOP\n")

    def test_initial_moving_baseline_rejected(self):
        self.motor.start_test(3)
        self.tick()
        self.motor._accept("CTRL OK STOP")
        self.feed(dt=.1, speed=3)
        self.tick()
        self.assertFalse(any(b"SPEED" in item for item in self.port.writes))
        self.assertIn("未静止", self.motor.status()["motion"]["message"])

    def test_paper_heartbeat_and_change_targets_are_separate_from_serial_renewal(self):
        self.begin(rpm=0, owner="paper")
        self.assertEqual(self.motor.status()["motion"]["last_speed_targets"], {"3":0, "4":0})
        self.motor.update_motion({"3":-60, "4":-65})
        self.assertEqual(self.motor.status()["motion"]["last_speed_targets"], {"3":0, "4":0})
        self.feed(dt=.11, mode="BENCH")
        self.tick()
        self.assertEqual(self.port.writes[-1], b"CTRL SPEED M1=0 M2=0 M3=-60 M4=-65\n")
        self.assertEqual(self.motor.status()["motion"]["last_speed_targets"], {"3":-60, "4":-65})
        self.motor._accept("CTRL OK M1=0 M2=0 M3=-60 M4=-65")
        for i in range(4):
            self.feed(dt=.11, mode="BENCH", targets={"3":-60,"4":-65})
            self.tick()
            self.motor._accept("CTRL OK M1=0 M2=0 M3=-60 M4=-65")
        self.assertEqual(self.port.writes[-1], b"CTRL STOP\n")
        self.assertIn("策略续租超时", self.motor.status()["motion"]["message"])

    def test_late_heartbeat_cannot_revive_expired_preparation(self):
        self.motor.start_motion({"3":0,"4":0},10,owner="paper")
        self.now += .501
        with self.assertRaises(MotorError):
            self.motor.update_motion({"3":-60,"4":-60})
        self.tick()
        self.assertEqual(self.port.writes, [b"CTRL STOP\n"])

    def test_mode_or_speed_ack_mismatch_causes_stop(self):
        self.begin()
        self.feed(dt=.11, mode="BENCH", targets={"3":60})
        self.tick()
        self.motor._accept("CTRL OK M1=0 M2=0 M3=500 M4=0")
        self.tick()
        self.assertEqual(self.port.writes[-1], b"CTRL STOP\n")
        self.assertIn("确认不符", self.motor.status()["motion"]["message"])

    def test_target_echo_mismatch_causes_stop(self):
        self.begin()
        for i in range(3):
            self.feed(dt=.1, mode="BENCH", targets={"3":50})
            self.tick()
            if b"SPEED" in self.port.writes[-1]:
                self.motor._accept("CTRL OK M1=0 M2=0 M3=60 M4=0")
        self.assertEqual(self.port.writes[-1], b"CTRL STOP\n")
        self.assertIn("回显不符", self.motor.status()["motion"]["message"])

    def test_stop_timeout_is_not_reported_as_success(self):
        self.begin()
        self.motor.stop_motion()
        self.tick()
        self.tick(2.01)
        result = self.motor.status()["motion"]
        self.assertEqual(result["state"], "failed")
        self.assertFalse(result["stop_confirmed"])

    def test_validation_and_exclusive_owner(self):
        for rpm in (501, -501, 1.5, True, float("nan"), float("inf"), "60"):
            with self.subTest(rpm=rpm), self.assertRaises(MotorError):
                self.motor.start_test(3, rpm)
        for duration in (0, -1, 11, float("nan"), True):
            with self.subTest(duration=duration), self.assertRaises(MotorError):
                self.motor.start_test(3,60,duration)
        with self.assertRaises(MotorError):
            self.motor.start_motion({"1":0,"3":60},3)
        self.motor.start_test(3)
        with self.assertRaises(MotorError):
            self.motor.start_test(4)
        with self.assertRaises(MotorError):
            self.motor.update_motion({"3":60})

    def test_stop_set_before_speed_prevents_write(self):
        self.begin()
        count = sum(b"SPEED" in item for item in self.port.writes)
        self.motor._stop.set()
        self.feed(dt=.11,mode="BENCH",targets={"3":60})
        self.tick()
        self.assertEqual(sum(b"SPEED" in item for item in self.port.writes), count)
        self.assertEqual(self.port.writes[-1],b"CTRL STOP\n")


class OwnerPort(FakePort):
    """Minimal fake firmware with live telemetry and protocol ACKs."""
    def __init__(self):
        super().__init__()
        self.writers = []
        self.mode = "PAIR"
        self.targets = {"3":0,"4":0}
        self.count = 100
        self.last_report = 0

    def write(self, payload):
        self.writers.append(threading.current_thread().name)
        super().write(payload)
        line = payload.decode().strip()
        if line == "CTRL STOP":
            self.targets = {"3":0,"4":0}
            ack = "CTRL OK STOP"
        elif line == "CTRL MODE BENCH":
            self.mode = "BENCH"
            ack = "CTRL OK MODE BENCH"
        elif line.startswith("CTRL SPEED"):
            fields = dict(word.split("=") for word in line.split()[2:])
            self.targets = {ch:int(fields["M"+ch]) for ch in ("3","4")}
            ack = line.replace("CTRL SPEED","CTRL OK")
        else:
            ack = "CTRL ERR TEST_UNEXPECTED_COMMAND"
        self.queue.put((ack+"\n").encode())
        return len(payload)

    def read(self, size):
        now = time.monotonic()
        if now - self.last_report > .015:
            self.last_report = now
            self.count += 1
            parts = fixture(self.count).replace("mode=PAIR","mode="+self.mode).replace("spd=-10","spd=0").split(" | ")
            for ch,target in self.targets.items():
                parts[int(ch)] = parts[int(ch)].replace("target=0",f"target={target}")
            self.queue.put((" | ".join(parts)+"\n").encode())
        return super().read(size)


class MotorOwnerThreadTests(unittest.TestCase):
    def setUp(self):
        self.port = OwnerPort()
        entry = SimpleNamespace(device="MOCK",vid=0x0483,pid=0x5740,serial_number="206B377B5948")
        self.motor = WebMotor(port_lister=lambda:[entry],serial_factory=lambda _:self.port)

    def tearDown(self):
        self.motor.close()

    def run_motion(self):
        self.motor.connect()
        wait_until(lambda:self.motor.status()["state"]=="connected")
        with patch("spa_control.web_motor.MOTION_BASELINE_S",.08):
            self.motor.start_test(4,-60,3)
            wait_until(lambda:self.motor.status()["motion"]["state"]=="running")

    def test_all_commands_have_one_owner_and_disconnect_sends_stop(self):
        self.run_motion()
        self.motor.close()
        self.assertEqual(self.port.writes[-1],b"CTRL STOP\n")
        self.assertEqual(set(self.port.writers),{"c-board-telemetry"})
        self.assertEqual(self.motor.status()["motion"]["state"],"completed")
        self.assertTrue(self.motor.status()["motion"]["stop_confirmed"])

    def test_malformed_stream_during_motion_sends_stop_before_close(self):
        self.run_motion()
        self.port.queue.put((fixture().replace("v4-full","unknown")+"\n").encode())
        wait_until(lambda:self.motor.status()["state"]=="error")
        self.assertEqual(self.port.writes[-1],b"CTRL STOP\n")
        self.assertTrue(self.port.closed)
        self.assertEqual(self.motor.status()["motion"]["state"],"failed")


if __name__ == "__main__":
    unittest.main()
