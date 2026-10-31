import json
import queue
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from spa_control.web_motor import MotorError, WebMotor, parse_frame


def fixture(count=100, age=1, fault="NONE"):
    line = ("RM_C MOTOR_USB v4-full can_ready=1 can_healthy=1 active=0x0C ESR=0x00000000 "
            f"fault={fault} lease_ms=300 limit=500 output_limit=25000 current_trip=7000 "
            "mode=PAIR dir3=-1 dir4=-1 roll=UNAVAILABLE trip_mask=0x00 trip_current=0 trip_ms=0")
    for i in range(1, 5):
        line += (f" | M{i} id=0x{0x204+i:03X} angle=123 spd=-10 torque=20 setI=0 setV=0 "
                 f"target=0 temp=25 seen={int(i > 2)} rx={count if i > 2 else 0} age_ms={age if i > 2 else 999999}")
    return line


class FakePort:
    def __init__(self):
        self.writes = []
        self.queue = queue.Queue()
        self.closed = False

    def write(self, payload):
        self.writes.append(payload)
        return len(payload)

    def read(self, size):
        try:
            return self.queue.get(timeout=0.01)
        except queue.Empty:
            return b""

    def close(self):
        self.closed = True


def wait_until(predicate):
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("worker did not reach expected state")


class MotorTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.port = FakePort()
        self.entries = [SimpleNamespace(device="COM9", vid=0x0483, pid=0x5740,
                                        serial_number="206B377B5948")]
        self.opens = []
        def open_port(name):
            self.opens.append(name)
            return self.port
        self.motor = WebMotor(lambda: self.entries, open_port, lambda: self.now)

    def tearDown(self):
        self.motor.close()

    def test_explicit_connect_stop_and_freshness(self):
        self.assertEqual(self.opens, [])
        self.motor.connect()
        wait_until(lambda: self.motor.status()["stop_sent"])
        self.assertEqual(self.port.writes, [b"CTRL STOP\n"])
        self.port.queue.put(("CTRL OK STOP\r\n" + fixture() + "\r\n").encode())
        wait_until(lambda: self.motor.status()["frames_received"] == 1)
        status = self.motor.status()
        self.assertTrue(status["stop_acknowledged"])
        self.assertTrue(status["frame_fresh"])
        self.assertTrue(status["motors"]["3"]["fresh"])
        self.assertEqual(status["directions"], {"3": -1, "4": -1})
        self.assertEqual(status["motors"]["4"]["spd"], -10)
        json.dumps(status, allow_nan=False)
        # Repeated CAN rx counters do not renew freshness, even with age=0.
        self.now += 0.3
        self.port.queue.put((fixture(age=0) + "\n").encode())
        wait_until(lambda: self.motor.status()["frames_received"] == 2)
        self.assertTrue(self.motor.status()["frame_fresh"])
        self.assertFalse(self.motor.status()["motors"]["3"]["fresh"])
        self.port.queue.put((fixture(count=101) + "\n").encode())
        wait_until(lambda: self.motor.status()["frames_received"] == 3)
        self.assertTrue(self.motor.status()["motors"]["3"]["fresh"])
        self.now += 0.6
        self.assertFalse(self.motor.status()["frame_fresh"])
        self.assertFalse(self.motor.status()["motors"]["4"]["fresh"])
        self.motor.disconnect()
        wait_until(lambda: self.port.closed)
        wait_until(lambda: self.motor.status()["state"] == "disconnected")
        self.assertIn("USB_CLOSED", self.motor.status()["message"])
        self.assertEqual(self.port.writes, [b"CTRL STOP\n"])

    def test_unmatched_or_ambiguous_board_never_opens(self):
        self.entries.append(self.entries[0])
        self.motor.connect()
        wait_until(lambda: self.motor.status()["state"] == "error")
        self.assertEqual(self.opens, [])
        self.assertIn("未唯一识别", self.motor.status()["message"])

    def test_fault_is_visible_and_never_cleared(self):
        self.motor.connect()
        wait_until(lambda: self.motor.status()["stop_sent"])
        self.port.queue.put((fixture(fault="USB_CLOSED") + "\n").encode())
        wait_until(lambda: self.motor.status()["frames_received"] == 1)
        self.assertEqual(self.motor.status()["fault"], "USB_CLOSED")
        self.assertEqual(self.port.writes, [b"CTRL STOP\n"])

    def test_parser_rejects_bad_version_and_duplicate_or_missing_fields(self):
        self.assertIsNone(parse_frame("CTRL OK STOP"))
        for text in (fixture().replace("v4-full", "v3"),
                     fixture().replace("dir4=-1", "dir4=-1 dir4=1"),
                     fixture().replace(" | M4 ", " | M3 "),
                     fixture().replace("id=0x207", "id=0x205")):
            with self.assertRaises(ValueError):
                parse_frame(text)

    def test_malformed_stream_causes_disconnect(self):
        self.motor.connect()
        wait_until(lambda: self.motor.status()["stop_sent"])
        self.port.queue.put((fixture().replace("v4-full", "v3") + "\n").encode())
        wait_until(lambda: self.motor.status()["state"] == "error")
        self.assertTrue(self.port.closed)
        self.assertIn("版本不匹配", self.motor.status()["message"])

    def prepare_clear(self):
        self.motor.connect()
        wait_until(lambda: self.motor.status()["stop_sent"])
        self.port.queue.put(("CTRL OK STOP\n" + fixture(fault="USB_CLOSED") + "\n").encode())
        wait_until(lambda: self.motor.status()["fault"] == "USB_CLOSED")
        self.motor.clear_usb_closed()
        wait_until(lambda: len(self.port.writes) == 2)
        self.port.queue.put(b"CTRL OK STOP\n")

    def feed_baseline(self):
        for count in range(101, 116):
            self.port.queue.put((fixture(count=count, fault="USB_CLOSED").replace("spd=-10", "spd=0") + "\n").encode())
            time.sleep(.095)

    def test_clear_success_requires_new_stop_ack_baseline_ack_and_fresh_state(self):
        self.prepare_clear()
        self.feed_baseline()
        wait_until(lambda: b"CTRL CLEAR\n" in self.port.writes)
        self.assertEqual(self.motor.status()["clear_state"], "waiting_ack")
        self.port.queue.put(("CTRL OK CLEAR\n" + fixture(count=119, fault="NONE").replace("spd=-10", "spd=0") + "\n").encode())
        wait_until(lambda: self.motor.status()["clear_state"] == "succeeded")
        self.assertEqual(self.motor.status()["fault"], "NONE")
        self.assertEqual(self.port.writes, [b"CTRL STOP\n", b"CTRL STOP\n", b"CTRL CLEAR\n"])

    def test_other_fault_cannot_be_cleared(self):
        self.motor.connect()
        wait_until(lambda: self.motor.status()["stop_sent"])
        self.port.queue.put((fixture(fault="OVER_CURRENT") + "\n").encode())
        wait_until(lambda: self.motor.status()["fault"] == "OVER_CURRENT")
        with self.assertRaises(MotorError):
            self.motor.clear_usb_closed()
        self.assertEqual(self.port.writes, [b"CTRL STOP\n"])

    def test_unsafe_samples_refuse_clear(self):
        self.prepare_clear()
        unsafe = fixture(count=101, fault="USB_CLOSED")  # speed -10
        self.port.queue.put((unsafe + "\n").encode())
        wait_until(lambda: self.motor.status()["clear_state"] == "failed")
        self.assertNotIn(b"CTRL CLEAR\n", self.port.writes)

    def test_guard_rejects_other_fault_and_dangerous_values(self):
        good = fixture(fault="USB_CLOSED").replace("spd=-10", "spd=0")
        changes = [
            ("fault=USB_CLOSED", "fault=OVER_CURRENT"),
            ("can_healthy=1", "can_healthy=0"),
            ("ESR=0x00000000", "ESR=0x00000001"),
            ("trip_mask=0x00", "trip_mask=0x04"),
            ("trip_current=0", "trip_current=100"),
            ("target=0", "target=1"),
            ("setV=0", "setV=1"),
            ("age_ms=1", "age_ms=50"),
            ("torque=20", "torque=1000"),
            ("temp=25", "temp=50"),
        ]
        self.motor._clear_guard(parse_frame(good), "USB_CLOSED")
        for old, new in changes:
            with self.subTest(change=new), self.assertRaises(MotorError):
                self.motor._clear_guard(parse_frame(good.replace(old, new)), "USB_CLOSED")

    def test_no_stop_ack_times_out_without_clear(self):
        with patch("spa_control.web_motor.CLEAR_TIMEOUT_S", .15):
            self.motor.connect()
            wait_until(lambda: self.motor.status()["stop_sent"])
            self.port.queue.put(("CTRL OK STOP\n" + fixture(fault="USB_CLOSED") + "\n").encode())
            wait_until(lambda: self.motor.status()["fault"] == "USB_CLOSED" and self.motor.status()["stop_acknowledged"])
            self.motor.clear_usb_closed()
            wait_until(lambda: self.motor.status()["clear_state"] == "failed")
        self.assertNotIn(b"CTRL CLEAR\n", self.port.writes)

    def test_duplicate_feedback_counts_abort(self):
        self.prepare_clear()
        frame = (fixture(count=101, fault="USB_CLOSED").replace("spd=-10", "spd=0") + "\n").encode()
        self.port.queue.put(frame)
        self.port.queue.put(frame)
        wait_until(lambda: self.motor.status()["clear_state"] == "failed")
        self.assertNotIn(b"CTRL CLEAR\n", self.port.writes)

    def test_clear_ack_missing_remains_unconfirmed_and_never_retries(self):
        with patch("spa_control.web_motor.CLEAR_TIMEOUT_S", 1.8):
            self.prepare_clear()
            self.feed_baseline()
            wait_until(lambda: b"CTRL CLEAR\n" in self.port.writes)
            self.port.queue.put((fixture(count=109, fault="NONE").replace("spd=-10", "spd=0") + "\n").encode())
            wait_until(lambda: self.motor.status()["clear_state"] == "failed")
        self.assertIn("未确认", self.motor.status()["clear_message"])
        self.assertEqual(self.port.writes.count(b"CTRL CLEAR\n"), 1)

    def test_board_rejection_does_not_retry(self):
        self.prepare_clear()
        self.feed_baseline()
        wait_until(lambda: b"CTRL CLEAR\n" in self.port.writes)
        self.port.queue.put(b"CTRL ERR CAN\n")
        wait_until(lambda: self.motor.status()["clear_state"] == "failed")
        self.assertEqual(self.port.writes.count(b"CTRL CLEAR\n"), 1)

    def test_disconnect_pending_clear_reports_unknown(self):
        self.prepare_clear()
        self.motor.disconnect()
        wait_until(lambda: self.motor.status()["state"] == "disconnected")
        self.assertEqual(self.motor.status()["clear_state"], "failed")
        self.assertIn("未确认", self.motor.status()["clear_message"])
        self.assertNotIn(b"CTRL CLEAR\n", self.port.writes)


if __name__ == "__main__":
    unittest.main()
