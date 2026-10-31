"""Transport tests against a loopback peer; never connect to the robot."""
from __future__ import annotations

import queue
import socket
import threading
import time
import unittest
from unittest.mock import patch

from spa_control.transport import RobotLink, motor_feedback_state


class FakePeer:
    def __init__(self):
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        self.connection = None
        self.ready = threading.Event()
        self.closed = threading.Event()
        self.received = queue.Queue()
        self.worker = threading.Thread(target=self._serve, daemon=True)
        self.worker.start()

    def _serve(self):
        try:
            self.connection, _ = self.listener.accept()
            self.connection.settimeout(0.05)
            self.ready.set()
            pending = b""
            while not self.closed.is_set():
                try:
                    data = self.connection.recv(8192)
                    if not data:
                        return
                    pending += data
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        self.received.put(line + b"\n")
                except socket.timeout:
                    continue
        except OSError:
            return

    def send(self, data):
        if not self.ready.wait(1):
            raise AssertionError("loopback peer was not connected")
        self.connection.sendall(data)

    def close(self):
        self.closed.set()
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.connection.close()
        self.listener.close()
        self.worker.join(timeout=0.3)


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not reached before timeout")


class RobotLinkTests(unittest.TestCase):
    def setUp(self):
        self.peer = FakePeer()
        self.link = RobotLink("127.0.0.1", self.peer.port)
        self.link.connect()
        self.assertTrue(self.peer.ready.wait(1))

    def tearDown(self):
        self.link.close()
        self.peer.close()

    def test_connect_is_silent_and_lease_sends_one_stop(self):
        with self.assertRaises(queue.Empty):
            self.peer.received.get(timeout=0.06)
        before = time.monotonic()
        sent = self.link.send_speeds([-40, 40, -40, 40, 0, 0], lease_s=0.12)
        self.assertGreaterEqual(sent, before)
        self.assertEqual(self.peer.received.get(timeout=0.5), b"CTRL SPEED M1=-40 M2=40 M3=-40 M4=40 M5=0 M6=0\n")
        self.assertEqual(self.peer.received.get(timeout=0.7), b"CTRL STOP\n")
        self.assertGreaterEqual(time.monotonic() - sent, 0.11)
        with self.assertRaises(queue.Empty):
            self.peer.received.get(timeout=0.15)

    def test_lease_fault_latches_motion_until_disconnect_and_reconnect(self):
        motion = [-20, 20, -20, 20, 0, 0]
        self.link.send_speeds(motion, lease_s=0.06)
        self.peer.received.get(timeout=0.5)
        self.assertEqual(self.peer.received.get(timeout=0.5), b"CTRL STOP\n")
        wait_until(lambda: not self.link._stopping)
        self.assertTrue(self.link.lease_fault)
        self.assertIn("需重新连接", self.link.status)
        with self.assertRaises(InterruptedError):
            self.link.send_speeds(motion)
        self.link.connect()  # Calling connect on a live socket cannot rearm it.
        with self.assertRaises(InterruptedError):
            self.link.send_speeds(motion)
        self.assertTrue(self.link.stop())
        self.assertEqual(self.peer.received.get(timeout=0.5), b"CTRL STOP\n")
        self.link.send_speeds([0] * 6)
        self.assertEqual(self.peer.received.get(timeout=0.5), b"CTRL SPEED M1=0 M2=0 M3=0 M4=0 M5=0 M6=0\n")
        self.assertTrue(self.link.lease_fault)
        with self.assertRaises(queue.Empty):
            self.peer.received.get(timeout=0.08)
        self.link.close()
        self.peer.close()
        self.peer = FakePeer()
        self.link.port = self.peer.port
        self.link.connect()
        self.assertFalse(self.link.lease_fault)
        self.link.send_speeds(motion)
        self.assertEqual(self.peer.received.get(timeout=0.5), b"CTRL SPEED M1=-20 M2=20 M3=-20 M4=20 M5=0 M6=0\n")

    def test_fragmented_telemetry_and_partial_values_do_not_refresh(self):
        self.peer.send(b"CAN rx=2 | M1 id=0x205 angle=23 spd=-42 tor")
        self.peer.send(b"que=120 setI=999 setV=3 temp=28 | M4 id=520 speed=41 current=-130\n")
        wait_until(lambda: self.link.snapshot().get(1, {}).get("speed") == -42)
        first = self.link.snapshot()[1]
        self.assertEqual(first["current_raw"], 120)
        self.assertEqual(first["set_current"], 999)
        self.assertEqual(first["can_id"], 0x205)
        self.assertEqual(self.link.snapshot()[4]["current_raw"], -130)
        self.peer.send(b"CAN rx=3 | M1 spd=-50 setI=400 temp=29\n")
        wait_until(lambda: self.link.snapshot()[1]["temp"] == 29)
        partial = self.link.snapshot()[1]
        self.assertEqual(partial["last_seen"], first["last_seen"])
        self.assertEqual(partial["speed"], first["speed"])
        self.assertEqual(partial["current_raw"], first["current_raw"])
        self.peer.send(b"CAN rx=4 | M1 spd=nan torque=111 temp=30\n")
        wait_until(lambda: self.link.snapshot()[1]["temp"] == 30)
        self.assertEqual(self.link.snapshot()[1]["last_seen"], first["last_seen"])
        # snapshot is a copy, not shared mutable telemetry state.
        partial["current_raw"] = 123456
        self.assertEqual(self.link.snapshot()[1]["current_raw"], 120)

    def test_explicit_stop_cancels_a_waiting_old_command(self):
        self.link._send_lock.acquire()
        errors = []
        entered = threading.Event()
        def queued_send():
            entered.set()
            try:
                self.link.send_speeds([20, -20, 20, -20, 0, 0])
            except Exception as exc:
                errors.append(exc)
        worker = threading.Thread(target=queued_send)
        worker.start()
        entered.wait(0.5)
        time.sleep(0.02)
        stopper = threading.Thread(target=self.link.stop)
        stopper.start()
        wait_until(lambda: self.link._stopping)
        self.link._send_lock.release()
        worker.join(timeout=0.5)
        stopper.join(timeout=0.5)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], InterruptedError)
        self.assertEqual(self.peer.received.get(timeout=0.5), b"CTRL STOP\n")
        with self.assertRaises(queue.Empty):
            self.peer.received.get(timeout=0.1)

    def test_device_fault_cancels_waiting_motion_and_stops_without_rearming(self):
        self.peer.send(b"M1 speed=2 torque=3\n")
        wait_until(lambda: 1 in self.link.snapshot())
        self.link._send_lock.acquire()
        errors = []
        entered = threading.Event()

        def queued_send():
            entered.set()
            try:
                self.link.send_speeds([20, 0, 0, 0, 0, 0])
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=queued_send)
        worker.start()
        self.assertTrue(entered.wait(0.5))
        try:
            self.peer.send(b"CTRL STATUS link=1 fault=1 reason=CAN_ERROR\n")
            wait_until(lambda: self.link.diagnostics()["device_fault"] == "CAN_ERROR")
        finally:
            self.link._send_lock.release()
        worker.join(timeout=0.5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], InterruptedError)
        self.assertEqual(self.peer.received.get(timeout=0.5), b"CTRL STOP\n")
        self.assertFalse(motor_feedback_state(self.link.snapshot(), 1, 0.8, 7000)[0])
        self.peer.send(b"CTRL STATUS link=1 fault=0 reason=NONE\n")
        wait_until(lambda: self.link.last_feedback == "CTRL STATUS link=1 fault=0 reason=NONE")
        self.assertEqual(self.link.diagnostics()["device_fault"], "CAN_ERROR")
        with self.assertRaises(InterruptedError):
            self.link.send_speeds([20, 0, 0, 0, 0, 0])
        with self.assertRaises(queue.Empty):
            self.peer.received.get(timeout=0.08)

    def test_peer_disconnect_invalidates_feedback_and_commands(self):
        self.peer.send(b"CAN rx=1 | M1 spd=2 torque=3\n")
        wait_until(lambda: 1 in self.link.snapshot())
        self.peer.close()
        wait_until(lambda: not self.link.connected)
        self.assertEqual(self.link.snapshot(), {})
        self.assertFalse(self.link.stop())
        with self.assertRaises(ConnectionError):
            self.link.send_speeds([0] * 6)

    def test_zero_command_clears_lease_and_invalid_commands_are_rejected(self):
        for speeds in ([1] * 5, [501] * 6, [1.0] * 6, [True] * 6):
            with self.assertRaises(ValueError):
                self.link.send_speeds(speeds)
        self.link.send_speeds([10, -10, 10, -10, 0, 0], lease_s=0.12)
        self.peer.received.get(timeout=0.5)
        self.link.send_speeds([0] * 6, lease_s=0.12)
        self.assertEqual(self.peer.received.get(timeout=0.5), b"CTRL SPEED M1=0 M2=0 M3=0 M4=0 M5=0 M6=0\n")
        with self.assertRaises(queue.Empty):
            self.peer.received.get(timeout=0.2)

    def test_close_sends_stop_and_reconnect_discards_old_telemetry(self):
        self.peer.send(b"CAN rx=1 | M1 spd=2 torque=3\n")
        wait_until(lambda: 1 in self.link.snapshot())
        self.link.close()
        self.assertEqual(self.peer.received.get(timeout=0.5), b"CTRL STOP\n")
        self.assertFalse(self.link.connected)
        self.assertEqual(self.link.snapshot(), {})
        self.peer.close()
        self.peer = FakePeer()
        self.link.port = self.peer.port
        self.link.connect()
        self.assertTrue(self.link.connected)
        self.assertEqual(self.link.snapshot(), {})
        self.link.send_speeds([0] * 6)
        self.assertEqual(self.peer.received.get(timeout=0.5), b"CTRL SPEED M1=0 M2=0 M3=0 M4=0 M5=0 M6=0\n")

    def test_diagnostics_zero_byte_connection_is_silent_and_not_motor_ready(self):
        first = self.link.diagnostics()
        self.assertTrue(first["connected"])
        self.assertGreaterEqual(first["connected_for_s"], 0)
        self.assertEqual(first["received_bytes"], 0)
        self.assertEqual(first["received_lines"], 0)
        self.assertIsNone(first["last_receive_age_s"])
        self.assertEqual(first["last_line"], "")
        self.assertEqual(first["recent_lines"], [])
        ready, reason = motor_feedback_state(self.link.snapshot(), 1, 0.8, 7000)
        self.assertFalse(ready)
        self.assertIn("未收到该通道", reason)
        with self.assertRaises(queue.Empty):
            self.peer.received.get(timeout=0.06)
        self.assertGreater(self.link.diagnostics()["connected_for_s"], first["connected_for_s"])

    def test_diagnostics_counts_health_messages_and_fragment_bytes_without_motor_data(self):
        fragment = b"HEALTH can_rx=0 status=waiting"
        self.peer.send(fragment)
        wait_until(lambda: self.link.diagnostics()["received_bytes"] == len(fragment))
        partial = self.link.diagnostics()
        self.assertEqual(partial["received_lines"], 0)
        self.assertEqual(partial["last_line"], "")
        self.assertIsNotNone(partial["last_receive_age_s"])
        self.assertEqual(self.link.snapshot(), {})
        tail = b"\nCAN rx=0\n\n"
        self.peer.send(tail)
        wait_until(lambda: self.link.diagnostics()["received_lines"] == 3)
        complete = self.link.diagnostics()
        self.assertEqual(complete["received_bytes"], len(fragment) + len(tail))
        self.assertEqual(complete["recent_lines"], [fragment.decode(), "CAN rx=0", ""])
        self.assertEqual(complete["last_line"], "")
        complete["recent_lines"].append("external mutation")
        self.assertNotIn("external mutation", self.link.diagnostics()["recent_lines"])
        self.assertEqual(self.link.snapshot(), {})
        with self.assertRaises(queue.Empty):
            self.peer.received.get(timeout=0.06)

    def test_diagnostics_partial_motor_frame_never_becomes_ready_until_complete(self):
        first = b"M1 spd=20 setI=400 temp=30\n"
        self.peer.send(first)
        wait_until(lambda: self.link.diagnostics()["received_lines"] == 1)
        ready, reason = motor_feedback_state(self.link.snapshot(), 1, 0.8, 7000)
        self.assertFalse(ready)
        self.assertIn("缺失或无效", reason)
        fragment = b"M1 spd=20 current_raw="
        self.peer.send(fragment)
        wait_until(lambda: self.link.diagnostics()["received_bytes"] == len(first) + len(fragment))
        self.assertEqual(self.link.diagnostics()["received_lines"], 1)
        self.assertFalse(motor_feedback_state(self.link.snapshot(), 1, 0.8, 7000)[0])
        self.peer.send(b"-42\n")
        wait_until(lambda: self.link.diagnostics()["received_lines"] == 2)
        self.assertTrue(motor_feedback_state(self.link.snapshot(), 1, 0.8, 7000)[0])
        state = self.link.diagnostics()
        self.assertEqual(state["received_bytes"], len(first) + len(fragment) + 4)
        self.assertEqual(state["last_line"], "M1 spd=20 current_raw=-42")

    def test_diagnostics_reconnect_resets_counts_lines_and_timing(self):
        payload = b"HEALTH ok\nM1 speed=5 torque=30\n"
        self.peer.send(payload)
        wait_until(lambda: self.link.diagnostics()["received_lines"] == 2)
        self.link.close()
        self.assertFalse(self.link.diagnostics()["connected"])
        self.assertEqual(self.link.diagnostics()["connected_for_s"], 0)
        self.assertEqual(self.link.diagnostics()["received_bytes"], len(payload))
        self.peer.close()
        self.peer = FakePeer()
        self.link.port = self.peer.port
        self.link.connect()
        self.assertTrue(self.peer.ready.wait(1))
        state = self.link.diagnostics()
        self.assertTrue(state["connected"])
        self.assertEqual(state["received_bytes"], 0)
        self.assertEqual(state["received_lines"], 0)
        self.assertIsNone(state["last_receive_age_s"])
        self.assertEqual(state["last_line"], "")
        self.assertEqual(state["recent_lines"], [])
        self.assertEqual(self.link.snapshot(), {})
        with self.assertRaises(queue.Empty):
            self.peer.received.get(timeout=0.06)

    def test_transmit_metrics_measure_completed_writes_and_reset_gap_after_stop(self):
        command = [-300, 0, 0, 0, 0, 0]
        first = self.link.send_speeds(command)
        second = self.link.send_speeds(command)
        data = self.link.diagnostics()
        self.assertEqual(data["sent_motion_count"], 2)
        self.assertAlmostEqual(data["last_motion_gap_s"], second - first)
        self.assertAlmostEqual(data["max_motion_gap_s"], second - first)
        self.assertGreaterEqual(data["last_send_duration_s"], 0)
        self.assertEqual(data["last_tx_line"], "CTRL SPEED M1=-300 M2=0 M3=0 M4=0 M5=0 M6=0")
        self.assertTrue(self.link.stop())
        stopped = self.link.diagnostics()
        self.assertEqual(stopped["sent_stop_count"], 1)
        self.assertEqual(stopped["last_tx_line"], "CTRL STOP")
        self.link.send_speeds(command)
        next_run = self.link.diagnostics()
        self.assertEqual(next_run["sent_motion_count"], 3)
        self.assertIsNone(next_run["last_motion_gap_s"])
        self.assertIsNone(next_run["max_motion_gap_s"])

    def test_send_error_retains_original_exception_and_diagnostics(self):
        failure = OSError(10054, "injected reset from peer")
        with patch.object(socket.socket, "sendall", side_effect=failure):
            with self.assertRaises(ConnectionError) as raised:
                self.link.send_speeds([300, 0, 0, 0, 0, 0])
        self.assertIs(raised.exception.__cause__, failure)
        self.assertIn("injected reset", str(raised.exception))
        data = self.link.diagnostics()
        self.assertFalse(data["connected"])
        self.assertEqual(data["sent_motion_count"], 0)
        self.assertEqual(data["last_transport_error"]["operation"], "send")
        self.assertEqual(data["last_transport_error"]["errno"], 10054)
        self.assertIn("injected reset", data["status"])
        data["last_transport_error"]["message"] = "changed copy"
        self.assertIn("injected reset", self.link.diagnostics()["last_transport_error"]["message"])

    def test_peer_close_retains_disconnect_reason_but_normal_close_is_not_an_error(self):
        self.peer.close()
        wait_until(lambda: not self.link.connected)
        self.assertEqual(self.link.diagnostics()["last_transport_error"]["type"], "PeerClosed")
        self.peer = FakePeer()
        self.link.port = self.peer.port
        self.link.connect()
        self.assertEqual(self.link.diagnostics()["sent_motion_count"], 0)
        self.assertIsNone(self.link.diagnostics()["last_transport_error"])
        self.link.close()
        self.assertIsNone(self.link.diagnostics()["last_transport_error"])


class RobotLinkParserTests(unittest.TestCase):
    """Deterministic firmware-format regressions, with no hardware or TCP."""

    def setUp(self):
        class CaptureSocket:
            def __init__(self):
                self.sent = []

            def sendall(self, data):
                self.sent.append(data)

        self.link = RobotLink("unused", 0)
        self.sock = CaptureSocket()
        self.link._socket = self.sock

    def consume(self, line, now=100.0):
        with patch("spa_control.transport.time.monotonic", return_value=now):
            self.link._consume_line(line, self.sock)

    def ready(self, now=100.0):
        return motor_feedback_state(self.link.snapshot(), 1, 0.8, 7000, now=now)

    def test_usb_zero_placeholders_do_not_create_measured_feedback(self):
        self.consume("RM_C MOTOR_USB v2 rx=0 fault=NONE | M1 seen=0 rx=0 age_ms=999999 spd=0 torque=0")
        self.assertFalse(self.ready()[0])
        self.assertIsNone(self.link.snapshot()[1]["last_seen"])
        self.assertNotIn("speed", self.link.snapshot()[1])
        self.assertEqual(self.sock.sent, [])

    def test_reported_age_is_included_and_duplicate_rx_never_renews_sample(self):
        self.consume("RM_C MOTOR_USB v3 fault=NONE | M1 seen=1 rx=27 age_ms=100 spd=2 torque=3")
        self.assertAlmostEqual(self.link.snapshot()[1]["last_seen"], 99.9)
        self.assertTrue(self.ready()[0])
        self.consume("M1 seen=1 rx=27 age_ms=0 spd=2 torque=3", now=101.0)
        self.assertAlmostEqual(self.link.snapshot()[1]["last_seen"], 99.9)
        self.assertFalse(self.ready(now=101.0)[0])
        self.consume("M1 seen=1 rx=28 age_ms=10000 spd=2 torque=3", now=102.0)
        self.assertAlmostEqual(self.link.snapshot()[1]["last_seen"], 92.0)
        self.assertFalse(self.ready(now=102.0)[0])
        self.consume("M1 seen=1 rx=29 age_ms=10 spd=4 torque=5", now=103.0)
        self.assertTrue(self.ready(now=103.0)[0])
        self.assertAlmostEqual(self.link.snapshot()[1]["last_seen"], 102.99)

    def test_invalid_and_partial_sample_metadata_cannot_refresh_old_pair(self):
        self.consume("M1 seen=1 rx=4 age_ms=0 spd=2 torque=3")
        for line in ("M1 seen=0 rx=0 age_ms=999999 spd=0 torque=0",
                     "M1 seen=1 rx=5 age_ms=nan spd=2 torque=3",
                     "M1 seen=1 rx=5.5 age_ms=0 spd=2 torque=3",
                     "M1 seen=1 rx=5 age_ms=-1 spd=2 torque=3",
                     "M1 spd=2 torque=3"):
            with self.subTest(line=line):
                self.consume(line, now=100.1)
                self.assertFalse(self.ready(now=100.1)[0])
        # A complete measured pair is required before consuming a new count.
        self.consume("M1 seen=1 rx=5 age_ms=0 spd=4", now=100.2)
        self.assertFalse(self.ready(now=100.2)[0])
        self.consume("M1 seen=1 rx=5 age_ms=0 spd=4 torque=5", now=100.3)
        self.assertTrue(self.ready(now=100.3)[0])

    def test_counter_replay_is_not_new_feedback_but_uint32_wrap_is(self):
        self.consume("M1 seen=1 rx=4294967294 age_ms=0 spd=2 torque=3")
        self.consume("M1 seen=1 rx=1 age_ms=0 spd=4 torque=5", now=100.1)
        self.assertEqual(self.link.snapshot()[1]["speed"], 4)
        self.assertAlmostEqual(self.link.snapshot()[1]["last_seen"], 100.1)
        self.consume("M1 seen=1 rx=4294967294 age_ms=0 spd=20 torque=30", now=101.0)
        self.assertEqual(self.link.snapshot()[1]["speed"], 4)
        self.assertFalse(self.ready(now=101.0)[0])

    def test_device_faults_stop_once_latch_readiness_and_reject_motion(self):
        for fault_line in ("RM_C MOTOR_USB v2 fault=CAN_ERROR",
                           "CTRL STATUS link=1 fault=1 reason=CAN_ERROR M1=0",
                           "CTRL ERR CAN_ERROR"):
            with self.subTest(fault_line=fault_line):
                self.setUp()
                self.consume("M1 spd=2 torque=3")
                self.link.send_speeds([20, 0, 0, 0, 0, 0])
                self.consume(fault_line)
                self.assertEqual(self.link.diagnostics()["device_fault"], "CAN_ERROR")
                self.assertIn("设备故障", self.ready()[1])
                self.assertIsNone(self.link.snapshot()[1]["last_seen"])
                self.assertIsNone(self.link._lease_deadline)
                self.assertEqual(self.sock.sent[-1], b"CTRL STOP\n")
                count = len(self.sock.sent)
                self.consume(fault_line)
                self.consume("CTRL STATUS link=1 fault=0 reason=NONE M1=0")
                self.consume("RM_C MOTOR_USB v3 fault=NONE | M1 seen=1 rx=30 age_ms=0 spd=0 torque=0")
                self.assertEqual(len(self.sock.sent), count)
                self.assertFalse(self.ready()[0])
                with self.assertRaises(InterruptedError):
                    self.link.send_speeds([20, 0, 0, 0, 0, 0])
                # Zero/STOP remain available. The host never sends CLEAR.
                self.link.send_speeds([0] * 6)
                self.assertNotIn(b"CTRL CLEAR\n", self.sock.sent)

    def test_legacy_telemetry_and_healthy_status_still_work(self):
        self.consume("CTRL STATUS link=1 fault=0 reason=NONE")
        self.consume("CAN rx=0 | M1 spd=2 torque=3")
        self.assertTrue(self.ready()[0])
        self.assertIsNone(self.link.diagnostics()["device_fault"])
        self.assertEqual(self.sock.sent, [])

    def test_v2_stopped_reason_is_normal_and_does_not_poison_later_motion(self):
        self.consume("RM_C MOTOR_USB v2 fault=STOP | M1 seen=1 rx=4 age_ms=10 spd=0 torque=0")
        self.assertIsNone(self.link.diagnostics()["device_fault"])
        self.assertTrue(self.ready()[0])
        self.assertEqual(self.sock.sent, [])
        self.consume("RM_C MOTOR_USB v2 fault=NONE | M1 seen=1 rx=5 age_ms=0 spd=0 torque=0", now=100.1)
        self.assertTrue(self.ready(now=100.1)[0])
        self.link.send_speeds([20, 0, 0, 0, 0, 0])
        self.assertEqual(len(self.sock.sent), 1)
        self.assertTrue(self.sock.sent[0].startswith(b"CTRL SPEED M1=20"))
        # The numeric fault flag is authoritative, even with reason=STOP.
        self.consume("CTRL STATUS link=1 fault=1 reason=STOP")
        self.assertEqual(self.link.diagnostics()["device_fault"], "STOP")
        self.assertFalse(self.ready()[0])
        self.assertEqual(self.sock.sent[-1], b"CTRL STOP\n")
        with self.assertRaises(InterruptedError):
            self.link.send_speeds([20, 0, 0, 0, 0, 0])


class MotorFeedbackStateTests(unittest.TestCase):
    def check(self, motor, now=100.0, max_age_s=0.8, current_limit_raw=7000):
        return motor_feedback_state({1: motor} if motor is not None else {}, 1,
                                    max_age_s, current_limit_raw, now=now)

    def test_missing_fields_report_specific_failures(self):
        cases = [(None, "未收到该通道"), ({"last_seen": 100.0}, "速度缺失或无效"),
                 ({"speed": float("nan"), "current_raw": 3, "last_seen": 100}, "速度缺失或无效"),
                 ({"speed": 4, "last_seen": 100.0}, "电流缺失或无效"),
                 ({"speed": 4, "current_raw": float("inf"), "last_seen": 100}, "电流缺失或无效"),
                 ({"speed": 4, "current_raw": 3}, "时间戳缺失或无效"),
                 ({"speed": 4, "current_raw": 3, "last_seen": float("nan")}, "时间戳缺失或无效")]
        for motor, expected in cases:
            with self.subTest(motor=motor):
                ready, reason = self.check(motor)
                self.assertFalse(ready)
                self.assertIn(expected, reason)

    def test_age_limits_and_overcurrent_must_pass_before_ready(self):
        base = {"speed": -300, "current_raw": -500, "last_seen": 99.5}
        self.assertTrue(self.check(base)[0])
        for values, expected in (({"last_seen": 99.0}, "过时"),
                                 ({"last_seen": 100.1}, "当前时基"),
                                 ({"current_raw": 7000}, "电流超限"),
                                 ({"current_raw": -7001}, "电流超限")):
            with self.subTest(values=values):
                ready, reason = self.check({**base, **values})
                self.assertFalse(ready)
                self.assertIn(expected, reason)
        self.assertEqual(self.check({**base, "last_seen": 100})[1], "M1 遥测就绪")

    def test_invalid_time_limits_and_boolean_values_are_never_ready(self):
        base = {"speed": 1, "current_raw": 2, "last_seen": 100}
        for kwargs in ({"now": float("nan")}, {"max_age_s": -1}, {"current_limit_raw": 0}):
            self.assertFalse(self.check(base, **kwargs)[0])
        for field in base:
            self.assertFalse(self.check({**base, field: True})[0])


if __name__ == "__main__":
    unittest.main()
