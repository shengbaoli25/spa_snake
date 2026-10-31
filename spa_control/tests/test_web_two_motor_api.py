"""HTTP contract tests for passive two-motor verification; no hardware I/O."""
import http.client
import json
import threading
import unittest
from unittest.mock import patch

from spa_control.web_server import LocalServer, Workbench
from spa_control.tests.test_web_server import FakeCamera


class FakeMotor:
    def __init__(self):
        self.state = "disconnected"
        self.calls = []

    def status(self):
        return {"state": self.state, "active": self.state == "connected",
                "motors": {"3": {"spd": -12, "fresh": True},
                           "4": {"spd": 13, "fresh": True}},
                "fault": "NONE", "stop_acknowledged": True}

    def connect(self):
        self.calls.append("connect")
        self.state = "connected"
        return self.status()

    def disconnect(self):
        self.calls.append("disconnect")
        self.state = "disconnected"
        return self.status()

    def clear_usb_closed(self):
        self.calls.append("clear_usb_closed")
        return {"clear": {"state": "checking"}}

    def close(self):
        self.calls.append("close")


class FakeRecorder:
    def __init__(self):
        self.calls = []
        self.active = False

    def status(self):
        return {"active": self.active, "state": "recording" if self.active else "idle",
                "directory": None, "samples": 0, "images": 0}

    def start(self, duration_s):
        if type(duration_s) not in (int, float) or not 2 <= duration_s <= 30:
            raise ValueError("duration must be 2 to 30 seconds")
        self.calls.append(("start", duration_s))
        self.active = True
        return self.status()

    def stop(self):
        self.calls.append("stop")
        self.active = False
        return self.status()

    def close(self):
        self.calls.append("close")


class WebTwoMotorAPITests(unittest.TestCase):
    def setUp(self):
        self.camera, self.motor, self.recorder = FakeCamera(), FakeMotor(), FakeRecorder()
        self.workbench = Workbench(camera=self.camera, motor=self.motor, recorder=self.recorder)
        self.server = LocalServer(("127.0.0.1", 0), self.workbench)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.workbench.close()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        if body is not None:
            body = json.dumps(body).encode()
            headers = {"Content-Type": "application/json", **(headers or {})}
        try:
            connection.request(method, path, body, headers or {})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_status_is_passive_and_connect_is_explicit(self):
        code, status = self.request("GET", "/api/status")
        self.assertEqual(code, 200)
        self.assertEqual(status["motor"]["state"], "disconnected")
        self.assertFalse(status["recorder"]["active"])
        self.assertEqual(self.motor.calls, [])
        self.assertEqual(self.recorder.calls, [])
        code, result = self.request("POST", "/api/motor/connect", {})
        self.assertEqual(code, 200)
        self.assertEqual(result["state"], "connected")
        self.assertEqual(self.motor.calls, ["connect"])
        code, result = self.request("GET", "/api/status")
        self.assertEqual(result["motor"]["motors"]["3"]["spd"], -12)
        self.assertEqual(result["motor"]["motors"]["4"]["spd"], 13)
        self.assertEqual(self.request("POST", "/api/motor/disconnect", {})[0], 200)
        self.assertEqual(self.motor.calls, ["connect", "disconnect"])

    def test_record_duration_validation_and_no_motor_commands(self):
        self.assertEqual(self.request("POST", "/api/record/start", {"duration_s": 1})[0], 400)
        self.assertEqual(self.request("POST", "/api/record/start", {"duration_s": True})[0], 400)
        self.assertEqual(self.request("POST", "/api/record/start", {"duration_s": 31})[0], 400)
        self.assertEqual(self.recorder.calls, [])
        code, result = self.request("POST", "/api/record/start", {"duration_s": 5})
        self.assertEqual(code, 200)
        self.assertTrue(result["active"])
        self.assertEqual(self.recorder.calls, [("start", 5)])
        self.assertTrue(self.request("GET", "/api/status")[1]["recorder"]["active"])
        self.assertEqual(self.request("POST", "/api/record/stop", {})[0], 200)
        self.assertEqual(self.motor.calls, [])

    def test_clear_requires_explicit_same_origin_post(self):
        path = "/api/motor/clear-usb-closed"
        self.assertEqual(self.request("POST", path, {}, {"Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.motor.calls, [])
        code, result = self.request("POST", path, {})
        self.assertEqual(code, 200)
        self.assertEqual(result["clear"]["state"], "checking")
        self.assertEqual(self.motor.calls, ["clear_usb_closed"])

    def test_two_motor_demo_explicit_hypotheses_and_no_physical_control(self):
        def demo(front_motor):
            if type(front_motor) is not int or front_motor not in (3, 4):
                raise ValueError("front motor must be 3 or 4")
            return {"front_motor": front_motor, "mapping_confirmed": False,
                    "drive_source": "synthetic", "physical_policy_allowed": False}
        with patch("spa_control.two_motor_preview.run_two_motor_demo", side_effect=demo):
            for front in (3, 4):
                code, result = self.request("POST", "/api/model/two", {"front_motor": front})
                self.assertEqual(code, 200)
                self.assertEqual(result["front_motor"], front)
                self.assertFalse(result["mapping_confirmed"])
                self.assertFalse(result["physical_policy_allowed"])
                self.assertEqual(result["drive_source"], "synthetic")
            self.assertEqual(self.request("POST", "/api/model/two", {"front_motor": 2})[0], 400)
        self.assertEqual(self.motor.calls, [])
        self.assertEqual(self.recorder.calls, [])

    def test_rejects_cross_origin_and_nonexistent_motion_commands(self):
        for path in ("/api/motor/connect", "/api/record/start", "/api/model/two"):
            self.assertEqual(self.request("POST", path, {}, {"Origin": "https://other.example"})[0], 403)
        for path in ("/api/motors/start", "/api/motor/start", "/api/motor/speed"):
            self.assertEqual(self.request("POST", path, {})[0], 404)
        self.assertEqual(self.motor.calls, [])
        self.assertEqual(self.recorder.calls, [])


if __name__ == "__main__":
    unittest.main()
