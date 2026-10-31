import http.client
import json
import threading
import unittest

from spa_control.web_server import LocalServer, Workbench


class FakeCamera:
    def __init__(self):
        self.active = False
        self.light = False
        self.opens = []

    def ports(self):
        return [{"device": "COM14", "friendly_name": "SPA-CAM", "preferred": True,
                 "available_for_camera": True}]

    def connect(self, port):
        if port != "COM14":
            raise ValueError("unknown port")
        self.opens.append(port)
        self.active = True
        return self.status()

    def disconnect(self):
        self.active = False
        return self.status()

    def set_light(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError("boolean required")
        self.light = enabled
        return self.status()

    def status(self):
        return {"active": self.active, "message": "fake", "port": "COM14",
                "light": self.light, "frames_received": int(self.active),
                "frame_age_s": 0.2 if self.active else None,
                "frame_stale": not self.active}

    def latest_frame(self):
        return {"jpeg": b"\xff\xd8test\xff\xd9", "sequence": 7, "age_s": .2} if self.active else None


class WebServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.camera = FakeCamera()
        cls.server = LocalServer(("127.0.0.1", 0), Workbench(camera=cls.camera))
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=2)
        if body is not None:
            body = json.dumps(body).encode()
            headers = {"Content-Type": "application/json", **(headers or {})}
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        status, content = response.status, response.read()
        connection.close()
        return status, content

    def test_status_and_simulation_are_local_and_do_not_open_camera(self):
        self.camera.disconnect()
        before = list(self.camera.opens)
        status, data = self.request("GET", "/api/status")
        self.assertEqual(status, 200)
        parsed = json.loads(data)
        self.assertTrue(parsed["simulation"]["synthetic"])
        self.assertFalse(parsed["simulation"]["trained"])
        status, data = self.request("POST", "/api/simulation/reset", {"seed": 8})
        self.assertEqual(status, 200)
        status, data = self.request("POST", "/api/simulation/step", {})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["step"], 1)
        status, data = self.request("POST", "/api/simulation/train", {})
        self.assertEqual(status, 200)
        trained = json.loads(data)
        self.assertEqual(trained["train_samples"], 240)
        self.assertEqual(trained["policy_version"], 1)
        self.assertTrue(trained["synthetic"])
        self.assertFalse(trained["trained"])
        self.assertEqual(self.camera.opens, before)
        self.assertEqual(self.request("POST", "/api/simulation/reset", {"seed": -1})[0], 400)
        self.assertEqual(self.request("POST", "/api/motors/start", {})[0], 404)

    def test_camera_requires_explicit_connection_and_rejects_cross_origin(self):
        self.camera.disconnect()
        self.assertEqual(self.request("GET", "/api/camera/frame")[0], 503)
        self.assertEqual(self.request("POST", "/api/camera/connect", {"port": "COM14"},
                                      {"Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.request("POST", "/api/camera/connect", {"port": "COM14"})[0], 200)
        code, frame = self.request("GET", "/api/camera/frame")
        self.assertEqual(code, 200)
        self.assertTrue(frame.startswith(b"\xff\xd8"))
        self.assertEqual(self.request("POST", "/api/camera/disconnect", {})[0], 200)
        self.assertEqual(self.request("GET", "/api/camera/frame")[0], 503)

    def test_host_and_static_path_restrictions(self):
        self.assertEqual(self.request("GET", "/")[0], 200)
        self.assertEqual(self.request("GET", "/../AGENTS.md")[0], 404)
        self.assertEqual(self.request("GET", "/", headers={"Host": "outside.example"})[0], 403)


if __name__ == "__main__":
    unittest.main()
