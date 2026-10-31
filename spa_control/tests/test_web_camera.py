import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from spa_control.bluetooth_camera import CameraPacket
from spa_control.web_camera import CameraError, WebCamera, list_camera_ports, _bluetooth_name


class FakeReceiver:
    def __init__(self):
        self.lock = threading.Lock()
        self.light = False
        self.active = False
        self.latest = None
        self.calls = []

    def connect(self, port):
        self.calls.append(("connect", port))
        self.active = True

    def disconnect(self):
        self.calls.append(("disconnect",))
        self.active = False

    def snapshot(self):
        with self.lock:
            return self.active, "测试状态", self.latest, int(self.latest is not None)


class WebCameraTests(unittest.TestCase):
    def setUp(self):
        self.receiver = FakeReceiver()
        self.ports = [{"device": "COM14", "available_for_camera": True},
                      {"device": "COM9", "available_for_camera": False}]
        self.camera = WebCamera(self.receiver, lambda: self.ports)

    def test_explicit_connect_rejects_motor_board_and_unknown(self):
        self.assertEqual(self.receiver.calls, [])
        with self.assertRaises(CameraError):
            self.camera.connect("COM9")
        with self.assertRaises(CameraError):
            self.camera.connect("COM3")
        self.assertEqual(self.receiver.calls, [])
        self.assertTrue(self.camera.connect("com14")["active"])
        with self.assertRaises(CameraError):
            self.camera.connect("COM14")
        self.camera.disconnect()
        self.assertEqual(self.receiver.calls, [("connect", "COM14"), ("disconnect",)])

    def test_status_frame_and_light(self):
        packet = CameraPacket(42, 7, 1234, b"jpeg")
        self.receiver.latest = (packet, time.monotonic_ns(), None)
        frame = self.camera.latest_frame()
        self.assertEqual((frame["jpeg"], frame["sequence"], frame["capture_us"]),
                         (b"jpeg", 7, 1234))
        self.assertEqual(frame["clock_sync"], "not_calibrated")
        self.assertFalse(self.camera.status()["frame_stale"])
        self.assertTrue(self.camera.set_light(True)["light"])
        with self.assertRaises(CameraError):
            self.camera.set_light("true")

    def test_enumeration_marks_c_board_and_does_not_open_ports(self):
        motor = SimpleNamespace(device="COM9", description="C board", vid=0x0483,
                                pid=0x5740, hwid="USB VID:PID=0483:5740")
        camera = SimpleNamespace(device="COM14", description="BT SPP", vid=None,
                                 pid=None, hwid="BTHENUM")
        with patch("serial.tools.list_ports.comports", return_value=[motor, camera]):
            listed = list_camera_ports()
        self.assertFalse(listed[0]["available_for_camera"])
        self.assertTrue(listed[1]["available_for_camera"])
        self.assertEqual(listed[1]["friendly_name"], "BT SPP")
        self.assertFalse(listed[1]["preferred"])

    def test_known_address_moves_between_com_ports_and_unknown_falls_back(self):
        known = SimpleNamespace(device="COM37", description="serial", vid=None, pid=None,
                                hwid="BTHENUM\\{00001101}&0&20500D8D471A_C00000000")
        unknown = SimpleNamespace(device="COM38", description="other", vid=None, pid=None,
                                  hwid="BTHENUM\\{00001101}&0&ABCDEF123456_C00000000")
        with patch("serial.tools.list_ports.comports", return_value=[known, unknown]), \
                patch("spa_control.web_camera._bluetooth_name", return_value=None):
            listed = list_camera_ports()
        self.assertEqual(listed[0]["device"], "COM37")
        self.assertEqual(listed[0]["friendly_name"], "SPA-CAM")
        self.assertTrue(listed[0]["preferred"])
        self.assertEqual(listed[1]["friendly_name"], "other")
        self.assertFalse(listed[1]["preferred"])

    def test_device_name_evidence_and_platform_fallback(self):
        dev = SimpleNamespace(device="COM18", description="BT", vid=None, pid=None,
                              hwid="BTHENUM\\{00001101}&0&ABCDEF123456_C00000000")
        with patch("serial.tools.list_ports.comports", return_value=[dev]), \
                patch("spa_control.web_camera._bluetooth_name", return_value="SPA-CAM"):
            listed = list_camera_ports()
        self.assertTrue(listed[0]["preferred"])
        self.assertEqual(listed[0]["friendly_name"], "SPA-CAM")
        with patch.dict("sys.modules", {"winreg": None}):
            self.assertIsNone(_bluetooth_name("ABCDEF123456"))

    def test_windows_registry_name_bytes(self):
        class Key:
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
        registry = SimpleNamespace(HKEY_LOCAL_MACHINE=object(),
                                   OpenKey=lambda root, path: Key(),
                                   QueryValueEx=lambda key, field:
                                       (b"\x00" if field == "FriendlyName" else b"SPA-CAM\x00", 3))
        with patch.dict("sys.modules", {"winreg": registry}):
            self.assertEqual(_bluetooth_name("20500D8D471A"), "SPA-CAM")


if __name__ == "__main__":
    unittest.main()
