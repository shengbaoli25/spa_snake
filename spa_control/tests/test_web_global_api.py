"""HTTP integration checks; all devices are fakes and no motor is driven."""
import http.client
import json
import threading
import unittest

from spa_control.web_server import LocalServer, Workbench
from spa_control.tests.test_web_server import FakeCamera
from spa_control.tests.test_web_two_motor_api import FakeMotor, FakeRecorder


class FakeGlobal:
    def __init__(self):
        self.calls = []
        self.frame = None
        self.valid = False
        self.speed = None
        self.position = None

    def status(self):
        return {'valid': self.valid, 'velocity_mps': self.speed,
                'position_m': self.position, 'state': 'connected' if self.frame else 'disconnected'}

    def latest_frame(self):
        return self.frame

    def connect(self, camera_index, width, height, fps):
        self.calls.append(('connect', camera_index, width, height, fps))
        self.frame = b'\xff\xd8global\xff\xd9'
        return self.status()

    def disconnect(self):
        self.calls.append(('disconnect',))
        self.frame = None
        self.valid = False
        return self.status()

    def set_calibration(self, points, direction):
        self.calls.append(('calibration', points, direction))
        return self.status()

    def load_calibration(self, direction):
        self.calls.append(('load-calibration', direction))
        return self.status()

    def set_roi(self, bbox):
        self.calls.append(('roi', bbox))
        return self.status()

    def clear_roi(self):
        self.calls.append(('clear-roi',))
        return self.status()

    close = disconnect


class WebGlobalAPITests(unittest.TestCase):
    def setUp(self):
        self.global_camera, self.motor = FakeGlobal(), FakeMotor()
        self.workbench = Workbench(camera=FakeCamera(), motor=self.motor,
                                   recorder=FakeRecorder(), global_camera=self.global_camera)
        self.server = LocalServer(('127.0.0.1', 0), self.workbench)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.workbench.close()
        self.server.server_close()
        self.thread.join(2)

    def request(self, method, path, data=None, headers=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        body = None if data is None else json.dumps(data).encode()
        if body is not None:
            headers = {'Content-Type': 'application/json', **(headers or {})}
        try:
            conn.request(method, path, body, headers or {})
            response = conn.getresponse()
            return response.status, response.getheader('Content-Type'), response.read()
        finally:
            conn.close()

    def post(self, path, data):
        status, _, payload = self.request('POST', path, data)
        return status, json.loads(payload)

    def test_reference_units_and_missing_or_stale_tracking(self):
        payload = {'motor_rpm': 100, 'lead_mm': 30, 'motor_to_helix_ratio': 1,
                   'forward_sign': 1, 'distance_m': 1}
        code, result = self.post('/api/kinematics/preview', payload)
        self.assertEqual(code, 200)
        self.assertAlmostEqual(result['theory_speed_mps'], .05)
        self.assertAlmostEqual(result['ideal_duration_s'], 20)
        self.assertIsNone(result['tracking'])
        self.assertIsNone(result['camera_position_m'])
        self.global_camera.speed = .05
        self.global_camera.position = .3
        # Even a numerical but invalid sample cannot be treated as a reward.
        self.assertIsNone(self.post('/api/kinematics/preview', payload)[1]['tracking'])
        self.global_camera.valid = True
        code, result = self.post('/api/kinematics/preview', payload)
        self.assertEqual(code, 200)
        self.assertEqual(result['tracking']['tracking_reward'], 1)
        self.assertAlmostEqual(result['camera_position_m'], .3)
        self.assertEqual(self.motor.calls, [])
        self.assertEqual(self.post('/api/kinematics/preview', {**payload, 'lead_mm': 0})[0], 400)

    def test_explicit_camera_endpoints_and_jpeg(self):
        self.assertEqual(self.request('GET', '/api/global-camera/frame')[0], 503)
        self.assertEqual(self.post('/api/global-camera/connect', {'camera_index': 2})[0], 200)
        self.assertEqual(self.global_camera.calls[0], ('connect', 2, 1280, 720, 30))
        code, mime, data = self.request('GET', '/api/global-camera/frame')
        self.assertEqual((code, mime, data), (200, 'image/jpeg', b'\xff\xd8global\xff\xd9'))
        points = [{'image_x_px': 100, 'image_y_px': 50, 'position_m': 0},
                  {'image_x_px': 200, 'image_y_px': 50, 'position_m': .3}]
        for path, data in [('/api/global-camera/calibration', {'points': points, 'direction': 'right'}),
                           ('/api/global-camera/load-calibration', {'direction': 'left'}),
                           ('/api/global-camera/roi', {'bbox': [10, 20, 30, 40]}),
                           ('/api/global-camera/clear-roi', {})]:
            self.assertEqual(self.post(path, data)[0], 200)
        self.assertEqual(self.global_camera.calls[1:], [('calibration', points, 'right'),
                         ('load-calibration', 'left'), ('roi', [10, 20, 30, 40]), ('clear-roi',)])
        self.assertEqual(self.post('/api/global-camera/disconnect', {})[0], 200)
        self.assertEqual(self.request('GET', '/api/global-camera/frame')[0], 503)
        self.assertEqual(self.motor.calls, [])

    def test_cross_origin_and_host_rejected(self):
        for path in ('/api/global-camera/connect', '/api/global-camera/calibration',
                     '/api/global-camera/roi', '/api/kinematics/preview'):
            self.assertEqual(self.request('POST', path, {}, {'Origin': 'https://evil.example'})[0], 403)
        self.assertEqual(self.request('GET', '/api/global-camera/frame', headers={'Host': 'evil.example'})[0], 403)
        self.assertEqual(self.global_camera.calls, [])
        self.assertEqual(self.motor.calls, [])


if __name__ == '__main__':
    unittest.main()
