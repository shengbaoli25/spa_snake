"""HTTP control integration uses fakes; never opens hardware."""
import http.client
import json
import threading
import time
import unittest

from spa_control.web_server import LocalServer, Workbench
from spa_control.tests.test_web_server import FakeCamera
from spa_control.tests.test_web_two_motor_api import FakeMotor, FakeRecorder
from spa_control.tests.test_web_global_api import FakeGlobal


class Motor(FakeMotor):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.motion = 'idle'

    def status(self):
        return {**super().status(), 'motion': {'state': self.motion}}

    def start_test(self, motor, rpm, duration_s):
        self.events.append(('test', motor, rpm, duration_s))
        self.motion = 'preparing'
        return self.status()

    def stop_motion(self, reason):
        self.events.append(('motor_stop', reason))
        self.motion = 'completed'
        return self.status()


class Experiment:
    def __init__(self, events):
        self.events = events
        self.active = False

    def status(self):
        return {'active': self.active, 'state': 'running' if self.active else 'idle'}

    def configure(self, data):
        self.events.append(('config', data))
        return self.status()

    def start(self, mode, options):
        self.events.append(('experiment_start', mode, options))
        self.active = True
        return self.status()

    def pretrain(self, options):
        self.events.append(('pretrain', options))
        self.active = True
        return self.status()

    def stop(self):
        self.events.append(('experiment_stop',))
        self.active = False
        return self.status()

    def close(self):
        self.active = False


class WebControlAPITests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.motor = Motor(self.events)
        self.experiment = Experiment(self.events)
        self.workbench = Workbench(camera=FakeCamera(), motor=self.motor,
            recorder=FakeRecorder(), global_camera=FakeGlobal(), experiment=self.experiment)
        self.server = LocalServer(('127.0.0.1', 0), self.workbench)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.workbench.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self, path, data=None, origin=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=3)
        headers = {'Content-Type': 'application/json'}
        if origin:
            headers['Origin'] = origin
        conn.request('POST', path, json.dumps(data or {}), headers)
        response = conn.getresponse()
        result = response.status, json.loads(response.read())
        conn.close()
        return result

    def test_individual_test_defaults_and_stop_order(self):
        self.assertEqual(self.request('/api/motor/test', {'motor': 4})[0], 200)
        self.assertEqual(self.events, [('test', 4, -60, 3)])
        self.assertEqual(self.request('/api/motor/stop')[0], 200)
        self.assertEqual(self.events[-2:], [('experiment_stop',), ('motor_stop', '网页停止')])

    def test_mutually_exclusive_manual_test_and_real_session(self):
        self.assertEqual(self.request('/api/experiment/start', {'mode': 'collect'})[0], 200)
        self.assertEqual(self.request('/api/motor/test', {'motor': 3})[0], 400)
        self.request('/api/experiment/stop')
        self.assertEqual(self.request('/api/motor/test', {'motor': 3})[0], 200)
        for path in ('configure', 'start', 'pretrain'):
            self.assertEqual(self.request('/api/experiment/' + path)[0], 400)

    def test_cross_origin_cannot_start_or_renew(self):
        for path in ('/api/motor/test', '/api/experiment/start', '/api/control/heartbeat'):
            self.assertEqual(self.request(path, {'motor': 3}, 'https://example.com')[0], 403)
        self.assertEqual(self.events, [])

    def test_lost_page_heartbeat_stops_both_producer_and_motor(self):
        self.request('/api/motor/test', {'motor': 3})
        with self.workbench._control_lock:
            self.workbench._control_at -= 3
        deadline = time.monotonic() + 1
        while not any(e[0] == 'motor_stop' for e in self.events) and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(self.events[-2:], [('experiment_stop',), ('motor_stop', '网页心跳超时')])

    def test_geometry_defaults_match_confirmed_machine(self):
        code, result = self.request('/api/kinematics/preview', {'motor_rpm': 100})
        self.assertEqual(code, 200)
        self.assertEqual(result['lead_mm'], 30)
        self.assertEqual(result['motor_to_helix_ratio'], 1)
        self.assertAlmostEqual(result['theory_speed_mps'], .05)
        self.assertEqual(self.events, [])

    def test_watchdog_survives_completed_test_and_shutdown_rejects_start(self):
        self.request('/api/motor/test', {'motor': 3})
        self.request('/api/motor/stop')
        time.sleep(.15)
        self.assertTrue(self.workbench._control_watchdog.is_alive())
        self.request('/api/motor/test', {'motor': 4})
        with self.workbench._control_lock:
            self.workbench._control_at -= 3
        deadline = time.monotonic() + 1
        while self.motor.motion != 'completed' and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(self.events[-1], ('motor_stop', '网页心跳超时'))
        self.workbench.close()
        self.assertEqual(self.request('/api/motor/test', {'motor': 3})[0], 400)


if __name__ == '__main__':
    unittest.main()
