import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

from spa_control.web_global_camera import WebGlobalCamera, GlobalCameraError
from spa_control.camera import CameraSample


class Capture:
    def __init__(self, *args):
        self.closed = False
    def isOpened(self):
        return True
    def set(self, *args):
        return True
    def read(self):
        time.sleep(.05)
        return (True, np.zeros((240, 320, 3), dtype=np.uint8))
    def release(self):
        self.closed = True


class Tracker:
    def __init__(self):
        self.x = 10
    def init(self, frame, roi):
        self.x = roi[0]
        return True
    def update(self, frame):
        self.x += 1
        return True, (self.x, 30, 20, 20)


FAKE_CV = types.SimpleNamespace(VideoCapture=Capture, TrackerCSRT_create=Tracker,
    CAP_DSHOW=0, CAP_ANY=0, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
    CAP_PROP_FPS=5, CAP_PROP_BUFFERSIZE=6, IMWRITE_JPEG_QUALITY=1,
    imencode=lambda *args: (True, np.array([255, 216, 255, 217], dtype=np.uint8)))


def wait_for(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError('timed out')


class WebGlobalCameraTest(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules, {'cv2': FAKE_CV})
        self.modules.start()
        self.camera = WebGlobalCamera()

    def tearDown(self):
        self.camera.close()
        self.modules.stop()

    def connected(self):
        self.camera.connect(0, 320, 240, 20)
        wait_for(lambda: self.camera.status()['frame_size'] == [320, 240])

    def calibrate(self):
        return self.camera.set_calibration([
            {'image_x_px': 0, 'image_y_px': 40, 'position_m': 0},
            {'image_x_px': 100, 'image_y_px': 40, 'position_m': 1}], 'left')

    def test_synthetic_capture_calibration_tracking_and_disconnect(self):
        self.connected()
        self.assertTrue(self.camera.latest_frame().startswith(b'\xff\xd8'))
        self.assertFalse(self.camera.status()['valid'])
        self.calibrate()
        self.camera.set_roi([10, 30, 20, 20])
        wait_for(lambda: self.camera.status()['valid'])
        status = self.camera.status()
        self.assertGreater(status['velocity_mps'], .1)
        self.assertLess(status['velocity_mps'], .3)
        self.assertEqual(status['clock'], 'host_monotonic_frame_read')
        self.camera.disconnect()
        self.assertFalse(self.camera.status()['valid'])
        self.assertIsNone(self.camera.status()['velocity_mps'])
        self.assertIsNone(self.camera.latest_frame())

    def test_bad_calibration_roi_and_clearing_fail_closed(self):
        self.connected()
        with self.assertRaises(GlobalCameraError):
            self.camera.set_roi([10, 30, 20, 20])
        with self.assertRaises(GlobalCameraError):
            self.camera.set_calibration([{'image_x_px': -1, 'image_y_px': 40, 'position_m': 0},
                                         {'image_x_px': 100, 'image_y_px': 40, 'position_m': 1}])
        self.calibrate()
        with self.assertRaises(GlobalCameraError):
            self.camera.set_roi([315, 30, 20, 20])
        self.camera.set_roi([10, 30, 20, 20])
        wait_for(lambda: self.camera.status()['valid'])
        self.camera.clear_roi()
        self.assertFalse(self.camera.status()['valid'])
        time.sleep(.1)
        self.assertFalse(self.camera.status()['valid'])

    def test_recalibration_invalidates_measurement_and_old_tracker(self):
        self.connected()
        self.calibrate()
        self.camera.set_roi([10, 30, 20, 20])
        wait_for(lambda: self.camera.status()['valid'])
        self.calibrate()
        time.sleep(.1)
        self.assertFalse(self.camera.status()['valid'])

    def test_age_and_tracking_loss_never_return_speed(self):
        self.connected()
        old = time.monotonic() - 2
        with self.camera._lock:
            self.camera._sample = CameraSample(old, old, .4, .2, True, 'test')
            self.camera._sample_revision = self.camera._revision
        self.assertFalse(self.camera.status()['valid'])
        self.assertIsNone(self.camera.status()['velocity_mps'])
        now = time.monotonic()
        with self.camera._lock:
            self.camera._sample = CameraSample(now, now, None, None, False, 'test', 'lost')
            self.camera._sample_revision = self.camera._revision
        self.assertFalse(self.camera.status()['valid'])
        self.assertIsNone(self.camera.status()['position_m'])

    def test_inflight_tracker_result_cannot_restore_cleared_tracking(self):
        entered, release = threading.Event(), threading.Event()
        class BlockedTracker(Tracker):
            def update(self, frame):
                entered.set()
                release.wait(2)
                return super().update(frame)
        with patch.object(FAKE_CV, 'TrackerCSRT_create', BlockedTracker):
            self.connected()
            self.calibrate()
            self.camera.set_roi([10, 30, 20, 20])
            self.assertTrue(entered.wait(2))
            self.camera.clear_roi()
            release.set()
            time.sleep(.15)
            status = self.camera.status()
            self.assertFalse(status['valid'])
            self.assertIsNone(status['velocity_mps'])

    def test_reselecting_head_invalidates_previous_speed_until_warmup(self):
        self.connected()
        self.calibrate()
        self.camera.set_roi([10, 30, 20, 20])
        wait_for(lambda: self.camera.status()['valid'])
        self.camera.set_roi([40, 30, 20, 20])
        self.assertFalse(self.camera.status()['valid'])
        self.assertIsNone(self.camera.status()['velocity_mps'])
        wait_for(lambda: self.camera.status()['valid'])

    def test_revision_identifies_the_calibration_and_roi_session(self):
        initial = self.camera.status()['revision']
        self.connected()
        connected = self.camera.status()['revision']
        self.assertGreater(connected, initial)
        self.calibrate()
        calibrated = self.camera.status()['revision']
        self.assertGreater(calibrated, connected)
        self.camera.set_roi([10, 30, 20, 20])
        selected = self.camera.status()['revision']
        self.assertGreater(selected, calibrated)
        self.assertIsNone(self.camera.status()['sample_revision'])
        wait_for(lambda: self.camera.status()['valid'])
        self.assertEqual(self.camera.status()['sample_revision'], selected)
        self.camera.clear_roi()
        cleared = self.camera.status()
        self.assertGreater(cleared['revision'], selected)
        self.assertFalse(cleared['valid'])
        self.assertIsNone(cleared['sample_revision'])
        self.camera.disconnect()
        self.assertGreater(self.camera.status()['revision'], cleared['revision'])
        self.connected()
        self.assertGreater(self.camera.status()['revision'], cleared['revision'])


if __name__ == '__main__':
    unittest.main()
