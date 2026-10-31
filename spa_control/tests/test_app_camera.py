"""Camera/UI integration without a Tk window, physical camera or robot."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from spa_control.camera_workbench import Workbench
from spa_control.camera import RgbCameraSpeedSource


class AppCameraTests(unittest.TestCase):
    def roi_app(self, start):
        camera = Mock()
        app = SimpleNamespace(
            roi_start=start, frame_rect=(40, 80, .5), frame_shape=(720, 1280, 3),
            runner=SimpleNamespace(camera=camera, message=""),
            video=Mock(), roi_dragging=True,
        )
        return app, camera

    def test_letterboxed_preview_reverse_drag_returns_original_pixels(self):
        app, camera = self.roi_app((240, 230))
        Workbench.roi_release(app, SimpleNamespace(x=140, y=130))
        camera.set_roi.assert_called_once_with((200, 100, 200, 200))
        self.assertFalse(app.roi_dragging)
        self.assertIsNone(app.roi_start)

    def test_roi_crossing_letterbox_is_clipped_to_actual_frame(self):
        app, camera = self.roi_app((20, 60))
        Workbench.roi_release(app, SimpleNamespace(x=140, y=130))
        camera.set_roi.assert_called_once_with((0, 0, 200, 100))

    def test_drag_entirely_in_letterbox_does_not_select_robot(self):
        app, camera = self.roi_app((10, 20))
        Workbench.roi_release(app, SimpleNamespace(x=20, y=40))
        camera.set_roi.assert_not_called()

    def test_camera_device_open_occurs_in_background_thread(self):
        entered = threading.Event()
        release = threading.Event()
        thread_ids = []
        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.return_value = (False, None)

        def fake_video_capture(*args):
            thread_ids.append(threading.get_ident())
            entered.set()
            if not release.wait(2):
                raise RuntimeError("test camera release timed out")
            return capture

        cv2 = SimpleNamespace(
            TrackerCSRT_create=lambda: object(), VideoCapture=fake_video_capture,
            CAP_DSHOW=0, CAP_ANY=0, CAP_PROP_FRAME_WIDTH=1, CAP_PROP_FRAME_HEIGHT=2,
            CAP_PROP_FPS=3, CAP_PROP_BUFFERSIZE=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"calibration.json"
            path.write_text(json.dumps({
                "schema_version": 1, "track_type": "straight", "points": [
                    {"image_x_px": 0, "image_y_px": 0, "position_m": 0},
                    {"image_x_px": 100, "image_y_px": 0, "position_m": 1},
                ],
            }), encoding="utf-8")
            source = RgbCameraSpeedSource(path)
            self.assertFalse(entered.is_set(), "constructing a camera must not open hardware")
            with patch.dict("sys.modules", {"cv2": cv2}):
                try:
                    source.start()
                    self.assertTrue(entered.wait(1))
                    self.assertNotEqual(thread_ids[0], threading.get_ident())
                    self.assertIsNone(source.latest())
                finally:
                    release.set()
                    source.stop()
            self.assertIsNone(source.latest())


if __name__ == "__main__":
    unittest.main()
