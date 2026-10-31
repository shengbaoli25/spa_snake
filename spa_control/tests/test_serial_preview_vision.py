import io
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from spa_control.serial_preview_vision import MobileNetWindowEncoder, VisionModelError


def jpeg_bytes():
    image = Image.new("RGB", (24, 16), (20, 100, 180))
    target = io.BytesIO()
    image.save(target, "JPEG")
    return target.getvalue()


class VisionTests(unittest.TestCase):
    def test_real_embeddings_are_frozen_and_deterministic(self):
        import torch
        old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            model = MobileNetWindowEncoder()
            data = jpeg_bytes()
            frames = [{"jpeg": data, "receive_monotonic_ns": 100},
                      {"jpeg": data, "receive_monotonic_ns": 101}]
            with patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("network")):
                result = model.encode_window(frames, 102)
                again = model.encode_window(frames[:1], 102)
            self.assertEqual(result["valid_count"], 2)
            self.assertEqual(len(result["frames"][0]["features"]), 576)
            self.assertEqual(result["frames"][0]["features"], result["frames"][1]["features"])
            # Batch size may cause very small floating-point differences.
            for a, b in zip(result["frames"][0]["features"], again["frames"][0]["features"]):
                self.assertAlmostEqual(a, b, places=5)
            self.assertFalse(model._model.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in model._model.parameters()))
        finally:
            torch.set_num_threads(old_threads)

    def test_cutoff_and_missing_values_do_not_invent_features(self):
        model = MobileNetWindowEncoder()
        frames = [{"receive_monotonic_ns": i, "jpeg": None} for i in range(10)]
        frames.extend([{"receive_monotonic_ns": 100, "jpeg": jpeg_bytes()},
                       {"receive_monotonic_ns": None, "jpeg": jpeg_bytes()}])
        with patch.object(model, "_load", side_effect=AssertionError("should not load model")):
            result = model.encode_window(frames, 9)
            empty = model.encode_window([], 9)
        self.assertEqual([f["receive_monotonic_ns"] for f in result["frames"]], list(range(2, 10)))
        self.assertEqual(result["future_frames_excluded"], 1)
        self.assertEqual(result["invalid_timestamps_excluded"], 1)
        self.assertEqual(result["valid_count"], 0)
        self.assertEqual(result["frames"][0]["age_s"], 7e-9)
        self.assertTrue(all(not f["valid"] and f["features"] is None for f in result["frames"]))
        self.assertEqual(empty["frames"], [])
        self.assertEqual(empty["valid_count"], 0)

    def test_bad_image_and_bad_arguments(self):
        model = MobileNetWindowEncoder()
        result = model.encode_window([{"receive_monotonic_ns": 1, "jpeg": b"bad"}], 2)
        self.assertEqual(result["frames"][0]["error"], "invalid_jpeg")
        for bad in [-1, True, 1.5]:
            with self.assertRaises(ValueError):
                model.encode_window([], bad)
        with self.assertRaises(ValueError):
            model.encode_window([], 9, max_frames=9)

    def test_missing_weight_reports_official_source_without_download(self):
        import torch
        model = MobileNetWindowEncoder()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(torch.hub, "get_dir", return_value=directory), \
                patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("network")):
            with self.assertRaisesRegex(VisionModelError, "download.pytorch.org"):
                model.encode_window([{"receive_monotonic_ns": 1, "jpeg": jpeg_bytes()}], 1)


if __name__ == "__main__":
    unittest.main()
