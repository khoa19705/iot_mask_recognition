"""Optional real-model checks; use the export environment with Torch and ORT."""
import importlib.util
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np


@unittest.skipUnless(Path("best.onnx").exists() and importlib.util.find_spec("onnxruntime"), "Requires exported model and ORT")
class RuntimeTests(unittest.TestCase):
    def test_actual_onnx_api_and_metadata(self):
        import app
        with patch.dict(os.environ, {"MASK_MODEL_PATH": "best.onnx"}):
            with patch.object(app, "_model", None):
                client = app.app.test_client()
                meta = client.get("/api/model")
                self.assertEqual(meta.status_code, 200)
                self.assertEqual(meta.json["backend"], "onnxruntime")
                self.assertTrue(meta.json["three_classes_ready"])
                _, encoded = cv2.imencode(".jpg", np.zeros((240, 320, 3), np.uint8))
                result = client.post("/api/detect", data={"file": (io.BytesIO(encoded.tobytes()), "blank.jpg")})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json["input_size"], 320)
                self.assertEqual(result.json["summary"]["total"], 0)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("ultralytics"), "Requires original model runtime")
    def test_export_raw_outputs_match_original_model_at_same_size(self):
        import torch
        from ultralytics import YOLO
        from onnx_detector import OnnxDetector
        torch.set_num_threads(1)
        original = YOLO("best.pt").model.float().eval().fuse(verbose=False)
        exported = OnnxDetector("best.onnx")
        tensor = np.random.default_rng(2026).random((1, 3, 320, 320), dtype=np.float32)
        with torch.inference_mode():
            expected = original(torch.from_numpy(tensor))[0].numpy()
        actual = exported.session.run(None, {exported.input_name: tensor})[0]
        np.testing.assert_allclose(actual, expected, rtol=.001, atol=.01)


if __name__ == "__main__":
    unittest.main()
