import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import app as service
import start_cloud


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = service.app.test_client()
        self.temp = tempfile.TemporaryDirectory()
        self.log_patch = patch.object(service, "LOG_FILE", Path(self.temp.name) / "events.jsonl")
        self.log_patch.start()
        service._prediction_count = 0
        service._last_log_signature = None
        ok, image = cv2.imencode(".jpg", np.zeros((96, 128, 3), np.uint8))
        assert ok
        self.image = image.tobytes()

    def tearDown(self):
        self.log_patch.stop()
        self.temp.cleanup()

    def post(self, content=None, confidence="0.35"):
        return self.client.post("/api/detect", data={
            "file": (io.BytesIO(self.image if content is None else content), "frame.jpg"),
            "confidence": confidence,
        })

    def test_invalid_uploads(self):
        self.assertEqual(self.client.post("/api/detect").status_code, 400)
        self.assertEqual(self.post(b"").status_code, 400)
        self.assertEqual(self.post(b"not-an-image").status_code, 400)
        for confidence in ("nan", "inf", "bad", "0", "1"):
            self.assertEqual(self.post(confidence=confidence).status_code, 400)

    def test_large_upload_returns_json(self):
        with patch.dict(service.app.config, {"MAX_CONTENT_LENGTH": 32}):
            response = self.post()
            self.assertEqual(response.status_code, 413)
            self.assertIn("error", response.json)

    def test_real_timing_path_and_cold_start(self):
        model = SimpleNamespace(predict=lambda **kw: [SimpleNamespace(names={}, boxes=None)])
        with patch.object(service, "get_model", return_value=model):
            for cold in (True, False):
                response = self.post()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json["summary"]["total"], 0)
                timing = response.json["timing"]
                self.assertEqual(timing["cold_start"], cold)
                for key in ("inference_ms", "queue_ms", "model_load_ms", "server_ms"):
                    self.assertGreaterEqual(timing[key], 0)
                self.assertGreaterEqual(timing["server_ms"] + .01, timing["inference_ms"])
        self.assertFalse(service.LOG_FILE.exists())

    def test_multiple_classes_and_logs(self):
        detections = [{"label": label, "category": service.classify_label(label),
                       "confidence": .9, "box": [0, 0, 10, 10]} for label in
                      ["with_mask", "without_mask", "mask_weared_incorrect"]]
        with patch.object(service, "analyze_frame", return_value=(detections, {"inference_ms": 3})):
            response = self.post()
            self.assertEqual(response.json["summary"], {"total": 3, "mask": 1, "no_mask": 1, "incorrect": 1, "other": 0})
            self.assertTrue(response.json["logged"])
            self.assertFalse(self.post().json["logged"])
        self.assertEqual(self.client.get("/api/logs").json["total"], 1)
        download = self.client.get("/api/logs/download")
        self.assertEqual(download.status_code, 200)
        download.close()

    def test_inference_failure_is_json(self):
        with patch.object(service, "analyze_frame", side_effect=RuntimeError("private path")):
            with self.assertLogs(service.app.logger, level="ERROR"):
                response = self.post()
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private path", response.json["error"])

    def test_missing_model_health(self):
        with patch.object(service, "_model_path", return_value=Path(self.temp.name) / "missing.pt"):
            response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json["ready"])
        self.assertNotIn("model_path", response.json)

    def test_page_includes_metrics(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'metric-end_to_end_ms', response.data)
        self.assertIn(b'metrics.js', response.data)

    def test_busy_inference_rejected_without_queueing(self):
        service._request_slot.acquire()
        try:
            response = self.post()
            self.assertEqual(response.status_code, 429)
            self.assertIn("error", response.json)
            self.assertEqual(self.client.get("/").status_code, 200)
        finally:
            service._request_slot.release()


class DownloadTests(unittest.TestCase):
    def test_verified_download_and_reject_mismatch(self):
        import hashlib
        payload = b"test weights"
        for expected, succeeds in ((hashlib.sha256(payload).hexdigest(), True), ("0" * 64, False)):
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "best.pt"
                response = io.BytesIO(payload)
                response.url = "https://example.test/model"
                with patch.dict("os.environ", {"MASK_MODEL_PATH": str(target), "MODEL_URL": response.url, "MODEL_SHA256": expected}):
                    with patch("urllib.request.urlopen", return_value=response):
                        if succeeds:
                            start_cloud.prepare_model()
                            self.assertEqual(target.read_bytes(), payload)
                        else:
                            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                                start_cloud.prepare_model()
                            self.assertFalse(target.exists())
                            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
