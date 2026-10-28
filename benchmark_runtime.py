"""Fresh-process API smoke benchmark. No webcam or accuracy claims."""
import argparse
import io
import json
import os
import sys
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    os.environ["MASK_MODEL_PATH"] = args.model
    import cv2
    import numpy as np
    import psutil
    import app
    process = psutil.Process()
    client = app.app.test_client()
    metadata = client.get("/api/model")
    assert metadata.status_code == 200, metadata.json
    _, encoded = cv2.imencode(".jpg", np.zeros((480, 640, 3), np.uint8))
    samples = []
    for _ in range(6):
        started = time.perf_counter()
        response = client.post("/api/detect", data={"file": (io.BytesIO(encoded.tobytes()), "blank.jpg")})
        assert response.status_code == 200, response.json
        samples.append({"timing": response.json["timing"], "api_ms": (time.perf_counter() - started) * 1000,
                        "rss_mb": process.memory_info().rss / 1024 ** 2})
    memory = process.memory_info()
    print(json.dumps({"model": metadata.json, "samples": samples,
                      "peak_rss_mb": getattr(memory, "peak_wset", memory.rss) / 1024 ** 2,
                      "torch_loaded": "torch" in sys.modules}, indent=2))
