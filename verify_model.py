"""Run real CPU inference on blank images; not an accuracy benchmark."""
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
import app


def main():
    client = app.app.test_client()
    metadata = client.get("/api/model")
    if metadata.status_code != 200:
        raise RuntimeError(metadata.json)
    ok, encoded = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
    assert ok
    samples = []
    for _ in range(5):
        response = client.post("/api/detect", data={
            "file": (io.BytesIO(encoded.tobytes()), "blank.jpg"), "confidence": "0.35",
        })
        if response.status_code != 200:
            raise RuntimeError(response.json)
        assert response.json["width"] == 640 and response.json["height"] == 480
        samples.append({"timing": response.json["timing"], "summary": response.json["summary"]})
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "environment": "local",
              "description": "Real model on five blank frames, CPU. Does not measure accuracy, webcam, browser latency or Cloud.",
              "model": metadata.json,
              "sha256": hashlib.sha256(app._model_path().read_bytes()).hexdigest(), "samples": samples}
    output = Path("outputs/model-verification.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
