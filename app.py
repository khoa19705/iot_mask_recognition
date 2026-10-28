"""Webcam-only mask detection app powered by an Ultralytics YOLO model."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file

try:
    from ultralytics import YOLO
except ImportError:  # Keep the UI able to explain a missing dependency.
    YOLO = None  # type: ignore[assignment,misc]


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "detections.jsonl"

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_MB", "8")) * 1024 * 1024

_model: Any = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_prediction_count = 0
_log_lock = threading.Lock()
_last_log_signature: tuple[int, ...] | None = None
_last_logged_at = 0.0
LOG_REPEAT_SECONDS = 10


def _model_path() -> Path:
    path = Path(os.getenv("MASK_MODEL_PATH", "best.pt"))
    return path if path.is_absolute() else BASE_DIR / path


def get_model() -> Any:
    """Load the configured model once, on the first inference request."""
    global _model
    if _model is not None:
        return _model
    path = _model_path()
    if YOLO is None:
        raise RuntimeError("Chưa cài Ultralytics. Hãy chạy: pip install -r requirements.txt")
    if not path.is_file():
        raise RuntimeError(f"Không tìm thấy model tại '{path}'. Hãy đặt file .pt vào đây hoặc cấu hình MASK_MODEL_PATH.")
    with _model_lock:
        if _model is None:
            _model = YOLO(str(path))
    return _model


def classify_label(label: str) -> str:
    normalized = label.lower().strip().replace("-", "_").replace(" ", "_")
    if any(token in normalized for token in ("incorrect", "improper", "wrong")):
        return "incorrect"
    if any(token in normalized for token in ("no_mask", "without_mask", "nomask", "unmasked")):
        return "no_mask"
    if "mask" in normalized or "masked" in normalized:
        return "mask"
    return "other"


def analyze_frame(frame: np.ndarray, confidence: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Mask classes can compete for the same face. Suppress overlapping boxes
    # across classes so one person is not counted as two different mask states.
    global _prediction_count
    queued_at = time.perf_counter()
    with _inference_lock:
        acquired_at = time.perf_counter()
        load_started = time.perf_counter()
        was_loaded = _model is not None
        model = get_model()
        loaded_at = time.perf_counter()
        result = model.predict(
            source=frame, conf=confidence, iou=0.45,
            agnostic_nms=True, verbose=False, device="cpu",
        )[0]
        predicted_at = time.perf_counter()
        cold_start = _prediction_count == 0
        _prediction_count += 1
    timing = {
        "queue_ms": round((acquired_at - queued_at) * 1000, 3),
        "model_load_ms": round((loaded_at - load_started) * 1000, 3) if not was_loaded else 0,
        "inference_ms": round((predicted_at - loaded_at) * 1000, 3),
        "cold_start": cold_start,
    }
    names = result.names
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return detections, timing
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        label = str(names.get(class_id, class_id) if isinstance(names, dict) else names[class_id])
        detections.append({
            "label": label,
            "category": classify_label(label),
            "confidence": round(float(box.conf[0].item()), 4),
            "box": box.xyxy[0].cpu().numpy().astype(int).tolist(),
        })
    return detections, timing


def summarize(detections: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"total": len(detections), "mask": 0, "no_mask": 0, "incorrect": 0, "other": 0}
    for detection in detections:
        summary[detection["category"]] += 1
    return summary


def record_detection_event(detections: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Persist changes in detected class counts, repeating unchanged states every 10 seconds."""
    global _last_log_signature, _last_logged_at
    if not detections:
        with _log_lock:
            _last_log_signature = None
        return None
    summary = summarize(detections)
    signature = (summary["mask"], summary["no_mask"], summary["incorrect"], summary["other"])
    now = time.monotonic()
    with _log_lock:
        if signature == _last_log_signature and now - _last_logged_at < LOG_REPEAT_SECONDS:
            return None
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "summary": summary,
            "detections": detections,
        }
        with LOG_FILE.open("a", encoding="utf-8") as log:
            log.write(json.dumps(event, ensure_ascii=False) + "\n")
        _last_log_signature = signature
        _last_logged_at = now
        return event


def read_recent_logs(limit: int = 30) -> tuple[list[dict[str, Any]], int]:
    if not LOG_FILE.exists():
        return [], 0
    with _log_lock:
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    events: list[dict[str, Any]] = []
    for line in reversed(lines):
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
        if len(events) >= limit:
            break
    return events, len(lines)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    path = _model_path()
    ready = path.is_file() and YOLO is not None
    return jsonify({"ready": ready, "loaded": _model is not None,
                    "environment": os.getenv("APP_ENV", "local"),
                    "model_file": path.name}), 200 if ready else 503


@app.get("/api/model")
def model_info():
    try:
        model = get_model()
        names = model.names
        labels = list(names.values()) if isinstance(names, dict) else list(names)
        categories = {classify_label(label) for label in labels}
        return jsonify({"labels": labels, "task": model.task,
                        "three_classes_ready": {"mask", "no_mask", "incorrect"} <= categories,
                        "parameters": sum(p.numel() for p in model.model.parameters()),
                        "model_bytes": _model_path().stat().st_size})
    except Exception:
        app.logger.exception("Model inspection failed")
        return jsonify({"error": "Không tải được model. Kiểm tra trọng số và log máy chủ."}), 503


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "Ảnh vượt giới hạn dung lượng cho phép."}), 413


@app.post("/api/detect")
def detect_webcam_frame():
    started_at = time.perf_counter()
    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "Không nhận được khung hình từ webcam."}), 400
    raw = np.frombuffer(uploaded.read(), dtype=np.uint8)
    if not raw.size:
        return jsonify({"error": "Ảnh webcam rỗng."}), 400
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Không đọc được khung hình webcam."}), 400
    try:
        confidence = float(request.form.get("confidence", "0.35"))
        if not math.isfinite(confidence) or not 0.05 <= confidence <= 0.95:
            raise ValueError()
    except ValueError:
        return jsonify({"error": "Ngưỡng tin cậy phải từ 0.05 đến 0.95."}), 400
    try:
        detections, timing = analyze_frame(frame, confidence)
    except Exception:
        app.logger.exception("Inference failed")
        return jsonify({"error": "Không chạy được model. Kiểm tra trọng số và log máy chủ."}), 503
    logged_event = record_detection_event(detections)
    timing["server_ms"] = round((time.perf_counter() - started_at) * 1000, 3)
    return jsonify({
        "detections": detections,
        "summary": summarize(detections),
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "logged": logged_event is not None,
        "timing": timing,
        "environment": os.getenv("APP_ENV", "local"),
    })


@app.get("/api/logs")
def recent_logs():
    events, total = read_recent_logs()
    return jsonify({"events": events, "total": total})


@app.get("/api/logs/download")
def download_logs():
    if not LOG_FILE.exists():
        LOG_FILE.touch()
    return send_file(LOG_FILE, as_attachment=True, download_name="masklens-detections.jsonl", mimetype="application/x-ndjson")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG", "0") == "1", threaded=True)
