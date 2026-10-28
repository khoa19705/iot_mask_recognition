"""CPU-only inference for the exported YOLOv8-style detection head (no Torch)."""
import ast
import os
import cv2
import numpy as np


def decode_predictions(output, names, source_shape, ratio, padding, confidence):
    """Decode [1,4+classes,anchors], suppress boxes across classes, undo letterbox."""
    predictions = np.asarray(output)
    if predictions.ndim != 3 or predictions.shape[0] != 1 or predictions.shape[1] != 4 + len(names):
        raise ValueError("Unsupported ONNX output; export with nms=False.")
    rows = predictions[0].T
    classes = np.argmax(rows[:, 4:], axis=1)
    scores = rows[np.arange(len(rows)), classes + 4]
    valid = np.isfinite(rows).all(axis=1) & (scores >= confidence) & (rows[:, 2] > 0) & (rows[:, 3] > 0)
    rows, scores, classes = rows[valid], scores[valid], classes[valid]
    if not len(rows):
        return []
    # Bound NMS work even when a low threshold produces many candidates.
    order = np.argsort(scores)[::-1][:3000]
    rows, scores, classes = rows[order], scores[order], classes[order]
    xywh = rows[:, :4].copy()
    xywh[:, :2] -= xywh[:, 2:] / 2
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), confidence, .45, top_k=3000)
    height, width = source_shape[:2]
    detections = []
    for index in np.asarray(keep).reshape(-1)[:100]:
        left, top, box_width, box_height = xywh[index]
        coordinates = np.array([left, top, left + box_width, top + box_height], dtype=np.float32)
        coordinates -= np.array([padding[0], padding[1], padding[0], padding[1]])
        coordinates /= ratio
        coordinates[[0, 2]] = np.clip(coordinates[[0, 2]], 0, width)
        coordinates[[1, 3]] = np.clip(coordinates[[1, 3]], 0, height)
        if coordinates[2] <= coordinates[0] or coordinates[3] <= coordinates[1]:
            continue
        detections.append({"label": names[int(classes[index])],
                           "confidence": round(float(scores[index]), 4),
                           "box": coordinates.astype(int).tolist()})
    return detections


class OnnxDetector:
    def __init__(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(os.getenv("INFERENCE_THREADS", "1")))
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.enable_cpu_mem_arena = False
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.height, self.width = model_input.shape[2:]
        if not all(isinstance(value, int) and value > 0 for value in (self.height, self.width)):
            raise ValueError("Use a static ONNX export.")
        metadata = self.session.get_modelmeta().custom_metadata_map
        names = ast.literal_eval(metadata["names"])
        self.names = {int(key): str(value) for key, value in names.items()}
        if set(self.names) != set(range(len(self.names))):
            raise ValueError("Model class IDs must be contiguous.")
        self.task = metadata.get("task", "detect")
        if self.task != "detect":
            raise ValueError("Only object detection models are supported.")
        self.parameters = int(metadata.get("source_parameters", "0"))
        cv2.setNumThreads(1)

    def detect(self, frame, confidence):
        height, width = frame.shape[:2]
        ratio = min(self.width / width, self.height / height)
        resized_width, resized_height = round(width * ratio), round(height * ratio)
        left = round((self.width - resized_width) / 2 - .1)
        top = round((self.height - resized_height) / 2 - .1)
        image = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        image = cv2.copyMakeBorder(image, top, self.height - resized_height - top,
                                  left, self.width - resized_width - left,
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114))
        tensor = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= 255.0
        output = self.session.run(None, {self.input_name: tensor})[0]
        return decode_predictions(output, self.names, frame.shape, ratio, (left, top), confidence)
