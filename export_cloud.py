"""Export the existing trained model for the low-memory Cloud runtime."""
from pathlib import Path
import hashlib
from ultralytics import YOLO
import onnx


if __name__ == "__main__":
    source = Path("best.pt")
    model = YOLO(str(source))
    parameters = sum(value.numel() for value in model.model.parameters())
    result = model.export(format="onnx", imgsz=320, opset=17, simplify=False,
                          dynamic=False, nms=False, device="cpu")
    graph = onnx.load(result)
    metadata = {item.key: item.value for item in graph.metadata_props}
    metadata["source_parameters"] = str(parameters)
    metadata["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    onnx.helper.set_model_props(graph, metadata)
    onnx.checker.check_model(graph)
    onnx.save(graph, result)
    if Path(result).stat().st_size >= 100 * 1024 * 1024:
        raise RuntimeError("ONNX file exceeds GitHub's regular file limit.")
    print(f"Cloud model: {result}, {Path(result).stat().st_size:,} bytes")
