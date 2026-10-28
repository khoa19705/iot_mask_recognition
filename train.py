"""Train a YOLO detector from a dataset YAML file."""

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện YOLO phát hiện khẩu trang.")
    parser.add_argument("--data", required=True, help="Đường dẫn đến file dataset.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="Model nền hoặc trọng số có sẵn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Ví dụ: 0 cho GPU đầu tiên, cpu cho CPU")
    parser.add_argument("--name", default="mask-detection")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/detect",
        name=args.name,
    )


if __name__ == "__main__":
    main()
