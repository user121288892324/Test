from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a YOLO COCO model to ONNX for the browser."
    )

    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Ultralytics model name or local .pt path.",
    )

    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT /
            "models" /
            "traffic-light.onnx"
        ),
        help="Destination ONNX file.",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    destination = Path(
        arguments.output
    ).resolve()

    destination.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    model = YOLO(arguments.model)

    exported_path_value = model.export(
        format="onnx",
        imgsz=640,
        opset=12,
        simplify=True,
        dynamic=False,
        half=False,
    )

    exported_path = Path(
        exported_path_value
    ).resolve()

    shutil.copy2(
        exported_path,
        destination
    )

    print(f"Export source: {exported_path}")
    print(f"Browser model: {destination}")
    print("COCO traffic-light class ID: 9")


if __name__ == "__main__":
    main()
