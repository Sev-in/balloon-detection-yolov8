from pathlib import Path
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "runs" / "balloon_yolov8n" / "weights" / "best.pt"


def main():
    model = YOLO(str(MODEL_PATH))

    model.export(
        format="onnx"
    )


if __name__ == "__main__":
    main()