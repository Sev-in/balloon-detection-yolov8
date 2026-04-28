from pathlib import Path
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]

def main():
    model_path = ROOT_DIR / "runs" / "balloon_yolov8n" / "weights" / "best.pt"

    model = YOLO(str(model_path))

    model.predict(
        source=str(ROOT_DIR / "dataset_merged" / "images" / "test"),
        conf=0.4,
        save=True,
    )


if __name__ == "__main__":
    main()