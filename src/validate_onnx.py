from pathlib import Path
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT_DIR / "runs" / "balloon_yolov8n" / "weights" / "best.onnx"
DATASET_YAML = ROOT_DIR / "dataset_merged.yaml"


def main():
    model = YOLO(str(MODEL_PATH), task="detect")

    metrics = model.val(
        data=str(DATASET_YAML),
        imgsz=640,
        conf=0.35,
        split="test"
    )

    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)


if __name__ == "__main__":
    main()