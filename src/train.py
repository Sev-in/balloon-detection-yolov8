from pathlib import Path
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_YAML = ROOT_DIR / "dataset_merged.yaml"
RUNS_DIR = ROOT_DIR / "runs"

def main():
    model = YOLO("yolov8n.pt")

    # 🔹 TEST (ilk çalıştır)
    #model.train(
    #   data=str(DATASET_YAML),
    #    epochs=1,
    #    imgsz=640,
    #    device=0
    #)

    # 🔹 GERÇEK EĞİTİM
    model.train(
        data=str(DATASET_YAML),
        epochs=50,  # önce 5 dene
        imgsz=640,
        batch=8,
        device=0,
        workers=2,  # Windows + GPU için daha stabil
        project=str(RUNS_DIR),
        name="balloon_yolov8n"
    )

if __name__ == "__main__":
    main()