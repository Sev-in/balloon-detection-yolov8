from pathlib import Path
import shutil
import hashlib
import random


# Projenin ana klasörünü bulur.
# Örnek:
# AI_egitim/src/merge_datasets.py
# parents[1] -> AI_egitim
ROOT_DIR = Path(__file__).resolve().parents[1]

# İndirdiğimiz ham datasetlerin bulunduğu klasör.
# İçinde balloon_01, balloon_02 gibi klasörler var.
RAW_DATASETS_DIR = ROOT_DIR / "datasets_raw"

# Merge edilmiş final dataset buraya oluşturulacak.
OUTPUT_DIR = ROOT_DIR / "dataset_merged"

# Görsel olarak kabul edeceğimiz dosya uzantıları.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Yeni dataset oranları.
# Tüm datasetleri birleştirip tekrar böleceğiz.
SPLITS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}


def file_hash(path: Path) -> str:
    """
    Görsel dosyasının hash değerini üretir.

    Amaç:
    Aynı görsel farklı datasetlerde varsa bunu yakalamak.

    Not:
    Aynı isimli dosyaya değil, dosyanın içeriğine bakar.
    """
    return hashlib.md5(path.read_bytes()).hexdigest()


def find_image_label_pairs():
    """
    datasets_raw içindeki tüm datasetleri gezer.

    Beklenen Roboflow yapısı:
    dataset/
      train/images
      train/labels
      valid/images
      valid/labels
      test/images
      test/labels

    Her image için aynı isimli label var mı kontrol eder.
    Örnek:
    image: abc.jpg
    label: abc.txt

    Eşleşen image-label çiftlerini liste olarak döndürür.
    """
    pairs = []

    for dataset_dir in RAW_DATASETS_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue

        # Bazı datasetlerde valid, bazılarında val olabilir.
        for split_name in ["train", "valid", "val", "test"]:
            images_dir = dataset_dir / split_name / "images"
            labels_dir = dataset_dir / split_name / "labels"

            if not images_dir.exists() or not labels_dir.exists():
                continue

            for image_path in images_dir.iterdir():
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                label_path = labels_dir / f"{image_path.stem}.txt"

                if label_path.exists():
                    pairs.append((image_path, label_path))

    return pairs


def prepare_output_dirs():
    """
    Eski dataset_merged klasörü varsa siler.
    Sonra temiz final klasör yapısını oluşturur.

    Final yapı:
    dataset_merged/
      images/train
      images/val
      images/test
      labels/train
      labels/val
      labels/test
    """
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for split in SPLITS:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml():
    """
    YOLO eğitimi için dataset_merged.yaml dosyası oluşturur.

    Bu dosya YOLO'ya şunu söyler:
    - dataset nerede?
    - train/val/test klasörleri nerede?
    - kaç class var?
    - class ismi ne?
    """
    yaml_content = f"""path: {OUTPUT_DIR.as_posix()}

train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: balloon
"""

    (ROOT_DIR / "dataset_merged.yaml").write_text(yaml_content, encoding="utf-8")


def main():
    """
    Ana akış:

    1. Output klasörlerini hazırla
    2. Tüm datasetlerden image-label çiftlerini bul
    3. Aynı görselleri hash ile temizle
    4. Verileri karıştır
    5. Train / val / test olarak yeniden böl
    6. Yeni isimlerle final klasöre kopyala
    7. dataset_merged.yaml oluştur
    """
    prepare_output_dirs()

    pairs = find_image_label_pairs()
    print(f"Found image-label pairs: {len(pairs)}")

    seen_hashes = set()
    unique_pairs = []

    for image_path, label_path in pairs:
        img_hash = file_hash(image_path)

        if img_hash in seen_hashes:
            continue

        seen_hashes.add(img_hash)
        unique_pairs.append((image_path, label_path))

    print(f"Unique image-label pairs: {len(unique_pairs)}")
    print(f"Duplicate removed: {len(pairs) - len(unique_pairs)}")

    # Her çalıştırmada aynı bölünmeyi almak için sabit seed.
    random.seed(42)
    random.shuffle(unique_pairs)

    total = len(unique_pairs)
    train_end = int(total * SPLITS["train"])
    val_end = train_end + int(total * SPLITS["val"])

    split_data = {
        "train": unique_pairs[:train_end],
        "val": unique_pairs[train_end:val_end],
        "test": unique_pairs[val_end:],
    }

    counter = 0

    for split, items in split_data.items():
        for image_path, label_path in items:
            counter += 1

            # Dosya isimlerini yeniden veriyoruz.
            # Böylece farklı datasetlerde aynı isim varsa çakışma olmaz.
            new_name = f"balloon_{counter:06d}"

            new_image_path = OUTPUT_DIR / "images" / split / f"{new_name}{image_path.suffix.lower()}"
            new_label_path = OUTPUT_DIR / "labels" / split / f"{new_name}.txt"

            shutil.copy2(image_path, new_image_path)
            shutil.copy2(label_path, new_label_path)

        print(f"{split}: {len(items)}")

    write_dataset_yaml()

    print("\nMerge completed.")
    print(f"Output dataset: {OUTPUT_DIR}")
    print(f"YAML file: {ROOT_DIR / 'dataset_merged.yaml'}")


if __name__ == "__main__":
    main()