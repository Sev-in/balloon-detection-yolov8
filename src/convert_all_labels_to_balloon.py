from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATASETS_DIR = ROOT_DIR / "datasets_raw"


def convert_labels_to_single_class():
    label_dirs = list(RAW_DATASETS_DIR.rglob("labels"))

    print("RAW_DATASETS_DIR:", RAW_DATASETS_DIR)
    print("Found label dirs:", len(label_dirs))

    converted_files = 0
    converted_lines = 0

    for label_dir in label_dirs:
        for label_file in label_dir.glob("*.txt"):
            lines = label_file.read_text(encoding="utf-8").splitlines()
            new_lines = []

            for line in lines:
                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                parts[0] = "0"
                new_lines.append(" ".join(parts))

            label_file.write_text(
                "\n".join(new_lines) + ("\n" if new_lines else ""),
                encoding="utf-8"
            )

            converted_files += 1
            converted_lines += len(new_lines)

    print(f"Converted label files: {converted_files}")
    print(f"Converted label lines: {converted_lines}")


if __name__ == "__main__":
    convert_labels_to_single_class()