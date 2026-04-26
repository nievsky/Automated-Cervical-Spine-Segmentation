import json
import random
import shutil
from pathlib import Path
import cv2

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PNG_DIR = PROJECT_ROOT / "dataset" / "unziped" / "datasets-PNG"
RAW_JSON_DIR = PROJECT_ROOT / "dataset" / "unziped" / "datasets-JSON"
STAGING_DIR = PROJECT_ROOT / "dataset" / "staging"
FINAL_IMAGES_DIR = PROJECT_ROOT / "dataset" / "images"
FINAL_LABELS_DIR = PROJECT_ROOT / "dataset" / "labels"

# --- Ratios (70% Train, 15% Validation, 15% Test) ---
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# Test ratio will be whatever is left (~0.15)
SEED = 42

VERTEBRAE = ["C3", "C4", "C5", "C6", "C7"]
CLASS_MAP = {name: index for index, name in enumerate(VERTEBRAE)}
CORNER_SUFFIXES = ["top left", "top right", "bottom right", "bottom left"]


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def convert_json_to_yolo_seg(json_file: Path, img_w: int, img_h: int) -> list[str]:
    with json_file.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    shapes_by_label = {shape.get("label"): shape for shape in data.get("shapes", [])}
    yolo_data: list[str] = []

    for vertebra in VERTEBRAE:
        polygon_points = []
        for suffix in CORNER_SUFFIXES:
            shape = shapes_by_label.get(f"{vertebra} {suffix}")
            if not shape or not shape.get("points"):
                polygon_points = []
                break

            raw_pt = shape["points"][0]
            norm_x = raw_pt[0] / img_w
            norm_y = raw_pt[1] / img_h
            polygon_points.append(f"{norm_x:.6f} {norm_y:.6f}")

        if not polygon_points:
            continue

        class_id = CLASS_MAP[vertebra]
        yolo_line = f"{class_id} " + " ".join(polygon_points)
        yolo_data.append(yolo_line)

    return yolo_data


def main() -> None:
    # 1. Create ALL necessary directories
    splits = ["train", "val", "test"]
    for split in splits:
        ensure_clean_dir(FINAL_IMAGES_DIR / split)
        ensure_clean_dir(FINAL_LABELS_DIR / split)
    ensure_clean_dir(STAGING_DIR / "images")
    ensure_clean_dir(STAGING_DIR / "labels")

    print("🚀 Converting JSONs to YOLO Segmentation format...")
    paired_items = []

    # 2. Process and Stage Files
    for img_path in sorted(RAW_PNG_DIR.glob("*.png")):
        json_path = RAW_JSON_DIR / f"{img_path.stem}.json"
        if not json_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        height, width = image.shape[:2]
        yolo_lines = convert_json_to_yolo_seg(json_path, width, height)

        if not yolo_lines:
            continue

        staged_image_path = STAGING_DIR / "images" / img_path.name
        staged_label_path = STAGING_DIR / "labels" / f"{img_path.stem}.txt"

        shutil.copy2(img_path, staged_image_path)
        staged_label_path.write_text("\n".join(yolo_lines), encoding="utf-8")
        paired_items.append((staged_image_path, staged_label_path))

    # 3. Perform the 3-way Split
    print(f"📦 Splitting {len(paired_items)} items into 70/15/15...")
    random.Random(SEED).shuffle(paired_items)

    total = len(paired_items)
    idx_val = int(total * TRAIN_RATIO)
    idx_test = int(total * (TRAIN_RATIO + VAL_RATIO))

    dataset_splits = {
        "train": paired_items[:idx_val],
        "val": paired_items[idx_val:idx_test],
        "test": paired_items[idx_test:]
    }

    # 4. Move files to Final Destination
    for split_name, items in dataset_splits.items():
        for image_path, label_path in items:
            shutil.copy2(image_path, FINAL_IMAGES_DIR / split_name / image_path.name)
            shutil.copy2(label_path, FINAL_LABELS_DIR / split_name / label_path.name)

    print(f"✅ Setup complete!")
    print(f"   Train: {len(dataset_splits['train'])}")
    print(f"   Val:   {len(dataset_splits['val'])}")
    print(f"   Test:  {len(dataset_splits['test'])}")


if __name__ == "__main__":
    main()