import os
import csv
import re
BASE = r"D:\NutriVision\data\raw\nutrition5k_dataset"
ROOT_DIR = os.path.join(BASE, "side_frames_valid")
LABEL_CSV = os.path.join(BASE, "labels_corrected.csv")
TRAIN_TXT = os.path.join(BASE, "train_ids.txt")
VAL_TXT = os.path.join(BASE, "val_ids.txt")
TEST_TXT = os.path.join(BASE, "test_ids.txt")
OUT_MANIFEST = os.path.join(BASE, "full_frame_manifest.csv")

# Read labels
label_dict = {}
with open(LABEL_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        did = row["dish_id"].strip()
        label_dict[did] = {
            "calories": row["calories"],
            "protein": row["protein"],
            "carbohydrate": row["carbohydrate"],
            "fat": row["fat"]
        }

def load_split(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

train_set = load_split(TRAIN_TXT)
val_set = load_split(VAL_TXT)
test_set = load_split(TEST_TXT)

def get_split(dish_id):
    if dish_id in train_set:
        return "train"
    elif dish_id in val_set:
        return "val"
    elif dish_id in test_set:
        return "test"
    else:
        return "unknown"

rows = []
for dish_id in os.listdir(ROOT_DIR):
    dish_folder = os.path.join(ROOT_DIR, dish_id)
    if not os.path.isdir(dish_folder) or not dish_id.startswith("dish_"):
        continue
    if dish_id not in label_dict:
        print(f" {dish_id} missing label, skipped")
        continue
    info = label_dict[dish_id]
    split_tag = get_split(dish_id)
    for filename in os.listdir(dish_folder):
        if not filename.lower().endswith(".jpg"):
            continue
        name_no_suffix = filename.replace(".jpg", "")
        match = re.search(r'f(\d+)', name_no_suffix)
        if not match:
            print(f" Failed to extract frame index from filename: {filename}, skipped")
            continue
        frame_idx = int(match.group(1))
        cam_part = name_no_suffix.split("_f")[0]
        rel_path = os.path.join(dish_id, filename).replace("\\", "/")
        rows.append([
            dish_id,
            cam_part,
            frame_idx,
            rel_path,
            info["calories"],
            info["protein"],
            info["carbohydrate"],
            info["fat"],
            split_tag
        ])

with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as out_f:
    writer = csv.writer(out_f)
    writer.writerow([
        "dish_id",
        "camera",
        "frame_idx",
        "image_path",
        "calories",
        "protein",
        "carbohydrate",
        "fat",
        "split"
    ])
    writer.writerows(rows)

print(f"\nDone, total {len(rows)} image records generated")
print(f"Output file: {OUT_MANIFEST}")
