import os
import csv
BASE = r"D:\NutriVision\data\raw\nutrition5k_dataset"
CAFE1_CSV = r"D:\NutriVision\data\raw\metadata\nutrition5k_dataset_metadata_dish_metadata_cafe1.csv"
CAFE2_CSV = r"D:\NutriVision\data\raw\metadata\nutrition5k_dataset_metadata_dish_metadata_cafe2.csv"
IMAGE_ROOT = os.path.join(BASE, "side_frames_valid")
OUT_LABEL = os.path.join(BASE, "labels_corrected.csv")

# Read all dish_ids that have images
valid_dish_set = set()
for name in os.listdir(IMAGE_ROOT):
    folder_path = os.path.join(IMAGE_ROOT, name)
    if os.path.isdir(folder_path) and name.startswith("dish_"):
        valid_dish_set.add(name)
print(f" Number of valid dishes in image folder: {len(valid_dish_set)}")

# No header, read by index: 0:dish_id,1:calories,2:protein,3:carbohydrate,4:fat
def read_raw_csv(file_path):
    dish_dict = {}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            did = row[0].strip()
            dish_dict[did] = {
                "calories": row[1],
                "protein": row[2],
                "carbohydrate": row[3],
                "fat": row[4]
            }
    return dish_dict

dict1 = read_raw_csv(CAFE1_CSV)
dict2 = read_raw_csv(CAFE2_CSV)
all_raw_dict = {**dict1, **dict2}

# Keep only dishes that have both images and metadata
out_rows = []
for did in valid_dish_set:
    if did in all_raw_dict:
        info = all_raw_dict[did]
        out_rows.append([did, info["calories"], info["protein"], info["carbohydrate"], info["fat"]])
    else:
        print(f" {did} has images but no nutrition label, skipped")

with open(OUT_LABEL, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["dish_id", "calories", "protein", "carbohydrate", "fat"])
    writer.writerows(out_rows)

print(f"\n labels_corrected.csv generated, valid dishes: {len(out_rows)}")
