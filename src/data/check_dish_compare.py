import os
import csv
# 改为相对路径（项目根目录下）
FRAMES_ROOT = r"data/raw/nutrition5k_dataset/side_frames"
TRAIN_TXT = r"data/raw/nutrition5k_dataset/dish_ids/rgb_train_ids.txt"
TEST_TXT = r"data/raw/nutrition5k_dataset/dish_ids/rgb_test_ids.txt"
# No validation file, commented out
# VAL_TXT = r"data/raw/nutrition5k_dataset/dish_ids/val.txt"
OUTPUT_CSV = "dish_compare_result.csv"

def load_dish_txt(filepath):
    """Read txt file, return dish set, clean blank lines and spaces automatically"""
    dish_set = set()
    if not os.path.exists(filepath):
        print(f" File not found: {filepath}")
        return dish_set
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            d = line.strip()
            if d.startswith("dish_"):
                dish_set.add(d)
    return dish_set

# Load official data split
train_dishes = load_dish_txt(TRAIN_TXT)
test_dishes = load_dish_txt(TEST_TXT)
all_official = train_dishes | test_dishes
print(f" Total official dishes: {len(all_official)}")
print(f"   train: {len(train_dishes)}")
print(f"   test:  {len(test_dishes)}")

# Scan all dish folders with extracted frames under side_frames
success_dishes = set()
if os.path.exists(FRAMES_ROOT):
    for name in os.listdir(FRAMES_ROOT):
        full_path = os.path.join(FRAMES_ROOT, name)
        if os.path.isdir(full_path) and name.startswith("dish_"):
            success_dishes.add(name)
else:
    print(f" Directory not found: {FRAMES_ROOT}, please check the path!")
print(f"\n Total locally extracted dishes: {len(success_dishes)}")

# Set comparison
missing_in_your_result = all_official - success_dishes   # In official list but extraction failed (missing)
extra_you_have = success_dishes - all_official           # Extracted but not in official train/test split
both_have = all_official & success_dishes                # Exists in both, usable normally

print("\n===== Comparison Summary =====")
print(f" Matched & usable dishes: {len(both_have)}")
print(f" In official list but frame extraction failed: {len(missing_in_your_result)}")
print(f" Extracted but not in official train/test split: {len(extra_you_have)}")

# Write to CSV with original train/test split label
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["dish_id", "status", "official_split"])

    def get_split_name(did):
        if did in train_dishes:
            return "train"
        elif did in test_dishes:
            return "test"
        else:
            return "none"

    for did in sorted(both_have):
        writer.writerow([did, "ok: exist in both", get_split_name(did)])
    for did in sorted(missing_in_your_result):
        writer.writerow([did, "missing: official exists, frame extraction failed", get_split_name(did)])
    for did in sorted(extra_you_have):
        writer.writerow([did, "extra: extracted, not in official split", "none"])

print(f"\n Detailed comparison saved to: {OUTPUT_CSV}")
