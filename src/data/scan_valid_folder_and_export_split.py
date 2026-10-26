import os
# ===================== Path Configuration =====================
BASE = r"D:\NutriVision\data\raw\nutrition5k_dataset"
VALID_FRAMES = os.path.join(BASE, "side_frames_valid")
TRAIN_TXT = os.path.join(BASE, "dish_ids", "rgb_train_ids.txt")
TEST_TXT = os.path.join(BASE, "dish_ids", "rgb_test_ids.txt")
OUT_TRAIN = os.path.join(BASE, "valid_train_ids.txt")
OUT_TEST = os.path.join(BASE, "valid_test_ids.txt")
# ============================================================
def load_dish_set(txt_path):
    s = set()
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            did = line.strip()
            if did.startswith("dish_"):
                s.add(did)
    return s
# 1. Scan side_frames_valid and get existing dishes
exist_valid_dishes = set()
for name in os.listdir(VALID_FRAMES):
    fullp = os.path.join(VALID_FRAMES, name)
    if os.path.isdir(fullp) and name.startswith("dish_"):
        exist_valid_dishes.add(name)
# 2. Load official dataset split
official_train = load_dish_set(TRAIN_TXT)
official_test = load_dish_set(TEST_TXT)
# 3. Compute intersection
final_train = sorted(official_train & exist_valid_dishes)
final_test = sorted(official_test & exist_valid_dishes)
# 4. Write output files
with open(OUT_TRAIN, "w", encoding="utf-8") as f:
    for d in final_train:
        f.write(d + "\n")
with open(OUT_TEST, "w", encoding="utf-8") as f:
    for d in final_test:
        f.write(d + "\n")
print(f" Total dishes scanned in side_frames_valid: {len(exist_valid_dishes)}")
print(f" Original official train: {len(official_train)}, overlapping valid train: {len(final_train)}")
print(f" Original official test: {len(official_test)}, overlapping valid test: {len(final_test)}")
print(f"\n Output completed:")
print(f"   {OUT_TRAIN}")
print(f"   {OUT_TEST}")
