import os
import random
BASE = r"D:\NutriVision\data\raw\nutrition5k_dataset"
INPUT_TRAIN = os.path.join(BASE, "valid_train_ids.txt")
OUT_SPLIT_TRAIN = os.path.join(BASE, "split_train_85.txt")
OUT_SPLIT_VAL = os.path.join(BASE, "split_val_15.txt")
VAL_RATIO = 0.15   # 15% for validation set
random.seed(42)     # fixed random seed for reproducibility
# Load filtered train dish ids
with open(INPUT_TRAIN, "r", encoding="utf-8") as f:
    dish_list = [line.strip() for line in f if line.strip()]
random.shuffle(dish_list)
val_count = int(len(dish_list) * VAL_RATIO)
val_dishes = dish_list[:val_count]
new_train_dishes = dish_list[val_count:]
# Write result files
with open(OUT_SPLIT_TRAIN, "w", encoding="utf-8") as f:
    for d in new_train_dishes:
        f.write(d + "\n")
with open(OUT_SPLIT_VAL, "w", encoding="utf-8") as f:
    for d in val_dishes:
        f.write(d + "\n")
print(f"Original valid train total: {len(dish_list)}")
print(f" After split, train set split_train_85.txt: {len(new_train_dishes)}")
print(f" After split, validation set split_val_15.txt: {len(val_dishes)}")
print("\n Test set remains unchanged, keep using valid_test_ids.txt")
print(f"\nOutput files:")
print(OUT_SPLIT_TRAIN)
print(OUT_SPLIT_VAL)
