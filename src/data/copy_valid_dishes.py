import os
import csv
import shutil
# ===================== Path Configuration =====================
BASE = r"D:\NutriVision\data\raw\nutrition5k_dataset"
SRC_SIDE = os.path.join(BASE, "side_frames")
DST_VALID = os.path.join(BASE, "side_frames_valid")
COMPARE_CSV = os.path.join(BASE, "dish_compare_result.csv")
# =================================================================
os.makedirs(DST_VALID, exist_ok=True)
ok_set = set()
# Read comparison csv and filter valid dishes
with open(COMPARE_CSV, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        did = row["dish_id"]
        status = row["status"]
        if status == "ok: exist in both":
            ok_set.add(did)
print(f" Number of valid dishes to copy: {len(ok_set)}")
copied = 0
for did in ok_set:
    src_path = os.path.join(SRC_SIDE, did)
    dst_path = os.path.join(DST_VALID, did)
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dst_path)
        copied += 1
        print(f"📋 Copied {did} --> side_frames_valid")
    else:
        print(f" Skip: {did} not found in source folder (frame extraction missing)")
print(f"\n Operation completed!")
print(f" Valid dataset location: {DST_VALID}")
print(f" Successfully copied: {copied} dishes")
print(f" Original side_frames folder remains unchanged")
