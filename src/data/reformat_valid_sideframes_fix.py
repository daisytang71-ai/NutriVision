import os
import csv
import shutil
# ===================== Configuration =====================
BASE = r"D:\NutriVision\data\raw\nutrition5k_dataset"
ROOT_DIR = os.path.join(BASE, "side_frames_valid")
MANIFEST_OUT = os.path.join(BASE, "side_frames_valid", "frame_manifest.csv")
# ===============================================
cam_map = {
    "A": "camera_A",
    "B": "camera_B",
    "C": "camera_C",
    "D": "camera_D"
}
manifest_rows = []
total_image = 0
for dish_id in os.listdir(ROOT_DIR):
    dish_path = os.path.join(ROOT_DIR, dish_id)
    if not os.path.isdir(dish_path) or not dish_id.startswith("dish_"):
        continue
    for cam_short in cam_map.keys():
        cam_subdir = os.path.join(dish_path, cam_short)
        if not os.path.isdir(cam_subdir):
            continue
        cam_full_name = cam_map[cam_short]
        for fname in os.listdir(cam_subdir):
            if not fname.lower().endswith(".jpg"):
                continue
            # parse number from frame_001.jpg
            prefix, num_str = fname.replace(".jpg", "").split("_")
            frame_idx = int(num_str)
            new_filename = f"{cam_full_name}_f{frame_idx:02d}.jpg"
            old_full = os.path.join(cam_subdir, fname)
            new_full = os.path.join(dish_path, new_filename)
            shutil.move(old_full, new_full)
            total_image += 1
            rel_img_path = os.path.join(dish_id, new_filename).replace("\\", "/")
            manifest_rows.append([dish_id, cam_full_name, frame_idx, rel_img_path])
        # Remove empty A/B/C/D sub‑folders
        try:
            shutil.rmtree(cam_subdir)
            print(f" Removed folder: {cam_subdir}")
        except Exception as e:
            print(f" Failed to remove {cam_subdir}: {e}")
# Write manifest file
with open(MANIFEST_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["dish_id","camera","frame_idx","image_path"])
    writer.writerows(manifest_rows)
print("\n===== Processing finished =====")
print(f"✅ Total processed images: {total_image}")
print(f"📄 Manifest file: {MANIFEST_OUT}")
print(f"✅ Images are now directly under dish folder with name like camera_A_f01.jpg")
