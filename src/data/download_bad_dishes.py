import subprocess
from pathlib import Path
# Path for bad_dish.txt
bad_list_path = Path("./imagery/side_angles/bad_dish.txt")
target_dir = Path("./imagery/side_angles")
if not bad_list_path.exists():
    print("ERROR: bad_dish.txt not found, please check path!")
    exit(1)
# Read file, filter blank lines, keep only lines starting with dish_
with open(bad_list_path, "r", encoding="utf-8") as f:
    dish_ids = []
    for line in f:
        line = line.strip()
        if line and line.startswith("dish_"):
            dish_ids.append(line)
total = len(dish_ids)
print(f"Total {total} dishes loaded, preparing to download 4‑channel videos")
print("=" * 60)
success_count = 0
fail_list = []
for idx, dish in enumerate(dish_ids, 1):
    print(f"\n[{idx}/{total}] Processing: {dish}")
    source = f"gs://nutrition5k_dataset/nutrition5k_dataset/imagery/side_angles/{dish}/"
    # gcloud recursive copy entire dish folder
    cmd = [
        "gcloud", "storage", "cp",
        "-r",
        source,
        str(target_dir)
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print(f" {dish} download completed")
        success_count += 1
    except subprocess.CalledProcessError:
        print(f" {dish} download failed (network timeout / access error)")
        fail_list.append(dish)
print("\n===== All tasks finished =====")
print(f"Successfully downloaded: {success_count}")
print(f"Failed count: {len(fail_list)}")
if fail_list:
    print("Failed dish list:")
    for d in fail_list:
        print(d)
    # Save failed list for later retry
    with open("./imagery/side_angles/retry_bad_dish.txt", "w", encoding="utf-8") as outf:
        for d in fail_list:
            outf.write(d + "\n")
    print("Failed list saved to retry_bad_dish.txt, you can retry with this file later")
