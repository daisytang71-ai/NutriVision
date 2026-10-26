"""Check manifest paths and dish split overlap; does not validate label mapping."""
import csv
import json
import argparse
from pathlib import Path
from collections import Counter


def audit(root):
    splits = {s: set((root / (s + '_ids.txt')).read_text(encoding='utf-8-sig').split()) for s in ('train', 'val', 'test')}
    images = list((root / 'side_frames_valid').rglob('*.jpg'))
    actual = {p.relative_to(root / 'side_frames_valid').as_posix() for p in images}
    seen, suspicious = set(), set()
    dishes = {s: set() for s in splits}
    counts = Counter()
    missing = duplicate = mismatch = 0
    with (root / 'full_frame_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            dish, split = row['dish_id'], row['split']
            path = row['image_path'].replace('\\', '/')
            duplicate += path in seen
            seen.add(path)
            missing += path not in actual
            mismatch += dish not in splits.get(split, set())
            counts[split] += 1
            dishes.setdefault(split, set()).add(dish)
            if 4 * float(row['protein']) > float(row['calories']):
                suspicious.add(dish)
    return dict(check_date='2026-10-26', image_count=len(images), image_bytes=sum(p.stat().st_size for p in images),
                manifest_rows=sum(counts.values()), split_image_counts=dict(counts),
                split_manifest_dish_counts={s: len(v) for s, v in dishes.items()},
                split_id_counts={s: len(v) for s, v in splits.items()},
                split_overlap={a+'_'+b: len(splits[a] & splits[b]) for a,b in [('train','val'),('train','test'),('val','test')]},
                missing_images=missing, duplicate_manifest_paths=duplicate, split_mismatch_rows=mismatch,
                unlisted_images=len(actual-seen), ids_without_manifest={s:len(splits[s]-dishes[s]) for s in splits},
                dishes_with_4x_protein_exceeding_calories=len(suspicious),
                label_status='Unverified; possible column mapping issue. Original values unchanged.',
                limitations='Does not establish official split provenance, decode images, check duplicate image content, or verify labels against source metadata.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
