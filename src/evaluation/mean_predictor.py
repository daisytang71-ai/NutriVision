"""Fit training-label means and evaluate a fixed validation manifest. Standard library only."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from evaluate import evaluate

LABELS = ('calories', 'protein', 'carbohydrate', 'fat')
SHORT = ('cal', 'p', 'c', 'f')


def load_manifest(path, sampling):
    groups = {'train': {}, 'val': {}, 'test': {}}
    seen = set()
    dish_splits = {}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'dish_id', 'image_path', 'split', *LABELS}
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames) or not required <= set(reader.fieldnames):
            raise ValueError('Manifest has missing/duplicate column names')
        for line, row in enumerate(reader, 2):
            if None in row or any(row[c] is None or not row[c].strip() for c in required):
                raise ValueError(f'Line {line}: missing fields')
            dish, image, split = (row[c].strip() for c in ('dish_id', 'image_path', 'split'))
            if split not in groups:
                raise ValueError(f'Unknown split: {split}')
            if dish in dish_splits and dish_splits[dish] != split:
                raise ValueError(f'Dish leakage: {dish}')
            dish_splits[dish] = split
            if image in seen:
                raise ValueError(f'Duplicate image path: {image}')
            seen.add(image)
            values = tuple(float(row[c]) for c in LABELS)
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise ValueError(f'Line {line}: invalid label')
            items = groups[split].setdefault(dish, [])
            if items and values != items[0]['labels']:
                raise ValueError(f'Inconsistent labels for dish {dish}')
            items.append(dict(dish_id=dish, image_path=image, labels=values))
    selected = {}
    for split, dishes in groups.items():
        selected[split] = []
        for dish in sorted(dishes):
            rows = sorted(dishes[dish], key=lambda r: r['image_path'])
            selected[split].extend(rows[:1] if sampling == 'one-per-dish' else rows)
    if not selected['train'] or not selected['val']:
        raise ValueError('Training and validation sets must be nonempty')
    return selected


def fit_mean(rows):
    return [math.fsum(row['labels'][i] for row in rows) / len(rows) for i in range(4)]


def run(manifest, out, sampling, label_status):
    selected = load_manifest(manifest, sampling)
    means = fit_mean(selected['train'])
    out.mkdir(parents=True, exist_ok=True)
    prediction_path = out / 'mean_baseline.csv'
    fields = ['image_path', 'dish_id'] + ['y_' + s for s in SHORT] + ['pred_' + s for s in SHORT]
    with prediction_path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in selected['val']:
            record = {c: row[c] for c in ('image_path', 'dish_id')}
            record.update({'y_' + s: row['labels'][i] for i, s in enumerate(SHORT)})
            record.update({'pred_' + s: means[i] for i, s in enumerate(SHORT)})
            writer.writerow(record)
    metrics = evaluate([prediction_path], out / 'metrics.csv')
    for split in ('train', 'val'):
        with (out / (split + '_selected.csv')).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['dish_id', 'image_path'])
            writer.writeheader()
            writer.writerows({c: row[c] for c in writer.fieldnames} for row in selected[split])
    metadata = dict(model='Mean Predictor', label_status=label_status,
                    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    sampling=sampling, selection_rule='Sort dish_id and image_path lexicographically; first image per dish unless all-images selected',
                    fit_split='train', evaluation_split='val', test_evaluated=False,
                    sample_counts={s: len(v) for s, v in selected.items()},
                    training_means=dict(zip(LABELS, means)),
                    weighting='Equal selected-image weights; with one-per-dish this is equal dish weights',
                    limitations='Labels used unchanged. Does not verify source label mapping or image decoding. Other models must use identical selected samples and labels.')
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    warning = '标签映射未验证，本次仅为流程试运行；不得作为正式营养预测结果。' if label_status == 'unverified' else '标签状态由运行者声明为已验证。'
    lines = ['# Mean Predictor 验证记录', '', warning, '',
             f'采样：{sampling}；训练样本 {len(selected["train"])}；验证样本 {len(selected["val"])}。仅训练标签拟合均值，未评估测试集。', '',
             '|目标（按当前列名）|训练集均值|验证 MAE|验证 RMSE|验证 R²|', '|---|---:|---:|---:|---:|']
    for i, row in enumerate(metrics):
        lines.append(f'|{LABELS[i]}|{means[i]:.6f}|{row["MAE"]:.6f}|{row["RMSE"]:.6f}|{row["R2"]}|')
    lines.extend(['', '具体样本见 train_selected.csv 和 val_selected.csv；预测见 mean_baseline.csv。标签修复或清单变更后必须重新运行。'])
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return metadata


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sampling', choices=['one-per-dish', 'all-images'], default='one-per-dish')
    p.add_argument('--label-status', choices=['unverified', 'verified'], default='unverified')
    args = p.parse_args()
    if args.manifest.resolve().is_relative_to(args.output_dir.resolve()):
        p.error('Keep source manifest outside output directory')
    try:
        run(args.manifest, args.output_dir, args.sampling, args.label_status)
    except (ValueError, OSError, OverflowError) as exc:
        p.exit(1, f'Error: {exc}\n')
