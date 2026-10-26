"""Evaluate NutriVision prediction CSVs using Python 3 (no extra packages)."""
import argparse
import csv
import math
import sys
from pathlib import Path

TARGETS = (
    ('calories', 'kcal', 'y_cal', 'pred_cal'),
    ('protein', 'g', 'y_p', 'pred_p'),
    ('carbohydrate', 'g', 'y_c', 'pred_c'),
    ('fat', 'g', 'y_f', 'pred_f'),
)
REQUIRED = ['image_path', 'dish_id'] + [c for t in TARGETS for c in t[2:]]


def read_predictions(path):
    """Reject incomplete data rather than silently dropping validation samples."""
    records = {}
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f'{path}: missing or duplicate column names')
        missing = set(REQUIRED) - set(reader.fieldnames)
        if missing:
            raise ValueError(f'{path}: missing columns: {sorted(missing)}')
        for line, row in enumerate(reader, 2):
            if None in row or any(row[c] is None for c in REQUIRED):
                raise ValueError(f'{path}, line {line}: wrong number of fields')
            key = (row['dish_id'].strip(), row['image_path'].strip())
            if not all(key):
                raise ValueError(f'{path}, line {line}: empty dish_id/image_path')
            if key in records:
                raise ValueError(f'{path}, line {line}: duplicate sample {key}')
            values = {}
            for target in TARGETS:
                for column in target[2:]:
                    try:
                        value = float(row[column])
                    except (ValueError, TypeError):
                        raise ValueError(f'{path}, line {line}: {column} is not numeric') from None
                    if not math.isfinite(value):
                        raise ValueError(f'{path}, line {line}: {column} is NaN/infinite')
                    if column.startswith('y_') and value < 0:
                        raise ValueError(f'{path}, line {line}: negative true label {column}')
                    values[column] = value
            records[key] = values
    if not records:
        raise ValueError(f'{path}: no samples')
    return records


def calculate_metrics(truth, prediction):
    n = len(truth)
    if not n or n != len(prediction):
        raise ValueError('Expected equally sized, nonempty arrays')
    errors = [p - y for y, p in zip(truth, prediction)]
    mse = math.fsum(e * e for e in errors) / n
    mean = math.fsum(truth) / n
    total = math.fsum((y - mean) ** 2 for y in truth)
    reason = 'fewer than two samples' if n < 2 else ('constant true labels' if total == 0 else '')
    r2 = None if reason else 1 - (mse * n) / total
    result = (math.fsum(abs(e) for e in errors) / n, math.sqrt(mse), r2, reason)
    if any(v is not None and not math.isfinite(v) for v in result[:3]):
        raise ValueError('Metric overflow: check input magnitudes')
    return result


def evaluate(files, output):
    names = [p.stem for p in files]
    if len(set(names)) != len(names):
        raise ValueError('Input filenames must have different stems (used as model names)')
    datasets = [read_predictions(p) for p in files]
    reference = datasets[0]
    for path, data in zip(files[1:], datasets[1:]):
        if set(data) != set(reference):
            raise ValueError(f'{path}: sample set differs from first file; comparison stopped')
        for key in reference:
            for _, _, true_col, _ in TARGETS:
                if not math.isclose(reference[key][true_col], data[key][true_col], rel_tol=1e-9, abs_tol=1e-6):
                    raise ValueError(f'{path}: inconsistent true label for {key}, {true_col}')
    rows = []
    for name, data in zip(names, datasets):
        for target, unit, true_col, pred_col in TARGETS:
            truth = [v[true_col] for v in data.values()]
            pred = [v[pred_col] for v in data.values()]
            mae, rmse, r2, reason = calculate_metrics(truth, pred)
            rows.append(dict(model=name, target=target, unit=unit, n_samples=len(data),
                             n_dishes=len({k[0] for k in data}), MAE=mae, RMSE=rmse,
                             R2='' if r2 is None else r2, R2_note=reason))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        r2_text = 'undefined' if row['R2'] == '' else f"{row['R2']:.4f}"
        print(f"{row['model']:20} {row['target']:14} MAE={row['MAE']:.4f} "
              f"RMSE={row['RMSE']:.4f} {row['unit']} R2={r2_text} n={row['n_samples']}")
    print(f'Saved: {output.resolve()}')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+', type=Path, help='Prediction CSVs; filename stem is model name')
    parser.add_argument('--output', type=Path, default=Path('metrics.csv'))
    args = parser.parse_args()
    if any(args.output.resolve() == p.resolve() for p in args.files):
        parser.error('Output must not overwrite an input file')
    try:
        evaluate(args.files, args.output)
    except (ValueError, OSError, OverflowError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
