"""Export CNN predictions, then score with the project's evaluate.py."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models' / 'nutrivision_cnn'))
from nutrivision.model import CustomCNN
from nutrivision.data import LabelScaler, NutritionDataset, load_manifest
from evaluate import evaluate

FIELDS = ['image_path','dish_id','y_cal','y_p','y_c','y_f','pred_cal','pred_p','pred_c','pred_f']


def compatible_records(manifest, root, checkpoint, split_dir=None):
    """Validate current project manifest against stored training moments and split lists.

    Matching moments do not establish byte-identical training data. Both hashes
    are recorded in output metadata for traceability.
    """
    with manifest.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'image_path','dish_id','split','calories','protein','carbohydrate','fat'}
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames) or not required <= set(reader.fieldnames):
            raise ValueError('Missing or duplicate manifest columns')
        rows = list(reader)
    seen, dishes, groups = set(), {}, {}
    for row in rows:
        if None in row or any(not row[k] or not row[k].strip() for k in required):
            raise ValueError('Missing manifest field')
        dish, split, path = row['dish_id'], row['split'], row['image_path']
        if split not in {'train','val','test'}:
            raise ValueError('Invalid split')
        values = tuple(float(row[k]) for k in ('calories','protein','carbohydrate','fat'))
        if not np.isfinite(values).all() or min(values)<0:
            raise ValueError('Invalid labels')
        info = (split, values)
        if dish in dishes and dishes[dish] != info:
            raise ValueError('Dish leakage or inconsistent labels')
        dishes[dish] = info
        row['group_id'] = row.get('group_id') or dish
        if row['group_id'] in groups and groups[row['group_id']] != split:
            raise ValueError('Physical group leakage')
        groups[row['group_id']] = split
        full = (root/path).resolve()
        if not full.is_relative_to(root.resolve()) or str(full) in seen:
            raise ValueError('Duplicate or out-of-root image path')
        seen.add(str(full))
        if not full.is_file():
            raise FileNotFoundError(full)
        row['_path'] = str(full)
        for key, value in zip(('calories','protein','carbs','fat'), values):
            row[key] = value
    splits = Path(split_dir) if split_dir else Path(__file__).resolve().parents[2]/'splits'
    for split in ('train','val','test'):
        expected = set((splits/(split+'_ids.txt')).read_text(encoding='utf-8-sig').split())
        actual = {dish for dish,(s,_) in dishes.items() if s==split}
        if actual != expected:
            raise ValueError('Current manifest does not match repository split: '+split)
    train = np.array([values for split,values in dishes.values() if split=='train'])
    expected_std = train.std(0)
    expected_std[expected_std<1e-6] = 1
    for calculated, stored, name in [(train.mean(0),checkpoint['train_mean'],'training means'),
                                     (train.mean(0),checkpoint['label_scaler']['mean'],'scaler means'),
                                     (expected_std,checkpoint['label_scaler']['std'],'scaler std')]:
        if not np.allclose(calculated,stored,rtol=1e-6,atol=1e-5):
            raise ValueError('Checkpoint mismatch: '+name)
    print('Current manifest split lists and training moments match checkpoint.',flush=True)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--split', choices=['val','test'], default='test')
    p.add_argument('--compatible-manifest', action='store_true', help='Use current project manifest after checking split IDs and stored training moments; retain both hashes')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.batch_size < 1 or a.threads < 1:
        p.error('Batch size and threads must be positive')
    checkpoint = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    digest = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    if digest != checkpoint['manifest_sha256'] and not a.compatible_manifest:
        p.error('Manifest differs from training snapshot. Supply full_frame_manifest_cnn.csv from the training run.')
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        p.error('Output directory must be empty')
    records = (compatible_records(a.manifest, a.data_root, checkpoint) if a.compatible_manifest
               else load_manifest(a.manifest, a.data_root, verify_images=False))
    selected = [r for r in records if r['split'] == a.split]
    if not selected:
        p.error('No selected samples')
    torch.set_num_threads(a.threads)
    model = CustomCNN(checkpoint['config']['dropout']).eval()
    model.load_state_dict(checkpoint['model'])
    scaler = LabelScaler(**checkpoint['label_scaler'])
    dataset = NutritionDataset(selected, scaler, checkpoint['config']['image_size'])
    loader = DataLoader(dataset,batch_size=a.batch_size,shuffle=False,num_workers=0)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    labels = {r['image_path']: [r[k] for k in ('calories','protein','carbs','fat')] for r in selected}
    groups = {}
    with (a.output_dir/'cnn_images.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f); writer.writerow(FIELDS)
        with torch.inference_mode():
            done=0
            for x, _, ids, paths in loader:
                predictions=scaler.decode(model(x).numpy())
                if not np.isfinite(predictions).all():
                    raise ValueError('Nonfinite model prediction')
                for dish,path,pred in zip(ids,paths,predictions):
                    truth=labels[path]
                    writer.writerow([path,dish,*truth,*map(float,pred)])
                    g=groups.setdefault(dish,dict(truth=truth,total=np.zeros(4,dtype=np.float64),n=0))
                    g['total']+=pred; g['n']+=1
                done+=len(ids)
                if done % (a.batch_size*10)==0 or done==len(dataset):
                    print(f'{done}/{len(dataset)} images',flush=True)
    for filename, baseline in [('cnn_dishes.csv',False),('mean_baseline_dishes.csv',True)]:
        with (a.output_dir/filename).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.writer(f); writer.writerow(FIELDS)
            for dish,g in sorted(groups.items()):
                pred=checkpoint['train_mean'] if baseline else g['total']/g['n']
                # Synthetic aggregation identity; not an actual image path.
                writer.writerow(['dish_average/'+dish,dish,*g['truth'],*map(float,pred)])
    evaluate([a.output_dir/'cnn_images.csv'],a.output_dir/'image_metrics.csv')
    evaluate([a.output_dir/'cnn_dishes.csv',a.output_dir/'mean_baseline_dishes.csv'],a.output_dir/'dish_comparison.csv')
    meta=dict(split=a.split,checkpoint_epoch=checkpoint['epoch'],manifest_sha256=digest,
              checkpoint_manifest_sha256=checkpoint['manifest_sha256'],
              manifest_check=('split IDs and training moments; hashes differ' if digest!=checkpoint['manifest_sha256'] else 'exact hash'),
              checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
              images=len(dataset),dishes=len(groups),device='cpu',torch=str(torch.__version__),
              evaluation_script='src/evaluation/evaluate.py',
              aggregation='Unclipped predictions averaged across all available images of each dish',
              baseline='Training-dish means saved in checkpoint; not fitted on evaluation labels',
              targets=['calories','protein','carbohydrate','fat'])
    (a.output_dir/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
