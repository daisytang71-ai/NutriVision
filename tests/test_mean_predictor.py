import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/evaluation'))
from mean_predictor import load_manifest, fit_mean


class MeanTests(unittest.TestCase):
    def test_training_only_and_dish_weighting(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
            path = Path(folder) / 'manifest.csv'
            def write(rows):
                with path.open('w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['dish_id','image_path','split','calories','protein','carbohydrate','fat'])
                    w.writerows(rows)
            rows = [['a','a/z.jpg','train',10,1,2,3], ['a','a/a.jpg','train',10,1,2,3],
                    ['b','b/a.jpg','train',30,3,4,5], ['v','v/a.jpg','val',1000,100,200,300]]
            write(rows)
            selected = load_manifest(path, 'one-per-dish')
            self.assertEqual(selected['train'][0]['image_path'], 'a/a.jpg')
            self.assertEqual(fit_mean(selected['train']), [20,2,3,4])
            rows[-1][3:] = [9999,999,999,999]
            write(rows)
            self.assertEqual(fit_mean(load_manifest(path,'one-per-dish')['train']), [20,2,3,4])
            rows.append(['a','a/new.jpg','val',10,1,2,3])
            write(rows)
            with self.assertRaisesRegex(ValueError, 'leakage'):
                load_manifest(path, 'one-per-dish')
