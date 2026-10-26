import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/evaluation'))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CNN checks require PyTorch')
class CNNPipelineTests(unittest.TestCase):
    def test_current_manifest_compatibility_checks(self):
        from test_cnn import compatible_records
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root=Path(directory)
            fields=['image_path','dish_id','split','calories','protein','carbohydrate','fat']
            rows=[['a.jpg','a','train',10,1,2,3],['b.jpg','b','train',30,3,4,5],
                  ['v.jpg','v','val',20,2,3,4],['t.jpg','t','test',20,2,3,4]]
            for row in rows: (root/row[0]).touch()
            for split,ids in [('train','a\nb\n'),('val','v\n'),('test','t\n')]:
                (root/(split+'_ids.txt')).write_text(ids)
            def save():
                with (root/'manifest.csv').open('w',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(fields);writer.writerows(rows)
            save()
            checkpoint={'train_mean':[20,2,3,4],'label_scaler':{'mean':[20,2,3,4],'std':[10,1,1,1]}}
            records=compatible_records(root/'manifest.csv',root,checkpoint,root)
            self.assertEqual(records[0]['carbs'],2)
            self.assertEqual(len(records),4)
            checkpoint['train_mean'][0]=999
            with self.assertRaisesRegex(ValueError,'training means'):
                compatible_records(root/'manifest.csv',root,checkpoint,root)
            checkpoint['train_mean'][0]=20
            rows[-1][1]='a'
            save()
            with self.assertRaisesRegex(ValueError,'leakage'):
                compatible_records(root/'manifest.csv',root,checkpoint,root)
