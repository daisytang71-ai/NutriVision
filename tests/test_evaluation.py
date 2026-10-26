import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'src/evaluation/evaluate.py'
spec = importlib.util.spec_from_file_location('evaluate', path)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class MetricsTests(unittest.TestCase):
    def test_perfect_predictions(self):
        self.assertEqual(evaluation.calculate_metrics([1, 2, 3], [1, 2, 3])[:3], (0, 0, 1))

    def test_known_error(self):
        mae, rmse, r2, _ = evaluation.calculate_metrics([1, 2, 3], [2, 2, 2])
        self.assertAlmostEqual(mae, 2 / 3)
        self.assertAlmostEqual(rmse, (2 / 3) ** 0.5)
        self.assertEqual(r2, 0)

    def test_undefined_r2(self):
        self.assertIsNone(evaluation.calculate_metrics([2, 2], [1, 3])[2])
        self.assertIsNone(evaluation.calculate_metrics([2], [3])[2])


if __name__ == '__main__':
    unittest.main()
