"""ROOT-free checks for ensemble completeness and bias-summary conventions."""
import importlib.util
import math
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "bias_verification", Path(__file__).resolve().parents[1] / "validate_bias_repair.py")
bias = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bias)


class BiasMergeTests(unittest.TestCase):
    def test_missing_toys_are_not_silently_dropped(self):
        with self.assertRaisesRegex(RuntimeError, "missing=\\[2\\]"):
            bias.combine_records({1: {"interval": [-1, 0, 1]}}, {}, 2)

    def test_repair_restores_missing_toy_without_counting_it_twice(self):
        original = {1: {"interval": [-1, 0, 1]}, 2: {"interval": [-50, 0, 100]}}
        replacement = {2: {"interval": [0, 1, 2]}}
        result = bias.combine_records(original, replacement, 2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[2]["interval"], [0, 1, 2])
        self.assertEqual(original[2]["interval"], [-50, 0, 100])

    def test_duplicate_best_fits_cannot_be_selected_arbitrarily(self):
        rows = [{"toy": 1, "quantile": q, "r": r}
                for q, r in [(-1, 0), (-1, 0.1), (-0.32, -1), (0.32, 1)]]
        self.assertNotIn(1, bias.intervals(rows))

    def test_capped_and_collapsed_intervals_are_invalid(self):
        for value in [[-50, 0, 1], [-1, 0, 100], [0, 0, 1], [1, 0, 2]]:
            self.assertFalse(bias.valid_interval(value))

    def test_directional_pulls_use_the_side_towards_injection(self):
        records = {1: {"interval": [0, 2, 3]}, 2: {"interval": [-3, -2, 0]}}
        result = bias.summarize(records, 0)
        self.assertEqual(result["n_toys"], 2)
        self.assertEqual(result["mean_pull"], 0)
        self.assertAlmostEqual(result["pull_width"], math.sqrt(2))
        self.assertAlmostEqual(result["mean_bias_mc_standard_error"], 2)
        self.assertFalse(result["coverage_test"])


if __name__ == "__main__":
    unittest.main()
