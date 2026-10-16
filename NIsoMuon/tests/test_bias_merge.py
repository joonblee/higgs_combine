"""ROOT-free checks for ensemble completeness and bias-summary conventions."""
import importlib.util
import math
import tempfile
import unittest
from unittest.mock import patch
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

    def test_relative_shift_requires_independent_center_nll_check(self):
        interval = [5.9803009, 15.7671747, 24.0259151]
        self.assertTrue(bias.reference_consistent(
            0.0327845, interval, -1e-5, 0.01, 0.01, 0.001))
        self.assertFalse(bias.reference_consistent(
            0.0327845, interval, -0.02, 0.01, 0.01, 0.001))
        self.assertFalse(bias.reference_consistent(
            0.2, interval, -1e-5, 0.01, 0.01, 0.001))

    def test_saved_reference_profiles_skip_only_global_fit(self):
        original = ["combine", "-M", "MultiDimFit", "workspace.root",
                    "--algo", "singles", "--setParameters", "r=0",
                    "--saveFitResult", "--cminPoiOnlyFit",
                    "--cminFallbackAlgo", "Minuit2,Migrad,2:0.1"]
        result = bias.fixed_command(original, {"parameters": {"r": 1.2, "nuis": 0.3}},
                                    -8, "low", 2, 0.01)
        self.assertIn("--skipInitialFit", result)
        self.assertNotIn("--saveFitResult", result)
        self.assertNotIn("--cminPoiOnlyFit", result)
        self.assertNotIn("--freezeParameters", result)
        self.assertNotIn("--cminFallbackAlgo", result)
        self.assertEqual(bias.option(result, "--fixedPointPOIs"), "r=-8")
        self.assertEqual(bias.option(original, "--algo"), "singles")

    def test_center_delta_is_read_even_if_derived_pvalue_is_undefined(self):
        command = ["combine", "-M", "MultiDimFit", "w.root", "--setParameters", "r=0"]
        rows = [{"toy": 8, "quantile": float("nan"), "r": 0, "deltaNLL": -1e-6}]
        with tempfile.TemporaryDirectory() as directory, patch.object(bias, "run", return_value=rows):
            result = bias.profile_saved_reference(command, {"parameters": {"r": 0}},
                8, 0, Path(directory) / "center", [], 0.001)
        self.assertEqual(result["deltaNLL"], -1e-6)

    def test_disagreeing_profile_minima_are_not_selected_to_match_target(self):
        command = ["combine", "-M", "MultiDimFit", "w.root", "--setParameters", "r=0"]
        variants = [[{"toy": 8, "quantile": 0.32, "r": 1, "deltaNLL": d}]
                    for d in (0.49447, 0.3, 0.49448)]
        with tempfile.TemporaryDirectory() as directory, patch.object(bias, "run", side_effect=variants):
            with self.assertRaisesRegex(RuntimeError, "Inconsistent conditional minima"):
                bias.profile_saved_reference(command, {"parameters": {"r": 0}},
                    8, 1, Path(directory) / "high", [], 0.01)


if __name__ == "__main__":
    unittest.main()
