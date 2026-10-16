import importlib.util
import math
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "audit_bias_intervals", Path(__file__).resolve().parents[1] / "audit_bias_intervals.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def rows(toy=1, interval=(-0.5, 0.1, 0.7)):
    return [{"toy": toy, "quantile": q, "r": r}
            for (_, q), r in zip(audit.QUANTILES, interval)]


class AuditTests(unittest.TestCase):
    def test_valid(self):
        result = audit.audit_rows(rows(), 1, (-2, 100))
        self.assertTrue(result["passed"])
        self.assertEqual(result["toys"]["1"]["interval"], [-0.5, 0.1, 0.7])

    def test_missing_toy(self):
        result = audit.audit_rows(rows(), 2, (-2, 100))
        self.assertEqual(result["flagged_toy_ids"], [2])

    def test_duplicate_not_overwritten(self):
        result = audit.audit_rows(rows() + rows()[:1], 1, (-2, 100))
        self.assertFalse(result["passed"])
        self.assertIsNone(result["toys"]["1"]["interval"])

    def test_nonfinite_collapsed_and_capped(self):
        for interval in [(math.nan, 0, 1), (-1, 0, 0), (-2, 0, 1), (-1, 0, 100), (1, 0, 2)]:
            with self.subTest(interval=interval):
                self.assertFalse(audit.audit_rows(rows(interval=interval), 1, (-2, 100))["passed"])

    def test_unknown_quantile_and_extra_toy(self):
        self.assertFalse(audit.audit_rows(rows() + [{"toy": 1, "quantile": 0.5, "r": 0.8}], 1, (-2, 100))["passed"])
        self.assertEqual(audit.audit_rows(rows() + rows(2), 1, (-2, 100))["unexpected_toy_ids"], [2])

    def test_range_is_read_from_log(self):
        text = "Set Range of Parameter r To : (-2,100)\n" * 2
        self.assertEqual(audit.fit_range(text), (-2, 100))
        with self.assertRaises(RuntimeError):
            audit.fit_range(text + "Set Range of Parameter r To : (-50,100)")
        with self.assertRaises(RuntimeError):
            audit.fit_range("")


if __name__ == "__main__":
    unittest.main()
