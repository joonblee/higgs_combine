"""Counting-yield extraction checks without requiring CMSSW/PyROOT."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prefit_postfit_yields as tool


class Value:
    def __init__(self, value, error):
        self.value, self.error = value, error

    def getVal(self):
        return self.value

    def getError(self):
        return self.error


class Norms(dict):
    def find(self, key):
        return self.get(key)


class Fit:
    def status(self):
        return 0

    def covQual(self):
        return 3

    def edm(self):
        return 1e-8

    def floatParsFinal(self):
        return Norms()


class Source(dict):
    def Get(self, key):
        return self.get(key)


class YieldTests(unittest.TestCase):
    def setUp(self):
        self.nominal = {"bin_2018": {"QCD": 10., "tt": 20.}}
        self.source = Source(fit_b=Fit(), fit_s=Fit())
        for stage in tool.STAGES:
            self.source[f"norm_{stage}"] = Norms({
                "bin_2018/QCD": Value(10., 3.), "bin_2018/tt": Value(20., 4.),
                "bin_2018/total_background": Value(30., 2.)})

    def test_correlated_total_is_retained_not_quadrature(self):
        stages, fits = tool.extract(self.source, self.nominal)
        self.assertEqual(stages["fit_s"]["bin_2018"]["total_background"]["error"], 2.)
        self.assertEqual(fits["fit_b"]["status"], 0)

    def test_missing_total_refuses_independent_error_fallback(self):
        del self.source["norm_fit_b"]["bin_2018/total_background"]
        with self.assertRaisesRegex(RuntimeError, "Missing saved counting yield"):
            tool.extract(self.source, self.nominal)

    def test_bad_process_closure_is_rejected(self):
        self.source["norm_fit_s"]["bin_2018/total_background"].value = 35.
        with self.assertRaisesRegex(RuntimeError, "process closure"):
            tool.extract(self.source, self.nominal)

    def test_prefit_rateparam_mismatch_is_rejected(self):
        nominal = copy.deepcopy(self.nominal)
        nominal["bin_2018"]["QCD"] = 1.
        with self.assertRaisesRegex(RuntimeError, "prefit bin_2018/QCD"):
            tool.extract(self.source, nominal)

    def test_nonfinite_uncertainty_is_rejected(self):
        self.source["norm_fit_b"]["bin_2018/total_background"].error = float("nan")
        with self.assertRaisesRegex(RuntimeError, "Invalid error"):
            tool.extract(self.source, self.nominal)

    def test_production_card_includes_effective_qcd_rate(self):
        card = Path(tool.__file__).parent / "datacards_NIsoMuon/alpha/blind/combined/Run2Run3/datacard_M-30_Run2Run3.txt"
        nominal = tool.nominal_backgrounds(card)
        self.assertEqual(len(nominal), 8)
        self.assertNotEqual(nominal["bin_2018"]["QCD"], 1.)
        self.assertAlmostEqual(sum(nominal["bin_2018"].values()), 274.352327954, places=5)


if __name__ == "__main__":
    unittest.main()
