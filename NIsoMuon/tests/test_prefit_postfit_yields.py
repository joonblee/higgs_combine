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

class WorkspaceTests(unittest.TestCase):
    def test_joint_sampling_keeps_anticorrelation(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        state = {'x': 0.}
        class Vars:
            def assignValueOnly(self, values):
                state.update(values)
        class Function:
            def __init__(self, sign): self.sign = sign
            def getVal(self): return (10. if self.sign == 1 else 20.) + self.sign*state['x']
        class Draws:
            def numEntries(self): return 2
            def get(self, i): return {'x': (-2., 2.)[i]}
        class Workspace:
            def obj(self, key): return NS(GetNuisanceParameters=lambda: None)
            def var(self, key): return NS(setVal=lambda v: None) if key == 'r' else None
            def allVars(self): return Vars()
            def saveSnapshot(self, *args): pass
            def loadSnapshot(self, *args): state.update(x=0.)
            def function(self, key): return Function(1 if key.endswith('QCD') else -1)
            def pdf(self, key): return NS(generate=lambda *args: Draws())
        class SampleFit(Fit):
            i = 0
            def floatParsFinal(self): return Norms(x=0.)
            def randomizePars(self):
                self.i += 1
                return {'x': -2. if self.i % 2 else 2.}
        file = NS(IsZombie=lambda: False, Get=lambda k: Workspace(), Close=lambda: None)
        root = NS(TFile=NS(Open=lambda _: file), RooRandom=NS(randomGenerator=lambda: NS(SetSeed=lambda _: None)))
        with patch.dict(sys.modules, ROOT=root):
            stages, _ = tool.extract_counting_workspace(Source(fit_b=SampleFit(), fit_s=SampleFit()),
                Path('workspace.root'), {'bin_2018': {'QCD': 10., 'tt': 20.}}, samples=2)
        for stage in tool.STAGES:
            self.assertEqual(stages[stage]['bin_2018']['total_background']['error'], 0.)
            self.assertEqual(stages[stage]['bin_2018']['processes']['QCD']['error'], 2.)


if __name__ == "__main__":
    unittest.main()
