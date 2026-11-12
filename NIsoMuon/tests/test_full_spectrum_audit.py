"""Bounded-POI reporting and correlated SR covariance propagation."""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import full_spectrum_fit as fs
import full_spectrum_audit as audit


class POITests(unittest.TestCase):
    def fit(self, reported=73., variance=1.64, lo=0., hi=1.44):
        parameter = SimpleNamespace(GetName=lambda: 'r', getVal=lambda: .007,
            getError=lambda: reported, getMin=lambda: 0., getMax=lambda: 100.,
            hasAsymError=lambda: True, getAsymErrorLo=lambda: lo, getAsymErrorHi=lambda: hi)
        return SimpleNamespace(floatParsFinal=lambda: [parameter], covarianceMatrix=lambda: [[variance]])

    def test_bounded_reported_error_is_not_covariance_or_profile_error(self):
        result = fs.poi_diagnostics(self.fit())
        self.assertEqual(result['r_error'], 73.)
        self.assertAlmostEqual(result['r_covariance_error'], math.sqrt(1.64))
        self.assertEqual(result['r_profile_error_high'], 1.44)
        self.assertAlmostEqual(result['r_profile_crossing_high'], 1.447)
        self.assertIsNone(result['r_profile_error_low'])
        self.assertIsNone(result['r_profile_crossing_low'])

    def test_measured_lower_crossing_and_invalid_covariance(self):
        result = fs.poi_diagnostics(self.fit(lo=-.003))
        self.assertAlmostEqual(result['r_profile_crossing_low'], .004)
        with self.assertRaisesRegex(ValueError, 'covariance'):
            fs.poi_diagnostics(self.fit(variance=-1.))

    def test_fixed_poi_is_not_reported_as_a_measurement(self):
        self.assertEqual(fs.poi_diagnostics(SimpleNamespace(floatParsFinal=lambda: [])), {})


@unittest.skipUnless(importlib.util.find_spec('ROOT'), 'Requires PyROOT')
class WorkspaceTests(unittest.TestCase):
    def test_combined_sr_retains_cross_era_covariance(self):
        import ROOT
        workspace = ROOT.RooWorkspace('w')
        workspace.factory('theta[0,-7,7]')
        workspace.factory('r[0,0,100]')
        rows = [dict(channel='era1', era='era1'), dict(channel='era2', era='era2')]
        for i, row in enumerate(rows):
            for process in ('QCD', 'DY', 'tt', 'ST', 'Others'):
                name = f'n_exp_bin{row["channel"]}_proc_{process}'
                if process == 'QCD':
                    workspace.factory(f'expr::{name}("10+{i+1}*@0", theta)')
                else:
                    workspace.factory(name+'[1]')
        parameters = ROOT.RooArgList(workspace.var('theta'), workspace.var('r'))
        fit = SimpleNamespace(floatParsFinal=lambda: parameters,
                              covarianceMatrix=lambda: [[1., 0.], [0., 10000.]])
        result = audit.sample_workspace(ROOT, workspace, fit, rows, 10000, 42)
        sigma = result['sample_std']
        self.assertAlmostEqual(sigma[0], 1., delta=.03)
        self.assertAlmostEqual(sigma[1], 2., delta=.06)
        self.assertAlmostEqual(sigma[2], 3., delta=.09)
        self.assertGreater(sigma[2], math.hypot(*sigma[:2]))
        np.testing.assert_allclose(result['linearised_error'], [1., 2., 3.], atol=1e-8)
        np.testing.assert_allclose(result['linearised_error_half_step'], [1., 2., 3.], atol=1e-8)
        self.assertEqual(workspace.var('theta').getVal(), 0.)
        self.assertEqual(workspace.var('theta').getMin(), -7.)
        self.assertEqual(workspace.var('r').getVal(), 0.)


if __name__ == '__main__':
    unittest.main()
