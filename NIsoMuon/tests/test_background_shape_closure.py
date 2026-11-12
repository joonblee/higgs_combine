"""Shape-only closure with shared normalisation, signed MC and exact fit bins."""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import background_shape_closure as closure


class ShapeTests(unittest.TestCase):
    def test_pure_normalisation_difference_closes_with_one_null_direction(self):
        result = closure.shape_closure([10, 20, 30], [20, 40, 60], [10, 20, 30], [20, 40, 60])
        self.assertEqual(result['normalisation'], 2.)
        self.assertAlmostEqual(result['chi2'], 0.)
        self.assertEqual(result['covariance_rank'], 2)
        np.testing.assert_allclose(np.array(result['residual_covariance']).sum(axis=0), 0, atol=1e-12)

    def test_ratio_uncertainty_includes_same_sample_integral_correlation(self):
        # Shape ratios for a two-bin, unit-count source/target each have
        # variance 1 after integral matching; a fixed-factor treatment gives 2.
        result = closure.shape_closure([1, 1], [1, 1], [1, 1], [1, 1])
        np.testing.assert_allclose(result['ratio_error'], [1, 1])
        self.assertAlmostEqual(result['normalisation_error'], 1.)

    def test_covariance_agrees_with_independent_mc_fluctuations(self):
        source = np.array([10000., 20000., 30000.])
        target = np.array([12000., 19000., 35000.])
        vs, vt = source*2, target*3
        result = closure.shape_closure(source, target, vs, vt)
        rng = np.random.default_rng(817)
        s = rng.normal(source, np.sqrt(vs), size=(100000, 3))
        t = rng.normal(target, np.sqrt(vt), size=(100000, 3))
        residual = t-s*(t.sum(axis=1)/s.sum(axis=1))[:, None]
        np.testing.assert_allclose(np.cov(residual, rowvar=False), result['residual_covariance'], rtol=.025)

    def test_signed_bins_retained_and_undefined_ratio_not_clipped(self):
        result = closure.shape_closure([-1, 20, 30], [2, 18, 40], [10, 20, 30], [2, 18, 40])
        self.assertEqual(result['source'][0], -1.)
        self.assertLess(result['prediction'][0], 0.)
        self.assertIsNone(result['ratio'][0])
        self.assertEqual(closure.shape_closure([0, 0], [1, 2], [1, 1], [1, 2])['status'],
                         'inconclusive_nonpositive_total')

    def test_missing_mc_variance_discrepancy_is_not_false_success(self):
        result = closure.shape_closure([1, 1], [1, 2], [0, 0], [0, 0])
        self.assertIsNone(result['chi2'])
        self.assertGreater(result['residual_in_zero_variance_subspace'], 0.)

    def test_collision_data_filename_is_rejected_before_opening(self):
        reader = closure.MCReader(None)
        with self.assertRaisesRegex(ValueError, 'Only the nominal'):
            reader.histogram('/tmp/data.root', 'OS_POGMedium_tight_BJet_NIsoDimuon')


@unittest.skipUnless(importlib.util.find_spec('ROOT'), 'Requires PyROOT')
class RootTests(unittest.TestCase):
    def test_integrals_preserve_whole_bins_and_signed_sumw2(self):
        import ROOT
        hist = ROOT.TH1D('closure_test', '', 3, 11., 14.)
        hist.Sumw2()
        hist.SetBinContent(1, -2.)
        hist.SetBinError(1, 3.)
        hist.SetBinContent(2, 5.)
        hist.SetBinError(2, 4.)
        reader = closure.MCReader(ROOT)
        values, variance = reader.integrate(hist, [dict(low=11., high=13.)])
        self.assertEqual(values, [3.])
        self.assertEqual(variance, [25.])
        with self.assertRaisesRegex(ValueError, 'native binning'):
            reader.integrate(hist, [dict(low=11.1, high=13.)])


if __name__ == '__main__':
    unittest.main()
