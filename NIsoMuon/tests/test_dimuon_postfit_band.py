import sys
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_dimuon_mass import replay_yield_covariance, saved_yield_covariance
import full_spectrum_fit as fs


class PostfitBandTests(unittest.TestCase):
    def test_replay_uses_joint_deviations_from_best_fit(self):
        fit, parameters = Mock(), Mock()
        fit.randomizePars.side_effect = [-1., 1.]
        parameters.assignValueOnly.side_effect = lambda x: setattr(parameters, 'theta', x)
        fit.floatParsFinal.return_value = 0.

        def prediction():
            scale = 2.**parameters.theta
            return np.array([[scale, 1./scale], [scale, 0.]])

        # The two draws give the same first-bin yield of 2.5; its
        # standard deviation about the sampled mean is zero, but its RMS
        # difference from the best fit of 2 is 0.5.
        cov = replay_yield_covariance(fit, parameters, prediction, np.array([2., 1.]), 2)
        np.testing.assert_allclose(cov, [[0.25, 0.125], [0.125, 0.625]])
        self.assertEqual(parameters.theta, 0.)

    def test_overall_shapes_are_opt_in(self):
        opt = fs.make_parser().parse_args([])
        self.assertNotIn('--saveOverallShapes', fs.fit_commands(opt, 12, Path('/tmp/model'))[1])
        opt.fit_output = 'overall'
        command = fs.fit_commands(opt, 12, Path('/tmp/model'), 'fit_b')[1]
        for flag in ('--saveOverallShapes', '--saveWithUncertainties', '--skipSBFit'):
            self.assertIn(flag, command)

    @unittest.skipUnless(find_spec('ROOT'), 'PyROOT required')
    def test_read_root_covariance_by_label_and_merge_in_event_units(self):
        import ROOT
        with ROOT.TMemFile('postfit_band_test.root', 'RECREATE') as f:
            f.mkdir('shapes_fit_b').cd()
            cov = ROOT.TH2D('overall_total_covar', '', 2, 0, 2, 2, 0, 2)
            width = ROOT.TH1D('total_bin_width', '', 2, 0, 2)
            bkg = ROOT.TH1D('total_background', '', 2, 0, 2)
            for axis in (cov.GetXaxis(), cov.GetYaxis(), width.GetXaxis(), bkg.GetXaxis()):
                axis.SetBinLabel(1, 'SR_B_0')
                axis.SetBinLabel(2, 'SR_A_0')
            for i, dx, density, error in [(1, 2., 8., 3.), (2, 3., 5., 4.)]:
                width.SetBinContent(i, dx)
                bkg.SetBinContent(i, density)
                bkg.SetBinError(i, error)
            for i, row in enumerate([[9., -6.], [-6., 16.]], start=1):
                for j, value in enumerate(row, start=1):
                    cov.SetBinContent(i, j, value)
            for h in (width, bkg, cov):
                h.Write()
            rows = [{'channel': 'SR_A'}, {'channel': 'SR_B'}]
            fitted = np.array([[15.], [16.]])
            result = saved_yield_covariance(f, rows, fitted, [[0, 1]])
            np.testing.assert_allclose(result, [[12.**2 + 6.**2 - 2 * 36.]])
            with self.assertRaisesRegex(ValueError, 'labels'):
                saved_yield_covariance(f, [rows[0], rows[0]], fitted, [[0, 1]])


if __name__ == '__main__':
    unittest.main()
