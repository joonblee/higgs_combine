"""Geometry, correlation, blindness and card regression tests; optional Combine smoke test."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import limit_workflow_sideband_fit as sb
import limit_workflow as base


def example_model():
    rows = []
    for era in ('2018', '2023'):
        for region, low, high, qcd, dy in [('L00', 11., 20., 60., 20.), ('SR', 20., 24., 10., 15.), ('R00', 24., 30., 5., 30.)]:
            rates = dict(sig=5. if region == 'SR' else .01, QCD=qcd, DY=dy, tt=40., ST=3., Others=2.)
            rows.append(dict(era=era, region=region, channel=f'bin_{era}_{region}', low=low, high=high,
                             rates=rates, asimov=sum(rates[p] for p in sb.BACKGROUNDS),
                             qcd_shape=qcd*.03, qcd_modes=[qcd*.02, dy*.01], dy_stat=dy*.1,
                             nuisances={'tt_stat': {'tt': '0.95/1.05'}, 'QCD_norm': {'QCD': '0.8/1.2'},
                                        'DY_NFModel': {'DY': '0.9/1.1'}, 'DY_LightJetStat': {'DY': '0.9/1.1'},
                                        'PDF_scale_tt_muR': {'tt': '0.9/1.1'}}))
    opt = sb.make_parser().parse_args(['--masses', '22'])
    return dict(schema_version=1, blind=True, observation='nominal_background_Asimov', mass=22., target='Run2Run3',
                rows=rows, eras={'2018':{},'2023':{}}, alpha_internal_unit=.01, settings=vars(opt))


class GeometryTests(unittest.TestCase):
    def test_real_resolution_edges_and_gaps(self):
        args = base.build_parser().parse_args(['--target','full','--sigfit-dir',str(Path(base.__file__).parent/'sigfit_inputs')])
        base.prepare_resolution_coefficients(args)
        edges = np.arange(0, 100.001, .02)
        for era in base.YEARS:
            for mass in (12., 15., 30., 70.):
                sigma = mass*base.relative_resolution(args, era, mass)
                for exclude in (5., 7., 10.):
                    rows = sb.regions(edges, mass, sigma, exclude_nsigma=exclude)
                    sr = rows[0]
                    ordered = sorted(rows, key=lambda r:r['low'])
                    self.assertGreaterEqual(ordered[0]['low'], 11.)
                    self.assertLessEqual(ordered[-1]['high'], 80.)
                    self.assertTrue(all(a['high'] <= b['low'] for a,b in zip(ordered,ordered[1:])))
                    for r in rows[1:]:
                        self.assertTrue(r['high'] <= mass-exclude*sigma+1e-7 or r['low'] >= mass+exclude*sigma-1e-7)
                    expected = base.mass_window(args,era,mass)
                    self.assertLessEqual(sr['low'], expected[0]+1e-7)
                    self.assertGreaterEqual(sr['high'], expected[1]-1e-7)

    def test_one_sided_low_and_high(self):
        grid = np.arange(11.,80.01,.02)
        low = sb.regions(grid,12.,.2,exclude_nsigma=10.)
        high = sb.regions(grid,70.,1.2,exclude_nsigma=10.)
        self.assertTrue(all(r['region'].startswith('R') for r in low[1:]))
        self.assertTrue(all(r['region'].startswith('L') for r in high[1:]))

    def test_not_enough_sideband(self):
        with self.assertRaisesRegex(ValueError,'Fewer than two'):
            sb.regions([11.,20.,30.,40.,80.],30.,10.,exclude_nsigma=10.)

    def test_bad_options_rejected(self):
        with self.assertRaises(ValueError):
            sb.configuration(sb.make_parser().parse_args(['--exclude-nsigma','4']))
        with self.assertRaises(ValueError):
            sb.configuration(sb.make_parser().parse_args(['--outer-nsigma','5']))


class StatisticalTests(unittest.TestCase):
    def test_joint_sum_retains_covariance(self):
        covariance = np.array([[4.,-1.],[-1.,2.]])
        gradient = np.array([[1.,2.],[3.,-1.],[-.2,4.]])
        response = gradient @ sb.covariance_factor(covariance)
        np.testing.assert_allclose(response @ response.T, gradient @ covariance @ gradient.T, atol=1e-12)
        self.assertAlmostEqual(np.sum(response,axis=0) @ np.sum(response,axis=0),
                               np.sum(gradient,axis=0) @ covariance @ np.sum(gradient,axis=0))
        self.assertNotAlmostEqual(np.sum(response**2), float(np.sum(response,axis=0) @ np.sum(response,axis=0)))

    def test_indefinite_covariance_fails(self):
        with self.assertRaises(ValueError):
            sb.covariance_factor([[1.,2.],[2.,1.]])

    def test_qcd_dy_degeneracy_detected(self):
        rows = example_model()['rows']
        self.assertTrue(sb.separation_diagnostic(rows)['identifiable'])
        for r in rows:
            r['rates']['DY'] = 2*r['rates']['QCD']
        self.assertFalse(sb.separation_diagnostic(rows)['identifiable'])

    def test_card_nominal_equals_asimov_and_signal_leakage_included(self):
        model = example_model()
        text = sb.card_text(model)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'datacard.txt'
            path.write_text(text)
            predictions = sb.nominal_predictions(path.read_text())
        for row in model['rows']:
            self.assertAlmostEqual(predictions[row['channel']],row['asimov'])
            self.assertGreater(row['rates']['sig'],0.)

    def test_card_correlations_and_replaced_norms(self):
        text = sb.card_text(example_model())
        self.assertEqual(text.count('beta_QCD_2018 rateParam'),3)
        self.assertEqual(text.count('SB_QCDFitMode0_2018 param'),1)
        self.assertEqual(text.count('QCDscale_ren_ttbar lnN'),1)
        self.assertNotIn('NFModel',text)
        self.assertNotIn('bckgNorm_QCD',text)
        self.assertIn('MCstat_ttbar_BJetOS_2018_L00 lnN',text)
        self.assertIn('MCstat_ttbar_BJetOS_2018_SR lnN',text)

    def test_zero_dy_keeps_absolute_uncertainty(self):
        model=example_model()
        row=model['rows'][0]
        row['asimov']-=row['rates']['DY']
        row['rates']['DY']=0.
        row['nuisances']['DY_LightJetStat']['DY']='-'
        text=sb.card_text(model)
        self.assertIn('max(0.,2*@0)',text)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'datacard.txt'; path.write_text(text)
            predictions=sb.nominal_predictions(path.read_text())
        for row in model['rows']:
            self.assertAlmostEqual(predictions[row['channel']],row['asimov'])

    def test_sr_masked_and_signal_not_fitted(self):
        commands=sb.fit_commands(example_model(),Path('/tmp/study'))
        fit=commands[1]
        self.assertIn('--channel-masks',commands[0])
        self.assertIn('--skipSBFit',fit)
        settings=fit[fit.index('--setParameters')+1]
        self.assertIn('mask_bin_2018_SR=1',settings)
        self.assertIn('mask_bin_2018_L00=0',settings)
        self.assertIn('-t',fit)
        self.assertEqual(fit[fit.index('--expectSignal')+1],'0')

    def test_target_data_blocked_before_open(self):
        reader=object.__new__(sb.BlindedReader)
        with patch.object(base.RootReader,'_file',side_effect=AssertionError('opened target data')):
            with self.assertRaisesRegex(base.WorkflowError,'forbidden'):
                reader._file('/tmp/nominal/data.root')


@unittest.skipUnless(shutil.which('combine') and shutil.which('text2workspace.py') and importlib.util.find_spec('ROOT'), 'Requires CMSSW/Combine/PyROOT')
class CombineSmokeTest(unittest.TestCase):
    def test_asimov_sideband_and_joint_limit(self):
        model=example_model()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'M22'
            path.mkdir()
            (path/'datacard.txt').write_text(sb.card_text(model))
            for command in sb.fit_commands(model,path):
                subprocess.run(command,cwd=path,check=True,stdout=subprocess.DEVNULL)
            result=sb.extract_fit(model,path,200,12345)
            for value in result['fit']['normalisations'].values():
                self.assertAlmostEqual(value['value'],1.,delta=.05)
            subprocess.run(['combine','-M','AsymptoticLimits',str(path/'datacard.txt'),'-m','22','--run','blind','--rMax','100'],cwd=path,check=True,stdout=subprocess.DEVNULL)
            # extract_fit's public JSON sits next to the mass working directory.
            (path.parent/'sideband_fit_Run2Run3_M22.json').unlink(missing_ok=True)


if __name__=='__main__':
    unittest.main()
