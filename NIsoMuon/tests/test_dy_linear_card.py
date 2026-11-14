"""Common shape, independent existing normalisations and exact card-level conservation."""
import copy
from pathlib import Path
import sys
import unittest
import subprocess
import importlib.util
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import dy_closure_shape as c
import full_spectrum_fit as fs
import limit_workflow_sideband_fit as shared
from test_full_spectrum_fit import example_model


class LinearCardTests(unittest.TestCase):
    def linear_model(self):
        model=example_model()
        for row in model['rows']:
            row['dy_closure_raw_up']=row['rates']['DY']*(c.INTERCEPT+c.SLOPE*(row['low']+row['high'])/2.)
            row['dy_closure_metadata']=dict(c.configuration(),source_sha256='fixture_'+row['era'])
        opt=fs.make_parser().parse_args(['--dy-shape-model','linear'])
        return fs.decorate_model(model,opt,{e:[r for r in model['rows'] if r['era']==e] for e in model['eras']})

    def test_one_shared_shape_prior_and_existing_lnn_not_nf_model(self):
        model=self.linear_model();card=shared.card_text(model,normalisation_kappa=1.5,prefix='FS')
        self.assertEqual(card.count(c.NUISANCE+' param 0 1'),1)
        self.assertNotIn('NFModel',card)
        for era in model['eras']:
            self.assertIn(fs.normalisation_name('DY',era)+' lnN',card)
        self.assertEqual(shared.nominal_predictions(card),
                         shared.nominal_predictions(shared.card_text(example_model(),normalisation_kappa=1.5,prefix='FS')))
        for theta in (-5.,-1.,-.4,0.,.4,1.,5.):
            varied=card.replace(c.NUISANCE+' param 0 1',c.NUISANCE+f' param {theta} 1')
            totals=shared.nominal_predictions(varied)
            for era in model['eras']:
                selected=[r for r in model['rows'] if r['era']==era]
                self.assertAlmostEqual(sum(totals[r['channel']] for r in selected),sum(r['asimov'] for r in selected),places=8)

    def test_legacy_name_changes_only_the_model_parameter(self):
        model = self.linear_model()
        self.assertEqual(c.configuration()['nuisance'], c.SOURCE_NUISANCE)
        current = shared.card_text(model, normalisation_kappa=1.5, prefix='FS')
        model['dy_linear_shape']['configuration']['nuisance'] = c.SOURCE_NUISANCE
        old = shared.card_text(model, normalisation_kappa=1.5, prefix='FS')
        self.assertEqual(old.replace(c.SOURCE_NUISANCE, c.NUISANCE), current)
        self.assertEqual(fs.base.spectrum_parameter_name(c.SOURCE_NUISANCE), c.NUISANCE)

    def test_nominal_zero_keeps_additive_stat_and_no_shape_multiplier(self):
        model=self.linear_model();row=model['rows'][0]
        row['rates']['DY']=0.;row['dy_stat']=2.;row['dy_closure_raw_up']=3.
        fs.decorate_model(model,fs.make_parser().parse_args(['--dy-shape-model','linear']),{e:[r for r in model['rows'] if r['era']==e] for e in model['eras']})
        card=shared.card_text(model,normalisation_kappa=1.5,prefix='FS')
        self.assertIn('CMS_NPS26009_DYStat_'+row['channel']+' param 0 1',card)
        self.assertNotIn('DYClosureBin_'+row['channel']+' rateParam',card)
        self.assertEqual(row['dy_closure_up'],0.)

    def test_opt_in_changes_signature_and_rejects_double_modelling(self):
        old=fs.make_parser().parse_args([]);new=fs.make_parser().parse_args(['--dy-shape-model','linear'])
        self.assertNotEqual(fs.signature(old),fs.signature(new))
        self.assertIn('DYlinearShape_v1',str(fs.model_directory(new,30.)))
        both=fs.make_parser().parse_args(['--dy-shape-model','linear','--dy-shape-uncertainty','.3'])
        with self.assertRaisesRegex(ValueError,'not both'):fs.configuration(both)
        with self.assertRaisesRegex(ValueError,'fresh preparation'):
            fs.decorate_model(example_model(),new,{e:[r for r in example_model()['rows'] if r['era']==e] for e in example_model()['eras']})

    @unittest.skipUnless(importlib.util.find_spec('ROOT'),'PyROOT unavailable')
    def test_rooformula_matches_normalisation_at_nonzero_shape(self):
        script = r"""
import sys
sys.path.insert(0,'tests')
import ROOT
from pathlib import Path
from test_dy_linear_card import LinearCardTests
import dy_closure_shape as c
case=LinearCardTests();model=case.linear_model()
import tempfile
workspace=ROOT.RooWorkspace('w')
workspace.factory(c.NUISANCE+'[0,-7,7]')
for era in model['eras']:
 lines=c.card_lines(model['rows'],era)
 for line in lines:
  name,_,channel,_,formula,dependencies=line.split()
  if not workspace.arg(name):
   assert workspace.factory('expr::'+name+'("'+formula+'",'+dependencies+')')
 for row in [r for r in model['rows'] if r['era']==era]:
  channel=row['channel'];norm=('DYClosureNorm_'+era if row['channel']==next(r['channel'] for r in model['rows'] if r['era']==era) else 'DYClosureNormUse_'+channel)
  assert workspace.factory('expr::n_exp_bin'+channel+'_proc_DY("'+str(row['rates']['DY'])+'*@0*@1",DYClosureBin_'+channel+','+norm+')')
for v in (-4.,-.5,0.,.7,3.):
 workspace.var(c.NUISANCE).setVal(v)
 for era in model['eras']:
  selected=[r for r in model['rows'] if r['era']==era]
  yields=[workspace.function('n_exp_bin'+row['channel']+'_proc_DY').getVal() for row in selected]
  case.assertAlmostEqual(sum(yields),sum(row['rates']['DY'] for row in selected),places=8)
workspace.var(c.NUISANCE).setVal(0.)
with tempfile.TemporaryDirectory() as directory:
 workspace.writeToFile(directory+'/workspace.root')
 import full_spectrum_fit as fs
 fs.audit_dy_shape_workspace(model,directory)
 case.assertTrue(Path(directory+'/dy_shape_audit.json').is_file())

"""
        subprocess.run([sys.executable,'-c',script],check=True)



if __name__=='__main__':unittest.main()
