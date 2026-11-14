"""Review-model isolation, joint QCD screening and toy accounting."""
import copy
import hashlib
import json
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import tarfile
from types import SimpleNamespace
import unittest

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import full_spectrum_fit as fs
import full_spectrum_toys as toys
import limit_workflow_sideband_fit as shared
import prepare_full_spectrum_review as review
import qcd_joint_closure as joint
from test_full_spectrum_fit import example_model


def window_example():
    model=example_model()
    model['fit_kind']='window_count_with_sidebands'
    model['settings']['fit_layout']='window'
    for row in model['rows']:
        if row['low']==29.:
            row['region']='SR';row['channel']='bin_'+row['era']+'_SR'
    text=shared.card_text(model,normalisation_kappa=1.5,prefix='FS')
    model['card_sha256']=hashlib.sha256(text.encode()).hexdigest()
    return model,text


class ReviewTests(unittest.TestCase):
    def test_default_options_preserve_legacy_signatures_and_paths(self):
        current=fs.make_parser().parse_args(['--fit-layout','window'])
        old=copy.copy(current)
        for name in ('dy_shape_uncertainty','dy_normalisation','dy_norm_max'):
            delattr(old,name)
        self.assertEqual(fs.signature(old),fs.signature(current))
        self.assertEqual(fs.model_directory(old,30),fs.model_directory(current,30))

    def test_free_scale_has_no_prior_constraint(self):
        parameter=SimpleNamespace(getVal=lambda:1.,getError=lambda:.2)
        result=fs.normalisation_diagnostics(parameter,dict(kind='rateParam',bounds=[0,10],kappa=None))
        self.assertEqual(result['scale'],1.)
        self.assertIsNone(result['constraint'])
        self.assertAlmostEqual(result['scale_low'],.8)

    def test_review_card_preserves_yields_and_unrelated_nuisance_lines(self):
        model,text=window_example()
        opt=fs.make_parser().parse_args(['--fit-layout','window','--dy-shape-uncertainty','.3','--dy-normalisation','free'])
        windows={e:[r for r in model['rows'] if r['era']==e] for e in model['eras']}
        transformed=fs.decorate_model(copy.deepcopy(model),opt,windows)
        changed=review.review_card(text,transformed,opt)
        self.assertEqual(shared.nominal_predictions(text),shared.nominal_predictions(changed))
        closure_lines=[line.split() for line in changed.splitlines() if line.startswith('CMS_NPS26009_DYClosure_')]
        self.assertEqual(len(closure_lines),len(model['rows']))
        self.assertEqual(len({line[0] for line in closure_lines}),len(model['rows']))
        for line in closure_lines:
            values=[v for v in line[2:] if v!='-']
            self.assertEqual(len(values),1)
            np.testing.assert_allclose([float(v) for v in values[0].split('/')],[.7,1.3])
        removed={fs.normalisation_name('DY',e) for e in model['eras']}
        self.assertFalse(any(line.split()[:2]==[name,'lnN'] for name in removed for line in changed.splitlines()))
        for line in text.splitlines():
            if line.split() and line.split()[0] not in removed:
                self.assertIn(line,changed.splitlines())
        regenerated=shared.card_text(transformed,normalisation_kappa=1.5,prefix='FS')
        self.assertCountEqual(regenerated.splitlines(),changed.splitlines())

    def test_legacy_review_card_and_impact_labels(self):
        model, text = window_example()
        old = copy.deepcopy(model)
        old['normalisations'] = {
            name.replace('_Norm_', '_inSituNorm_'): spec
            for name, spec in model['normalisations'].items()
        }
        legacy = shared.card_text(old, normalisation_kappa=1.5, prefix='FS')
        self.assertIn('FS_QCDFitMode1_2018 param', legacy)
        opt = fs.make_parser().parse_args(['--fit-layout', 'window'])
        self.assertEqual(review.review_card(legacy, model, opt), text)
        self.assertEqual(fs.base.spectrum_parameter_name('FS_QCDFitMode4_2018'),
                         'CMS_NPS26009_QCDStat_mode4_2018')

    def test_preparation_reuses_saved_card_without_root_inputs(self):
        model,text=window_example()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            path=source/'model.json';path.write_text(json.dumps(model))
            (source/'datacard.txt').write_text(text)
            result=review.prepare(path,root/'review')
            dest=Path(result['directory']);after=json.loads((dest/'model.json').read_text())
            self.assertEqual(after['review_parent']['card_sha256'],model['card_sha256'])
            self.assertIn('DYshape0p3_DYfree10',str(dest))
            self.assertEqual(path.read_text(),json.dumps(model))
            self.assertEqual((source/'datacard.txt').read_text(),text)
            self.assertTrue(all(a['rates']==b['rates'] for a,b in zip(model['rows'],after['rows'])))
            with self.assertRaisesRegex(ValueError,'already exists'):
                review.prepare(path,root/'review')

    def test_zero_dy_keeps_additive_source_stat_and_zero_nominal(self):
        model,text=window_example();row=model['rows'][0]
        row['rates']['DY']=0.;row['dy_stat']=2.;row['asimov']=sum(row['rates'][p] for p in shared.BACKGROUNDS)
        opt=fs.make_parser().parse_args(['--dy-shape-uncertainty','.3','--dy-normalisation','free'])
        fs.decorate_model(model,opt,{e:[r for r in model['rows'] if r['era']==e] for e in model['eras']})
        card=shared.card_text(model,normalisation_kappa=1.5,prefix='FS')
        self.assertIn('CMS_NPS26009_DYStat_'+row['channel']+' param 0 1',card)
        self.assertAlmostEqual(shared.nominal_predictions(card)[row['channel']],row['asimov'])


class JointTests(unittest.TestCase):
    def test_joint_response_reduces_transport_residual(self):
        rows=[dict(region=str(i),rates={'QCD':q},qcd_shape=0.,qcd_modes=[mode],asimov=q+50.)
              for i,(q,mode) in enumerate(zip([100.,100.,100.],[20.,-20.,0.]))]
        record=dict(source=[100.,100.,100.],target=[120.,80.,100.],
                    source_variance=[10.,10.,10.],target_variance=[10.,10.,10.],ratio=[1.2,.8,1.])
        result=joint.profile_qcd(rows,record,1.5,'prior')
        self.assertEqual(result['status'],'diagnostic_converged')
        self.assertLess(result['objective'],result['baseline_quadratic'])
        self.assertGreater(result['ss_mode_pulls'][0],0)

    def test_zero_ratio_residual_recovers_nominal(self):
        rows=[dict(region=str(i),rates={'QCD':10.},qcd_shape=2.,qcd_modes=[1.],asimov=30.) for i in range(3)]
        record=dict(source=[100.]*3,target=[100.]*3,source_variance=[100.]*3,target_variance=[100.]*3,ratio=[1.]*3)
        for mode in ('prior','free'):
            result=joint.profile_qcd(rows,record,1.5,mode)
            self.assertAlmostEqual(result['objective'],0.,places=12)
            self.assertAlmostEqual(result['normalisation_scale'],1.)


class ToyTests(unittest.TestCase):
    def test_commands_fix_truth_nuisances_but_fit_profiles_them(self):
        task=dict(mass=30.,count=5,seed=7,injected_r=2.,r_min=-10.,r_max=100.,toy_errors=False)
        gen,fit=toys.commands(task,'fit.root','truth.root')
        self.assertIn('truth.root',gen);self.assertIn('fit.root',fit)
        self.assertIn('--toysNoSystematics',gen);self.assertIn('--toysNoSystematics',fit)
        self.assertNotIn('--freezeParameters',fit)
        self.assertIn('--noErrors',fit)
        self.assertEqual(fit[fit.index('--rMin')+1],'-10.0')

    def test_alternative_truth_requires_matching_card_and_positive_factors(self):
        model,text=window_example();channel=model['rows'][0]['channel']
        truth=dict(mass=30.,fit_card_sha256='abc',factors={channel:{'QCD':1.4}})
        result=toys.truth_card(text,model,truth,'abc')
        self.assertIn(f'Truth_{channel}_QCD rateParam {channel} QCD 1.399',result)
        with self.assertRaisesRegex(ValueError,'match'):
            toys.truth_card(text,model,truth,'def')
        truth['factors'][channel]['QCD']=-1
        with self.assertRaisesRegex(ValueError,'Invalid'):
            toys.truth_card(text,model,truth,'abc')

    def test_collection_reports_failed_missing_and_boundary_toys(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            task=dict(index=0,mass=30.,injected_r=0.,reference_r=1.,truth_label='nominal',truth_sha256='a',fit_card_sha256='b',count=3,r_min=-5.,r_max=10.)
            second=dict(task,index=1)
            payload=dict(tasks=[task,second]);payload['manifest_sha256']=toys.manifest_digest(payload)
            (root/'manifest.json').write_text(json.dumps(payload))
            path=root/'jobs/job_00000/result';path.mkdir(parents=True)
            (path/'result.json').write_text(json.dumps(dict(task=task,entries=[dict(r=-5.,status=0),dict(r=1.,status=0),dict(r=3.,status=4)])))
            toys.collect(root)
            result=json.loads((root/'toy_summary.json').read_text())['ensembles'][0]
            self.assertFalse(result['complete']);self.assertEqual(result['failed'],1)
            self.assertEqual(result['boundary_fits'],1);self.assertEqual(result['missing_chunks'],[1])
            self.assertEqual(result['mean_r'],-2.)
            with tarfile.open(path.parent/'result.tar.gz','w:gz') as archive:
                archive.add(path,arcname='result')
            shutil.rmtree(path)
            toys.collect(root)
            packed=json.loads((root/'toy_summary.json').read_text())['ensembles'][0]
            self.assertEqual(result,packed)


@unittest.skipUnless(importlib.util.find_spec('ROOT') and shutil.which('combine') and shutil.which('text2workspace.py'),
                     'Requires CMSSW/Combine/PyROOT')
class CombineReviewTests(unittest.TestCase):
    def test_free_dy_with_independent_closure_recovers_injection(self):
        model,_=window_example()
        opt=fs.make_parser().parse_args(['--fit-layout','window','--dy-shape-uncertainty','.3','--dy-normalisation','free','--inject-r','1'])
        fs.decorate_model(model,opt,{e:[r for r in model['rows'] if r['era']==e] for e in model['eras']})
        card=shared.card_text(model,normalisation_kappa=1.5,prefix='FS')
        model['card_sha256']=hashlib.sha256(card.encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'datacard.txt').write_text(card)
            fs.run_fit(model,root,opt)
            summary=json.loads((root/'fit_r1/fit_summary.json').read_text())
            self.assertAlmostEqual(summary['fits']['fit_s']['r'],1.,delta=.05)
            for value in summary['fits']['fit_s']['normalisations'].values():
                self.assertAlmostEqual(value['scale'],1.,delta=.05)
                if value['process']=='DY':self.assertIsNone(value['constraint'])


if __name__=='__main__':unittest.main()
