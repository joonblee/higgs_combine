"""Full-domain geometry, card semantics, blinding and optional Combine test."""
import copy
from contextlib import ExitStack, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import full_spectrum_fit as fs
import limit_workflow as base
import limit_workflow_sideband_fit as shared


def example_model():
    opt = fs.make_parser().parse_args(['--masses', '30'])
    rows = []
    for era in ('2018', '2023'):
        for i, (lo, hi, qcd, dy, tt, sig) in enumerate([
                (11.,20.,1000.,120.,150.,.01), (20.,29.,200.,300.,350.,.2),
                (29.,31.,20.,80.,100.,12.29), (31.,80.,40.,600.,1200.,0.)]):
            rates = dict(sig=sig, QCD=qcd, DY=dy, tt=tt, ST=3., Others=2.)
            region = f'B{i:04d}'
            rows.append(dict(era=era, region=region, channel=f'bin_{era}_{region}',
                low=lo, high=hi, rates=rates, asimov=sum(rates[p] for p in shared.BACKGROUNDS),
                qcd_shape=.02*qcd, qcd_modes=[.01*qcd, (-1)**i*.01*qcd], dy_stat=.1*dy,
                nuisances={'tt_stat': {'tt':'0.95/1.05'}, 'QCD_norm': {'QCD':'0.8/1.2'},
                           'DY_NFModel': {'DY':'0.9/1.1'}, 'DY_LightJetStat': {'DY':'0.9/1.1'},
                           'PDF_scale_tt_muR': {'tt':'0.9/1.1'}}))
    return dict(schema_version=fs.SCHEMA, blind=True, fit_kind='full_spectrum',
        mass=30., target=opt.target, rows=rows, eras={'2018':{},'2023':{}},
        alpha_internal_unit=.01, settings=vars(opt),
        normalisations={fs.normalisation_name(p,e):dict(process=p,era=e,kappa=1.5)
                        for e in ('2018','2023') for p in fs.MAJOR})


class GeometryTests(unittest.TestCase):
    def test_full_domain_partition_all_eras(self):
        coeffs = json.loads((Path(base.__file__).parent/'sigfit_inputs/resolution_coefficients.json').read_text())
        grid = np.round(np.arange(11., 80.001, .02),8)
        for era, ab in coeffs.items():
            resolution = lambda m: m*(ab['a']*m+ab['b'])
            for factor in (.5,1.,2.):
                rows = fs.resolution_bins(grid,resolution,factor)
                self.assertEqual(rows[0]['low'],11.)
                self.assertEqual(rows[-1]['high'],80.)
                self.assertTrue(all(a['high']==b['low'] for a,b in zip(rows,rows[1:])))
                self.assertTrue(all(r['low'] in grid and r['high'] in grid for r in rows))
                self.assertAlmostEqual(sum(r['high']-r['low'] for r in rows),69.)
                # Integrate an arbitrary native template once and only once.
                counts = np.arange(len(grid)-1)+1
                integrated = sum(counts[np.searchsorted(grid,r['low']):np.searchsorted(grid,r['high'])].sum() for r in rows)
                self.assertEqual(integrated,counts.sum())
                ratios = [(r['high']-r['low'])/resolution((r['low']+r['high'])/2) for r in rows]
                self.assertLess(abs(np.median(ratios)-factor),.15)

    def test_coarse_grid_is_not_split_and_tiny_terminal_bin_merged(self):
        rows=fs.resolution_bins([11.,12.,13.,20.,40.,79.9,80.],lambda m:2.)
        self.assertEqual(rows[-1]['high'],80.)
        self.assertNotEqual(rows[-1]['low'],79.9)
        self.assertEqual(rows[0]['high'],13.)

    def test_no_domain_truncation(self):
        for edges in ([11.1,20.,80.],[11.,20.,79.9]):
            with self.assertRaisesRegex(ValueError,'both 11 and 80'):
                fs.resolution_bins(edges,lambda m:1.)

    def test_cli_targets_and_invalid_inputs(self):
        defaults=fs.make_parser().parse_args([])
        self.assertEqual(defaults.backend,'condor')
        self.assertEqual(defaults.condor_maxjobs,0)
        with patch.dict('os.environ',{'NPS26009_CONDOR_FILESYSTEM_DOMAIN':'old-domain'}):
            self.assertIsNone(fs.make_parser().parse_args([]).condor_filesystem_domain)
        for value,expected in [('full','Run2Run3'),('Run2+3','Run2Run3'),('Run2','Run2'),('run3','Run3'),('2018','2018')]:
            opt=fs.make_parser().parse_args(['--target',value])
            fs.configuration(opt)
            self.assertEqual(opt.target,expected)
        for arguments in (['--norm-prior','1'],['--inject-r','nan'],['--masses','30,30'],['--masses','10']):
            with self.assertRaises(ValueError):
                fs.configuration(fs.make_parser().parse_args(arguments))


class CardTests(unittest.TestCase):
    def build_synthetic_model(self, audit=None):
        opt=fs.make_parser().parse_args(['--target','2018','--masses','30'])
        args=fs.configuration(opt)
        windows={'2018':[dict(region='B0000',low=11.,high=30.),
                         dict(region='B0001',low=30.,high=80.)]}
        metadata={'fit':{'covariance':[[.04]]}}
        axis=SimpleNamespace(FindFixBin=lambda x:x)
        native=SimpleNamespace(GetXaxis=lambda:axis)
        derivative=SimpleNamespace(Integral=lambda a,b:.2*(b-a+2e-8))
        inputs=(metadata,native,[None]*6+[derivative,derivative])
        reader=SimpleNamespace(_file=lambda path:object())
        def channel_builder(reader,args,mass,files,eras,windows):
            low,high=windows['2018']
            width=high-low
            ch=SimpleNamespace(raw_rates=dict(sig=width,QCD=2*width,DY=width,tt=width,ST=1.,Others=1.),
                nuisances={'PDF_scale_ST_muF':{'ST':'0.8/1.2'},
                           'PDF_scale_ST_muR':{'ST':'0.9/1.1'}},
                qcd_shape_sigma_down=.1,qcd_shape_sigma_up=.2,
                dy_nf_amc=.1,dy_lightjet_error=1.,warnings=[
                    'The ST diagonal scale variation (Down) does not factorise into the muF and muR '
                    'responses in 2018: observed ratio=2, product=0.5.'])
            return {'2018':ch}, audit if audit is not None else base.AuditLog()
        with ExitStack() as stack:
            stack.enter_context(patch.object(base,'prepare_resolution_coefficients'))
            stack.enter_context(patch.object(base,'relative_resolution',return_value=.01))
            stack.enter_context(patch.object(base,'read_qcd_stat_inputs',return_value=inputs))
            stack.enter_context(patch.object(base,'read_qcd_stat_metadata',return_value=metadata))
            stack.enter_context(patch.object(base,'qcd_root_window_statistics',return_value={'gradient':[1.]}))
            stack.enter_context(patch.object(base,'build_channels_for_mass',side_effect=channel_builder))
            model=fs.build_model(reader,args,opt,30.,{'2018':'signal.root'},windows)
        return model

    def test_full_builder_preserves_all_bins_and_scales_full_signal(self):
        model=self.build_synthetic_model()
        self.assertEqual(model['eras']['2018']['bin_edges'],[11.,30.,80.])
        self.assertEqual(len(model['rows']),2)
        self.assertAlmostEqual(sum(r['rates']['sig'] for r in model['rows']),25.)
        self.assertAlmostEqual(model['alpha_internal_unit'],25./69.)
        self.assertEqual(model['rows'][0]['qcd_modes'],[.2])
        self.assertEqual(len(model['normalisations']),3)
        text=shared.card_text(model,normalisation_kappa=1.5,prefix='FS')
        for r in model['rows']:
            self.assertAlmostEqual(shared.nominal_predictions(text)[r['channel']],r['asimov'])

    def test_diagonal_scale_diagnostic_is_saved_without_dropping_axis_nuisances(self):
        audit=base.AuditLog()
        audit.add('PDF_scale_factorisation/ST','2018:Down',False)
        audit.add('PDF_scale_factorisation/ST','2018:Up',True)
        output=io.StringIO()
        with redirect_stdout(output):
            model=self.build_synthetic_model(audit)
        self.assertIn('[SCALE DIAGNOSTIC] 2018/B0000',output.getvalue())
        self.assertIn('observed ratio=2, product=0.5',output.getvalue())
        for entry in model['input_audits']:
            diagnostic=entry['diagnostic_failures'][0]
            self.assertEqual(diagnostic['check'],'PDF_scale_factorisation/ST')
            self.assertEqual(diagnostic['labels'],['2018:Down'])
            self.assertEqual((diagnostic['found'],diagnostic['expected']),(1,2))
            self.assertIn('observed ratio=2, product=0.5',diagnostic['messages'][0])
        card=shared.card_text(model,normalisation_kappa=1.5,prefix='FS')
        self.assertIn('CMS_NPS26009_QCDscale_fac_singleTop lnN',card)
        self.assertIn('CMS_NPS26009_QCDscale_ren_singleTop lnN',card)
        self.assertNotIn('muRmuF',card)
        baseline=shared.card_text(self.build_synthetic_model(),normalisation_kappa=1.5,prefix='FS')
        self.assertEqual(card,baseline)

    def test_required_scale_and_consistency_failures_still_stop_build(self):
        for key in ('PDF_scale_muF/ST','PDF_scale_muRmuF/ST',
                    'PDF_scale_central_closure/ST','nominal/ST','background/DY_factorisation'):
            with self.subTest(key=key):
                audit=base.AuditLog()
                audit.add('PDF_scale_factorisation/ST','2018:Down',False)
                audit.add(key,'2018:missing-or-inconsistent',False)
                with self.assertRaises(base.WorkflowError) as context:
                    self.build_synthetic_model(audit)
                self.assertIn('failed required audits',str(context.exception))
                self.assertIn(key,str(context.exception))
                self.assertIn('2018:missing-or-inconsistent',str(context.exception))

    def test_nominal_closure_and_broad_prior_semantics(self):
        model=example_model()
        text=shared.card_text(model,normalisation_kappa=1.5,prefix='FS')
        predictions=shared.nominal_predictions(text)
        columns=[(r,p) for r in model['rows'] for p in base.PROCESSES]
        for row in model['rows']:
            self.assertAlmostEqual(predictions[row['channel']],row['asimov'])
        lines=[line.split() for line in text.splitlines()]
        norms=[line for line in lines if line and 'inSituNorm' in line[0]]
        self.assertEqual(len(norms),6)
        for line in norms:
            spec=model['normalisations'][line[0]]
            for (row,p),value in zip(columns,line[2:]):
                self.assertEqual(value,'1.5' if p==spec['process'] and row['era']==spec['era'] else '-')
        self.assertNotIn('beta_',text)
        self.assertNotIn('NFModel',text)
        self.assertNotIn('bckgNorm_QCD',text)
        self.assertEqual(text.count('QCDscale_ren_ttbar lnN'),1)
        self.assertEqual(text.count('FS_QCDFitMode0_2018 param'),1)
        self.assertEqual(text.count('MCstat_ttbar_BJetOS_2018_B000'),4)
        self.assertIn('# alpha_internal_signal_yield = 25',text)

    def test_zero_dy_additive_uncertainty_has_norm_prior(self):
        model=example_model()
        row=model['rows'][0]
        row['asimov']-=row['rates']['DY']
        row['rates']['DY']=0.
        row['nuisances']['DY_LightJetStat']['DY']='-'
        text=shared.card_text(model,normalisation_kappa=1.5,prefix='FS')
        self.assertIn('FS_DYYield_bin_2018_B0000 rateParam',text)
        self.assertIn('max(0.,12*@0)',text)
        self.assertAlmostEqual(shared.nominal_predictions(text)[row['channel']],row['asimov'])

    def test_all_bins_fitted_both_asimov_hypotheses(self):
        for injection in ('0','1'):
            opt=fs.make_parser().parse_args(['--inject-r',injection])
            commands=fs.fit_commands(opt,30,Path('/tmp/model'))
            text=' '.join(sum(commands,[]))
            self.assertNotIn('mask',text)
            self.assertNotIn('--skipSBFit',text)
            fit=commands[1]
            self.assertEqual(fit[fit.index('--expectSignal')+1],str(float(injection)))
            self.assertEqual(fit[fit.index('-t')+1],'-1')
            self.assertIn('--saveShapes',fit)
            self.assertIn(f'fit_r{injection}',text)

    def test_fit_card_keeps_legacy_model_and_uses_simultaneous_builder(self):
        model=example_model()
        opt=fs.make_parser().parse_args(['--masses','30'])
        original=shared.card_text(model,normalisation_kappa=1.5,prefix='FS')
        model['card_sha256']=hashlib.sha256(original.encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'datacard.txt').write_text(original)
            with patch.object(base,'run_command') as run, patch.object(fs,'extract_summary'):
                fs.run_fit(model,path,opt)
            fit_card=path/'fit_r0/datacard.txt'
            self.assertEqual(fit_card.read_text(),'shapes * * FAKE\n'+original)
            self.assertEqual((path/'datacard.txt').read_text(),original)
            self.assertEqual(run.call_args_list[0].args[0][1],str(fit_card))
            self.assertEqual(shared.nominal_predictions(fit_card.read_text()),
                             shared.nominal_predictions(original))
            self.assertEqual(fs.fit_card_text(fit_card.read_text()),fit_card.read_text())
            (path/'datacard.txt').write_text(original+'# changed\n')
            with self.assertRaisesRegex(ValueError,'Datacard changed'):
                fs.run_fit(model,path,opt)

    def test_saved_card_fingerprint_and_configuration(self):
        opt=fs.make_parser().parse_args([])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            text=shared.card_text(example_model(),normalisation_kappa=1.5,prefix='FS')
            model=example_model()
            model['signature']=fs.signature(opt)
            model['card_sha256']=hashlib.sha256(text.encode()).hexdigest()
            (path/'datacard.txt').write_text(text)
            (path/'model.json').write_text(json.dumps(model))
            self.assertEqual(fs.load_model(path,opt)['schema_version'],fs.SCHEMA)
            (path/'datacard.txt').write_text(text+'# edited\n')
            with self.assertRaisesRegex(ValueError,'Datacard changed'):
                fs.load_model(path,opt)
            (path/'datacard.txt').write_text(text)
            opt.bin_sigma=.5
            with self.assertRaisesRegex(ValueError,'configuration differs'):
                fs.load_model(path,opt)

    def test_fit_only_missing_or_partial_model_explains_how_to_rebuild(self):
        opt=fs.make_parser().parse_args([])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            for content in (None,'model.json','datacard.txt'):
                for name in ('model.json','datacard.txt'):
                    (path/name).unlink(missing_ok=True)
                if content:
                    (path/content).write_text('{}')
                with self.assertRaisesRegex(ValueError,'--stage all --overwrite') as context:
                    fs.load_model(path,opt)
                self.assertIn('before --stage fit',str(context.exception))

    def test_target_data_blocked(self):
        reader=object.__new__(shared.BlindedReader)
        with patch.object(base.RootReader,'_file',side_effect=AssertionError('data opened')):
            with self.assertRaisesRegex(base.WorkflowError,'forbidden'):
                reader._file('/tmp/data.root')

    def test_dry_run_does_not_construct_reader(self):
        with patch.object(shared,'BlindedReader',side_effect=AssertionError('opened ROOT')):
            self.assertEqual(fs.main(['--dry-run','--masses','30','--target','full']),0)

    def test_old_sideband_norm_ranges_are_closed(self):
        model=example_model()
        model['settings'].update(norm_scope='era',norm_max=10.)
        lines=shared.card_text(model).splitlines()
        free=[line for line in lines if line.startswith('beta_')]
        self.assertTrue(free)
        self.assertTrue(all(line.endswith('[0,10]') for line in free))


@unittest.skipUnless(shutil.which('combine') and shutil.which('text2workspace.py') and importlib.util.find_spec('ROOT'),
                     'Requires CMSSW/Combine/PyROOT')
class CombineSmokeTest(unittest.TestCase):
    def test_full_spectrum_asimov(self):
        model=example_model()
        opt=fs.make_parser().parse_args(['--masses','30'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            text=shared.card_text(model,normalisation_kappa=1.5,prefix='FS')
            model['card_sha256']=hashlib.sha256(text.encode()).hexdigest()
            (path/'datacard.txt').write_text(text)
            for injection in (0.,1.):
                opt.inject_r=injection
                fit_dir=path/f'fit_r{base.mass_label(injection)}'
                fit_dir.mkdir()
                (fit_dir/'datacard.txt').write_text(fs.fit_card_text(text))
                for command in fs.fit_commands(opt,30,path):
                    subprocess.run(command,cwd=fit_dir,check=True,stdout=subprocess.DEVNULL)
                result=fs.extract_summary(model,path,opt)
                for specification in result['fits']['fit_s']['normalisations'].values():
                    self.assertAlmostEqual(specification['scale'],1.,delta=.05)
                self.assertAlmostEqual(result['fits']['fit_s']['r'],injection,delta=.05)

    def test_large_counting_model_has_finite_nominal_nll(self):
        import ROOT
        # A product of 1055 nominal Poisson(100) probabilities underflows.
        # The equivalent one-bin-shape likelihood must remain finite.
        channels=[f'b{i}' for i in range(1055)]
        text='\n'.join([
            'imax 1055', 'jmax 1', 'kmax 0', 'bin '+' '.join(channels),
            'observation '+' '.join(['100']*1055),
            'bin '+' '.join(c for c in channels for _ in range(2)),
            'process '+' '.join(['sig bkg']*1055),
            'process '+' '.join(['0 1']*1055),
            'rate '+' '.join(['1 100']*1055)])+'\n'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            (path/'card.txt').write_text(fs.fit_card_text(text))
            subprocess.run(['text2workspace.py',str(path/'card.txt'),'-o',str(path/'w.root')],
                           check=True,stdout=subprocess.DEVNULL)
            handle=ROOT.TFile.Open(str(path/'w.root'))
            try:
                workspace=handle.Get('w')
                workspace.var('r').setVal(0.)
                pdf=workspace.pdf('model_s')
                self.assertTrue(pdf.InheritsFrom('RooSimultaneous'))
                nll=pdf.createNLL(workspace.data('data_obs'))
                self.assertTrue(np.isfinite(nll.getVal()))
            finally:
                handle.Close()


if __name__=='__main__':
    unittest.main()

