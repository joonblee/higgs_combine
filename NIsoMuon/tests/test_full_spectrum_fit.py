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
            for factor in (.5,1.,2.,4.):
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

    def test_fast_preparation_has_fewer_bins_and_separate_outputs(self):
        normal=fs.make_parser().parse_args(['--target','2018','--masses','30'])
        fast=fs.make_parser().parse_args(['--target','2018','--masses','30','--fast-mode'])
        edges=np.round(np.arange(11.,80.001,.02),8)
        windows=[]
        with ExitStack() as stack:
            stack.enter_context(patch.object(base,'scan_signal_files',return_value={30.:{'2018':'signal.root'}}))
            stack.enter_context(patch.object(base,'prepare_resolution_coefficients'))
            stack.enter_context(patch.object(shared,'common_edges',return_value=edges))
            stack.enter_context(patch.object(base,'relative_resolution',return_value=.01))
            for opt in (normal,fast):
                _,w=fs.prepare_inputs(None,SimpleNamespace(),opt)
                windows.append(w['2018'])
        self.assertLess(len(windows[1]),.3*len(windows[0]))
        for bins in windows:
            self.assertEqual(bins[0]['low'],11.)
            self.assertEqual(bins[-1]['high'],80.)
            self.assertTrue(all(a['high']==b['low'] for a,b in zip(bins,bins[1:])))
        self.assertNotEqual(fs.model_directory(normal,30),fs.model_directory(fast,30))
        self.assertIn('binsigma4_lnN1p5_fast',str(fs.model_directory(fast,30)))
        self.assertNotEqual(fs.signature(normal),fs.signature(fast))
        legacy=copy.copy(normal)
        del legacy.fast_mode
        self.assertEqual(fs.signature(normal),fs.signature(legacy))
        self.assertEqual(fs.fit_commands(normal,30,Path('/tmp/test')),
                         fs.fit_commands(fast,30,Path('/tmp/test')))

    def test_no_domain_truncation(self):
        for edges in ([11.1,20.,80.],[11.,20.,79.9]):
            with self.assertRaisesRegex(ValueError,'both 11 and 80'):
                fs.resolution_bins(edges,lambda m:1.)

    def test_window_geometry_covers_native_bins_once_for_every_era_and_mass(self):
        coeffs=json.loads((Path(base.__file__).parent/'sigfit_inputs/resolution_coefficients.json').read_text())
        grid=np.round(np.arange(11.,80.001,.02),8)
        counts=np.arange(len(grid)-1,dtype=float)+1
        for era,ab in coeffs.items():
            for mass in (12.,30.,70.):
                sigma=mass*(ab['a']*mass+ab['b'])
                low,high=max(11.,mass-5*sigma),min(80.,mass+5*sigma)
                for refine in (1,2):
                    bins=fs.window_bins(grid,low,high,refine=refine)
                    self.assertEqual(bins[0]['low'],11.)
                    self.assertEqual(bins[-1]['high'],80.)
                    self.assertTrue(all(a['high']==b['low'] for a,b in zip(bins,bins[1:])))
                    sr=[w for w in bins if w['region']=='SR']
                    self.assertEqual(len(sr),1)
                    self.assertLessEqual(sr[0]['low'],low+1e-8)
                    self.assertGreaterEqual(sr[0]['high'],high-1e-8)
                    integrated=sum(counts[np.searchsorted(grid,w['low']):np.searchsorted(grid,w['high'])].sum() for w in bins)
                    self.assertEqual(integrated,counts.sum())
                    self.assertTrue(all(w['low'] in grid and w['high'] in grid for w in bins))

    def test_window_preflight_builds_distinct_plans_for_each_mass(self):
        opt=fs.make_parser().parse_args(['--fit-layout','window','--target','2018','--masses','12,30,70'])
        args=fs.configuration(opt)
        args.resolution_coefficients={'2018':[0.,.01]}
        signals={m:{'2018':f'signal{m}.root'} for m in opt.masses}
        grid=np.round(np.arange(11.,80.001,.02),8)
        with ExitStack() as stack:
            stack.enter_context(patch.object(base,'scan_signal_files',return_value=signals))
            stack.enter_context(patch.object(base,'prepare_resolution_coefficients'))
            stack.enter_context(patch.object(shared,'common_edges',return_value=grid))
            found,plans=fs.prepare_inputs(None,args,opt)
        self.assertEqual(found,signals)
        windows=[]
        for mass in opt.masses:
            bins=fs.windows_for_mass(plans,mass,opt)['2018']
            sr=next(w for w in bins if w['region']=='SR')
            self.assertAlmostEqual(sr['low'],max(11.,mass*.95))
            self.assertAlmostEqual(sr['high'],min(80.,mass*1.05))
            windows.append((sr['low'],sr['high']))
        self.assertEqual(len(set(windows)),3)

    def test_small_sideband_fragments_merge_away_from_sr(self):
        grid=np.arange(11.,80.001,.1)
        bins=fs.window_bins(grid,30.1,39.9)
        sr=next(w for w in bins if w['region']=='SR')
        self.assertEqual((sr['low'],sr['high']),(30.1,39.9))
        self.assertEqual(next(w for w in bins if w['high']==30.1)['low'],28.)
        self.assertEqual(next(w for w in bins if w['low']==39.9)['high'],50.)
        self.assertFalse(any(w['region'].startswith('L') for w in fs.window_bins(grid,11.,14.)))
        self.assertFalse(any(w['region'].startswith('R') for w in fs.window_bins(grid,76.,80.)))

    def test_nested_sideband_presets_share_sr_and_have_exact_reference_ratios(self):
        presets=fs.SIDEBAND_PRESETS
        self.assertEqual({k:len(v)-1 for k,v in presets.items()},dict(central=20,fine=40,coarse=10))
        self.assertTrue(set(presets['coarse']) < set(presets['central']) < set(presets['fine']))
        self.assertTrue(all(b-a==1. for a,b in zip(presets['central'],presets['central'][1:]) if b<=20.))
        grid=np.round(np.arange(11.,80.001,.02),8)
        paths=set()
        signatures=[]
        for name in ('central','fine','coarse'):
            opt=fs.make_parser().parse_args(['--fit-layout','window','--sideband-binning',name])
            fs.configuration(opt)
            paths.add(fs.model_directory(opt,30))
            signatures.append(fs.signature(opt))
            for low,high in ((11.,13.),(28.3,31.7),(76.,80.)):
                bins=fs.window_bins(grid,low,high,fs.sideband_reference(opt))
                self.assertEqual([(w['low'],w['high']) for w in bins if w['region']=='SR'],[(low,high)])
                self.assertEqual(sum(w['high']-w['low'] for w in bins),69.)
        self.assertEqual(len(paths),3)
        self.assertEqual(len({json.dumps(s,sort_keys=True) for s in signatures}),3)
        # A custom-edge saved model predating named presets stays compatible.
        old=copy.copy(opt)
        old.sideband_edges=[11.,15.,20.,25.,30.,40.,50.,60.,80.]
        old.sideband_binning='central'
        legacy=copy.copy(old)
        del legacy.sideband_binning
        self.assertEqual(fs.signature(old),fs.signature(legacy))
        self.assertEqual(fs.model_directory(old,30),fs.model_directory(legacy,30))
        with self.assertRaisesRegex(ValueError,'either'):
            fs.configuration(fs.make_parser().parse_args(['--fit-layout','window',
                '--sideband-binning','fine','--sideband-edges','11,20,80']))

    def test_window_cli_is_isolated_and_rejects_invalid_or_ignored_settings(self):
        parser=fs.make_parser()
        normal=parser.parse_args([])
        window=parser.parse_args(['--fit-layout','window'])
        refined=parser.parse_args(['--fit-layout','window','--sideband-refine','2'])
        fs.configuration(window)
        self.assertNotEqual(fs.model_directory(normal,30),fs.model_directory(window,30))
        self.assertNotEqual(fs.model_directory(window,30),fs.model_directory(refined,30))
        legacy=copy.copy(normal)
        for name in ('fit_layout','sr_nsigma','sideband_edges','sideband_refine','sideband_fragment_fraction'):
            delattr(legacy,name)
        self.assertEqual(fs.signature(legacy),fs.signature(normal))
        for extra in (['--fast-mode'],['--bin-sigma','2'],['--sideband-refine','0'],
                      ['--sideband-edges','11,20,20,80'],['--sideband-edges','12,20,80'],
                      ['--sideband-fragment-fraction','nan']):
            with self.assertRaises(ValueError):
                fs.configuration(parser.parse_args(['--fit-layout','window',*extra]))

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

    def test_independent_fits_keep_full_diagnostics_and_share_workspace(self):
        opt=fs.make_parser().parse_args(['--masses','30'])
        self.assertEqual(opt.condor_memory,16000)
        commands={k:fs.fit_commands(opt,30,Path('/tmp/model'),k) for k in ('fit_b','fit_s')}
        self.assertEqual(commands['fit_b'][0],commands['fit_s'][0])
        for kind,skip in (('fit_b','--skipSBFit'),('fit_s','--skipBOnlyFit')):
            command=commands[kind][1]
            self.assertIn(skip,command)
            self.assertIn('--saveWithUncertainties',command)
            self.assertEqual(command[command.index('--robustFit')+1],'1')
            self.assertEqual(command[command.index('-t')+1],'-1')
            self.assertNotIn('--noErrors',command)

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
    def test_split_fits_match_sequential_model(self):
        model=example_model()
        opt=fs.make_parser().parse_args(['--masses','30','--inject-r','1'])
        text=shared.card_text(model,normalisation_kappa=1.5,prefix='FS')
        model['card_sha256']=hashlib.sha256(text.encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            outputs=[]
            for mode in ('sequential','split'):
                path=Path(directory)/mode
                path.mkdir()
                (path/'datacard.txt').write_text(text)
                if mode=='sequential':
                    fs.run_fit(model,path,opt)
                    result=json.loads((path/'fit_r1/fit_summary.json').read_text())
                else:
                    fs.prepare_workspace(model,path,opt)
                    fs.run_fit_component(model,path,opt,'fit_s')
                    fs.run_fit_component(model,path,opt,'fit_b')
                    result=fs.combine_fit_summaries(model,path,opt)
                outputs.append(result)
            for kind in ('fit_b','fit_s'):
                left,right=[r['fits'][kind] for r in outputs]
                self.assertAlmostEqual(left['r'],right['r'],delta=.05)
                for name in model['normalisations']:
                    self.assertAlmostEqual(left['normalisations'][name]['scale'],
                                           right['normalisations'][name]['scale'],delta=.05)

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

