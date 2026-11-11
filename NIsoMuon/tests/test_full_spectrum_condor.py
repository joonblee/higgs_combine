"""Distributed preparation must produce the same joint card as local execution."""
import argparse
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import full_spectrum_condor as batch
import full_spectrum_fit as fs
import limit_workflow as base
import limit_workflow_sideband_fit as shared


def raw_part(reader, args, opt, mass, files, era, windows, full_spectrum=False):
    rows = []
    for window in windows:
        first = window['region'] == 'B0000'
        # Some preparation shards contain no signal: only the assembled model
        # may require positive total signal or apply the 25-event scaling.
        rates = dict(sig=0. if first else 17., QCD=100. if first else 10.,
                     DY=10. if first else 50., tt=20., ST=2., Others=1.)
        rows.append(dict(window, era=era, channel=f'bin_{era}_{window["region"]}',
            rates=rates, nuisances={'tt_stat':{'tt':'0.9/1.1'},
                'DY_LightJetStat':{'DY':'0.9/1.1'}, 'QCD_norm':{'QCD':'0.8/1.2'}},
            qcd_modes=[.1*rates['QCD']], qcd_shape=.2*rates['QCD'], dy_stat=1.,
            dy_nf=.1, warnings=[]))
    return dict(era=era, rows=rows, transfer_ratios=[[.1,.2] for _ in rows],
                input_audits=[dict(era=era,region=w['region'],requirements=1,
                                  diagnostic_failures=[]) for w in windows])


class FakeReader:
    def __init__(self):
        self.fingerprints = {}
    def check_inputs(self):
        pass
    def close(self):
        pass


class CondorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.eras = base.target_years('Run2')
        self.windows = {e:[dict(region='B0000',low=11.,high=30.),
                           dict(region='B0001',low=30.,high=80.)] for e in self.eras}
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(batch, 'InputReader', FakeReader))
        self.stack.enter_context(patch.object(base, 'condor_environment',
            return_value=(self.root/'cmssw', 'test_arch')))
        self.stack.enter_context(patch.object(shared, 'build_bin_rows', side_effect=raw_part))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def make_batch(self, name='batch', grouping=1, stage='all', extra=()):
        opt = fs.make_parser().parse_args(['--backend','condor','--condor-dry-run',
            '--masses','30','--target','Run2','--stage',stage,
            '--condor-dir',str(self.root/name),'--bins-per-job',str(grouping), *extra])
        args = fs.configuration(opt)
        def inputs(reader, args, opt):
            args.resolution_coefficients = {e:[0.,.01] for e in self.eras}
            return {30.:{e:'signal.root' for e in self.eras}}, self.windows
        with patch.object(fs, 'prepare_inputs', side_effect=inputs):
            directory = batch.submit(opt, args)
        return opt, args, batch.load_manifest(directory)

    def workers(self, payload, kind):
        for job in reversed(payload['jobs']):
            if job['kind'] == kind:
                batch.run_worker(Path(payload['batch'])/'manifest.json', job['index'])

    def fake_fit(self, model, directory, opt):
        self.assertAlmostEqual(sum(r['rates']['sig'] for r in model['rows']),25.)
        self.assertEqual(len(model['rows']),8)
        directory = directory/f'fit_r{base.mass_label(opt.inject_r)}'
        directory.mkdir()
        for name in ('workspace.root','fitDiagnostics.FullSpectrum.root','fit_summary.json'):
            (directory/name).write_text('synthetic fit output')

    def test_raw_shards_assemble_to_identical_local_card_and_one_joint_fit(self):
        opt,args,payload = self.make_batch()
        jobs = payload['jobs']
        self.assertEqual([j['kind'] for j in jobs].count('prepare'),8)
        self.assertEqual(jobs[-2]['dependencies'],list(range(8)))
        self.assertEqual(jobs[-1]['dependencies'],[8])
        with self.assertRaisesRegex(ValueError,'Dependency job'):
            batch.run_worker(Path(payload['batch']),8)
        self.workers(payload,'prepare')
        # Worker completions may arrive in any order.
        self.workers(payload,'assemble')
        saved_opt,_ = batch.worker_options(payload)
        path = fs.model_directory(saved_opt,30.)
        saved = fs.load_model(path,saved_opt)
        self.assertAlmostEqual(sum(r['rates']['sig'] for r in saved['rows']),25.)
        with patch.object(base,'prepare_resolution_coefficients'):
            local = fs.build_model(FakeReader(),args,opt,30.,
                                  {e:'signal.root' for e in self.eras},self.windows)
        self.assertEqual((path/'datacard.txt').read_text(),
                         shared.card_text(local,normalisation_kappa=1.5,prefix='FS'))
        with patch.object(fs,'run_fit',side_effect=self.fake_fit) as fit:
            self.workers(payload,'fit')
            fit.assert_called_once()
        for job in jobs:
            self.assertIsNotNone(batch.completed(payload,job))
        opt.stage='collect'
        self.assertEqual(batch.manage(opt),0)

    def test_grouping_preserves_card_and_configuration_signature(self):
        cards=[]
        for name,grouping in (('one',1),('two',2)):
            opt,args,payload=self.make_batch(name,grouping,stage='cards')
            self.workers(payload,'prepare')
            self.workers(payload,'assemble')
            saved_opt,_=batch.worker_options(payload)
            cards.append((fs.model_directory(saved_opt,30.)/'datacard.txt').read_text())
        self.assertEqual(cards[0],cards[1])
        local=fs.make_parser().parse_args(['--target','Run2','--masses','30'])
        self.assertEqual(fs.signature(local),fs.signature(opt))

    def test_failed_worker_is_recorded_and_can_be_retried(self):
        opt,args,payload=self.make_batch()
        manifest=Path(payload['batch'])/'manifest.json'
        with patch.object(shared,'build_bin_rows',side_effect=ValueError('missing template')):
            with self.assertRaisesRegex(ValueError,'missing template'):
                batch.run_worker(manifest,0)
        status=json.loads((batch.workdir(payload,payload['jobs'][0])/'status.json').read_text())
        self.assertEqual(status['state'],'failed')
        self.assertIn('missing template',status['error'])
        self.assertIsNone(batch.completed(payload,payload['jobs'][0]))
        batch.run_worker(manifest,0)
        self.assertIsNotNone(batch.completed(payload,payload['jobs'][0]))

    def test_tampered_shard_invalidates_descendants_and_resume_keeps_other_bins(self):
        opt,args,payload=self.make_batch()
        self.workers(payload,'prepare')
        self.workers(payload,'assemble')
        path=batch.workdir(payload,payload['jobs'][0])/'part.json'
        path.write_text(path.read_text()+'\n')
        self.assertIsNone(batch.completed(payload,payload['jobs'][-2]))
        opt.stage='resume'
        batch.manage(opt)
        dag=next(Path(payload['batch']).glob('resume_*.dag')).read_text()
        self.assertIn('JOB J0 submit.jds',dag)
        self.assertNotIn('JOB J1 submit.jds',dag)
        self.assertIn('PARENT J0 CHILD J8',dag)
        self.assertIn('PARENT J8 CHILD J9',dag)
        self.assertNotIn('PARENT J1',dag)
        with patch.object(batch.subprocess,'run',return_value=SimpleNamespace(stdout='12345\n')):
            opt.condor_dry_run=False
            with patch.object(base,'require_command'):
                with self.assertRaisesRegex(ValueError,'still has queued jobs'):
                    batch.manage(opt)

    def test_dependency_rerun_invalidates_a_previously_successful_merge(self):
        opt,args,payload=self.make_batch()
        self.workers(payload,'prepare')
        self.workers(payload,'assemble')
        batch.run_worker(Path(payload['batch']),0)
        self.assertIsNone(batch.completed(payload,payload['jobs'][-2]))

    def test_missing_or_duplicate_bins_and_changed_inputs_rejected(self):
        opt,args,payload=self.make_batch()
        self.workers(payload,'prepare')
        saved_opt,saved_args=batch.worker_options(payload)
        job=payload['jobs'][-2]
        changed=copy.deepcopy(job)
        changed['dependencies']=changed['dependencies'][1:]
        with self.assertRaisesRegex(ValueError,'bin coverage'):
            batch.assemble(payload,changed,saved_opt,saved_args)
        changed['dependencies']=job['dependencies']+[0]
        with self.assertRaisesRegex(ValueError,'bin coverage'):
            batch.assemble(payload,changed,saved_opt,saved_args)
        # A transfer that is constant inside each shard but changes across
        # shards must still fail the global mass-independence validation.
        path=batch.workdir(payload,payload['jobs'][0])/'part.json'
        part=json.loads(path.read_text())
        part['part']['transfer_ratios']=[[.3,.2]]
        batch.write_json(path,part)
        with self.assertRaisesRegex(ValueError,'QCD transfer changes shape'):
            batch.assemble(payload,job,saved_opt,saved_args)
        source=self.root/'input.root'
        source.write_text('initial')
        fingerprint={str(source):batch.stamp(source)}
        source.write_text('changed input')
        with self.assertRaisesRegex(ValueError,'Input changed'):
            batch.check_inputs(fingerprint)

    def test_snapshot_and_manifest_are_checked_and_wrapper_is_valid_bash(self):
        opt,args,payload=self.make_batch()
        directory=Path(payload['batch'])
        subprocess.run(['bash','-n',str(directory/'run_job.sh')],check=True)
        report=subprocess.run([sys.executable,str(directory/'source/full_spectrum_fit.py'),
            '--stage','status','--condor-dir',str(directory)],check=True,
            text=True,capture_output=True)
        self.assertIn('[BINS] 2018: 0/2 complete',report.stdout)
        self.assertIn('should_transfer_files = NO',(directory/'submit.jds').read_text())
        self.assertIn('RETRY J0 2',(directory/'workflow.dag').read_text())
        self.assertNotIn('-maxjobs',batch.submission(payload,directory/'workflow.dag'))
        capped=copy.deepcopy(payload)
        capped['options']['condor_maxjobs']=100
        self.assertIn('-maxjobs',batch.submission(capped,directory/'workflow.dag'))
        with self.assertRaisesRegex(ValueError,'already exists'):
            batch.submit(opt,args)
        with batch.job_lock(batch.workdir(payload,payload['jobs'][0])):
            with self.assertRaisesRegex(ValueError,'already active'):
                batch.run_worker(directory,0)
        path=directory/'source/full_spectrum_fit.py'
        path.write_text(path.read_text()+'# edited\n')
        with self.assertRaisesRegex(ValueError,'snapshot changed'):
            batch.load_manifest(directory)
        path=directory/'manifest.json'
        value=json.loads(path.read_text())
        value['jobs'][0]['mass']=70.
        batch.write_json(path,value)
        with self.assertRaisesRegex(ValueError,'manifest changed'):
            batch.load_manifest(directory)

    def test_fit_only_batch_reuses_card_without_template_preparation(self):
        opt,args,payload=self.make_batch(stage='cards')
        self.workers(payload,'prepare')
        self.workers(payload,'assemble')
        results=str(Path(payload['batch'])/'results')
        with patch.object(fs,'prepare_inputs',side_effect=AssertionError('templates read')):
            opt,args,new=self.make_batch('injection',stage='fit',extra=(
                '--output-base',results,'--inject-r','1'))
        self.assertEqual([j['kind'] for j in new['jobs']],['fit'])
        with patch.object(fs,'run_fit',side_effect=self.fake_fit):
            self.workers(new,'fit')
        self.assertIsNotNone(batch.completed(new,new['jobs'][0]))

    def test_bin_job_count_and_invalid_resources(self):
        opt,args,payload=self.make_batch()
        for name,value in (('bins_per_job',0),('condor_memory',0),
                           ('condor_maxjobs',-1),('condor_retries',-1)):
            other=copy.copy(opt)
            setattr(other,name,value)
            with self.assertRaises(ValueError):
                batch.validate_options(other)


class ReaderTests(unittest.TestCase):
    def test_qcd_validation_cached_across_bins_but_integrals_recomputed(self):
        reader=object.__new__(shared.BlindedReader)
        reader.qcd_input_cache={}
        inputs=object()
        with patch.object(shared.BlindedReader,'_file',return_value=object()), \
             patch.object(base,'read_qcd_stat_inputs',return_value=inputs) as read, \
             patch.object(base,'qcd_root_window_statistics',side_effect=[{'sigma':1.},{'sigma':2.}]) as evaluate:
            a=reader.qcd_statistics('ss.root','hist',11.,20.,'2018',10.)
            b=reader.qcd_statistics('ss.root','hist',20.,30.,'2018',4.)
        self.assertEqual(read.call_count,1)
        self.assertEqual(evaluate.call_count,2)
        self.assertEqual((a['sigma'],b['sigma']),(1.,2.))

    def test_batch_reader_blocks_target_data_before_filesystem_access(self):
        reader=object.__new__(batch.InputReader)
        reader.fingerprints={}
        with patch.object(batch,'stamp',side_effect=AssertionError('target accessed')):
            with self.assertRaisesRegex(base.WorkflowError,'forbidden'):
                reader._file('/analysis/data.root')


if __name__=='__main__':
    unittest.main()
