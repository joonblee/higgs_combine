"""Saved-toy provenance, chunk selection and paired comparison accounting."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import full_spectrum_toy_refits as refit


class SavedToyRefits(unittest.TestCase):
    def test_fresh_retry_uses_standard_tolerance_with_recorded_options(self):
        payload=self.prepare('failed')
        options=payload['fit_options']
        self.assertEqual(options,dict(strategy=2,tolerance=.1,max_calls=None))
        task=payload['tasks'][0]
        command=refit.fit_command(task['original_task'],self.batch/task['workspace'],
                                  self.batch/task['dataset'],options)
        self.assertEqual(command[command.index('--cminDefaultMinimizerTolerance')+1],'0.1')
        self.assertEqual(command[command.index('--cminFallbackAlgo')+1],'Minuit2,Migrad,1:0.1')
        self.assertNotIn('--X-rtd',command)
        self.assertEqual(task['original_task']['seed'],123)
        self.assertEqual((self.batch/task['dataset']).read_text(),'saved dataset 0')

    def test_legacy_manifest_without_options_preserves_old_retry_numerics(self):
        payload=self.prepare('failed')
        del payload['fit_options']
        payload['manifest_sha256']=refit.toys.manifest_digest(payload)
        (self.batch/'manifest.json').write_text(json.dumps(payload))
        commands=[]
        def run(command,cwd,**kwargs):
            commands.append(command);(cwd/'fitDiagnostics.Bias.root').write_text('fit')
        with patch.object(refit.base,'run_command',side_effect=run), \
             patch.object(refit,'read_fits',return_value=[dict(r=0.,status=0)]*3):
            refit.worker(self.batch/'manifest.json',0,self.batch/'jobs/job_00000/result')
        self.assertEqual(commands[0][commands[0].index('--cminDefaultMinimizerTolerance')+1],'0.0001')

    def test_workspace_hash_is_shared_across_saved_toy_chunks(self):
        original=Path.open;reads=[]
        def opened(path,*args,**kwargs):
            if path.name=='fit_abc.root' and args and args[0]=='rb':
                reads.append(str(path))
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',new=opened):
            self.prepare()
        self.assertEqual(len(reads),2) # original workspace + copied snapshot
        self.assertEqual(len(set(reads)),2)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='saved_refit_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root/'original'
        (self.source/'input').mkdir(parents=True)
        self.batch = self.root/'comparison'
        workspace = self.source/'input/fit_abc.root'
        workspace.write_text('original workspace bytes')
        tasks = []
        for index in range(2):
            task = dict(index=index, mass=30., injected_r=0., reference_r=1., count=3,
                seed=123+index, r_min=-5., r_max=100., toy_errors=False,
                truth_sha256='nominal', truth_label='nominal', fit_card_sha256='card',
                fit_workspace='input/fit_abc.root', truth_workspace='input/fit_abc.root',
                inputs={'input/fit_abc.root':refit.toys.digest(workspace)})
            tasks.append(task)
            directory = self.source/'jobs'/f'job_{index:05d}'/'result'
            directory.mkdir(parents=True)
            entries = [dict(r=-.1, status=0), dict(r=.2, status=0), dict(r=.3, status=4 if index==0 else 0)]
            (directory/'result.json').write_text(json.dumps(dict(task=task, entries=entries)))
            (directory/'higgsCombine.Truth.GenerateOnly.mH30.123.root').write_text('saved dataset '+str(index))
            with tarfile.open(directory.parent/'result.tar.gz', 'w:gz') as archive:
                archive.add(directory, arcname='result')
            import shutil
            shutil.rmtree(directory)
        payload = dict(tasks=tasks, source_hashes={})
        payload['manifest_sha256'] = refit.toys.manifest_digest(payload)
        (self.source/'manifest.json').write_text(json.dumps(payload))
        (self.root/'CMSSW/src').mkdir(parents=True)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def prepare(self, selection='all'):
        refit.main(['--stage','refit','--source-batch',str(self.source), '--batch', str(self.batch),
            '--selection',selection,'--dry-run','--condor-cmssw-base',str(self.root/'CMSSW'),
            '--condor-scram-arch','el9_amd64_gcc12'])
        return refit.load(self.batch)

    def test_extracts_original_datasets_and_never_generates(self):
        payload = self.prepare()
        self.assertEqual(len(payload['tasks']), 2)
        for task in payload['tasks']:
            old = task['original_task']
            self.assertEqual((self.batch/task['dataset']).read_text(), 'saved dataset '+str(old['index']))
            command = refit.fit_command(old, self.batch/task['workspace'], self.batch/task['dataset'])
            self.assertEqual(command[command.index('-M')+1], 'FitDiagnostics')
            self.assertEqual(command[command.index('-t')+1], '3')
            self.assertEqual(command[command.index('-s')+1], str(old['seed']))
            self.assertEqual(command[command.index('--rMin')+1], '-5.0')
            self.assertEqual(command[command.index('--cminDefaultMinimizerStrategy')+1], '2')
            self.assertIn('--noErrors', command)
            self.assertNotIn('GenerateOnly', command)
        subprocess.run(['bash','-n',str(self.batch/'run_job.sh')], check=True)

    def test_failed_selection_keeps_successful_controls_in_same_chunk(self):
        payload = self.prepare('failed')
        self.assertEqual(len(payload['tasks']), 1)
        self.assertEqual(payload['tasks'][0]['original_task']['index'], 0)
        self.assertEqual(payload['tasks'][0]['original_task']['count'], 3)
        _, _, failures = refit.original(self.source)
        self.assertEqual([(x['chunk'], x['entry']) for x in failures], [(0,3)])

    def test_worker_and_collect_keep_new_failures_and_do_not_overwrite_originals(self):
        payload = self.prepare()
        before = (self.source/'jobs/job_00000/result.tar.gz').read_bytes()
        comparisons = [[dict(r=-.11,status=0),dict(r=.19,status=0),dict(r=.25,status=0)],
                       [dict(r=-.1,status=4),dict(r=.21,status=0),dict(r=.31,status=0)]]
        commands = []
        def run(command, cwd, **kwargs):
            commands.append(command)
            (cwd/'fitDiagnostics.Bias.root').write_text('synthetic ROOT')
        with patch.object(refit.base,'run_command',side_effect=run), patch.object(refit,'read_fits',side_effect=comparisons):
            for task in payload['tasks']:
                refit.worker(self.batch/'manifest.json',task['index'],
                    self.batch/'jobs'/f'job_{task["index"]:05d}'/'result')
        refit.collect(self.batch)
        summary = json.loads((self.batch/'toy_refit_summary.json').read_text())
        self.assertEqual(summary['recovered_failed_fits'],1)
        self.assertEqual(summary['newly_failed_fits'],1)
        self.assertEqual(summary['pending_chunks'],[])
        self.assertEqual(summary['refitted_ensembles'][0]['failed'],1)
        self.assertEqual(summary['refit_status_counts'],{'0':5,'4':1})
        self.assertEqual(len(summary['failed_refit_entries']),1)
        failure=summary['failed_refit_entries'][0]
        self.assertTrue(failure['newly_failed'])
        self.assertEqual((failure['original_chunk'],failure['entry'],failure['seed']),(1,1,124))
        self.assertEqual(commands[0][commands[0].index('--cminDefaultMinimizerTolerance')+1],'0.1')
        self.assertEqual(len(summary['paired_entries']),4)
        self.assertTrue(all(c[c.index('-M')+1]=='FitDiagnostics' for c in commands))
        self.assertEqual(before,(self.source/'jobs/job_00000/result.tar.gz').read_bytes())

    def test_incomplete_comparison_does_not_present_a_mixed_bias_estimate(self):
        self.prepare()
        refit.collect(self.batch)
        data = json.loads((self.batch/'toy_refit_summary.json').read_text())
        self.assertEqual(data['pending_chunks'],[0,1])
        self.assertIsNone(data['refitted_ensembles'])

    def test_modified_saved_workspace_is_rejected(self):
        (self.source/'input/fit_abc.root').write_text('changed')
        with self.assertRaisesRegex(ValueError,'workspace changed'):
            self.prepare()

    def test_modified_manifest_is_rejected(self):
        self.prepare()
        path = self.batch/'manifest.json'
        payload = json.loads(path.read_text())
        payload['tasks'][0]['original_task']['seed'] = 999
        path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(ValueError,'modified'):
            refit.load(self.batch)


if __name__=='__main__':
    unittest.main()
