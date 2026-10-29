"""ROOT-free contracts for review DAG partitioning, isolation and recovery."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import combine_review as review
import review_condor as condor
import limit_workflow

CARD = '''imax 1
jmax 1
kmax *
bin channel
observation 100
bin channel channel
process sig bkg
process 0 1
rate 25 100
scale lnN 1.1 0.9/1.2
# alpha_internal_unit = 0.00001
'''


class ReviewCondorTests(unittest.TestCase):
    def test_combine_filename_seed_rules_include_positive_toys(self):
        # Literal expectations from Combine bin/combine.cpp, not the planner.
        base = ['combine', '-M', 'MultiDimFit', '-m', '20', '-n', '.Bias']
        self.assertEqual(condor.root_name(base + ['-t', '500', '--pickToy', '17']),
                         'higgsCombine.Bias.MultiDimFit.mH20.123456.root')
        self.assertEqual(condor.root_name(base + ['-t', '-1']),
                         'higgsCombine.Bias.MultiDimFit.mH20.root')
        self.assertEqual(condor.root_name(base + ['-s', '123456']),
                         'higgsCombine.Bias.MultiDimFit.mH20.root')
        self.assertEqual(condor.root_name(base + ['-t', '-1', '--saveToys']),
                         'higgsCombine.Bias.MultiDimFit.mH20.123456.root')
        self.assertEqual(condor.root_name(base + ['-s', '7']),
                         'higgsCombine.Bias.MultiDimFit.mH20.7.root')

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.args = review.build_parser().parse_args(['20'])
        self.args.tasks = self.args.task
        self.args.card_base = str(self.root / 'cards')
        self.args.output_base = str(self.root / 'results')
        self.args.expected_json = str(self.root / 'limits.json')
        Path(self.args.expected_json).write_text(json.dumps({str(float(m)): {'exp0': 2.0} for m in (12, 20, 30, 70)}))
        for mass in (12, 20, 30, 70):
            args = copy.copy(self.args)
            args.mass = mass
            card = Path(args.card_base) / args.target / f'datacard_M-{mass}_{args.target}.txt'
            card.parent.mkdir(parents=True, exist_ok=True)
            card.write_text(CARD)
            workspace = review.workspace_path(args)
            workspace.parent.mkdir(parents=True, exist_ok=True)
            workspace.write_text('fake workspace for scheduler tests')
        self.batch = self.root / 'batch'
        self.batch.mkdir()
        self.manifest = {'batch': str(self.batch), 'batch_id': 'test', 'suite': True,
                         'jobs': [], 'output_base': str(self.root / 'published')}
        self.planner = condor.Planner(review, self.batch, self.manifest)
        self.patch_commands = patch.object(review, 'require_command', side_effect=lambda name: name)
        self.patch_commands.start()
        self.addCleanup(self.patch_commands.stop)

    def plan(self, task):
        self.planner.task(self.args, task, review.card_path(self.args))
        return self.manifest['jobs']

    def test_scan_partitions_cover_exact_grid_and_preserve_command(self):
        self.args.scan_points = 53
        self.args.scan_chunk_size = 20
        jobs = self.plan('scan')
        self.assertEqual([j['kind'] for j in jobs], ['scan_chunk'] * 3 + ['scan_collect'])
        points = []
        for job in jobs[:-1]:
            command = job['commands'][0]['cmd']
            self.assertEqual(condor.option(command, '--points'), '53')
            self.assertEqual(condor.option(command, '--expectSignal'), '0')
            self.assertEqual(condor.option(command, '-t'), '-1')
            points += list(range(int(condor.option(command, '--firstPoint')),
                                 int(condor.option(command, '--lastPoint')) + 1))
        self.assertEqual(points, list(range(53)))
        self.assertEqual(jobs[-1]['dependencies'], [0, 1, 2])
        self.assertEqual(len({j['commands'][0]['cwd'] for j in jobs[:-1]}), 3)
        original = jobs[-1]['skip_commands'][0]['cmd']
        self.assertNotIn('--firstPoint', original)

    def test_hybrid_preserves_blind_dataset_grid_and_distinct_seeds(self):
        self.args.hybrid_grid_points = 5
        jobs = self.plan('hybrid')
        self.assertEqual([j['kind'] for j in jobs], ['hybrid_asimov'] + ['hybrid_point'] * 5 + ['hybrid_collect'])
        generation = jobs[0]['commands'][0]['cmd']
        self.assertEqual(condor.option(generation, '--expectSignal'), '0')
        self.assertEqual(condor.option(generation, '-t'), '-1')
        seeds, points = [], []
        for job in jobs[1:-1]:
            cmd = job['commands'][0]['cmd']
            seeds.append(condor.option(cmd, '-s'))
            points.append(float(condor.option(cmd, '--singlePoint')))
            self.assertIn(':toys/toy_asimov', condor.option(cmd, '-D'))
            self.assertNotIn('--expectedFromGrid', cmd)
            self.assertEqual(job['dependencies'], [0])
        self.assertEqual(len(set(seeds)), 5)
        self.assertEqual(points, [0.4, 1.8, 3.2, 4.6, 6.0])

    def test_impact_nuisances_isolated_and_depend_on_initial(self):
        self.args.inject_r = 1.0
        with patch.object(limit_workflow, 'impact_workspace_parameters', return_value={'scale': (-5, 5), 'shape': (0, 10)}):
            jobs = self.plan('sb-impacts')
        self.assertEqual([j['kind'] for j in jobs], ['impact_initial', 'impact_nuisance', 'impact_nuisance', 'sb-impacts_collect'])
        for job in jobs[1:3]:
            command = job['commands'][0]['cmd']
            self.assertEqual(condor.option(command, '--expectSignal'), '1')
            self.assertEqual(condor.option(command, '--rMin'), '0')
            self.assertEqual(job['dependencies'], [0])
            self.assertEqual(len(job['copies_before']), 1)
            self.assertTrue(job['copies_before'][0][1].startswith(job['workdir']))
            self.assertNotIn('--parallel', command)
        self.assertNotEqual(jobs[1]['workdir'], jobs[2]['workdir'])
        self.assertEqual(jobs[-1]['dependencies'], [0, 1, 2])
        self.assertEqual(len(jobs[-1]['copies_before']), 3)

    def test_bias_generation_once_and_original_toy_ids_preserved(self):
        self.args.bias_toys = 7
        self.args.bias_chunk_size = 3
        jobs = self.plan('bias')
        generations = [j for j in jobs if j['kind'] == 'bias_generate']
        self.assertEqual(len(generations), 3)
        self.assertEqual([condor.option(j['commands'][0]['cmd'], '--expectSignal') for j in generations], ['0', '2', '4'])
        for gen in generations:
            chunks = [j for j in jobs if j['kind'] == 'bias_chunk' and j['dependencies'] == [gen['index']]]
            self.assertEqual(len(chunks), 3)
            commands = [c['cmd'] for j in chunks for c in j['commands']]
            self.assertEqual([int(condor.option(c, '--pickToy')) for c in commands], list(range(1, 8)))
            self.assertEqual({condor.option(c, '-t') for c in commands}, {'7'})
            self.assertEqual(len({condor.option(c, '--toysFile') for c in commands}), 1)
        final = next(j for j in jobs if j['kind'] == 'bias_collect')
        self.assertEqual(len(final['merges']), 3)
        self.assertTrue(all(len(m['sources']) == 7 for m in final['merges']))
        self.assertEqual(jobs[-1]['kind'], 'bias_audit')
        self.assertEqual(jobs[-1]['dependencies'], [final['index']])

    def test_suite_contains_r1_r0_validation_bias_and_hybrid(self):
        specs = condor.suite_configs(self.args)
        self.assertEqual([m for m, t, _ in specs if t == 'validate'], [12, 20, 30, 70])
        self.assertEqual([m for m, t, _ in specs if t == 'sb-impacts'], [12, 30, 70])
        self.assertTrue(all(o['inject_r'] == 1 for _, t, o in specs if t in ('sb-impacts', 'fitdiag')))
        self.assertEqual([m for m, t, _ in specs if t == 'hybrid'], [70])
        self.assertEqual([m for m, t, _ in specs if t == 'bias'], [20, 70])

    def test_failed_command_cannot_be_collected(self):
        index = self.planner.add('failure', self.args,
            commands=[{'cmd': ['combine', '-M', 'MultiDimFit'], 'cwd': str(self.root)}],
            files=[self.root / 'missing.root'])
        with patch.object(review, 'run_command', return_value=1):
            with self.assertRaisesRegex(RuntimeError, 'Command 0 failed'):
                condor.execute(review, self.manifest, index)
        status = json.loads((Path(self.manifest['jobs'][index]['workdir']) / 'status.json').read_text())
        self.assertFalse(status['success'])
        with self.assertRaisesRegex(RuntimeError, 'nothing collected'):
            condor.collect(self.batch, self.manifest)
        self.assertFalse(Path(self.manifest['output_base']).exists())

    def test_dependency_rerun_invalidates_descendant_and_changed_files_fail(self):
        first = self.root / 'first.root'
        second = self.root / 'second.root'
        first.write_text('first')
        second.write_text('second')
        parent = self.planner.add('parent', self.args, files=[first])
        child = self.planner.add('child', self.args, [parent], files=[second])
        def write_outputs(job):
            for filename in job['files']:
                Path(filename).write_text('completed output')
        with patch.object(review, 'task_validate', side_effect=lambda *_: write_outputs(self.manifest['jobs'][parent])):
            self.manifest['jobs'][parent]['helper'] = 'validate'
            condor.execute(review, self.manifest, parent)
        with patch.object(review, 'task_validate', side_effect=lambda *_: write_outputs(self.manifest['jobs'][child])):
            self.manifest['jobs'][child]['helper'] = 'validate'
            condor.execute(review, self.manifest, child)
        condor.complete(self.manifest, child)
        with patch.object(review, 'task_validate', side_effect=lambda *_: write_outputs(self.manifest['jobs'][parent])):
            condor.execute(review, self.manifest, parent)
        with self.assertRaisesRegex(RuntimeError, 'stale after dependency'):
            condor.complete(self.manifest, child)
        first.write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'output missing or changed'):
            condor.complete(self.manifest, parent)

    def test_collection_replaces_only_selected_mass_and_preserves_previous(self):
        source = Path(self.args.output_base) / self.args.target / 'M20'
        (source / 'result.json').write_text('new')
        target = Path(self.manifest['output_base']) / self.args.target / 'M20'
        target.mkdir(parents=True)
        (target / 'result.json').write_text('old')
        untouched = target.parent / 'M50'
        untouched.mkdir()
        (untouched / 'keep.txt').write_text('keep')
        # results must live under the batch, as in real submissions.
        self.manifest['jobs'] = []
        shutil = __import__('shutil')
        shutil.copytree(Path(self.args.output_base), self.batch / 'results')
        condor.collect(self.batch, self.manifest)
        self.assertEqual((target / 'result.json').read_text(), 'new')
        self.assertEqual((untouched / 'keep.txt').read_text(), 'keep')
        backups = list((self.batch / 'previous_outputs').rglob('result.json'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), 'old')

    def test_submit_wrapper_resources_and_resume_dag_edges(self):
        for i in range(3):
            self.planner.add('job', self.args, [i - 1] if i else [])
        self.manifest.update(cmssw_base='/shared/CMSSW', scram_arch='el9_amd64_gcc13',
                             memory=4000, max_jobs=40, requirements=None, filesystem_domain='tamsa2-ib0')
        condor.write_submission(self.batch, self.manifest, [1, 2], 'resume.dag')
        dag = (self.batch / 'resume.dag').read_text()
        self.assertNotIn('PARENT J0', dag)
        self.assertIn('PARENT J1 CHILD J2', dag)
        submit = (self.batch / 'submit.jds').read_text()
        self.assertIn('request_cpus = 1', submit)
        self.assertIn('should_transfer_files = NO', submit)
        self.assertIn('+FileSystemDomain = "tamsa2-ib0"', submit)
        wrapper = (self.batch / 'run_job.sh').read_text()
        self.assertIn('scramv1 runtime -sh', wrapper)
        with patch.object(condor.subprocess, 'run') as run:
            condor.submit(self.batch, self.manifest, 'resume.dag')
        self.assertEqual(run.call_args.args[0], ['condor_submit_dag', '-maxjobs', '40', 'resume.dag'])

    def test_active_batch_blocks_duplicate_submission(self):
        with patch.object(condor.subprocess, 'run', return_value=argparse.Namespace(stdout='123 0\n')):
            with self.assertRaisesRegex(RuntimeError, 'still has queued'):
                condor.active_batch(self.batch)

    def test_worker_runs_command_publishes_and_refuses_stale_artifact(self):
        index = self.planner.add('actual_process', self.args)
        job = self.manifest['jobs'][index]
        work = Path(job['workdir'])
        artifact = work / 'result.root'
        published = self.root / 'published.root'
        command = [sys.executable, '-c', "from pathlib import Path; Path('result.root').write_text('new fit')"]
        job.update(commands=[{'cmd': command, 'cwd': str(work)}],
                   files=[str(artifact), str(published)],
                   copies_after=[[str(artifact), str(published)]])
        condor.execute(review, self.manifest, index)
        self.assertEqual(published.read_text(), 'new fit')
        condor.complete(self.manifest, index)
        self.assertIn('result.root', (work / 'command_0.log').read_text() or str(command))
        # Exit zero without writing a new file must not reuse the old artifact.
        job['commands'][0]['cmd'] = [sys.executable, '-c', 'pass']
        with self.assertRaises(FileNotFoundError):
            condor.execute(review, self.manifest, index)
        self.assertFalse(json.loads((work / 'status.json').read_text())['success'])

    def test_local_command_cache_does_not_launch_distributed_fit_twice(self):
        command = ['combine', '-M', 'MultiDimFit']
        review._REVIEW_COMMAND_CACHE = {review.command_key(command, self.root)}
        try:
            with patch.object(review.subprocess, 'run') as run:
                self.assertEqual(review.run_command(command, self.root), 0)
                run.assert_not_called()
        finally:
            review._REVIEW_COMMAND_CACHE = set()

    def test_snapshot_tampering_is_rejected(self):
        source = self.batch / 'card.txt'
        source.write_text('old')
        self.manifest.update(schema='NIsoMuon_review_condor_v1', inputs={str(source): condor.digest(source)})
        condor.dump(self.batch / 'manifest.json', self.manifest)
        condor.load_batch(self.batch)
        source.write_text('new')
        with self.assertRaisesRegex(RuntimeError, 'Submitted input'):
            condor.load_batch(self.batch)

    def test_full_suite_preparation_freezes_inputs_without_submitting(self):
        self.args.mass = None
        self.args.review_suite = True
        self.args.condor_dry_run = True
        self.args.condor_dir = str(self.root / 'prepared')
        self.args.condor_cmssw_base = str(self.root / 'CMSSW')
        self.args.condor_scram_arch = 'el9_amd64_gcc13'
        (Path(self.args.condor_cmssw_base) / 'src').mkdir(parents=True)

        def workspace(args, card):
            path = review.workspace_path(args)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('immutable workspace ' + str(args.mass))
            return path

        with patch.object(review, 'ensure_workspace', side_effect=workspace), \
             patch.object(limit_workflow, 'impact_workspace_parameters', return_value={'scale': (-5, 5)}), \
             patch.object(condor, 'submit') as submit, \
             __import__('contextlib').redirect_stdout(__import__('io').StringIO()):
            batch = condor.prepare(review, self.args)
        submit.assert_not_called()
        _, manifest = condor.load_batch(batch)
        self.assertGreater(len(manifest['jobs']), 100)
        self.assertEqual(len([j for j in manifest['jobs'] if j['kind'] == 'bias_audit']), 2)
        self.assertTrue(all(j['config']['scan_parallel'] == 1 for j in manifest['jobs']))
        self.assertTrue(all(j['config']['impact_parallel'] == 1 for j in manifest['jobs']))
        self.assertIn(str(batch / 'expected_limits.json'), manifest['inputs'])
        self.assertIn(str(batch / 'review_condor.py'), manifest['inputs'])
        original = review.card_path(argparse.Namespace(**{**vars(self.args), 'mass': 20}))
        original.write_text('edited after submission')
        condor.load_batch(batch)  # The submitted card snapshot is independent.

    def test_legacy_upgrade_recovers_finished_toys_and_preserves_other_statuses(self):
        self.manifest['inputs'] = {}
        old_module = self.batch / 'review_condor.py'
        old_helper = self.batch / 'combine_review.py'
        old_module.write_text('# known legacy backend')
        old_helper.write_text('# known legacy helper')
        for p in (old_module, old_helper):
            self.manifest['inputs'][str(p)] = condor.digest(p)
        toy = self.root / 'toys.root'
        toy.write_text('saved toys')
        parent = self.planner.add('bias_generate', self.args, files=[toy])
        chunk = self.planner.add('bias_chunk', self.args, [parent])
        work = Path(self.manifest['jobs'][chunk]['workdir'])
        command = ['combine', '-M', 'MultiDimFit', '-m', '20', '-n', '.Bias', '-t', '500', '--pickToy', '1']
        old_output = str(work / 'higgsCombine.Bias.MultiDimFit.mH20.root')
        new_output = str(work / 'higgsCombine.Bias.MultiDimFit.mH20.123456.root')
        Path(new_output).write_text('completed fit')
        (work / 'command_0.log').write_text('fit exited zero')
        job = self.manifest['jobs'][chunk]
        job.update(commands=[{'cmd': command, 'cwd': str(work)}], files=[old_output])
        final = self.planner.add('bias_collect', self.args, [chunk],
                                 merges=[{'sources': [old_output], 'destination': str(self.root / 'merged.root')}])
        parent_job = self.manifest['jobs'][parent]
        parent_status = {'success': True, 'batch_id': 'test', 'completion_id': 'generation',
                         'job_sha256': condor.hashlib.sha256(json.dumps(parent_job, sort_keys=True).encode()).hexdigest(),
                         'outputs': {str(toy): condor.digest(toy)}, 'dependencies': {}}
        parent_path = Path(parent_job['workdir']) / 'status.json'
        condor.dump(parent_path, parent_status)
        parent_before = parent_path.read_bytes()
        status = {'success': False, 'batch_id': 'test', 'completion_id': 'fit-completion',
                  'job_sha256': condor.hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
                  'dependencies': {str(parent): 'generation'},
                  'error': 'RuntimeError: Required output missing or empty: ' + old_output}
        condor.dump(work / 'status.json', status)
        condor.dump(self.batch / 'manifest.json', self.manifest)
        original_command = copy.deepcopy(job['commands'])
        with patch.object(condor, 'LEGACY_BIAS_BACKEND_SHA256', condor.digest(old_module)), \
             patch.object(condor, 'LEGACY_BIAS_HELPER_SHA256', condor.digest(old_helper)):
            upgraded = condor.upgrade_backend(review, self.batch, self.manifest)
        self.assertEqual(upgraded['jobs'][chunk]['files'], [new_output])
        self.assertEqual(upgraded['jobs'][chunk]['commands'], original_command)
        self.assertEqual(upgraded['jobs'][final]['merges'][0]['sources'], [new_output])
        self.assertEqual(parent_path.read_bytes(), parent_before)
        recovered = condor.complete(upgraded, chunk)
        self.assertTrue(recovered['recovered_output_names'])
        self.assertEqual(recovered['completion_id'], 'fit-completion')
        archive = Path(upgraded['backend_upgrade']['archive'])
        self.assertEqual((archive / 'review_condor.py').read_text(), '# known legacy backend')
        self.assertTrue((archive / 'status_job_1.json').is_file())


if __name__ == '__main__':
    unittest.main()
