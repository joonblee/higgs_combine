"""Per-toy Condor repair contracts with synthetic inputs and mocked fits."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import combine_review as review
import review_condor as backend
import review_bias_parallel as parallel
import repair_bias_intervals as repair


class ParallelBiasTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.batch = self.root / 'main'
        self.batch.mkdir()
        script_dir = Path(review.__file__).parent
        for name in parallel.SOURCES:
            shutil.copy2(script_dir / name, self.batch / name)
        self.args = review.build_parser().parse_args(['70'])
        self.args.tasks = ['bias']
        self.args.output_base = str(self.batch / 'results')
        self.source = review.outdir(self.args)
        (self.source / 'bias').mkdir(parents=True)
        (self.source / 'bias_checks').mkdir()
        self.workspace = review.workspace_path(self.args)
        self.workspace.write_text('workspace')
        self.audit = self.root / 'audit.json'
        fingerprints, ensembles, points = {}, [], []
        for i, injection in enumerate((0., 1., 2.)):
            tag = f'r{i}'
            fit = self.source / 'bias' / f'higgsCombine.ReviewBiasFit_Run2Run3_M70_{tag}.MultiDimFit.mH70.123456.root'
            fit.write_text('original toy fits')
            fingerprints[str(fit)] = backend.digest(fit)
            dataset = self.source / 'bias' / f'higgsCombine.ReviewBiasGen_Run2Run3_M70_{tag}.GenerateOnly.mH70.12345.root'
            dataset.write_text('saved toy datasets')
            rows = []
            for toy in (1, 2):
                interval = (-2., .3, .8) if i == 1 and toy == 1 else (-.2, .3, .8)
                rows.extend({'toy': toy, 'quantile': q, 'r': value}
                            for (_, q), value in zip(repair.audit.QUANTILES, interval))
            ensemble = repair.audit.audit_rows(rows, 2, (-2., 100.))
            ensemble.update(tag=tag, injected_r=injection, fit_file=str(fit), r_range=[-2., 100.])
            ensembles.append(ensemble)
            points.append({'injected_r': injection, 'n_toys_requested': 2, 'fit_file': str(fit)})
        summary = self.source / 'bias' / 'bias_summary_Run2Run3_M70.json'
        summary.write_text(json.dumps(points))
        fingerprints[str(summary)] = backend.digest(summary)
        self.audit.write_text(json.dumps({'errors': [], 'ensembles': ensembles, 'input_sha256': fingerprints}))
        self.main = {'batch': str(self.batch), 'batch_id': 'test', 'cmssw_base': '/shared/CMSSW',
                     'scram_arch': 'arch', 'memory': 4000, 'max_jobs': 40}
        self.parent = {'index': 7, 'repair_dir': str(self.batch / 'bias_repairs' / 'M70')}
        self.record = {'job': 'r1/toy_0001', 'passed': True, 'interval': [-.2, .3, .8],
                       'status': 0, 'covQual': 3, 'EDM': .00001,
                       'center_profile': {'r': .3, 'deltaNLL': 0., 'candidate_spread': 0.},
                       'endpoints': [
                           {'side': 'low', 'r': -.2, 'deltaNLL': parallel.TARGET_NLL, 'candidate_spread': 0.},
                           {'side': 'high', 'r': .8, 'deltaNLL': parallel.TARGET_NLL, 'candidate_spread': 0.}],
                       'source_directory': str(self.root / 'verified_fit')}

    def prepare(self):
        parallel.prepare(review, self.main, self.parent, self.args, self.audit, 'audit-completion')
        return backend.load_batch(self.parent['repair_dir'])[1]

    def test_only_flagged_toys_are_submitted_and_merge_waits_for_them(self):
        child = self.prepare()
        self.assertEqual([j['kind'] for j in child['jobs']], ['bias_repair_toy', 'bias_repair_merge'])
        self.assertEqual(child['jobs'][0]['bias_one']['toy'], 1)
        self.assertEqual(child['jobs'][0]['bias_one']['tag'], 'r1')
        self.assertEqual(child['jobs'][1]['dependencies'], [0])
        self.assertEqual(child['parent_completions'], {'7': 'audit-completion'})
        dag = (Path(child['batch']) / 'workflow.dag').read_text()
        self.assertIn('PARENT J0 CHILD J1', dag)
        self.assertIn('CONFIG dagman.config', dag)
        self.assertEqual((Path(child['batch']) / 'dagman.config').read_text(), 'DAGMAN_MAX_JOBS_SUBMITTED = 40\n')

    def test_worker_and_merge_preserve_all_original_toy_ids_and_inputs(self):
        child = self.prepare()
        before = {name: backend.digest(name) for name in child['inputs']}
        with patch.object(repair, 'repair_one_toy', return_value=self.record):
            backend.execute(review, child, 0)
        backend.execute(review, child, 1)
        backend.complete(child, 1)
        for path, fingerprint in before.items():
            self.assertEqual(backend.digest(path), fingerprint)
        merged = json.loads((self.source / 'bias_checks/parallel_verified/merged_toys.json').read_text())
        self.assertEqual(set(merged), {'r0', 'r1', 'r2'})
        self.assertTrue(all(set(v) == {'1', '2'} for v in merged.values()))
        self.assertTrue(merged['r1']['1']['repaired'])
        self.assertFalse(merged['r1']['2']['repaired'])
        summary = json.loads((self.source / 'bias_checks/parallel_verified/bias_summary_Run2Run3_M70.json').read_text())
        self.assertEqual([p['n_toys'] for p in summary], [2, 2, 2])

    def test_failed_toy_blocks_merge_and_does_not_publish_summary(self):
        child = self.prepare()
        with patch.object(repair, 'repair_one_toy', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'No independently verified'):
                backend.execute(review, child, 0)
        with self.assertRaises(RuntimeError):
            backend.execute(review, child, 1)
        self.assertFalse((self.source / 'bias_checks/review_bias_validation.json').exists())

    def test_strict_numerical_criteria_are_identical_for_reused_and_new_results(self):
        task = parallel.context(self.audit, self.source, 'Run2Run3', 70)[2][0]
        parallel.valid_record(self.record, task)
        for key, value in (('status', 1), ('covQual', 1), ('EDM', .1)):
            bad = copy.deepcopy(self.record)
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                parallel.valid_record(bad, task)
        bad = copy.deepcopy(self.record)
        bad['endpoints'][0]['deltaNLL'] = 0.
        with self.assertRaisesRegex(RuntimeError, 'differs from target'):
            parallel.valid_record(bad, task)

    def test_saved_serial_repairs_are_reused_only_for_the_same_dataset_and_toy(self):
        task = parallel.context(self.audit, self.source, 'Run2Run3', 70)[2][0]
        report = self.source / 'bias_checks/bias_M70_verified.previous'
        report.mkdir()
        fit_dir = Path(self.record['source_directory'])
        fit_dir.mkdir()
        for name in ('fit.log', 'higgsCombine.Repair.MultiDimFit.mH70.123456.root', 'multidimfit.Repair.root'):
            (fit_dir / name).write_text('saved evidence')
        command = repair.retry_command(Path(task['workspace']), Path(task['toy_file']), 70, 2, 1, 1., (-2., 100.), 1, .01)
        (report / 'validation.json').write_text(json.dumps({'passed': False, 'repairs': [self.record]}))
        history = [{'directory': str(fit_dir), 'command': command, 'exit_code': 0}]
        (report / 'run_manifest.json').write_text(json.dumps(history))
        child = self.prepare()
        status = backend.complete(child, 0)
        result = json.loads(Path(child['jobs'][0]['files'][0]).read_text())
        self.assertTrue(result['reused_verified_repair'])
        dag = (Path(child['batch']) / 'workflow.dag').read_text()
        self.assertNotIn('JOB J0 ', dag)
        self.assertIn('JOB J1 ', dag)
        command[command.index('--pickToy') + 1] = '2'
        (report / 'run_manifest.json').write_text(json.dumps(history))
        adopted, _ = parallel.find_completed(self.source, [task])
        self.assertEqual(adopted, {})

    def test_modified_audited_input_is_rejected_before_worker_generation(self):
        fit = next((self.source / 'bias').glob('*MultiDimFit*'))
        fit.write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'Audited bias input changed'):
            self.prepare()
        self.assertFalse(Path(self.parent['repair_dir']).exists())

    def test_main_dag_uses_dependent_lazy_subdag_with_its_own_working_directory(self):
        manifest = {**self.main, 'jobs': [], 'requirements': None, 'filesystem_domain': None}
        planner = backend.Planner(review, self.batch, manifest)
        prepare, child = backend.add_parallel_audit(planner, self.args, [])
        backend.write_submission(self.batch, manifest, [prepare, child], 'workflow.dag')
        dag = (self.batch / 'workflow.dag').read_text()
        self.assertIn('SUBDAG EXTERNAL J1 ', dag)
        self.assertIn(' DIR ', dag)
        self.assertIn('PARENT J0 CHILD J1', dag)
        self.assertFalse(Path(manifest['jobs'][child]['subdag']).exists())
        with patch.object(backend.subprocess, 'run') as run:
            backend.submit(self.batch, manifest, 'workflow.dag')
        self.assertIn('-no_recurse', run.call_args.args[0])

    def test_child_dependency_rerun_invalidates_main_and_resume_keeps_good_toy(self):
        self.parent['index'] = 0
        child = self.prepare()
        with patch.object(repair, 'repair_one_toy', return_value=self.record):
            backend.execute(review, child, 0)
        backend.execute(review, child, 1)
        main = {**self.main, 'jobs': [], 'requirements': None, 'filesystem_domain': None}
        planner = backend.Planner(review, self.batch, main)
        parent = planner.add('bias_audit_prepare', self.args, files=[Path(child['batch']) / 'manifest.json'])
        subdag = planner.add('bias_repair_dag', self.args, [parent], subdag=str(Path(child['batch']) / 'workflow.dag'))
        job = main['jobs'][parent]
        backend.dump(Path(job['workdir']) / 'status.json', {
            'success': True, 'batch_id': 'test', 'completion_id': 'audit-completion',
            'job_sha256': backend.hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
            'outputs': {str(Path(child['batch']) / 'manifest.json'): backend.digest(Path(child['batch']) / 'manifest.json')},
            'dependencies': {}})
        backend.complete(main, subdag)
        with patch.object(repair, 'repair_one_toy', return_value=self.record):
            backend.execute(review, child, 0)
        with self.assertRaisesRegex(RuntimeError, 'stale after dependency'):
            backend.complete(main, subdag)
        old_rescue = Path(child['batch']) / 'workflow.dag.rescue001'
        old_rescue.write_text('old native rescue')
        backend.refresh_child_dags(self.batch, main, [subdag])
        dag = (Path(child['batch']) / 'workflow.dag').read_text()
        self.assertNotIn('JOB J0 ', dag)
        self.assertIn('JOB J1 ', dag)
        self.assertFalse(old_rescue.exists())
        self.assertTrue(list(Path(child['batch']).glob('scheduling_archive_*/workflow.dag.rescue001')))

    def test_migration_retains_completed_jobs_and_replaces_only_incomplete_audits(self):
        main = {**self.main, 'jobs': [], 'inputs': {}, 'requirements': None, 'filesystem_domain': None}
        planner = backend.Planner(review, self.batch, main)
        completed = planner.add('bias_chunk', self.args)
        legacy = planner.add('bias_audit', self.args, [completed], audit=True)
        completed_before = copy.deepcopy(main['jobs'][completed])
        for name in ('review_condor.py', 'combine_review.py', 'repair_bias_intervals.py'):
            (self.batch / name).write_text('legacy ' + name)
            main['inputs'][str(self.batch / name)] = backend.digest(self.batch / name)
        backend.dump(self.batch / 'manifest.json', main)
        with patch.object(backend, 'SERIAL_BACKEND_SHA256', backend.digest(self.batch / 'review_condor.py')), \
             patch.object(backend, 'SERIAL_HELPER_SHA256', backend.digest(self.batch / 'combine_review.py')), \
             patch.object(backend, 'SERIAL_REPAIR_SHA256', backend.digest(self.batch / 'repair_bias_intervals.py')):
            updated = backend.parallelize_bias(review, self.batch, main)
        self.assertEqual(updated['jobs'][completed], completed_before)
        self.assertEqual(updated['jobs'][legacy]['kind'], 'bias_repair_dag')
        added = updated['jobs'][-1]
        self.assertEqual(added['kind'], 'bias_audit_prepare')
        self.assertEqual(added['dependencies'], [completed])
        self.assertEqual(updated['jobs'][legacy]['dependencies'], [added['index']])
        self.assertIn(str(self.batch / 'review_bias_parallel.py'), updated['inputs'])


if __name__ == '__main__':
    unittest.main()
