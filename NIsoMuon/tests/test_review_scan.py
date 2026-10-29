"""Scan completeness, retry provenance and preservation of completed jobs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import combine_review as review
import review_condor as condor
import review_scan as scan
import validate_bias_repair as verify


def row(r, quantile=.5, delta=None, nll=0.):
    return {'r': r, 'deltaNLL': .5 * r*r if delta is None else delta,
            'quantileExpected': quantile, 'nll': nll, 'nll0': 100.}


def grid(low=-3., high=3., count=12):
    return [row(0., -1., 0.)] + [row(scan.coordinate(i, low, high, count)) for i in range(count)]


class ScanTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_five_m20_holes_are_found_without_confusing_repeated_chunk_minima(self):
        missing = {178, 179, 254, 255, 256}
        rows = [row(0., -1., 0.) for _ in range(30)]
        rows += [row(scan.coordinate(i, -50., 100., 600), delta=.5*(scan.coordinate(i, -50., 100., 600)/7.5)**2)
                 for i in range(600) if i not in missing]
        _, _, issues, holes, _ = scan.analyse(rows, -50., 100., 600)
        self.assertEqual(set(holes), missing)
        self.assertEqual(sorted(issues[i]['r'] for i in holes), [-5.375, -5.125, 13.625, 13.875, 14.125])
        self.assertTrue(all('found 0' in issues[i]['issues'][0] for i in holes))

    def test_missing_invalid_and_filtered_spike_have_distinct_diagnostics(self):
        rows = grid()
        rows = [r for r in rows if r['r'] != -1.25]
        next(r for r in rows if r['r'] == -.75)['deltaNLL'] = 9990.
        next(r for r in rows if r['r'] == 1.75)['deltaNLL'] = 4.
        _, _, problems, holes, _ = scan.analyse(rows, -3., 3., 12)
        self.assertEqual(holes, [3, 4, 9])
        self.assertIn('found 0', problems[3]['issues'][0])
        self.assertIn('sentinel', problems[4]['issues'][0])
        self.assertIn('plotting cut', problems[9]['issues'][0])

    def test_tail_failures_are_reported_without_unnecessary_tail_refits(self):
        rows = grid()
        next(r for r in rows if r['r'] == -2.75)['deltaNLL'] = 9990.
        _, _, problems, holes, extent = scan.analyse(rows, -3., 3., 12)
        self.assertIn(0, problems)
        self.assertEqual(holes, [])
        self.assertEqual(extent, (1, 10))

    def test_nonfinite_and_duplicate_grid_rows_remain_visible(self):
        rows = grid()
        next(r for r in rows if r['r'] == .75)['deltaNLL'] = float('nan')
        rows.append(row(-1.25))
        _, _, problems, holes, _ = scan.analyse(rows, -3., 3., 12)
        self.assertEqual(holes, [3, 7])
        self.assertIn('found 2', problems[3]['issues'][0])
        self.assertEqual(problems[7]['issues'], ['non-finite scan row'])

    def test_inconsistent_chunk_reference_and_off_grid_rows_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'Inconsistent initial minima'):
            scan.analyse(grid() + [row(0., -1., 0., nll=.01)], -3., 3., 12)
        with self.assertRaisesRegex(RuntimeError, 'Off-grid'):
            scan.analyse(grid() + [row(.33)], -3., 3., 12)

    def test_crossing_checks_reject_missing_high_side_support(self):
        points = [r for r in grid() if 2*r['deltaNLL'] <= 7]
        brackets = scan.crossing_brackets(points, 0.)
        self.assertEqual([r['r'] for r in brackets['4']['high']], [1.75, 2.25])
        with self.assertRaisesRegex(RuntimeError, 'both sides.*4'):
            scan.crossing_brackets([p for p in points if p['r'] < 2], 0.)

    def retry(self, profiles, initial=None):
        command = ['combine', '-M', 'MultiDimFit', '/inputs/workspace.root', '-m', '20',
                   '-t', '-1', '--expectSignal', '0', '--setParameters', 'r=0',
                   '--algo', 'grid', '--points', '12', '--rMin', '-3', '--rMax', '3', '--saveNLL', '-n', '.Scan']
        commands = []
        def run(cmd, directory, **kwargs):
            commands.append(cmd)
            (directory / 'higgsCombine.Scan.MultiDimFit.mH20.root').write_text('mock ROOT')
            return 0
        info = {'status': 0, 'covQual': 3, 'EDM': 1e-6, 'parameters': {'r': 0.}}
        with patch.object(scan, 'read_rows', side_effect=profiles), patch.object(verify, 'fit_info', return_value=info):
            directory = Path(tempfile.mkdtemp(prefix='retry_', dir=self.root))
            value = scan.retry_point(command, 9, 1.75, initial or row(0., -1., 0.), directory, run)
        return value, commands

    def test_single_point_retries_preserve_asimov_grid_and_require_two_agreeing_minima(self):
        ((point, attempts), commands) = self.retry([
            [row(0., -1., 0.), row(1.75, delta=1.532)],
            [row(0., -1., 0.), row(1.75, delta=1.53125)]])
        self.assertEqual(point['deltaNLL'], 1.53125)
        self.assertEqual(len(attempts), 2)
        for cmd in commands:
            for flag, expected in [('-t', '-1'), ('--expectSignal', '0'), ('--setParameters', 'r=0'),
                                   ('--points', '12'), ('--rMin', '-3'), ('--rMax', '3'),
                                   ('--firstPoint', '9'), ('--lastPoint', '9')]:
                self.assertEqual(condor.option(cmd, flag), expected)
            self.assertIn('/inputs/workspace.root', cmd)

    def test_one_good_retry_or_disagreeing_retries_cannot_pass(self):
        good = [row(0., -1., 0.), row(1.75)]
        missing = [row(0., -1., 0.)]
        with self.assertRaisesRegex(RuntimeError, 'No two consistent profiles'):
            self.retry([good, missing, missing, missing])
        with self.assertRaisesRegex(RuntimeError, 'No two consistent profiles'):
            self.retry([[row(0., -1., 0.), row(1.75, delta=d)] for d in [1., 2., 3., 4.]])

    def test_changed_retry_nll_reference_is_rejected(self):
        changed = [row(0., -1., 0., nll=.1), row(1.75, nll=.1)]
        with self.assertRaisesRegex(RuntimeError, 'No two consistent profiles'):
            self.retry([changed] * 4)

    def test_validator_preserves_raw_file_and_fails_before_publishing_unresolved_holes(self):
        source = self.root / 'source.root'
        source.write_bytes(b'original scan')
        before = scan.digest(source)
        rows = [r for r in grid() if r['r'] != 1.75]
        with patch.object(scan, 'read_rows', return_value=rows), \
             patch.object(scan, 'retry_point', side_effect=RuntimeError('failed conditional fit')), \
             patch.object(scan, 'write_rows') as write:
            with self.assertRaisesRegex(RuntimeError, 'failed conditional fit'):
                scan.validate_scan(source, [], self.root, '20', -3., 3., 12, None)
        self.assertEqual(scan.digest(source), before)
        self.assertFalse(write.called)
        report = json.loads((self.root / 'scan_validation_M20.json').read_text())
        self.assertFalse(report['passed'])
        self.assertEqual(report['repair_indices'], [9])
        self.assertIn('failed conditional fit', report['error'])

    def test_complete_scan_skips_refits_and_binds_report_to_validated_root(self):
        source = self.root / 'source.root'
        source.write_bytes(b'original scan')
        def write(path, rows):
            self.assertEqual(len(rows), 11)  # Ten below-cut grid rows + best fit.
            path.write_text(json.dumps(rows))
        with patch.object(scan, 'read_rows', return_value=grid()), \
             patch.object(scan, 'retry_point') as retry, patch.object(scan, 'write_rows', side_effect=write):
            validated = scan.validate_scan(source, [], self.root, '70', -3., 3., 12, None)
        self.assertFalse(retry.called)
        report = json.loads((self.root / 'scan_validation_M70.json').read_text())
        self.assertTrue(report['passed'])
        self.assertEqual(report['validated_sha256'], scan.digest(validated))
        self.assertFalse(report['original_per_point_fit_status_available'])

    def test_successful_repair_changes_only_missing_point_and_keeps_exact_best_fit(self):
        source = self.root / 'source.root'
        source.write_bytes(b'original scan')
        rows = grid(count=13)  # This grid also contains the exact minimum r=0.
        absent = scan.coordinate(9, -3., 3., 13)
        rows = [p for p in rows if p['r'] != absent]
        original = {p['r']: p for p in rows if 0 <= p['quantileExpected'] and 2*p['deltaNLL'] <= 7}
        written = []
        def write(path, points):
            written.extend(points)
            path.write_text(json.dumps(points))
        with patch.object(scan, 'read_rows', return_value=rows), \
             patch.object(scan, 'retry_point', return_value=(row(absent), [])), \
             patch.object(scan, 'write_rows', side_effect=write):
            scan.validate_scan(source, [], self.root, '20', -3., 3., 13, None)
        self.assertEqual([p for p in written if p['r'] == 0.], [row(0., -1., 0.)])
        self.assertEqual(next(p for p in written if p['r'] == absent), row(absent))
        for p in written:
            if p['r'] not in (0., absent):
                self.assertEqual(p, original[p['r']])
        self.assertEqual(source.read_bytes(), b'original scan')

    def test_scan_plot_passes_correct_asimov_label_and_validated_input(self):
        args = review.build_parser().parse_args(['20', '--task', 'scan'])
        args.output_base = str(self.root)
        output = review.outdir(args)
        output.mkdir(parents=True)
        raw = output / 'higgsCombine.ReviewScan_Run2Run3_M20.MultiDimFit.mH20.root'
        raw.touch()
        validated = output / 'scan_validated_M20.root'
        commands = []
        with patch.object(review, 'require_command'), patch.object(review, 'ensure_workspace', return_value=self.root/'workspace.root'), \
             patch.object(review, 'safe_negative_rmin', return_value=-2.), \
             patch.object(review, 'run_command', side_effect=lambda cmd, *a, **kw: commands.append(cmd) or 0), \
             patch.object(review.shutil, 'which', return_value='plot1DScan.py'), \
             patch.object(scan, 'validate_scan', return_value=validated):
            review.task_scan(args, self.root/'card.txt')
        plot = commands[-1]
        self.assertEqual(plot[1], str(validated))
        self.assertEqual(condor.option(plot, '--main-label'), 'Background-only Asimov')


class ScanMigrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.batch = Path(temporary.name)
        args = review.build_parser().parse_args(['20'])
        args.tasks = ['scan']
        args.output_base = str(self.batch/'results')
        self.manifest = {'batch': str(self.batch), 'batch_id': 'test', 'inputs': {}, 'jobs': []}
        planner = condor.Planner(review, self.batch, self.manifest)
        self.chunk = planner.add('scan_chunk', args, files=[self.batch/'chunk.root'])
        self.fit = planner.add('unrelated_fit', args, files=[self.batch/'fit.root'])
        self.collector = planner.add('scan_collect', args, [self.chunk], helper='scan',
                                     files=[self.batch/'merged.root', self.batch/'scan.pdf'])
        for index in (self.chunk, self.fit):
            job = self.manifest['jobs'][index]
            Path(job['files'][0]).write_text('original verified result')
            condor.dump(Path(job['workdir'])/'status.json', {
                'success': True, 'batch_id': 'test', 'completion_id': str(index), 'dependencies': {},
                'job_sha256': hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
                'outputs': {job['files'][0]: condor.digest(job['files'][0])}})
        self.known = {}
        for name in condor.PRE_SCAN_AUDIT_HELPERS:
            path = self.batch/name
            path.write_text('published pre-audit ' + name)
            self.known[name] = {condor.digest(path)}
            self.manifest['inputs'][str(path)] = condor.digest(path)
        condor.dump(self.batch/'manifest.json', self.manifest)

    def test_only_scan_collectors_become_stale_and_migration_is_idempotent(self):
        before = copy.deepcopy(self.manifest)
        with patch.object(condor, 'PRE_SCAN_AUDIT_HELPERS', self.known):
            updated = condor.refresh_scan_validation(review, self.batch, self.manifest)
        for index in (self.chunk, self.fit):
            self.assertEqual(updated['jobs'][index], before['jobs'][index])
            condor.complete(updated, index)
        self.assertEqual(updated['jobs'][self.collector]['scan_validation'], scan.SCHEMA)
        self.assertEqual(len(updated['jobs'][self.collector]['files']), 4)
        self.assertIn(str(self.batch/'review_scan.py'), updated['inputs'])
        archive = Path(updated['scan_validation_patch']['archive'])
        self.assertEqual(json.loads((archive/'manifest.json').read_text()), before)
        self.assertIs(condor.refresh_scan_validation(review, self.batch, updated), updated)

    def test_unknown_snapshot_is_rejected_without_mutation(self):
        before = (self.batch/'manifest.json').read_bytes()
        with self.assertRaisesRegex(RuntimeError, 'Unknown pre-audit scan helper'):
            condor.refresh_scan_validation(review, self.batch, self.manifest)
        self.assertEqual((self.batch/'manifest.json').read_bytes(), before)

    def test_old_successful_collector_cannot_be_collected_without_point_audit(self):
        with self.assertRaisesRegex(RuntimeError, 'scan point audit missing; run --stage resume'):
            condor.collect(self.batch, self.manifest)


if __name__ == '__main__':
    unittest.main()
