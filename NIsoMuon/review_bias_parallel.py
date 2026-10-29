"""Per-toy bias repairs and strict merging for the Condor review backend."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import statistics

import repair_bias_intervals as repair
import review_condor as backend
from uuid import uuid4

TARGET_NLL = .5 * statistics.NormalDist().inv_cdf(.84) ** 2
SOURCES = ('combine_review.py', 'review_condor.py', 'review_bias_parallel.py', 'limit_workflow.py',
           'audit_bias_intervals.py', 'repair_bias_intervals.py', 'validate_bias_repair.py')


def context(audit_path, source, target, mass):
    audit_path, source = Path(audit_path).resolve(), Path(source).resolve()
    data = json.loads(audit_path.read_text())
    if data.get('errors') or len(data.get('ensembles', [])) != 3:
        raise RuntimeError('Incomplete bias audit')
    inputs = {str(audit_path): backend.digest(audit_path)}
    for name, value in data['input_sha256'].items():
        if not Path(name).is_file() or backend.digest(name) != value:
            raise RuntimeError(f'Audited bias input changed: {name}')
        inputs[name] = value
    summary = source / 'bias' / f'bias_summary_{target}_M{mass}.json'
    if str(summary) not in inputs:
        raise RuntimeError('Bias summary is not among the audited inputs')
    original = json.loads(summary.read_text())
    workspace = repair.verify.one(source, f'workspace_{target}_M{mass}.root')
    inputs[str(workspace)] = backend.digest(workspace)
    tasks = []
    for ensemble in data['ensembles']:
        tag, injection = ensemble['tag'], ensemble['injected_r']
        matches = [s for s in original if s['injected_r'] == injection]
        if len(matches) != 1 or matches[0]['n_toys_requested'] != ensemble['n_expected']:
            raise RuntimeError(f'Bias injection/count mismatch: {tag}')
        fit = source / 'bias' / Path(matches[0]['fit_file']).name
        if str(fit) != ensemble['fit_file'] or str(fit) not in inputs:
            raise RuntimeError(f'Fit input differs from the audit: {tag}')
        dataset = repair.verify.one(source / 'bias',
            f'higgsCombine.ReviewBiasGen_{target}_M{mass}_{tag}.GenerateOnly.mH*.root')
        inputs[str(dataset)] = backend.digest(dataset)
        for toy in ensemble['flagged_toy_ids']:
            tasks.append({'tag': tag, 'toy': toy, 'injection': injection,
                'count': ensemble['n_expected'], 'bounds': ensemble['r_range'], 'mass': mass,
                'workspace': str(workspace), 'toy_file': str(dataset)})
    return data, inputs, tasks


def valid_record(record, task):
    """Apply the serial worker's fit, centre and endpoint acceptance criteria."""
    if record.get('job') != f"{task['tag']}/toy_{task['toy']:04d}" or not record.get('passed'):
        raise RuntimeError('Wrong or failed repaired toy')
    repair.verify.check_fit(record, .001)
    interval = record['interval']
    if not repair.verify.valid_interval(interval, *task['bounds']):
        raise RuntimeError('Invalid repaired interval')
    center = record['center_profile']
    repair.validate_point(center, interval[1], 0., .001)
    if center.get('candidate_spread', 0.) > .001:
        raise RuntimeError('Inconsistent centre profiles')
    endpoints = {p['side']: p for p in record['endpoints']}
    if set(endpoints) != {'low', 'high'}:
        raise RuntimeError('Missing repaired interval endpoint')
    for side, value in (('low', interval[0]), ('high', interval[2])):
        repair.validate_point(endpoints[side], value, TARGET_NLL, .01)
        if endpoints[side].get('candidate_spread', 0.) > .01:
            raise RuntimeError('Inconsistent endpoint profiles')


def find_completed(source, tasks):
    """Reuse only individually verified serial results with matching toy inputs."""
    available = {}
    expected = {f"{t['tag']}/toy_{t['toy']:04d}": t for t in tasks}
    proofs = {}
    for report_path in sorted((source / 'bias_checks').glob('bias_M*_verified.*/validation.json')):
        history_path = report_path.parent / 'run_manifest.json'
        if not history_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text())
            history = json.loads(history_path.read_text())
            for record in report.get('repairs', []):
                task = expected.get(record.get('job'))
                if not task:
                    continue
                valid_record(record, task)
                saved = Path(record['source_directory'])
                if (not saved.is_dir() or not (saved / 'fit.log').is_file()
                    or not list(saved.glob('higgsCombine*.MultiDimFit.mH*.root'))
                    or not list(saved.glob('multidimfit*.root'))):
                    continue
                attempts = [x for x in history if x.get('directory') == record['source_directory'] and x.get('exit_code') == 0]
                if len(attempts) != 1:
                    continue
                cmd = attempts[0]['command']
                if (task['workspace'] not in cmd or backend.option(cmd, '--toysFile') != task['toy_file']
                    or int(backend.option(cmd, '--pickToy')) != task['toy']
                    or int(backend.option(cmd, '-t')) != task['count']
                    or float(backend.option(cmd, '-m')) != task['mass']
                    or float(backend.option(cmd, '--rMin')) != task['bounds'][0]
                    or float(backend.option(cmd, '--rMax')) != task['bounds'][1]
                    or not math.isclose(float(backend.option(cmd, '--setParameters').split('=')[1]), task['injection'])):
                    continue
                available[record['job']] = copy.deepcopy(record)
                proofs[str(report_path)] = backend.digest(report_path)
                proofs[str(history_path)] = backend.digest(history_path)
        except (ValueError, KeyError, TypeError, RuntimeError, OSError):
            continue
    return available, proofs


def prepare(review, main_manifest, parent_job, args, audit_path, completion_id):
    child = Path(parent_job['repair_dir'])
    if child.exists() and any(child.iterdir()):
        raise RuntimeError(f'Repair DAG directory already exists: {child}; inspect the prepared batch instead of replacing it')
    mass = int(args.mass)
    source = review.outdir(args)
    data, inputs, tasks = context(audit_path, source, args.target, mass)
    available, proofs = find_completed(source, tasks)
    inputs.update(proofs)
    child.mkdir(parents=True, exist_ok=True)
    for name in SOURCES:
        original = Path(main_manifest['batch']) / name
        dest = child / name
        shutil.copy2(original, dest)
        inputs[str(dest)] = backend.digest(dest)
    manifest = {'schema': 'NIsoMuon_review_condor_v1', 'batch': str(child),
        'batch_id': main_manifest['batch_id'] + '_bias_' + str(parent_job['index']),
        'output_base': str(source / 'bias_checks'), 'suite': False,
        'cmssw_base': main_manifest['cmssw_base'], 'scram_arch': main_manifest['scram_arch'],
        'memory': main_manifest['memory'], 'max_jobs': main_manifest['max_jobs'],
        'requirements': main_manifest.get('requirements'), 'filesystem_domain': main_manifest.get('filesystem_domain'),
        'inputs': inputs, 'jobs': [], 'audit': str(audit_path), 'source': str(source),
        'target': args.target, 'mass': mass,
        'parent_completions': {str(parent_job['index']): completion_id}}
    planner = backend.Planner(review, child, manifest)
    workers = []
    for task in tasks:
        index = planner.add('bias_repair_toy', args, bias_one=task)
        job = manifest['jobs'][index]
        result = Path(job['workdir']) / 'repair_result.json'
        job['files'] = [str(result)]
        record = available.get(f"{task['tag']}/toy_{task['toy']:04d}")
        if record:
            backend.dump(result, {'passed': True, 'repair': record, 'reused_verified_repair': True})
            backend.dump(Path(job['workdir']) / 'status.json', {
                'success': True, 'batch_id': manifest['batch_id'], 'completion_id': uuid4().hex,
                'job_sha256': backend.hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
                'outputs': {str(result): backend.digest(result)}, 'dependencies': {}})
        else:
            workers.append(index)
    final = planner.add('bias_repair_merge', args, list(range(len(tasks))), bias_merge=True,
        files=[source / 'bias_checks' / 'review_bias_validation.json', child / 'validation.json',
               source / 'bias_checks' / 'parallel_verified' / f'bias_summary_{args.target}_M{mass}.json'])
    workers.append(final)
    backend.dump(child / 'manifest.json', manifest)
    backend.write_submission(child, manifest, workers, 'workflow.dag')
    print(f"[BIAS CONDOR] {len(tasks)} flagged toys; {len(tasks) - len(workers) + 1} verified repairs reused; {len(workers) - 1} toy workers")


def execute(manifest, job):
    work = Path(job['workdir'])
    if job.get('bias_one'):
        task = job['bias_one']
        history, failures = [], []
        attempt = work / ('attempt_' + uuid4().hex[:8])
        attempt.mkdir()
        try:
            record = repair.repair_one_toy(Path(task['workspace']), Path(task['toy_file']),
                task['mass'], task['count'], task['toy'], task['injection'], task['bounds'],
                task['tag'], attempt, TARGET_NLL, history, failures)
            if record is None:
                raise RuntimeError(f"No independently verified repair for {task['tag']}/toy_{task['toy']:04d}")
            valid_record(record, task)
            backend.dump(work / 'repair_result.json', {'passed': True, 'repair': record})
        finally:
            backend.dump(attempt / 'run_manifest.json', history)
            backend.dump(attempt / 'failed_attempts.json', failures)
    else:
        merge(manifest)


def merge(manifest):
    data, _inputs, tasks = context(manifest['audit'], manifest['source'], manifest['target'], manifest['mass'])
    records = {}
    for job in manifest['jobs']:
        if not job.get('bias_one'):
            continue
        payload = json.loads(Path(job['files'][0]).read_text())
        if not payload.get('passed'):
            raise RuntimeError('Failed repaired toy in merge')
        record = payload['repair']
        valid_record(record, job['bias_one'])
        if record['job'] in records:
            raise RuntimeError('Duplicate repaired toy')
        records[record['job']] = record
    expected = {f"{t['tag']}/toy_{t['toy']:04d}" for t in tasks}
    if set(records) != expected:
        raise RuntimeError('Incomplete or unexpected repaired-toy set')
    summaries, merged = [], {}
    for ensemble in data['ensembles']:
        tag, bounds = ensemble['tag'], ensemble['r_range']
        values = {}
        for toy, entry in ensemble['toys'].items():
            name = f'{tag}/toy_{int(toy):04d}'
            fixed = records.get(name)
            interval = fixed['interval'] if fixed else entry['interval']
            if not repair.verify.valid_interval(interval or [], *bounds):
                raise RuntimeError(f'Invalid final interval: {name}')
            values[toy] = {'interval': interval, 'repaired': fixed is not None,
                           'source': fixed['source_directory'] if fixed else ensemble['fit_file']}
        if set(map(int, values)) != set(range(1, ensemble['n_expected'] + 1)):
            raise RuntimeError('Incomplete original toy IDs')
        summary = repair.verify.summarize(values, ensemble['injected_r'])
        summary['n_repaired'] = sum(x['repaired'] for x in values.values())
        summaries.append(summary)
        merged[tag] = values
    child = Path(manifest['batch'])
    output = Path(manifest['source']) / 'bias_checks'
    directory = output / 'parallel_verified'
    directory.mkdir(parents=True, exist_ok=True)
    backend.dump(directory / 'merged_toys.json', merged)
    backend.dump(directory / f"bias_summary_{manifest['target']}_M{manifest['mass']}.json", summaries)
    backend.dump(child / 'validation.json', {'passed': True, 'n_verified_repairs': len(records),
        'repairs': list(records.values()), 'errors': [], 'coverage_test': False,
        'scope': 'Independent profiles for repaired toys; structural checks for other toys'})
    backend.dump(output / 'review_bias_validation.json', {'passed': True, 'audit': manifest['audit'],
        'repair': str(child / 'validation.json'), 'parallel': True})
