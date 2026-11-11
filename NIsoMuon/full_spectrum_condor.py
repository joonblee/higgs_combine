"""Shared-filesystem DAG: raw era/bin jobs -> combined card -> joint fit.

Invoked through full_spectrum_fit.py. Workers use a source snapshot and never
fit bins separately. Changing job grouping cannot change the likelihood.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from uuid import uuid4

import full_spectrum_fit as fs
import limit_workflow as base
import limit_workflow_sideband_fit as shared

SCHEMA = 'NPS26009_full_spectrum_condor_v1'
SOURCES = ('full_spectrum_fit.py', 'full_spectrum_condor.py',
           'limit_workflow.py', 'limit_workflow_sideband_fit.py')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def stamp(path):
    info = Path(path).stat()
    return dict(size=info.st_size, mtime_ns=info.st_mtime_ns)


class InputReader(shared.BlindedReader):
    """Record every opened input, checking that it did not change during a job."""
    def __init__(self):
        super().__init__()
        self.fingerprints = {}

    def _file(self, filename):
        # Apply the blinding check before even stat-ing a target-data file.
        if Path(filename).name.lower() == 'data.root':
            raise base.WorkflowError('OS data.root is forbidden in this blinded workflow.')
        path = str(Path(filename).resolve())
        if path not in self.fingerprints and Path(path).is_file():
            self.fingerprints[path] = stamp(path)
        return super()._file(filename)

    def check_inputs(self):
        check_inputs(self.fingerprints)


def check_inputs(fingerprints):
    for path, expected in fingerprints.items():
        if stamp(path) != expected:
            raise ValueError(f'Input changed during the batch: {path}; submit a new batch.')


def make_jobs(opt, windows):
    """Pure planning: one preparation node per era/bin chunk, then dependencies."""
    if opt.bins_per_job < 1:
        raise ValueError('--bins-per-job must be at least 1.')
    jobs = []
    def add(kind, mass, dependencies=(), **fields):
        job = dict(index=len(jobs), kind=kind, mass=mass,
                   dependencies=list(dependencies), **fields)
        jobs.append(job)
        return job['index']
    for mass in opt.masses:
        parents = []
        if opt.stage in ('cards', 'all'):
            for era in base.target_years(opt.target):
                bins = windows[era]
                for start in range(0, len(bins), opt.bins_per_job):
                    parents.append(add('prepare', mass, era=era,
                                       windows=bins[start:start+opt.bins_per_job]))
            parents = [add('assemble', mass, parents)]
        if opt.stage in ('fit', 'all'):
            add('fit', mass, parents)
    return jobs


def workdir(payload, job):
    return Path(payload['batch'])/'jobs'/f'job_{job["index"]:05d}'


def source_check(payload):
    for name, expected in payload['source_hashes'].items():
        if file_hash(Path(payload['batch'])/'source'/name) != expected:
            raise ValueError(f'Batch source snapshot changed: {name}')


def load_manifest(directory):
    path = Path(directory).resolve()
    if path.is_dir():
        path /= 'manifest.json'
    payload = json.loads(path.read_text())
    if payload.get('schema') != SCHEMA or payload.get('batch') != str(path.parent):
        raise ValueError('Invalid batch manifest or moved batch directory.')
    expected = payload.get('manifest_sha256')
    if digest({k:v for k,v in payload.items() if k != 'manifest_sha256'}) != expected:
        raise ValueError('Batch manifest changed; submit a new batch.')
    source_check(payload)
    return payload


def completed(payload, job, cache=None):
    """Accept only current products with matching dependency completions."""
    cache = {} if cache is None else cache
    index = job['index']
    if index in cache:
        return cache[index]
    path = workdir(payload, job)/'status.json'
    try:
        status = json.loads(path.read_text())
        if (status.get('state') != 'success' or status.get('batch_id') != payload['batch_id']
                or status.get('job_sha256') != digest(job)):
            raise ValueError('No valid completion')
        for parent in job['dependencies']:
            previous = completed(payload, payload['jobs'][parent], cache)
            if not previous or status['dependency_completions'].get(str(parent)) != previous['completion_id']:
                raise ValueError('Dependency was rerun or is incomplete')
        if not status.get('products'):
            raise ValueError('Missing products')
        for filename, checksum in status['products'].items():
            if file_hash(filename) != checksum:
                raise ValueError('Product changed')
    except (OSError, ValueError, KeyError, TypeError):
        status = None
    cache[index] = status
    return status


def worker_options(payload):
    opt = argparse.Namespace(**payload['options'])
    opt.backend = 'local'
    opt.output_base = str(Path(payload['batch'])/'results')
    args = fs.configuration(opt)
    args.base_dir = payload['input_base_dir']
    args.resolution_coefficients = payload['resolution_coefficients']
    return opt, args


def assemble(payload, job, opt, args):
    parts, fingerprints = [], dict(payload['preflight_inputs'])
    expected = [(era, window) for era in base.target_years(opt.target)
                for window in payload['windows'][era]]
    seen = []
    for index in job['dependencies']:
        parent = payload['jobs'][index]
        data = json.loads((workdir(payload, parent)/'part.json').read_text())
        if (data.get('batch_id') != payload['batch_id'] or data.get('job_sha256') != digest(parent)
                or data.get('mass') != job['mass']):
            raise ValueError(f'Stale or foreign shard: job {index}')
        part = data['part']
        if part['era'] != parent['era'] or len(part['input_audits']) != len(parent['windows']):
            raise ValueError(f'Invalid shard era/audits: job {index}')
        actual = [dict(region=r['region'], low=r['low'], high=r['high']) for r in part['rows']]
        if actual != parent['windows'] or any(r['era'] != parent['era'] or
                r['channel'] != f'bin_{parent["era"]}_{r["region"]}' for r in part['rows']):
            raise ValueError(f'Shard bins differ from plan: job {index}')
        for audit, window in zip(part['input_audits'], parent['windows']):
            if audit['era'] != parent['era'] or audit['region'] != window['region']:
                raise ValueError(f'Shard audits differ from plan: job {index}')
        seen.extend((part['era'], w) for w in actual)
        for filename, value in data['inputs'].items():
            if filename in fingerprints and fingerprints[filename] != value:
                raise ValueError(f'Inconsistent input versions across shards: {filename}')
            fingerprints[filename] = value
        parts.append(part)
    if seen != expected:
        raise ValueError('Incomplete, duplicate or reordered bin coverage; refusing joint card.')
    check_inputs(fingerprints)
    study = argparse.Namespace(**vars(opt), norm_scope='era')
    model = shared.assemble_model(args, study, job['mass'], parts, full_spectrum=True)
    model = fs.decorate_model(model, opt, payload['windows'])
    model['batch_inputs'] = fingerprints
    model['condor_batch_id'] = payload['batch_id']
    directory = fs.model_directory(opt, job['mass'])
    fs.save_model(model, opt, job['mass'], directory)
    return [directory/'datacard.txt', directory/'model.json']


@contextmanager
def job_lock(directory):
    with (directory/'worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(f'A worker is already active in {directory}') from None
        yield


def run_worker(manifest, index):
    payload = load_manifest(manifest)
    if index is None or not 0 <= index < len(payload['jobs']):
        raise ValueError('Invalid --condor-index.')
    job = payload['jobs'][index]
    directory = workdir(payload, job)
    with job_lock(directory):
        status = dict(batch_id=payload['batch_id'], job_sha256=digest(job),
                      state='running', started=time.time(), kind=job['kind'], index=index)
        write_json(directory/'status.json', status)
        try:
            dependencies = {}
            for parent in job['dependencies']:
                previous = completed(payload, payload['jobs'][parent])
                if not previous:
                    raise ValueError(f'Dependency job {parent} is incomplete or stale.')
                dependencies[str(parent)] = previous['completion_id']
            opt, args = worker_options(payload)
            check_inputs(payload['preflight_inputs'])
            if job['kind'] == 'prepare':
                reader = InputReader()
                try:
                    part = shared.build_bin_rows(reader, args, opt, job['mass'],
                        payload['signal_files'][base.mass_label(job['mass'])], job['era'],
                        job['windows'], full_spectrum=True)
                    reader.check_inputs()
                    path = directory/'part.json'
                    write_json(path, dict(batch_id=payload['batch_id'], job_sha256=digest(job),
                        mass=job['mass'], part=part, inputs=reader.fingerprints))
                    products = [path]
                finally:
                    reader.close()
            elif job['kind'] == 'assemble':
                products = assemble(payload, job, opt, args)
            elif job['kind'] == 'fit':
                model_dir = fs.model_directory(opt, job['mass'])
                model = fs.load_model(model_dir, opt)
                check_inputs(model.get('batch_inputs', {}))
                fs.run_fit(model, model_dir, opt)
                fit_dir = model_dir/f'fit_r{base.mass_label(opt.inject_r)}'
                products = [fit_dir/name for name in
                            ('fit_summary.json', 'fitDiagnostics.FullSpectrum.root', 'workspace.root')]
            else:
                raise ValueError('Unknown batch job kind.')
            status.update(state='success', completion_id=uuid4().hex,
                          dependency_completions=dependencies,
                          products={str(p):file_hash(p) for p in products})
        except Exception as error:
            status.update(state='failed', error=str(error))
            raise
        finally:
            status.update(finished=time.time(), elapsed_seconds=time.time()-status['started'])
            write_json(directory/'status.json', status)
            print(f'[JOB] {index} {job["kind"]}: {status["state"]}; {status["elapsed_seconds"]:.1f} s', flush=True)


def write_dag(payload, indices, path):
    selected = set(indices)
    lines = []
    for index in indices:
        job = payload['jobs'][index]
        lines += [f'JOB J{index} submit.jds', f'VARS J{index} jobindex="{index}"',
                  f'RETRY J{index} {payload["options"]["condor_retries"]}']
        parents = [i for i in job['dependencies'] if i in selected]
        if parents:
            lines.append('PARENT '+' '.join(f'J{i}' for i in parents)+f' CHILD J{index}')
    Path(path).write_text('\n'.join(lines)+'\n')


def submission(payload, dag):
    command = ['condor_submit_dag', '-batch-name', 'NPSFullSpectrum_'+payload['batch_id']]
    cap = payload['options']['condor_maxjobs']
    if cap:
        command += ['-maxjobs', str(cap)]
    return command+[Path(dag).name]


def validate_options(opt):
    if opt.bins_per_job < 1 or opt.condor_memory < 1 or opt.condor_maxjobs < 0 or opt.condor_retries < 0:
        raise ValueError('Require positive bins-per-job/memory and nonnegative maxjobs/retries.')
    if any(c in opt.condor_requirements for c in ('\n', '\r')):
        raise ValueError('Condor requirements must be one line.')


def submit(opt, args):
    validate_options(opt)
    cmssw, arch = base.condor_environment(opt)
    if not opt.condor_dry_run:
        base.require_command('condor_submit_dag')
    batch_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid4().hex[:8]
    batch = (Path(opt.condor_dir).expanduser() if opt.condor_dir else
             Path(opt.output_base)/'condor'/batch_id).resolve()
    # These paths appear in both Condor submit syntax and shell commands.
    if any(c.isspace() or c in '$"' for c in str(batch)):
        raise ValueError('Use a Condor directory without whitespace, dollar signs or quotes.')
    if batch.exists():
        raise ValueError(f'Batch already exists: {batch}; use --stage resume or a new --condor-dir.')
    signals, windows, preflight = {}, {}, {}
    if opt.stage in ('cards', 'all'):
        reader = InputReader()
        try:
            signals, windows = fs.prepare_inputs(reader, args, opt)
            reader.check_inputs()
            preflight = reader.fingerprints
        finally:
            reader.close()
    selected = {}
    for mass in opt.masses:
        if opt.stage in ('cards', 'all'):
            found = next((m for m in signals if abs(m-mass) < 1e-6), None)
            if found is None:
                raise ValueError(f'No simulated signal at M={mass}.')
            selected[base.mass_label(mass)] = {era:str(Path(signals[found][era]).resolve())
                for era in base.target_years(opt.target)}
        else:
            fs.load_model(fs.model_directory(opt, mass), opt)
    jobs = make_jobs(opt, windows)
    batch.mkdir(parents=True)
    (batch/'source').mkdir()
    for name in SOURCES:
        shutil.copy2(Path(__file__).parent/name, batch/'source'/name)
    payload = dict(schema=SCHEMA, batch_id=batch_id, batch=str(batch), options=vars(opt),
        jobs=jobs, windows=windows, signal_files=selected, preflight_inputs=preflight,
        input_base_dir=str(Path(args.base_dir).expanduser().resolve()),
        resolution_coefficients=getattr(args, 'resolution_coefficients', {}),
        cmssw_base=str(cmssw), scram_arch=arch,
        source_hashes={name:file_hash(batch/'source'/name) for name in SOURCES})
    for job in jobs:
        workdir(payload, job).mkdir(parents=True)
    if opt.stage == 'fit':
        destination = argparse.Namespace(**vars(opt))
        destination.output_base = str(batch/'results')
        for mass in opt.masses:
            old, new = fs.model_directory(opt, mass), fs.model_directory(destination, mass)
            new.mkdir(parents=True)
            for name in ('datacard.txt', 'model.json'):
                shutil.copy2(old/name, new/name)
    payload['manifest_sha256'] = digest(payload)
    write_json(batch/'manifest.json', payload)
    wrapper = batch/'run_job.sh'
    wrapper.write_text('\n'.join([
        '#!/bin/bash', 'set -eo pipefail', 'export LC_ALL=C ROOT_HIST=0',
        'export CMS_PATH=/cvmfs/cms.cern.ch', 'source "$CMS_PATH/cmsset_default.sh"',
        f'export SCRAM_ARCH={shlex.quote(arch)}', f'cd {shlex.quote(str(cmssw/"src"))}',
        'eval "$(scramv1 runtime -sh)"',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        f'cd {shlex.quote(str(batch))}',
        f'exec python3 -u {shlex.quote(str(batch/"source/full_spectrum_fit.py"))} '
        f'--condor-worker {shlex.quote(str(batch/"manifest.json"))} --condor-index "$1"', '']))
    wrapper.chmod(0o755)
    submit_lines = [f'executable = {wrapper}', f'initialdir = {batch}', 'universe = vanilla',
        'arguments = $(jobindex)', 'log = condor.log', 'getenv = True',
        'should_transfer_files = NO', 'output = jobs/job_$(jobindex).out',
        'error = jobs/job_$(jobindex).err', 'request_cpus = 1',
        f'request_memory = {opt.condor_memory}MB', f'jobbatchname = NPSFullSpectrum_{batch_id}']
    if opt.condor_requirements:
        submit_lines.append('requirements = '+opt.condor_requirements)
    (batch/'submit.jds').write_text('\n'.join(submit_lines+['queue 1', '']))
    dag = batch/'workflow.dag'
    write_dag(payload, range(len(jobs)), dag)
    counts = Counter(job['kind'] for job in jobs)
    print('[CONDOR] '+', '.join(f'{kind}={count}' for kind,count in counts.items()), flush=True)
    print(f'[CONDOR] batch={batch}\n[CONDOR] results={batch/"results"}', flush=True)
    if opt.condor_dry_run:
        print('[CONDOR] prepared only; submit from the batch directory: '+shlex.join(submission(payload, dag)))
    else:
        base.run_command(submission(payload, dag), batch)
    print(f'[CONDOR] status: python3 full_spectrum_fit.py --stage status --condor-dir {shlex.quote(str(batch))}')
    return batch


def manage(opt):
    if not opt.condor_dir:
        raise ValueError('--stage status/resume/collect requires --condor-dir.')
    payload = load_manifest(opt.condor_dir)
    cache, states, pending = {}, Counter(), []
    bin_totals, bin_done, failures = Counter(), Counter(), []
    for job in payload['jobs']:
        if completed(payload, job, cache):
            state = 'success'
        else:
            pending.append(job['index'])
            path = workdir(payload, job)/'status.json'
            status = json.loads(path.read_text()) if path.exists() else {}
            state = status.get('state', 'pending')
            if state == 'success':
                state = 'stale'
            if state == 'failed':
                failures.append((job, status.get('error', 'See worker stderr.')))
        states[job['kind'], state] += 1
        if job['kind'] == 'prepare':
            bin_totals[job['era']] += len(job['windows'])
            if state == 'success':
                bin_done[job['era']] += len(job['windows'])
    for (kind,state), count in sorted(states.items()):
        print(f'[STATUS] {kind}: {state}={count}')
    for era in bin_totals:
        print(f'[BINS] {era}: {bin_done[era]}/{bin_totals[era]} complete')
    for job, error in failures[:10]:
        print(f'[FAILED] job {job["index"]} {job["kind"]} M{job["mass"]:g}: {error}')
    print(f'[RESULTS] {Path(payload["batch"])/"results"}')
    if opt.stage == 'status':
        for job in payload['jobs']:
            if job['kind'] == 'prepare':
                continue
            status = cache[job['index']]
            if status:
                print(f'[TIME] job {job["index"]} {job["kind"]} M{job["mass"]:g}: {status["elapsed_seconds"]:.1f} s')
        return 0
    if opt.stage == 'collect':
        if pending:
            raise ValueError(f'{len(pending)} incomplete/stale jobs; inspect logs and use --stage resume.')
        print('[COMPLETE] All planned jobs and products verified.')
        return 0
    if not pending:
        print('[COMPLETE] No jobs need resubmission.')
        return 0
    if not opt.condor_dry_run:
        base.require_command('condor_q')
        constraint = 'JobBatchName == "NPSFullSpectrum_'+payload['batch_id']+'"'
        active = subprocess.run(['condor_q', '-constraint', constraint, '-af', 'ClusterId'],
                                check=True, text=True, capture_output=True)
        if active.stdout.strip():
            raise ValueError('This batch still has queued jobs. Wait for completion or remove the old DAG before resuming.')
    dag = Path(payload['batch'])/('resume_'+uuid4().hex[:8]+'.dag')
    write_dag(payload, pending, dag)
    if opt.condor_dry_run:
        print('[RESUME] prepared only: '+shlex.join(submission(payload, dag)))
    else:
        base.run_command(submission(payload, dag), Path(payload['batch']))
    return 0
