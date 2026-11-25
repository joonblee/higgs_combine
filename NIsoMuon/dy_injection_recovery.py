#!/usr/bin/env python3
"""Fit DY-shifted background pseudo-data with the nominal Run-correlated model."""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

INJECTIONS = (0.8, 1.0, 1.2)
MAJOR = ('QCD', 'DY', 'tt')
SOURCE_FILES = ('full_spectrum_fit.py', 'limit_workflow.py',
                'limit_workflow_sideband_fit.py', 'dy_closure_shape.py')
LIMIT_KEYS = ('exp-2', 'exp-1', 'exp0', 'exp+1', 'exp+2')


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate_card(model, card):
    from limit_workflow_sideband_fit import nominal_predictions

    lines = [line.split() for line in card.splitlines()]
    observations = [line[1:] for line in lines if line[:1] == ['observation']]
    bins = [line[1:] for line in lines if line[:1] == ['bin']]
    rows = model['rows']
    channels = [row['channel'] for row in rows]
    asimov = [row['asimov'] for row in rows]
    if len(set(channels)) != len(channels) or len(bins) != 2 or bins[0] != channels:
        raise ValueError('Saved model and card channels differ.')
    if len(observations) != 1 or list(map(float, observations[0])) != asimov:
        raise ValueError('Datacard observations must be the nominal background Asimov counts.')
    nominal = nominal_predictions(card)
    if any(not math.isclose(nominal[c], y, rel_tol=1e-10, abs_tol=1e-12)
           for c, y in zip(channels, asimov)):
        raise ValueError('Datacard nominal prediction differs from the background Asimov counts.')


def mass_list(text):
    if text == 'all':
        return None
    values = tuple(map(float, text.split(',')))
    if any(not math.isfinite(mass) or mass <= 0 for mass in values):
        raise argparse.ArgumentTypeError('Use positive finite masses or all.')
    return set(values)


def find_inputs(source, target, masses):
    import full_spectrum_fit as fs
    from limit_workflow import mass_label

    source = source.resolve(strict=True)
    manifest = source / 'manifest.json'
    if manifest.is_file():
        mode = read_json(manifest).get('options', {}).get('mode')
        if mode not in (None, 'blind'):
            raise ValueError('The source batch must be blinded.')
    results = source / 'results'
    if results.is_dir():
        source = results
    found = {}
    requested = None if masses is None else {mass_label(mass) for mass in masses}
    for path in sorted(source.rglob('model.json')):
        if re.fullmatch(r'M[0-9]+(?:p[0-9]+)?', path.parent.name) is None:
            continue
        label = path.parent.name[1:]
        if requested is not None and label not in requested:
            continue
        model = read_json(path)
        if model.get('target') != target:
            continue
        mass = float(model['mass'])
        if (model.get('blind') is not True or model.get('test_only')
                or model.get('schema_version') != fs.SCHEMA or mass_label(mass) != label):
            raise ValueError(f'Need a blinded production full-spectrum model: {path}.')
        if mass in found:
            raise ValueError(f'Duplicate M{mass:g} under {source}. Select a single fit layout.')
        found[mass] = (path.parent, model)
    if not found or (requested is not None and requested != {mass_label(mass) for mass in found}):
        raise ValueError(f'Missing input models for {target} under {source}. '
                         f'Found masses: {sorted(found)}.')
    reference = next(iter(found.values()))[1]
    for _, model in found.values():
        if (model.get('signature') != reference.get('signature')
                or model.get('fit_kind') != reference.get('fit_kind')
                or set(model['eras']) != set(reference['eras'])):
            raise ValueError('Input models use inconsistent fit configurations.')
    return found


def fit_options(config):
    return argparse.Namespace(mode='blind', inject_r=0, r_max=config['r_max'],
                              fit_components='b-only', fit_output='norms',
                              fit_strategy=config['fit_strategy'],
                              fit_max_calls=config['fit_max_calls'],
                              fit_robust_hesse=config['robust_hesse'])


def worker(batch, index, *, regenerate=False):
    import full_spectrum_fit as fs

    manifest = read_json(batch / 'manifest.json')
    job = manifest['jobs'][index]
    directory = batch / job['directory']
    with (batch / 'jobs' / f'{index}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = batch / 'jobs' / f'{index}.done'
        if marker.is_file() and not regenerate:
            return
        print(f"[WORKER] {job['kind']} {job['directory']}", flush=True)
        if job['kind'].startswith('collect_'):
            collect(batch, manifest, job['kind'])
        else:
            model = read_json(directory / 'model.json')
            if model['blind'] is not True or model['card_sha256'] != digest(
                    (directory / 'datacard.txt').read_text()):
                raise ValueError('Worker requires an intact blinded card.')
            opt = fit_options(manifest['config'])
            if job['kind'] == 'workspace':
                fs.prepare_workspace(model, directory, opt)
            elif job['kind'] == 'fit_b':
                opt.mode = 'unblind'  # Read the synthetic observation stored in this workspace.
                fs.run_fit_component(model, directory, opt, 'fit_b')
            elif job['kind'] == 'limits':
                fs.run_expected_limits(model, directory, opt, condition_on_pseudodata=True)
            else:
                raise ValueError(f"Unknown worker task {job['kind']}.")
        marker.touch()


def write_table(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def injected_card(model, card, factor):
    from limit_workflow import run_group

    validate_card(model, card)
    for process in MAJOR:
        expected = {frozenset(e for e in model['eras'] if run_group(e) == group)
                    for group in {run_group(e) for e in model['eras']}}
        groups = [frozenset(v['eras']) for v in model['normalisations'].values()
                  if v['process'] == process]
        if len(groups) != len(expected) or set(groups) != expected:
            raise ValueError(f'{process} normalisations must be shared within each Run.')
    if factor == 1:
        return card
    counts = [row['asimov'] + (factor - 1) * row['rates']['DY'] for row in model['rows']]
    if any(not math.isfinite(n) or n < 0 for n in counts):
        raise ValueError('Invalid injected observation.')
    return re.sub(r'^([ \t]*observation)[ \t]+[^\n]+', lambda m:
                  m[1] + ' ' + ' '.join(format(n, '.17g') for n in counts),
                  card, count=1, flags=re.MULTILINE)


def collect(batch, manifest, kind):
    norms, limits = [], []
    for case in manifest['cases']:
        directory = batch / case['directory']
        labels = dict(mass=case['mass'], injected_DY=case['injection'])
        if kind == 'collect_fits':
            fit = read_json(directory / 'fit_r0/fit_b/fit_summary.json')['fits']['fit_b']
            for name, info in fit['normalisations'].items():
                norms.append(dict(**labels, run=info['run'], process=info['process'],
                                  nuisance=name, scale=info['scale'], low=info['scale_low'],
                                  high=info['scale_high'], status=fit['status'], covQual=fit['covQual']))
        else:
            values = read_json(directory / 'fit_r0/limits/expected_limits.json')['alpha']
            limits.append(dict(**labels, **{key: values[key] for key in LIMIT_KEYS}))
    if norms:
        write_table(batch / 'postfit_normalisations.csv', norms)
    if limits:
        baseline = {r['mass']: r['exp0'] for r in limits if r['injected_DY'] == 1}
        for row in limits:
            row['median_over_nominal'] = row['exp0'] / baseline[row['mass']]
        write_table(batch / 'expected_limits.csv', limits)


def stage_batch(args):
    from limit_workflow import mass_label

    batch = args.batch.resolve()
    if batch.exists():
        raise ValueError(f'Batch already exists: {batch}. Choose a new directory.')
    cmssw_base = args.cmssw_base or os.environ.get('CMSSW_BASE')
    arch = args.scram_arch or os.environ.get('SCRAM_ARCH')
    if not cmssw_base or not arch or not (Path(cmssw_base) / 'src').is_dir():
        raise ValueError('Activate CMSSW first, or provide --cmssw-base and --scram-arch.')
    cmssw = Path(cmssw_base).resolve()
    if not args.dry_run and not shutil.which('condor_submit_dag'):
        raise ValueError('condor_submit_dag is unavailable on this host.')

    inputs = find_inputs(args.source, args.target, args.masses)
    prepared = []
    for mass, (source, model) in sorted(inputs.items()):
        card = (source / 'datacard.txt').read_text()
        if digest(card) != model['card_sha256']:
            raise ValueError(f'Source datacard hash mismatch: {source}')
        for factor in INJECTIONS:
            variant = injected_card(model, card, factor)
            case = dict(mass=mass, injection=factor,
                        directory=f'results/M{mass_label(mass)}/DY{factor:.1f}')
            changed = dict(model, card_sha256=digest(variant), injected_DY=factor,
                           source_card_sha256=model['card_sha256'])
            prepared.append((case, changed, variant))

    snapshot = batch / 'source'
    snapshot.mkdir(parents=True)
    (batch / 'jobs').mkdir()
    script = Path(__file__).resolve()
    for name in (script.name, *SOURCE_FILES):
        shutil.copy2(script.with_name(name), snapshot / name)

    config = dict(target=args.target, r_max=args.r_max,
                  fit_strategy=args.fit_strategy, fit_max_calls=args.fit_max_calls,
                  robust_hesse=args.robust_hesse)
    jobs = []

    def add_job(kind, directory, memory, parents=()):
        index = len(jobs)
        jobs.append(dict(kind=kind, directory=directory, memory=memory, parents=list(parents)))
        return index

    for case, model, card in prepared:
        directory = batch / case['directory']
        directory.mkdir(parents=True)
        (directory / 'datacard.txt').write_text(card)
        write_json(directory / 'model.json', model)
        parent = add_job('workspace', case['directory'], args.workspace_memory)
        add_job('fit_b', case['directory'], args.fit_memory, (parent,))
        if not args.no_limits:
            add_job('limits', case['directory'], args.limit_memory, (parent,))
    for task, collector in [('fit_b', 'collect_fits'), ('limits', 'collect_limits')]:
        parents = [i for i, job in enumerate(jobs) if job['kind'] == task]
        if parents:
            add_job(collector, '.', args.limit_memory, parents)
    write_json(batch / 'manifest.json', dict(config=config, jobs=jobs,
                                             cases=[case for case, _, _ in prepared]))

    wrapper = batch / 'run_job.sh'
    wrapper.write_text('\n'.join([
        '#!/bin/bash', 'set -eo pipefail', 'ulimit -s unlimited',
        'export LC_ALL=C ROOT_HIST=0',
        'source /cvmfs/cms.cern.ch/cmsset_default.sh',
        f'export SCRAM_ARCH={shlex.quote(arch)}',
        f'cd {shlex.quote(str(cmssw / "src"))}', 'eval "$(scram runtime -sh)"',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        f'exec python3 -u {shlex.quote(str(snapshot / script.name))} worker '
        f'--batch {shlex.quote(str(batch))} --index "$1"', '',
    ]))
    wrapper.chmod(0o755)
    requirements = f'requirements = {args.requirements}\n' if args.requirements else ''
    (batch / 'submit.jds').write_text(
        'universe = vanilla\nexecutable = run_job.sh\narguments = $(idx)\n'
        'should_transfer_files = YES\nwhen_to_transfer_output = ON_EXIT\n'
        'transfer_output_files = ""\nrequest_cpus = 1\nrequest_memory = $(memory)MB\n'
        'output = jobs/$(idx).out\nerror = jobs/$(idx).err\nlog = jobs/events.log\n'
        f'{requirements}queue 1\n')
    dag = []
    for i, job in enumerate(jobs):
        dag.extend([f'JOB J{i} submit.jds', f'VARS J{i} idx="{i}" memory="{job["memory"]}"'])
        if args.retries:
            dag.append(f'RETRY J{i} {args.retries}')
        if job['parents']:
            names = ' '.join(f'J{j}' for j in job['parents'])
            dag.append(f'PARENT {names} CHILD J{i}')
    (batch / 'workflow.dag').write_text('\n'.join(dag) + '\n')
    counts = {kind: sum(job['kind'] == kind for job in jobs)
              for kind in ('workspace', 'fit_b', 'limits', 'collect_fits', 'collect_limits')}
    print(f'[PLAN] {batch} {counts}', flush=True)
    command = ['condor_submit_dag']
    if args.max_jobs:
        command += ['-maxjobs', str(args.max_jobs)]
    command.append('workflow.dag')
    if args.dry_run:
        print(f'[DRY RUN] cd {shlex.quote(str(batch))} && {shlex.join(command)}')
    else:
        subprocess.run(command, cwd=batch, check=True)


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Use a positive finite number.')
    return value


def nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError('Use a nonnegative integer.')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    submit = commands.add_parser('submit', help='Stage and submit the comparison DAG.')
    submit.add_argument('--source', type=Path, required=True, help='Existing blind batch or results directory.')
    submit.add_argument('--target', choices=('Run2Run3', 'Run2', 'Run3'), default='Run2Run3')
    submit.add_argument('--masses', type=mass_list, default={12., 40., 70.})
    submit.add_argument('--no-limits', action='store_true')
    submit.add_argument('--r-max', type=positive_float, default=100.)
    submit.add_argument('--fit-strategy', type=int, choices=(0, 1, 2), default=1)
    submit.add_argument('--fit-max-calls', type=nonnegative_int)
    submit.add_argument('--robust-hesse', action='store_true', help='Apply the existing audited robustHesse workflow.')
    submit.add_argument('--cmssw-base')
    submit.add_argument('--scram-arch')
    submit.add_argument('--requirements', default='', help='Optional Condor ClassAd expression.')
    submit.add_argument('--workspace-memory', type=int, default=2048, metavar='MB')
    submit.add_argument('--fit-memory', type=int, default=4096, metavar='MB')
    submit.add_argument('--limit-memory', type=int, default=2048, metavar='MB')
    submit.add_argument('--max-jobs', type=nonnegative_int, default=0, help='DAG throttle; 0 leaves it unset.')
    submit.add_argument('--retries', type=nonnegative_int, default=1)
    submit.add_argument('--dry-run', action='store_true', help='Write the DAG without submitting it.')
    status = commands.add_parser('status', help='Show completed jobs and pending job IDs.')
    summary = commands.add_parser('collect', help='Regenerate tables and plots once all analysis jobs finish.')
    internal = commands.add_parser('worker', help='Run one DAG node.')
    internal.add_argument('--index', type=nonnegative_int, required=True)
    for command in (submit, status, summary, internal):
        command.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == 'submit':
        if min(args.workspace_memory, args.fit_memory, args.limit_memory) < 1:
            parser.error('Memory requests must be positive.')
        if any(char in args.requirements for char in '\r\n'):
            parser.error('--requirements must be a single line.')
        stage_batch(args)
        return

    batch = args.batch.resolve(strict=True)
    if args.command == 'worker':
        worker(batch, args.index)
    else:
        manifest = read_json(batch / 'manifest.json')
        if args.command == 'collect':
            for index, job in enumerate(manifest['jobs']):
                if job['kind'].startswith('collect_'):
                    worker(batch, index, regenerate=True)
        else:
            for kind in ('workspace', 'fit_b', 'limits', 'collect_fits', 'collect_limits'):
                ids = [i for i, job in enumerate(manifest['jobs']) if job['kind'] == kind]
                pending = [i for i in ids if not (batch / 'jobs' / f'{i}.done').is_file()]
                print(f'{kind}: {len(ids) - len(pending)}/{len(ids)} complete')
                for i in pending:
                    print(f"  J{i} {manifest['jobs'][i]['directory']}  jobs/{i}.err")


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f'Error: {error}')
