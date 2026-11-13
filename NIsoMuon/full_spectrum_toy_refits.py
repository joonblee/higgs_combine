#!/usr/bin/env python3
"""Audit/re-fit saved conditional toys without regenerating or changing datasets.

The default comparison refits every original chunk with strategy 2 and tighter
minimisation. A failed-only selection refits chunks containing failures (and
therefore also the successful control toys inside those chunks). Originals and
comparison fits remain separate. This checks numerical stability, not coverage.
"""
import argparse
import json
import math
from pathlib import Path
import shlex
import shutil
import sys
import tarfile

import numpy as np
import full_spectrum_toys as toys
import limit_workflow as base

SOURCES = ('full_spectrum_toy_refits.py', *toys.SOURCES)


def valid(entry):
    return (entry.get('status') == 0 and isinstance(entry.get('r'), (int, float))
            and math.isfinite(entry['r']))


def record(directory):
    path = directory/'result/result.json'
    if path.is_file():
        return json.loads(path.read_text())
    archive = directory/'result.tar.gz'
    if archive.is_file():
        with tarfile.open(archive, 'r:gz') as stream:
            try:
                data = stream.extractfile('result/result.json')
            except KeyError:
                return None
            return json.loads(data.read()) if data else None
    return None


def original(batch):
    batch = Path(batch).resolve()
    payload = json.loads((batch/'manifest.json').read_text())
    if payload.get('manifest_sha256') != toys.manifest_digest(payload):
        raise ValueError('Original toy manifest changed.')
    rows, failures = [], []
    for task in payload['tasks']:
        directory = batch/'jobs'/f'job_{task["index"]:05d}'
        item = record(directory)
        if item is None or item.get('task') != task or len(item['entries']) != task['count']:
            raise ValueError('Missing/mismatched original chunk: '+str(directory))
        rows.append(dict(task=task, entries=item['entries']))
        for i, entry in enumerate(item['entries'], 1):
            if not valid(entry):
                failures.append(dict(chunk=task['index'], entry=i, mass=task['mass'],
                    injected_r=task['injected_r'], seed=task['seed'], **entry))
    return payload, rows, failures


def ensembles(rows):
    groups = {}
    for row in rows:
        task = row['task']
        key = (task['mass'], task['injected_r'], task['truth_sha256'])
        group = groups.setdefault(key, dict(mass=task['mass'], injected_r=task['injected_r'],
            truth=task['truth_label'], r_min=task['r_min'], r_max=task['r_max'], entries=[]))
        group['entries'].extend(row['entries'])
    results = []
    for group in groups.values():
        entries = group.pop('entries')
        values = [entry['r'] for entry in entries if valid(entry)]
        result = dict(group, recorded=len(entries), successful=len(values), failed=len(entries)-len(values),
            boundary_fits=sum(min(abs(v-group['r_min']), abs(v-group['r_max'])) < 1e-4 for v in values))
        if values:
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
            result.update(mean_r=float(np.mean(values)), bias=float(np.mean(values))-group['injected_r'],
                empirical_std=sd, bias_standard_error=sd/math.sqrt(len(values)) if sd is not None else None,
                mean_is_conditional_on_fit_success=len(values) != len(entries))
        results.append(result)
    return results


def extract_toys(source, task, destination):
    directory = source/'jobs'/f'job_{task["index"]:05d}'
    unpacked = list((directory/'result').glob('higgsCombine.Truth.GenerateOnly.mH*.root'))
    if len(unpacked) == 1:
        shutil.copy2(unpacked[0], destination)
        return
    with tarfile.open(directory/'result.tar.gz', 'r:gz') as stream:
        members = [m for m in stream.getmembers() if m.isfile()
            and m.name.startswith('result/higgsCombine.Truth.GenerateOnly.mH') and m.name.endswith('.root')]
        if len(members) != 1:
            raise ValueError('Need exactly one saved GenerateOnly file in chunk '+str(task['index']))
        with stream.extractfile(members[0]) as src, destination.open('wb') as dst:
            shutil.copyfileobj(src, dst)


def fit_command(task, workspace, dataset):
    _, command = toys.commands(task, workspace, workspace, dataset)
    command[command.index('--cminDefaultMinimizerStrategy')+1] = '2'
    command += ['--cminDefaultMinimizerTolerance', '0.0001',
        '--cminFallbackAlgo', 'Minuit2,Migrad,1:0.0001',
        '--X-rtd', 'MINIMIZER_MaxCalls=5000000']
    return command


def read_fits(path, count):
    import ROOT
    source = ROOT.TFile.Open(str(path))
    try:
        tree = source.Get('tree_fit_sb') if source and not source.IsZombie() else None
        if not tree or tree.GetEntries() != count:
            raise ValueError('Unexpected number of comparison toy fits.')
        entries = []
        for entry in tree:
            value = float(entry.r)
            entries.append(dict(r=value if math.isfinite(value) else None, status=int(entry.fit_status)))
        return entries
    finally:
        if source:
            source.Close()


def load(batch):
    batch = Path(batch).resolve()
    payload = json.loads((batch/'manifest.json').read_text())
    if payload.get('schema') != 'NPS26009_saved_toy_refits_v1' or payload.get('manifest_sha256') != toys.manifest_digest(payload):
        raise ValueError('Invalid/modified saved-toy refit manifest.')
    return payload


def worker(path, index, destination):
    root = Path(path).resolve().parent
    payload = load(root)
    task = payload['tasks'][index]
    def locate(name):
        return root/name if (root/name).is_file() else root/Path(name).name
    for name, fingerprint in {**payload['source_hashes'], **task['inputs']}.items():
        if toys.digest(locate(name)) != fingerprint:
            raise ValueError('Changed refit input: '+name)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError('Comparison output is not empty; choose a new batch.')
    command = fit_command(task['original_task'], locate(task['workspace']), locate(task['dataset']))
    (destination/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    base.run_command(command, destination, stdout_path=destination/'fit.log')
    entries = read_fits(destination/'fitDiagnostics.Bias.root', task['original_task']['count'])
    result = dict(task=task, entries=entries, comparison='same saved datasets, strategy 2, tolerance 1e-4')
    (destination/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


def prepare(opt):
    with base.validation_cache():
        return _prepare(opt)


def _prepare(opt):
    source = Path(opt.source_batch).resolve()
    parent, rows, failures = original(source)
    selected = {f['chunk'] for f in failures}
    chosen = rows if opt.selection == 'all' else [r for r in rows if r['task']['index'] in selected]
    if not chosen:
        raise ValueError('No chunks selected.')
    batch = Path(opt.batch).resolve()
    if batch.exists() or any(c.isspace() or c in '$"' for c in str(batch)):
        raise ValueError('Choose a new batch path without whitespace, quotes or dollar signs.')
    cmssw, arch = base.condor_environment(opt) if opt.backend == 'condor' else (None, None)
    for name in ('source', 'input', 'jobs'):
        (batch/name).mkdir(parents=True, exist_ok=True)
    source_hashes = {}
    for name in SOURCES:
        shutil.copy2(Path(__file__).parent/name, batch/'source'/name)
        source_hashes['source/'+name] = toys.digest(batch/'source'/name)
    tasks = []
    for row in chosen:
        task = row['task']
        workspace = source/task['fit_workspace']
        if toys.digest(workspace) != task['inputs'][task['fit_workspace']]:
            raise ValueError('Original fit workspace changed.')
        copied = batch/'input'/workspace.name
        if not copied.exists():
            shutil.copy2(workspace, copied)
        dataset = batch/'input'/f'toys_{task["index"]:05d}.root'
        extract_toys(source, task, dataset)
        current = dict(index=len(tasks), original_task=task, original_entries=row['entries'],
            workspace=str(copied.relative_to(batch)), dataset=str(dataset.relative_to(batch)))
        current['inputs'] = {name: toys.digest(batch/name) for name in (current['workspace'], current['dataset'])}
        tasks.append(current)
        (batch/'jobs'/f'job_{current["index"]:05d}').mkdir()
    payload = dict(schema='NPS26009_saved_toy_refits_v1', blind=True, source_batch=str(source),
        original_manifest_sha256=parent['manifest_sha256'], original_rows=rows, selection=opt.selection,
        tasks=tasks, source_hashes=source_hashes)
    payload['manifest_sha256'] = toys.manifest_digest(payload)
    (batch/'manifest.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    if opt.backend == 'local':
        for task in tasks:
            worker(batch/'manifest.json', task['index'], batch/'jobs'/f'job_{task["index"]:05d}'/'result')
        collect(batch)
        return
    wrapper = batch/'run_job.sh'
    wrapper.write_text('\n'.join(['#!/bin/bash', 'set -eo pipefail', 'ulimit -s unlimited',
        'job_scratch="$PWD"', 'source /cvmfs/cms.cern.ch/cmsset_default.sh',
        'export SCRAM_ARCH='+shlex.quote(arch), 'cd '+shlex.quote(str(cmssw/'src')),
        'eval "$(scram runtime -sh)"', 'cd "$job_scratch"',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1', 'mkdir -p result', 'set +e',
        'python3 full_spectrum_toy_refits.py --worker manifest.json --index "$1"',
        'job_status=$?', 'set -e', 'tar -czf result.tar.gz result', 'exit "$job_status"', '']))
    wrapper.chmod(0o755)
    transfer = [str(batch/'source'/name) for name in SOURCES]+[str(batch/'manifest.json')]
    (batch/'submit.jds').write_text('\n'.join(['universe = vanilla', f'executable = {wrapper}',
        f'initialdir = {batch}', 'arguments = $(taskid)', 'request_cpus = 1',
        f'request_memory = {opt.condor_memory}MB', 'should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT', 'transfer_input_files = '+','.join(transfer)+',$(fitfile),$(toyfile)',
        'transfer_output_files = result.tar.gz', 'transfer_output_remaps = "result.tar.gz=$(destination)"',
        'output = jobs/job_$(taskid).out', 'error = jobs/job_$(taskid).err', 'log = condor.log',
        'queue taskid,fitfile,toyfile,destination from tasks.txt', '']))
    (batch/'tasks.txt').write_text('\n'.join(f'{t["index"]} {batch/t["workspace"]} {batch/t["dataset"]} '
        f'{batch}/jobs/job_{t["index"]:05d}/result.tar.gz' for t in tasks)+'\n')
    print(f'[PLAN] {len(tasks)} chunks; {sum(t["original_task"]["count"] for t in tasks)} saved-toy refits; no generation.')
    if not opt.dry_run:
        base.run_command(['condor_submit', 'submit.jds'], batch, stdout_path=batch/'submit.log')
    print('[BATCH]', batch)


def collect(batch):
    batch = Path(batch).resolve()
    payload = load(batch)
    merged = {r['task']['index']: r for r in payload['original_rows']}
    pending, pairs, recovered, regressions = [], [], 0, 0
    for task in payload['tasks']:
        item = record(batch/'jobs'/f'job_{task["index"]:05d}')
        if item is None:
            pending.append(task['index'])
            continue
        if item.get('task') != task or len(item['entries']) != task['original_task']['count']:
            raise ValueError('Stale/malformed comparison result.')
        old = task['original_task']
        merged[old['index']] = dict(task=old, entries=item['entries'])
        for i, (before, after) in enumerate(zip(task['original_entries'], item['entries']), 1):
            recovered += not valid(before) and valid(after)
            regressions += valid(before) and not valid(after)
            if valid(before) and valid(after):
                pairs.append(dict(mass=old['mass'], injected_r=old['injected_r'], chunk=old['index'],
                    entry=i, before=before['r'], after=after['r'], delta_r=after['r']-before['r']))
    paired_groups = {}
    for pair in pairs:
        paired_groups.setdefault((pair['mass'], pair['injected_r']), []).append(pair['delta_r'])
    changes = []
    for (mass, injection), values in paired_groups.items():
        changes.append(dict(mass=mass, injected_r=injection, count=len(values),
            mean_delta_r=float(np.mean(values)), max_abs_delta_r=float(np.max(np.abs(values))),
            delta_standard_error=float(np.std(values, ddof=1)/math.sqrt(len(values))) if len(values)>1 else None))
    output = batch/'toy_refit_summary.json'
    result = dict(blind=True, selection=payload['selection'], pending_chunks=pending,
        recovered_failed_fits=recovered, newly_failed_fits=regressions,
        original_ensembles=ensembles(payload['original_rows']),
        refitted_ensembles=ensembles(list(merged.values())) if not pending else None,
        paired_changes=changes, paired_entries=pairs,
        note='Same saved datasets, ranges and nuisance model. Selected chunks use ALL refitted entries, '
             'including regressions; no selection by desired bias. Means are conditional on successful fits. '
             'Failed-only chunk controls are selected by failure occurrence. Not a coverage/pull test.')
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('[REFITS] pending=', len(pending), 'recovered=', recovered, 'newly_failed=', regressions)
    print('[SAVED]', output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('audit', 'refit', 'collect'), default='audit')
    parser.add_argument('--source-batch')
    parser.add_argument('--batch')
    parser.add_argument('--selection', choices=('failed', 'all'), default='all')
    parser.add_argument('--backend', choices=('condor', 'local'), default='condor')
    parser.add_argument('--condor-memory', type=int, default=16000)
    parser.add_argument('--condor-cmssw-base')
    parser.add_argument('--condor-scram-arch')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--index', type=int, help=argparse.SUPPRESS)
    opt = parser.parse_args(argv)
    if opt.worker:
        worker(Path(opt.worker), opt.index, Path.cwd()/'result')
    elif opt.stage == 'collect':
        if not opt.batch:
            raise ValueError('--batch is required.')
        collect(opt.batch)
    else:
        if not opt.source_batch:
            raise ValueError('--source-batch is required.')
        _, rows, failures = original(opt.source_batch)
        if opt.stage == 'audit':
            print(json.dumps(dict(failed_fits=failures, ensembles=ensembles(rows)), indent=2, allow_nan=False))
        else:
            if not opt.batch:
                raise ValueError('Choose a new --batch for comparison refits.')
            if opt.dry_run and opt.backend == 'local':
                raise ValueError('--dry-run requires Condor backend.')
            prepare(opt)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, base.WorkflowError) as error:
        print('ERROR:', error, file=sys.stderr)
        raise SystemExit(1)
