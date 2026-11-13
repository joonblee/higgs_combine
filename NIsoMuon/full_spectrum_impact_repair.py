#!/usr/bin/env python3
"""Independent final-impact retries from immutable completed parent scans."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import full_spectrum_inference as inference
import limit_workflow as base


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def run_worker(batch, index):
    plan = json.loads((batch/'plan.json').read_text())
    for name, expected in plan['source_hashes'].items():
        if inference.digest(batch/name) != expected:
            raise ValueError('Repair source changed: '+name)
    manifest = batch/'source_manifest.json'
    if inference.digest(manifest) != plan['manifest_sha256']:
        raise ValueError('Source manifest copy changed')
    payload = json.loads(manifest.read_text())
    job = payload['jobs'][index]
    args = inference.options(payload, job)
    args.impact_step_size = plan['step_size']
    args.impact_crossing_tolerance = plan['crossing_tolerance']
    args.impact_parallel = 1
    args.plot_limits_py = str(batch/'source/python/plotLimits.py')
    directory = batch/f'job_{index:05d}'
    directory.mkdir(exist_ok=False)
    started = time.time()
    status = dict(index=index, mass=job['mass'], injected_r=job.get('injected_r'),
                  state='running', started=started, step_size=args.impact_step_size,
                  crossing_tolerance=args.impact_crossing_tolerance)
    write(directory/'status.json', status)
    try:
        card = Path(job['card'])
        if inference.digest(card) != job['card_sha256']:
            raise ValueError('Source card changed')
        # The common collector validates initial context, parent completion IDs,
        # input hashes and nonempty files. All new outputs stay in directory.
        products = base.run_condor_impact_final(payload, args, job, card,
            base.mass_label(job['mass']), job['mass'], directory)
        status.update(products, state='success')
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        raise
    finally:
        status['elapsed_seconds'] = time.time()-started
        write(directory/'status.json', status)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-batch')
    p.add_argument('--batch', required=True)
    p.add_argument('--indices', default='2221,2983,4524')
    p.add_argument('--step-size', type=float, default=0.1)
    p.add_argument('--crossing-tolerance', type=float, default=1e-4)
    p.add_argument('--memory', type=int, default=16000)
    p.add_argument('--stage', choices=('submit','collect'), default='submit')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    opt = p.parse_args(argv)
    batch = Path(opt.batch).resolve()
    if opt.worker is not None:
        run_worker(batch, opt.worker)
        return
    if opt.stage == 'collect':
        plan = json.loads((batch/'plan.json').read_text())
        counts = Counter()
        for index in plan['indices']:
            path = batch/f'job_{index:05d}'/'status.json'
            status = json.loads(path.read_text()) if path.exists() else dict(state='pending')
            counts[status['state']] += 1
            print(index, json.dumps(status))
        print('[STATUS]', dict(counts))
        return
    base.impact_scan_options(argparse.Namespace(impact_step_size=opt.step_size,
        impact_crossing_tolerance=opt.crossing_tolerance))
    if opt.memory <= 0:
        raise ValueError('Memory must be positive')
    if not opt.source_batch:
        raise ValueError('--source-batch is required for submission')
    source = Path(opt.source_batch).resolve()
    manifest = source/'manifest.json'
    payload = json.loads(manifest.read_text())
    if payload.get('manifest_sha256') != inference.manifest_digest(payload):
        raise ValueError('Source manifest digest mismatch')
    indices = [int(i) for i in opt.indices.split(',')]
    if len(set(indices)) != len(indices):
        raise ValueError('Duplicate indices')
    for index in indices:
        if payload['jobs'][index]['kind'] != 'impact_final':
            raise ValueError('Only impact_final jobs can be retried')
    cmssw = Path(os.environ.get('CMSSW_BASE',''))
    arch = os.environ.get('SCRAM_ARCH','')
    if not arch or not (cmssw/'src').is_dir():
        raise ValueError('Activate the CMSSW environment first')
    batch.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest, batch/'source_manifest.json')
    names = (*inference.SOURCES, 'full_spectrum_impact_repair.py')
    for name in names:
        dest = batch/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).parent/name, dest)
    write(batch/'plan.json', dict(source_batch=str(source), indices=indices,
        step_size=opt.step_size, crossing_tolerance=opt.crossing_tolerance,
        minimizer_tolerance=0.1, first_strategy=1, repair_strategies=[1,2,0],
        manifest_sha256=inference.digest(batch/'source_manifest.json'),
        source_hashes={'source/'+n:inference.digest(batch/'source'/n) for n in names}))
    wrapper = batch/'run.sh'
    wrapper.write_text('\n'.join(['#!/bin/bash','set -eo pipefail','ulimit -s unlimited',
        'source /cvmfs/cms.cern.ch/cmsset_default.sh',
        'export SCRAM_ARCH='+shlex.quote(arch),
        'cd '+shlex.quote(str(cmssw/'src')), 'eval "$(scramv1 runtime -sh)"',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        'exec python3 -u '+shlex.quote(str(batch/'source/full_spectrum_impact_repair.py'))+
        ' --batch '+shlex.quote(str(batch))+' --worker "$1"','']))
    wrapper.chmod(0o755)
    submit = batch/'submit.jds'
    submit.write_text('\n'.join(['universe = vanilla',f'executable = {wrapper}',
        f'initialdir = {batch}','arguments = $(index)',f'request_memory = {opt.memory}MB',
        'request_cpus = 1','getenv = True','should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT','transfer_output_files = ""',
        'output = repair_$(index).out','error = repair_$(index).err','log = condor.log',
        'jobbatchname = NPSImpactCrossing_'+batch.name,
        'queue index in '+','.join(map(str,indices)),'']))
    print('[PLAN]',len(indices),'independent final-impact repair jobs; original batch unchanged',flush=True)
    if not opt.dry_run:
        subprocess.run(['condor_submit',str(submit)],cwd=batch,check=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, base.WorkflowError) as exc:
        print('ERROR:',exc,file=sys.stderr)
        raise SystemExit(1)
