#!/usr/bin/env python3
"""Expected limits and distributed r=0/1 impacts from completed blinded models.

Reuse the validated FAKE-shape workspace without rebuilding input estimates.
The DAG has no toy dependency. All nuisance fits are included; only the plotted
ranking is limited to the largest 30 impacts. Shared-storage Condor backend.
"""
import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
import shlex
import shutil
import sys
from uuid import uuid4

import limit_workflow as base
import condor_memory_policy as memory_policy

SOURCES = ('condor_memory_policy.py', 'full_spectrum_inference.py', 'limit_workflow.py',
           'python/plotLimits.py', 'python/plotting.py')
SCHEMA = 'NPS26009_full_spectrum_inference_v1'


def digest(path):
    return base.validation_digest(path)


def manifest_digest(payload):
    return hashlib.sha256(json.dumps({k: v for k, v in payload.items()
        if k != 'manifest_sha256'}, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_manifest(batch, index=None, check_inputs=True):
    batch = Path(batch).resolve()
    payload = json.loads((batch/'manifest.json').read_text())
    if payload.get('schema') != SCHEMA or payload.get('manifest_sha256') != manifest_digest(payload):
        raise ValueError('Invalid or modified inference manifest.')
    if index is not None and (not isinstance(index,int) or not 0 <= index < len(payload['jobs'])):
        raise ValueError('Invalid inference job index.')
    if check_inputs:
        _check_manifest_inputs(batch,payload,index)
    return payload


def _check_manifest_inputs(batch,payload,index=None):
    for name, expected in payload['source_hashes'].items():
        if digest(batch/name) != expected:
            raise ValueError('Submitted source changed: '+name)
    selected = set()
    def include(current):
        if not isinstance(current, int) or not 0 <= current < len(payload['jobs']):
            raise ValueError('Invalid inference job/dependency index.')
        if current in selected:
            return
        selected.add(current)
        row = payload['jobs'][current]
        for parent in row.get('dependencies', []):
            include(parent)
        if 'initial' in row:
            include(row['initial'])
    if index is None:
        selected.update(range(len(payload['jobs'])))
    else:
        include(index)
    checked = {}
    for current in sorted(selected):
        job = payload['jobs'][current]
        for key in ('card', 'workspace'):
            if job[key] in checked and checked[job[key]] != job[key+'_sha256']:
                raise ValueError('Conflicting input fingerprints: '+job[key])
            if job[key] not in checked:
                if digest(job[key]) != job[key+'_sha256']:
                    raise ValueError('Submitted card/workspace changed: '+job[key])
                checked[job[key]] = job[key+'_sha256']


def options(payload, job=None):
    args = base.build_parser().parse_args([])
    vars(args).update(payload['config'])
    if job:
        args.impact_expect_signal = job.get('injected_r', 0.)
        args.r_min = job.get('r_min', 0.)
        if job['kind'] == 'impact_final':
            args.impact_parallel = payload['config'].get('impact_repair_cpus', 1)
    return args


def input_case(path):
    summary = json.loads(path.read_text())
    model_path = path.parent.parent/'model.json'
    model = json.loads(model_path.read_text())
    card = path.parent/'datacard.txt'
    original = path.parent.parent/'datacard.txt'
    workspace = path.parent/'workspace.root'
    if (summary.get('blind') is not True or model.get('blind') is not True
            or summary.get('inject_r') != 0 or model.get('test_only')
            or summary.get('test_only')):
        raise ValueError('Require completed, non-test background-only Asimov results: '+str(path))
    if (summary['mass'] != model['mass'] or summary['target'] != model['target']
            or summary['card_sha256'] != model['card_sha256']
            or digest(original) != model['card_sha256']
            or digest(card) != summary['fit_card_sha256']):
        raise ValueError('Summary, model and card provenance mismatch: '+str(path))
    if 'shapes * * FAKE' not in card.read_text().splitlines():
        raise ValueError('Require the validated compiled-fit card with FAKE shapes.')
    if not workspace.is_file():
        raise ValueError('Missing saved workspace: '+str(workspace))
    for kind in ('fit_b', 'fit_s'):
        fit = summary['fits'][kind]
        audit = fit.get('covariance_audit', {})
        external = (fit['covQual'] == -1 and audit.get('method') == 'robustHesse'
                    and audit.get('validated') is True)
        if fit['status'] != 0 or not (fit['covQual'] == 3 or external):
            raise ValueError('Failed Asimov fit quality: '+str(path))
    # This scale only seeds the signed initial POI range and recovery tolerance.
    # Every impact context still obtains/validates its own profile crossings.
    # An audited HESSE covariance is not relabelled as a profile interval.
    info = summary['fits']['fit_s']
    reference = info.get('r_profile_error_high')
    method = 'positive_profile_error'
    if not isinstance(reference,(int,float)) or not math.isfinite(reference) or reference <= 0:
        reference = info.get('r_covariance_error')
        method = 'audited_covariance_error_range_seed'
    if not isinstance(reference,(int,float)) or not math.isfinite(reference) or reference <= 0:
        import ROOT
        source = ROOT.TFile.Open(str(path.parent/'fit_s/fitDiagnostics.FullSpectrum.root'))
        try:
            fit = source.Get('fit_s') if source and not source.IsZombie() else None
            parameter = fit.floatParsFinal().find('r') if fit else None
            reference = float(parameter.getAsymErrorHi()) if parameter and parameter.hasAsymError() else None
            method = 'saved_positive_profile_error'
        finally:
            if source:
                source.Close()
    if reference is None or not math.isfinite(reference) or reference <= 0:
        raise ValueError('Missing positive profile/covariance range seed: '+str(path))
    safe = min(row['asimov']/row['rates']['sig'] for row in model['rows'] if row['rates']['sig'] > 0)
    unit = base.alpha_unit_from_card(card)
    if (not math.isfinite(safe) or safe <= 0
            or not math.isclose(unit, model['alpha_internal_unit'], rel_tol=1e-12)
            or not math.isclose(unit, summary['alpha_internal_unit'], rel_tol=1e-12)):
        raise ValueError('Invalid nominal Poisson means or signal unit.')
    return model, card, workspace, reference, -.9*safe, method


def initial_job(payload, args, job, directory):
    """Check the signed POI range before releasing any nuisance jobs."""
    card, workspace = Path(job['card']), Path(job['workspace'])
    mass, label = job['mass'], base.mass_label(job['mass'])
    name = f'.Impacts_{base.tag_for(args, job["target"])}_M{label}'
    initial = directory/f'higgsCombine_initialFit_{name}.MultiDimFit.mH{base.format_mass_for_combine(mass)}.root'
    rmin, rmax = job['r_min'], args.r_max
    for attempt in range(args.impact_range_retries+1):
        initial.unlink(missing_ok=True)
        command = base.impact_common_command(args, card, job['target'], label, mass, workspace, rmin, rmax)
        base.run_command([*command, '--doInitialFit'], directory)
        poi = {'POIs': [{'name': 'r', 'fit': base.read_impact_initial_poi(initial)}]}
        centre = poi['POIs'][0]['fit'][1]
        if not math.isfinite(centre) or not math.isclose(centre, args.impact_expect_signal,
                rel_tol=1e-3, abs_tol=1e-3*job.get('planning_reference_error',job.get('reference_positive_profile_error'))):
            raise ValueError('Impact Asimov best fit does not recover the requested injection.')
        boundary, _ = base.impact_fit_reaches_lower_boundary(poi, rmin)
        if boundary:
            expanded = base.expanded_impact_rmin(rmin, job['safe_r_min'])
            if attempt < args.impact_range_retries and expanded < rmin-1e-9:
                rmin = expanded
                continue
            raise ValueError('Impact lower interval reaches the nominal-safe POI boundary; see initial job log.')
        base.repair_impact_initial_poi(args, card, job['target'], label, mass, directory,
            workspace, directory/'initial_poi.json', poi, rmin, rmax)
        context = dict(impact_run_id=uuid4().hex, rmin=rmin, rmax=rmax,
            workspace=str(workspace), workspace_sha256=job['workspace_sha256'],
            initial_file=str(initial), initial_file_sha256=digest(initial))
        context_path = directory/'impact_context.json'
        base.write_condor_status(context_path, context)
        return dict(impact_run_id=context['impact_run_id'], files=[str(initial), str(context_path)])
    raise ValueError('Impact initial range retries exhausted.')


def worker(batch, index):
    payload = read_manifest(batch,index=index,check_inputs=False)
    directory = Path(payload['jobs'][index]['workdir'])
    with (directory/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with base.validation_cache():
                _check_manifest_inputs(Path(batch).resolve(),payload,index)
                return _worker(payload,index)
        except base.WorkflowError as error:
            # Cache exit checks run while the worker lock is still held.
            if str(error).startswith('Input changed during validation:'):
                path = directory/'status.json'
                if path.is_file():
                    status = json.loads(path.read_text())
                    status.update(state='failed', error=str(error))
                    base.write_condor_status(path, status)
            raise


def _worker(payload, index):
    job = payload['jobs'][index]
    args = options(payload, job)
    directory = Path(job['workdir'])
    try:
        base.completed_condor_job(payload, args, job)
        print('[REUSE]', index)
        return
    except base.WorkflowError:
        pass
    status = dict(batch_id=payload['batch_id'], index=index, target=job['target'], mode='blind',
        kind=job['kind'], card_sha256=job['card_sha256'], state='running')
    status_path = directory/'status.json'
    base.write_condor_status(status_path, status)
    try:
        parents = {str(i): base.completed_condor_job(payload, args, payload['jobs'][i])['completion_id']
                   for i in job['dependencies']}
        card = Path(job['card'])
        mass, label = job['mass'], base.mass_label(job['mass'])
        if job['kind'] == 'limits':
            result = base.run_asymptotic(args, Path(job['workspace']), job['target'], label, mass, directory)
            products = dict(expected=str(result.expected), files=[str(result.expected)])
        elif job['kind'] == 'impact_initial':
            products = initial_job(payload, args, job, directory)
        elif job['kind'] == 'impact_nuisance':
            products = base.run_condor_impact_nuisance(payload, args, job, card, label, mass, directory)
        elif job['kind'] == 'impact_final':
            products = base.run_condor_impact_final(payload, args, job, card, label, mass, directory)
        elif job['kind'] == 'limit_plot':
            rows = [payload['jobs'][i] for i in job['dependencies']]
            outputs = [Path(base.completed_condor_job(payload, args, j)['expected']) for j in rows]
            base.collect_and_plot_limits(args, job['target'], [Path(j['card']) for j in rows],
                outputs, outputs, directory)
            prefix = directory/f'limits_{job["target"]}_alpha'
            products = dict(files=[str(prefix.with_suffix(s)) for s in ('.json', '.pdf', '.png')])
        else:
            raise ValueError('Unknown job kind.')
        if any(not Path(p).is_file() or Path(p).stat().st_size == 0 for p in products['files']):
            raise ValueError('Missing/empty inference output.')
        status.update(products, state='success', completion_id=uuid4().hex, dependency_completions=parents)
    except Exception as error:
        status.update(state='failed', error=str(error))
        base.write_condor_status(status_path, status)
        raise
    base.write_condor_status(status_path, status)
    print('[DONE]', job['kind'], job['target'], mass, job.get('injected_r'))


def write_dag(batch, payload, resume=False):
    with base.validation_cache():
        return _write_dag(batch, payload, resume)


def _write_dag(batch, payload, resume=False):
    lines = []
    args = options(payload) if resume else None
    for job in payload['jobs']:
        done = False
        if resume:
            try:
                base.completed_condor_job(payload, args, job)
                done = True
            except base.WorkflowError:
                pass
        cpus = payload['config'].get('impact_repair_cpus', 1) if job['kind'] == 'impact_final' else 1
        node = f'J{job["index"]}'
        lines += [f'JOB {node} submit.jds'+(' DONE' if done else ''),
                  f'VARS {node} jobindex="{job["index"]}" nodecpus="{cpus}"']
        if job['dependencies']:
            lines += ['PARENT '+' '.join(f'J{i}' for i in job['dependencies'])+' CHILD '+node]
    name = 'resume_'+uuid4().hex[:8]+'.dag' if resume else 'workflow.dag'
    (batch/name).write_text('\n'.join(lines)+'\n')
    return name


def submit(opt):
    if (not math.isfinite(opt.r_max) or opt.r_max <= 0 or opt.condor_memory <= 0
            or opt.impact_repair_cpus < 1 or opt.max_jobs < 0 or not opt.impact_injections
            or any(not math.isfinite(x) or x < 0 or x >= opt.r_max for x in opt.impact_injections)):
        raise ValueError('Invalid POI range, injection, memory or concurrency.')
    batch = Path(opt.batch).resolve()
    if batch.exists() or any(c.isspace() or c in '$"' for c in str(batch)):
        raise ValueError('Choose a new batch path without whitespace, quotes or dollar signs.')
    paths = set()
    for root in opt.results or []:
        root = Path(root).resolve()
        if not root.exists():
            raise ValueError('Missing input result path: '+str(root))
        paths.update([root] if root.is_file() else root.rglob('fit_r0/fit_summary.json'))
    if not paths:
        raise ValueError('Supply --results with completed fit_r0 results.')
    cases = [input_case(path) for path in sorted(paths)]
    keys = [(m['target'], m['mass']) for m, *_ in cases]
    if len(set(keys)) != len(keys):
        raise ValueError('More than one model for the same target/mass; select one results batch.')
    cmssw, arch = base.condor_environment(opt)
    batch.mkdir(parents=True)
    source_hashes = {}
    for name in SOURCES:
        destination = batch/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).parent/name, destination)
        source_hashes[str(destination.relative_to(batch))] = digest(destination)
    jobs, limits = [], {}
    for model, source_card, source_workspace, reference, safe, reference_method in cases:
        target, mass = model['target'], model['mass']
        print(f'[RANGE SEED] {target} M{mass:g}: {reference:g} ({reference_method})',flush=True)
        label = base.mass_label(mass)
        directory = batch/'input'/target/f'M{label}'
        directory.mkdir(parents=True)
        card = directory/f'datacard_M-{label}_{target}.txt'
        workspace = directory/'workspace.root'
        shutil.copy2(source_card, card)
        shutil.copy2(source_workspace, workspace)
        shutil.copy2(source_card.parent.parent/'model.json', directory/'model.json')
        shutil.copy2(source_card.parent/'fit_summary.json', directory/'fit_summary.json')
        common = dict(target=target, mass=mass, card=str(card), card_sha256=digest(card),
            workspace=str(workspace), workspace_sha256=digest(workspace),
            original_workspace=str(source_workspace.resolve()), planning_reference_error=reference,
            planning_reference_method=reference_method,
            original_card_sha256=model['card_sha256'], alpha_internal_unit=model['alpha_internal_unit'])
        def add(kind, dependencies=(), **extra):
            index = len(jobs)
            work = batch/'jobs'/f'job_{index:05d}'
            work.mkdir(parents=True)
            job = dict(common, index=index, kind=kind, dependencies=list(dependencies), workdir=str(work), **extra)
            jobs.append(job)
            return job
        if opt.task in ('limits', 'all'):
            limits.setdefault(target, []).append(add('limits')['index'])
        if opt.task in ('impacts', 'all'):
            pois, bounds = base.impact_workspace_model(workspace)
            nuisances = sorted(bounds)
            if not nuisances:
                raise ValueError('No floating nuisance parameters in saved workspace.')
            for injection in dict.fromkeys(opt.impact_injections):
                rmin = max(safe, -5*reference)
                initial = add('impact_initial', injected_r=injection, r_min=rmin, safe_r_min=safe)
                scans = [add('impact_nuisance', [initial['index']], initial=initial['index'],
                    nuisance=n, impact_pois=pois, injected_r=injection, r_min=rmin) for n in nuisances]
                add('impact_final', [initial['index'], *[s['index'] for s in scans]],
                    initial=initial['index'], nuisance_jobs=[s['index'] for s in scans],
                    injected_r=injection, r_min=rmin)
    # Plot limits as soon as limit jobs finish, independently of all impacts.
    for target, indices in limits.items():
        job = dict(jobs[indices[0]], index=len(jobs), kind='limit_plot', dependencies=indices,
            workdir=str(batch/'jobs'/f'job_{len(jobs):05d}'))
        Path(job['workdir']).mkdir()
        jobs.append(job)
    payload = dict(schema=SCHEMA, batch_id=uuid4().hex, blind=True, jobs=jobs, source_hashes=source_hashes,
        config=dict(mode='blind', parameter='alpha', strict=True, r_max=opt.r_max,
                    allow_negative_r=True, impact_profile_strategy=opt.impact_profile_strategy,
                    impact_repair_cpus=opt.impact_repair_cpus, impact_parallel=1, jobs=1,
                    plot_limits_py=str(batch/'source/python/plotLimits.py')))
    payload['manifest_sha256'] = manifest_digest(payload)
    (batch/'manifest.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    wrapper = batch/'run_job.sh'
    wrapper.write_text('\n'.join(['#!/bin/bash', 'set -eo pipefail', 'ulimit -s unlimited',
        'source /cvmfs/cms.cern.ch/cmsset_default.sh', 'export SCRAM_ARCH='+shlex.quote(arch),
        'cd '+shlex.quote(str(cmssw/'src')), 'eval "$(scram runtime -sh)"',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        'exec python3 -u '+shlex.quote(str(batch/'source/full_spectrum_inference.py'))+
        ' --batch '+shlex.quote(str(batch))+' --worker "$1"', '']))
    wrapper.chmod(0o755)
    memory = str(opt.condor_memory)+'MB' if opt.impact_repair_cpus == 1 else f'{opt.condor_memory} * $(nodecpus)'
    (batch/'submit.jds').write_text('\n'.join(['universe = vanilla', f'executable = {wrapper}',
        f'initialdir = {batch}', 'arguments = $(jobindex)', 'should_transfer_files = NO',
        'getenv = True', 'request_cpus = $(nodecpus)', f'request_memory = {memory}',
        'output = jobs/job_$(jobindex).out', 'error = jobs/job_$(jobindex).err',
        'log = condor.log', *memory_policy.retry_lines(str(max(opt.condor_memory * opt.impact_repair_cpus,8192))), 'queue 1', '']))
    dag = write_dag(batch, payload)
    command = ['condor_submit_dag']+(['-maxjobs', str(opt.max_jobs)] if opt.max_jobs else [])+[dag]
    print('[PLAN]', dict(Counter(j['kind'] for j in jobs)))
    print('[BATCH]', batch)
    if not opt.dry_run:
        base.run_command(command, batch, stdout_path=batch/'submit.log')
    else:
        print('[SUBMIT]', shlex.join(command))
    print('[STATUS] python3 full_spectrum_inference.py --stage status --batch '+str(batch))


def report(batch):
    with base.validation_cache():
        payload = read_manifest(batch)
        args = options(payload)
        counts, products = Counter(), []
        for job in payload['jobs']:
            try:
                status = base.completed_condor_job(payload, args, job)
                state = 'success'
                if job['kind'] in ('limit_plot', 'impact_final'):
                    products.append(dict(kind=job['kind'], target=job['target'], mass=job['mass'],
                        injected_r=job.get('injected_r'), files=status['files']))
            except base.WorkflowError:
                path = Path(job['workdir'])/'status.json'
                state = json.loads(path.read_text()).get('state', 'pending') if path.exists() else 'pending'
                if state == 'success':
                    state = 'stale'
            counts[job['kind']+':'+state] += 1
    for key, value in sorted(counts.items()):
        print('[STATUS]', key, value)
    destination = Path(batch)/'inference_summary.json'
    destination.write_text(json.dumps(dict(blind=True, counts=dict(counts), products=products), indent=2)+'\n')
    for product in products:
        print('[PLOT]', product['kind'], 'r='+str(product['injected_r']), *product['files'])
    return payload


def index_condition(indices, variable):
    indices = sorted(set(indices))
    ranges = []
    for index in indices:
        if ranges and index == ranges[-1][1]+1:
            ranges[-1][1] = index
        else:
            ranges.append([index,index])
    return ' || '.join(f'{variable} == {a}' if a == b else
                      f'({variable} >= {a} && {variable} <= {b})' for a,b in ranges) or 'false'


def set_memory(opt):
    """Re-size only nuisance fits and plots; do not change active fits or inputs."""
    if opt.dag_id is None or min(opt.impact_memory, opt.plot_memory, opt.condor_memory) < 1:
        raise ValueError('set-memory requires --dag-id and positive memory values.')
    batch = Path(opt.batch).resolve()
    payload = read_manifest(batch)
    nuisance = [j['index'] for j in payload['jobs'] if j['kind']=='impact_nuisance']
    plots = [j['index'] for j in payload['jobs'] if j['kind']=='limit_plot']
    expression = (f'ifThenElse(({index_condition(nuisance,"$(jobindex)")}), {opt.impact_memory}, '
        f'ifThenElse(({index_condition(plots,"$(jobindex)")}), {opt.plot_memory}, {opt.condor_memory}))')
    if payload['config'].get('impact_repair_cpus', 1) > 1:
        expression = f'({expression}) * $(nodecpus)'
    submit = batch/'submit.jds'
    text, count = re.subn(r'^request_memory\s*=.*$', 'request_memory = '+expression,
                         submit.read_text(), flags=re.MULTILINE)
    if count != 1:
        raise ValueError('Expected exactly one request_memory in submit.jds.')
    ads = json.loads(subprocess.check_output(['condor_q','-constraint',
        f'DAGManJobId == {opt.dag_id} && JobStatus == 1',
        '-attributes','ClusterId,ProcId,DAGNodeName,Iwd','-json'], text=True))
    selected = {**{i:opt.impact_memory for i in nuisance}, **{i:opt.plot_memory for i in plots}}
    ids = []
    for ad in ads:
        match = re.fullmatch(r'J(\d+)',str(ad.get('DAGNodeName','')))
        if not match or int(match.group(1)) not in selected:
            continue
        if ad.get('Iwd') and Path(ad['Iwd']).resolve() != batch:
            raise ValueError('DAG ID does not match the requested batch directory.')
        ids.append((int(ad['ClusterId']),int(ad['ProcId']),selected[int(match.group(1))]))
    edits = []
    for memory in sorted({opt.impact_memory,opt.plot_memory}):
        subset = [(c,p) for c,p,m in ids if m==memory]
        if not subset:
            continue
        # Bound command size even for a newly submitted large impact DAG.
        for start in range(0,len(subset),500):
            condition = ' || '.join(f'(ClusterId == {c} && ProcId == {p})' for c,p in subset[start:start+500])
            edits.append(['condor_qedit','-constraint',f'JobStatus == 1 && ({condition})',
                          'RequestMemory',str(memory)])
    if opt.dry_run:
        print('[MEMORY] would update',len(ids),'idle jobs and future submit requests.')
        return
    backup = submit.with_name('submit.jds.before_impact_memory')
    if not backup.exists():
        shutil.copy2(submit,backup)
    temporary = submit.with_name('submit.jds.'+uuid4().hex+'.tmp')
    temporary.write_text(text)
    temporary.replace(submit)
    for command in edits:
        base.run_command(command,batch)
    print('[MEMORY] Updated',len(ids),'idle jobs and future submissions; running jobs unchanged.')
    print('[MEMORY] nuisance=',opt.impact_memory,'plot=',opt.plot_memory,
          'initial/final/limits=',opt.condor_memory,'MB')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', nargs='+')
    parser.add_argument('--batch', required=True)
    parser.add_argument('--stage', choices=('submit', 'status', 'collect', 'resume', 'set-memory'), default='submit')
    parser.add_argument('--task', choices=('limits', 'impacts', 'all'), default='all')
    parser.add_argument('--impact-repair-cpus', type=int, default=1,
                        help='CPUs for independent retries within each final impact job. Memory scales per CPU.')
    parser.add_argument('--impact-profile-strategy', type=int, choices=(0,1,2), default=0,
                        help='Nominal robust crossing strategy. Use 1 to reproduce the former setting.')
    parser.add_argument('--impact-injections', type=base.parse_float_list, default=[0., 1.])
    parser.add_argument('--r-max', type=float, default=100.)
    parser.add_argument('--condor-memory', type=int, default=1024)
    parser.add_argument('--impact-memory', type=int, default=1024)
    parser.add_argument('--plot-memory', type=int, default=1024)
    parser.add_argument('--dag-id', type=int)
    parser.add_argument('--max-jobs', type=int, default=0, help='DAG concurrency cap; 0 means pool scheduling only')
    parser.add_argument('--condor-cmssw-base')
    parser.add_argument('--condor-scram-arch')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    opt = parser.parse_args(argv)
    if opt.worker is not None:
        worker(opt.batch, opt.worker)
    elif opt.stage == 'set-memory':
        set_memory(opt)
    elif opt.stage == 'submit':
        submit(opt)
    else:
        payload = report(opt.batch)
        if opt.stage == 'resume':
            dag = write_dag(Path(opt.batch).resolve(), payload, resume=True)
            command = ['condor_submit_dag']+(['-maxjobs', str(opt.max_jobs)] if opt.max_jobs else [])+[dag]
            if not opt.dry_run:
                base.run_command(command, Path(opt.batch).resolve())


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, base.WorkflowError) as error:
        print('ERROR:', error, file=sys.stderr)
        raise SystemExit(1)
