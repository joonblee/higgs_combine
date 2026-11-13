#!/usr/bin/env python3
"""Split invalid impact nuisance scans into independent Condor DAG nodes."""
import argparse
import fcntl
from collections import Counter
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

SOURCES = (*inference.SOURCES, 'full_spectrum_impact_repair_dag.py')


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def pending_scans(data, bounds):
    missing, bad = base.impact_interval_issues(data, bounds)
    params = {p['name']:p for p in data['params']}
    special = {n for n, sides in bad.items()
        if (sides == ['lower'] and '_bckgShape_QCD_' in n and bounds[n][0] == 0.)
        or base.tiny_gaussian_impact_candidate(params[n], bounds[n])}
    return sorted(set(missing) | (set(bad)-special))


def case_options(payload, case, batch):
    job = payload['jobs'][case['source_index']]
    args = inference.options(payload, job)
    args.impact_parallel = 1
    args.impact_step_size = .1
    args.impact_crossing_tolerance = 1e-4
    args.plot_limits_py = str(batch/'source/python/plotLimits.py')
    return job, args


def check_case(payload, case, args, task=None):
    job = payload['jobs'][case['source_index']]
    context = base.condor_impact_context(payload, args, job['initial'])
    if context != case['context']:
        raise ValueError('Initial fit context changed')
    selected = set(case['parents'])
    if task and task['kind'] == 'scan':
        selected = {str(job['initial']),str(case['nuisance_parents'][task['nuisance']])}
    for index in selected:
        expected = case['parents'][index]
        status = base.completed_condor_job(payload, args, payload['jobs'][int(index)])
        if status['completion_id'] != expected:
            raise ValueError('Completed parent scan changed')


def scan_case(case, args, task, directory):
    """One NP per job; only this NP is repeated on a failed interval audit."""
    nuisance = task['nuisance']
    context = case['context']
    shutil.copy2(context['initial_file'], directory/Path(context['initial_file']).name)
    mass = case['mass']; label = base.mass_label(mass)
    name = f'.Impacts_{base.tag_for(args,case["target"])}_M{label}'
    mass_arg = base.format_mass_for_combine(mass)
    output = directory/f'higgsCombine_paramFit_{name}_{nuisance}.MultiDimFit.mH{mass_arg}.root'
    for strategy in (1,2,0):
        if output.exists():
            shutil.copy2(output, directory/f'previous_strategy{strategy}.root')
            output.unlink()
        command = base.impact_common_command(args, Path(case['card']), case['target'],
            label, mass, Path(context['workspace']), context['rmin'], context['rmax'])
        for flag in ('--cminDefaultMinimizerStrategy','--setRobustFitStrategy'):
            command[command.index(flag)+1] = str(strategy)
        code = base.run_command([*command,'--doFits','--named',nuisance], directory,
            allow_failure=True, stdout_path=directory/f'fit_strategy{strategy}.log')
        if code or not output.is_file():
            continue
        collected = directory/'single.json'
        collected.unlink(missing_ok=True)
        code = base.run_command(['combineTool.py','-M','Impacts','-d',context['workspace'],
            '-m',mass_arg,'-n',name,'--named',nuisance,'-o',collected.name], directory,
            allow_failure=True, stdout_path=directory/f'collect_strategy{strategy}.log')
        if code or not collected.is_file():
            continue
        data = json.loads(collected.read_text())
        missing, bad = base.impact_interval_issues(data,[nuisance])
        if not missing and not bad:
            return dict(files=[str(output)], strategy=strategy)
    raise base.WorkflowError('Nuisance interval failed strategies 1/2/0: '+nuisance)


def worker(batch, index):
    directory = batch/f'task_{index:05d}'
    directory.mkdir(exist_ok=True)
    with (directory/'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _worker(batch,index)


def _worker(batch, index):
    plan = json.loads((batch/'plan.json').read_text())
    if inference.digest(batch/'source_manifest.json') != plan['manifest_sha256']:
        raise ValueError('Saved source manifest changed')
    for name, expected in plan['source_hashes'].items():
        if inference.digest(batch/name) != expected:
            raise ValueError('Saved worker source changed')
    payload = json.loads((batch/'source_manifest.json').read_text())
    task = plan['tasks'][index]; case = plan['cases'][task['case']]
    job,args = case_options(payload,case,batch)
    directory = batch/f'task_{index:05d}'
    previous = directory/'status.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        if old.get('state') == 'success':
            with base.validation_cache():
                check_case(payload,case,args,task)
                for filename, expected in old['hashes'].items():
                    if inference.digest(filename) != expected:
                        raise ValueError('Completed repair file changed')
            print('[REUSE]',index,flush=True)
            return
        shutil.copy2(previous,directory/f'previous_status_{time.time_ns()}.json')
    status = dict(state='running', task=task, started=time.time())
    write(directory/'status.json',status)
    try:
        with base.validation_cache():
            check_case(payload,case,args,task)
        if task['kind'] == 'scan':
            products = scan_case(case,args,task,directory)
        else:
            out = batch/case['directory']
            repair = out/f'impact_repair_M{base.mass_label(case["mass"])}'
            repair.mkdir(exist_ok=True)
            for dependency in task['dependencies']:
                parent = batch/f'task_{dependency:05d}'/'status.json'
                result = json.loads(parent.read_text())
                if result['state'] != 'success':
                    raise ValueError('Incomplete distributed scan')
                for filename in result['files']:
                    if inference.digest(filename) != result['hashes'][filename]:
                        raise ValueError('Distributed scan file changed')
                    shutil.copy2(filename,repair/Path(filename).name)
            # Reuse the full existing collector/auditor, including special
            # boundary profiles. Unexpected invalid ordinary scans fail here.
            args.impact_repair_no_scans = True
            products = base.run_condor_impact_final(payload,args,job,Path(case['card']),
                base.mass_label(case['mass']),case['mass'],out)
        status.update(products, hashes={p:inference.digest(p) for p in products['files']},state='success')
    except Exception as exc:
        status.update(state='failed',error=str(exc))
        raise
    finally:
        status['elapsed_seconds'] = time.time()-status['started']
        write(directory/'status.json',status)


def make_tasks(cases):
    tasks=[]
    for i,case in enumerate(cases):
        dependencies=[]
        for nuisance in case['pending']:
            dependencies.append(len(tasks))
            tasks.append(dict(case=i,kind='scan',nuisance=nuisance,dependencies=[]))
        tasks.append(dict(case=i,kind='final',dependencies=dependencies))
    return tasks


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-batch')
    p.add_argument('--batch',required=True)
    p.add_argument('--indices',default='2221,2983,4524')
    p.add_argument('--stage',choices=('submit','collect'),default='submit')
    p.add_argument('--memory',type=int,default=16000)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--worker',type=int,help=argparse.SUPPRESS)
    opt=p.parse_args(argv); batch=Path(opt.batch).resolve()
    if opt.worker is not None:
        worker(batch,opt.worker); return
    if opt.stage == 'collect':
        plan=json.loads((batch/'plan.json').read_text()); counts=Counter()
        for i,task in enumerate(plan['tasks']):
            path=batch/f'task_{i:05d}'/'status.json'
            result=json.loads(path.read_text()) if path.exists() else dict(state='pending')
            counts[task['kind']+':'+result['state']]+=1
            if task['kind']=='final' or result['state']=='failed':
                print(i,json.dumps(result))
        print('[STATUS]',dict(counts)); return
    if not opt.source_batch or opt.memory<=0:
        raise ValueError('Provide --source-batch and positive memory')
    cmssw=Path(os.environ.get('CMSSW_BASE','')); arch=os.environ.get('SCRAM_ARCH','')
    if not arch or not (cmssw/'src').is_dir():
        raise ValueError('Activate CMSSW first')
    source=Path(opt.source_batch).resolve()
    payload=json.loads((source/'manifest.json').read_text())
    if payload.get('manifest_sha256')!=inference.manifest_digest(payload):
        raise ValueError('Source manifest digest mismatch')
    indices=[int(i) for i in opt.indices.split(',')]
    if len(indices)!=len(set(indices)) or any(i<0 or i>=len(payload['jobs']) or
            payload['jobs'][i]['kind']!='impact_final' for i in indices):
        raise ValueError('Select unique valid impact_final indices')
    batch.mkdir(parents=True,exist_ok=False)
    shutil.copy2(source/'manifest.json',batch/'source_manifest.json')
    for name in SOURCES:
        dest=batch/'source'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(__file__).parent/name,dest)
    cases=[]
    with base.validation_cache():
        for index in indices:
            job=payload['jobs'][index]; args=inference.options(payload,job)
            print('[PREPARE]',index,'reading completed scans; no new fit',flush=True)
            context=base.condor_impact_context(payload,args,job['initial'])
            parents={str(i):base.completed_condor_job(payload,args,payload['jobs'][i])['completion_id']
                     for i in [job['initial'],*job['nuisance_jobs']]}
            out=batch/f'case_{index:05d}';out.mkdir()
            shutil.copy2(context['initial_file'],out/Path(context['initial_file']).name)
            for i in job['nuisance_jobs']:
                for filename in base.completed_condor_job(payload,args,payload['jobs'][i])['files']:
                    (out/Path(filename).name).symlink_to(Path(filename).resolve())
            label=base.mass_label(job['mass']);name=f'.Impacts_{base.tag_for(args,job["target"])}_M{label}'
            collected=out/'baseline.json'
            base.run_command(['combineTool.py','-M','Impacts','-d',context['workspace'],
                '-m',base.format_mass_for_combine(job['mass']),'-n',name,'-o',collected.name],
                out,stdout_path=out/'collect_baseline.log')
            bounds=base.impact_workspace_parameters(Path(context['workspace']))
            pending=pending_scans(json.loads(collected.read_text()),bounds)
            cases.append(dict(source_index=index,mass=job['mass'],target=job['target'],
                card=job['card'],context=context,parents=parents,pending=pending,directory=out.name,
                nuisance_parents={payload['jobs'][i]['nuisance']:i for i in job['nuisance_jobs']}))
            print('[SPLIT]',index,len(pending),'ordinary nuisance repairs',flush=True)
    tasks=make_tasks(cases)
    write(batch/'plan.json',dict(cases=cases,tasks=tasks,
        manifest_sha256=inference.digest(batch/'source_manifest.json'),
        source_hashes={'source/'+n:inference.digest(batch/'source'/n) for n in SOURCES}))
    wrapper=batch/'run.sh'
    wrapper.write_text('\n'.join(['#!/bin/bash','set -eo pipefail','ulimit -s unlimited',
        'source /cvmfs/cms.cern.ch/cmsset_default.sh','export SCRAM_ARCH='+shlex.quote(arch),
        'cd '+shlex.quote(str(cmssw/'src')),'eval "$(scramv1 runtime -sh)"',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        'exec python3 -u '+shlex.quote(str(batch/'source/full_spectrum_impact_repair_dag.py'))+
        ' --batch '+shlex.quote(str(batch))+' --worker "$1"','']))
    wrapper.chmod(0o755)
    (batch/'submit.jds').write_text('\n'.join(['universe = vanilla',f'executable = {wrapper}',
        f'initialdir = {batch}','arguments = $(index)',f'request_memory = {opt.memory}MB',
        'request_cpus = 1','getenv = True','should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT','transfer_output_files = ""',
        'output = task_$(index).out','error = task_$(index).err','log = condor.log',
        'jobbatchname = NPSImpactSplit_'+batch.name,'queue','']))
    lines=[]
    for i,task in enumerate(tasks):
        lines.extend([f'JOB J{i} submit.jds',f'VARS J{i} index="{i}"'])
        if task['dependencies']:
            lines.append('PARENT '+' '.join('J'+str(j) for j in task['dependencies'])+f' CHILD J{i}')
    (batch/'workflow.dag').write_text('\n'.join(lines)+'\n')
    print('[PLAN]',Counter(t['kind'] for t in tasks),flush=True)
    if not opt.dry_run:
        subprocess.run(['condor_submit_dag',str(batch/'workflow.dag')],cwd=batch,check=True)


if __name__=='__main__':
    try:
        main()
    except (ValueError,OSError,base.WorkflowError) as exc:
        print('ERROR:',exc,file=sys.stderr);raise SystemExit(1)
