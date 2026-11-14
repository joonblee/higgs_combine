#!/usr/bin/env python3
"""Conditional Poisson bias toys from saved blinded window models.

Default injection strengths are 0,1,2 times each model's own positive profile
error from its completed Asimov fit, not a fixed event yield for all masses.
Truth nuisances stay at nominal values; fitting still profiles nuisances.
Default minimum-only fits measure empirical bias, not interval coverage/pulls.
Optional positive per-channel/process truth factors generate alternative
background toys; sparse signed MC ratios are never made into truth automatically.
Condor snapshots and transfers the workspaces and Python sources per chunk.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import sys
import tarfile

import numpy as np
import full_spectrum_audit as audit
import full_spectrum_fit as fs
import limit_workflow as base
import condor_memory_policy as memory_policy

SOURCES=('condor_memory_policy.py','full_spectrum_toys.py','full_spectrum_audit.py','full_spectrum_fit.py',
         'limit_workflow.py','limit_workflow_sideband_fit.py')


def digest(path):
    return base.validation_digest(path)


def manifest_digest(payload):
    return hashlib.sha256(json.dumps({k:v for k,v in payload.items() if k!='manifest_sha256'},
        sort_keys=True,allow_nan=False).encode()).hexdigest()


def truth_card(text,model,truth,expected_hash):
    if truth['fit_card_sha256']!=expected_hash or truth['mass']!=model['mass']:
        raise ValueError('Alternative truth does not match this fit card/mass.')
    channels={r['channel'] for r in model['rows']}
    lines=[]
    for channel,processes in truth['factors'].items():
        if channel not in channels:
            raise ValueError('Unknown alternative-truth channel: '+channel)
        for process,value in processes.items():
            if process not in ('QCD','DY','tt','ST','Others') or not math.isfinite(value) or value<0:
                raise ValueError('Invalid alternative background process/factor.')
            lines.append(f'Truth_{channel}_{process} rateParam {channel} {process} {value:.17g}')
    if not lines:
        raise ValueError('Alternative truth factor map is empty.')
    return text+'\n# Alternative truth only; never fit these extra rateParams.\n'+'\n'.join(lines)+'\n'


def plan_case(summary_path,opt):
    summary=json.loads(summary_path.read_text())
    model=json.loads((summary_path.parent.parent/'model.json').read_text())
    if summary.get('blind') is not True or model.get('blind') is not True or summary['inject_r']!=0:
        raise ValueError('Require completed background-only Asimov inputs.')
    if summary['card_sha256']!=model['card_sha256']:
        raise ValueError('Summary/model hash mismatch.')
    for kind in ('fit_b','fit_s'):
        if summary['fits'][kind]['status']!=0 or summary['fits'][kind]['covQual']!=3:
            raise ValueError('Asimov fits must pass status=0, covQual=3 first.')
    import ROOT
    fitfile=summary_path.parent/'fit_s/fitDiagnostics.FullSpectrum.root'
    if not fitfile.exists():
        fitfile=summary_path.parent/'fitDiagnostics.FullSpectrum.root'
    handle=ROOT.TFile.Open(str(fitfile))
    try:
        fit=handle.Get('fit_s') if handle and not handle.IsZombie() else None
        if not fit or fit.status()!=0 or fit.covQual()!=3:
            raise ValueError('Missing/failed saved S+B fit.')
        reference=fs.poi_diagnostics(fit)['r_profile_error_high']
    finally:
        if handle: handle.Close()
    if reference is None or not math.isfinite(reference) or reference<=0:
        raise ValueError('No positive profile error for sensitivity-scaled injections.')
    safe=min(r['asimov']/r['rates']['sig'] for r in model['rows'] if r['rates']['sig']>0)
    rmin=opt.r_min if opt.r_min is not None else -min(5*reference,.8*safe)
    if not -safe<rmin<0:
        raise ValueError('Require a negative fit r-min with positive nominal Poisson means.')
    return summary,model,reference,rmin


def commands(task,fit_workspace,truth_workspace,toy_file=None):
    common=['-m',str(task['mass']),'-t',str(task['count']),'-s',str(task['seed'])]
    gen=['combine','-M','GenerateOnly',str(truth_workspace),*common,
         '--expectSignal',str(task['injected_r']),'--rMin','0','--rMax',str(task['r_max']),
         '--toysNoSystematics','--saveToys','-n','.Truth']
    toy_file=toy_file or f'higgsCombine.Truth.GenerateOnly.mH{base.mass_label(task["mass"])}.{task["seed"]}.root'
    fit=['combine','-M','FitDiagnostics',str(fit_workspace),*common,
         '--toysFile',str(toy_file),'--toysNoSystematics','--skipBOnlyFit',
         '--setParameters',f'r={task["injected_r"]}',
         '--rMin',str(task['r_min']),'--rMax',str(task['r_max']),
         '--minos','none','--cminDefaultMinimizerStrategy','1','-n','.Bias']
    if not task['toy_errors']:
        fit.append('--noErrors')
    return gen,fit


def run_worker(payload,index,destination,root):
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError('Toy result directory is not empty.')
    def input_path(relative):
        path=Path(root)/relative
        return path if path.exists() else Path(root)/Path(relative).name
    task=payload['tasks'][index]
    try:
        if payload.get('manifest_sha256')!=manifest_digest(payload):
            raise ValueError('Toy manifest changed after preparation.')
        for name,expected in {**payload['source_hashes'],**task['inputs']}.items():
            if digest(input_path(name))!=expected:
                raise ValueError('Changed toy input/source: '+name)
        gen,fit=commands(task,input_path(task['fit_workspace']),input_path(task['truth_workspace']))
        base.run_command(gen,destination,stdout_path=destination/'generate.log')
        generated=list(destination.glob('higgsCombine.Truth.GenerateOnly.mH*.root'))
        if len(generated)!=1:
            raise ValueError('Expected exactly one generated toy file.')
        _,fit=commands(task,input_path(task['fit_workspace']),input_path(task['truth_workspace']),generated[0])
        base.run_command(fit,destination,stdout_path=destination/'fit.log')
        import ROOT
        handle=ROOT.TFile.Open(str(destination/'fitDiagnostics.Bias.root'))
        try:
            tree=handle.Get('tree_fit_sb') if handle and not handle.IsZombie() else None
            if not tree:
                raise ValueError('Missing toy S+B diagnostics tree.')
            branches={b.GetName() for b in tree.GetListOfBranches()}
            if not {'r','fit_status'}<=branches:
                raise ValueError('Missing r/fit_status branches.')
            values=[]
            for entry in tree:
                value=float(entry.r)
                values.append(dict(r=value if math.isfinite(value) else None,status=int(entry.fit_status)))
        finally:
            if handle:handle.Close()
        record=dict(state='completed',task=task,entries=values,
                    fit_file_sha256=digest(destination/'fitDiagnostics.Bias.root'))
        (destination/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    except Exception as error:
        (destination/'failure.json').write_text(json.dumps(dict(state='failed',task=task,error=str(error)),indent=2)+'\n')
        raise


def collect(batch):
    batch=Path(batch).resolve();payload=json.loads((batch/'manifest.json').read_text())
    if payload.get('manifest_sha256')!=manifest_digest(payload):
        raise ValueError('Toy manifest changed after preparation.')
    groups={}
    for task in payload['tasks']:
        key=(task['fit_card_sha256'],task['injected_r'],task['truth_sha256'])
        group=groups.setdefault(key,dict(mass=task['mass'],injected_r=task['injected_r'],
            reference_r=task['reference_r'],truth=task['truth_label'],requested=0,missing_chunks=[],entries=[],
            r_min=task['r_min'],r_max=task['r_max']))
        group['requested']+=task['count']
        directory=batch/'jobs'/f'job_{task["index"]:05d}'
        path=directory/'result/result.json'
        archive=directory/'result.tar.gz'
        if path.exists():
            record=json.loads(path.read_text())
        elif archive.exists():
            with tarfile.open(archive,'r:gz') as handle:
                try:
                    member=handle.extractfile('result/result.json')
                except KeyError:
                    member=None
                record=json.loads(member.read()) if member else None
            if record is None:
                group['missing_chunks'].append(task['index']);continue
        else:
            group['missing_chunks'].append(task['index']);continue
        if record['task']!=task:
            raise ValueError('Stale or mismatched toy result: '+str(path))
        group['entries'].extend(record['entries'])
    results=[]
    for group in groups.values():
        entries=group.pop('entries');accepted=[e['r'] for e in entries if e['status']==0
            and isinstance(e['r'],(int,float)) and math.isfinite(e['r'])]
        group.update(recorded=len(entries),successful=len(accepted),failed=len(entries)-len(accepted),
            boundary_fits=sum(min(abs(v-group['r_min']),abs(v-group['r_max']))<1e-4 for v in accepted))
        group['complete']=not group['missing_chunks'] and len(entries)==group['requested'] and group['failed']==0 and group['boundary_fits']==0
        if accepted:
            mean=float(np.mean(accepted));sd=float(np.std(accepted,ddof=1)) if len(accepted)>1 else None
            group.update(mean_r=mean,bias=mean-group['injected_r'],empirical_std=sd,
                bias_standard_error=sd/math.sqrt(len(accepted)) if sd is not None else None,
                mean_is_conditional_on_fit_success=group['failed']>0)
        results.append(group)
    output=batch/'toy_summary.json'
    output.write_text(json.dumps(dict(schema_version=1,blind=True,ensembles=results,
        note='Conditional Poisson toys at fixed truth nuisance values; nuisance parameters profiled in fitting. '
             'Minimum-only by default: empirical bias, not pulls or interval coverage. Failed/boundary/missing toys remain visible.'),indent=2,allow_nan=False)+'\n')
    print('[SAVED]',output)


def submit(opt):
    with base.validation_cache():
        return _submit(opt)


def _submit(opt):
    if opt.toys<1 or opt.toys_per_job<1 or any(not math.isfinite(x) or x<0 for x in opt.multipliers):
        raise ValueError('Use positive toy counts and finite nonnegative injection multipliers.')
    batch=Path(opt.batch).resolve()
    if batch.exists() or any(c.isspace() or c in '$"' for c in str(batch)):
        raise ValueError('Use a new batch path without whitespace, dollar signs or quotes.')
    paths=audit.find_summaries(opt.results)
    cases=[(p,*plan_case(p,opt)) for p in paths]
    injections=json.loads(Path(opt.injection_r_json).read_text()) if opt.injection_r_json else None
    if injections is not None:
        for _,_,model,_,_ in cases:
            values=injections.get(base.mass_label(model['mass']))
            if not isinstance(values,list) or not values or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in values):
                raise ValueError('Need a finite nonnegative injection list for every mass in --injection-r-json.')
    batch.mkdir(parents=True);(batch/'source').mkdir();(batch/'input').mkdir();(batch/'jobs').mkdir()
    source_hashes={}
    for name in SOURCES:
        path=batch/'source'/name;shutil.copy2(Path(__file__).parent/name,path)
        source_hashes['source/'+name]=digest(path)
    tasks=[]
    for path,summary,model,reference,rmin in cases:
        source=path.parent/'workspace.root'
        label=digest(source)[:16]
        copied=batch/'input'/('fit_'+label+'.root');shutil.copy2(source,copied)
        truth_workspace=batch/'input'/('truth_'+label+'.root')
        shutil.copy2(copied,truth_workspace)
        truth_label='nominal';truth_hash=digest(copied)
        if opt.truth_json:
            truth=json.loads(Path(opt.truth_json).read_text())
            text=truth_card((path.parent/'datacard.txt').read_text(),model,truth,summary['fit_card_sha256'])
            card=batch/'input'/('truth_'+label+'.txt');card.write_text(text)
            truth_workspace=batch/'input'/('truth_'+label+'.root')
            base.run_command(['text2workspace.py',str(card),'-m',str(model['mass']),'-o',str(truth_workspace)],batch/'input',stdout_path=batch/'input'/('truth_'+label+'.log'))
            truth_label=Path(opt.truth_json).name;truth_hash=digest(Path(opt.truth_json))
        levels=[(None,float(v)) for v in injections[base.mass_label(model['mass'])]] if injections is not None else [(v,v*reference) for v in opt.multipliers]
        input_hashes={str(p.relative_to(batch)):digest(p) for p in (copied,truth_workspace)}
        for multiplier,injection in levels:
            for start in range(0,opt.toys,opt.toys_per_job):
                index=len(tasks)
                task=dict(index=index,mass=model['mass'],reference_r=reference,injected_r=injection,
                    multiplier_of_positive_profile_error=multiplier,count=min(opt.toys_per_job,opt.toys-start),
                    injection_definition='explicit per-mass r list' if injections is not None else 'multiple of positive Asimov profile error',
                    seed=opt.seed+index,r_min=rmin,r_max=max(100.,10*reference,2*injection+5),
                    toy_errors=opt.toy_errors,fit_card_sha256=summary['fit_card_sha256'],
                    truth_label=truth_label,truth_sha256=truth_hash,
                    fit_workspace=str(copied.relative_to(batch)),truth_workspace=str(truth_workspace.relative_to(batch)))
                task['inputs']=dict(input_hashes)
                tasks.append(task);(batch/'jobs'/f'job_{index:05d}').mkdir()
    payload=dict(schema_version=1,blind=True,tasks=tasks,source_hashes=source_hashes)
    payload['manifest_sha256']=manifest_digest(payload)
    (batch/'manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    if opt.backend=='local':
        for task in tasks:
            run_worker(payload,task['index'],batch/'jobs'/f'job_{task["index"]:05d}'/'result',batch)
        collect(batch);return
    cmssw,arch=base.condor_environment(opt)
    wrapper=batch/'run_job.sh'
    wrapper.write_text('\n'.join(['#!/bin/bash','set -eo pipefail','ulimit -s unlimited',
        'job_scratch="$PWD"','source /cvmfs/cms.cern.ch/cmsset_default.sh',
        'export SCRAM_ARCH='+shlex.quote(arch),'cd '+shlex.quote(str(cmssw/'src')),
        'eval "$(scram runtime -sh)"','cd "$job_scratch"','mkdir -p result','set +e',
        'python3 full_spectrum_toys.py --worker manifest.json --index "$1"',
        'job_status=$?','set -e','tar -czf result.tar.gz result','exit "$job_status"','']))
    wrapper.chmod(0o755)
    transfer=[str(batch/'source'/name) for name in SOURCES]+[str(batch/'manifest.json')]
    (batch/'submit.jds').write_text('\n'.join(['universe = vanilla',f'executable = {wrapper}',
        f'initialdir = {batch}','arguments = $(taskid)','should_transfer_files = YES',
        'when_to_transfer_output = ON_EXIT','transfer_input_files = '+','.join(transfer)+',$(fitfile),$(truthfile)',
        'transfer_output_files = result.tar.gz','transfer_output_remaps = "result.tar.gz=$(destination)"',
        'output = jobs/job_$(taskid).out','error = jobs/job_$(taskid).err','log = condor.log',
        'request_memory = '+str(opt.condor_memory),
        *memory_policy.retry_lines(str(max(opt.condor_memory,2048))),
        'queue taskid,fitfile,truthfile,destination from tasks.txt','']))
    task_lines=[]
    for task in tasks:
        destination=batch/'jobs'/('job_%05d'%task['index'])/'result.tar.gz'
        task_lines.append(f'{task["index"]} {batch/task["fit_workspace"]} {batch/task["truth_workspace"]} {destination}')
    (batch/'tasks.txt').write_text('\n'.join(task_lines)+'\n')
    print(f'[PLAN] {len(tasks)} independent Condor chunks; {sum(t["count"] for t in tasks)} toy fits.')
    if not opt.dry_run:
        base.run_command(['condor_submit','submit.jds'],batch,stdout_path=batch/'submit.log')
    print('[BATCH]',batch)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('submit','collect'),default='submit')
    parser.add_argument('--results',nargs='+')
    parser.add_argument('--batch',default='full_spectrum_outputs/toy_pilot')
    parser.add_argument('--toys',type=int,default=10)
    parser.add_argument('--toys-per-job',type=int,default=5)
    parser.add_argument('--multipliers',type=base.parse_float_list,default=[0.,1.,2.])
    parser.add_argument('--injection-r-json',help='Optional JSON mass->[r,...], overriding multipliers; supports later 0,r95,2*r95 studies.')
    parser.add_argument('--seed',type=int,default=123456)
    parser.add_argument('--r-min',type=float)
    parser.add_argument('--truth-json')
    parser.add_argument('--toy-errors',action='store_true',help='Calculate HESSE errors too; empirical bias summary does not use getError().')
    parser.add_argument('--backend',choices=('condor','local'),default='condor')
    parser.add_argument('--condor-memory',type=int,default=1024)
    parser.add_argument('--condor-cmssw-base')
    parser.add_argument('--condor-scram-arch')
    parser.add_argument('--dry-run',action='store_true',help='Prepare Condor files without submitting.')
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    parser.add_argument('--index',type=int,help=argparse.SUPPRESS)
    opt=parser.parse_args(argv)
    if opt.worker:
        path=Path(opt.worker).resolve();payload=json.loads(path.read_text())
        run_worker(payload,opt.index,Path.cwd()/'result',path.parent)
    elif opt.stage=='collect':collect(opt.batch)
    else:
        if not opt.results:raise ValueError('--results is required for submission.')
        if opt.dry_run and opt.backend=='local':raise ValueError('--dry-run is for Condor planning.')
        submit(opt)


if __name__=='__main__':
    try:main()
    except (ValueError,OSError,base.WorkflowError) as error:
        print('ERROR:',error,file=sys.stderr);raise SystemExit(1)
