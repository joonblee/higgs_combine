import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import limit_workflow as base
import full_spectrum_fit as fs
import full_spectrum_inference as inference
import full_spectrum_impact_repair_dag as dag


def parameter(fit):
    return dict(name='n',type='Gaussian',prefit=[-1,0,1],fit=fit,r=[-.1,0,.1])


class OptimisationTest(unittest.TestCase):
    def test_norm_only_keeps_model_and_covariance_options(self):
        full=fs.make_parser().parse_args([])
        norm=fs.make_parser().parse_args(['--fit-output','norms','--fit-robust-hesse'])
        self.assertEqual(fs.signature(full),fs.signature(norm))
        cmd=fs.fit_commands(norm,30,Path('/tmp/test'),'fit_b')[1]
        self.assertNotIn('--saveShapes',cmd)
        self.assertNotIn('--saveWithUncertainties',cmd)
        self.assertIn('--saveNormalizations',cmd)
        self.assertIn('--robustHesse',cmd)

    def test_scoped_cache_detects_mutation_and_expires(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);card=root/'card';card.write_text('card')
            jobs=[]
            for i in range(3):
                out=root/str(i);out.mkdir();product=out/'result';product.write_text('result')
                job=dict(index=i,kind='test',workdir=str(out),card=str(card),
                    card_sha256=hashlib.sha256(card.read_bytes()).hexdigest(),target='test',
                    dependencies=[] if i==0 else [0])
                status=dict(batch_id='batch',index=i,kind='test',state='success',mode='blind',
                    target='test',card_sha256=job['card_sha256'],files=[str(product)],completion_id=str(i),
                    dependency_completions={} if i==0 else {'0':'0'})
                (out/'status.json').write_text(json.dumps(status));jobs.append(job)
            payload=dict(batch_id='batch',jobs=jobs);args=SimpleNamespace(mode='blind')
            with patch.object(base,'_completed_condor_job',wraps=base._completed_condor_job) as check:
                with base.validation_cache():
                    base.completed_condor_job(payload,args,jobs[1])
                    base.completed_condor_job(payload,args,jobs[2])
                self.assertEqual(check.call_count,3)
            with self.assertRaises(base.WorkflowError):
                with base.validation_cache():
                    base.completed_condor_job(payload,args,jobs[1])
                    card.write_text('changed')
            with self.assertRaises(base.WorkflowError):
                with base.validation_cache():base.completed_condor_job(payload,args,jobs[1])

    def test_nuisance_dag_retry_and_complete_collection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);src=root/'old';src.mkdir();batch=root/'new'
            cmssw=root/'cmssw';(cmssw/'src').mkdir(parents=True)
            initial=src/'initial.root';initial.write_text('initial')
            workspace=src/'workspace.root';workspace.write_text('workspace')
            scan=src/'original_scan.root';scan.write_text('unchanged')
            card=src/'card';card.write_text('card')
            context=dict(workspace=str(workspace),initial_file=str(initial),rmin=-10.,rmax=100.)
            job=dict(index=2,kind='impact_final',mass=30.,target='Run2Run3',injected_r=1.,
                card=str(card),card_sha256=inference.digest(card),initial=0,nuisance_jobs=[1])
            payload=dict(batch_id='old',config=dict(mode='blind',parameter='alpha'),
                jobs=[dict(index=0),dict(index=1,nuisance='n'),job])
            payload['manifest_sha256']=inference.manifest_digest(payload)
            (src/'manifest.json').write_text(json.dumps(payload))
            strategies=[]
            def run(command,cwd,**kwargs):
                if '--doFits' in command:
                    strategy=command[command.index('--cminDefaultMinimizerStrategy')+1]
                    strategies.append(strategy)
                    self.assertEqual(command[command.index('--expectSignal')+1],'1')
                    name=command[command.index('-n')+1]
                    (cwd/f'higgsCombine_paramFit_{name}_n.MultiDimFit.mH30.root').write_text('new scan')
                else:
                    fit=[0,0,0] if not strategies or strategies[-1]=='1' else [-1,0,1]
                    (cwd/command[command.index('-o')+1]).write_text(json.dumps(dict(params=[parameter(fit)])))
                return 0
            def completed(payload,args,job):
                return dict(completion_id=str(job['index']),files=[str(scan)])
            def finish(payload,args,job,card,label,mass,out):
                self.assertTrue(args.impact_repair_no_scans)
                paths=list((out/'impact_repair_M30').glob('*.root'))
                self.assertEqual(len(paths),1)
                result=out/'validated.json';result.write_text('{}')
                return dict(files=[str(result)])
            with patch.dict(os.environ,CMSSW_BASE=str(cmssw),SCRAM_ARCH='test'), \
                 patch.object(base,'condor_impact_context',return_value=context), \
                 patch.object(base,'completed_condor_job',side_effect=completed), \
                 patch.object(base,'run_command',side_effect=run), \
                 patch.object(base,'impact_workspace_model',return_value=(['r'],{'n':(-7,7)})), \
                 patch.object(base,'run_condor_impact_final',side_effect=finish):
                dag.main(['--source-batch',str(src),'--batch',str(batch),'--indices','2','--serial-strategies','--dry-run'])
                plan=json.loads((batch/'plan.json').read_text())
                self.assertEqual([t['kind'] for t in plan['tasks']],['scan','final'])
                self.assertEqual(plan['tasks'][1]['dependencies'],[0])
                dag.worker(batch,0);dag.worker(batch,1)
                dag.worker(batch,0)  # reuse; no new scan
            self.assertEqual(strategies,['1','2'])
            self.assertEqual(scan.read_text(),'unchanged')
            for i in (0,1):
                self.assertEqual(json.loads((batch/f'task_{i:05d}'/'status.json').read_text())['state'],'success')
            result=batch/'task_00000'/'higgsCombine_paramFit_.Impacts_Run2Run3_alpha_M30_n.MultiDimFit.mH30.root'
            result.write_text('tampered')
            with self.assertRaises(ValueError):
                with patch.object(base,'condor_impact_context',return_value=context), \
                     patch.object(base,'completed_condor_job',side_effect=completed):
                    dag.worker(batch,0)


if __name__=='__main__':unittest.main()
