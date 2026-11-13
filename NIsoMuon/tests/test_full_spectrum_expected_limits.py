"""Optional limits share the workspace while retaining the original fit DAG."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import full_spectrum_fit as fs
import full_spectrum_condor as condor


class ExpectedLimitTests(unittest.TestCase):
    def test_optional_limit_and_table_dependencies_exclude_impacts_and_toys(self):
        opt=fs.make_parser().parse_args(['--stage','fit','--masses','12,30,70','--expected-limits'])
        jobs=condor.make_jobs(opt,{})
        self.assertEqual(len(jobs),16)
        self.assertEqual([j['kind'] for j in jobs].count('limits'),3)
        self.assertEqual([j['kind'] for j in jobs].count('fit_b'),3)
        self.assertEqual([j['kind'] for j in jobs].count('fit_s'),3)
        for job in jobs:
            if job['kind']=='limits':
                self.assertEqual([jobs[i]['kind'] for i in job['dependencies']],['workspace'])
            self.assertNotIn(job['kind'],('impact_initial','impact_nuisance','impact_final','toys'))
        self.assertEqual(jobs[-1]['kind'],'tables')
        self.assertEqual(sorted(jobs[i]['kind'] for i in jobs[-1]['dependencies']),['limits']*3+['summarize']*3)

    def test_execution_option_does_not_change_the_model_signature(self):
        opt=fs.make_parser().parse_args(['--fit-layout','window'])
        legacy=copy.copy(opt)
        delattr(legacy,'expected_limits')
        opt.expected_limits=True
        self.assertEqual(fs.signature(legacy),fs.signature(opt))
        self.assertEqual(fs.model_directory(legacy,30),fs.model_directory(opt,30))
        for extra in (['--inject-r','1'],['--stage','cards'],['--fast-mode']):
            bad=fs.make_parser().parse_args(['--expected-limits',*extra])
            with self.assertRaisesRegex(ValueError,'requires'):
                fs.configuration(bad)

    def test_limit_reuses_workspace_and_converts_the_saved_signal_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            (directory/'fit_r0').mkdir()
            (directory/'datacard.txt').write_text('saved original card\n')
            (directory/'fit_r0/datacard.txt').write_text(fs.fit_card_text('saved original card\n'))
            workspace=directory/'fit_r0/workspace.root'
            workspace.write_bytes(b'compiled original workspace')
            opt=fs.make_parser().parse_args(['--expected-limits'])
            model=dict(blind=True,mass=30.,target='Run2Run3',card_sha256='original',alpha_internal_unit=.01)
            values=dict(zip(fs.base.EXPECTED_LIMIT_KEYS,(1.,2.,3.,4.,5.)))
            output=directory/'fit_r0/limits/limit.root'
            def run(args,card,target,label,mass,work):
                self.assertEqual(card,workspace)
                self.assertEqual(args.mode,'blind')
                self.assertEqual(args.r_min,0.)
                output.write_bytes(b'synthetic limit result')
                return SimpleNamespace(expected=output)
            with patch.object(fs.base,'run_asymptotic',side_effect=run), \
                 patch.object(fs.base,'_read_single_limit_json',return_value=values):
                products=fs.run_expected_limits(model,directory,opt)
            result=json.loads(products[0].read_text())
            self.assertTrue(result['blind'])
            for key,value in values.items():
                self.assertEqual(result['alpha'][key],value*.01)
            self.assertEqual(workspace.read_bytes(),b'compiled original workspace')
            self.assertNotIn('obs',result['r'])

    def test_tables_export_only_qcd_dy_and_preserve_both_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            opt=fs.make_parser().parse_args(['--masses','30','--output-base',tmp])
            directory=fs.model_directory(opt,30.)/'fit_r0'
            (directory/'limits').mkdir(parents=True)
            normalisations={p:dict(era='2018',process=p,scale=1.,scale_low=.8,scale_high=1.2,constraint=.5)
                            for p in ('QCD','DY','tt')}
            summary=dict(target='Run2Run3',card_sha256='card',fit_card_sha256='fitcard',
                fits={k:dict(status=0,covQual=3,normalisations=normalisations) for k in ('fit_b','fit_s')})
            (directory/'fit_summary.json').write_text(json.dumps(summary))
            limit=dict(card_sha256='card',fit_card_sha256='fitcard',
                alpha={key:.1 for key in fs.base.EXPECTED_LIMIT_KEYS})
            (directory/'limits/expected_limits.json').write_text(json.dumps(limit))
            paths=condor.result_tables(dict(batch=tmp),opt)
            import csv
            rows=list(csv.DictReader(paths[1].read_text().splitlines()))
            self.assertEqual(len(rows),4)
            self.assertEqual({r['process'] for r in rows},{'QCD','DY'})
            self.assertEqual({r['fit'] for r in rows},{'fit_b','fit_s'})
            self.assertEqual(len(list(csv.DictReader(paths[0].read_text().splitlines()))),1)
            limit['card_sha256']='different'
            (directory/'limits/expected_limits.json').write_text(json.dumps(limit))
            with self.assertRaisesRegex(ValueError,'different cards'):
                condor.result_tables(dict(batch=tmp),opt)


class StageMemoryTests(unittest.TestCase):
    def test_active_batch_update_keeps_heavy_jobs_and_snapshots_intact(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            submit=directory/'submit.jds'
            submit.write_text('request_memory = 16000MB\nqueue 1\n')
            opt=fs.make_parser().parse_args(['--stage','set-memory'])
            payload=dict(batch=tmp,batch_id='example',options=dict(condor_memory=16000),
                jobs=[dict(index=i,kind=kind) for i,kind in enumerate(('assemble','workspace','fit_b','fit_s','limits','summarize'))])
            ads=[dict(ClusterId=123,ProcId=0,DAGNodeName='J1'),
                 dict(ClusterId=124,ProcId=0,DAGNodeName='J4')]
            import subprocess
            with patch.object(condor.subprocess,'check_output',return_value=json.dumps(ads)), \
                 patch.object(condor.base,'run_command') as run:
                condor.set_memory(opt,payload)
            self.assertEqual(run.call_args_list[0].args[0][-1],'2048')
            self.assertEqual(run.call_args_list[1].args[0][-1],'16000')
            self.assertIn('JobStatus == 1',run.call_args_list[0].args[0][2])
            expression=submit.read_text()
            self.assertIn('$(jobindex) == 2',expression)
            self.assertIn('$(jobindex) == 3',expression)
            self.assertIn('$(jobindex) == 4',expression)
            self.assertNotIn('$(jobindex) == 1',expression)
            self.assertEqual((directory/'submit.jds.before_stage_memory').read_text(),'request_memory = 16000MB\nqueue 1\n')

    def test_light_and_workspace_memory_preserve_fit_resources(self):
        options=vars(fs.make_parser().parse_args([]))
        for kind in ('prepare','assemble','summarize','tables'):
            self.assertEqual(condor.memory_for_job(options,kind),2048)
        self.assertEqual(condor.memory_for_job(options,'workspace'),2048)
        for kind in ('fit_b','fit_s','limits','fit'):
            self.assertEqual(condor.memory_for_job(options,kind),16000)
        for kind in ('prepare','workspace','fit_b'):
            self.assertEqual(condor.memory_for_job({'condor_memory':12000},kind),12000)
        options['condor_light_memory']=3072
        self.assertEqual(condor.memory_for_job(options,'assemble'),3072)


if __name__=='__main__':unittest.main()
