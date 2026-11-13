"""Exercise the complete DAG with synthetic Combine files, never physics fits."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import full_spectrum_inference as inf
from test_condor_limits import FAKE_COMBINE, FAKE_PLOT
from test_condor_impacts import IMPACT_TOOL, PLOT_IMPACTS


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='inference_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.batch = self.root/'batch'
        self.bin = self.root/'bin'
        self.bin.mkdir()
        tool = IMPACT_TOOL.replace('name = a[a.index("-n")+1]',
            'name = a[a.index("-n")+1]\ninjection = float(a[a.index("--expectSignal")+1]) if "--expectSignal" in a else 0.')
        tool = tool.replace('else min(0., rmin+1.)', 'else injection-1.')
        tool = tool.replace('[low, 0., 1.]', '[low, injection, injection+1.]')
        tool = tool.replace('"r": [-.2, 0., .2]', '"r": [injection-.2, injection, injection+.2]')
        for name, code in [('combine', FAKE_COMBINE), ('combineTool.py', tool),
                           ('plotImpacts.py', PLOT_IMPACTS)]:
            path = self.bin/name
            path.write_text(code)
            path.chmod(0o755)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, PATH=str(self.bin)+os.pathsep+os.environ['PATH']))
        self.stack.enter_context(patch.object(inf.base, 'impact_workspace_parameters',
            side_effect=lambda p: json.loads(Path(p).read_text())))
        self.stack.enter_context(patch.object(inf.base, 'read_impact_initial_poi',
            side_effect=lambda p: json.loads(Path(p).read_text())['POIs'][0]['fit']))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        (self.root/'CMSSW/src').mkdir(parents=True)
        self.results = self.root/'results'
        for mass in (12, 30, 70):
            directory = self.results/f'M{mass}'
            fit = directory/'fit_r0'
            fit.mkdir(parents=True)
            text = '# alpha_internal_unit = 0.01\n# alpha_internal_signal_yield = 25\n'
            (directory/'datacard.txt').write_text(text)
            (fit/'datacard.txt').write_text('shapes * * FAKE\n'+text)
            (fit/'workspace.root').write_text(json.dumps({'theta': [-7., 7.], 'phi': [-7., 7.]}))
            model = dict(blind=True, mass=mass, target='Run2Run3', alpha_internal_unit=.01,
                card_sha256=inf.digest(directory/'datacard.txt'),
                rows=[dict(asimov=100., rates=dict(sig=1.))])
            (directory/'model.json').write_text(json.dumps(model))
            summary = dict(model, inject_r=0., fit_card_sha256=inf.digest(fit/'datacard.txt'),
                fits={k: dict(status=0, covQual=3, r_profile_error_high=1.) for k in ('fit_b','fit_s')})
            (fit/'fit_summary.json').write_text(json.dumps(summary))

    def prepare(self, task='all'):
        inf.main(['--results', str(self.results), '--batch', str(self.batch), '--dry-run',
            '--task', task, '--condor-cmssw-base', str(self.root/'CMSSW'),
            '--condor-scram-arch', 'el9_amd64_gcc12'])
        payload = inf.read_manifest(self.batch)
        # Substitute a rendering stub; production snapshots the actual plotter.
        plot = self.batch/'source/python/plotLimits.py'
        plot.write_text(FAKE_PLOT+'\nfor suffix in (".pdf", ".png"):\n    Path(a[a.index("--output")+1]+suffix).write_text("synthetic plot")\n')
        payload['source_hashes']['source/python/plotLimits.py'] = inf.digest(plot)
        payload['manifest_sha256'] = inf.manifest_digest(payload)
        (self.batch/'manifest.json').write_text(json.dumps(payload))
        return payload

    def test_end_to_end_limits_and_both_injections(self):
        payload = self.prepare()
        self.assertEqual(len(payload['jobs']), 28)
        subprocess.run(['bash', '-n', str(self.batch/'run_job.sh')], check=True)
        limit_plot = payload['jobs'][-1]
        self.assertEqual(limit_plot['kind'], 'limit_plot')
        self.assertEqual([payload['jobs'][i]['kind'] for i in limit_plot['dependencies']], ['limits']*3)
        for job in payload['jobs']:
            inf.worker(self.batch, job['index'])
        inf.report(self.batch)
        summary = json.loads((self.batch/'inference_summary.json').read_text())
        self.assertEqual(len(summary['products']), 7)
        for job in payload['jobs']:
            directory = Path(job['workdir'])
            status = json.loads((directory/'status.json').read_text())
            self.assertEqual(status['state'], 'success')
            if job['kind'] == 'impact_nuisance':
                command = json.loads((directory/'fit_arguments.json').read_text())
                self.assertEqual(float(command[command.index('--expectSignal')+1]), job['injected_r'])
                self.assertLess(float(command[command.index('--rMin')+1]), 0.)
                self.assertEqual(command[command.index('--named')+1], job['nuisance'])
            elif job['kind'] == 'impact_final':
                result = json.loads(
                    (directory/f'impacts_Run2Run3_alpha_M{int(job["mass"])}.json').read_text())
                self.assertAlmostEqual(result['POIs'][0]['fit'][1], job['injected_r']*.01)
                self.assertEqual(len(result['params']), 2)
            elif job['kind'] == 'limit_plot':
                result = json.loads((directory/'limits_Run2Run3_alpha.json').read_text())
                self.assertEqual(len(result), 3)
                for values in result.values():
                    self.assertAlmostEqual(values['exp0'], .03)
                    self.assertNotIn('obs', values)
        before = (Path(payload['jobs'][0]['workdir'])/'status.json').read_bytes()
        inf.worker(self.batch, 0)
        self.assertEqual(before, (Path(payload['jobs'][0]['workdir'])/'status.json').read_bytes())
        name = inf.write_dag(self.batch, payload, resume=True)
        self.assertEqual((self.batch/name).read_text().count(' DONE'), 28)

    def test_changed_inputs_and_early_children_fail(self):
        payload = self.prepare()
        child = next(j for j in payload['jobs'] if j['kind'] == 'impact_nuisance')
        with self.assertRaisesRegex(inf.base.WorkflowError, 'pending'):
            inf.worker(self.batch, child['index'])
        parent = payload['jobs'][child['initial']]
        inf.worker(self.batch, parent['index'])
        inf.worker(self.batch, child['index'])
        Path(parent['workdir'], 'status.json').unlink()
        inf.worker(self.batch, parent['index'])
        with self.assertRaisesRegex(inf.base.WorkflowError, 'different initial fit'):
            inf.base.completed_condor_job(payload, inf.options(payload, child), child)
        Path(child['workspace']).write_text('modified')
        with self.assertRaisesRegex(ValueError, 'workspace changed'):
            inf.worker(self.batch, child['index'])

    def test_limits_only_never_enumerates_nuisances(self):
        with patch.object(inf.base, 'impact_workspace_parameters', side_effect=AssertionError):
            payload = self.prepare('limits')
        self.assertEqual([j['kind'] for j in payload['jobs']], ['limits']*3+['limit_plot'])

    def test_blinding_and_provenance_rejections(self):
        path = self.results/'M12/fit_r0/fit_summary.json'
        record = json.loads(path.read_text())
        record['inject_r'] = 1
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'background-only Asimov'):
            inf.input_case(path)
        record['inject_r'] = 0
        record['fit_card_sha256'] = 'wrong'
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
            inf.input_case(path)

    def test_impact_memory_update_excludes_initial_fits_and_preserves_manifest(self):
        payload=self.prepare()
        manifest=(self.batch/'manifest.json').read_bytes()
        kinds=('impact_nuisance','impact_initial','limit_plot')
        jobs=[next(j for j in payload['jobs'] if j['kind']==kind) for kind in kinds]
        ads=[dict(ClusterId=100+i,ProcId=0,DAGNodeName='J'+str(j['index']),Iwd=str(self.batch))
             for i,j in enumerate(jobs)]
        with patch.object(inf.subprocess,'check_output',return_value=json.dumps(ads)), \
             patch.object(inf.base,'run_command') as run:
            inf.main(['--stage','set-memory','--batch',str(self.batch),'--dag-id','19457414'])
        self.assertEqual(run.call_count,2)
        calls=[c.args[0] for c in run.call_args_list]
        self.assertEqual({c[-1] for c in calls},{'2048','4096'})
        self.assertFalse(any('ClusterId == 101' in c[2] for c in calls))
        self.assertTrue(all('JobStatus == 1' in c[2] for c in calls))
        self.assertEqual(manifest,(self.batch/'manifest.json').read_bytes())
        self.assertIn('ifThenElse',(self.batch/'submit.jds').read_text())
        self.assertTrue((self.batch/'submit.jds.before_impact_memory').exists())

    def test_modified_manifest_and_snapshot_rejected(self):
        self.prepare('limits')
        path = self.batch/'source/limit_workflow.py'
        path.write_text(path.read_text()+'\n# modified\n')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            inf.read_manifest(self.batch)


if __name__ == '__main__':
    unittest.main()
