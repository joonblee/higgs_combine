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


if __name__=='__main__':unittest.main()
