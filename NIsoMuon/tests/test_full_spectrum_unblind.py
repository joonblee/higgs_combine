"""Observed-data cards and limits, with unchanged blinded defaults."""
import contextlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]))
import full_spectrum_fit as fs
import full_spectrum_condor as batch
import limit_workflow as base
import limit_workflow_sideband_fit as shared
from test_full_spectrum_review import window_example


class UnblindTests(unittest.TestCase):
    def test_modes_keep_old_blind_signature_and_separate_observed_paths(self):
        blind=fs.make_parser().parse_args(['--fit-layout','window'])
        old=SimpleNamespace(**{k:v for k,v in vars(blind).items() if k!='mode'})
        observed=fs.make_parser().parse_args(['--mode','unblind','--fit-layout','window'])
        self.assertEqual(fs.signature(blind),fs.signature(old))
        self.assertNotEqual(fs.signature(blind),fs.signature(observed))
        self.assertNotEqual(fs.model_directory(blind,30),fs.model_directory(observed,30))
        self.assertEqual(fs.configuration(observed).mode,'unblind')
        self.assertIn('-t',fs.fit_commands(blind,30,Path('/tmp'))[1])
        command=fs.fit_commands(observed,30,Path('/tmp'))[1]
        self.assertNotIn('-t',command);self.assertNotIn('--expectSignal',command)
        for options in (['--inject-r','1'],['--fast-mode']):
            opt=fs.make_parser().parse_args(['--mode','unblind',*options])
            with self.assertRaisesRegex(ValueError,'actual observations'):
                fs.configuration(opt)

    def test_observed_card_uses_counts_including_zero_without_changing_backgrounds(self):
        model,_=window_example()
        model['blind']=False;model['observation']='OS_data'
        for i,row in enumerate(model['rows']):row['observation']=i
        text=shared.card_text(model,normalisation_kappa=1.5,prefix='FS')
        actual=next(line.split()[1:] for line in text.splitlines() if line.startswith('observation '))
        self.assertEqual(list(map(float,actual)),list(range(len(model['rows']))))
        self.assertEqual(list(shared.nominal_predictions(text).values()),[r['asimov'] for r in model['rows']])
        del model['rows'][0]['observation']
        with self.assertRaises(KeyError):shared.card_text(model,normalisation_kappa=1.5,prefix='FS')

    def test_reader_blocks_data_by_default_and_fingerprints_explicit_observations(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'data.root';path.write_text('OS counts')
            reader=object.__new__(batch.InputReader);reader.fingerprints={}
            with patch.object(base.RootReader,'_file',return_value='OS input'):
                with self.assertRaisesRegex(base.WorkflowError,'forbidden'):reader._file(str(path))
                self.assertFalse(reader.fingerprints)
                reader.allow_data=True
                self.assertEqual(reader._file(str(path)),'OS input')
                reader.check_inputs()
                path.write_text('changed OS input')
                with self.assertRaisesRegex(ValueError,'Input changed'):reader.check_inputs()

    def test_observed_preparation_reads_integer_data_for_every_bin_without_fallback(self):
        opt=fs.make_parser().parse_args(['--mode','unblind','--target','2018','--masses','30'])
        args=fs.configuration(opt)
        metadata={'fit':{'covariance':[[1.]]}}
        axis=SimpleNamespace(FindFixBin=lambda x:1)
        hist=SimpleNamespace(GetXaxis=lambda:axis)
        derivative=SimpleNamespace(Integral=lambda a,b:0.)
        inputs=(metadata,hist,[None]*6+[derivative,derivative])
        rates=dict(sig=10.,QCD=20.,DY=3.,tt=2.,ST=1.,Others=1.)
        channel=SimpleNamespace(raw_rates=rates,nuisances={},warnings=[],
            qcd_shape_sigma_down=1.,qcd_shape_sigma_up=1.,dy_nf_amc=.1,dy_lightjet_error=1.)
        windows=[dict(region='SR',low=29.,high=31.),dict(region='R0000',low=31.,high=80.)]
        reader=SimpleNamespace(qcd_inputs=lambda *a:inputs)
        def observed(filename,path,low,high):
            self.assertTrue(filename.endswith('/2018/data.root'))
            return base.YieldResult(0. if low==29. else 4.,0.,1,2)
        reader.integral=observed
        with patch.object(base,'build_channels_for_mass',return_value=({'2018':channel},base.AuditLog())), \
             patch.object(base,'qcd_root_window_statistics',return_value=dict(gradient=[1.])):
            part=shared.build_bin_rows(reader,args,opt,30.,{'2018':'signal.root'},'2018',windows,True)
            self.assertEqual([r['observation'] for r in part['rows']],[0,4])
            for value in (None,-1.,.5,float('nan')):
                reader.integral=lambda *a,value=value:None if value is None else base.YieldResult(value,0.,1,2)
                with self.assertRaisesRegex(base.WorkflowError,'integer OS data counts'):
                    shared.build_bin_rows(reader,args,opt,30.,{'2018':'signal.root'},'2018',windows,True)

    def test_observed_limits_keep_expected_quantiles_and_export_obs_to_plot(self):
        with tempfile.TemporaryDirectory() as temp:
            opt=fs.make_parser().parse_args(['--mode','unblind','--masses','30','--limits',
                '--fit-components','none','--output-base',temp])
            fs.configuration(opt)
            directory=fs.model_directory(opt,30);fit=directory/'fit_r0';fit.mkdir(parents=True)
            text='observation 13\n'
            (directory/'datacard.txt').write_text(text)
            (fit/'datacard.txt').write_text(fs.fit_card_text(text))
            (fit/'workspace.root').write_bytes(b'workspace containing actual counts')
            model=dict(blind=False,mass=30.,target='Run2Run3',alpha_internal_unit=.01,
                card_sha256=base.validation_digest(directory/'datacard.txt'))
            expected=fit/'expected.root';observed=fit/'observed.root'
            values={key:float(i+1) for i,key in enumerate(base.EXPECTED_LIMIT_KEYS)}
            def run(args,workspace,*args2):
                self.assertEqual(args.mode,'unblind')
                expected.write_text('expected');observed.write_text('observed')
                return SimpleNamespace(expected=expected,observed=observed)
            def read(path,*args):return dict(obs=2.5,**values) if path==observed else values
            with patch.object(base,'run_asymptotic',side_effect=run),patch.object(base,'_read_single_limit_json',side_effect=read):
                products=fs.run_expected_limits(model,directory,opt)
            result=json.loads(products[0].read_text())
            self.assertFalse(result['blind']);self.assertEqual(result['r']['obs'],2.5)
            self.assertEqual(result['alpha']['obs'],.025)
            self.assertEqual(set(result['alpha']),set(base.EXPECTED_LIMIT_KEYS)|{'obs'})
            self.assertIn(observed,products)
            def plot(command,cwd,**kwargs):
                self.assertEqual(command[command.index('--show')+1],'exp,obs')
                stem=cwd/command[command.index('--output')+1]
                for ext in ('.pdf','.png'):stem.with_suffix(ext).write_text('plot')
            with patch.object(base,'run_command',side_effect=plot):
                outputs=batch.result_tables(dict(batch=temp),opt)
            data=json.loads((Path(temp)/'limits_Run2Run3_alpha.json').read_text())
            self.assertEqual(data['30.0']['obs'],.025)
            self.assertIn('obs',(Path(temp)/'expected_limits.csv').read_text().splitlines()[0])
            blind=fs.make_parser().parse_args(['--masses','30','--limits','--fit-components','none','--output-base',temp])
            with self.assertRaisesRegex(ValueError,'matching observation mode'):
                fs.run_expected_limits(model,directory,blind)


if __name__=='__main__':unittest.main()
