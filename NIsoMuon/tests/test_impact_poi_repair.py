"""Regression coverage for finite singles output that did not find a crossing."""
import copy,json,math,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import limit_workflow as w

def payload(fit):
    return {'POIs':[{'name':'r','fit':fit}],
            'params':[{'name':'theta','fit':[-1.,0.,1.],
                       'prefit':[-1.,0.,1.],'r':[.1,0.,-.1],'impact_r':.1}]}

class InitialPOIRepairTests(unittest.TestCase):
    def test_all_three_uploaded_cap_failures_are_rejected(self):
        for low in [-1.4401483535766602,-1.1222114562988281,-.3838861882686615]:
            self.assertEqual(w.impact_poi_interval_issues(payload([low,0.,100.]),-5.,100.),
                             ['POI upper endpoint reaches rMax'])

    def test_physical_lower_boundary_is_allowed_but_upper_cap_is_not(self):
        self.assertEqual(w.impact_poi_interval_issues(payload([0.,0.,1.]),0.,100.),[])
        self.assertTrue(w.impact_poi_interval_issues(payload([0.,0.,100.]),0.,100.))
        self.assertTrue(w.impact_poi_interval_issues(payload([-5.,0.,1.]),-5.,100.,True))

    def test_missing_nonfinite_and_collapsed_poi_intervals_fail(self):
        for p in [{},payload([-1.,0.,math.nan]),payload([0.,0.,0.])]:
            self.assertTrue(w.impact_poi_interval_issues(p,-5.,100.))

    def test_valid_cached_expanded_lower_range_does_not_refit(self):
        args=w.build_parser().parse_args(['--allow-negative-r'])
        p=payload([-15.,0.,14.])
        with patch.object(w,'run_command') as run:
            report=w.repair_impact_initial_poi(args,Path('card'),'Run2','12',12.,
                       Path('out'),Path('workspace'),Path('input.json'),p,-5.,100.)
        run.assert_not_called();self.assertTrue(report['passed']);self.assertFalse(report['repaired'])

    def test_cached_physical_bonly_result_cannot_pass_negative_r_request(self):
        args=w.build_parser().parse_args(['--allow-negative-r','--r-min','-50'])
        for fit in ([0.,.002656,17.25235], [0.,0.,1.], [.01,.02,1.]):
            with self.subTest(fit=fit),patch.object(w,'run_command') as run:
                with self.assertRaisesRegex(w.WorkflowError,'nuisance scans cannot be reused'):
                    w.repair_impact_initial_poi(args,Path('card'),'Run2Run3','12',12.,
                        Path('out'),Path('workspace'),Path('input.json'),payload(fit),-50.,100.)
                run.assert_not_called()

    def test_explicit_physical_bonly_diagnostic_remains_supported(self):
        args=w.build_parser().parse_args([])
        with patch.object(w,'run_command') as run:
            report=w.repair_impact_initial_poi(args,Path('card'),'Run2Run3','12',12.,
                Path('out'),Path('workspace'),Path('input.json'),payload([0.,0.,1.]),0.,100.)
        run.assert_not_called();self.assertTrue(report['passed'])

    def test_valid_sb_interval_is_not_required_to_cross_zero(self):
        args=w.build_parser().parse_args(['--allow-negative-r'])
        args.impact_expect_signal=15.5625
        with patch.object(w,'run_command') as run:
            report=w.repair_impact_initial_poi(args,Path('card'),'Run2Run3','20',20.,
                Path('out'),Path('workspace'),Path('input.json'),payload([6.47,15.5625,23.79]),-50.,100.)
        run.assert_not_called();self.assertTrue(report['passed'])

    def setup_case(self,directory,mode='blind',injection=0.):
        out=Path(directory);card=out/'card.txt';card.write_text('# alpha_internal_unit = 0.0003\n')
        workspace=out/'workspace.root';workspace.touch()
        internal=out/'impacts_internal.json';p=payload([injection-.4,injection,100.])
        internal.write_text(json.dumps(p));args=w.build_parser().parse_args(['--mode',mode,'--allow-negative-r'])
        args.impact_expect_signal=injection
        return out,card,workspace,internal,p,args

    def test_repair_only_initial_fit_and_preserve_nuisance_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,card,workspace,internal,p,args=self.setup_case(tmp)
            original=p['params'].copy();calls=[]
            def run(command,cwd,**kwargs):
                calls.append(command)
                (cwd/'higgsCombine_initialFit_.Impacts_Run3_alpha_M50.MultiDimFit.mH50.root').touch()
                return 0
            with patch.object(w,'run_command',side_effect=run),patch.object(w,'read_impact_initial_poi',return_value=[-.384,0.,.359]):
                report=w.repair_impact_initial_poi(args,card,'Run3','50',50.,out,workspace,internal,p,-2.,10.)
            self.assertTrue(report['repaired']);self.assertEqual(p['params'],original)
            self.assertEqual(p['POIs'][0]['fit'],[-.384,0.,.359])
            self.assertEqual(json.loads(internal.read_text())['POIs'][0]['fit'],[-.4,0.,100.])
            self.assertEqual(len(calls),1);self.assertIn('--doInitialFit',calls[0]);self.assertNotIn('--doFits',calls[0])
            self.assertEqual(calls[0][calls[0].index('--setRobustFitStrategy')+1],'2')
            self.assertIn('--expectSignal',calls[0])

    def test_second_attempt_uses_smaller_range_only_after_failed_crossing(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,card,workspace,internal,p,args=self.setup_case(tmp);calls=[]
            def run(command,cwd,**kwargs):
                calls.append(command);(cwd/'higgsCombine_initialFit_.Impacts_Run3_alpha_M50.MultiDimFit.mH50.root').touch();return 0
            with patch.object(w,'run_command',side_effect=run),patch.object(w,'read_impact_initial_poi',side_effect=[[-.4,0.,100.],[-.384,0.,.359]]):
                w.repair_impact_initial_poi(args,card,'Run3','50',50.,out,workspace,internal,p,-2.,100.)
            self.assertEqual([float(c[c.index('--rMax')+1]) for c in calls],[100.,2.])

    def test_failed_repair_retains_original_json_and_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,card,workspace,internal,p,args=self.setup_case(tmp);original=internal.read_text();saved=copy.deepcopy(p)
            with patch.object(w,'run_command',return_value=1):
                with self.assertRaises(w.WorkflowError):w.repair_impact_initial_poi(args,card,'Run3','50',50.,out,workspace,internal,p,-2.,100.)
            self.assertEqual(internal.read_text(),original);self.assertEqual(p,saved)

    def test_material_change_of_centre_cannot_reuse_conditional_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,card,workspace,internal,p,args=self.setup_case(tmp)
            with patch.object(w,'run_command',return_value=0),patch.object(w,'read_impact_initial_poi',return_value=[-.2,.1,.5]):
                with self.assertRaisesRegex(w.WorkflowError,'centre shifted'):w.repair_impact_initial_poi(args,card,'Run3','50',50.,out,workspace,internal,p,-2.,10.)

    def test_review_sb_repair_keeps_original_signal_injection(self):
        with tempfile.TemporaryDirectory() as tmp:
            out,card,workspace,internal,p,args=self.setup_case(tmp,injection=15.5625);calls=[]
            args.impact_fit_name='.ReviewSBImpacts_Run2Run3_M20'
            def run(command,cwd,**kwargs):
                calls.append(command);(cwd/'higgsCombine_initialFit_.ReviewSBImpacts_Run2Run3_M20.MultiDimFit.mH20.root').touch();return 0
            with patch.object(w,'run_command',side_effect=run),patch.object(w,'read_impact_initial_poi',return_value=[6.47,15.5625,23.79]):
                w.repair_impact_initial_poi(args,card,'Run2Run3','20',20.,out,workspace,internal,p,0.,100.)
            self.assertEqual(calls[0][calls[0].index('--expectSignal')+1],'15.5625')
            self.assertIn('r=15.5625',calls[0])

if __name__=='__main__':unittest.main()
