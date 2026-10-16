import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import repair_bias_intervals as repair


class RepairTests(unittest.TestCase):
    def test_command_reuses_exact_toy_and_range(self):
        command = repair.retry_command(Path('/workspace.root'), Path('/toys.root'),
                                       70, 500, 96, .94921875, (-2, 100), 1, .01)
        def opt(name):
            return command[command.index(name)+1]
        self.assertEqual(opt('--pickToy'), '96')
        self.assertEqual(opt('-t'), '500')
        self.assertEqual(opt('-m'), '70')
        self.assertEqual(opt('--toysFile'), '/toys.root')
        self.assertEqual(opt('--setParameterRanges'), 'r=-2,100')
        self.assertNotIn('GenerateOnly', command)

    def test_profile_must_match_likelihood_target(self):
        repair.validate_point({'r': .2, 'deltaNLL': .49447}, .2, .49447, .01)
        with self.assertRaises(RuntimeError):
            repair.validate_point({'r': .2, 'deltaNLL': 0}, .2, .49447, .01)
        with self.assertRaises(RuntimeError):
            repair.validate_point({'r': .3, 'deltaNLL': .49447}, .2, .49447, .01)

    def test_runner_accepts_mass70_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'fit'
            def execute(*args, **kwargs):
                (directory / 'higgsCombine.Test.MultiDimFit.mH70.123456.root').touch()
                return types.SimpleNamespace(returncode=0)
            with patch.object(repair.verify.subprocess, 'run', side_effect=execute), \
                 patch.object(repair.verify, 'read_rows', return_value=['rows']):
                self.assertEqual(repair.verify.run(['combine'], directory, []), ['rows'])

    def test_success_and_failure_control_final_summaries(self):
        # Exercise the orchestration without ROOT/Combine. Actual numerical
        # minimisations must be run in the original CMSSW environment.
        for endpoint_pass in (True, False):
            with self.subTest(endpoint_pass=endpoint_pass), tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp); source = base / 'source'; (source/'bias').mkdir(parents=True)
                output = base/'output'; output.mkdir()
                workspace = source/'workspace_Run2Run3_M70.root'; workspace.touch()
                ensembles, points, fingerprints, original = [], [], {}, {}
                for index, inj in enumerate((0, 1, 2)):
                    tag = f'r{inj}'
                    fit = source/'bias'/f'higgsCombine.ReviewBiasFit_Run2Run3_M70_{tag}.MultiDimFit.mH70.123456.root'
                    fit.touch(); fingerprints[str(fit)] = repair.audit.sha256(fit)
                    (source/'bias'/f'higgsCombine.ReviewBiasGen_Run2Run3_M70_{tag}.GenerateOnly.mH70.12345.root').touch()
                    rows = []
                    for toy in (1, 2):
                        interval = (-2, .3, .8) if index == 1 and toy == 1 else (-.2, .3, .8)
                        rows.extend({'toy': toy, 'quantile': q, 'r': r} for (_, q), r in zip(repair.audit.QUANTILES, interval))
                    original[str(fit)] = rows
                    ensemble = repair.audit.audit_rows(rows, 2, (-2, 100))
                    ensemble.update(tag=tag, injected_r=inj, fit_file=str(fit), r_range=[-2, 100])
                    ensembles.append(ensemble)
                    points.append({'injected_r': inj, 'fit_file': str(fit), 'n_toys_requested': 2})
                summary = source/'bias/bias_summary_Run2Run3_M70.json'
                summary.write_text(json.dumps(points)); fingerprints[str(summary)] = repair.audit.sha256(summary)
                audit_file = base/'audit.json'
                audit_file.write_text(json.dumps({'errors': [], 'ensembles': ensembles, 'input_sha256': fingerprints}))
                def run(*args):
                    return [{'toy': 1, 'quantile': q, 'r': r} for (_, q), r in zip(repair.audit.QUANTILES, (-.2, .3, .8))]
                def profile(command, info, toy, value, directory, history, tolerance):
                    delta = 0 if directory.name == 'center' else .49447 if endpoint_pass else 0
                    return {'r': value, 'deltaNLL': delta}
                root = types.SimpleNamespace(gROOT=types.SimpleNamespace(SetBatch=lambda _: None),
                                             Math=types.SimpleNamespace(chisquared_quantile=lambda *_: .98894))
                argv = ['repair_bias_intervals.py', '--audit', str(audit_file), '--source-dir', str(source)]
                with patch.object(sys, 'argv', argv), patch.dict(sys.modules, ROOT=root), \
                     patch.object(repair.tempfile, 'mkdtemp', return_value=str(output)), \
                     patch.object(repair.audit, 'read_rows', side_effect=lambda p: original[str(p)]), \
                     patch.object(repair.verify, 'run', side_effect=run), \
                     patch.object(repair.verify, 'fit_info', return_value={'status': 0, 'covQual': 3, 'EDM': .00001}), \
                     patch.object(repair.verify, 'profile_saved_reference', side_effect=profile):
                    code = repair.main()
                self.assertEqual(code, 0 if endpoint_pass else 2)
                validation = json.loads((output/'validation.json').read_text())
                self.assertEqual(validation['passed'], endpoint_pass)
                self.assertEqual((output/'bias_summary_Run2Run3_M70.json').exists(), endpoint_pass)
                self.assertEqual((output/'merged_toys.json').exists(), endpoint_pass)
                if endpoint_pass:
                    data = json.loads((output/'merged_toys.json').read_text())
                    self.assertEqual(data['r1']['1']['interval'], [-.2, .3, .8])
                    self.assertTrue(data['r1']['1']['repaired'])


if __name__ == '__main__':
    unittest.main()
