"""Explicit profiles must measure an impact, even in tiny absolute units."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import limit_workflow as w

NAME = 'CMS_NPS26009_stat_QCD_BJetOS_2022EE'
WIDTH = 3.574945593e-13


def param():
    return {'name': NAME, 'type': 'Gaussian', 'prefit': [-WIDTH, 0., WIDTH],
            'fit': [0., 0., 0.], 'r': [1., 1., 1.], 'impact_r': 0.}


class File:
    def __init__(self, path):
        self.rows = json.loads(Path(path).read_text())
        self.tree = SimpleNamespace(GetEntries=lambda: len(self.rows),
                                    GetBranch=lambda key: key in self.rows[0])
        def entry(i):
            for key, value in self.rows[i].items():
                setattr(self.tree, key, value)
        self.tree.GetEntry = entry
    def IsZombie(self): return False
    def Get(self, _): return self.tree
    def Close(self): pass


class TinyProfileTests(unittest.TestCase):
    def reader(self):
        return patch.object(w, 'RootReader', return_value=SimpleNamespace(
            ROOT=SimpleNamespace(TFile=SimpleNamespace(Open=File))))

    def test_candidate_requires_collapsed_qcd_stat_gaussian_and_finite_symmetric_prior(self):
        p = param()
        self.assertTrue(w.tiny_gaussian_impact_candidate(p, (-7 * WIDTH, 7 * WIDTH)))
        for changes in ({'name': 'arbitrary'}, {'type': 'Unconstrained'},
                        {'fit': [-WIDTH, 0., WIDTH]}, {'prefit': [-1., 0., 1.]},
                        {'prefit': [0., 0., 0.]}, {'prefit': [-WIDTH, 0., 2 * WIDTH]}):
            with self.subTest(changes=changes):
                self.assertFalse(w.tiny_gaussian_impact_candidate({**p, **changes}, (-7., 7.)))

    def run_profile(self, output, commands, wrong_coordinate=False, wrong_delta=False):
        def run(command, cwd, **kwargs):
            commands.append(command)
            value = float(command[command.index('--fixedPointPOIs') + 1].split('=')[1])
            rows = [{NAME: 0., 'r': 1., 'deltaNLL': 0.},
                    {NAME: 0. if wrong_coordinate else value,
                     'r': 1. + .1 * value / WIDTH,
                     'deltaNLL': 0. if wrong_delta else .5 * (value / WIDTH) ** 2}]
            (cwd / 'higgsCombine.TinyProfile.MultiDimFit.mH70.root').write_text(json.dumps(rows))
            kwargs['stdout_path'].write_text('saved conditional fit')
            return 0
        return patch.object(w, 'run_command', side_effect=run)

    def test_tiny_width_does_not_imply_zero_impact_and_injection_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, commands = param(), []
            blind = ['-t', '-1', '--expectSignal', '1', '--setParameters', 'r=1']
            with self.reader(), self.run_profile(tmp, commands):
                w.repair_tiny_gaussian_impact(p, Path('workspace.root'), Path(tmp), '70', 0., 100.,
                                             ['--rMin', '0', '--rMax', '100'], blind, 1.)
            self.assertEqual(p['fit'], [-WIDTH, 0., WIDTH])
            self.assertAlmostEqual(p['impact_r'], .1)
            self.assertEqual(w.impact_interval_issues({'params': [p]}, [NAME]), ([], {}))
            self.assertEqual(len(commands), 3)
            for command in commands:
                self.assertEqual(command[command.index('--expectSignal') + 1], '1')
                self.assertEqual(command[command.index('--setParameters') + 1], 'r=1')
                self.assertEqual(command[command.index('--floatOtherPOIs') + 1], '1')
                self.assertNotIn('--freezeParameters', command)

    def test_zero_coordinate_cannot_pass_as_a_tiny_nonzero_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, commands = param(), []
            before = copy.deepcopy(p)
            with self.reader(), self.run_profile(tmp, commands, wrong_coordinate=True):
                with self.assertRaisesRegex(w.WorkflowError, 'lower profile failed'):
                    w.repair_tiny_gaussian_impact(p, Path('workspace.root'), Path(tmp), '70', 0., 100.,
                                                 ['--rMin', '0', '--rMax', '100'], [], 1.)
            self.assertEqual(p, before)

    def test_wrong_likelihood_rise_is_rejected_even_with_valid_file_and_zero_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, commands = param(), []
            with self.reader(), self.run_profile(tmp, commands, wrong_delta=True):
                with self.assertRaisesRegex(w.WorkflowError, 'lower profile failed'):
                    w.repair_tiny_gaussian_impact(p, Path('workspace.root'), Path(tmp), '70', 0., 100.,
                                                 ['--rMin', '0', '--rMax', '100'], [], 1.)

    def test_missing_conditional_entry_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'profile.root'
            path.write_text(json.dumps([{NAME: 0., 'r': 1., 'deltaNLL': 0.}]))
            with self.reader(), self.assertRaisesRegex(w.WorkflowError, 'initial and conditional'):
                w.read_tiny_impact_profile(path, NAME, WIDTH, WIDTH, .5, 0., 100.)

    def test_repair_integration_preserves_the_card_model_and_records_measured_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = w.build_parser().parse_args(['--stage', 'run', '--task', 'impacts'])
            args.impact_expect_signal = 1.
            args.impact_fit_name = '.ReviewSBImpacts_Run2Run3_M70'
            p = {'POIs': [{'name': 'r', 'fit': [.48, 1., 1.53]}], 'params': [param()]}
            internal = out / 'impacts.json'
            internal.write_text(json.dumps(p))
            commands = []
            with self.reader(), self.run_profile(out, commands), \
                 patch.object(w, 'impact_workspace_parameters', return_value={NAME: (-7 * WIDTH, 7 * WIDTH)}):
                result = w.repair_impact_payload(args, out / 'card', 'Run2Run3', '70', 70.,
                                                 out, out / 'workspace', internal, p, 0., 100.)
            self.assertEqual(result['params'][0]['prefit'], [-WIDTH, 0., WIDTH])
            self.assertEqual(result['params'][0]['type'], 'Gaussian')
            self.assertTrue(all('--doFits' not in command for command in commands))
            report = json.loads((out / 'impact_repair_M70/validation.json').read_text())
            self.assertEqual(report['tiny_gaussian_profiled'], [NAME])
            self.assertTrue(report['passed'])


if __name__ == '__main__':
    unittest.main()
