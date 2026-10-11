"""Regression tests for missing scans, tiny intervals and physical endpoints."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import limit_workflow as w


def parameter(name="theta", fit=None, r=None):
    return {"name": name, "fit": fit or [-1., 0., 1.],
            "r": r or [0., 0., 0.], "prefit": [-1., 0., 1.], "impact_r": 0.}


class ImpactRepairTests(unittest.TestCase):
    def test_zero_poi_impact_is_allowed(self):
        self.assertEqual(w.impact_interval_issues({"params": [parameter()]}, ["theta"]), ([], {}))

    def test_tiny_nonzero_interval_is_allowed(self):
        p = parameter(fit=[-3e-13, 0., 3e-13])
        self.assertEqual(w.impact_interval_issues({"params": [p]}, ["theta"]), ([], {}))

    def test_missing_and_collapsed_are_separate(self):
        p = parameter(fit=[-1., 0., 0.])
        self.assertEqual(w.impact_interval_issues({"params": [p]}, ["theta", "missing"]),
                         (["missing"], {"theta": ["upper"]}))

    def test_boundary_uses_conditional_r_and_recomputes_impact(self):
        p = parameter(fit=[2., 2., 5.], r=[0., 0., -0.2])
        w.apply_impact_boundary_endpoint(p, 0., 0.3, 0.1)
        self.assertEqual(p["fit"], [0., 2., 5.])
        self.assertEqual(p["r"], [0.3, 0., -0.2])
        self.assertEqual(p["impact_r"], 0.3)
        self.assertEqual(w.impact_interval_issues({"params": [p]}, ["theta"]), ([], {}))

    def test_centre_at_bound_needs_explicit_profile_evidence(self):
        p = parameter(fit=[0., 0., 1.])
        self.assertEqual(w.impact_interval_issues({"params": [p]}, ["theta"])[1], {"theta": ["lower"]})
        w.apply_impact_boundary_endpoint(p, 0., 0., 0.)
        self.assertEqual(w.impact_interval_issues({"params": [p]}, ["theta"]), ([], {}))

    def test_crossing_before_bound_cannot_be_labelled_truncated(self):
        with self.assertRaises(w.WorkflowError):
            w.apply_impact_boundary_endpoint(parameter(fit=[2., 2., 5.]), 0., 0., 0.7)

    def test_invalid_and_duplicates_fail(self):
        p = parameter(fit=[-1., 0., math.nan])
        self.assertEqual(w.impact_interval_issues({"params": [p]}, ["theta"])[1], {"theta": ["invalid"]})
        with self.assertRaises(w.WorkflowError):
            w.impact_interval_issues({"params": [parameter(), parameter()]}, ["theta"])

    def test_repair_flag_cannot_rebuild_cards(self):
        args = w.build_parser().parse_args(["--repair-impacts", "--task", "impacts"])
        with self.assertRaises(w.WorkflowError):
            w.validate_args(args)

    def test_failed_retry_does_not_overwrite_original_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = w.build_parser().parse_args(["--stage", "run", "--task", "impacts"])
            p = {"POIs": [{"name": "r", "fit": [-1., 0., 1.]}],
                 "params": [parameter(fit=[-1., 0., 0.])]}
            original = json.dumps(p)
            internal = out / "impacts.json"
            internal.write_text(original)
            (out / "higgsCombine_initialFit_.Impacts_Run2_alpha_M70.MultiDimFit.mH70.root").touch()
            def fake_run(command, cwd, **kwargs):
                if "-o" in command:
                    (cwd / "retry_internal.json").write_text(original)
                return 0
            with patch.object(w, "impact_workspace_parameters", return_value={"theta": (-7., 7.)}), \
                    patch.object(w, "run_command", side_effect=fake_run):
                with self.assertRaises(w.WorkflowError):
                    w.repair_impact_payload(args, out / "card.txt", "Run2", "70", 70.,
                                            out, out / "workspace.root", internal, copy.deepcopy(p), -2., 10.)
            self.assertEqual(internal.read_text(), original)

    def test_repair_prepare_does_not_clean_existing_scans(self):
        args = w.build_parser().parse_args(["--stage", "run", "--task", "impacts", "--repair-impacts", "--r-max", "10"])
        with patch.object(w, "extract_mass_from_card", return_value=("70", 70.)), \
                patch.object(w, "clean_mass_outputs") as clean, \
                patch.object(w, "alpha_unit_from_card", return_value=1.):
            w.prepare_card_run(args, Path("card.txt"), "Run2", Path("output"))
        clean.assert_not_called()

    def test_resume_reuses_three_scans_and_retries_only_missing_conditional_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = w.build_parser().parse_args(["--stage", "run", "--task", "impacts"])
            names = ["tiny", "jer", "mcstat", "prefiring"]
            payload = {"POIs": [{"name": "r", "fit": [-1., 0., 1.]}],
                       "params": [parameter(n, [-1., 0., 0.]) for n in names]}
            internal = out / "impacts.json"
            internal.write_text(json.dumps(payload))
            workspace = out / "workspace.root"
            workspace.write_bytes(b"workspace")
            initial = out / "higgsCombine_initialFit_.Impacts_Run2_alpha_M70.MultiDimFit.mH70.root"
            initial.write_bytes(b"initial")
            directory = out / "impact_repair_M70"
            directory.mkdir()
            scans = {n: directory / f"higgsCombine_paramFit_.Impacts_Run2_alpha_M70_{n}.MultiDimFit.mH70.root"
                     for n in names}
            for n in names[:3]:
                scans[n].write_bytes(b"good")
            scans["prefiring"].write_bytes(b"incomplete")
            attempts = []
            def fake_run(command, cwd, **kwargs):
                if "--doFits" in command:
                    self.assertEqual(command[command.index("--named") + 1], "prefiring")
                    self.assertFalse(scans["prefiring"].exists())
                    strategy = command[command.index("--cminDefaultMinimizerStrategy") + 1]
                    attempts.append(strategy)
                    self.assertEqual(command[command.index("--setRobustFitStrategy") + 1], strategy)
                    self.assertEqual(command.count("--cminFallbackAlgo"), 2)
                    self.assertEqual(command[command.index("-v") + 1], "2")
                    scans["prefiring"].write_bytes(b"good" if strategy == "1" else b"incomplete")
                else:
                    good = [parameter(n, [-3e-13, 0., 3e-13] if n == "tiny" else None)
                            for n in names if scans[n].exists() and scans[n].read_bytes() == b"good"]
                    (cwd / "retry_internal.json").write_text(json.dumps({"params": good}))
                return 0
            with patch.object(w, "impact_workspace_parameters", return_value={n: (-7., 7.) for n in names}), \
                    patch.object(w, "run_command", side_effect=fake_run):
                result = w.repair_impact_payload(args, out / "card", "Run2", "70", 70.,
                                                out, workspace, internal, payload, -2., 10.)
            self.assertEqual(attempts, ["2", "1"])
            self.assertEqual(len(result["params"]), 4)
            for n in names[:3]:
                self.assertEqual(scans[n].read_bytes(), b"good")

    def test_rebuilt_workspace_discards_old_retry_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = w.build_parser().parse_args(["--stage", "run", "--task", "impacts"])
            payload = {"POIs": [{"name": "r", "fit": [-1., 0., 1.]}],
                       "params": [parameter(fit=[-1., 0., 0.])]}
            internal = out / "impacts.json"
            internal.write_text(json.dumps(payload))
            workspace = out / "workspace.root"
            workspace.write_bytes(b"new workspace")
            (out / "higgsCombine_initialFit_.Impacts_Run2_alpha_M70.MultiDimFit.mH70.root").touch()
            directory = out / "impact_repair_M70"
            directory.mkdir()
            scan = directory / "higgsCombine_paramFit_.Impacts_Run2_alpha_M70_theta.MultiDimFit.mH70.root"
            scan.write_bytes(b"old result")
            os.utime(scan, (1., 1.))
            attempts = []
            def fake_run(command, cwd, **kwargs):
                if "--doFits" in command:
                    attempts.append(command)
                    self.assertFalse(scan.exists())
                    scan.write_bytes(b"new result")
                else:
                    if not attempts:
                        self.assertFalse(scan.exists())
                    (cwd / "retry_internal.json").write_text(json.dumps({"params": [parameter()] if attempts else []}))
                return 0
            with patch.object(w, "impact_workspace_parameters", return_value={"theta": (-7., 7.)}), \
                    patch.object(w, "run_command", side_effect=fake_run):
                w.repair_impact_payload(args, out / "card", "Run2", "70", 70.,
                                       out, workspace, internal, payload, -2., 10.)
            self.assertEqual(len(attempts), 1)
            self.assertEqual(scan.read_bytes(), b"new result")

    def test_successful_boundary_repair_retains_good_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = w.build_parser().parse_args(["--stage", "run", "--task", "impacts"])
            name = "CMS_NPS26009_bckgShape_QCD_BJetOS_2018"
            good = parameter("good")
            payload = {"POIs": [{"name": "r", "fit": [-1., 0., 1.]}],
                       "params": [good, parameter(name, [2., 2., 5.], [0., 0., -0.2])]}
            internal = out / "impacts.json"
            internal.write_text(json.dumps(payload))
            with patch.object(w, "impact_workspace_parameters", return_value={name: (0., 100.), "good": (-7., 7.)}), \
                    patch.object(w, "run_command", return_value=0) as run, \
                    patch.object(w, "read_boundary_impact_result", return_value=(0.3, 0.1)):
                result = w.repair_impact_payload(args, out / "card", "Run2", "70", 70.,
                                                out, out / "workspace", internal, payload, -2., 10.)
            run.assert_called_once()
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--fixedPointPOIs")+1], name+"=0")
            self.assertEqual(command[command.index("-P")+1], name)
            self.assertEqual(command[command.index("--floatOtherPOIs")+1], "1")
            self.assertNotIn("--freezeParameters", command)
            self.assertEqual(next(p for p in result["params"] if p["name"] == "good"), good)
            fixed = next(p for p in result["params"] if p["name"] == name)
            self.assertEqual(fixed["fit"][0], 0.)
            self.assertEqual(fixed["r"][0], 0.3)


if __name__ == "__main__":
    unittest.main()
