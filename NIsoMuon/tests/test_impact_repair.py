"""Regression tests for missing scans, tiny intervals and physical endpoints."""
import copy
import hashlib
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
import combine_review as review


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
                elif "--doFits" in command:
                    (cwd / "higgsCombine_paramFit_.Impacts_Run2_alpha_M70_theta.MultiDimFit.mH70.root").touch()
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

    def test_empty_retry_directory_fits_before_collecting(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = w.build_parser().parse_args(["--stage", "run", "--task", "impacts"])
            name = "CMS_NPS26009_pdf_NNPDF31"
            payload = {"POIs": [{"name": "r", "fit": [-1., 0., 1.]}],
                       "params": [parameter(name, [-1., 0., 0.])]}
            internal = out / "impacts.json"
            internal.write_text(json.dumps(payload))
            (out / "higgsCombine_initialFit_.Impacts_Run3_alpha_M70.MultiDimFit.mH70.root").touch()
            calls = []
            def fake_run(command, cwd, **kwargs):
                calls.append("fit" if "--doFits" in command else "collect")
                scan = cwd / f"higgsCombine_paramFit_.Impacts_Run3_alpha_M70_{name}.MultiDimFit.mH70.root"
                if "--doFits" in command:
                    scan.touch()
                else:
                    if not scan.exists():
                        raise w.WorkflowError("ROOT cannot open the missing paramFit file")
                    (cwd / "retry_internal.json").write_text(json.dumps({"params": [parameter(name)]}))
                return 0
            with patch.object(w, "impact_workspace_parameters", return_value={name: (-7., 7.)}), \
                    patch.object(w, "run_command", side_effect=fake_run):
                result = w.repair_impact_payload(args, out / "card", "Run3", "70", 70.,
                                                out, out / "workspace", internal, payload, -0.5, 10.)
            self.assertEqual(calls, ["fit", "collect"])
            self.assertEqual(result["params"][0]["fit"], [-1., 0., 1.])

    def test_partial_cache_collects_only_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = w.build_parser().parse_args(["--stage", "run", "--task", "impacts"])
            names = ["cached", "missing"]
            payload = {"POIs": [{"name": "r", "fit": [-1., 0., 1.]}],
                       "params": [parameter(n, [-1., 0., 0.]) for n in names]}
            internal = out / "impacts.json"
            internal.write_text(json.dumps(payload))
            (out / "higgsCombine_initialFit_.Impacts_Run2_alpha_M70.MultiDimFit.mH70.root").touch()
            directory = out / "impact_repair_M70"
            directory.mkdir()
            scans = {n: directory / f"higgsCombine_paramFit_.Impacts_Run2_alpha_M70_{n}.MultiDimFit.mH70.root"
                     for n in names}
            scans["cached"].touch()
            collections = []
            def fake_run(command, cwd, **kwargs):
                selected = command[command.index("--named") + 1].split(",")
                if "--doFits" in command:
                    self.assertEqual(selected, ["missing"])
                    scans["missing"].touch()
                else:
                    collections.append(selected)
                    self.assertTrue(all(scans[n].exists() for n in selected))
                    (cwd / "retry_internal.json").write_text(json.dumps({"params": [parameter(n) for n in selected]}))
                return 0
            with patch.object(w, "impact_workspace_parameters", return_value={n: (-7., 7.) for n in names}), \
                    patch.object(w, "run_command", side_effect=fake_run):
                w.repair_impact_payload(args, out / "card", "Run2", "70", 70.,
                                       out, out / "workspace", internal, payload, -2., 10.)
            self.assertEqual(collections, [["cached"], ["cached", "missing"]])

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

    def test_sb_retry_recovers_upper_crossing_then_profiles_lower_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = w.build_parser().parse_args(["--stage", "run", "--task", "impacts"])
            args.impact_expect_signal = 1.25
            args.impact_fit_name = ".ReviewSBImpacts_Run2Run3_M70"
            qcd = "CMS_NPS26009_bckgShape_QCD_BJetOS_2016postVFP"
            pdf = "CMS_NPS26009_pdf_NNPDF31"
            good = parameter("good", r=[1.1, 1.25, 1.4])
            payload = {"POIs": [{"name": "r", "fit": [.4, 1.25, 2.]}],
                       "params": [good, parameter(qcd, [.1, .1, .1], [1.25]*3)]}
            internal = out / "impacts.json"
            internal.write_text(json.dumps(payload))
            initial = out / f"higgsCombine_initialFit_{args.impact_fit_name}.MultiDimFit.mH70.root"
            initial.touch()
            commands = []
            def fake_run(command, cwd, **kwargs):
                commands.append(command)
                if "--doFits" in command:
                    selected = command[command.index("--named")+1].split(",")
                    self.assertEqual(set(selected), {pdf, qcd})
                    for n in selected:
                        (cwd / f"higgsCombine_paramFit_{args.impact_fit_name}_{n}.MultiDimFit.mH70.root").touch()
                elif "-o" in command:
                    repaired = [parameter(pdf, r=[1.2, 1.25, 1.3]),
                                parameter(qcd, [.1, .1, 3.], [1.25, 1.25, 1.1])]
                    (cwd / "retry_internal.json").write_text(json.dumps({"params": repaired}))
                return 0
            bounds = {pdf: (-7., 7.), qcd: (0., 20.), "good": (-7., 7.)}
            with patch.object(w, "impact_workspace_parameters", return_value=bounds), \
                    patch.object(w, "run_command", side_effect=fake_run), \
                    patch.object(w, "read_boundary_impact_result", return_value=(1.3, .02)):
                result = w.repair_impact_payload(args, out / "card", "Run2Run3", "70", 70.,
                                               out, out / "workspace", internal, payload, 0., 100.)
            fits = [c for c in commands if "--doFits" in c or "--fixedPointPOIs" in c]
            self.assertEqual(len(fits), 2)
            for c in fits:
                self.assertEqual(c[c.index("--expectSignal")+1], "1.25")
                self.assertEqual(c[c.index("--setParameters")+1], "r=1.25")
            params = {p["name"]: p for p in result["params"]}
            self.assertEqual(params["good"], good)
            self.assertEqual(params[qcd]["fit"], [0., .1, 3.])
            self.assertEqual(params[qcd]["r"][0], 1.3)
            self.assertEqual(w.impact_interval_issues(result, bounds), ([], {}))
            report = json.loads((out / "impact_repair_M70/validation.json").read_text())
            self.assertEqual(report["expected_signal"], 1.25)
            self.assertEqual(report["boundary"], [qcd])


class ReviewSBRepairTests(unittest.TestCase):
    def make_inputs(self, root):
        args = review.build_parser().parse_args(["20", "--task", "sb-impacts",
            "--repair-sb-impacts", "--output-base", str(root)])
        output = review.outdir(args)
        output.mkdir(parents=True)
        card = root / "card.txt"
        card.write_text("# alpha_internal_unit = 0.001\n")
        workspace = review.workspace_path(args)
        workspace.write_bytes(b"workspace")
        initial = output / "higgsCombine_initialFit_.ReviewSBImpacts_Run2Run3_M20.MultiDimFit.mH20.root"
        initial.write_bytes(b"initial fit")
        os.utime(workspace, (1., 1.))
        os.utime(initial, (2., 2.))
        metadata = {"mass": 20., "target": "Run2Run3", "injected_r": 15.5625}
        (output / "sb_impacts_metadata_M20.json").write_text(json.dumps(metadata))
        (root / "run_manifest.json").write_text(json.dumps({"cards": {"20": {
            "sha256": hashlib.sha256(card.read_bytes()).hexdigest()}}}))
        return args, card, workspace

    def test_repair_uses_original_injection_and_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, card, workspace = self.make_inputs(Path(tmp))
            self.assertEqual(review.sb_repair_inputs(args, card), (workspace, 15.5625))
            args.inject_r = 1.
            with self.assertRaisesRegex(RuntimeError, "original.*injection"):
                review.sb_repair_inputs(args, card)
            args.inject_r = None
            card.write_text("changed card")
            with self.assertRaisesRegex(RuntimeError, "card differs"):
                review.sb_repair_inputs(args, card)

    def test_repair_rejects_workspace_newer_than_initial_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, card, workspace = self.make_inputs(Path(tmp))
            os.utime(workspace, (3., 3.))
            with self.assertRaisesRegex(RuntimeError, "newer than"):
                review.sb_repair_inputs(args, card)

    def test_review_repair_does_not_rebuild_workspace_or_initial_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, card, workspace = self.make_inputs(Path(tmp))
            with patch.object(review, "require_command", return_value="combineTool.py"), \
                    patch.object(review, "ensure_workspace") as build, \
                    patch.object(review, "produce_sb_scans") as scans, \
                    patch.object(review, "repair_sb_payload") as repair, \
                    patch.object(review.shutil, "which", return_value=None):
                review.task_sb_impacts(args, card)
            build.assert_not_called()
            scans.assert_not_called()
            self.assertEqual(repair.call_args.args[-1], 15.5625)

    def test_failed_fresh_repair_preserves_metadata_and_skips_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, card, workspace = self.make_inputs(Path(tmp))
            args.repair_sb_impacts = False
            with patch.object(review, "require_command", return_value="combineTool.py"), \
                    patch.object(review, "ensure_workspace", return_value=workspace), \
                    patch.object(review, "impact_injection", return_value=(15.5625, .0155625)), \
                    patch.object(review, "produce_sb_scans"), \
                    patch.object(review, "repair_sb_payload", side_effect=RuntimeError("Incomplete impacts")), \
                    patch.object(review.shutil, "which") as plot:
                with self.assertRaisesRegex(RuntimeError, "Incomplete"):
                    review.task_sb_impacts(args, card)
            plot.assert_not_called()
            data = json.loads((review.outdir(args)/"sb_impacts_metadata_M20.json").read_text())
            self.assertEqual(data["injected_r"], 15.5625)
            self.assertEqual(data["rmax"], 100.)
            self.assertIn("workspace_sha256", data)

    def test_sb_repair_rejects_background_only_initial_poi(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, card, workspace = self.make_inputs(Path(tmp))
            path = review.outdir(args) / "impacts.json"
            path.write_text(json.dumps({"POIs": [{"name": "r", "fit": [-1., 0., 1.]}]}))
            with self.assertRaisesRegex(RuntimeError, "initial POI"):
                review.repair_sb_payload(args, card, workspace, path, 15.5625)


if __name__ == "__main__":
    unittest.main()
