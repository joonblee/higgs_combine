"""Condor impact DAG tests using synthetic Combine files and mocked ROOT reads."""
import contextlib
import io
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from test_condor_limits import CondorFixture, w


IMPACT_TOOL = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
a = sys.argv
if a[a.index("-M")+1] == "CollectLimits":
    stop = a.index("-o")
    payload = {}
    for name in a[3:stop]:
        payload.update(json.loads(Path(name).read_text()))
    Path(a[stop+1]).write_text(json.dumps(payload))
    raise SystemExit(0)
mass = a[a.index("-m")+1]
name = a[a.index("-n")+1]
initial = Path("higgsCombine_initialFit_"+name+".MultiDimFit.mH"+mass+".root")
if "--doInitialFit" in a:
    rmin = float(a[a.index("--rMin")+1])
    low = rmin if os.environ.get("FAKE_IMPACT_BOUNDARY") and rmin > -5 else min(0., rmin+1.)
    initial.write_text(json.dumps({"POIs": [{"name": "r", "fit": [low, 0., 1.]}]}))
elif "--doFits" in a:
    assert initial.is_file(), "initial fit was not copied to the nuisance worker"
    for nuisance in a[a.index("--named")+1].split(","):
        p = {"name": nuisance, "fit": [-1., 0., 1.], "prefit": [-1., 0., 1.],
             "type": "Gaussian", "r": [-.2, 0., .2], "impact_r": .2}
        scan = Path("higgsCombine_paramFit_"+name+"_"+nuisance+".MultiDimFit.mH"+mass+".root")
        scan.write_text(json.dumps(p))
    Path("fit_arguments.json").write_text(json.dumps(a))
else:
    result = json.loads(initial.read_text())
    result["params"] = []
    for nuisance in sorted(json.loads(Path(a[a.index("-d")+1]).read_text())):
        scan = Path("higgsCombine_paramFit_"+name+"_"+nuisance+".MultiDimFit.mH"+mass+".root")
        if scan.is_file():
            result["params"].append(json.loads(scan.read_text()))
    Path(a[a.index("-o")+1]).write_text(json.dumps(result))
'''

PLOT_IMPACTS = '''#!/usr/bin/env python3
"""Synthetic plotting anchors:
    pads[0].SetGrid(1, 0)
    h_impacts.Draw()
"""
import json, sys
from pathlib import Path
a = sys.argv
Path(a[a.index("-o")+1]+".pdf").write_bytes(b"synthetic PDF")
'''


class CondorImpactTests(CondorFixture):
    def setUp(self):
        super().setUp()
        for name, code in (("combineTool.py", IMPACT_TOOL), ("plotImpacts.py", PLOT_IMPACTS),
                           ("text2workspace.py", "#!/bin/bash\nexit 0\n")):
            path = self.bin/name
            path.write_text(code)
            path.chmod(0o755)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, self.env))
        self.stack.enter_context(patch.object(w, "impact_workspace_parameters",
                                              side_effect=lambda p: json.loads(Path(p).read_text())))
        self.stack.enter_context(patch.object(w, "read_impact_initial_poi",
                                              side_effect=lambda p: json.loads(Path(p).read_text())["POIs"][0]["fit"]))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def run_node(self, job):
        w.run_condor_worker(self.batch/"manifest.json", job["index"])

    def complete_impacts(self):
        for job in self.manifest["jobs"]:
            if job["kind"] == "impact_initial":
                self.run_node(job)
        nuisance_jobs = [j for j in self.manifest["jobs"] if j["kind"] == "impact_nuisance"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(self.run_node, nuisance_jobs))
        for job in self.manifest["jobs"]:
            if job["kind"] == "impact_final":
                self.run_node(job)

    def check_impacts(self, mode):
        args = self.prepare(mode=mode, masses="12", task="impacts")
        jobs = self.manifest["jobs"]
        self.assertEqual(Counter(j["kind"] for j in jobs),
                         {"impact_initial": 2, "impact_nuisance": 4, "impact_final": 2})
        jds = (self.batch/"submit.jds").read_text()
        self.assertIn("arguments = $(jobindex)", jds)
        self.assertNotIn("concurrency_limits", jds)
        dag = (self.batch/"workflow.dag").read_text()
        for job in jobs:
            if job["dependencies"]:
                parents = " ".join(f"J{i}" for i in job["dependencies"])
                self.assertIn(f"PARENT {parents} CHILD J{job['index']}", dag)
        self.complete_impacts()
        w.collect_condor_workflow(self.batch)
        for target, unit in (("2018", .01), ("2022", .02)):
            out = w.output_dir(args, target)
            result = json.loads((out/f"impacts_{target}_alpha_M12.json").read_text())
            self.assertEqual(len(result["params"]), 2)
            self.assertEqual(result["POIs"][0]["fit"], [0., 0., unit])
            self.assertAlmostEqual(result["params"][0]["impact_r"], .2*unit)
            self.assertTrue((out/f"impacts_{target}_alpha_M12.pdf").is_file())
            report = json.loads((out/"impact_repair_M12/validation.json").read_text())
            self.assertTrue(report["passed"])
        for job in jobs:
            if job["kind"] != "impact_nuisance":
                continue
            command = json.loads((Path(job["workdir"])/"fit_arguments.json").read_text())
            self.assertEqual(command[command.index("--named")+1], job["nuisance"])
            self.assertEqual("--expectSignal" in command, mode == "blind")
            self.assertNotIn("--parallel", command)

    def test_blind_impacts_split_nuisances_and_validate_before_publication(self):
        self.check_impacts("blind")

    def test_observed_impacts_preserve_data_mode_in_every_nuisance_fit(self):
        self.check_impacts("unblind")

    def test_dependencies_block_early_children_and_stale_final_results(self):
        self.prepare(masses="12", task="impacts")
        initial, nuisance, _, final = self.manifest["jobs"][:4]
        with self.assertRaisesRegex(w.WorkflowError, "pending"):
            self.run_node(nuisance)
        self.complete_impacts()
        # Even a successful nuisance rerun invalidates the old final collection.
        self.run_node(nuisance)
        with self.assertRaisesRegex(w.WorkflowError, "stale result after dependency"):
            w.collect_condor_workflow(self.batch)
        self.run_node(final)
        # Rebuilding the initial fit invalidates every downstream scan.
        self.run_node(initial)
        with self.assertRaisesRegex(w.WorkflowError, "different initial fit"):
            w.collect_condor_workflow(self.batch)

    def test_initial_range_is_expanded_before_nuisance_jobs_start(self):
        self.prepare(masses="12", task="impacts", extra=("--allow-negative-r", "--r-min", "-1"))
        with patch.object(w, "initial_impact_range", return_value=(-1., 100., -10.)), \
                patch.dict(os.environ, {"FAKE_IMPACT_BOUNDARY": "1"}):
            self.complete_impacts()
        for job in self.manifest["jobs"]:
            if job["kind"] == "impact_nuisance":
                command = json.loads((Path(job["workdir"])/"fit_arguments.json").read_text())
                self.assertEqual(float(command[command.index("--rMin")+1]), -7.5)

    def test_task_all_includes_limits_diagnostics_and_split_impacts(self):
        args = self.prepare(masses="12", task="all")
        counts = Counter(j["kind"] for j in self.manifest["jobs"])
        self.assertEqual(counts, {"limits": 2, "fitdiag": 2, "impact_initial": 2,
                                  "impact_nuisance": 4, "impact_final": 2})
        for job in self.manifest["jobs"]:
            if job["kind"] in {"limits", "fitdiag"}:
                self.run_node(job)
        self.complete_impacts()
        w.collect_condor_workflow(self.batch)
        for target in ("2018", "2022"):
            out = w.output_dir(args, target)
            self.assertTrue((out/f"limits_{target}_alpha.json").is_file())
            self.assertTrue((out/f"fitDiagnostics.FitDiag_{target}_alpha_M12.root").is_file())
            self.assertTrue((out/f"impacts_{target}_alpha_M12.pdf").is_file())

    def test_repair_snapshots_and_runs_existing_results_on_condor(self):
        self.prepare(masses="12", task="impacts")
        self.complete_impacts()
        w.collect_condor_workflow(self.batch)
        args = self.prepare(masses="12", task="impacts", extra=(
            "--repair-impacts", "--condor-dir", str(self.root/"repair_batch")))
        self.assertEqual([j["kind"] for j in self.manifest["jobs"]], ["impact_repair"]*2)
        for job in self.manifest["jobs"]:
            self.run_node(job)
        w.collect_condor_workflow(self.batch)
        self.assertTrue((w.output_dir(args, "2018")/"impacts_2018_alpha_M12.pdf").is_file())

    def test_resume_uses_current_code_and_only_failed_final_nodes(self):
        self.prepare(masses="12", task="impacts")
        self.complete_impacts()
        finals = [j for j in self.manifest["jobs"] if j["kind"] == "impact_final"]
        successful = [j for j in self.manifest["jobs"] if j["kind"] != "impact_final"]
        preserved = {j["index"]: (Path(j["workdir"])/"status.json").read_bytes() for j in successful}
        original_snapshot = (self.batch/"limit_workflow.py").read_bytes()
        for job in finals:
            path = Path(job["workdir"])/"status.json"
            status = json.loads(path.read_text())
            status.update(state="failed", error="Boundary fit needs two entries")
            path.write_text(json.dumps(status))
        directory = w.resume_condor_workflow(self.batch, dry_run=True)
        request = json.loads((directory/"resume.json").read_text())
        self.assertEqual(set(map(int, request["jobs"])), {j["index"] for j in finals})
        self.assertEqual((self.batch/"limit_workflow.py").read_bytes(), original_snapshot)
        self.assertEqual((directory/"limit_workflow.py").read_bytes(), Path(w.__file__).read_bytes())
        for job in successful:
            self.assertEqual((Path(job["workdir"])/"status.json").read_bytes(), preserved[job["index"]])
        self.assertNotIn("concurrency_limits", (directory/"submit.jds").read_text())
        self.assertNotIn("PARENT", (directory/"resume.dag").read_text())
        for job in finals:
            self.run_node(job)
        w.collect_condor_workflow(self.batch)
        self.assertIsNone(w.resume_condor_workflow(self.batch, dry_run=True))

    def test_resume_includes_downstream_nodes_of_failed_initial_fit(self):
        self.prepare(masses="12", task="impacts")
        self.complete_impacts()
        initial = self.manifest["jobs"][0]
        path = Path(initial["workdir"])/"status.json"
        status = json.loads(path.read_text())
        status.update(state="failed", error="synthetic initial fit failure")
        path.write_text(json.dumps(status))
        directory = w.resume_condor_workflow(self.batch, dry_run=True)
        request = json.loads((directory/"resume.json").read_text())
        self.assertEqual(set(map(int, request["jobs"])), {0, 1, 2, 3})
        dag = (directory/"resume.dag").read_text()
        self.assertIn("PARENT J0 CHILD J1", dag)
        self.assertIn("PARENT J0 J1 J2 CHILD J3", dag)

