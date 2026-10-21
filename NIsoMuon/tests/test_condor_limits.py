"""ROOT-free integration tests; fake executables model Combine's file interface."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import limit_workflow as w


# These executables write JSON in place of ROOT, solely for orchestration tests.
FAKE_COMBINE = '''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
if os.environ.get("FAIL_SYNTHETIC_COMBINE"):
    raise SystemExit(7)
a = sys.argv
mass = a[a.index("-m")+1]
name = a[a.index("-n")+1]
blind = "--run" in a and a[a.index("--run")+1] == "blind"
keys = ("exp-2", "exp-1", "exp0", "exp+1", "exp+2")
values = dict(zip(keys, (1., 2., 3., 4., 5.) if blind else (99.,)*5))
if not blind:
    values["obs"] = 3.5
Path("higgsCombine"+name+".AsymptoticLimits.mH"+mass+".root").write_text(json.dumps({mass: values}))
'''

FAKE_COLLECT = '''#!/usr/bin/env python3
import json, sys
from pathlib import Path
a = sys.argv
stop = a.index("-o")
payload = {}
for name in a[3:stop]:
    payload.update(json.loads(Path(name).read_text()))
Path(a[stop+1]).write_text(json.dumps(payload))
'''

FAKE_PLOT = '''#!/usr/bin/env python3
import json, sys
from pathlib import Path
a = sys.argv[1:]
Path(a[a.index("--output")+1]+".plot.json").write_text(json.dumps(a))
'''


class EraSelectionTests(unittest.TestCase):
    def test_subsets_do_not_change_run_combinations(self):
        args = w.build_parser().parse_args(["--target", "eras", "--eras", "2018,2022EE,2018"])
        w.validate_args(args)
        self.assertEqual(w.requested_targets(args), ["2018", "2022EE"])
        self.assertEqual(w.years_needed_for_request(args), ("2018", "2022EE"))
        for target, eras in (("run2", w.RUN2_ERAS), ("run3", w.RUN3_ERAS),
                             ("full", w.YEARS), ("runs", w.YEARS), ("eras", w.YEARS)):
            args = w.build_parser().parse_args(["--target", target])
            self.assertEqual(w.years_needed_for_request(args), eras)
        for era in w.YEARS:
            args = w.build_parser().parse_args(["--target", era.lower()])
            self.assertEqual(w.requested_targets(args), [era])
            self.assertIn("per_year/"+era, str(w.card_pattern(args, era)))
            self.assertIn("fb^{-1}", w.lumi_label(era))
        args = w.build_parser().parse_args(["--target", "runs", "--eras", "2018"])
        with self.assertRaisesRegex(w.WorkflowError, "full era set"):
            w.validate_args(args)

    def test_signal_discovery_and_card_building_only_use_selected_eras(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = w.build_parser().parse_args([
                "--target", "eras", "--eras", "2018", "--base-dir", tmp,
                "--card-base", str(Path(tmp)/"cards"), "--qcd-method", "mc", "--dy-method", "mc",
            ])
            for era, mass in (("2018", 12), ("2022", 70)):
                path = Path(w.root_dir(args, era))/f"NIsoMuon_Zp_M-{mass}.root"
                path.parent.mkdir(parents=True)
                path.touch()
            self.assertEqual(set(w.scan_signal_files(args)), {12.})

            def channels(reader, args, mass, files, years):
                self.assertEqual(years, ("2018",))
                rates = {p: 1. for p in w.PROCESSES}
                return {"2018": w.ChannelResult("2018", "12", 12., 11., 13., 5.,
                                                 rates.copy(), rates.copy(), {})}, w.AuditLog()

            with patch.object(w, "RootReader"), patch.object(w, "build_channels_for_mass", channels):
                cards = w.build_cards(args)
            self.assertEqual([c.target for c in cards], ["2018"])
            text = cards[0].path.read_text()
            self.assertIn("bin_2018", text)
            self.assertNotIn("bin_2022", text)


class CondorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="condor limits ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root/"bin"
        self.bin.mkdir()
        for name, content in (("combine", FAKE_COMBINE), ("combineTool.py", FAKE_COLLECT),
                              ("plotLimits.py", FAKE_PLOT)):
            path = self.bin/name
            path.write_text(content)
            path.chmod(0o755)
        (self.root/"CMSSW/src").mkdir(parents=True)
        self.env = dict(os.environ, PATH=str(self.bin)+os.pathsep+os.environ["PATH"])

    def prepare(self, mode="blind", masses="12,30"):
        args = w.build_parser().parse_args([
            "--stage", "run", "--backend", "condor", "--condor-dry-run",
            "--target", "eras", "--eras", "2018,2022", "--mode", mode,
            "--masses", masses, "--card-base", str(self.root/"cards"),
            "--output-base", str(self.root/"outputs"), "--condor-dir", str(self.root/"batch"),
            "--condor-cmssw-base", str(self.root/"CMSSW"),
            "--condor-scram-arch", "el9_amd64_gcc12", "--condor-nmax", "40",
            "--qcd-method", "mc", "--dy-method", "mc", "--strict",
            "--plot-limits-py", str(self.bin/"plotLimits.py"),
        ])
        for target, unit in (("2018", .01), ("2022", .02), ("Run2", .03)):
            for mass in (12, 30, 70):
                card = w.datacard_path(args, target, str(mass))
                card.parent.mkdir(parents=True, exist_ok=True)
                card.write_text(f"# alpha_internal_unit = {unit}\n# alpha_internal_signal_yield = 25\n")
        w.validate_args(args)
        with patch.object(w, "run_command") as command, patch.dict(os.environ, {"USER": "tester"}), \
                contextlib.redirect_stdout(io.StringIO()):
            batch = w.submit_condor_limits(args)
        command.assert_not_called()
        self.batch = batch
        self.manifest = json.loads((batch/"manifest.json").read_text())
        return args

    def cli(self, *options, env=None):
        return subprocess.run([sys.executable, str(self.batch/"limit_workflow.py"), *options],
                              cwd=self.root, env=env or self.env, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    def worker(self, index, env=None):
        return self.cli("--condor-worker", str(self.batch/"manifest.json"),
                        "--condor-index", str(index), env=env)

    def test_blind_batch_preserves_selection_scaling_and_plot_labels(self):
        # Each mode uses an independent fixture, just like independent submissions.
        self.run_batch("blind")

    def test_unblind_uses_separate_blind_expected_quantiles(self):
        self.run_batch("unblind")

    def run_batch(self, mode):
        args = self.prepare(mode)
        jobs = self.manifest["jobs"]
        self.assertEqual(len(jobs), 4)
        self.assertEqual({j["target"] for j in jobs}, {"2018", "2022"})
        subprocess.run(["bash", "-n", str(self.batch/"run_job.sh")], check=True)
        jds = (self.batch/"submit.jds").read_text()
        submit_values = dict(line.split(" = ", 1) for line in jds.splitlines() if " = " in line)
        self.assertTrue(Path(submit_values["executable"]).is_file())
        self.assertTrue(Path(submit_values["initialdir"]).is_dir())
        for line in ("queue 4", "request_cpus = 1", "should_transfer_files = NO",
                     "arguments = $(Process)", "concurrency_limits = n40."):
            self.assertIn(line, jds)
        for job in jobs:
            # A later card build must not change the submitted fit or conversion.
            Path(job["original_card"]).write_text("# alpha_internal_unit = 999\n")
        pending = self.cli("--stage", "collect", "--condor-dir", str(self.batch))
        self.assertNotEqual(pending.returncode, 0)
        self.assertIn("incomplete", pending.stdout)
        self.assertFalse((self.root/"outputs/alpha").exists())
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(self.worker, range(4)))
        for result in results:
            self.assertEqual(result.returncode, 0, result.stdout)
        result = self.cli("--stage", "collect", "--condor-dir", str(self.batch))
        self.assertEqual(result.returncode, 0, result.stdout)
        for target, unit in (("2018", .01), ("2022", .02)):
            out = w.output_dir(args, target)
            data = json.loads((out/f"limits_{target}_alpha.json").read_text())
            self.assertEqual(set(data), {"12", "30"})
            for row in data.values():
                self.assertAlmostEqual(row["exp0"], 3.*unit)
                if mode == "blind":
                    self.assertNotIn("obs", row)
                else:
                    self.assertAlmostEqual(row["obs"], 3.5*unit)
            plot = json.loads((out/f"limits_{target}_alpha.plot.json").read_text())
            self.assertEqual(plot[plot.index("--title-right")+1], w.lumi_label(target))
            self.assertEqual(plot[plot.index("--show")+1], "exp" if mode == "blind" else "exp,obs")
        self.assertFalse(w.output_dir(args, "Run2").exists())

    def test_failed_retry_cannot_collect_previous_success(self):
        self.prepare(masses="12")
        for index in range(2):
            result = self.worker(index)
            self.assertEqual(result.returncode, 0, result.stdout)
        failed = self.worker(0, dict(self.env, FAIL_SYNTHETIC_COMBINE="1"))
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("exit status 7", failed.stdout)
        result = self.cli("--stage", "collect", "--condor-dir", str(self.batch))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("failed", result.stdout)
        self.assertFalse((self.root/"outputs/alpha").exists())

    def test_changed_snapshot_and_collection_option_override_are_rejected(self):
        self.prepare("unblind", masses="12")
        Path(self.manifest["jobs"][0]["card"]).write_text("modified")
        result = self.worker(0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Submitted datacard changed", result.stdout)
        result = self.cli("--stage", "collect", "--condor-dir", str(self.batch), "--mode", "blind")
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stdout)


if __name__ == "__main__":
    unittest.main()
