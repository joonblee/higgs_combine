"""Exercise interpolated scan cards through Condor snapshot/worker/collect."""
import contextlib
import ctypes
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from test_condor_limits import CondorFixture, w
from test_signal_interpolation import SyntheticReader, arguments, signal_files


class CondorSignalInterpolationTests(CondorFixture):
    def prepare_scan(self, masses="12,12.5,15", mode="unblind", custom_grid=True):
        options = ["--target", "2016preVFP", "--mode", mode, "--interpolate-signal",
                   "--card-base", str(self.root/"cards"), "--output-base", str(self.root/"outputs"),
                   "--condor-dir", str(self.root/"batch"), "--condor-dry-run",
                   "--condor-cmssw-base", str(self.root/"CMSSW"),
                   "--condor-scram-arch", "el9_amd64_gcc12",
                   "--plot-limits-py", str(self.bin/"plotLimits.py")]
        if masses:
            options += ["--masses", masses]
        if custom_grid:
            grid = self.root/"original_grid.json"
            grid.write_text(json.dumps({"schema_version": 1, "masses": [12., 12.5, 15.]}))
            options += ["--mass-grid-file", str(grid)]
        args = arguments(*options)
        w.validate_args(args)
        w.prepare_mass_scan(args)
        with patch.object(w, "RootReader", return_value=SyntheticReader()), \
                patch.object(w, "scan_signal_files", return_value=signal_files(args)), \
                contextlib.redirect_stdout(io.StringIO()):
            w.build_cards(args)
            args.stage = "run"
            self.batch = w.submit_condor_workflow(args)
        self.manifest = json.loads((self.batch/"manifest.json").read_text())
        return args

    def test_fractional_unblind_scan_uses_snapshot_and_collects_alpha(self):
        args = self.prepare_scan()
        self.assertEqual(len(self.manifest["jobs"]), 3)
        config = self.manifest["config"]
        self.assertNotIn("signal_interpolator", config)
        self.assertEqual(Path(config["mass_grid_file"]), self.batch/"mass_scan_grid.json")
        self.assertEqual(config["mass_grid_sha256"], args.mass_grid_sha256)
        # Neither source grid nor original cards/fit inputs are needed by workers.
        Path(args.mass_grid_source).unlink()
        for job in self.manifest["jobs"]:
            Path(job["original_card"]).unlink()
        for index in range(3):
            result = self.worker(index)
            self.assertEqual(result.returncode, 0, result.stdout)
        result = self.cli("--stage", "collect", "--condor-dir", str(self.batch))
        self.assertEqual(result.returncode, 0, result.stdout)
        output = w.output_dir(args, "2016preVFP")
        limits = json.loads((output/"limits_2016preVFP_alpha.json").read_text())
        self.assertEqual(set(limits), {"12", "12.5", "15"})
        for job in self.manifest["jobs"]:
            _, mass = w.extract_mass_from_card(Path(job["card"]), "2016preVFP")
            unit = w.alpha_unit_from_card(Path(job["card"]))
            self.assertAlmostEqual(limits[f"{mass:g}"]["exp0"], 3.*unit)
            self.assertAlmostEqual(limits[f"{mass:g}"]["obs"], 3.5*unit)

    def test_changed_submitted_grid_is_rejected(self):
        self.prepare_scan()
        grid = self.batch/"mass_scan_grid.json"
        grid.write_text(grid.read_text()+"\n")
        result = self.worker(0)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Submitted interpolation mass grid changed", result.stdout)

    def test_float32_collected_masses_preserve_alpha_conversion(self):
        args = self.prepare_scan(masses="32.553,66.784,70", custom_grid=False)
        collect = self.bin/"combineTool.py"
        code = collect.read_text().replace("import json, sys", "import ctypes, json, sys")
        code = code.replace('Path(a[stop+1]).write_text(json.dumps(payload))',
                            'payload = {str(ctypes.c_float(float(k)).value): v for k, v in payload.items()}\n'
                            'Path(a[stop+1]).write_text(json.dumps(payload))')
        collect.write_text(code)
        for index in range(3):
            result = self.worker(index)
            self.assertEqual(result.returncode, 0, result.stdout)
        result = self.cli("--stage", "collect", "--condor-dir", str(self.batch))
        self.assertEqual(result.returncode, 0, result.stdout)
        limits = json.loads((w.output_dir(args, "2016preVFP")/"limits_2016preVFP_alpha.json").read_text())
        self.assertIn("32.553001403808594", limits)
        for job in self.manifest["jobs"]:
            _, mass = w.extract_mass_from_card(Path(job["card"]), "2016preVFP")
            row = limits[str(ctypes.c_float(mass).value)]
            unit = w.alpha_unit_from_card(Path(job["card"]))
            self.assertAlmostEqual(row["exp0"], 3.*unit)
            self.assertAlmostEqual(row["obs"], 3.5*unit)

    def test_resume_snapshot_uses_same_grid_and_fractional_hypotheses(self):
        self.prepare_scan()
        result = self.worker(0)
        self.assertEqual(result.returncode, 0, result.stdout)
        with patch.dict(os.environ, self.env), contextlib.redirect_stdout(io.StringIO()):
            resume_dir = w.resume_condor_workflow(self.batch, dry_run=True)
        resumed_script = resume_dir/"limit_workflow.py"
        self.assertTrue(resumed_script.is_file())
        for index in (1, 2):
            result = subprocess.run([sys.executable, str(resumed_script), "--condor-worker",
                                     str(self.batch/"manifest.json"), "--condor-index", str(index)],
                                    cwd=self.root, env=self.env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        result = self.cli("--stage", "collect", "--condor-dir", str(self.batch))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_full_shared_grid_creates_229_single_era_jobs(self):
        args = self.prepare_scan(masses=None, custom_grid=False, mode="blind")
        self.assertEqual(len(self.manifest["jobs"]), 229)
        self.assertEqual({j["target"] for j in self.manifest["jobs"]}, {"2016preVFP"})
        self.assertEqual(self.manifest["config"]["masses"], args.masses)
        self.assertIn("queue 229", (self.batch/"submit.jds").read_text())
