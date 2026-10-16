#!/usr/bin/env python3
"""Refit only toys flagged by audit_bias_intervals.py and verify profiles.

Uses the original workspace, saved datasets and fitted r ranges. Never generates
toys or overwrites original results. Final summaries require all repairs to pass.
The untouched original fits receive structural interval checks only.
"""
import argparse
import json
import math
import tempfile
from pathlib import Path

import audit_bias_intervals as audit
import validate_bias_repair as verify


def retry_command(workspace, toy_file, mass, count, toy, injection, bounds, strategy, tolerance):
    return ["combine", "-M", "MultiDimFit", str(workspace), "-m", str(mass),
            "-t", str(count), "--pickToy", str(toy), "--toysFile", str(toy_file),
            "--algo", "singles", "--setParameters", f"r={injection:.17g}",
            "--rMin", str(bounds[0]), "--rMax", str(bounds[1]),
            "--setParameterRanges", f"r={bounds[0]},{bounds[1]}",
            "--robustFit", "1", "--cminPoiOnlyFit",
            "--cminDefaultMinimizerStrategy", str(strategy),
            "--cminDefaultMinimizerTolerance", str(tolerance),
            "--cminFallbackAlgo", "Minuit2,Migrad,2:0.1",
            "--setRobustFitStrategy", str(strategy),
            "--setRobustFitTolerance", str(tolerance),
            "--setCrossingTolerance", "1e-4", "--saveFitResult", "--saveNLL",
            "--X-rtd", "MINIMIZER_MaxCalls=1000000", "-v", "3",
            "-n", f".BiasRepair_M{mass}_toy{toy}"]


def validate_point(profile, value, target, tolerance):
    if not math.isfinite(profile["deltaNLL"]) or abs(profile["r"]-value) > 1e-4:
        raise RuntimeError("Invalid conditional profile point")
    if abs(profile["deltaNLL"]-target) > tolerance:
        raise RuntimeError(f"Profile deltaNLL={profile['deltaNLL']} differs from target={target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--mass", type=int, default=70)
    parser.add_argument("--target", default="Run2Run3")
    args = parser.parse_args()
    import ROOT
    ROOT.gROOT.SetBatch(True)
    source = args.source_dir.resolve()
    output = Path(tempfile.mkdtemp(prefix=f"bias_M{args.mass}_verified.", dir=Path.cwd()))
    report = {"passed": False, "errors": [], "n_verified_repairs": 0,
              "repairs": [], "failed_attempts": [], "coverage_test": False,
              "original_per_toy_fit_status_available": False}
    history, merged, summaries = [], {}, []
    manifest = {"audit_file": str(args.audit.resolve()), "input_sha256": {},
                "code_sha256": {p.name: audit.sha256(p) for p in
                                (Path(__file__), Path(audit.__file__), Path(verify.__file__))},
                "screening": {"edm_max": .001, "center_nll_tolerance": .001,
                              "endpoint_nll_tolerance": .01},
                "scope": "Independent profiles for repaired toys; structural checks for other toys"}
    try:
        previous = verify.load(args.audit)
        if previous["errors"] or len(previous["ensembles"]) != 3:
            raise RuntimeError("Audit is incomplete or has input errors")
        manifest["input_sha256"][str(args.audit.resolve())] = audit.sha256(args.audit)
        # Exact audit inputs (including original ROOT files) must still match.
        for name, fingerprint in previous["input_sha256"].items():
            path = Path(name)
            if audit.sha256(path) != fingerprint:
                raise RuntimeError(f"Input changed since audit: {path}")
            manifest["input_sha256"][str(path)] = fingerprint
        original_summary = verify.load(source / "bias" / f"bias_summary_{args.target}_M{args.mass}.json")
        workspace = verify.one(source, f"workspace_{args.target}_M{args.mass}.root")
        manifest["input_sha256"][str(workspace)] = audit.sha256(workspace)
        target_nll = float(.5 * ROOT.Math.chisquared_quantile(.68, 1))
        expected_repairs = sum(len(e["flagged_toy_ids"]) for e in previous["ensembles"])
        for ensemble in previous["ensembles"]:
            tag, injection = ensemble["tag"], ensemble["injected_r"]
            count, bounds = ensemble["n_expected"], tuple(ensemble["r_range"])
            old = [s for s in original_summary if s["injected_r"] == injection]
            if len(old) != 1 or old[0]["n_toys_requested"] != count:
                raise RuntimeError(f"Injection/ensemble mismatch: {tag}")
            fit = source / "bias" / Path(old[0]["fit_file"]).name
            if str(fit) != ensemble["fit_file"] or str(fit) not in manifest["input_sha256"]:
                raise RuntimeError(f"Source path differs from audited inputs: {fit}")
            fresh = audit.audit_rows(audit.read_rows(fit), count, bounds)
            if fresh["unexpected_toy_ids"] or fresh["unrecognised_rows"]:
                raise RuntimeError(f"Ambiguous original entries: {tag}")
            if fresh["flagged_toy_ids"] != ensemble["flagged_toy_ids"]:
                raise RuntimeError(f"Flagged toy IDs changed since audit: {tag}")
            records = {toy: {"interval": row["interval"], "source": str(fit), "repaired": False}
                       for toy, row in fresh["toys"].items()}
            toy_file = verify.one(source / "bias",
                                 f"higgsCombine.ReviewBiasGen_{args.target}_M{args.mass}_{tag}.GenerateOnly.mH*.root")
            manifest["input_sha256"][str(toy_file)] = audit.sha256(toy_file)
            for toy in fresh["flagged_toy_ids"]:
                job = f"{tag}/toy_{toy:04d}"
                success = None
                # Fixed order; selection is by numerical checks, never by bias value.
                for strategy, tolerance in ((1, .01), (2, .01), (2, .001)):
                    directory = output / "repairs" / job / f"s{strategy}_t{tolerance}"
                    command = retry_command(workspace, toy_file, args.mass, count, toy,
                                            injection, bounds, strategy, tolerance)
                    try:
                        rows = verify.run(command, directory, history)
                        result = audit.audit_rows(rows, count, bounds)
                        if result["unexpected_toy_ids"] or result["unrecognised_rows"]:
                            raise RuntimeError("Unexpected entries in single-toy retry")
                        selected = result["toys"][str(toy)]
                        if not selected["passed"] or {r["toy"] for r in rows} != {toy}:
                            raise RuntimeError(f"Invalid retry interval/toy selection: {selected['issues']}")
                        interval = selected["interval"]
                        info = verify.fit_info(directory)
                        verify.check_fit(info, .001)
                        center = verify.profile_saved_reference(command, info, toy, interval[1],
                                                               directory / "center", history, .001)
                        validate_point(center, interval[1], 0, .001)
                        endpoints = []
                        for side, value in (("low", interval[0]), ("high", interval[2])):
                            point = verify.profile_saved_reference(command, info, toy, value,
                                                                  directory / side, history, .01)
                            validate_point(point, value, target_nll, .01)
                            endpoints.append(point | {"side": side, "target_deltaNLL": target_nll})
                        success = {"job": job, "passed": True, "interval": interval,
                                   "status": info["status"], "covQual": info["covQual"],
                                   "EDM": info["EDM"], "center_profile": center,
                                   "endpoints": endpoints, "source_directory": str(directory)}
                        break
                    except Exception as exc:
                        report["failed_attempts"].append({"job": job, "directory": str(directory),
                                                          "error": str(exc)})
                        print(f"[RETRY CHECK] {job}: {exc}", flush=True)
                if success is None:
                    report["errors"].append(f"No independently verified repair: {job}")
                    continue
                report["repairs"].append(success)
                records[str(toy)] = {"interval": success["interval"],
                                     "source": success["source_directory"], "repaired": True}
                print(f"[PROFILE PASS] {job}: {success['interval']}", flush=True)
                verify.dump(output / "validation.json", report)
                verify.dump(output / "run_manifest.json", history)
            if set(map(int, records)) != set(range(1, count+1)):
                raise RuntimeError(f"Incomplete merged ensemble: {tag}")
            for toy, record in records.items():
                if record["interval"] is None or not verify.valid_interval(record["interval"], *bounds):
                    if f"No independently verified repair: {tag}/toy_{int(toy):04d}" not in report["errors"]:
                        report["errors"].append(f"Invalid merged interval: {tag}/toy_{int(toy):04d}")
            merged[tag] = records
            if not report["errors"]:
                row = verify.summarize(records, injection)
                row["n_repaired"] = sum(x["repaired"] for x in records.values())
                summaries.append(row)
        report["n_verified_repairs"] = len(report["repairs"])
        report["passed"] = not report["errors"] and len(report["repairs"]) == expected_repairs and len(summaries) == 3
        if report["passed"]:
            verify.dump(output / "merged_toys.json", merged)
            verify.dump(output / f"bias_summary_{args.target}_M{args.mass}.json", summaries)
            print(f"[COMPLETE] {expected_repairs} repaired profiles checked; three {count}-toy ensembles merged", flush=True)
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        print(f"[ERROR] {exc}", flush=True)
    finally:
        report["n_verified_repairs"] = len(report["repairs"])
        verify.dump(output / "validation.json", report)
        verify.dump(output / "input_manifest.json", manifest)
        verify.dump(output / "run_manifest.json", history)
        print(f"[OUTPUT] {output}", flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
