#!/usr/bin/env python3
"""Validate saved-toy bias repairs without changing cards or generating toys.

Run under the same CMSSW/Combine runtime as the original review. Original fit
files are read-only. New minimisations and summaries live in a new directory.
The final summaries require complete toy IDs and independent endpoint checks.
Numerical consistency is not a coverage test; original ensemble fits generally
lack per-toy saved RooFitResults, and their status cannot be reconstructed here.
"""
import argparse
import copy
import hashlib
import json
import math
import statistics
import subprocess
import tempfile
from pathlib import Path


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load(path):
    return json.loads(path.read_text())


def option(command, name):
    return command[command.index(name) + 1]


def change(command, name, value):
    if name in command:
        command[command.index(name) + 1] = str(value)
    else:
        command.extend([name, str(value)])


def remove(command, name, takes_value=True):
    while name in command:
        i = command.index(name)
        del command[i:i + (2 if takes_value else 1)]


def intervals(rows):
    """Keep duplicate entries visible instead of silently overwriting them."""
    grouped = {}
    for row in rows:
        for name, q in (("best", -1.0), ("low", -0.32), ("high", 0.32)):
            if abs(row["quantile"] - q) < 1e-5:
                grouped.setdefault(row["toy"], {}).setdefault(name, []).append(row["r"])
    result = {}
    for toy, values in grouped.items():
        if all(len(values.get(k, [])) == 1 for k in ("low", "best", "high")):
            result[toy] = [values[k][0] for k in ("low", "best", "high")]
    return result


def valid_interval(value, low=-50.0, high=100.0):
    return (len(value) == 3 and all(math.isfinite(v) for v in value)
            and low + 1e-5 < value[0] < value[1] < value[2] < high - 1e-5)


def combine_records(original, replacements, count):
    records = copy.deepcopy(original)
    records.update(replacements)
    expected = set(range(1, count + 1))
    if set(records) != expected:
        raise RuntimeError(f"Toy IDs mismatch: missing={sorted(expected-set(records))}, "
                           f"extra={sorted(set(records)-expected)}")
    for toy, record in records.items():
        if not valid_interval(record["interval"]):
            raise RuntimeError(f"Invalid final interval for toy {toy}: {record['interval']}")
    return records


def summarize(records, injection):
    values = [r["interval"][1] for r in records.values()]
    # Directional profile errors: use the side pointing towards the injected r.
    pulls = []
    for record in records.values():
        low, best, high = record["interval"]
        sigma = best - low if best >= injection else high - best
        pulls.append((best - injection) / sigma)
    spread = statistics.stdev(values)
    return {"injected_r": injection, "n_toys": len(values),
            "mean_fitted_r": statistics.fmean(values),
            "mean_bias_r": statistics.fmean(values) - injection,
            "std_fitted_r": spread,
            "mean_bias_mc_standard_error": spread / math.sqrt(len(values)),
            "n_pulls": len(pulls), "mean_pull": statistics.fmean(pulls),
            "pull_width": statistics.stdev(pulls),
            "pull_definition": "(rhat-r_inj)/directional_68_percent_profile_error",
            "coverage_test": False,
            "original_per_toy_fit_status_available": False}


def root_open(path):
    import ROOT
    f = ROOT.TFile.Open(str(path))
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {path}")
    return f


def read_rows(path):
    f = root_open(path)
    try:
        tree = f.Get("limit")
        if not tree:
            raise RuntimeError(f"No limit tree in {path}")
        rows = []
        for e in tree:
            row = {"toy": int(e.iToy), "quantile": float(e.quantileExpected), "r": float(e.r)}
            if hasattr(e, "deltaNLL"):
                row["deltaNLL"] = float(e.deltaNLL)
            rows.append(row)
        return rows
    finally:
        f.Close()


def one(directory, pattern):
    paths = list(directory.glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {pattern} in {directory}, found {len(paths)}")
    return paths[0]


def fit_info(directory):
    f = root_open(one(directory, "multidimfit*.root"))
    try:
        fit = f.Get("fit_mdf")
        if not fit:
            raise RuntimeError(f"No fit_mdf in {directory}")
        parameters = {p.GetName(): float(p.getVal()) for p in fit.floatParsFinal()}
        return {"status": int(fit.status()), "covQual": int(fit.covQual()),
                "EDM": float(fit.edm()), "parameters": parameters}
    finally:
        f.Close()


def run(command, directory, history):
    directory.mkdir(parents=True, exist_ok=False)
    print(f"[FIT] {directory}", flush=True)
    with (directory / "fit.log").open("w") as log:
        p = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
    history.append({"directory": str(directory), "command": command, "exit_code": p.returncode})
    if p.returncode:
        raise RuntimeError(f"Command exit {p.returncode}: {directory}")
    return read_rows(one(directory, "higgsCombine*.MultiDimFit.mH20*.root"))


def seed_command(command, info):
    command = list(command)
    # A full saved best-fit seed supersedes the preliminary POI-only search.
    remove(command, "--cminPoiOnlyFit", False)
    change(command, "--setParameters", ",".join(
        f"{name}={value:.17g}" for name, value in sorted(info["parameters"].items())))
    return command


def check_fit(info, edm_limit):
    if info["status"] != 0 or info["covQual"] < 2:
        raise RuntimeError(f"Unsatisfactory saved fit: status={info['status']}, covQual={info['covQual']}")
    if not math.isfinite(info["EDM"]) or not 0 <= info["EDM"] <= edm_limit:
        raise RuntimeError(f"EDM requires further inspection: {info['EDM']}")


def reference_consistent(shift, interval, center_delta, absolute_tol, relative_tol, nll_tol):
    """A relative shift alone is insufficient: reprofile the saved centre too."""
    scale = 0.5 * (interval[2] - interval[0])
    return (math.isfinite(center_delta) and abs(center_delta) <= nll_tol
            and math.isfinite(shift)
            and shift <= max(absolute_tol, relative_tol * scale))


def fixed_command(command, info, value, label, strategy, tolerance):
    fixed = seed_command(command, info)
    change(fixed, "--algo", "fixed")
    remove(fixed, "--saveFitResult", False)
    remove(fixed, "--cminFallbackAlgo")
    remove(fixed, "--skipInitialFit", False)
    fixed.append("--skipInitialFit")
    change(fixed, "--fixedPointPOIs", f"r={value:.17g}")
    change(fixed, "--cminDefaultMinimizerStrategy", strategy)
    change(fixed, "--cminDefaultMinimizerTolerance", tolerance)
    change(fixed, "-n", f".Verify_{label}")
    if "--saveNLL" not in fixed:
        fixed.append("--saveNLL")
    return fixed


def profile_saved_reference(command, info, toy, value, directory, history, agreement_tol):
    # --algo fixed starts its deltaNLL reference at the supplied parameter
    # state when --skipInitialFit is used. All nuisance parameters still float
    # in the conditional fit. This avoids needlessly refitting a verified
    # global minimum (which previously failed for an already-converged toy).
    candidates, failures = [], []
    for strategy, tolerance in ((1, 0.01), (2, 0.01), (2, 0.1)):
        attempt = directory / f"s{strategy}_t{tolerance}"
        fixed = fixed_command(command, info, value, directory.name, strategy, tolerance)
        try:
            rows = run(fixed, attempt, history)
            # With --skipInitialFit there is no initial best-fit row. A tiny
            # negative centre delta can give an undefined chi-square p-value;
            # identify the conditional entry by toy and deltaNLL, not that p-value.
            points = [r for r in rows if r["toy"] == toy and "deltaNLL" in r]
            if len(points) != 1:
                raise RuntimeError("Missing conditional profile entry")
            point = points[0]
            delta = point.get("deltaNLL", float("nan"))
            if not math.isfinite(delta) or abs(point["r"]-value) > 1e-4:
                raise RuntimeError("Non-finite or mismatched conditional point")
            candidates.append({"strategy": strategy, "tolerance": tolerance,
                               "deltaNLL": delta, "directory": str(attempt)})
        except Exception as exc:
            failures.append(str(exc))
    dump(directory / "attempts.json", {"candidates": candidates, "failures": failures})
    if not candidates:
        raise RuntimeError(f"All conditional fits failed: {directory}")
    spread = max(c["deltaNLL"] for c in candidates) - min(c["deltaNLL"] for c in candidates)
    if spread > agreement_tol:
        raise RuntimeError(f"Inconsistent conditional minima: deltaNLL spread={spread}")
    # Select the smallest conditional NLL, never the one closest to the target.
    best = min(candidates, key=lambda c: c["deltaNLL"])
    return {"r": value, "deltaNLL": best["deltaNLL"],
            "reference": "saved_best_fit_parameters", "candidate_spread": spread,
            "attempts": candidates, "failed_attempts": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("review_outputs_updated.P2ejoE9l/Run2Run3/M20"))
    parser.add_argument("--fallback-dir", type=Path, required=True)
    parser.add_argument("--poiseed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path,
                        help="Reuse successful profile checks and refinement from this run")
    parser.add_argument("--edm-review-threshold", type=float, default=1e-3,
                        help="Conservative numerical screening value, not a universal Minuit criterion")
    parser.add_argument("--endpoint-nll-tolerance", type=float, default=0.01)
    parser.add_argument("--minimum-shift-tolerance", type=float, default=0.01)
    parser.add_argument("--minimum-relative-shift-tolerance", type=float, default=0.01,
                        help="Fraction of interval half-width; also requires centre NLL check")
    parser.add_argument("--minimum-nll-tolerance", type=float, default=0.001)
    args = parser.parse_args()
    import ROOT
    ROOT.gROOT.SetBatch(True)
    if min(args.edm_review_threshold, args.endpoint_nll_tolerance,
           args.minimum_shift_tolerance, args.minimum_relative_shift_tolerance,
           args.minimum_nll_tolerance) <= 0:
        parser.error("Numerical screening tolerances must be positive")
    output = (args.output_dir.resolve() if args.output_dir else
              Path(tempfile.mkdtemp(prefix="bias_M20_verified.", dir=Path.cwd())))
    if args.output_dir:
        output.mkdir(parents=True, exist_ok=False)
    history, checks, refinements = [], [], []
    summary_source = args.source_dir.resolve() / "bias/bias_summary_Run2Run3_M20.json"
    original_summaries = load(summary_source)
    commands, selected = {}, {}
    injections = {}
    fingerprints = {}
    errors = []
    try:
        # Retain only the successful fallback results, then the successful
        # POI-seeded results; never choose a result by its fitted signal value.
        for base in (args.fallback_dir.resolve(), args.poiseed_dir.resolve()):
            validation = load(base / "validation.json")
            passed = {row["job"] for row in validation["results"] if row["passed"]}
            for item in load(base / "run_manifest.json"):
                job = item["directory"]
                if job in passed:
                    if job in selected:
                        raise RuntimeError(f"Ambiguous successful replacement: {job}")
                    selected[job] = base / job
                    commands[job] = item["command"]
                    cmd = item["command"]
                    toy = int(option(cmd, "--pickToy"))
                    tag = Path(job).parts[0]
                    initial = option(cmd, "--setParameters")
                    injection = float(initial.removeprefix("r="))
                    if tag in injections and injections[tag] != injection:
                        raise RuntimeError(f"Inconsistent injection for {tag}")
                    injections[tag] = injection
                    if toy != int(Path(job).name.split("_")[-1]):
                        raise RuntimeError(f"Toy selection mismatch: {job}")
                    if int(option(cmd, "-t")) != 500:
                        raise RuntimeError("This verification expects the original 500-toy ensembles")
                    # These are input files only; no GenerateOnly command occurs.
                    for path in (Path(option(cmd, "--toysFile")), Path(cmd[cmd.index("MultiDimFit")+1])):
                        if not path.is_file():
                            raise RuntimeError(f"Missing original input: {path}")
                        if str(path) not in fingerprints:
                            fingerprints[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(selected) != 32:
            raise RuntimeError(f"Expected 32 distinct repaired toys, found {len(selected)}")
        resume = args.resume_from.resolve() if args.resume_from else None
        previous = load(resume / "validation.json") if resume else {}
        if resume:
            old_inputs = load(resume / "input_manifest.json")
            if old_inputs["input_sha256"] != fingerprints:
                raise RuntimeError("Resume input hashes do not match current workspace/toys")
            if old_inputs["selected_repairs"] != {k: str(v) for k, v in selected.items()}:
                raise RuntimeError("Resume selected-repair directories differ")
            old_options = old_inputs["numerical_screening_options"]
            for name in ("edm_review_threshold", "endpoint_nll_tolerance", "minimum_shift_tolerance"):
                if old_options[name] != getattr(args, name):
                    raise RuntimeError(f"Cannot reuse checks with a changed {name}")
            checks = copy.deepcopy(previous["repairs"])
            jobs_checked = {c["job"] for c in checks}
            if len(jobs_checked) != len(checks) or not jobs_checked <= set(selected):
                raise RuntimeError("Duplicate or unknown cached profile checks")
            for cached in checks:
                if not cached["passed"] or len(cached["endpoints"]) != 2:
                    raise RuntimeError("Invalid cached profile check")
                for endpoint in cached["endpoints"]:
                    if abs(endpoint["deltaNLL"]-endpoint["target_deltaNLL"]) > args.endpoint_nll_tolerance:
                        raise RuntimeError("Cached endpoint fails current NLL tolerance")
            refinements = copy.deepcopy(previous.get("refinements", []))
            print(f"[REUSE] {len(checks)} completed profile checks", flush=True)
        else:
            jobs_checked = set()
        dump(output / "input_manifest.json", {
            "source_summary": str(summary_source), "input_sha256": fingerprints,
            "selected_repairs": {k: str(v) for k, v in selected.items()},
            "numerical_screening_options": vars(args) | {
                "source_dir": str(args.source_dir), "fallback_dir": str(args.fallback_dir),
                "poiseed_dir": str(args.poiseed_dir), "output_dir": str(output),
                "resume_from": str(resume) if resume else None},
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        for job in sorted(selected):
            try:
                if job in jobs_checked:
                    cached = next(c for c in checks if c["job"] == job)
                    toy = int(option(commands[job], "--pickToy"))
                    directory = Path(cached["source_directory"])
                    value = intervals(read_rows(one(directory, "higgsCombine*.MultiDimFit.mH20*.root"))).get(toy)
                    if value != cached["interval"]:
                        raise RuntimeError("Cached source interval changed")
                    check_fit(fit_info(directory), args.edm_review_threshold)
                    continue
                directory = selected[job]
                command = commands[job]
                toy = int(option(command, "--pickToy"))
                if resume and (resume / "refined" / job).is_dir():
                    refined_dir = resume / "refined" / job
                    recorded = [r for r in load(resume / "run_manifest.json")
                                if r["directory"] == str(refined_dir) and r["exit_code"] == 0]
                    if len(recorded) != 1:
                        raise RuntimeError("Missing unique successful refinement command")
                    directory, command = refined_dir, recorded[0]["command"]
                    print(f"[REUSE REFINEMENT] {job}", flush=True)
                info = fit_info(directory)
                values = intervals(read_rows(one(directory, "higgsCombine*.MultiDimFit.mH20*.root")))
                if toy not in values or not valid_interval(values[toy]):
                    raise RuntimeError("Invalid selected repair interval")
                interval = values[toy]
                # Re-minimise a high-EDM result from all its saved best-fit
                # parameters, rather than returning to the failing nominal seed.
                if (not math.isfinite(info["EDM"]) or info["EDM"] > args.edm_review_threshold):
                    refined = seed_command(command, info)
                    remove(refined, "--cminFallbackAlgo")
                    change(refined, "--cminDefaultMinimizerStrategy", 2)
                    change(refined, "--cminDefaultMinimizerTolerance", 0.001)
                    change(refined, "--setRobustFitStrategy", 2)
                    change(refined, "--setRobustFitTolerance", 0.001)
                    refined_dir = output / "refined" / job
                    rows = run(refined, refined_dir, history)
                    newer = fit_info(refined_dir)
                    check_fit(newer, args.edm_review_threshold)
                    replacement = intervals(rows).get(toy)
                    if replacement is None or not valid_interval(replacement):
                        raise RuntimeError("Refined interval is invalid")
                    refinements.append({"job": job, "before_interval": interval,
                                        "after_interval": replacement,
                                        "before_EDM": info["EDM"], "after_EDM": newer["EDM"]})
                    directory, info, interval, command = refined_dir, newer, replacement, refined
                check_fit(info, args.edm_review_threshold)
                endpoint_checks = []
                center = profile_saved_reference(command, info, toy, interval[1],
                    output / "centers" / job, history, args.minimum_nll_tolerance)
                if abs(center["deltaNLL"]) > args.minimum_nll_tolerance:
                    raise RuntimeError(f"Saved centre is not profile-stable: {center['deltaNLL']}")
                # In v11 --algo singles uses CL=0.68 exactly, whose threshold is
                # 0.494473..., rather than treating it as precisely one sigma.
                target = float(0.5 * ROOT.Math.chisquared_quantile(0.68, 1))
                for label, value in (("low", interval[0]), ("high", interval[2])):
                    check = None
                    cached_dir = resume / "endpoints" / job / label if resume else None
                    if cached_dir and cached_dir.is_dir():
                        cached_rows = read_rows(one(cached_dir, "higgsCombine*.MultiDimFit.mH20*.root"))
                        best = [r for r in cached_rows if r["toy"] == toy and abs(r["quantile"]+1) < 1e-5]
                        points = [r for r in cached_rows if r["toy"] == toy and r["quantile"] >= 0]
                        if len(best) == 1 and len(points) == 1:
                            p = points[0]
                            shift = abs(best[0]["r"]-interval[1])
                            if (math.isfinite(p.get("deltaNLL", float("nan")))
                                    and abs(p["deltaNLL"]-target) <= args.endpoint_nll_tolerance
                                    and abs(p["r"]-value) <= 1e-4
                                    and reference_consistent(shift, interval, center["deltaNLL"],
                                        args.minimum_shift_tolerance, args.minimum_relative_shift_tolerance,
                                        args.minimum_nll_tolerance)):
                                check = {"r": value, "deltaNLL": p["deltaNLL"],
                                         "refitted_best_r": best[0]["r"], "best_r_shift": shift,
                                         "source_directory": str(cached_dir), "reused": True,
                                         "saved_center_deltaNLL": center["deltaNLL"]}
                                print(f"[REUSE ENDPOINT] {job}/{label}", flush=True)
                    if check is None:
                        check = profile_saved_reference(command, info, toy, value,
                            output / "endpoints" / job / label, history, args.endpoint_nll_tolerance)
                    check.update(side=label, target_deltaNLL=target)
                    if abs(check["deltaNLL"]-target) > args.endpoint_nll_tolerance:
                        raise RuntimeError(f"Independent profile check failed: {check}")
                    endpoint_checks.append(check)
                record = {"job": job, "passed": True, "interval": interval,
                          "status": info["status"], "covQual": info["covQual"],
                          "EDM": info["EDM"], "source_directory": str(directory),
                          "endpoints": endpoint_checks, "center_profile": center}
                checks.append(record)
                print(f"[PROFILE PASS] {job}: {interval}", flush=True)
            except Exception as exc:
                message = f"{job}: {exc}"
                errors.append(message)
                print(f"[CHECK] {message}", flush=True)
        if errors or len(checks) != 32:
            raise RuntimeError(f"Only {len(checks)}/32 independent profile checks passed")
        merged, summaries = {}, []
        for old in original_summaries:
            injection = float(old["injected_r"])
            tags = [tag for tag, value in injections.items() if value == injection]
            if len(tags) != 1:
                raise RuntimeError(f"Ambiguous injection tag: {injection}")
            tag = tags[0]
            original_fit = Path(old["fit_file"])
            original = {toy: {"interval": value, "source": str(original_fit), "repaired": False}
                        for toy, value in intervals(read_rows(original_fit)).items()}
            replacements = {int(Path(c["job"]).name.split("_")[-1]):
                            {"interval": c["interval"], "source": c["source_directory"], "repaired": True}
                            for c in checks if Path(c["job"]).parts[0] == tag}
            records = combine_records(original, replacements, 500)
            merged[tag] = {str(toy): record for toy, record in sorted(records.items())}
            summary = summarize(records, injection)
            summary["n_repaired"] = len(replacements)
            summaries.append(summary)
        if len(summaries) != 3:
            raise RuntimeError("Expected three original injection ensembles")
        dump(output / "merged_toys.json", merged)
        dump(output / "bias_summary_Run2Run3_M20.json", summaries)
        print("[COMPLETE] 32 repaired profiles checked; three 500-toy ensembles merged", flush=True)
    except Exception as exc:
        errors.append(str(exc))
        print(f"[INCOMPLETE] {exc}", flush=True)
    finally:
        dump(output / "run_manifest.json", history)
        dump(output / "validation.json", {"passed": not errors, "errors": errors,
             "n_verified_repairs": len(checks), "repairs": checks, "refinements": refinements,
             "coverage_test": False, "original_per_toy_fit_status_available": False})
        print(f"[OUTPUT] {output}", flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
