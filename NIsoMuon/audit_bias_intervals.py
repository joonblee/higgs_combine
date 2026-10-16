#!/usr/bin/env python3
"""Read-only audit of saved Combine --algo singles toy intervals.

No fits or toy generation are run. Exit 0 means the stored intervals pass
structural checks, 2 means toys need inspection, and 1 means an input error.
This does not establish minimiser convergence, endpoint deltaNLL or coverage.
"""
import argparse
import hashlib
import json
import math
import re
import statistics
import tempfile
from pathlib import Path


QUANTILES = (("low", -0.32), ("best", -1.0), ("high", 0.32))


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def audit_rows(rows, count, bounds):
    grouped, extra, unrecognised = {}, set(), []
    for row in rows:
        toy = row["toy"]
        if toy not in range(1, count + 1):
            extra.add(toy)
        matches = [name for name, q in QUANTILES
                   if math.isfinite(row["quantile"]) and abs(row["quantile"] - q) < 1e-5]
        if len(matches) != 1:
            unrecognised.append({"toy": toy, "quantile": row["quantile"]})
            continue
        grouped.setdefault(toy, {}).setdefault(matches[0], []).append(row["r"])
    records, flagged, valid = {}, [], {}
    for toy in range(1, count + 1):
        values = grouped.get(toy, {})
        issues = []
        for name, _ in QUANTILES:
            n = len(values.get(name, []))
            if n != 1:
                issues.append(f"{name}: expected one row, found {n}")
        interval = None
        if not issues:
            interval = [values[name][0] for name, _ in QUANTILES]
            lo, best, hi = interval
            if not all(math.isfinite(x) for x in interval):
                issues.append("Non-finite interval")
            else:
                if not lo < best < hi:
                    issues.append("Collapsed or unordered interval")
                if lo <= bounds[0] + 1e-5 or hi >= bounds[1] - 1e-5:
                    issues.append("Interval at or outside fitted r range")
        record = {"toy": toy, "passed": not issues, "interval": interval,
                  "row_counts": {name: len(values.get(name, [])) for name, _ in QUANTILES},
                  "issues": issues}
        records[str(toy)] = record
        if issues:
            flagged.append(toy)
        else:
            valid[toy] = interval
    return {"passed": not flagged and not extra and not unrecognised,
            "n_expected": count, "n_valid_intervals": len(valid),
            "flagged_toy_ids": flagged, "unexpected_toy_ids": sorted(extra),
            "unrecognised_rows": unrecognised, "toys": records}


def read_rows(path):
    import ROOT
    f = ROOT.TFile.Open(str(path), "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open {path}")
    try:
        tree = f.Get("limit")
        if not tree:
            raise RuntimeError(f"No limit tree: {path}")
        for branch in ("iToy", "quantileExpected", "r"):
            if not tree.GetBranch(branch):
                raise RuntimeError(f"Missing branch {branch}: {path}")
        return [{"toy": int(e.iToy), "quantile": float(e.quantileExpected),
                 "r": float(e.r)} for e in tree]
    finally:
        f.Close()


def fit_range(log):
    ranges = {(float(a), float(b)) for a, b in re.findall(
        r"Set Range of Parameter r To\s*:\s*\(\s*([^,]+),\s*([^\)]+)\)", log)}
    if len(ranges) != 1:
        raise RuntimeError(f"Expected one unambiguous fitted r range in log, found {ranges}")
    bounds = ranges.pop()
    if not all(math.isfinite(x) for x in bounds) or bounds[0] >= bounds[1]:
        raise RuntimeError(f"Invalid fitted r range: {bounds}")
    return bounds


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Review mass directory containing bias/")
    parser.add_argument("--mass", type=int, default=70)
    parser.add_argument("--target", default="Run2Run3")
    parser.add_argument("--expected-toys", type=int, default=500)
    args = parser.parse_args()
    if args.expected_toys < 2:
        parser.error("--expected-toys must be at least 2")
    source = args.source_dir.resolve()
    output = Path(tempfile.mkdtemp(prefix=f"bias_M{args.mass}_audit.", dir=Path.cwd()))
    report = {"passed": False, "errors": [], "ensembles": [],
              "check_scope": "Saved interval completeness, ordering, finiteness and range only",
              "original_per_toy_fit_status_available": False,
              "independent_endpoint_profiles_checked": False, "coverage_test": False,
              "script_sha256": sha256(Path(__file__)), "input_sha256": {}}
    fatal = False
    try:
        summary_path = source / "bias" / f"bias_summary_{args.target}_M{args.mass}.json"
        summary = json.loads(summary_path.read_text())
        report["input_sha256"][str(summary_path)] = sha256(summary_path)
        if len(summary) != 3 or len({x["injected_r"] for x in summary}) != 3:
            raise RuntimeError("Expected three distinct injection ensembles")
        for point in summary:
            if point["n_toys_requested"] != args.expected_toys:
                raise RuntimeError("Requested toy count differs from original summary")
            # Use the source directory explicitly supplied, not an old absolute path.
            fit = source / "bias" / Path(point["fit_file"]).name
            tag = re.search(r"_r([^.]*)\.MultiDimFit\.mH", fit.name)
            if not tag:
                raise RuntimeError(f"Cannot identify injection tag: {fit.name}")
            log_path = source / "bias" / f"bias_fit_r{tag[1]}.log"
            text = log_path.read_text()
            bounds = fit_range(text)
            result = audit_rows(read_rows(fit), args.expected_toys, bounds)
            result.update({"tag": "r" + tag[1], "injected_r": point["injected_r"],
                           "fit_file": str(fit), "r_range": list(bounds),
                           "log_minos_failure_count": text.count("Minos error calculation failed"),
                           "log_initial_failure_count": text.count("Initial minimization failed"),
                           "log_invalid_low_error_count": text.count("No valid low-error found"),
                           "log_invalid_high_error_count": text.count("No valid high-error found")})
            for path in (fit, log_path):
                report["input_sha256"][str(path)] = sha256(path)
            # Only publish ensemble statistics when every stored interval passes.
            if result["passed"]:
                values = [record["interval"][1] for record in result["toys"].values()]
                pulls = []
                for record in result["toys"].values():
                    lo, best, hi = record["interval"]
                    error = best-lo if best >= point["injected_r"] else hi-best
                    pulls.append((best-point["injected_r"]) / error)
                result["structural_summary"] = {
                    "mean_bias_r": statistics.fmean(values)-point["injected_r"],
                    "mean_bias_mc_standard_error": statistics.stdev(values)/math.sqrt(len(values)),
                    "mean_pull": statistics.fmean(pulls), "pull_width": statistics.stdev(pulls)}
            report["ensembles"].append(result)
            print(f"[AUDIT] r={point['injected_r']:g}: {result['n_valid_intervals']}/{args.expected_toys} valid intervals", flush=True)
            print(f"        retry/inspect IDs: {result['flagged_toy_ids']}", flush=True)
            for toy in result["flagged_toy_ids"]:
                record = result["toys"][str(toy)]
                print(f"[CHECK] {result['tag']}/toy_{toy:04d}: {record['interval']} {record['issues']}", flush=True)
        report["passed"] = len(report["ensembles"]) == 3 and all(x["passed"] for x in report["ensembles"])
    except Exception as exc:
        fatal = True
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        print(f"[ERROR] {exc}", flush=True)
    (output / "audit.json").write_text(json.dumps(clean(report), indent=2, allow_nan=False) + "\n")
    print(f"[SAVED] {output / 'audit.json'}", flush=True)
    print("[SCOPE] Structural interval audit only; no new minimisations or endpoint profiles.", flush=True)
    return 1 if fatal else 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
