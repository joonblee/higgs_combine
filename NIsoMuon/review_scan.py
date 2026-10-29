"""Audit and repair the plotted region of a background-only Asimov scan.

The original grid ROOT file is read-only. Tail failures outside the region
bracketed by points below 2*deltaNLL=7 are reported, not called converged.
Only holes in that region are retried, at the original grid coordinates and
with the original dataset/model/range. This is not a per-point EDM audit of
the original grid: Combine does not save those RooFitResults.
"""
from array import array
import hashlib
import json
import math
from pathlib import Path
import tempfile


SCHEMA = "NIsoMuon_scan_validation_v1"
Y_CUT = 7.0
REFERENCE_TOLERANCE = 0.001
PROFILE_TOLERANCE = 0.005  # deltaNLL; 0.01 in -2*Delta ln L
FIELDS = ("r", "deltaNLL", "quantileExpected", "nll", "nll0")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    # ROOT can contain NaNs precisely in the failed rows being diagnosed.
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        return value
    Path(path).write_text(json.dumps(clean(data), indent=2, allow_nan=False) + "\n")


def read_rows(path):
    import ROOT
    source = ROOT.TFile.Open(str(path))
    if not source or source.IsZombie():
        raise RuntimeError(f"Cannot open scan ROOT file: {path}")
    try:
        tree = source.Get("limit")
        if not tree or any(not tree.GetBranch(k) for k in FIELDS):
            raise RuntimeError(f"Missing scan branches (including --saveNLL): {path}")
        return [{k: float(getattr(row, k)) for k in FIELDS} for row in tree]
    finally:
        source.Close()


def write_rows(path, rows):
    import ROOT
    destination = ROOT.TFile.Open(str(path), "RECREATE")
    if not destination or destination.IsZombie():
        raise RuntimeError(f"Cannot write validated scan: {path}")
    try:
        tree = ROOT.TTree("limit", "validated background-only Asimov scan")
        values = {k: array("d", [0.]) for k in FIELDS}
        for key, value in values.items():
            tree.Branch(key, value, key + "/D")
        for row in rows:
            for key in FIELDS:
                values[key][0] = row[key]
            tree.Fill()
        tree.Write()
    finally:
        destination.Close()


def coordinate(index, low, high, count):
    # Combine's default grid samples bin centres (no --alignEdges).
    return low + (index + 0.5) * (high - low) / count


def reference(rows):
    best = [r for r in rows if r["quantileExpected"] == -1.]
    if not best or any(not all(math.isfinite(r[k]) for k in FIELDS) for r in best):
        raise RuntimeError("Missing or non-finite initial scan minimum")
    origin = best[0]
    for row in best:
        if (abs(row["deltaNLL"]) > REFERENCE_TOLERANCE
                or abs(row["nll"] + row["nll0"] - origin["nll"] - origin["nll0"]) > REFERENCE_TOLERANCE
                or abs(row["r"] - origin["r"]) > 1e-4):
            raise RuntimeError("Inconsistent initial minima across scan chunks")
    return origin


def row_issues(row, origin):
    issues = []
    if not all(math.isfinite(row[k]) for k in FIELDS):
        return ["non-finite scan row"]
    if not 0 <= row["quantileExpected"] <= 1:
        issues.append("invalid scan quantile")
    if row["deltaNLL"] >= 1000:
        issues.append("invalid/sentinel deltaNLL")
    if row["deltaNLL"] < -REFERENCE_TOLERANCE:
        issues.append("scan point below the initial minimum")
    if abs(row["nll"] + row["nll0"] - origin["nll"] - origin["nll0"]) > REFERENCE_TOLERANCE:
        issues.append("different initial NLL reference")
    return issues


def analyse(rows, low, high, count):
    origin = reference(rows)
    step = (high - low) / count
    groups = {}
    for row in rows:
        if row["quantileExpected"] == -1.:
            continue  # One repeated global minimum per distributed chunk.
        if not math.isfinite(row["r"]):
            raise RuntimeError("Non-finite scan coordinate")
        index = round((row["r"] - low) / step - 0.5)
        if not 0 <= index < count or abs(row["r"] - coordinate(index, low, high, count)) > step * .01:
            raise RuntimeError(f"Off-grid scan coordinate: {row['r']}")
        groups.setdefault(index, []).append(row)
    good, problems = {}, {}
    for index in range(count):
        candidates = groups.get(index, [])
        issues = ([f"expected one grid row, found {len(candidates)}"] if len(candidates) != 1
                  else row_issues(candidates[0], origin))
        if issues:
            problems[index] = {"r": coordinate(index, low, high, count),
                               "issues": issues, "rows": candidates}
        else:
            good[index] = candidates[0]
    visible = [i for i, row in good.items() if 2 * row["deltaNLL"] <= Y_CUT]
    if len(visible) < 2:
        raise RuntimeError("Too few valid scan points below the plotting cut")
    first, last = min(visible), max(visible)
    holes = [i for i in range(first, last + 1)
             if i not in good or 2 * good[i]["deltaNLL"] > Y_CUT]
    for index in holes:
        if index not in problems:
            problems[index] = {"r": coordinate(index, low, high, count),
                "issues": ["point above plotting cut inside the displayed region"], "rows": groups[index]}
    return origin, good, problems, holes, (first, last)


def crossing_brackets(points, best_r):
    result = {}
    for level in (1., 4.):
        bounds = {}
        for side, selected in (("low", sorted((p for p in points if p["r"] <= best_r), key=lambda p: -p["r"])),
                               ("high", sorted((p for p in points if p["r"] >= best_r), key=lambda p: p["r"]))):
            for inner, outer in zip(selected, selected[1:]):
                if 2 * inner["deltaNLL"] <= level <= 2 * outer["deltaNLL"]:
                    bounds[side] = [{"r": p["r"], "two_deltaNLL": 2 * p["deltaNLL"]} for p in (inner, outer)]
                    break
            if side not in bounds:
                raise RuntimeError(f"Scan does not bracket both sides of 2*deltaNLL={level:g}")
        result[str(int(level))] = bounds
    return result


def retry_point(command, index, value, origin, directory, run_command):
    import validate_bias_repair as verify
    candidates, attempts = [], []
    for strategy, tolerance in ((1, .01), (2, .01), (1, .001), (2, .001)):
        attempt = directory / f"s{strategy}_t{tolerance}"
        attempt.mkdir(parents=True)
        cmd = list(command)
        for flag, setting in (("--firstPoint", index), ("--lastPoint", index),
                              ("--cminDefaultMinimizerStrategy", strategy),
                              ("--cminDefaultMinimizerTolerance", tolerance)):
            verify.change(cmd, flag, setting)
        cmd.append("--saveFitResult")
        record = {"command": cmd, "directory": str(attempt)}
        attempts.append(record)
        try:
            record["exit_code"] = run_command(cmd, attempt, log=attempt / "fit.log")
            if record["exit_code"]:
                raise RuntimeError("Single-point scan command failed")
            rows = read_rows(verify.one(attempt, "higgsCombine*.MultiDimFit.mH*.root"))
            # --saveFitResult saves the INITIAL global fit, not the grid profile.
            info = verify.fit_info(attempt)
            verify.check_fit(info, .001)
            record["initial_fit"] = info
            initial = reference(rows)
            step = (float(verify.option(cmd, '--rMax')) - float(verify.option(cmd, '--rMin'))) / int(verify.option(cmd, '--points'))
            if abs(initial["r"] - origin["r"]) > step:
                raise RuntimeError("Retry changed the initial best-fit r")
            points = [r for r in rows if r["quantileExpected"] != -1.]
            if len(points) != 1 or abs(points[0]["r"] - value) > 1e-4:
                raise RuntimeError("Missing or wrong single-point profile entry")
            point = points[0]
            issues = row_issues(point, origin)
            if issues:
                raise RuntimeError("; ".join(issues))
            # Bring tiny permitted reference differences back to the original grid.
            point = dict(point)
            point["deltaNLL"] += point["nll"] + point["nll0"] - origin["nll"] - origin["nll0"]
            point["nll"], point["nll0"] = origin["nll"], origin["nll0"]
            if row_issues(point, origin):
                raise RuntimeError("Rebased profile is below the original minimum")
            record["point"] = point
            candidates.append(point)
        except (RuntimeError, OSError, ValueError) as exc:
            record["error"] = str(exc)
        dump(directory / 'attempts.json', {'attempts': attempts})
        if len(candidates) >= 2:
            spread = max(p["deltaNLL"] for p in candidates) - min(p["deltaNLL"] for p in candidates)
            if spread <= PROFILE_TOLERANCE:
                # Select the lower NLL, never the point closest to an expected curve.
                return min(candidates, key=lambda p: p["deltaNLL"]), attempts
    raise RuntimeError(f"No two consistent profiles at r={value:g}; see {directory}")


def validate_scan(root_file, command, output, label, low, high, count, run_command):
    report_path = output / f"scan_validation_M{label}.json"
    validated = output / f"scan_validated_M{label}.root"
    report = {"schema": SCHEMA, "passed": False, "dataset": "background-only Asimov",
              "source": str(root_file), "source_sha256": digest(root_file),
              "grid": {"r_min": low, "r_max": high, "points": count}, "y_cut": Y_CUT,
              "original_per_point_fit_status_available": False, "repairs": []}
    try:
        origin, good, problems, holes, (first, last) = analyse(read_rows(root_file), low, high, count)
        report.update(initial_minimum=origin, initial_problems=problems,
                      repair_indices=holes, displayed_grid_indices=[first, last],
                      tail_problems={i: p for i, p in problems.items() if i < first or i > last})
        dump(report_path, report)
        print(f"[SCAN CHECK] M{label}: {len(holes)} holes in the plotted region; "
              f"{len(report['tail_problems'])} tail problems recorded", flush=True)
        if holes:
            attempt = Path(tempfile.mkdtemp(prefix=f"scan_repair_M{label}_", dir=output))
            report['repair_directory'] = str(attempt)
            dump(report_path, report)
            for index in holes:
                value = coordinate(index, low, high, count)
                print(f"[SCAN RETRY] grid point {index}, r={value:g}", flush=True)
                point, history = retry_point(command, index, value, origin, attempt / f"point_{index}", run_command)
                good[index] = point
                report["repairs"].append({"index": index, "r": value, "point": point, "attempts": history})
                dump(report_path, report)
        selected = [good[i] for i in range(first, last + 1)]
        if any(2 * p["deltaNLL"] > Y_CUT for p in selected):
            raise RuntimeError("The plotted region still has points above the plotting cut")
        selected.append(origin)
        selected.sort(key=lambda p: p["r"])
        report["crossing_brackets"] = crossing_brackets(selected, origin["r"])
        # Keep the exact deltaNLL=0 minimum when it coincides with a grid point;
        # plot1DScan otherwise removes one of the equal-x entries arbitrarily.
        selected = [origin] + [p for p in selected if p['r'] != origin['r']]
        selected.sort(key=lambda p: p['r'])
        write_rows(validated, selected)
        report.update(passed=True, validated_root=str(validated), validated_sha256=digest(validated),
                      n_displayed_grid_points=last - first + 1)
        print(f"[SCAN PASS] M{label}: plotted grid complete, levels 1 and 4 bracketed", flush=True)
        return validated
    except (RuntimeError, OSError, ValueError) as exc:
        report["error"] = str(exc)
        raise
    finally:
        dump(report_path, report)
