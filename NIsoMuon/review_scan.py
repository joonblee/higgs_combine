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


def continuation_profile(command, value, origin, seed, anchor, directory,
                         strategy, tolerance, steps, run_command):
    """Profile with r fixed, carrying nuisance seeds from the previous fit.

    Absolute NLL differences must reproduce both the global minimum and an
    existing good grid neighbour. Never infer a missing point from the curve.
    """
    import validate_bias_repair as verify
    directory.mkdir(parents=True)
    record = {'method': 'frozen_poi_continuation', 'directory': str(directory),
              'passed': False, 'strategy': strategy,
              'tolerance': tolerance, 'anchor': anchor, 'steps': []}
    info = seed
    reference_nll = origin['nll'] + origin['nll0']
    try:
        # Generate at the ORIGINAL nominal settings before changing nuisance
        # seeds. Fresh -t -1 fits with shifted --setParameters could otherwise
        # generate a different Asimov dataset at every continuation step.
        generation = list(command)
        for flag in ('--algo', '--points', '--firstPoint', '--lastPoint'):
            verify.remove(generation, flag)
        for flag in ('--saveNLL', '--saveFitResult'):
            verify.remove(generation, flag, False)
        verify.change(generation, '-M', 'GenerateOnly')
        if '--saveToys' not in generation:
            generation.append('--saveToys')
        if verify.option(generation, '-t') != '-1' or float(verify.option(generation, '--expectSignal')) != 0:
            raise RuntimeError('Continuation requires the original background-only Asimov setup')
        data_dir = directory / 'asimov'
        record['dataset_generation'] = {'command': generation, 'directory': str(data_dir)}
        record['dataset_generation']['exit_code'] = run_command(generation, data_dir, log=data_dir / 'generation.log')
        if record['dataset_generation']['exit_code']:
            raise RuntimeError('Nominal Asimov dataset generation failed')
        data_file = verify.one(data_dir, 'higgsCombine*.GenerateOnly.mH*.root')
        record['dataset_generation'].update(file=str(data_file), sha256=digest(data_file))

        def fit_at(r, name, fixed=True):
            nonlocal info
            cmd = list(command)
            for flag in ('--points', '--firstPoint', '--lastPoint'):
                verify.remove(cmd, flag)
            for flag in ('-t', '--expectSignal', '--toysFile'):
                verify.remove(cmd, flag)
            verify.change(cmd, '-D', str(data_file) + ':toys/toy_asimov')
            # CachingAddNLL otherwise chooses a seed-dependent constant offset.
            # Remove only that constant and compute all differences against a
            # reference fitted with the same setting and fixed saved dataset.
            cmd.extend(['--X-rtd', 'REMOVE_CONSTANT_ZERO_POINT=1'])
            verify.remove(cmd, '--skipInitialFit', False)
            verify.remove(cmd, '--cminPoiOnlyFit', False)
            verify.change(cmd, '--algo', 'none')
            parameters = {**info['parameters'], 'r': r}
            verify.change(cmd, '--setParameters', ','.join(
                f'{k}={v:.17g}' for k, v in sorted(parameters.items())))
            if fixed:
                frozen = set((cmd[cmd.index('--freezeParameters') + 1].split(',')
                              if '--freezeParameters' in cmd else [])) | {'r'}
                verify.change(cmd, '--freezeParameters', ','.join(sorted(frozen)))
            verify.change(cmd, '--cminDefaultMinimizerStrategy', strategy)
            verify.change(cmd, '--cminDefaultMinimizerTolerance', tolerance)
            if '--saveFitResult' not in cmd:
                cmd.append('--saveFitResult')
            where = directory / name
            item = {'r': r, 'command': cmd, 'directory': str(where)}
            record['steps'].append(item)
            item['exit_code'] = run_command(cmd, where, log=where / 'fit.log')
            if item['exit_code']:
                raise RuntimeError(f'Conditional fit command failed at r={r:g}')
            info = verify.fit_info(where)
            item['conditional_fit'] = info
            verify.check_fit(info, .001)
            if fixed and 'r' in info['parameters']:
                raise RuntimeError('The supposedly fixed POI is still floating')
            rows = read_rows(verify.one(where, 'higgsCombine*.MultiDimFit.mH*.root'))
            if len(rows) != 1 or abs(rows[0]['r'] - r) > 1e-4:
                raise RuntimeError(f'Wrong frozen-r output at r={r:g}')
            saved = reference(rows)
            item['nll'] = saved['nll'] + saved['nll0']
            item['deltaNLL'] = item['nll'] - reference_nll
            return item['deltaNLL']

        fit_at(origin['r'], 'global_reference', fixed=False)
        reference_nll = record['steps'][-1]['nll']
        center = fit_at(origin['r'], 'center')
        if abs(center) > REFERENCE_TOLERANCE:
            raise RuntimeError('Conditional centre does not reproduce the global NLL')
        for i in range(1, steps + 1):
            target = origin['r'] + (anchor['r'] - origin['r']) * i / steps
            delta = fit_at(target, f'walk_{i}')
            if delta < -REFERENCE_TOLERANCE or delta >= 1000:
                raise RuntimeError('Invalid conditional NLL on the continuation path')
        if abs(delta - anchor['deltaNLL']) > PROFILE_TOLERANCE:
            raise RuntimeError('Conditional neighbour does not reproduce the existing grid NLL')
        delta = fit_at(value, 'target')
        point = {'r': value, 'deltaNLL': delta,
                 'quantileExpected': math.erfc(math.sqrt(max(0., delta))),
                 'nll': origin['nll'], 'nll0': origin['nll0']}
        if row_issues(point, origin):
            raise RuntimeError('Invalid final conditional profile')
        record['point'] = point
        record['passed'] = True
        return point, record
    except (RuntimeError, OSError, ValueError) as exc:
        record['error'] = str(exc)
        return None, record
    finally:
        dump(directory / 'continuation.json', record)


def retry_point(command, index, value, origin, directory, run_command, anchor=None):
    import validate_bias_repair as verify
    candidates, attempts, references = [], [], []
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
            # MultiDimFit's Grid branch never calls saveResult(), even with
            # --saveFitResult. Obtain a saved reference using the None branch.
            baseline = list(cmd)
            verify.change(baseline, '--algo', 'none')
            for flag in ('--firstPoint', '--lastPoint', '--points'):
                verify.remove(baseline, flag)
            initial_dir = attempt / 'initial'
            record['initial_fit_command'] = baseline
            record['initial_fit_directory'] = str(initial_dir)
            record['initial_fit_exit_code'] = run_command(baseline, initial_dir, log=initial_dir / 'fit.log')
            if record['initial_fit_exit_code']:
                raise RuntimeError('Independent initial fit command failed')
            info = verify.fit_info(initial_dir)
            record["initial_fit"] = info
            verify.check_fit(info, .001)
            initial = reference(rows)
            baseline_rows = read_rows(verify.one(initial_dir, 'higgsCombine*.MultiDimFit.mH*.root'))
            saved = reference(baseline_rows)
            if (abs(saved['nll'] + saved['nll0'] - initial['nll'] - initial['nll0']) > REFERENCE_TOLERANCE
                    or abs(saved['r'] - initial['r']) > 1e-4):
                raise RuntimeError('Independent saved fit does not match the grid initial minimum')
            step = (float(verify.option(cmd, '--rMax')) - float(verify.option(cmd, '--rMin'))) / int(verify.option(cmd, '--points'))
            if abs(initial["r"] - origin["r"]) > step:
                raise RuntimeError("Retry changed the initial best-fit r")
            if abs(saved['nll'] + saved['nll0'] - origin['nll'] - origin['nll0']) > REFERENCE_TOLERANCE:
                raise RuntimeError('Saved reference differs from the original global NLL')
            references.append(info)
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
    if anchor is not None and references:
        print(f'[SCAN CONTINUATION] r={value:g}, checked neighbour r={anchor["r"]:g}', flush=True)
        for strategy, tolerance, steps in ((1, .01, 4), (2, .01, 4), (1, .001, 8), (2, .001, 8)):
            point, record = continuation_profile(command, value, origin, references[0], anchor,
                directory / f'continuation_s{strategy}_t{tolerance}_n{steps}',
                strategy, tolerance, steps, run_command)
            attempts.append(record)
            dump(directory / 'attempts.json', {'attempts': attempts})
            if point is not None:
                candidates.append(point)
            if len(candidates) >= 2 and max(p['deltaNLL'] for p in candidates) - min(p['deltaNLL'] for p in candidates) <= PROFILE_TOLERANCE:
                return min(candidates, key=lambda p: p['deltaNLL']), attempts
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
                anchor = min((p for i, p in good.items() if first <= i <= last and i not in holes),
                             key=lambda p: abs(p['r'] - value))
                point, history = retry_point(command, index, value, origin, attempt / f"point_{index}", run_command, anchor=anchor)
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
