#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NPS-26-009 Combine review helper.

This script is deliberately separate from limit_workflow.py.
It does NOT rebuild production datacards or rerun the nominal expected-limit
workflow.  Instead it runs review/validation diagnostics on already-produced
blind datacards.

Default assumptions
-------------------
* Run from the NIsoMuon repository root after cmsenv.
* Datacards live under
    datacards_NIsoMuon/alpha/blind/combined/<TARGET>/
* Existing internal-r expected-limit JSON lives under
    limit_outputs/alpha/blind/<TARGET>/limits_<TARGET>_alpha_internal.json
* Review outputs are isolated under review_outputs/<TARGET>/M<MASS>/.

Examples
--------
  # The immediate NPS validation check requested for M=20
  python3 combine_review.py 20 --task validate

  # Validate more than one representative point
  python3 combine_review.py 70 --task validate

  # B-only 1D likelihood scan; workspace is created automatically
  python3 combine_review.py 20 --task scan

  # Correlation matrix from a fresh B-only FitDiagnostics run
  python3 combine_review.py 20 --task correlation

  # HybridNew median expected limit, compared with the already-existing
  # AsymptoticLimits median in limits_<TARGET>_alpha_internal.json
  python3 combine_review.py 70 --task hybrid --hybrid-toys 500 --hybrid-parallel 24

  # S+B Asimov impacts.  By default inject the existing median expected r95.
  python3 combine_review.py 20 --task sb-impacts --impact-parallel 12

  # Override the injected internal r explicitly
  python3 combine_review.py 20 --task sb-impacts --inject-r 18.125

  # Signal-recovery/bias toys at 0, 1, 2 times the existing median expected r95
  python3 combine_review.py 20 --task bias --bias-toys 500 --bias-multipliers 0,1,2

  # Several tasks in one invocation; workspace is generated only once
  python3 combine_review.py 70 --task validate,scan,hybrid,correlation

Tasks already covered by limit_workflow.py and therefore intentionally NOT
reimplemented here:
* nominal blind AsymptoticLimits production;
* B-only FitDiagnostics/pulls used in the nominal workflow;
* B-only impacts used in the nominal workflow.

The correlation task reruns a small B-only FitDiagnostics job only because the
fitDiagnostics ROOT files are transient/ignored in the analysis Git repository
and the RooFitResult is needed to extract the correlation matrix.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TASKS = (
    "validate",
    "workspace",
    "scan",
    "hybrid",
    "sb-impacts",
    "bias",
    "correlation",
)


def mass_label(mass: float) -> str:
    if abs(mass - round(mass)) < 1.0e-9:
        return str(int(round(mass)))
    return f"{mass:.8g}".replace(".", "p")


def mass_for_combine(mass: float) -> str:
    return str(int(round(mass))) if abs(mass - round(mass)) < 1.0e-9 else f"{mass:.8g}"


def parse_csv_floats(value: str) -> List[float]:
    out: List[float] = []
    for token in re.split(r"[,;:\s]+", value.strip()):
        if token:
            out.append(float(token))
    if not out:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    return out


def parse_tasks(value: str) -> List[str]:
    raw = [x.strip().lower() for x in value.split(",") if x.strip()]
    if not raw:
        raise argparse.ArgumentTypeError("At least one task is required.")
    if raw == ["all"]:
        return list(TASKS)
    bad = [x for x in raw if x not in TASKS]
    if bad:
        raise argparse.ArgumentTypeError(
            "Unknown task(s): " + ", ".join(bad) + ". Allowed: " + ", ".join(TASKS)
        )
    # Preserve user order while removing duplicates.
    out: List[str] = []
    for task in raw:
        if task not in out:
            out.append(task)
    return out


def require_command(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    raise RuntimeError(
        f"Required command '{name}' is not in PATH. Run cmsenv in the CMSSW area first."
    )


def validate_script() -> List[str]:
    exe = shutil.which("ValidateDatacards.py")
    if exe:
        return [exe]
    cmssw = os.environ.get("CMSSW_BASE", "")
    if cmssw:
        script = Path(cmssw) / "src/CombineHarvester/CombineTools/scripts/ValidateDatacards.py"
        if script.is_file():
            return [sys.executable, str(script)]
    raise RuntimeError(
        "ValidateDatacards.py was not found. Install/build CombineHarvester and run cmsenv."
    )


def run_command(
    cmd: Sequence[str],
    cwd: Path,
    *,
    dry_run: bool = False,
    log: Optional[Path] = None,
) -> int:
    cwd.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(x) for x in cmd)
    print(f"[RUN] (cd {cwd} && {printable})", flush=True)
    if dry_run:
        return 0

    if log is None:
        completed = subprocess.run(list(cmd), cwd=str(cwd), check=False)
        return int(completed.returncode)

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        completed = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert completed.stdout is not None
        for line in completed.stdout:
            print(line, end="")
            stream.write(line)
        completed.wait()
        return int(completed.returncode)


def find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    if not matches:
        raise RuntimeError(f"No output matched {directory / pattern}")
    return matches[-1]


def card_path(args: argparse.Namespace) -> Path:
    label = mass_label(args.mass)
    path = (
        Path(args.card_base)
        / args.target
        / f"datacard_M-{label}_{args.target}.txt"
    )
    if not path.is_file():
        raise RuntimeError(f"Datacard not found: {path}")
    return path.resolve()


def outdir(args: argparse.Namespace) -> Path:
    return (Path(args.output_base) / args.target / f"M{mass_label(args.mass)}").resolve()


def parse_alpha_unit(card: Path) -> Optional[float]:
    pattern = re.compile(r"^#\s*alpha_internal_unit\s*=\s*([+\-0-9.eE]+)\s*$")
    for line in card.read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            return float(match.group(1))
    return None


def parse_rate_block(card: Path) -> Tuple[List[str], List[str], List[int], List[float]]:
    """Return (bins, process_names, process_ids, rates) from a standard card."""
    lines = [line.strip() for line in card.read_text().splitlines()]
    proc_name_idx = None
    for i, line in enumerate(lines):
        if not line.startswith("process "):
            continue
        toks = line.split()[1:]
        if toks and any(not re.fullmatch(r"[+\-]?\d+", tok) for tok in toks):
            proc_name_idx = i
            break
    if proc_name_idx is None:
        raise RuntimeError(f"Could not locate process-name row in {card}")

    # Find the closest preceding bin row with the same number of columns.
    process_names = lines[proc_name_idx].split()[1:]
    bins: Optional[List[str]] = None
    for j in range(proc_name_idx - 1, -1, -1):
        if lines[j].startswith("bin "):
            candidate = lines[j].split()[1:]
            if len(candidate) == len(process_names):
                bins = candidate
                break
    if bins is None:
        raise RuntimeError(f"Could not locate process-bin row in {card}")

    proc_ids: Optional[List[int]] = None
    rates: Optional[List[float]] = None
    for j in range(proc_name_idx + 1, min(len(lines), proc_name_idx + 5)):
        if lines[j].startswith("process "):
            toks = lines[j].split()[1:]
            if len(toks) == len(process_names) and all(re.fullmatch(r"[+\-]?\d+", x) for x in toks):
                proc_ids = [int(x) for x in toks]
        if lines[j].startswith("rate "):
            toks = lines[j].split()[1:]
            if len(toks) == len(process_names):
                rates = [float(x) for x in toks]
    if proc_ids is None or rates is None:
        raise RuntimeError(f"Could not locate process-id/rate row in {card}")
    return bins, process_names, proc_ids, rates


def safe_negative_rmin(card: Path, requested: float) -> float:
    """Keep nominal expected yields positive in every counting channel."""
    bins, _names, proc_ids, rates = parse_rate_block(card)
    by_bin: Dict[str, Dict[str, float]] = {}
    for channel, pid, rate in zip(bins, proc_ids, rates):
        slot = by_bin.setdefault(channel, {"sig": 0.0, "bkg": 0.0})
        if pid <= 0:
            slot["sig"] += rate
        else:
            slot["bkg"] += rate

    boundaries: List[float] = []
    for payload in by_bin.values():
        sig = payload["sig"]
        bkg = payload["bkg"]
        if sig > 0.0:
            boundaries.append(-bkg / sig)
    if not boundaries:
        return requested

    # r must be strictly above the largest (-B/S) boundary.
    nominal_boundary = max(boundaries)
    margin = 1.0e-4 * max(1.0, abs(nominal_boundary))
    safe = nominal_boundary + margin
    return max(requested, safe)


def expected_limit_json(args: argparse.Namespace) -> Path:
    if args.expected_json:
        return Path(args.expected_json).resolve()
    return (
        Path(args.limit_output_base)
        / args.target
        / f"limits_{args.target}_alpha_internal.json"
    ).resolve()


def existing_expected_r95(args: argparse.Namespace) -> float:
    path = expected_limit_json(args)
    if not path.is_file():
        raise RuntimeError(
            f"Existing internal-limit JSON not found: {path}. "
            "Supply --inject-r explicitly or --expected-json PATH."
        )
    payload = json.loads(path.read_text())
    candidates = [str(float(args.mass)), mass_for_combine(args.mass), mass_label(args.mass)]
    for key in candidates:
        if key in payload and isinstance(payload[key], dict) and "exp0" in payload[key]:
            value = float(payload[key]["exp0"])
            if not math.isfinite(value) or value <= 0:
                break
            return value
    # Numeric-key fallback.
    for key, values in payload.items():
        try:
            key_mass = float(key)
        except Exception:
            continue
        if abs(key_mass - args.mass) < 1.0e-6 and isinstance(values, dict) and "exp0" in values:
            return float(values["exp0"])
    raise RuntimeError(f"No exp0 value for M={args.mass:g} in {path}")


def workspace_path(args: argparse.Namespace) -> Path:
    return outdir(args) / f"workspace_{args.target}_M{mass_label(args.mass)}.root"


def ensure_workspace(args: argparse.Namespace, card: Path) -> Path:
    workspace = workspace_path(args)
    if args.dry_run:
        print(f"[WORKSPACE] would create/reuse {workspace}")
        return workspace
    if (
        workspace.is_file()
        and not args.force
        and workspace.stat().st_mtime >= card.stat().st_mtime
    ):
        print(f"[WORKSPACE] reuse {workspace}")
        return workspace
    require_command("text2workspace.py")
    cmd = [
        "text2workspace.py",
        str(card),
        "-m",
        mass_for_combine(args.mass),
        "-o",
        workspace.name,
    ]
    status = run_command(cmd, outdir(args), dry_run=args.dry_run, log=outdir(args) / "text2workspace.log")
    if status != 0 or not workspace.is_file():
        raise RuntimeError(f"text2workspace.py failed for {card}")
    return workspace


def task_validate(args: argparse.Namespace, card: Path) -> None:
    output = outdir(args)
    json_file = output / f"validation_M{mass_label(args.mass)}.json"
    log_file = output / f"validation_M{mass_label(args.mass)}.log"
    cmd = [
        *validate_script(),
        str(card),
        "--mass",
        mass_for_combine(args.mass),
        "--printLevel",
        str(args.print_level),
        "--jsonFile",
        str(json_file),
    ]
    status = run_command(cmd, output, dry_run=args.dry_run, log=log_file)
    if status != 0:
        raise RuntimeError(f"ValidateDatacards.py failed with exit code {status}")
    if args.dry_run:
        return
    if not json_file.is_file():
        raise RuntimeError(f"Validation JSON was not produced: {json_file}")
    print("\n[VALIDATION JSON]")
    print(json.dumps(json.loads(json_file.read_text()), indent=2, sort_keys=True))
    print(f"\n[SAVED] {json_file}")
    print(f"[SAVED] {log_file}")


def task_scan(args: argparse.Namespace, card: Path) -> None:
    require_command("combine")
    workspace = ensure_workspace(args, card)
    output = outdir(args)
    rmin = safe_negative_rmin(card, args.r_min)
    tag = f"ReviewScan_{args.target}_M{mass_label(args.mass)}"
    cmd = [
        "combine",
        "-M",
        "MultiDimFit",
        str(workspace),
        "-m",
        mass_for_combine(args.mass),
        "-t",
        "-1",
        "--expectSignal",
        "0",
        "--setParameters",
        "r=0",
        "--algo",
        "grid",
        "--points",
        str(args.scan_points),
        "--rMin",
        f"{rmin:.12g}",
        "--rMax",
        f"{args.r_max:.12g}",
        "--saveNLL",
        "-n",
        f".{tag}",
    ]
    if args.scan_parallel > 1:
        cmssw = Path(os.environ.get("CMSSW_BASE", ""))
        parallel_scan = cmssw / "src/HiggsAnalysis/CombinedLimit/test/parallelScan.py"
        if not parallel_scan.is_file():
            raise RuntimeError(f"parallelScan.py not found: {parallel_scan}")
        scan_cmd = [
            sys.executable, str(parallel_scan), *cmd,
            "-j", str(args.scan_parallel), "--hadd",
        ]
    else:
        scan_cmd = cmd
    status = run_command(scan_cmd, output, dry_run=args.dry_run, log=output / f"scan_M{mass_label(args.mass)}.log")
    if status != 0:
        raise RuntimeError("MultiDimFit likelihood scan failed")
    if args.dry_run:
        return

    root_file = find_one(output, f"higgsCombine.{tag}.MultiDimFit.mH*.root")
    plotter = shutil.which("plot1DScan.py")
    if not plotter:
        print(f"[WARNING] plot1DScan.py not found; retained {root_file}", file=sys.stderr)
        return
    plot_base = output / f"scan_{args.target}_M{mass_label(args.mass)}_Bonly"
    status = run_command(
        [plotter, str(root_file), "--POI", "r", "-o", str(plot_base)],
        output,
        dry_run=args.dry_run,
        log=output / f"scan_plot_M{mass_label(args.mass)}.log",
    )
    if status != 0:
        raise RuntimeError("plot1DScan.py failed")
    print(f"[SAVED] {plot_base}.pdf")


def read_limit_tree_value(root_path: Path) -> float:
    try:
        import ROOT  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"PyROOT is required to inspect {root_path}: {exc}")
    root_file = ROOT.TFile.Open(str(root_path), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {root_path}")
    tree = root_file.Get("limit")
    if not tree or tree.GetEntries() < 1:
        root_file.Close()
        raise RuntimeError(f"No limit tree entries in {root_path}")
    tree.GetEntry(tree.GetEntries() - 1)
    value = float(getattr(tree, "limit"))
    root_file.Close()
    return value


def task_hybrid(args: argparse.Namespace, card: Path) -> None:
    """Run the blinded HybridNew low-statistics cross-check.

    NPS-specific blinding prescription:
      1) make a *pre-fit* B-only Asimov dataset with GenerateOnly -t -1;
      2) pass that dataset to HybridNew as the observed dataset with -D.

    For this review check we deliberately do NOT combine that prescription with
    --expectedFromGrid in the adaptive command.  Instead we build a fixed grid of
    HybridNew singlePoint calculations around the already-known asymptotic median
    expected r95, merge the grid, and extract the CLs upper limit for the B-only
    Asimov dataset.  This is robust, remains fully blinded, and the expensive grid
    points can be run in parallel without using HybridNew's internal --fork mode.
    """
    require_command("combine")
    require_command("hadd")
    workspace = ensure_workspace(args, card)
    output = outdir(args)
    label = mass_label(args.mass)

    # ------------------------------------------------------------------
    # 1. Pre-fit B-only Asimov dataset required by the NPS blinding recipe.
    # ------------------------------------------------------------------
    asimov_tag = f"ReviewAsimov_{args.target}_M{label}"
    if args.force and not args.dry_run:
        for old in output.glob(f"higgsCombine.{asimov_tag}.GenerateOnly.mH*.root"):
            old.unlink()

    asimov_files = sorted(output.glob(f"higgsCombine.{asimov_tag}.GenerateOnly.mH*.root"))
    if not asimov_files or args.force:
        status = run_command(
            [
                "combine", "-M", "GenerateOnly", str(workspace),
                "-m", mass_for_combine(args.mass),
                "-t", "-1",
                "--expectSignal", "0",
                "--saveToys",
                "-n", f".{asimov_tag}",
            ],
            output,
            dry_run=args.dry_run,
            log=output / f"hybrid_asimov_M{label}.log",
        )
        if status != 0:
            raise RuntimeError("GenerateOnly failed for HybridNew Asimov data")
        if args.dry_run:
            asimov = output / f"higgsCombine.{asimov_tag}.GenerateOnly.mH{mass_for_combine(args.mass)}.123456.root"
        else:
            asimov = find_one(output, f"higgsCombine.{asimov_tag}.GenerateOnly.mH*.root")
    else:
        asimov = asimov_files[-1]
        print(f"[HYBRID] reuse Asimov file {asimov}")

    # ------------------------------------------------------------------
    # 2. Fixed-r toy grid around the existing asymptotic expected limit.
    #    This is the documented robust strategy for more complicated models.
    # ------------------------------------------------------------------
    asym_r95 = existing_expected_r95(args)
    if asym_r95 <= 0:
        raise RuntimeError(f"Existing expected median r95 must be positive, got {asym_r95}")

    npoints = args.hybrid_grid_points
    fmin = args.hybrid_grid_min_factor
    fmax = args.hybrid_grid_max_factor
    if npoints < 3:
        raise RuntimeError("--hybrid-grid-points must be at least 3")
    if not (0 < fmin < fmax):
        raise RuntimeError("Require 0 < --hybrid-grid-min-factor < --hybrid-grid-max-factor")

    factors = [fmin + (fmax - fmin) * i / (npoints - 1) for i in range(npoints)]
    r_points = [asym_r95 * f for f in factors]
    grid_rmax = max(args.r_max if args.r_max > 0 else 0.0, max(r_points) * 1.05)

    grid_dir = output / "hybrid_grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    if args.force and not args.dry_run:
        for old in grid_dir.glob("higgsCombine.ReviewHybridGrid_*.HybridNew.mH*.root"):
            old.unlink()

    print(
        f"[HYBRID GRID] asymptotic exp0 r95={asym_r95:.8g}; "
        f"points={npoints}; range=[{r_points[0]:.8g}, {r_points[-1]:.8g}]; "
        f"toys/point={args.hybrid_toys}; parallel={args.hybrid_parallel}"
    )

    def run_point(item: Tuple[int, float]) -> Tuple[int, float, int, Optional[Path]]:
        idx, rval = item
        tag = f"ReviewHybridGrid_{args.target}_M{label}_p{idx:03d}"
        # Deterministic but different seed for every point.
        seed = int(args.seed + idx + 1)
        cmd = [
            "combine", "-M", "HybridNew", str(workspace),
            "-m", mass_for_combine(args.mass),
            "--LHCmode", "LHC-limits",
            "--singlePoint", f"{rval:.12g}",
            "--saveHybridResult",
            "--saveToys",
            "--clsAcc", "0",
            "-T", str(args.hybrid_toys),
            "-D", f"{asimov}:toys/toy_asimov",
            "--rMin", "0",
            "--rMax", f"{grid_rmax:.12g}",
            "-s", str(seed),
            "-n", f".{tag}",
        ]
        status = run_command(
            cmd,
            grid_dir,
            dry_run=args.dry_run,
            log=grid_dir / f"point_{idx:03d}_r{rval:.6g}.log",
        )
        if args.dry_run:
            return idx, rval, status, None
        matches = sorted(grid_dir.glob(f"higgsCombine.{tag}.HybridNew.mH*.root"))
        root_path = matches[-1] if matches else None
        return idx, rval, status, root_path

    items = list(enumerate(r_points))
    results: List[Tuple[int, float, int, Optional[Path]]] = []
    if args.hybrid_parallel <= 1:
        for item in items:
            results.append(run_point(item))
    else:
        workers = min(args.hybrid_parallel, len(items))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(run_point, item): item for item in items}
            for fut in concurrent.futures.as_completed(future_map):
                results.append(fut.result())

    bad = [(i, r, status, path) for i, r, status, path in results if status != 0 or (not args.dry_run and path is None)]
    if bad:
        details = ", ".join(f"p{i}:r={r:.5g}:status={status}:file={path}" for i, r, status, path in bad)
        raise RuntimeError(f"One or more HybridNew grid points failed: {details}")
    if args.dry_run:
        return

    point_files = [path for _, _, _, path in sorted(results) if path is not None]
    if len(point_files) != npoints:
        raise RuntimeError(f"Expected {npoints} HybridNew point files, found {len(point_files)}")

    merged_grid = output / f"hybrid_grid_{args.target}_M{label}.root"
    status = run_command(
        ["hadd", "-f", str(merged_grid)] + [str(p) for p in point_files],
        output,
        dry_run=args.dry_run,
        log=output / f"hybrid_hadd_M{label}.log",
    )
    if status != 0:
        raise RuntimeError("hadd failed while merging HybridNew grid")

    # ------------------------------------------------------------------
    # 3. Extract the *Asimov-observed* CLs limit from the toy grid.
    #    No observed SR data are fitted, and no --expectedFromGrid is mixed in.
    # ------------------------------------------------------------------
    read_tag = f"ReviewHybridAsimovLimit_{args.target}_M{label}"
    cls_plot = output / f"hybrid_cls_scan_{args.target}_M{label}.png"
    read_cmd = [
        "combine", "-M", "HybridNew", str(workspace),
        "-m", mass_for_combine(args.mass),
        "--LHCmode", "LHC-limits",
        "--readHybridResults",
        f"--grid={merged_grid}",
        "--noUpdateGrid",
        "-D", f"{asimov}:toys/toy_asimov",
        "--rMin", "0",
        "--rMax", f"{grid_rmax:.12g}",
        f"--plot={cls_plot}",
        "-n", f".{read_tag}",
    ]
    status = run_command(
        read_cmd,
        output,
        dry_run=args.dry_run,
        log=output / f"hybrid_readgrid_M{label}.log",
    )
    if status != 0:
        raise RuntimeError(
            "HybridNew failed while extracting the Asimov limit from the fixed grid. "
            "Inspect hybrid_readgrid log and, if needed, widen the grid factors."
        )

    root_file = find_one(output, f"higgsCombine.{read_tag}.HybridNew.mH*.root")
    hybrid_limit = read_limit_tree_value(root_file)
    ratio = hybrid_limit / asym_r95 if asym_r95 > 0 else math.nan
    alpha_unit = parse_alpha_unit(card)
    summary = {
        "mass": args.mass,
        "target": args.target,
        "comparison_type": "HybridNew CLs limit on pre-fit B-only Asimov dataset vs AsymptoticLimits exp0",
        "hybrid_asimov_r95": hybrid_limit,
        "existing_asymptotic_expected_median_r95": asym_r95,
        "hybrid_over_asymptotic": ratio,
        "hybrid_toys_per_grid_point": args.hybrid_toys,
        "hybrid_grid_points": npoints,
        "hybrid_grid_r_min": r_points[0],
        "hybrid_grid_r_max": r_points[-1],
        "hybrid_parallel_jobs": min(args.hybrid_parallel, npoints),
        "alpha_internal_unit": alpha_unit,
        "hybrid_asimov_alpha95": hybrid_limit * alpha_unit if alpha_unit else None,
        "asymptotic_expected_median_alpha95": asym_r95 * alpha_unit if alpha_unit else None,
    }
    summary_path = output / f"hybrid_comparison_M{label}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("\n[HYBRID COMPARISON]")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[SAVED] {summary_path}")
    print(f"[SAVED] {cls_plot}")

def impact_injection(args: argparse.Namespace, card: Path) -> Tuple[float, Optional[float]]:
    r_inj = args.inject_r if args.inject_r is not None else existing_expected_r95(args)
    alpha_unit = parse_alpha_unit(card)
    alpha_inj = r_inj * alpha_unit if alpha_unit is not None else None
    return r_inj, alpha_inj


def task_sb_impacts(args: argparse.Namespace, card: Path) -> None:
    require_command("combineTool.py")
    workspace = ensure_workspace(args, card)
    output = outdir(args)
    label = mass_label(args.mass)
    r_inj, alpha_inj = impact_injection(args, card)
    tag = f"ReviewSBImpacts_{args.target}_M{label}"
    json_path = output / f"impacts_{args.target}_M{label}_SB.json"

    common = [
        "combineTool.py",
        "-M",
        "Impacts",
        "-d",
        str(workspace),
        "-m",
        mass_for_combine(args.mass),
        "--robustFit",
        "1",
        "--setParameters",
        f"r={r_inj:.12g}",
        "--rMin",
        "0",
        "--rMax",
        f"{args.r_max:.12g}",
        "-t",
        "-1",
        "--expectSignal",
        f"{r_inj:.12g}",
        "-n",
        f".{tag}",
    ]

    status = run_command(
        [*common, "--doInitialFit"],
        output,
        dry_run=args.dry_run,
        log=output / f"sb_impacts_initial_M{label}.log",
    )
    if status != 0:
        raise RuntimeError("S+B impact initial fit failed")

    fits = [*common, "--doFits"]
    if args.impact_parallel > 1:
        fits.extend(["--parallel", str(args.impact_parallel)])
    status = run_command(
        fits,
        output,
        dry_run=args.dry_run,
        log=output / f"sb_impacts_fits_M{label}.log",
    )
    if status != 0:
        raise RuntimeError("S+B impact nuisance fits failed")

    collect = [
        "combineTool.py",
        "-M",
        "Impacts",
        "-d",
        str(workspace),
        "-m",
        mass_for_combine(args.mass),
        "-n",
        f".{tag}",
        "-o",
        json_path.name,
    ]
    status = run_command(
        collect,
        output,
        dry_run=args.dry_run,
        log=output / f"sb_impacts_collect_M{label}.log",
    )
    if status != 0:
        raise RuntimeError("S+B impact JSON collection failed")

    if args.dry_run:
        return
    plotter = shutil.which("plotImpacts.py")
    if plotter:
        status = run_command(
            [plotter, "-i", json_path.name, "-o", f"impacts_{args.target}_M{label}_SB"],
            output,
            log=output / f"sb_impacts_plot_M{label}.log",
        )
        if status != 0:
            print("[WARNING] plotImpacts.py failed; JSON is retained.", file=sys.stderr)

    metadata = {
        "mass": args.mass,
        "target": args.target,
        "injected_r": r_inj,
        "alpha_internal_unit": parse_alpha_unit(card),
        "injected_alpha": alpha_inj,
        "source_for_default_injection": str(expected_limit_json(args)) if args.inject_r is None else "--inject-r",
    }
    meta_path = output / f"sb_impacts_metadata_M{label}.json"
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print("\n[S+B IMPACT INJECTION]")
    print(json.dumps(metadata, indent=2, sort_keys=True))


def best_fit_values_from_multidimfit(root_path: Path, injected_r: float) -> Tuple[List[float], List[float]]:
    """Return best-fit r values and pulls when an uncertainty branch is available."""
    try:
        import ROOT  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"PyROOT is required to inspect bias output: {exc}")

    root_file = ROOT.TFile.Open(str(root_path), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Could not open ROOT file: {root_path}")
    tree = root_file.Get("limit")
    if not tree:
        root_file.Close()
        raise RuntimeError(f"No limit tree in {root_path}")

    branch_names = {str(b.GetName()) for b in tree.GetListOfBranches()}
    r_values: List[float] = []
    pulls: List[float] = []
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        if "quantileExpected" in branch_names:
            quant = float(getattr(tree, "quantileExpected"))
            # Best-fit entry convention used by Combine.
            if quant >= 0.0:
                continue
        rhat = float(getattr(tree, "r"))
        if not math.isfinite(rhat):
            continue
        r_values.append(rhat)

        sigma: Optional[float] = None
        if "rErr" in branch_names:
            candidate = abs(float(getattr(tree, "rErr")))
            if candidate > 0 and math.isfinite(candidate):
                sigma = candidate
        elif "rLoErr" in branch_names and "rHiErr" in branch_names:
            lo = abs(float(getattr(tree, "rLoErr")))
            hi = abs(float(getattr(tree, "rHiErr")))
            candidate = 0.5 * (lo + hi)
            if candidate > 0 and math.isfinite(candidate):
                sigma = candidate
        if sigma is not None:
            pulls.append((rhat - injected_r) / sigma)

    root_file.Close()
    return r_values, pulls


def task_bias(args: argparse.Namespace, card: Path) -> None:
    require_command("combine")
    workspace = ensure_workspace(args, card)
    output = outdir(args) / "bias"
    output.mkdir(parents=True, exist_ok=True)
    label = mass_label(args.mass)
    expected = existing_expected_r95(args)
    alpha_unit = parse_alpha_unit(card)
    rmin = safe_negative_rmin(card, args.r_min)

    all_summaries: List[Dict[str, object]] = []
    for multiplier in args.bias_multipliers:
        r_inj = multiplier * expected
        if args.bias_injections is not None:
            # Explicit injections replace multiplier mode entirely.
            continue
        all_summaries.append(
            run_one_bias_point(args, workspace, output, label, rmin, r_inj, multiplier, alpha_unit)
        )

    if args.bias_injections is not None:
        all_summaries = []
        for r_inj in args.bias_injections:
            all_summaries.append(
                run_one_bias_point(args, workspace, output, label, rmin, r_inj, None, alpha_unit)
            )

    if args.dry_run:
        return
    summary_path = output / f"bias_summary_{args.target}_M{label}.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2, sort_keys=True) + "\n")
    print("\n[BIAS SUMMARY]")
    print(json.dumps(all_summaries, indent=2, sort_keys=True))
    print(f"[SAVED] {summary_path}")



def summarize_bias_fit(
    fit_file: Path,
    args: argparse.Namespace,
    r_inj: float,
    multiplier: Optional[float],
    alpha_unit: Optional[float],
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "injected_r": r_inj,
        "multiplier_of_existing_expected_r95": multiplier,
        "injected_alpha": r_inj * alpha_unit if alpha_unit else None,
        "n_toys_requested": args.bias_toys,
        "fit_file": str(fit_file),
    }
    r_values, pulls = best_fit_values_from_multidimfit(fit_file, r_inj)
    summary["n_best_fits_read"] = len(r_values)
    if r_values:
        mean_r = statistics.fmean(r_values)
        summary["mean_fitted_r"] = mean_r
        summary["mean_bias_r"] = mean_r - r_inj
        summary["std_fitted_r"] = statistics.stdev(r_values) if len(r_values) > 1 else 0.0
    if pulls:
        summary["n_pulls"] = len(pulls)
        summary["mean_pull"] = statistics.fmean(pulls)
        summary["pull_width"] = statistics.stdev(pulls) if len(pulls) > 1 else 0.0
    return summary

def run_one_bias_point(
    args: argparse.Namespace,
    workspace: Path,
    output: Path,
    label: str,
    rmin: float,
    r_inj: float,
    multiplier: Optional[float],
    alpha_unit: Optional[float],
) -> Dict[str, object]:
    safe_tag = f"{r_inj:.6g}".replace("-", "m").replace(".", "p")
    gen_tag = f"ReviewBiasGen_{args.target}_M{label}_r{safe_tag}"
    fit_tag = f"ReviewBiasFit_{args.target}_M{label}_r{safe_tag}"

    # Resume support: a previous review run may already have completed this
    # injection before a later injection failed.  Reuse the completed MultiDimFit
    # result unless --force was requested.
    existing_fit_files = sorted(output.glob(f"higgsCombine.{fit_tag}.MultiDimFit.mH*.root"))
    if existing_fit_files and not args.force and not args.dry_run:
        fit_file = existing_fit_files[-1]
        print(f"[BIAS] reuse completed fit for r={r_inj:g}: {fit_file}")
        return summarize_bias_fit(fit_file, args, r_inj, multiplier, alpha_unit)

    # GenerateOnly uses the POI range stored in the workspace unless we explicitly
    # enlarge it.  This matters for high injections such as 2 x the M20 expected
    # limit (r=36.25), while text2workspace.py leaves the default r range at [0,20].
    # Toy generation itself is physical, so keep the lower generation bound at 0.
    fit_rmax = max(args.r_max, r_inj * 2.0 + 5.0)
    gen_rmax = max(20.0, fit_rmax)

    existing_gen_files = sorted(output.glob(f"higgsCombine.{gen_tag}.GenerateOnly.mH*.root"))
    if existing_gen_files and not args.force and not args.dry_run:
        toy_file = existing_gen_files[-1]
        print(f"[BIAS] reuse generated toys for r={r_inj:g}: {toy_file}")
    else:
        status = run_command(
            [
                "combine",
                "-M",
                "GenerateOnly",
                str(workspace),
                "-m",
                mass_for_combine(args.mass),
                "-t",
                str(args.bias_toys),
                "--expectSignal",
                f"{r_inj:.12g}",
                "--setParameters",
                f"r={r_inj:.12g}",
                "--setParameterRanges",
                f"r=0,{gen_rmax:.12g}",
                "--saveToys",
                "-s",
                str(args.seed),
                "-n",
                f".{gen_tag}",
            ],
            output,
            dry_run=args.dry_run,
            log=output / f"bias_generate_r{safe_tag}.log",
        )
        if status != 0:
            raise RuntimeError(f"Bias toy generation failed for r={r_inj:g}")
        if args.dry_run:
            toy_file = output / f"higgsCombine.{gen_tag}.GenerateOnly.mH{mass_for_combine(args.mass)}.{args.seed}.root"
        else:
            toy_file = find_one(output, f"higgsCombine.{gen_tag}.GenerateOnly.mH*.root")

    status = run_command(
        [
            "combine",
            "-M",
            "MultiDimFit",
            str(workspace),
            "-m",
            mass_for_combine(args.mass),
            "-t",
            str(args.bias_toys),
            "--toysFile",
            str(toy_file),
            "--algo",
            "singles",
            "--setParameters",
            f"r={r_inj:.12g}",
            "--rMin",
            f"{rmin:.12g}",
            "--rMax",
            f"{fit_rmax:.12g}",
            "--setParameterRanges",
            f"r={rmin:.12g},{fit_rmax:.12g}",
            "-n",
            f".{fit_tag}",
        ],
        output,
        dry_run=args.dry_run,
        log=output / f"bias_fit_r{safe_tag}.log",
    )
    if status != 0:
        raise RuntimeError(f"Bias toy fitting failed for r={r_inj:g}")

    if args.dry_run:
        return {
            "injected_r": r_inj,
            "multiplier_of_existing_expected_r95": multiplier,
            "injected_alpha": r_inj * alpha_unit if alpha_unit else None,
            "n_toys_requested": args.bias_toys,
        }

    fit_file = find_one(output, f"higgsCombine.{fit_tag}.MultiDimFit.mH*.root")
    return summarize_bias_fit(fit_file, args, r_inj, multiplier, alpha_unit)


def task_correlation(args: argparse.Namespace, card: Path) -> None:
    require_command("combine")
    workspace = ensure_workspace(args, card)
    output = outdir(args)
    label = mass_label(args.mass)
    rmin = safe_negative_rmin(card, args.r_min)
    tag = f"ReviewCorrelation_{args.target}_M{label}"

    status = run_command(
        [
            "combine",
            "-M",
            "FitDiagnostics",
            str(workspace),
            "-m",
            mass_for_combine(args.mass),
            "-t",
            "-1",
            "--expectSignal",
            "0",
            "--setParameters",
            "r=0",
            "--rMin",
            f"{rmin:.12g}",
            "--rMax",
            f"{args.r_max:.12g}",
            "--saveWithUncertainties",
            "--saveNormalizations",
            "-n",
            f".{tag}",
        ],
        output,
        dry_run=args.dry_run,
        log=output / f"correlation_fit_M{label}.log",
    )
    if status != 0:
        raise RuntimeError("FitDiagnostics failed for correlation matrix")
    if args.dry_run:
        return

    fit_file = output / f"fitDiagnostics.{tag}.root"
    if not fit_file.is_file():
        fit_file = find_one(output, f"fitDiagnostics*{tag}*.root")

    try:
        import ROOT  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"PyROOT is required for correlation matrix: {exc}")
    ROOT.gROOT.SetBatch(True)
    f = ROOT.TFile.Open(str(fit_file), "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Could not open {fit_file}")
    fit = f.Get("fit_s")
    if not fit:
        f.Close()
        raise RuntimeError(f"fit_s RooFitResult not found in {fit_file}")
    hist = fit.correlationHist()
    if not hist:
        f.Close()
        raise RuntimeError("RooFitResult::correlationHist() returned null")
    hist.SetName(f"correlation_{args.target}_M{label}")
    canvas = ROOT.TCanvas("c_correlation", "", 1800, 1600)
    canvas.SetRightMargin(0.16)
    canvas.SetLeftMargin(0.18)
    canvas.SetBottomMargin(0.18)
    hist.SetStats(0)
    hist.Draw("COLZ")
    pdf = output / f"correlation_{args.target}_M{label}.pdf"
    root_out = output / f"correlation_{args.target}_M{label}.root"
    canvas.SaveAs(str(pdf))
    out = ROOT.TFile.Open(str(root_out), "RECREATE")
    hist.Write()
    out.Close()
    f.Close()
    print(f"[SAVED] {pdf}")
    print(f"[SAVED] {root_out}")


def write_metadata(args: argparse.Namespace, card: Path) -> None:
    output = outdir(args)
    output.mkdir(parents=True, exist_ok=True)
    alpha_unit = parse_alpha_unit(card)
    metadata = {
        "mass": args.mass,
        "target": args.target,
        "card": str(card),
        "tasks": args.tasks,
        "alpha_internal_unit": alpha_unit,
        "cmssw_base": os.environ.get("CMSSW_BASE"),
        "scram_arch": os.environ.get("SCRAM_ARCH"),
    }
    try:
        metadata["existing_expected_median_r95"] = existing_expected_r95(args)
    except Exception:
        metadata["existing_expected_median_r95"] = None
    if metadata["existing_expected_median_r95"] is not None and alpha_unit is not None:
        metadata["existing_expected_median_alpha95"] = (
            float(metadata["existing_expected_median_r95"]) * alpha_unit
        )
    if not args.dry_run:
        (output / "review_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
    print("[REVIEW CONFIG]")
    print(json.dumps(metadata, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="combine_review.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("mass", type=float, help="signal mass hypothesis in GeV, e.g. 20 or 70")
    parser.add_argument(
        "--task",
        type=parse_tasks,
        default=["validate"],
        help=(
            "comma-separated tasks: validate,workspace,scan,hybrid,sb-impacts,bias,correlation,all; "
            "default: validate"
        ),
    )
    parser.add_argument(
        "--target",
        choices=("Run2", "Run3", "Run2Run3"),
        default="Run2Run3",
        help="datacard target; default: %(default)s",
    )
    parser.add_argument(
        "--card-base",
        default="datacards_NIsoMuon/alpha/blind/combined",
        help="base directory containing Run2/Run3/Run2Run3 card directories",
    )
    parser.add_argument(
        "--output-base",
        default="review_outputs",
        help="review-only output base; default: %(default)s",
    )
    parser.add_argument(
        "--limit-output-base",
        default="limit_outputs/alpha/blind",
        help="base containing existing internal expected-limit JSON files",
    )
    parser.add_argument(
        "--expected-json",
        default=None,
        help="optional explicit limits_<TARGET>_alpha_internal.json path",
    )
    parser.add_argument("--print-level", type=int, choices=(0, 1, 2, 3), default=3)
    parser.add_argument("--r-min", type=float, default=-2.0, help="diagnostic lower r request; clipped to positive-yield safety")
    parser.add_argument("--r-max", type=float, default=100.0)
    parser.add_argument("--scan-points", type=int, default=120)
    parser.add_argument(
        "--scan-parallel", type=int, default=1,
        help="local processes for MultiDimFit grid via Combine parallelScan.py; default: 1",
    )
    parser.add_argument(
        "--hybrid-toys", type=int, default=500,
        help="HybridNew toys per fixed-r grid point (default: 500)",
    )
    parser.add_argument(
        "--hybrid-parallel", type=int, default=12,
        help="maximum simultaneous fixed-r HybridNew grid jobs (default: 12)",
    )
    parser.add_argument(
        "--hybrid-fork", type=int, default=None,
        help="deprecated compatibility alias for --hybrid-parallel; NOT passed to Combine --fork",
    )
    parser.add_argument("--hybrid-grid-points", type=int, default=29)
    parser.add_argument("--hybrid-grid-min-factor", type=float, default=0.20)
    parser.add_argument("--hybrid-grid-max-factor", type=float, default=3.00)
    parser.add_argument("--impact-parallel", type=int, default=12)
    parser.add_argument(
        "--inject-r",
        type=float,
        default=None,
        help="S+B impact injected internal r; default is existing expected median r95",
    )
    parser.add_argument("--bias-toys", type=int, default=200)
    parser.add_argument(
        "--bias-multipliers",
        type=parse_csv_floats,
        default=[0.0, 1.0, 2.0],
        help="multiples of existing expected median r95; default: 0,1,2",
    )
    parser.add_argument(
        "--bias-injections",
        type=parse_csv_floats,
        default=None,
        help="explicit internal-r bias injections; overrides --bias-multipliers",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--force", action="store_true", help="regenerate workspace/intermediate review products")
    parser.add_argument("--dry-run", action="store_true", help="print commands without executing them")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.tasks = args.task
    delattr(args, "task")

    try:
        if "CMSSW_BASE" not in os.environ:
            raise RuntimeError("CMSSW_BASE is not set. cd to the CMSSW src area and run cmsenv first.")
        if args.scan_points < 5:
            raise RuntimeError("--scan-points must be at least 5")
        if args.scan_parallel < 1:
            raise RuntimeError("--scan-parallel must be at least 1")
        if args.hybrid_fork is not None:
            if args.hybrid_fork < 1:
                raise RuntimeError("--hybrid-fork compatibility alias must be >= 1")
            # Backward-compatible with the review.sh already in use.  The value
            # now controls independent fixed-r jobs; it is NOT forwarded to
            # HybridNew's internal --fork option.
            args.hybrid_parallel = args.hybrid_fork
        if args.hybrid_parallel < 1:
            raise RuntimeError("--hybrid-parallel must be at least 1")
        if args.hybrid_toys < 1 or args.bias_toys < 1:
            raise RuntimeError("Toy counts must be positive")
        if args.r_max <= 0:
            raise RuntimeError("--r-max must be positive")

        card = card_path(args)
        write_metadata(args, card)
        output = outdir(args)
        output.mkdir(parents=True, exist_ok=True)

        dispatch = {
            "validate": lambda: task_validate(args, card),
            "workspace": lambda: ensure_workspace(args, card),
            "scan": lambda: task_scan(args, card),
            "hybrid": lambda: task_hybrid(args, card),
            "sb-impacts": lambda: task_sb_impacts(args, card),
            "bias": lambda: task_bias(args, card),
            "correlation": lambda: task_correlation(args, card),
        }
        for task in args.tasks:
            print("\n" + "=" * 88)
            print(f"[TASK] {task} | target={args.target} | M={args.mass:g}")
            print("=" * 88)
            dispatch[task]()

        print(f"\n[DONE] review outputs: {output}")
        return 0
    except (RuntimeError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

