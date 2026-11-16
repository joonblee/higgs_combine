#!/usr/bin/env python3
"""Plot the signal-window interpolation and leave-one-out MC closure by era or run.

Run python3 plot_signal_interpolation.py from NIsoMuon or pass --models (the
directory containing M12/model.json etc.) and --scan (expected_limits.csv).
Both splines interpolate log yield versus mass. The linear spline is used in
production. The cubic cross-check uses SciPy with not-a-knot end conditions.
No fit or input ROOT histograms are required. The signal MC uncertainties are
recovered at the precision of the stored sig_stat up factors (1 + Sumw2 error/yield).
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from limit_workflow import (
    YEARS, YieldResult, canonical_target, interpolate_signal_yield, lumi_label,
    selected_targets, signal_bracket, target_years,
)

HERE = Path(__file__).resolve().parent
REVIEW = HERE / "review_results"
DEFAULT_MODELS = (REVIEW / "20261114_130732/condor/dense_central_limits_r0/results" /
                  "Run2Run3_window5sigma_sbbb1f9e8e59_lnN1p5_signalSR25_v1_DYlinearShape_v1_signalCounting_v2")
CENTRAL = REVIEW / "20261115_central"


def anchors_from_models(directory):
    masses = json.loads((HERE / "mass_scan_grid.json").read_text())["generation"]["include_simulated_masses"]
    anchors = {}
    for mass in masses:
        path = directory / f"M{mass}/model.json"
        model = json.loads(path.read_text())
        if (model["mass"] != mass or model["signal_interpolation"] or model["target"] != "Run2Run3"
                or model["signal_normalisation_reference"] != "SR"):
            raise ValueError(f"Expected a generated Run 2+3 signal-window model in {path}")
        unit = model["alpha_internal_unit"]
        rows = [r for r in model["rows"] if r["region"] == "SR"]
        if len(rows) != len(YEARS) or {r["era"] for r in rows} != set(YEARS) or not unit > 0:
            raise ValueError(f"Invalid era coverage or signal normalisation in {path}")
        counts = {}
        for row in rows:
            value = row["rates"]["sig"] / unit
            factor = row["nuisances"]["sig_stat"]["sig"]
            if "/" not in factor:
                raise ValueError(f"Missing signal MC error in {path}, {row['era']}")
            relative_error = float(factor.split("/")[1]) - 1
            if not (math.isfinite(value) and value > 0 and math.isfinite(relative_error) and relative_error >= 0):
                raise ValueError(f"Invalid signal yield or MC error in {path}, {row['era']}")
            counts[row["era"]] = YieldResult(value, value * relative_error, 0, 0)
        if not math.isclose(sum(y.value for y in counts.values()), model["raw_reference_signal_yield"], rel_tol=1e-9):
            raise ValueError(f"Signal-window sum disagrees with recorded production yield in {path}")
        anchors[mass] = counts
    return anchors


def combined(anchors, mass, left, right, eras=YEARS):
    yields = [interpolate_signal_yield(mass, left, anchors[left][era], right, anchors[right][era])
              for era in eras]
    return sum(y.value for y in yields), math.sqrt(sum(y.error**2 for y in yields))


def cubic_yield(anchors, masses, eras, evaluate_at, with_error=False):
    """Interpolate log yields in each era, then sum; no mass extrapolation."""
    from scipy.interpolate import CubicSpline

    samples = [[anchors[m][era] for era in eras] for m in masses]
    logs = np.log([[y.value for y in row] for row in samples])
    # With fixed knots and not-a-knot ends, the spline is linear in the
    # anchor log yields. Interpolating unit vectors gives their exact weights.
    basis = CubicSpline(masses, np.eye(len(masses)), bc_type="not-a-knot", extrapolate=False)
    weights = basis(evaluate_at)
    prediction = np.exp(weights @ logs)
    total = prediction.sum(axis=-1)
    if not with_error:
        return total

    relative_errors = np.array([[y.error / y.value for y in row] for row in samples])
    log_variance = weights**2 @ (relative_errors**2)
    return total, np.sqrt((prediction**2 * log_variance).sum(axis=-1))


def plot(anchors, grid, alpha, output, target="Run2Run3", method="both"):
    eras = target_years(target)
    label = {"Run2": "Run 2", "Run3": "Run 3", "Run2Run3": "Run 2+3"}.get(target, target)
    print(label)
    masses = sorted(anchors)
    native = [combined(anchors, m, m, m, eras) for m in masses]
    styles = {
        "linear": ("Linear spline (nominal)", "#2475a8", "-", "o"),
        "cubic": ("Cubic spline", "#bd581b", "--", "s"),
    }
    methods = list(styles) if method == "both" else [method]
    # Match the relative text sizes of sigFit.py's 850 x 700 resolution
    # canvas (AN Fig. 32), including the shorter closure panel.
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.5, 7), sharex=True,
                                     gridspec_kw={"height_ratios": [2.1, 1]})
    height_pt = fig.get_figheight() * 72
    axis_size = .050 * height_pt
    tick_size = .043 * height_pt
    note_size = .035 * height_pt
    panel_top = .80 if target == "Run2Run3" else .86
    fig.subplots_adjust(left=.22, right=.97, bottom=.16, top=panel_top, hspace=.09)
    fig.text(.05, .94, r"$\mathbf{CMS}$ $\it{Simulation\ Preliminary}$", fontsize=tick_size)
    lumi_y = .868 if target == "Run2Run3" else .94
    fig.text(.97, lumi_y, lumi_label(target).replace("^{-1}", "$^{-1}$"),
             ha="right", fontsize=.037 * height_pt)
    for name in methods:
        legend, colour, line, marker = styles[name]
        if name == "cubic":
            scan = cubic_yield(anchors, masses, eras, grid)
        else:
            scan = np.array([combined(anchors, m, *signal_bracket(masses, m), eras=eras)[0] for m in grid])
        top.plot(grid, alpha * scan, color=colour, linestyle=line, label=legend)
        ratios, errors = [], []
        for i, mass in enumerate(masses[1:-1], start=1):
            if name == "cubic":
                retained = masses[:i] + masses[i+1:]
                pred, pred_err = cubic_yield(anchors, retained, eras, mass, with_error=True)
            else:
                pred, pred_err = combined(anchors, mass, masses[i-1], masses[i+1], eras)
            truth, truth_err = native[i]
            ratio = pred / truth
            error = ratio * math.hypot(pred_err / pred, truth_err / truth)
            ratios.append(ratio)
            errors.append(error)
            print(f"{name} {mass:g} GeV: predicted / simulated = {ratio:.4f} +/- {error:.4f} (MC stat.)")
        residual = 100 * (np.array(ratios) - 1)
        worst = int(np.argmax(abs(residual)))
        print(f"{name}: RMS = {np.sqrt(np.mean(residual**2)):.2f}%, "
              f"largest residual = {residual[worst]:+.2f}% at {masses[1+worst]:g} GeV")
        bottom.errorbar(masses[1:-1], ratios, yerr=errors, fmt=marker, color=colour,
                        markerfacecolor="white" if name == "cubic" else colour, ms=4, capsize=2)
    top.errorbar(masses, [y*alpha for y, _ in native], yerr=[e*alpha for _, e in native],
                 fmt="o", color="black", ms=4, capsize=2, label="Signal MC")
    top.set_yscale("log")
    top.set_ylabel("Signal yield\n[events]", fontsize=axis_size)
    top.legend(frameon=False, loc="upper right", fontsize=note_size, labelspacing=.16, borderaxespad=.25)
    alpha_label = f"{alpha:g}"
    if "e" in alpha_label:
        mantissa, exponent = alpha_label.split("e")
        power = rf"10^{{{int(exponent)}}}"
        alpha_label = power if mantissa == "1" else rf"{mantissa} \times {power}"
    heading = rf"{label}, OS, b-jet     $\alpha_{{qZ^\prime}}={alpha_label}$"
    top.set_title(heading, loc="left", fontsize=note_size, pad=3)
    bottom.axhline(1, color="0.5", lw=1)
    bottom.set_ylabel("Predicted\n/ MC", fontsize=axis_size)
    bottom.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
    bottom.set_xlabel("Omitted signal mass [GeV]", loc="right", fontsize=axis_size)
    for ax in (top, bottom):
        ax.tick_params(which="both", direction="in", top=True, right=True, labelsize=tick_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--scan", type=Path, default=CENTRAL / "blind_limits/expected_limits.csv")
    parser.add_argument("--alpha", type=float, default=1e-5, help="Fixed signal strength for the yield panel")
    parser.add_argument("--target", type=canonical_target, default="full",
                        help="One era, Run2, Run3, Run2+3 (default), eras, runs, or all")
    parser.add_argument("--method", choices=("both", "linear", "log-linear", "cubic"), default="both",
                        help="Compare both splines or draw one method (log-linear is an alias for linear)")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output", type=Path, help="Output file for a single target")
    outputs.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for named plots")
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be finite and positive")
    targets = selected_targets(args.target)
    if args.output and len(targets) != 1:
        parser.error("Use --output-dir when plotting multiple targets")
    anchors = anchors_from_models(args.models)
    with args.scan.open() as stream:
        grid = sorted({float(r["mass"]) for r in csv.DictReader(stream) if r["target"] == "Run2Run3"})
    if not anchors.keys() <= set(grid):
        parser.error("Run2Run3 scan must include every generated anchor")
    for target in targets:
        suffix = "" if target == "Run2Run3" else f"_{target}"
        output = args.output or args.output_dir / f"signal_window_interpolation{suffix}.pdf"
        plot(anchors, grid, args.alpha, output, target, "linear" if args.method == "log-linear" else args.method)


if __name__ == "__main__":
    main()
