#!/usr/bin/env python3
"""Plot the signal-window interpolation and leave-one-out MC closure by era or run.

Run python3 plot_signal_interpolation.py from NIsoMuon or pass --models (the
directory containing M12/model.json etc.) and --scan (expected_limits.csv).
The default compares the production log-linear interpolation with log-yield PCHIP
(requires SciPy). Use --method log-linear to draw the production method alone.
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


def pchip_yield(anchors, masses, eras, evaluate_at, with_error=False):
    """Interpolate log yields in each era, then sum. MC inputs are independent."""
    from scipy.interpolate import PchipInterpolator

    samples = [[anchors[m][era] for era in eras] for m in masses]
    logs = np.log([[y.value for y in row] for row in samples])

    def evaluate(log_yields):
        curve = PchipInterpolator(masses, log_yields, axis=0, extrapolate=False)
        return np.exp(curve(evaluate_at))

    prediction = evaluate(logs)
    total = prediction.sum(axis=-1)
    if not with_error:
        return total

    # First-order propagation through the cubic, including its anchor-dependent
    # slopes. Columns are independent eras, so their gradients can be evaluated together.
    variance = np.zeros_like(prediction)
    step = 1e-5
    for i, row in enumerate(samples):
        shift = np.zeros_like(logs)
        shift[i] = step
        derivative = (evaluate(logs + shift) - evaluate(logs - shift)) / (2 * step)
        relative_errors = np.array([y.error / y.value for y in row])
        variance += (derivative * relative_errors)**2
    return total, np.sqrt(variance.sum(axis=-1))


def plot(anchors, grid, alpha, output, target="Run2Run3", method="both"):
    eras = target_years(target)
    label = {"Run2": "Run 2", "Run3": "Run 3", "Run2Run3": "Run 2+3"}.get(target, target)
    print(label)
    masses = sorted(anchors)
    native = [combined(anchors, m, m, m, eras) for m in masses]
    styles = {
        "log-linear": ("Log-linear (nominal)", "#2475a8", "-", "o"),
        "pchip": ("PCHIP (log yield)", "#bd581b", "--", "s"),
    }
    methods = list(styles) if method == "both" else [method]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.8), sharex=True,
                                     gridspec_kw={"height_ratios": [2.4, 1]})
    fig.subplots_adjust(left=.15, right=.97, bottom=.12, top=.88, hspace=.07)
    fig.text(.15, .945, "CMS", weight="bold", fontsize=18)
    fig.text(.25, .945, "Simulation Preliminary", style="italic", fontsize=12)
    fig.text(.97, .945, lumi_label(target).replace("^{-1}", "$^{-1}$"), ha="right", fontsize=11)
    for name in methods:
        legend, colour, line, marker = styles[name]
        if name == "pchip":
            scan = pchip_yield(anchors, masses, eras, grid)
        else:
            scan = np.array([combined(anchors, m, *signal_bracket(masses, m), eras=eras)[0] for m in grid])
        top.plot(grid, alpha * scan, color=colour, linestyle=line, label=legend)
        ratios, errors = [], []
        for i, mass in enumerate(masses[1:-1], start=1):
            if name == "pchip":
                retained = masses[:i] + masses[i+1:]
                pred, pred_err = pchip_yield(anchors, retained, eras, mass, with_error=True)
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
                        markerfacecolor="white" if name == "pchip" else colour, ms=4, capsize=2)
    top.errorbar(masses, [y*alpha for y, _ in native], yerr=[e*alpha for _, e in native],
                 fmt="o", color="black", ms=4, capsize=2, label="Generated signal samples")
    top.set_yscale("log")
    top.set_ylabel("Signal-window yield [events]")
    top.legend(frameon=False, loc="upper right")
    alpha_label = f"{alpha:g}"
    if "e" in alpha_label:
        mantissa, exponent = alpha_label.split("e")
        power = rf"10^{{{int(exponent)}}}"
        alpha_label = power if mantissa == "1" else rf"{mantissa} \times {power}"
    top.set_title(rf"{label}, OS, b-jet     $\alpha_{{qZ^\prime}}={alpha_label}$", loc="left", fontsize=11)
    bottom.axhline(1, color="0.5", lw=1)
    bottom.set_ylabel("Leave-one-out\npredicted / simulated")
    bottom.set_xlabel(r"Signal mass [GeV]", loc="right")
    for ax in (top, bottom):
        ax.tick_params(which="both", direction="in", top=True, right=True)
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
    parser.add_argument("--method", choices=("both", "log-linear", "pchip"), default="both",
                        help="Interpolation methods to plot (default: both)")
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
        plot(anchors, grid, args.alpha, output, target, args.method)


if __name__ == "__main__":
    main()
