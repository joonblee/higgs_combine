#!/usr/bin/env python3
"""Plot the Run 2+3 signal-window interpolation and leave-one-out MC closure.

Run python3 plot_signal_interpolation.py from NIsoMuon or pass --models (the
directory containing M12/model.json etc.) and --scan (expected_limits.csv).
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

from limit_workflow import YEARS, YieldResult, interpolate_signal_yield, signal_bracket

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


def combined(anchors, mass, left, right):
    yields = [interpolate_signal_yield(mass, left, anchors[left][era], right, anchors[right][era])
              for era in YEARS]
    return sum(y.value for y in yields), math.sqrt(sum(y.error**2 for y in yields))


def plot(anchors, grid, alpha, output):
    masses = sorted(anchors)
    native = [combined(anchors, m, m, m) for m in masses]
    scan = [combined(anchors, m, *signal_bracket(masses, m))[0] * alpha for m in grid]
    ratios, errors = [], []
    for i in range(1, len(masses)-1):
        pred, pred_err = combined(anchors, masses[i], masses[i-1], masses[i+1])
        truth, truth_err = native[i]
        ratio = pred / truth
        ratios.append(ratio)
        errors.append(ratio * math.hypot(pred_err / pred, truth_err / truth))
        print(f"{masses[i]:g} GeV: predicted / simulated = {ratio:.4f} +/- {errors[-1]:.4f} (MC stat.)")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.8), sharex=True,
                                     gridspec_kw={"height_ratios": [2.4, 1]})
    fig.subplots_adjust(left=.15, right=.97, bottom=.12, top=.88, hspace=.07)
    fig.text(.15, .945, "CMS", weight="bold", fontsize=18)
    fig.text(.25, .945, "Simulation Preliminary", style="italic", fontsize=12)
    fig.text(.97, .945, r"138 fb$^{-1}$ (13 TeV) + 62 fb$^{-1}$ (13.6 TeV)", ha="right", fontsize=11)
    top.plot(grid, scan, color="#2475a8", label="Interpolated signal yield")
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
    top.set_title(rf"Run 2+3, OS, b-jet     $\alpha_{{qZ^\prime}}={alpha_label}$", loc="left", fontsize=11)
    bottom.axhline(1, color="0.5", lw=1)
    bottom.errorbar(masses[1:-1], ratios, yerr=errors, fmt="o", color="#2475a8", ms=4, capsize=3)
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
    parser.add_argument("--output", type=Path, default=Path("signal_window_interpolation.pdf"))
    args = parser.parse_args()
    if not math.isfinite(args.alpha) or args.alpha <= 0:
        parser.error("--alpha must be finite and positive")
    anchors = anchors_from_models(args.models)
    with args.scan.open() as stream:
        grid = sorted({float(r["mass"]) for r in csv.DictReader(stream) if r["target"] == "Run2Run3"})
    if not anchors.keys() <= set(grid):
        parser.error("Run2Run3 scan must include every generated anchor")
    plot(anchors, grid, args.alpha, args.output)


if __name__ == "__main__":
    main()
