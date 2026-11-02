#!/usr/bin/env python3
"""Blind counting-channel pre/post-fit yields from a fresh Asimov FitDiagnostics.

Run after cmsenv, from NIsoMuon. Never reads SR data histograms. Uses the
existing review fit, nominal rateParam evaluation and signed-r safeguard.
Uncertainty bands come from Combine's correlated total-background response,
not a quadrature sum of process errors. Requires PyROOT and matplotlib.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import combine_review as review


STAGES = ("prefit", "fit_b", "fit_s")
LABELS = ("Prefit", "Postfit B-only (r fixed to 0)", "Postfit S+B (r floating)")


def checked(value, label, nonnegative=False):
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise RuntimeError(f"Invalid {label}: {value}")
    return value


def require_close(value, reference, label):
    if not math.isclose(value, reference, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(f"{label}: {value} != {reference}")


def nominal_backgrounds(card):
    bins, names, ids, rates = review.parse_rate_block(card)
    channels = {}
    for channel, process, pid, rate in zip(bins, names, ids, rates):
        slot = channels.setdefault(channel, {})
        if pid > 0:
            if process in slot:
                raise RuntimeError(f"Duplicate process {channel}/{process}")
            slot[process] = checked(rate, f"nominal {channel}/{process}", True)
    # This tool is deliberately restricted to the production blind cards.
    if "# observation = unrounded prefit background-only expectation" not in card.read_text():
        raise RuntimeError("Expected a production blind card with fractional Asimov observations")
    return channels


def saved_value(source, stage, channel, process):
    norms = source.Get(f"norm_{stage}")
    var = norms.find(f"{channel}/{process}") if norms else None
    if var:
        return {"yield": checked(var.getVal(), f"{stage}/{channel}/{process}", True),
                "error": checked(var.getError(), f"error {stage}/{channel}/{process}", True)}
    # Older Combine versions may only save totals in the shape directory.
    # A counting channel has exactly one bin; no covariance is lost here.
    hist = source.Get(f"shapes_{stage}/{channel}/{process}")
    if hist and hist.InheritsFrom("TH1") and hist.GetNbinsX() == 1:
        return {"yield": checked(hist.GetBinContent(1), f"{stage}/{channel}/{process}", True),
                "error": checked(hist.GetBinError(1), f"error {stage}/{channel}/{process}", True)}
    raise RuntimeError(f"Missing saved counting yield {stage}/{channel}/{process}. "
                       "Rerun FitDiagnostics with --saveNormalizations --saveShapes "
                       "--saveWithUncertainties; no process-error quadrature fallback is used.")


def extract(source, nominal):
    fits = {}
    for name in ("fit_b", "fit_s"):
        fit = source.Get(name)
        if not fit or fit.status() != 0 or fit.covQual() != 3:
            raise RuntimeError(f"{name}: require status=0 and covQual=3 before plotting")
        fits[name] = {"status": int(fit.status()), "covQual": int(fit.covQual()),
                      "edm": checked(fit.edm(), "EDM")}
        r = fit.floatParsFinal().find("r")
        if r:
            fits[name]["r"] = {"value": float(r.getVal()), "error": float(r.getError())}
    result = {}
    for stage in STAGES:
        result[stage] = {}
        for channel, processes in nominal.items():
            parts = {p: saved_value(source, stage, channel, p) for p in processes}
            total = saved_value(source, stage, channel, "total_background")
            require_close(sum(v["yield"] for v in parts.values()), total["yield"],
                          f"{stage}/{channel} process closure")
            if stage == "prefit":
                for process, expected in processes.items():
                    require_close(parts[process]["yield"], expected, f"prefit {channel}/{process}")
            result[stage][channel] = {"processes": parts, "total_background": total}
    return result, fits


def write_tables(payload, directory):
    (directory / "prefit_postfit_yields.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with (directory / "prefit_postfit_yields.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mass_GeV", "target", "stage", "channel", "process", "yield", "error", "asimov"])
        for stage, channels in payload["stages"].items():
            for channel, slot in channels.items():
                for process, values in dict(slot["processes"], total_background=slot["total_background"]).items():
                    writer.writerow([payload["mass"], payload["target"], stage, channel,
                                     process, values["yield"], values["error"], payload["asimov"][channel]])


def plot(payload, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    channels = list(payload["asimov"])
    x = np.arange(len(channels))
    processes = list(payload["stages"]["prefit"][channels[0]]["processes"])
    fig, axes = plt.subplots(2, 3, figsize=(17, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    colours = {"QCD": "#e6ab02", "tt": "#66a61e", "ST": "#1b9e77",
               "DY": "#7570b3", "Others": "#e7298a"}
    for col, (stage, label) in enumerate(zip(STAGES, LABELS)):
        ax, ratio = axes[:, col]
        slots = payload["stages"][stage]
        bottom = np.zeros(len(channels))
        for process in processes:
            vals = np.array([slots[c]["processes"][process]["yield"] for c in channels])
            ax.bar(x, vals, bottom=bottom, width=.8, label=process, color=colours.get(process))
            bottom += vals
        total = np.array([slots[c]["total_background"]["yield"] for c in channels])
        error = np.array([slots[c]["total_background"]["error"] for c in channels])
        data = np.array([payload["asimov"][c] for c in channels])
        # Expected Poisson precision is illustrative, not an extra fit uncertainty.
        ax.bar(x, 2*error, bottom=total-error, width=.8, color="none", edgecolor="0.3",
               hatch="////", label="Background uncertainty")
        ax.errorbar(x, data, yerr=np.sqrt(data), fmt="ko", ms=3, label="B-only Asimov")
        ax.set_title(label)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Events in counting window")
        valid = total > 0
        ratio.bar(x[valid], 2*error[valid]/total[valid], bottom=1-error[valid]/total[valid],
                  width=.8, color="0.85")
        ratio.errorbar(x[valid], data[valid]/total[valid], yerr=np.sqrt(data[valid])/total[valid],
                       fmt="ko", ms=3)
        ratio.axhline(1, color="0.3", lw=1)
        ratio.set_ylabel("Asimov / Bkg.")
        ratio.set_xticks(x)
        ratio.set_xticklabels([c.removeprefix("bin_") for c in channels], rotation=55, ha="right")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"CMS Internal · {payload['target']} · m(Z′) = {payload['mass']:g} GeV\n"
                 "Background-only Asimov; point errors = expected √N; no observed SR data")
    fig.tight_layout(rect=(0, 0, 1, .91))
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"prefit_postfit_yields.{suffix}", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mass", type=float)
    parser.add_argument("--target", choices=("Run2", "Run3", "Run2Run3"), default="Run2Run3")
    parser.add_argument("--card-base", default="datacards_NIsoMuon/alpha/blind/combined")
    parser.add_argument("--output-base", default="prefit_postfit_outputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plot-only", action="store_true", help="redraw the saved, validated JSON without refitting")
    args = parser.parse_args()
    directory = review.outdir(args)
    try:
        if args.plot_only:
            plot(json.loads((directory / "prefit_postfit_yields.json").read_text()), directory)
            return 0
        card = review.card_path(args)
        nominal = nominal_backgrounds(card)
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f"Output is not empty: {directory}; use a fresh --output-base or --plot-only")
        command = [sys.executable, str(Path(review.__file__).resolve()), str(args.mass),
                   "--task", "fitdiag", "--inject-r", "0", "--r-min", "-50",
                   "--target", args.target, "--card-base", str(Path(args.card_base).resolve()),
                   "--output-base", str(Path(args.output_base).resolve())]
        print("Fit command:", " ".join(command), flush=True)
        if args.dry_run:
            return 0
        import ROOT
        import matplotlib  # fail before starting fits if plotting dependencies are missing
        ROOT.gROOT.SetBatch(True)
        subprocess.run(command, check=True)
        label = review.mass_label(args.mass)
        root_path = directory / f"fitDiagnostics.ReviewFitDiag_{args.target}_M{label}.root"
        source = ROOT.TFile.Open(str(root_path))
        if not source or source.IsZombie():
            raise RuntimeError(f"Cannot read {root_path}")
        try:
            stages, fits = extract(source, nominal)
        finally:
            source.Close()
        payload = {"schema": "NPS26009-counting-prefit-postfit-v1", "mass": args.mass,
                   "target": args.target, "card": str(card),
                   "card_sha256": hashlib.sha256(card.read_bytes()).hexdigest(),
                   "root_file": str(root_path), "command": command, "injected_r": 0,
                   "asimov": {c: sum(p.values()) for c, p in nominal.items()},
                   "uncertainty": "Combine saved correlated total-background RMS; not sum of process errors",
                   "fits": fits, "stages": stages}
        write_tables(payload, directory)
        plot(payload, directory)
        print(f"Saved PDF, PNG, JSON and CSV under {directory}")
        return 0
    except (RuntimeError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
