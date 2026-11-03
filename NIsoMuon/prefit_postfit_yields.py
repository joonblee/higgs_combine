#!/usr/bin/env python3
"""Blind counting-channel pre/post-fit yields from a fresh Asimov FitDiagnostics.

Run after cmsenv, from NIsoMuon. Never reads SR data histograms. Uses the
existing review fit, nominal rateParam evaluation and signed-r safeguard.
Uncertainty bands use correlated joint nuisance responses,
not a quadrature sum of process errors. Requires PyROOT and matplotlib.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from copy import copy
from pathlib import Path
import subprocess
import sys

import combine_review as review


SCAN_MASSES = (12, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70)

STAGES = ("prefit", "fit_b", "fit_s")
LABELS = ("Prefit", "Postfit B-only (r fixed to 0)", "Postfit S+B (r floating)")


# SKPlotMaker plotter.py: default b-tag stack order and data-driven labels.
# RGB equivalents of kSpring-9, kViolet-4, kOrange-2, kAzure-9, kGray+1.
STACK_ORDER = ("Others", "ST", "tt", "QCD", "DY")
PROCESS_COLOURS = {"Others": "#99ff33", "ST": "#cc66ff", "tt": "#ffcc33",
                   "QCD": "#99ccff", "DY": "#999999"}
PROCESS_LABELS = {"Others": "Others", "ST": "single top", "tt": r"$t\bar{t}$",
                  "QCD": "QCD (SS data)", "DY": "DY (light-jet data)"}
UNCERTAINTY_STYLE = {"facecolor": "none", "edgecolor": "0.35",
                     "hatch": "////", "linewidth": .7}


def ordered_processes(payload):
    available = {p for slot in payload["stages"]["prefit"].values() for p in slot["processes"]}
    return [p for p in STACK_ORDER if p in available] + sorted(available-set(STACK_ORDER))


def background_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    entries = dict(zip(labels, handles))
    order = ["Bkg Asimov"] + [PROCESS_LABELS[p] for p in reversed(STACK_ORDER)]
    order += [label for label in labels if label not in order and label != "Background uncertainty"]
    order += ["Background uncertainty"]
    present = [label for label in order if label in entries]
    ax.legend([entries[label] for label in present], present, **kwargs)


def background_interval(totals):
    import numpy as np

    if all("q16" in item and "q84" in item for item in totals):
        return (np.array([item["q16"] for item in totals]),
                np.array([item["q84"] for item in totals]),
                "Background central 68% interval")
    # Legacy JSON contains RMS errors only.
    central = np.array([item["yield"] for item in totals])
    error = np.array([item["error"] for item in totals])
    return central - error, central + error, "Background RMS uncertainty"


def checked(value, label, nonnegative=False):
    value = float(value)
    if not math.isfinite(value) or (nonnegative and value < 0):
        raise RuntimeError(f"Invalid {label}: {value}")
    return value


def central_yield(value, label, signed_values):
    """Preserve the fitted model value exactly; do not impose per-process bounds."""
    value = checked(value, label)
    if value < 0:
        signed_values.append({"label": label, "raw_yield": value})
    return value


def signed_stack_bottom(values, positive, negative):
    """Place signed contributions above/below zero without changing yields."""
    bottom = []
    for i, value in enumerate(values):
        base = positive if value >= 0 else negative
        bottom.append(base[i])
        base[i] += value
    return bottom


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


def sampled_background(functions, total_expectation, label, diagnostics):
    """Keep signed Gaussian process responses; validate the Poisson mean."""
    values = {p: checked(f.getVal(), f"sample {label}/{p}") for p, f in functions.items()}
    mean = checked(total_expectation.getVal(), f"Poisson mean {label}", True)
    diagnostics["minimum_expected_count"] = min(diagnostics["minimum_expected_count"], mean)
    for p, value in values.items():
        if value < 0:
            entry = diagnostics["negative_process_draws"].setdefault(
                f"{label}/{p}", {"count": 0, "minimum": 0.})
            entry["count"] += 1
            entry["minimum"] = min(entry["minimum"], value)
    return values


def extract_counting_workspace(source, workspace_path, nominal, samples=2000):
    """Evaluate counting expectations directly; retain nonlinear nuisance responses.

    Counting models use RooPoisson rather than RooAddPdf. FitDiagnostics v10.6
    does not traverse these PDFs when saving norm_* or shapes_*.
    """
    import ROOT
    handle = ROOT.TFile.Open(str(workspace_path))
    if not handle or handle.IsZombie():
        raise RuntimeError(f"Cannot read workspace {workspace_path}")
    try:
        w = handle.Get("w")
        mc = w.obj("ModelConfig") if w else None
        if not mc or not w.var("r"):
            raise RuntimeError("Missing counting workspace ModelConfig/r")
        w.saveSnapshot("yield_initial", w.allVars())
        functions = {}
        expected_counts = {}
        for channel, processes in nominal.items():
            functions[channel] = {}
            expected_counts[channel] = w.function(f"n_exp_bin{channel}")
            if not expected_counts[channel]:
                raise RuntimeError(f"Missing total Poisson expectation n_exp_bin{channel}")
            for process in processes:
                name = f"n_exp_bin{channel}_proc_{process}"
                func = w.function(name) or w.var(name)
                if not func:
                    raise RuntimeError(f"Missing counting expectation {name}")
                functions[channel][process] = func
        fits, result = {}, {}
        ROOT.RooRandom.randomGenerator().SetSeed(123456)
        for stage in STAGES:
            w.loadSnapshot("yield_initial")
            w.var("r").setVal(0.)
            fit = None
            draws = None
            if stage == "prefit":
                prior = w.pdf("nuisancePdf")
                if not prior:
                    raise RuntimeError("Missing nuisancePdf for correlated prefit sampling")
                draws = prior.generate(mc.GetNuisanceParameters(), samples)
                if not draws or draws.numEntries() != samples:
                    raise RuntimeError("Prefit nuisance generation failed")
                w.loadSnapshot("yield_initial")
                w.var("r").setVal(0.)
            else:
                fit = source.Get(stage)
                if not fit or fit.status() != 0 or fit.covQual() != 3:
                    raise RuntimeError(f"{stage}: require status=0 and covQual=3")
                w.allVars().assignValueOnly(fit.floatParsFinal())
                fits[stage] = {"status": int(fit.status()), "covQual": int(fit.covQual()),
                               "edm": checked(fit.edm(), "EDM")}
                r = fit.floatParsFinal().find("r")
                if r:
                    fits[stage]["r"] = {"value": float(r.getVal()), "error": float(r.getError())}
            signed_values = fits.setdefault("central_signed_yields", [])
            central = {c: {p: central_yield(f.getVal(), f"{stage}/{c}/{p}", signed_values)
                           for p, f in fs.items()} for c, fs in functions.items()}
            for c in functions:
                checked(expected_counts[c].getVal(), f"central Poisson mean {stage}/{c}", True)
                checked(sum(central[c].values()), f"central total background {stage}/{c}", True)
            if stage == "prefit":
                for c, ps in nominal.items():
                    for p, value in ps.items():
                        require_close(central[c][p], value, f"prefit {c}/{p}")
            squares = {c: {p: 0. for p in (*ps, "total_background")} for c, ps in nominal.items()}
            combined_central = {p: sum(ps.get(p, 0.) for ps in central.values())
                                for p in {p for ps in central.values() for p in ps}}
            combined_squares = {p: 0. for p in (*combined_central, "total_background")}
            diagnostics = {"samples": samples, "minimum_expected_count": float("inf"),
                           "negative_process_draws": {}}
            total_draws = {channel: [] for channel in (*central, "__combined__")}
            for i in range(samples):
                draw = draws.get(i) if draws else fit.randomizePars()
                w.allVars().assignValueOnly(draw)
                combined_draw = {p: 0. for p in combined_central}
                for c, fs in functions.items():
                    values = sampled_background(fs, expected_counts[c], f"{stage}/{c}", diagnostics)
                    for p, value in values.items():
                        combined_draw[p] += value
                        squares[c][p] += (value-central[c][p])**2
                    # Sum on each joint draw BEFORE taking its RMS: includes correlations.
                    total = sum(values.values())
                    total_draws[c].append(total)
                    squares[c]["total_background"] += (total-sum(central[c].values()))**2
                for p, value in combined_draw.items():
                    combined_squares[p] += (value-combined_central[p])**2
                total = sum(combined_draw.values())
                total_draws["__combined__"].append(total)
                combined_squares["total_background"] += (total-sum(combined_central.values()))**2
            fits.setdefault("sampling", {})[stage] = diagnostics
            result[stage] = {c: {
                "processes": {p: {"yield": v, "error": math.sqrt(squares[c][p]/samples)}
                              for p, v in ps.items()},
                "total_background": {"yield": sum(ps.values()),
                                     "error": math.sqrt(squares[c]["total_background"]/samples)}}
                for c, ps in central.items()}
            result[stage]["__combined__"] = {
                "processes": {p: {"yield": v, "error": math.sqrt(combined_squares[p]/samples)}
                              for p, v in combined_central.items()},
                "total_background": {"yield": sum(combined_central.values()),
                                     "error": math.sqrt(combined_squares["total_background"]/samples)}}
            import numpy as np
            for channel, yields in total_draws.items():
                q16, q84 = np.percentile(yields, [16, 84])
                result[stage][channel]["total_background"].update(q16=float(q16), q84=float(q84))
        return result, fits
    finally:
        handle.Close()


def product_path(payload, directory, suffix, by_era=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = "prefit_postfit_yields_by_era" if by_era else "prefit_postfit_yields"
    return directory / f"{stem}_{payload['target']}_M{review.mass_label(payload['mass'])}.{suffix}"


def write_tables(payload, directory):
    product_path(payload, directory, "json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with product_path(payload, directory, "csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mass_GeV", "target", "stage", "channel", "process", "yield", "error", "asimov"])
        for stage, channels in payload["stages"].items():
            for channel, slot in dict(channels, combined=payload["combined_stages"][stage]).items():
                for process, values in dict(slot["processes"], total_background=slot["total_background"]).items():
                    writer.writerow([payload["mass"], payload["target"], stage, channel,
                                     process, values["yield"], values["error"], (sum(payload["asimov"].values()) if channel == "combined" else payload["asimov"][channel])])


def plot_by_era(payload, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    channels = list(payload["asimov"])
    x = np.arange(len(channels))
    processes = ordered_processes(payload)
    fig, axes = plt.subplots(2, 3, figsize=(17, 7), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for col, (stage, label) in enumerate(zip(STAGES, LABELS)):
        ax, ratio = axes[:, col]
        slots = payload["stages"][stage]
        positive, negative = np.zeros(len(channels)), np.zeros(len(channels))
        for process in processes:
            vals = np.array([slots[c]["processes"][process]["yield"] for c in channels])
            ax.bar(x, vals, bottom=signed_stack_bottom(vals, positive, negative), width=.8, label=PROCESS_LABELS.get(process, process), color=PROCESS_COLOURS.get(process, "#bbbbbb"))
        total = np.array([slots[c]["total_background"]["yield"] for c in channels])
        lower, upper, band_label = background_interval([slots[c]["total_background"] for c in channels])
        data = np.array([payload["asimov"][c] for c in channels])
        # Expected Poisson precision is illustrative, not an extra fit uncertainty.
        ax.bar(x, upper-lower, bottom=lower, width=.8, **UNCERTAINTY_STYLE, label=band_label)
        ax.errorbar(x, data, yerr=np.sqrt(data), fmt="ko", ms=3, label="Bkg Asimov")
        ax.set_title(label)
        ax.set_ylim(bottom=min(0., float(negative.min())*1.15))
        ax.set_ylabel("Events in counting window")
        valid = total > 0
        ratio.bar(x[valid], (upper[valid]-lower[valid])/total[valid], bottom=lower[valid]/total[valid],
                  width=.8, **UNCERTAINTY_STYLE)
        ratio.errorbar(x[valid], data[valid]/total[valid], yerr=np.sqrt(data[valid])/total[valid],
                       fmt="ko", ms=3)
        ratio.axhline(1, color="0.3", lw=1)
        ratio.set_ylabel("Asimov / Bkg.")
        ratio.set_xticks(x)
        ratio.set_xticklabels([c.removeprefix("bin_") for c in channels], rotation=55, ha="right")
    background_legend(axes[0, 0], fontsize=8)
    fig.suptitle(f"CMS Internal · {payload['target']} · m(Z′) = {payload['mass']:g} GeV\n"
                 "Background-only Asimov; point errors = expected √N; no observed SR data")
    fig.tight_layout(rect=(0, 0, 1, .91))
    for suffix in ("pdf", "png"):
        fig.savefig(product_path(payload, directory, suffix, by_era=True), dpi=160)
    plt.close(fig)


def plot(payload, directory):
    if "combined_stages" not in payload:
        raise RuntimeError("Combined covariance is absent in this older JSON. Run --extract-only once; era errors cannot be added in quadrature.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    slots = payload["combined_stages"]
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    x = np.arange(3)
    positive, negative = np.zeros(3), np.zeros(3)
    processes = ordered_processes(payload)
    for p in processes:
        values = np.array([slots[s]["processes"].get(p, {"yield": 0.})["yield"] for s in STAGES])
        ax.bar(x, values, bottom=signed_stack_bottom(values, positive, negative), width=.65, label=PROCESS_LABELS.get(p, p), color=PROCESS_COLOURS.get(p, "#bbbbbb"))
    totals = np.array([slots[s]["total_background"]["yield"] for s in STAGES])
    lower, upper, band_label = background_interval([slots[s]["total_background"] for s in STAGES])
    asimov = sum(payload["asimov"].values())
    ax.bar(x, upper-lower, bottom=lower, width=.65, **UNCERTAINTY_STYLE, label=band_label)
    ax.errorbar(x, [asimov]*3, yerr=[math.sqrt(asimov)]*3, fmt="ko", label="Bkg Asimov")
    ax.set_ylim(bottom=min(0., float(negative.min())*1.15))
    ax.set_ylabel("Events in counting windows (sum of eras)")
    background_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    valid = totals > 0
    ratio.bar(x[valid], (upper[valid]-lower[valid])/totals[valid], bottom=lower[valid]/totals[valid],
              width=.65, **UNCERTAINTY_STYLE)
    ratio.errorbar(x[valid], asimov/totals[valid], yerr=math.sqrt(asimov)/totals[valid], fmt="ko")
    ratio.axhline(1, color="0.3", lw=1)
    ratio.set_ylabel("Asimov / Bkg.")
    ratio.set_xticks(x)
    ratio.set_xticklabels(["Prefit", "Postfit B-only\n(r fixed to 0)", "Postfit S+B\n(r floating)"])
    target = {"Run2Run3": "Run 2+3", "Run2": "Run 2", "Run3": "Run 3"}[payload["target"]]
    fig.suptitle(f"CMS Internal · {target} · m(Z′) = {payload['mass']:g} GeV\n"
                 "Background-only Asimov; no observed SR data", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    for suffix in ("pdf", "png"):
        fig.savefig(product_path(payload, directory, suffix), dpi=160)
    plt.close(fig)
    plot_by_era(payload, directory)


def load_plot_payload(args, products):
    path = product_path({"target": args.target, "mass": args.mass}, products, "json")
    if not path.exists():
        path = review.outdir(args) / "prefit_postfit_yields.json"
    payload = json.loads(path.read_text())
    if payload["target"] != args.target or payload["mass"] != args.mass:
        raise RuntimeError(f"Target or mass mismatch in {path}")
    return payload


def plot_mass_scan(payloads, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    masses = np.array([p["mass"] for p in payloads])
    asimov = np.array([sum(p["asimov"].values()) for p in payloads])
    if not np.all(np.isfinite(asimov) & (asimov > 0)):
        raise RuntimeError("Mass-scan ratios require positive finite Asimov counts")
    target = payloads[0]["target"]
    fig, (ax, ratio) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [7, 3]}
    )
    log_y = np.all(asimov - np.sqrt(asimov) > 0)
    for stage, label, color in zip(STAGES, LABELS, ("black", "#c62828", "#1565c0")):
        totals = [p["combined_stages"][stage]["total_background"] for p in payloads]
        lower, upper, band_label = background_interval(totals)
        if band_label != "Background central 68% interval":
            raise RuntimeError("Mass-scan bands require q16/q84. Run the batch with --extract-only.")
        central = np.array([item["yield"] for item in totals])
        log_y = log_y and np.all(lower > 0) and np.all(central > 0)
        style = UNCERTAINTY_STYLE if stage == "prefit" else {"color": color, "alpha": .17}
        ax.fill_between(masses, lower, upper, **style, zorder=1)
        ratio.fill_between(masses, lower/asimov, upper/asimov, **style, zorder=1)
        ax.plot(masses, central, color=color, linewidth=1.5, label=label, zorder=2)
        ratio.plot(masses, central/asimov, color=color, linewidth=1.5, zorder=2)

    ax.errorbar(masses, asimov, yerr=np.sqrt(asimov), fmt="ko", ms=4, label="Bkg Asimov", zorder=4)
    ratio.errorbar(masses, np.ones_like(asimov), yerr=1/np.sqrt(asimov), fmt="ko", ms=4, zorder=4)
    ratio.axhline(1, color="0.5", linewidth=.8, zorder=0)
    if log_y:
        ax.set_yscale("log")
    # SKPlotMaker plotter.py uses ROOT pad-relative text sizes. At 7 inches
    # tall with a 70:30 split, these translate to roughly 14–18 points.
    ax.set_ylabel("Events", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ratio.set_ylabel("Yield / Asimov", fontsize=18)
    ratio.set_xlabel("Dimuon mass hypothesis [GeV]", fontsize=18)
    ratio.tick_params(axis="both", labelsize=15)
    ratio.set_xticks(masses, labels=[f"{mass:g}" for mass in masses])
    ax.legend(fontsize=14, ncol=2)
    ax.set_title(f"CMS Internal · {target} · Counting windows at each mass hypothesis\n"
                 "Bands: central 68% background yield intervals", fontsize=16)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"prefit_postfit_yields_{target}_mass_scan.{suffix}", dpi=160)
    plt.close(fig)


def run_mass_batch(args, products):
    from limit_workflow import parse_float_list

    masses = sorted(SCAN_MASSES if args.masses == "all" else parse_float_list(args.masses))
    if len(masses) < 2 or any(not math.isfinite(m) or m <= 0 for m in masses):
        raise RuntimeError("--masses requires at least two positive, finite mass hypotheses")
    if len({review.mass_label(m) for m in masses}) != len(masses):
        raise RuntimeError("--masses contains duplicate output mass labels")
    requests = []
    for mass in masses:
        request = copy(args)
        request.mass = mass
        requests.append(request)

    def compute(request):
        command = [sys.executable, str(Path(__file__).resolve()), str(request.mass),
                   "--target", request.target, "--card-base", request.card_base,
                   "--output-base", request.output_base, "--tables-only"]
        directory = review.outdir(request)
        label = review.mass_label(request.mass)
        # Reuse completed fit products when a batch contains previously run masses.
        required = ("review_metadata.json", f"fitdiag_summary_M{label}.json",
                    f"fitDiagnostics.ReviewFitDiag_{request.target}_M{label}.root",
                    f"workspace_{request.target}_M{label}.root")
        if request.extract_only or all((directory / name).is_file() for name in required):
            command.append("--extract-only")
        if args.dry_run:
            print(" ".join(command), flush=True)
        else:
            subprocess.run(command, check=True)

    if not args.plot_only:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(compute, requests))
    if args.dry_run:
        return
    payloads = [load_plot_payload(request, products) for request in requests]
    plot_mass_scan(payloads, products)
    for payload in payloads:
        plot(payload, products)
    print(f"Saved mass-scan and per-mass plots under {products}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mass", type=float, nargs="?")
    parser.add_argument("--masses", help="Comma-separated mass hypotheses, or all for the 12–70 GeV grid")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel mass workers in batch mode")
    parser.add_argument("--tables-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--target", choices=("Run2", "Run3", "Run2Run3"), default="Run2Run3")
    parser.add_argument("--card-base", default="datacards_NIsoMuon/alpha/blind/combined")
    parser.add_argument("--output-base", default="prefit_postfit_outputs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extract-only", action="store_true", help="extract completed background-only Asimov fits without rerunning Combine")
    parser.add_argument("--plot-only", action="store_true", help="redraw the saved, validated JSON without refitting")
    args = parser.parse_args()
    if (args.mass is None) == (args.masses is None):
        parser.error("Specify one positional mass or --masses")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.plot_only and (args.extract_only or args.dry_run):
        parser.error("--plot-only is incompatible with --extract-only and --dry-run")
    products = Path(args.output_base).resolve()
    try:
        if args.masses is not None:
            run_mass_batch(args, products)
            return 0
        directory = review.outdir(args)  # existing FitDiagnostics work files
        if args.plot_only:
            payload = load_plot_payload(args, products)
            write_tables(payload, products)
            plot(payload, products)
            return 0
        card = review.card_path(args)
        nominal = nominal_backgrounds(card)
        if not args.extract_only and directory.exists() and any(directory.iterdir()):
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
        if args.extract_only:
            summary = json.loads((directory / f"fitdiag_summary_M{review.mass_label(args.mass)}.json").read_text())
            metadata = json.loads((directory / "review_metadata.json").read_text())
            if (summary.get("injected_r") != 0 or metadata.get("mass") != args.mass
                    or metadata.get("target") != args.target or Path(metadata["card"]).resolve() != card):
                raise RuntimeError("Existing fit is not the requested background-only Asimov review")
        else:
            subprocess.run(command, check=True)
        label = review.mass_label(args.mass)
        root_path = directory / f"fitDiagnostics.ReviewFitDiag_{args.target}_M{label}.root"
        source = ROOT.TFile.Open(str(root_path))
        if not source or source.IsZombie():
            raise RuntimeError(f"Cannot read {root_path}")
        try:
            stages, fits = extract_counting_workspace(
                source, directory / f"workspace_{args.target}_M{label}.root", nominal)
            combined_stages = {stage: slots.pop("__combined__") for stage, slots in stages.items()}
            uncertainty_method = "Workspace joint central 68% interval and RMS: 2000 prior/fitted-covariance draws; sum eras on each draw; seed 123456"
        finally:
            source.Close()
        payload = {"schema": "NPS26009-counting-prefit-postfit-v1", "mass": args.mass,
                   "target": args.target, "card": str(card),
                   "card_sha256": hashlib.sha256(card.read_bytes()).hexdigest(),
                   "root_file": str(root_path), "command": command, "injected_r": 0,
                   "asimov": {c: sum(p.values()) for c, p in nominal.items()},
                   "uncertainty": uncertainty_method,
                   "fits": fits, "stages": stages, "combined_stages": combined_stages}
        write_tables(payload, products)
        if not args.tables_only:
            plot(payload, products)
        print(f"Saved mass {args.mass:g} yield products under {products}")
        return 0
    except (RuntimeError, OSError, ImportError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
