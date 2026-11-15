"""Draw postfit blinded Run 2+3 mass distributions from a collected NIsoMuon B-only fit."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from full_spectrum_component_uncertainties import RUN2, RUN3, common_groups

PROCESSES = ("Others", "ST", "tt", "QCD", "DY")
LABELS = ("Others", "single top", r"$t\bar{t}$", "QCD (SS data)", "DY (light-jet data)")
COLOURS = ("#99ff33", "#cc66ff", "#ffcc33", "#99ccff", "#999999")


def read(path):
    return json.loads(path.read_text())


def evaluate(functions, groups):
    native = np.array([[obj.getVal() for obj in row] for row in functions])
    if not np.isfinite(native).all() or native.min() < -1.e-7:
        raise ValueError("Invalid background prediction in the workspace")
    return np.array([native[indices].sum(axis=0) for indices in groups])


def band(variables, means, transform, evaluate_total, count, rng):
    # A shared draw for all channels preserves correlations across processes and eras.
    low = np.array([v.getMin() for v in variables])
    high = np.array([v.getMax() for v in variables])
    draws = []
    trials = 0
    try:
        while len(draws) < count and trials < 100 * count:
            z = rng.normal(size=(32, len(means)))
            values = means + (z @ transform.T)
            trials += len(values)
            for point in values[np.all((values >= low) & (values <= high), axis=1)]:
                for var, x in zip(variables, point):
                    var.setVal(float(x))
                draws.append(evaluate_total())
                if len(draws) == count:
                    break
        if len(draws) != count:
            raise ValueError("Too few joint Gaussian draws inside parameter ranges")
        return np.quantile(draws, [0.16, 0.84], axis=0)
    finally:
        for var, x in zip(variables, means):
            var.setVal(float(x))


def draw(edges, components, bounds, mass, output):
    widths = np.diff(edges)
    total = components.sum(axis=1)
    lower, upper = bounds / widths
    fig, (ax, ratio) = plt.subplots(2, 1, sharex=True, figsize=(8, 7.4),
                                    gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.055})
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.10, top=0.92)
    bottom = np.zeros(len(widths))
    handles = []
    for values, colour, label in zip(components.T / widths, COLOURS, LABELS):
        top = bottom + values
        patch = ax.stairs(top, edges, baseline=bottom, fill=True, facecolor=colour,
                          edgecolor="black", linewidth=0.65, label=label)
        handles.append(patch)
        bottom = top

    band_style = dict(fill=True, facecolor="#cccccc", edgecolor="#555555", linewidth=0.6)
    band_patch = ax.stairs(upper, edges, baseline=lower, **band_style)
    ax.stairs(bottom, edges, baseline=None, color="black", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylim(0.01, max(bottom.max(), upper.max(), 1.) * 120.)
    ax.set_ylabel("Events / GeV", loc="top", fontsize=13)
    ax.text(0, 1.015, r"$\bf{CMS}$  $\it{Preliminary}$", transform=ax.transAxes, fontsize=15)
    ax.text(1, 1.018, r"138 fb$^{-1}$ (13 TeV) + 62 fb$^{-1}$ (13.6 TeV)",
            transform=ax.transAxes, ha="right", fontsize=11)
    ax.text(0.035, 0.95, "Run 2+3, OS, b-jet\nBlinded, background-only Asimov\n"
            + rf"Postfit ($r=0$), $M={mass:g}$ GeV", transform=ax.transAxes, va="top",
            linespacing=1.45, fontsize=11)
    ax.legend(handles[::-1] + [band_patch], list(LABELS[::-1]) + ["Bkg stat ⊕ syst unc."],
              loc="upper right", frameon=False, fontsize=10, handlelength=2.4,
              borderaxespad=0.7, labelspacing=0.28)

    # Show the relative model band over the same bins. Empty predictions have no ratio.
    relative = np.divide(bounds, total, out=np.full_like(bounds, np.nan), where=total > 0)
    ratio.stairs(relative[1], edges, baseline=relative[0], **band_style)
    ratio.axhline(1, color="black", linestyle="--", linewidth=0.85)
    finite = relative[np.isfinite(relative)]
    span = max(0.5, 1.08 * np.max(np.abs(finite - 1))) if finite.size else 0.5
    ratio.set_ylim(max(0, 1 - span), 1 + span)
    ratio.set_ylabel("Bkg / postfit central", fontsize=11)
    ratio.set_xlabel(r"$m_{\mu^{+}\mu^{-}}$ [GeV]", loc="right", fontsize=13)
    ratio.set_xlim(edges[0], edges[-1])
    for axis in (ax, ratio):
        axis.minorticks_on()
        axis.tick_params(which="both", direction="in", top=True, right=True)
    for ext in ("pdf", "png"):
        path = output / f"dimuon_mass_M{mass:g}_postfit_b.{ext}"
        fig.savefig(path, dpi=160)
        print(path)
    plt.close(fig)


def plot_mass(ROOT, fitdir, mass, output, samples):
    model = read(fitdir.parent / "model.json")
    summary = read(fitdir / "fit_summary.json")
    fitinfo = summary["fits"]["fit_b"]
    audit = fitinfo.get("covariance_audit", {})
    quality = fitinfo["covQual"] == 3 or (fitinfo["covQual"] == -1 and
              audit.get("method") == "robustHesse" and audit.get("validated") is True)
    card = fitdir / "datacard.txt"
    if not (model["blind"] and summary["blind"] and model["mass"] == summary["mass"] == mass
            and summary["inject_r"] == fitinfo["r"] == 0 and fitinfo["status"] == 0 and quality
            and summary["card_sha256"] == model["card_sha256"]
            and hashlib.sha256(card.read_bytes()).hexdigest() == summary["fit_card_sha256"]):
        raise ValueError(f"Incompatible or unvalidated B-only result in {fitdir}")
    rows, edges, groups = common_groups(model["rows"], (*RUN2, *RUN3))
    with ROOT.TFile.Open(str(fitdir / "workspace.root")) as wf, ROOT.TFile.Open(
        str(fitdir / "fit_b" / "fitDiagnostics.FullSpectrum.root")) as ff:
        w, fit = wf.Get("w"), ff.Get("fit_b")
        if not w or not fit or fit.status() != fitinfo["status"] or fit.covQual() != fitinfo["covQual"]:
            raise ValueError("Workspace or saved RooFitResult does not match fit summary")
        w.var("r").setVal(0.)
        funcs = []
        for row in rows:
            channel = []
            for proc in PROCESSES:
                name = f'n_exp_bin{row["channel"]}_proc_{proc}'
                obj = w.function(name) or w.var(name)
                if not obj:
                    raise ValueError("Missing workspace expectation " + name)
                channel.append(obj)
            funcs.append(channel)
        def total():
            return evaluate(funcs, groups).sum(axis=1)
        rng = np.random.default_rng(73100 + int(mass))
        pars = list(fit.floatParsFinal())
        variables = [w.var(p.GetName()) for p in pars]
        means = np.array([p.getVal() for p in pars])
        for var, x in zip(variables, means):
            var.setVal(float(x))
        saved = {b["channel"]: b["postfit"] for b in fitinfo["bins"]}
        native = np.array([[saved[row["channel"]][p] for p in PROCESSES] for row in rows])
        post = evaluate(funcs, groups)
        np.testing.assert_allclose(post, [native[g].sum(axis=0) for g in groups], rtol=1.e-5, atol=1.e-6)
        cov = fit.covarianceMatrix()
        matrix = np.array([[cov[i][j] for j in range(len(pars))] for i in range(len(pars))])
        postband = band(variables, means, np.linalg.cholesky(matrix), total, samples, rng)
        draw(np.array(edges), post, postband, mass, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--masses", type=int, nargs="+", default=[12, 40, 70])
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit")
    output = args.batch / "dimuon_mass_plots"
    output.mkdir(exist_ok=True)
    for m in args.masses:
        paths = list((args.batch / "results").glob(f"**/M{m}/fit_r0/fit_summary.json"))
        if len(paths) != 1:
            raise SystemExit(f"M{m}: expected one collected fit_summary.json, found {len(paths)}")
        plot_mass(ROOT, paths[0].parent, m, output, args.samples)


if __name__ == "__main__":
    main()
