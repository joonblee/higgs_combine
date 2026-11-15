"""Draw blinded Run 2+3 mass distributions from a collected NIsoMuon B-only fit."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from full_spectrum_component_uncertainties import RUN2, RUN3, common_groups, nuisance_priors

PROCESSES = ("Others", "ST", "tt", "DY", "QCD")
LABELS = ("Other", "Single top", r"$t\bar{t}$", "Drell–Yan", "Multijet")
COLOURS = ("#b4a8cd", "#75a6d3", "#5173aa", "#eeb758", "#80b69d")


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
            values = means + (z * transform if transform.ndim == 1 else z @ transform.T)
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


def draw(edges, stages, mass, output):
    widths = np.diff(edges)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.9), sharey=True)
    for ax, (stage, components, bounds) in zip(axes, stages):
        bottom = np.zeros(len(widths))
        for y, colour, label in zip(components.T / widths, COLOURS, LABELS):
            ax.bar(edges[:-1], y, width=widths, bottom=bottom, align="edge",
                   color=colour, linewidth=0, label=label)
            bottom += y
        lower, upper = bounds / widths
        ax.stairs(upper, edges, baseline=lower, fill=True, hatch="////",
                  facecolor="none", edgecolor="#333333", linewidth=0.65, label="68% model band")
        ax.stairs(bottom, edges, color="#222222", linewidth=1)
        ax.set_title(stage)
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    axes[0].set_ylabel("Events / GeV")
    fig.suptitle(f"Run 2+3   |   Blinded, background-only Asimov   |   M = {mass:g} GeV")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95))
    for ext in ("pdf", "png"):
        path = output / f"dimuon_mass_M{mass:g}_prefit_postfit_b.{ext}"
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
    priors, free = nuisance_priors(card.read_text())
    if free - priors.keys():
        raise ValueError(f"Missing prefit prior width for {free - priors.keys()}")
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
        stages = []
        variables, means, widths = [], [], []
        for name, (mean, width) in priors.items():
            var = w.var(name)
            if var and not var.isConstant():
                var.setVal(mean)
                variables.append(var)
                means.append(mean)
                widths.append(width)
        nominal = evaluate(funcs, groups)
        preband = band(variables, np.array(means), np.array(widths), total, samples, rng)
        stages.append(("Prefit", nominal, preband))
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
        stages.append(("Postfit, B-only", post, postband))
        draw(np.array(edges), stages, mass, output)


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
