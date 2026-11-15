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


def evaluate(functions):
    native = np.array([[obj.getVal() for obj in row] for row in functions])
    if not np.isfinite(native).all() or native.min() < -1.e-7:
        raise ValueError("Invalid background prediction in the workspace")
    return native


def template_projection(reader, model, rows, edges):
    """Nominal within-bin fractions. Keep each fitted process/channel yield on every draw."""
    import limit_workflow as base

    args = base.build_parser().parse_args([])
    for key in ("base_dir", "region", "trigger"):
        setattr(args, key, model["settings"][key])
    weights = np.zeros((len(edges) - 1, len(rows), len(PROCESSES)))
    notes = []
    for i, row in enumerate(rows):
        lo = np.maximum(edges[:-1], row["low"])
        hi = np.minimum(edges[1:], row["high"])
        overlap = hi > lo + 1.e-8
        for j, proc in enumerate(PROCESSES):
            filename = base.file_for_process(args, row["era"], proc, "", "nominal")
            key = base.DY_AUX_SOURCE_PATH if proc == "DY" else base.hist_path(args.region)
            values = []
            for a, b in zip(lo[overlap], hi[overlap]):
                value = reader.integral(filename, key, float(a), float(b))
                if value is None or not np.isfinite(value.value):
                    raise ValueError(f"Missing template integral {filename}:{key} [{a}, {b}]")
                values.append(value.value)
            signed = np.array(values)
            # DY's transfer factor is constant within this era and cancels in fractions.
            scale = row["dy_nf"] if proc == "DY" else 1.
            expected = max(0., signed.sum()) * scale
            if not np.isclose(expected, row["rates"][proc], atol=1.e-7, rtol=2.e-6):
                raise ValueError(f"Nominal template differs from saved {row['channel']} {proc}: "
                                 f"{expected} versus {row['rates'][proc]}")
            fraction = np.maximum(signed, 0.)
            if fraction.sum() == 0:
                # No positive template support for a potential additive postfit yield.
                fraction = (hi - lo)[overlap]
                policy = "width fractions (no positive template support)"
            elif np.any(signed < 0):
                policy = "positive overlap integrals renormalised (signed template)"
            else:
                policy = None
            if policy and overlap.sum() > 1:
                notes.append(dict(channel=row["channel"], process=proc, allocation=policy))
            weights[overlap, i, j] = fraction / fraction.sum()
    np.testing.assert_allclose(weights.sum(axis=0), 1.)
    return weights, notes


def band(variables, means, transform, prediction, count, rng):
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
                draws.append(prediction())
                if len(draws) == count:
                    break
        if len(draws) != count:
            raise ValueError("Too few joint Gaussian draws inside parameter ranges")
        return np.asarray(draws)
    finally:
        for var, x in zip(variables, means):
            var.setVal(float(x))


def draw(edges, components, bounds, asimov, mass, stem, projected):
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
    centres = (edges[1:] + edges[:-1]) / 2
    points, = ax.plot(centres, asimov / widths, "ko", ms=3.0, zorder=5)
    ax.set_yscale("log")
    ax.set_ylim(0.01, max(bottom.max(), upper.max(), 1.) * 120.)
    ax.set_ylabel("Events / GeV", loc="top", fontsize=13)
    ax.text(0, 1.015, r"$\bf{CMS}$  $\it{Preliminary}$", transform=ax.transAxes, fontsize=15)
    ax.text(1, 1.018, r"138 fb$^{-1}$ (13 TeV) + 62 fb$^{-1}$ (13.6 TeV)",
            transform=ax.transAxes, ha="right", fontsize=11)
    ax.text(0.035, 0.95, "Run 2+3, OS, b-jet\nBlinded, background-only Asimov\n"
            + rf"Postfit ($r=0$), $M={mass:g}$ GeV" + ("\n1 GeV template projection" if projected else ""),
            transform=ax.transAxes, va="top",
            linespacing=1.45, fontsize=11)
    ax.legend([points] + handles[::-1] + [band_patch],
              ["Asimov"] + list(LABELS[::-1]) + ["Postfit bkg (16–84%)"],
              loc="upper right", frameon=False, fontsize=10, handlelength=2.4,
              borderaxespad=0.7, labelspacing=0.28)

    # Show the relative model band over the same bins. Empty predictions have no ratio.
    relative = np.divide(bounds, total, out=np.full_like(bounds, np.nan), where=total > 0)
    ratio.stairs(relative[1], edges, baseline=relative[0], **band_style)
    ratio.axhline(1, color="black", linestyle="--", linewidth=0.85)
    point_ratio = np.divide(asimov, total, out=np.full_like(total, np.nan), where=total > 0)
    ratio.plot(centres, point_ratio, "ko", ms=3.0, zorder=5)
    visible = np.concatenate((relative.ravel(), point_ratio))
    finite = visible[np.isfinite(visible)]
    span = max(0.5, 1.08 * np.max(np.abs(finite - 1))) if finite.size else 0.5
    ratio.set_ylim(max(0, 1 - span), 1 + span)
    ratio.set_ylabel("Asimov / postfit bkg", fontsize=11)
    ratio.set_xlabel(r"$m_{\mu^{+}\mu^{-}}$ [GeV]", loc="right", fontsize=13)
    ratio.set_xlim(edges[0], edges[-1])
    for axis in (ax, ratio):
        axis.minorticks_on()
        axis.tick_params(which="both", direction="in", top=True, right=True)
    for ext in ("pdf", "png"):
        path = stem.with_suffix("." + ext)
        fig.savefig(path, dpi=160)
        print(path)
    plt.close(fig)


def plot_mass(ROOT, fitdir, mass, output, samples, binning):
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
    if binning == "1gev":
        from limit_workflow_sideband_fit import BlindedReader
        from full_spectrum_condor import stamp
        edges = np.arange(edges[0], edges[-1] + 1., 1.)
        reader = BlindedReader()
        try:
            weights, notes = template_projection(reader, model, rows, edges)
            for filename in reader.files:
                path = str(Path(filename).resolve())
                recorded = model.get("batch_inputs", {}).get(path)
                if recorded and stamp(path) != recorded:
                    raise ValueError(f"Template changed since model production: {path}")
        finally:
            reader.close()
    else:
        weights = np.zeros((len(groups), len(rows), len(PROCESSES)))
        for k, indices in enumerate(groups):
            weights[k, indices, :] = 1.
        notes = []

    def project(native):
        return np.einsum("brp,rp->bp", weights, native)

    nominal = np.array([[row["rates"][p] for p in PROCESSES] for row in rows])
    observations = np.array([row["asimov"] for row in rows])
    np.testing.assert_allclose(nominal.sum(axis=1), observations, rtol=1.e-9, atol=1.e-8)
    asimov = (project(nominal).sum(axis=1) if binning == "1gev" else
              np.array([observations[indices].sum() for indices in groups]))
    np.testing.assert_allclose(asimov.sum(), observations.sum())
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
        def prediction():
            return project(evaluate(funcs))
        rng = np.random.default_rng(73100 + int(mass))
        pars = list(fit.floatParsFinal())
        variables = [w.var(p.GetName()) for p in pars]
        means = np.array([p.getVal() for p in pars])
        for var, x in zip(variables, means):
            var.setVal(float(x))
        saved = {b["channel"]: b["postfit"] for b in fitinfo["bins"]}
        native = np.array([[saved[row["channel"]][p] for p in PROCESSES] for row in rows])
        fitted = evaluate(funcs)
        np.testing.assert_allclose(fitted, native, rtol=1.e-5, atol=1.e-6)
        post = project(fitted)
        cov = fit.covarianceMatrix()
        matrix = np.array([[cov[i][j] for j in range(len(pars))] for i in range(len(pars))])
        draws = band(variables, means, np.linalg.cholesky(matrix), prediction, samples, rng)
        q16, median, q84 = np.quantile(draws.sum(axis=2), [0.16, 0.5, 0.84], axis=0)
        suffix = "_1GeV" if binning == "1gev" else ""
        stem = output / f"dimuon_mass_M{mass:g}_postfit_b{suffix}"
        draw(np.array(edges), post, np.array([q16, q84]), asimov, mass, stem, binning == "1gev")
        diagnostics = dict(binning=binning, samples=samples, edges=list(edges),
                           asimov=asimov.tolist(), central=post.sum(axis=1).tolist(),
                           q16=q16.tolist(), median=median.tolist(), q84=q84.tolist(),
                           mean_shift_by_process=dict(zip(PROCESSES, (draws.mean(axis=0)-post).T.tolist())),
                           projection_notes=notes)
        stem.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--masses", type=int, nargs="+", default=[12, 40, 70])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--binning", choices=("fit", "1gev"), default="fit",
                        help="Use common fit edges or project each process with fixed nominal template fractions.")
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
        plot_mass(ROOT, paths[0].parent, m, output, args.samples, args.binning)


if __name__ == "__main__":
    main()
