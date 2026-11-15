"""Draw postfit blinded Run 2+3 mass distributions from a collected NIsoMuon B-only fit."""
import argparse
import hashlib
import json
import re
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


def sample_points(variables, means, transform, count, rng):
    low = np.array([v.getMin() for v in variables])
    high = np.array([v.getMax() for v in variables])
    accepted = []
    proposed = kept = 0
    while kept < count and proposed < 100 * count:
        values = means + rng.normal(size=(32, len(means))) @ transform.T
        valid = values[np.all((values >= low) & (values <= high), axis=1)]
        accepted.append(valid)
        proposed += len(values)
        kept += len(valid)
    if kept < count:
        raise ValueError("Too few joint Gaussian draws inside parameter ranges")
    return np.concatenate(accepted)[:count], kept / proposed


def evaluate_points(variables, means, points, prediction):
    try:
        return np.array([evaluate_at(variables, point, prediction) for point in points])
    finally:
        for var, x in zip(variables, means):
            var.setVal(float(x))


def evaluate_at(variables, point, prediction):
    for var, x in zip(variables, point):
        var.setVal(float(x))
    return prediction()


def audit_band(variables, means, points, matrix, prediction, post, full, norm_names,
               freeze_pattern, acceptance, edges):
    """Counterfactual yield evaluations on the same retained draws, without refitting."""
    centre = post.sum(axis=1)
    names = [v.GetName() for v in variables]
    norms = np.array([name in norm_names for name in names])
    if not norms.any():
        raise ValueError("No floating saved normalisation parameters matched the workspace")

    def total():
        return prediction().sum(axis=1)

    def summarise(values):
        q = np.quantile(values, [0.16, 0.5, 0.84], axis=0)
        below = (values < centre).mean(axis=0)
        return dict(q16=q[0].tolist(), median=q[1].tolist(), q84=q[2].tolist(),
                    mean_shift=(values.mean(axis=0) - centre).tolist(),
                    fraction_below_best=below.tolist())

    # First-order prediction at the best fit, evaluated on the identical draws.
    # At a parameter bound use the available one-sided derivative.
    deriv = np.zeros((len(means), len(centre)))
    for i, (var, best, variance) in enumerate(zip(variables, means, matrix.diagonal())):
        step = max(1.e-6, 1.e-3 * np.sqrt(variance))
        low, high = max(var.getMin(), best-step), min(var.getMax(), best+step)
        if high <= low:
            raise ValueError(f"No derivative interval for {var.GetName()}")
        try:
            var.setVal(float(low))
            minus = total()
            var.setVal(float(high))
            deriv[i] = (total()-minus)/(high-low)
        finally:
            var.setVal(float(best))

    results = {"full": summarise(full.sum(axis=2)),
               "linear": summarise(centre + (points-means) @ deriv)}
    selections = {"norm_only": norms, "norm_fixed": ~norms}
    matched = []
    if freeze_pattern:
        matched = [name for name in names if re.search(freeze_pattern, name)]
        if not matched:
            raise ValueError(f"No floating nuisance matched {freeze_pattern!r}")
        selections["selected_fixed"] = ~np.isin(names, matched)
    for label, active in selections.items():
        subset = [v for v, use in zip(variables, active) if use]
        values = evaluate_points(subset, means[active], points[:, active], total)
        results[label] = summarise(values)

    for i in sorted({0, len(centre)-1}):
        if centre[i] <= 0:
            continue
        print(f"[BAND] {edges[i]:g}–{edges[i+1]:g} GeV, best = {centre[i]:.2f} events")
        for label, item in results.items():
            ratios = [item[k][i]/centre[i] for k in ("q16", "median", "q84")]
            print(f"  {label:16s} q16/50/84 / best = {ratios[0]:.4f} {ratios[1]:.4f} "
                  f"{ratios[2]:.4f}   P(y < best) = {item['fraction_below_best'][i]:.3f}")
        shifts = (full[:, i].mean(axis=0)-post[i]) / centre[i] * 100
        print("  Full mean shift (% of total best): " +
              ", ".join(f"{proc} {shift:+.3f}%" for proc, shift in zip(PROCESSES, shifts)))
    print(f"[BAND] Fraction of proposed joint draws inside parameter ranges: {acceptance:.4f}")
    return dict(acceptance=acceptance, norm_names=[n for n in names if n in norm_names],
                freeze_pattern=freeze_pattern, matched=matched, scenarios=results)


def saved_yield_covariance(directory, rows, fitted, groups):
    """Read the FitDiagnostics B-only density covariance in labelled one-bin channels."""
    cov = directory.Get("shapes_fit_b/overall_total_covar")
    if not cov:
        return None
    width = directory.Get("shapes_fit_b/total_bin_width")
    bkg = directory.Get("shapes_fit_b/total_background")
    if not width or not bkg:
        raise ValueError("Incomplete shapes_fit_b overall histograms")

    # FitDiagnostics labels the first observable bin of a channel "<channel>_0".
    labels = [r["channel"] + "_0" for r in rows]
    def indices(axis):
        by_name = {str(axis.GetBinLabel(i)): i for i in range(1, axis.GetNbins()+1)}
        if axis.GetNbins() != len(labels) or set(by_name) != set(labels):
            raise ValueError("Overall histogram labels do not match the fitted one-bin channels")
        return [by_name[name] for name in labels]

    dx = np.array([width.GetBinContent(i) for i in indices(width.GetXaxis())])
    if not np.all(np.isfinite(dx) & (dx > 0)):
        raise ValueError("Invalid FitDiagnostics observable bin widths")
    bi = indices(bkg.GetXaxis())
    centre = np.array([bkg.GetBinContent(i) for i in bi]) * dx
    np.testing.assert_allclose(centre, fitted.sum(axis=1), rtol=1.e-5, atol=1.e-6)
    ix, iy = indices(cov.GetXaxis()), indices(cov.GetYaxis())
    matrix = np.array([[cov.GetBinContent(i, j) for j in iy] for i in ix]) * np.outer(dx, dx)
    # At r=0 the total and total background are identical, including their errors.
    err = np.array([bkg.GetBinError(i) for i in bi]) * dx
    np.testing.assert_allclose(matrix.diagonal(), err**2, rtol=2.e-5, atol=1.e-8)
    np.testing.assert_allclose(matrix, matrix.T, rtol=1.e-9, atol=1.e-10)
    merge = np.zeros((len(groups), len(rows)))
    for i, group in enumerate(groups):
        merge[i, group] = 1.
    return merge @ matrix @ merge.T


def replay_yield_covariance(fit, parameters, prediction, centre, count):
    """FitDiagnostics v10.6 getNormalizations: second moment about the best fit.

    Each model row is one counting bin. Evaluate its process expectations on
    the same joint RooFit draw across all eras, then form the display-bin sums.
    """
    deltas = np.empty((count, len(centre)))
    try:
        for i in range(count):
            parameters.assignValueOnly(fit.randomizePars())
            deltas[i] = prediction().sum(axis=1) - centre
    finally:
        parameters.assignValueOnly(fit.floatParsFinal())
    return deltas.T @ deltas / count


def draw(edges, components, bounds, asimov, mass, stem, projected, band_label):
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
              ["Asimov"] + list(LABELS[::-1]) + [band_label],
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


def plot_mass(ROOT, fitdir, mass, output, samples, binning, audit_requested, freeze_pattern, seed, band):
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
    result = fitdir / "fit_b" / "fitDiagnostics.FullSpectrum.root"
    if not result.is_file():
        result = fitdir / "fitDiagnostics.FullSpectrum.root"
    with ROOT.TFile.Open(str(fitdir / "workspace.root")) as wf, ROOT.TFile.Open(str(result)) as ff:
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
        pars = list(fit.floatParsFinal())
        if any(p.GetName() == "r" for p in pars):
            raise ValueError("Expected a background-only RooFitResult with fixed r")
        variables = [w.var(p.GetName()) for p in pars]
        means = np.array([p.getVal() for p in pars])
        for var, x in zip(variables, means):
            var.setVal(float(x))
        saved = {b["channel"]: b["postfit"] for b in fitinfo["bins"]}
        native = np.array([[saved[row["channel"]][p] for p in PROCESSES] for row in rows])
        fitted = evaluate(funcs)
        np.testing.assert_allclose(fitted, native, rtol=1.e-5, atol=1.e-6)
        post = project(fitted)
        centre = post.sum(axis=1)
        suffix = "_1GeV" if binning == "1gev" else ""
        stem = output / f"dimuon_mass_M{mass:g}_postfit_b{suffix}"
        diagnostics = dict(binning=binning, band=band, edges=list(edges),
                           asimov=asimov.tolist(), central=centre.tolist(),
                           projection_notes=notes)
        if band == "combine":
            # Within a native bin a 1 GeV projection has different fractions
            # for each process, so a total-background covariance cannot be split.
            matrix = saved_yield_covariance(ff, rows, fitted, groups) if binning == "fit" else None
            if matrix is None:
                ROOT.RooRandom.randomGenerator().SetSeed(seed + mass)
                matrix = replay_yield_covariance(fit, w.allVars(), prediction, centre, samples)
                source = "RooFitResult.randomizePars"
                diagnostics.update(seed=seed + mass, samples=samples)
            else:
                source = "shapes_fit_b/overall_total_covar"
            if not np.all(np.isfinite(matrix)) or np.any(matrix.diagonal() < 0):
                raise ValueError("Invalid display-bin yield covariance")
            sigma = np.sqrt(matrix.diagonal())
            bounds = np.array([centre - sigma, centre + sigma])
            diagnostics.update(source=source, yield_covariance=matrix.tolist(), sigma=sigma.tolist())
            band_label = r"Postfit bkg ($\pm1\sigma$)"
            print(f"[BAND] M{mass} {source}, [{edges[0]:g}, {edges[1]:g}) GeV: "
                  f"{centre[0]:.2f} +/- {sigma[0]:.2f} events")
        else:
            rng = np.random.default_rng(seed + mass)
            cov = fit.covarianceMatrix()
            matrix = np.array([[cov[i][j] for j in range(len(pars))] for i in range(len(pars))])
            points, acceptance = sample_points(variables, means, np.linalg.cholesky(matrix), samples, rng)
            draws = evaluate_points(variables, means, points, prediction)
            q16, median, q84 = np.quantile(draws.sum(axis=2), [0.16, 0.5, 0.84], axis=0)
            bounds = np.array([q16, q84])
            band_label = "Postfit bkg (16–84%)"
            diagnostics.update(seed=seed + mass, samples=samples, q16=q16.tolist(),
                               median=median.tolist(), q84=q84.tolist(),
                               mean_shift_by_process=dict(zip(PROCESSES, (draws.mean(axis=0)-post).T.tolist())))
            if audit_requested:
                diagnostics["band_audit"] = audit_band(
                    variables, means, points, matrix, prediction, post, draws,
                    fitinfo["normalisations"], freeze_pattern, acceptance, edges)
        diagnostics.update(lower=bounds[0].tolist(), upper=bounds[1].tolist())
        draw(np.array(edges), post, bounds, asimov, mass, stem, binning == "1gev", band_label)
        stem.with_suffix(".json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--masses", type=int, nargs="+", default=[12, 40, 70])
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--band", choices=("combine", "quantile"),
                        help="Default: FitDiagnostics uncertainty about the best fit. "
                             "Use quantile for the 16–84%% diagnostic.")
    parser.add_argument("--binning", choices=("fit", "1gev"), default="fit",
                        help="Use common fit edges or project each process with fixed nominal template fractions.")
    parser.add_argument("--audit-band", action="store_true",
                        help="Compare full, linearised, normalisation-only and fixed-normalisation yields.")
    parser.add_argument("--audit-freeze", metavar="REGEX",
                        help="Also evaluate with matching nuisance parameters fixed at the best fit.")
    parser.add_argument("--seed", type=int, default=73100, help="Random seed offset (mass is added).")
    args = parser.parse_args()
    band = args.band or ("quantile" if args.audit_band else "combine")
    if args.audit_band and band != "quantile":
        parser.error("--audit-band studies quantiles; use --band quantile with this option")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.audit_freeze:
        if not args.audit_band:
            parser.error("--audit-freeze requires --audit-band")
        try:
            re.compile(args.audit_freeze)
        except re.error as exc:
            parser.error(str(exc))
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit")
    output = args.batch / "dimuon_mass_plots"
    if band == "combine":
        output = output / "combine"
    output.mkdir(parents=True, exist_ok=True)
    for m in args.masses:
        paths = list((args.batch / "results").glob(f"**/M{m}/fit_r0/fit_summary.json"))
        if len(paths) != 1:
            raise SystemExit(f"M{m}: expected one collected fit_summary.json, found {len(paths)}")
        plot_mass(ROOT, paths[0].parent, m, output, args.samples, args.binning,
                  args.audit_band, args.audit_freeze, args.seed, band)


if __name__ == "__main__":
    main()
