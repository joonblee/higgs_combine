"""Draw the blinded 1 GeV background-only prefit projection for a collected NIsoMuon workspace."""
import argparse
import json
from pathlib import Path

import numpy as np
from full_spectrum_component_uncertainties import RUN2, RUN3, common_groups
from plot_dimuon_mass import PROCESSES, LABELS, evaluate, template_projection

CONTROL_EDGES = np.arange(5., 10.)


def control_yields(ROOT, filename, key):
    """Integrate whole input bins within [5, 9) only."""
    with ROOT.TFile.Open(str(filename)) as source:
        hist = source.Get(key)
        if not hist or hist.GetDimension() != 1:
            raise ValueError(f"Missing dimuon mass histogram {filename}:{key}")
        axis = hist.GetXaxis()
        indices = [axis.FindFixBin(float(edge) + 1.e-7) for edge in CONTROL_EDGES]
        for edge, index in zip(CONTROL_EDGES, indices):
            if not 1 <= index <= hist.GetNbinsX() + 1 or abs(axis.GetBinLowEdge(index) - edge) > 1.e-7:
                raise ValueError(f"Control bin boundary {edge:g} unavailable in {filename}:{key}")
        values = np.array([hist.Integral(first, stop - 1) for first, stop in zip(indices, indices[1:])])
        if not np.isfinite(values).all():
            raise ValueError(f"Invalid control yields in {filename}:{key}")
        return values


def control_spectrum(ROOT, settings, eras, template_dir):
    import limit_workflow as base

    args = base.build_parser().parse_args([])
    args.base_dir = settings["base_dir"]
    args.trigger = settings["trigger"]
    key = base.hist_path(settings["region"])
    counts = np.zeros(len(CONTROL_EDGES) - 1, dtype=int)
    components = np.zeros((len(counts), len(PROCESSES)))
    for era in eras:
        filename = Path(base.root_dir(args, era)) / "data.root"
        observed = control_yields(ROOT, filename, key)
        if np.any(observed < 0) or np.any(observed != np.floor(observed)):
            raise ValueError(f"Invalid observed event count in {filename}:{key}")
        counts += observed.astype(int)
        for j, proc in enumerate(PROCESSES):
            filename = Path(base.file_for_process(args, era, proc, "", "nominal"))
            if template_dir is not None:
                filename = template_dir / filename.relative_to(args.base_dir)
            # The OS nominal histograms already contain the QCD and DY transfer factors.
            components[:, j] += control_yields(ROOT, filename, key)
    return counts, components


def draw(edges, components, bounds, asimov, mass, stem, projected, band_label, counts, control, fit_layout=None):
    """Use the ROOT presentation of SKPlotMaker/plotter.py's Run2+3 OS b-jet spectrum."""
    from array import array
    import ROOT

    style = ROOT.gStyle
    style.SetOptStat(0)
    style.SetOptTitle(0)
    style.SetPadTickX(1)
    style.SetPadTickY(1)
    style.SetEndErrorSize(0)
    for axis in ("X", "Y"):
        style.SetLabelFont(42, axis)
        style.SetTitleFont(42, axis)
        style.SetTitleOffset(1.05, axis)
        style.SetNdivisions(510, axis)
    ROOT.TGaxis.SetMaxDigits(3)

    canvas = ROOT.TCanvas("dimuon_canvas", "", 900, 850)
    upper = ROOT.TPad("dimuon_upper", "", 0., 0.30, 1., 1.)
    ratio = ROOT.TPad("dimuon_ratio", "", 0., 0., 1., 0.30)
    for pad, top, bottom in ((upper, 0.100, 0.030), (ratio, 0.040, 0.350)):
        pad.SetLeftMargin(0.120)
        pad.SetRightMargin(0.050)
        pad.SetTopMargin(top)
        pad.SetBottomMargin(bottom)
        pad.SetTicks(1, 1)
        pad.Draw()
    upper.SetLogy()
    widths = np.diff(edges)
    centres = (edges[1:] + edges[:-1]) / 2
    total = components.sum(axis=1)

    def frame(name, bins=edges):
        hist = ROOT.TH1D(name, "", len(bins) - 1, array("d", bins))
        hist.SetDirectory(0)
        return hist

    def graphs(scale):
        # A percentile band can lie entirely above the stack. Give E2 its
        # interval midpoint so the renderer draws either prescription exactly.
        band = ROOT.TGraphAsymmErrors()
        points = ROOT.TGraphAsymmErrors()
        for i in np.flatnonzero(scale > 0):
            lo, hi = bounds[:, i] / scale[i]
            middle = (lo + hi) / 2
            j = band.GetN()
            half = widths[i] / 2
            band.SetPoint(j, centres[i], middle)
            band.SetPointError(j, half, half, middle - lo, hi - middle)
            points.SetPoint(j, centres[i], asimov[i] / scale[i])
            points.SetPointError(j, half, half, 0., 0.)
        band.SetFillStyle(1001)
        band.SetFillColorAlpha(ROOT.kGray + 1, 0.30)
        band.SetLineColor(ROOT.kGray + 2)
        band.SetLineWidth(1)
        points.SetMarkerStyle(24)
        points.SetMarkerSize(0.7)
        points.SetMarkerColor(ROOT.kBlack)
        points.SetLineColor(ROOT.kBlack)
        points.SetLineWidth(1)
        return band, points

    def label(x, y, text, size=0.040, font=42, align=12, angle=0):
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextFont(font)
        latex.SetTextSize(size)
        latex.SetTextAlign(align)
        latex.SetTextAngle(angle)
        latex.DrawLatex(x, y, text)

    upper.cd()
    display_edges = np.concatenate((CONTROL_EDGES, edges))
    main_frame = frame("mass_axis", display_edges)
    main_frame.SetMinimum(0.01)
    main_frame.SetMaximum(80. * max(np.max(bounds[1] / widths), np.max(asimov / widths), counts.max(), control.sum(axis=1).max(), 1.))
    main_frame.GetYaxis().SetLabelSize(0.045)
    main_frame.GetXaxis().SetLabelSize(0.)
    main_frame.GetXaxis().SetTitleSize(0.)
    main_frame.Draw("AXIS")

    # Cumulative TH1s painted back to front, as in plotter.py.
    palette = (ROOT.kSpring - 9, ROOT.kViolet - 4, ROOT.kOrange - 2, ROOT.kAzure - 9, ROOT.kGray + 1)
    stack = []
    display_components = np.vstack((control, np.zeros((1, len(PROCESSES))), components))
    for name, colour, heights in zip(PROCESSES, palette, np.cumsum(display_components, axis=1).T / np.diff(display_edges)):
        hist = frame("mass_" + name, display_edges)
        hist.SetFillColor(colour)
        hist.SetLineColor(ROOT.kBlack)
        hist.SetLineWidth(1)
        for i, y in enumerate(heights, start=1):
            hist.SetBinContent(i, float(y))
        stack.append(hist)
    for hist in reversed(stack):
        hist.Draw("HIST SAME")
    band, points = graphs(widths)
    band.Draw("E2 SAME")
    points.Draw("PZ SAME")

    observed = ROOT.TGraphAsymmErrors()
    observed_ratio = ROOT.TGraphAsymmErrors()
    control_total = control.sum(axis=1)
    tail = (1. - 0.682689492137) / 2.
    for i, value in enumerate(counts):
        n = int(value)
        low = ROOT.Math.gamma_quantile(tail, n, 1.) if n else 0.
        high = ROOT.Math.gamma_quantile_c(tail, n + 1, 1.)
        observed.SetPoint(i, CONTROL_EDGES[i] + 0.5, n)
        observed.SetPointError(i, 0.5, 0.5, n - low, high - n)
        if control_total[i] > 0:
            j = observed_ratio.GetN()
            observed_ratio.SetPoint(j, CONTROL_EDGES[i] + 0.5, n / control_total[i])
            observed_ratio.SetPointError(j, 0.5, 0.5, (n-low) / control_total[i], (high-n) / control_total[i])
    observed_ratio.SetMarkerStyle(20)
    observed_ratio.SetMarkerSize(0.8)
    observed.SetMarkerStyle(20)
    observed.SetMarkerSize(0.8)
    observed.Draw("PZ SAME")

    leg = ROOT.TLegend(0.56, 0.53, 0.93, 0.86)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.040)
    leg.AddEntry(observed, "Data (5#minus9 GeV)", "pe")
    leg.AddEntry(points, "Bkg Asimov", "pe")
    for hist, name in reversed(list(zip(stack, LABELS))):
        leg.AddEntry(hist, name, "f")
    leg.AddEntry(band, band_label, "f")
    leg.Draw()

    label(0.145, 0.925, "CMS", 0.055, 61, 11)
    label(0.240, 0.925, "Preliminary", 0.040, 52, 11)
    label(0.95, 0.925, "138 fb^{-1} (13 TeV) + 62 fb^{-1} (13.6 TeV)", align=31)
    label(0.155, 0.85, "Run2+3, OS, b-jet", align=13)
    fit_label = "Fixed binning (no signal window)" if fit_layout == "fixed" else f"B-only prefit, M = {mass:g} GeV"
    label(0.155, 0.78, fit_label, align=13)
    if projected:
        label(0.155, 0.72, "1 GeV template projection", align=13)
    label(0.04, 0.65, "Events / GeV", 0.045, angle=90)
    upper.RedrawAxis()

    ratio.cd()
    ratio_frame = frame("ratio_axis", display_edges)
    ratio_frame.SetMinimum(0.7)
    ratio_frame.SetMaximum(1.3)
    ratio_frame.GetXaxis().SetTitle("m_{#mu^{+}#mu^{-}} [GeV]")
    ratio_frame.GetXaxis().SetTitleSize(0.12)
    ratio_frame.GetXaxis().SetLabelSize(0.10)
    ratio_frame.GetYaxis().SetLabelSize(0.10)
    ratio_frame.GetYaxis().SetNdivisions(505)
    ratio_frame.Draw("AXIS")
    label(0.04, 0.655, "(Data or Asimov) / Bkg", 0.11, align=22, angle=90)
    ratio_band, ratio_points = graphs(total)
    ratio_band.Draw("E2 SAME")
    line = ROOT.TLine(edges[0], 1., edges[-1], 1.)
    line.SetLineStyle(2)
    line.Draw("SAME")
    ratio_points.Draw("PZ SAME")
    observed_ratio.Draw("PZ SAME")
    ratio.RedrawAxis()

    canvas.cd()
    for ext in ("pdf", "png"):
        path = stem.with_suffix("." + ext)
        canvas.SaveAs(str(path))
        print(path)
    canvas.Close()



def plot(ROOT, path, output, mass, samples, template_dir, binning, use_current_templates):
    from limit_workflow_sideband_fit import BlindedReader

    model = json.loads((path.parent.parent / "model.json").read_text())
    if not model["blind"] or model["mass"] != mass:
        raise ValueError("Expected a matching blinded model")
    rows, native_edges, groups = common_groups(model["rows"], (*RUN2, *RUN3))
    counts, control = control_spectrum(ROOT, model["settings"], sorted({row["era"] for row in rows}), template_dir)
    if binning == "fit":
        edges = np.asarray(native_edges)
        weights = np.zeros((len(groups), len(rows), len(PROCESSES)))
        for i, group in enumerate(groups):
            weights[i, group, :] = 1.
        notes = []
    else:
        edges = np.arange(native_edges[0], native_edges[-1] + 1., 1.)
        reader = BlindedReader()
        try:
            weights, notes = template_projection(reader, model, rows, edges, template_dir, use_current_templates)
        finally:
            reader.close()
    nominal = np.array([[row["rates"][proc] for proc in PROCESSES] for row in rows])
    np.testing.assert_allclose(nominal.sum(axis=1), [row["asimov"] for row in rows], atol=1e-8)
    components = np.einsum("brp,rp->bp", weights, nominal)
    central = components.sum(axis=1)
    with ROOT.TFile.Open(str(path)) as f:
        w = f.Get("w")
        w.var("r").setVal(0.)
        funcs = []
        for row in rows:
            names = [f'n_exp_bin{row["channel"]}_proc_{p}' for p in PROCESSES]
            funcs.append([w.function(n) or w.var(n) for n in names])
        np.testing.assert_allclose(evaluate(funcs), nominal, rtol=1e-5, atol=1e-6)
        mc = w.obj("ModelConfig")
        nuis = mc.GetNuisanceParameters()
        prior = w.pdf("nuisancePdf")
        if not prior:
            constraints = mc.GetPdf().getAllConstraints(mc.GetObservables(), ROOT.RooArgSet(nuis), False)
            if not constraints.getSize():
                raise RuntimeError("No workspace nuisance constraints found")
            prior = ROOT.RooProdPdf("prefit_constraints", "prefit_constraints", ROOT.RooArgList(constraints))
        ROOT.RooRandom.randomGenerator().SetSeed(73100 + mass)
        draws = prior.generate(nuis, samples)
        if not draws or draws.numEntries() != samples:
            raise RuntimeError("Failed to sample the workspace prior")
        residuals = np.empty((samples, len(central)))
        for i in range(samples):
            nuis.assignValueOnly(draws.get(i))
            residuals[i] = np.einsum("brp,rp->b", weights, evaluate(funcs)) - central
    covariance = residuals.T @ residuals / samples
    sigma = np.sqrt(covariance.diagonal())
    suffix = "_1GeV" if binning == "1gev" else ""
    stem = output / f"dimuon_mass_M{mass}_prefit_b{suffix}"
    draw(edges, components, np.array([central-sigma, central+sigma]), central, mass, stem, binning == "1gev",
         "Prefit bkg stat #oplus syst unc. (11#minus80 GeV)", counts, control, model["settings"].get("fit_layout"))
    stem.with_suffix(".json").write_text(json.dumps(dict(edges=edges.tolist(), central=central.tolist(),
        control_data=dict(edges=CONTROL_EDGES.tolist(), counts=counts.tolist(),
                          background={proc: control[:, j].tolist() for j, proc in enumerate(PROCESSES)}),
        sigma=sigma.tolist(), yield_covariance=covariance.tolist(), projection_notes=notes,
        samples=samples, seed=73100+mass, stage="prefit", source="workspace nuisance constraints"),
        indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--masses", type=int, nargs="+", default=[12, 40, 70])
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--binning", choices=("fit", "1gev"), default="1gev")
    parser.add_argument("--use-current-templates", action="store_true", help="Verify current template integrals before using their within-bin shapes.")
    parser.add_argument("--template-dir", type=Path)
    args = parser.parse_args()
    if args.binning == "fit" and (args.template_dir or args.use_current_templates):
        parser.error("Template options require --binning 1gev")
    if args.samples < 2:
        parser.error("--samples must be at least 2")
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gSystem.Load("libHiggsAnalysisCombinedLimit")
    output = args.batch / "dimuon_mass_plots" / "prefit"
    if args.use_current_templates:
        output = output / "current_templates"
    output.mkdir(parents=True, exist_ok=True)
    for mass in args.masses:
        paths = list((args.batch / "results").glob(f"**/M{mass}/fit_r0/workspace.root"))
        if len(paths) != 1:
            raise SystemExit(f"M{mass}: expected one workspace.root, found {len(paths)}")
        plot(ROOT, paths[0], output, mass, args.samples, args.template_dir, args.binning, args.use_current_templates)

