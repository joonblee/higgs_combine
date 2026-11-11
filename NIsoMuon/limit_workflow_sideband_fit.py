#!/usr/bin/env python3
"""Blinded, hypothesis-specific QCD/DY sideband study. See SIDEBAND_FIT.md.

The existing single-window workflow is unchanged. This adds disjoint sideband
Poisson channels, shared QCD/DY normalisations, and correlated shape responses.
Only background Asimov observations are supported; OS data.root is never read.
"""
import argparse
import ast
import bisect
import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import limit_workflow as base

BACKGROUNDS = tuple(p for p in base.PROCESSES if p != 'sig')
DOMAIN = (11., 80.)
SCHEMA = 1


def finite_positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Must be finite and positive')
    return value


def single_target(value):
    targets = base.selected_targets(base.canonical_target(value))
    if len(targets) != 1:
        raise argparse.ArgumentTypeError('Choose full, Run2, Run3, or one era.')
    return targets[0]


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--masses', type=base.parse_float_list, default=[12., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60., 65., 70.])
    p.add_argument('--target', type=single_target, default='Run2Run3')
    p.add_argument('--stage', choices=['cards', 'fit', 'limits', 'all'], default='all')
    p.add_argument('--base-dir', default=base.DEFAULT_BASE_DIR)
    p.add_argument('--sigfit-dir', default='./sigfit_inputs')
    p.add_argument('--resolution-map')
    p.add_argument('--region', default=base.DEFAULT_REGION)
    p.add_argument('--trigger', default='')
    p.add_argument('--sr-nsigma', type=finite_positive, default=5.)
    p.add_argument('--exclude-nsigma', type=finite_positive, default=5.)
    p.add_argument('--outer-nsigma', type=finite_positive,
                   help='Optional local outer fit boundary; default: use the full validated 11--80 GeV domain.')
    p.add_argument('--bin-width', type=finite_positive, default=2., help='Sideband bin width in GeV, aligned to common input-bin edges.')
    p.add_argument('--norm-max', type=finite_positive, default=10., help='Upper bound of each freely fitted QCD/DY scale; boundary proximity is reported.')
    p.set_defaults(norm_scope='era')  # Each era has an independent free scale.
    p.add_argument('--output-base', default='sideband_fit_outputs')
    p.add_argument('--samples', type=int, default=2000)
    p.add_argument('--seed', type=int, default=123456)
    p.add_argument('--r-max', type=finite_positive)
    return p


def configuration(opt):
    if opt.exclude_nsigma < opt.sr_nsigma:
        raise ValueError('The excluded region must contain the SR.')
    if opt.outer_nsigma is not None and opt.outer_nsigma <= opt.exclude_nsigma:
        raise ValueError('outer-nsigma must exceed exclude-nsigma.')
    if opt.norm_max <= 1 or opt.samples < 200:
        raise ValueError('Require norm-max > 1 and at least 200 covariance draws.')
    if not opt.masses or any(not math.isfinite(m) or not DOMAIN[0] < m < DOMAIN[1] for m in opt.masses):
        raise ValueError('Masses must lie strictly inside 11--80 GeV.')
    if opt.target not in ('Run2', 'Run3', 'Run2Run3', *base.YEARS):
        raise ValueError('Choose one target: Run2, Run3, full, or one era.')
    args = base.build_parser().parse_args([])
    for name in ('base_dir', 'sigfit_dir', 'resolution_map', 'region', 'trigger', 'r_max'):
        setattr(args, name, getattr(opt, name))
    args.mode, args.backend, args.strict = 'blind', 'local', True
    args.target, args.masses, args.n_sigma = opt.target, opt.masses, opt.sr_nsigma
    args.output_base = str(Path(opt.output_base).resolve())
    base.validate_args(args)
    return args


def regions(edges, mass, sigma, sr_nsigma=5., exclude_nsigma=5., width=2., outer_nsigma=None):
    """Snap SR/exclusion outwards; domain/outer edge inwards. No split bins."""
    grid = sorted(set(round(float(x), 8) for x in edges if DOMAIN[0]-1e-7 <= x <= DOMAIN[1]+1e-7))
    if len(grid) < 4 or not grid[0] < mass < grid[-1]:
        raise ValueError('Insufficient common input-bin edges in the validated domain.')
    if sigma <= 0 or not math.isfinite(sigma) or exclude_nsigma < sr_nsigma:
        raise ValueError('Invalid resolution/exclusion.')
    def floor(x):
        return grid[max(0, min(len(grid)-1, bisect.bisect_right(grid, x+1e-8)-1))]
    def ceil(x):
        return grid[max(0, min(len(grid)-1, bisect.bisect_left(grid, x-1e-8)))]
    lo, hi = grid[0], grid[-1]
    if outer_nsigma is not None:
        lo, hi = ceil(max(lo, mass-outer_nsigma*sigma)), floor(min(hi, mass+outer_nsigma*sigma))
    srlo, srhi = floor(mass-sr_nsigma*sigma), ceil(mass+sr_nsigma*sigma)
    exlo, exhi = floor(mass-exclude_nsigma*sigma), ceil(mass+exclude_nsigma*sigma)
    out = [dict(region='SR', low=srlo, high=srhi)]
    for side, a, b in [('L', lo, exlo), ('R', exhi, hi)]:
        if b <= a:
            continue
        cuts = [a]
        for desired in np.arange(a+width, b-1e-7, width):
            edge = ceil(desired)
            if cuts[-1] < edge < b:
                cuts.append(edge)
        cuts.append(b)
        for i, (a, b) in enumerate(zip(cuts, cuts[1:])):
            out.append(dict(region=f'{side}{i:02d}', low=a, high=b))
    if len(out) < 3:
        raise ValueError('Fewer than two sideband bins; widen the fit range or reduce bin width.')
    return out


class BlindedReader(base.RootReader):
    """Prevent target-data reads and enforce exactly disjoint input integrals."""
    def _file(self, filename):
        if Path(filename).name.lower() == 'data.root':
            raise base.WorkflowError('OS data.root is forbidden in this blinded workflow.')
        return super()._file(filename)

    def integral(self, filename, path, low, high):
        if DOMAIN[0]-1e-7 <= low < high <= DOMAIN[1]+1e-7:
            handle = self._file(filename)
            hist = handle.Get(path) if handle else None
            if hist and hasattr(hist, 'GetNbinsX'):
                a, b = base.whole_bin_window(hist, low, high)
                if abs(a-low) > 1e-7 or abs(b-high) > 1e-7:
                    raise base.WorkflowError(f'Incompatible bin edges: {filename}:{path}: requested {low},{high}; actual {a},{b}')
        return super().integral(filename, path, low, high)


def common_edges(reader, args, era, signal_file):
    axes = []
    for process in base.PROCESSES:
        filename = base.file_for_process(args, era, process, signal_file, 'nominal')
        path = base.DY_AUX_SOURCE_PATH if process == 'DY' else base.hist_path(args.region)
        handle = reader._file(filename)
        hist = handle.Get(path) if handle else None
        if not hist or not hasattr(hist, 'GetNbinsX'):
            raise base.WorkflowError(f'Missing {filename}:{path}')
        axis = hist.GetXaxis()
        axes.append({round(float(axis.GetBinLowEdge(i)), 8) for i in range(1, hist.GetNbinsX()+2)})
    return sorted(set.intersection(*axes))


def covariance_factor(covariance):
    """Factor a PSD covariance, retaining its correlations and null directions."""
    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or not np.all(np.isfinite(cov)):
        raise ValueError('Invalid covariance matrix')
    if not np.allclose(cov, cov.T, rtol=1e-7, atol=1e-15):
        raise ValueError('Asymmetric covariance matrix')
    eigenvalues, vectors = np.linalg.eigh((cov+cov.T)/2)
    if eigenvalues.min() < -1e-9*max(abs(eigenvalues).max(), 1e-300):
        raise ValueError('Non-positive-semidefinite covariance matrix')
    return vectors * np.sqrt(np.maximum(eigenvalues, 0))


def norm_name(process, era, scope):
    group = era if scope == 'era' else ('Run2' if era in base.RUN2_ERAS else 'Run3')
    return f'beta_{process}_{group}'


def separation_diagnostic(rows):
    """Optimistic two-normalisation Fisher diagnostic, not the profiled error."""
    sb = [r for r in rows if r['region'] != 'SR']
    design = np.array([[r['rates']['QCD'], r['rates']['DY']] for r in sb])
    total = np.array([sum(r['rates'][p] for p in BACKGROUNDS) for r in sb])
    fisher = design.T @ (design / np.maximum(total[:, None], 1e-300))
    lengths = np.sqrt(np.diag(fisher))
    if min(lengths) == 0:
        return dict(identifiable=False, reason='No sideband sensitivity to at least one normalisation.')
    cosine = float(fisher[0, 1]/np.prod(lengths))
    return dict(identifiable=abs(cosine) < 1-1e-8, template_cosine=cosine,
                condition_number=float(np.linalg.cond(fisher)),
                interpretation='Nominal shapes with other nuisances fixed; full fit can be less constraining.')


def build_model(reader, args, opt, mass, files, windows_by_era=None):
    """Build integrated Poisson bins, preserving producer covariance modes.

    Explicit windows are used by the full-spectrum workflow. Omitting them
    retains the hypothesis-specific SR/sideband study and its checks.
    """
    base.prepare_resolution_coefficients(args)
    rows, era_info, audits = [], {}, []
    for era in base.target_years(opt.target):
        if era not in files:
            raise base.WorkflowError(f'Missing simulated signal for {era}, M={mass}.')
        sigma = mass*base.relative_resolution(args, era, mass)
        windows = (windows_by_era[era] if windows_by_era is not None else
                   regions(common_edges(reader, args, era, files[era]), mass, sigma,
                           opt.sr_nsigma, opt.exclude_nsigma, opt.bin_width, opt.outer_nsigma))
        qfile = base.file_for_process(args, era, 'QCD', '', 'nominal')
        inputs = base.read_qcd_stat_inputs(reader._file(qfile), era, base.hist_path(args.region))
        qother = base.file_for_process(args, era, 'QCD', '', 'qcd')
        if inputs[0] != base.read_qcd_stat_metadata(reader._file(qother), era, base.hist_path(args.region)):
            raise ValueError(f'{era}: stale nominal/systematic QCD metadata')
        factor = covariance_factor(inputs[0]['fit']['covariance'])
        transfer_ratios = []
        for window in windows:
            low, high = window['low'], window['high']
            channels, audit = base.build_channels_for_mass(reader, args, mass, files, [era], windows={era: (low, high)})
            failed = [k for k, v in audit.entries.items() if v.found != v.expected]
            # The diagonal scale pair is validation-only. In a spectrum bin,
            # its response need not equal the product of the two axis responses.
            # Keep that diagnostic visible without treating it as a missing
            # required histogram. Preserve the original sideband audit policy.
            diagnostics = [k for k in failed if windows_by_era is not None
                           and k.startswith('PDF_scale_factorisation/')]
            failed = [k for k in failed if k not in diagnostics]
            if failed:
                details = '; '.join(f'{k}: {audit.entries[k].missing}' for k in failed)
                raise base.WorkflowError(f'{era}/{window["region"]}: failed required audits: {details}')
            ch = channels[era]
            diagnostic_info = []
            for key in diagnostics:
                entry = audit.entries[key]
                process = key.split('/', 1)[1]
                messages = [message for message in ch.warnings
                            if message.startswith(f'The {process} diagonal scale variation ')]
                diagnostic_info.append(dict(check=key, found=entry.found, expected=entry.expected,
                                            labels=list(entry.missing), messages=messages))
                print(f'[SCALE DIAGNOSTIC] {era}/{window["region"]}: {key}; '
                      + (' '.join(messages) or ', '.join(entry.missing)))
            stats = base.qcd_root_window_statistics(inputs, low, high, ch.raw_rates['QCD'])
            responses = np.asarray(stats['gradient']) @ factor
            # A free normalisation may absorb transport/NF terms ONLY if their
            # response is a mass-independent factor over every included region.
            axis = inputs[1].GetXaxis()
            first, last = axis.FindFixBin(low+1e-8), axis.FindFixBin(high-1e-8)
            gradients = [float(h.Integral(first, last)) for h in inputs[2][6:8]]
            if ch.raw_rates['QCD'] > 0:
                transfer_ratios.append([g/ch.raw_rates['QCD'] for g in gradients])
            elif any(abs(g) > 1e-12 for g in gradients):
                raise ValueError('QCD transport is not a pure normalisation at zero nominal yield.')
            row = dict(window, channel=f'bin_{era}_{window["region"]}', era=era,
                       rates=dict(ch.raw_rates), nuisances=ch.nuisances,
                       qcd_modes=responses.tolist(),
                       qcd_shape=max(ch.qcd_shape_sigma_down or 0., ch.qcd_shape_sigma_up or 0.),
                       dy_stat=(ch.dy_nf_amc or 0.)*(ch.dy_lightjet_error or 0.),
                       dy_nf=ch.dy_nf_amc, warnings=ch.warnings)
            rows.append(row)
            audits.append(dict(era=era, region=window['region'], requirements=len(audit.entries),
                               diagnostic_failures=diagnostic_info))
        if transfer_ratios and not np.allclose(transfer_ratios, transfer_ratios[0], rtol=1e-5, atol=1e-10):
            raise ValueError(f'{era}: QCD transfer changes shape; cannot absorb it into a free normalisation.')
        selected = [r for r in rows if r['era'] == era]
        if not np.allclose([r['dy_nf'] for r in selected], selected[0]['dy_nf'], rtol=1e-6):
            raise ValueError(f'{era}: DY NF is not constant across regions.')
        has_left = any(r['region'].startswith('L') for r in selected)
        has_right = any(r['region'].startswith('R') for r in selected)
        signal_sr = sum(r['rates']['sig'] for r in selected if r['region'] == 'SR')
        signal_sb = sum(r['rates']['sig'] for r in selected if r['region'] != 'SR')
        era_info[era] = dict(sigma=sigma, sidebands='two-sided' if has_left and has_right else 'left-only' if has_left else 'right-only',
                            signal_sb_over_sr=signal_sb/signal_sr if signal_sr else None)
        if windows_by_era is None:
            print(f'[REGIONS] {era} M{mass:g}: sigma={sigma:.5g}, {era_info[era]["sidebands"]}, {len(selected)-1} SB bins, S_SB/S_SR={era_info[era]["signal_sb_over_sr"]}')
    raw_sr = sum(r['rates']['sig'] for r in rows
                 if windows_by_era is not None or r['region'] == 'SR')
    if raw_sr <= 0:
        raise ValueError('No signal in the combined signal window.')
    alpha_unit = args.alpha_card_yield/raw_sr
    for row in rows:
        row['rates']['sig'] *= alpha_unit
        row['asimov'] = sum(row['rates'][p] for p in BACKGROUNDS)
        if row['asimov'] <= 0:
            raise ValueError(f'{row["channel"]}: zero total background; rebin this study.')
    groups = {}
    for row in rows:
        group = norm_name('QCD', row['era'], opt.norm_scope)
        groups.setdefault(group, []).append(row)
    diagnostics = {g: separation_diagnostic(rs) for g, rs in groups.items()}
    if windows_by_era is None and any(not d['identifiable'] for d in diagnostics.values()):
        raise ValueError(f'QCD/DY not separately identifiable: {diagnostics}. Widen sidebands or reconsider the model; no automatic regularisation.')
    return dict(schema_version=SCHEMA, blind=True, observation='nominal_background_Asimov',
                target=opt.target, mass=mass, alpha_internal_unit=alpha_unit,
                settings=vars(opt), domain=list(DOMAIN), eras=era_info, rows=rows,
                separation=diagnostics, input_audits=audits,
                uncertainty_model='QCD: shared SS-fit covariance modes + coherent absolute functional envelope; DY source stats independent in disjoint regions; pure QCD/DY normalisation priors replaced by free scales.')


def card_text(model, normalisation_kappa=None, prefix='SB'):
    """Serialize bins; optionally use broad QCD/DY/tt lnN priors for step 1."""
    rows, opt = model['rows'], model['settings']
    columns = [(r, p) for r in rows for p in base.PROCESSES]
    f = lambda x: format(float(x), '.17g')
    def rate(row, process):
        # Absolute, positive yield formula for QCD; additive error at DY=0.
        return 1. if process == 'QCD' or (process == 'DY' and row['rates']['DY'] == 0 and row['dy_stat'] > 0) else row['rates'][process]
    lines = [f'imax {len(rows)}', 'jmax *', 'kmax *', '------------',
             'bin ' + ' '.join(r['channel'] for r in rows),
             'observation ' + ' '.join(f(r['asimov']) for r in rows), '------------',
             'bin ' + ' '.join(r['channel'] for r, p in columns),
             'process ' + ' '.join('Zprime' if p == 'sig' else p for r, p in columns),
             'process ' + ' '.join(str(base.PROC_ID[p]) for r, p in columns),
             'rate ' + ' '.join(f(rate(r, p)) for r, p in columns), '------------',
             '# BLINDED STUDY: all observations are nominal background Asimov.',
             '# Model-based closure is not a test of background modelling bias.',
             '# alpha_internal_unit = '+f(model['alpha_internal_unit']),
             '# alpha_internal_signal_yield = '+f(sum(r['rates']['sig'] for r in rows
                 if normalisation_kappa is not None or r['region'] == 'SR'))]
    lnn = {}
    for index, (row, process) in enumerate(columns):
        for local, values in row['nuisances'].items():
            if local in ('QCD_norm', 'DY_NFModel', 'DY_NFStat'):
                continue  # mass-independent factors replaced by unconstrained beta
            value = values.get(process, '-')
            if value == '-':
                continue
            name = base.nuisance_global_name(local, row['era'])
            if local.endswith('_stat') or local == 'DY_LightJetStat':
                name += '_'+row['region']  # disjoint template-source bins
            lnn.setdefault(name, ['-']*len(columns))[index] = value
    lines.extend(name+' lnN '+' '.join(values) for name, values in sorted(lnn.items()))
    if normalisation_kappa is not None:
        if not math.isfinite(normalisation_kappa) or normalisation_kappa <= 1:
            raise ValueError('Normalisation lnN kappa must exceed one.')
        for era in sorted({r['era'] for r in rows}):
            for process in ('QCD', 'DY', 'tt'):
                name = f'CMS_NPS26009_inSituNorm_{process}_BJetOS_{era}'
                lines.append(name+' lnN '+' '.join(
                    f(normalisation_kappa) if r['era'] == era and p == process else '-'
                    for r, p in columns))
    params = set()
    for row in rows:
        era, channel = row['era'], row['channel']
        if normalisation_kappa is None:
            for process in ('QCD', 'DY'):
                name = norm_name(process, era, opt['norm_scope'])
                lines.append(f'{name} rateParam {channel} {process} 1 [0,{f(opt["norm_max"])}]')
        shape = f'{prefix}_QCDShape_{era}'
        names = [shape] + [f'{prefix}_QCDFitMode{k}_{era}' for k in range(len(row['qcd_modes']))]
        for name in names:
            if name not in params:
                lines.append(f'{name} param 0 1')
                params.add(name)
        terms = [f(row['rates']['QCD']), f(row['qcd_shape'])+'*@0']
        terms += [f(v)+f'*@{k+1}' for k, v in enumerate(row['qcd_modes'])]
        lines.append(f'{prefix}_QCDYield_{channel} rateParam {channel} QCD max(0.,'+ '+'.join(terms)+') '+','.join(names))
        if row['rates']['DY'] == 0 and row['dy_stat'] > 0:
            name = f'{prefix}_DYStat_{channel}'
            lines += [f'{name} param 0 1',
                      f'{prefix}_DYYield_{channel} rateParam {channel} DY max(0.,{f(row["dy_stat"])}*@0) {name}']
    return '\n'.join(lines)+'\n'


def fit_commands(model, directory):
    card = directory/'datacard.txt'
    workspace = directory/'workspace.root'
    masks = ['mask_'+r['channel'] for r in model['rows']]
    values = [name+'='+('1' if r['region'] == 'SR' else '0') for name, r in zip(masks, model['rows'])]
    return [
        ['text2workspace.py', str(card), '--channel-masks', '-m', str(model['mass']), '-o', str(workspace)],
        ['combine', '-M', 'FitDiagnostics', str(workspace), '-m', str(model['mass']),
         '-t', '-1', '--expectSignal', '0', '--setParameters', ','.join(values),
         '--freezeParameters', ','.join(masks), '--skipSBFit', '--robustFit', '1',
         '--cminDefaultMinimizerStrategy', '1', '-n', '.Sideband'],
    ]


def nominal_predictions(text):
    """Independently audit serialized rates/formulae before writing a card.

    Only arithmetic and max() are accepted; no Python eval or arbitrary names.
    Multiple rateParams on one process multiply, as they do in Combine.
    """
    lines = [line.split() for line in text.splitlines() if line and not line.startswith('#')]
    means = {line[0]: float(line[2]) for line in lines if len(line) > 3 and line[1] == 'param'}
    modifiers = {}
    def arithmetic(node):
        if isinstance(node, ast.Expression):
            return arithmetic(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (float, int)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            return arithmetic(node.operand)*(1 if isinstance(node.op, ast.UAdd) else -1)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            left, right = arithmetic(node.left), arithmetic(node.right)
            return left+right if isinstance(node.op, ast.Add) else left*right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'max' and not node.keywords:
            return max(arithmetic(arg) for arg in node.args)
        raise ValueError('Unsupported rateParam arithmetic')
    for line in lines:
        if len(line) < 5 or line[1] != 'rateParam':
            continue
        try:
            value = float(line[4])
        except ValueError:
            formula = line[4]
            parameters = line[5].split(',')
            for index in reversed(range(len(parameters))):
                formula = formula.replace('@'+str(index), '('+str(means[parameters[index]])+')')
            value = arithmetic(ast.parse(formula, mode='eval'))
        key = (line[2], line[3])
        modifiers[key] = modifiers.get(key, 1.)*value
    bins = [line[1:] for line in lines if line[0] == 'bin'][-1]
    processes = [line[1:] for line in lines if line[0] == 'process'][0]
    rates = next(line[1:] for line in lines if line[0] == 'rate')
    totals = {}
    for channel, process, rate in zip(bins, processes, rates):
        if process != 'Zprime':
            totals[channel] = totals.get(channel, 0.)+float(rate)*modifiers.get((channel, process), 1.)
    return totals


def extract_fit(model, directory, samples, seed):
    """Joint covariance draws of the SB-only fit, projected into the masked SR.

    These are Gaussian-covariance propagation intervals, not profile intervals.
    Never sum per-bin/process errors in quadrature. Report bounded draw fraction.
    """
    import ROOT
    wf = ROOT.TFile.Open(str(directory/'workspace.root'))
    ff = ROOT.TFile.Open(str(directory/'fitDiagnostics.Sideband.root'))
    if not wf or wf.IsZombie() or not ff or ff.IsZombie():
        raise ValueError('Missing fit/workspace ROOT file')
    try:
        workspace, fit = wf.Get('w'), ff.Get('fit_b')
        if not workspace or not fit:
            raise ValueError('Missing workspace or sideband fit_b')
        diagnostics = dict(status=int(fit.status()), covQual=int(fit.covQual()), edm=float(fit.edm()))
        if fit.status() != 0 or fit.covQual() != 3:
            (directory/'fit_failure.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
            raise ValueError(f'Sideband fit failed quality checks: {diagnostics}; no plot/result is published.')
        workspace.allVars().assignValueOnly(fit.floatParsFinal())
        workspace.var('r').setVal(0.)
        names = [str(p.GetName()) for p in fit.floatParsFinal()]
        means = np.array([float(p.getVal()) for p in fit.floatParsFinal()])
        cov = np.array([[float(fit.covarianceMatrix()[i][j]) for j in range(len(names))] for i in range(len(names))])
        factor = covariance_factor(cov)
        variables = [workspace.var(n) for n in names]
        if any(not v for v in variables):
            raise ValueError('A fitted parameter is absent from the projection workspace.')
        betas = [n for n in names if n.startswith('beta_')]
        beta_info = {}
        for n in betas:
            i = names.index(n)
            sigma = math.sqrt(max(cov[i,i], 0.))
            low, high = float(variables[i].getMin()), float(variables[i].getMax())
            beta_info[n] = dict(value=float(means[i]), error=sigma,
                                within_two_sigma_of_boundary=means[i]-low < 2*sigma or high-means[i] < 2*sigma)
        diagnostics['normalisations'] = beta_info
        diagnostics['normalisation_correlations'] = {
            a: {b: float(cov[names.index(a),names.index(b)]/math.sqrt(max(cov[names.index(a),names.index(a)]*cov[names.index(b),names.index(b)],1e-300))) for b in betas} for a in betas}
        funcs = {}
        for row in model['rows']:
            fs = []
            for process in BACKGROUNDS:
                name = f'n_exp_bin{row["channel"]}_proc_{process}'
                obj = workspace.function(name) or workspace.var(name)
                if not obj:
                    raise ValueError('Missing expectation '+name)
                fs.append(obj)
            funcs[row['channel']] = fs
        def evaluate():
            result = np.array([[float(f.getVal()) for f in funcs[r['channel']]] for r in model['rows']])
            if not np.all(np.isfinite(result)) or np.min(result) < -1e-8:
                raise ValueError('Non-finite/negative background in covariance propagation.')
            return np.maximum(result, 0.)
        central = evaluate()
        rng = np.random.default_rng(seed)
        draws = np.empty((samples, len(model['rows']), len(BACKGROUNDS)))
        rejected, attempts = 0, 0
        for i in range(samples):
            while True:
                attempts += 1
                if attempts > samples*100:
                    raise ValueError('Covariance draws mostly outside parameter bounds; use profile likelihood or toys.')
                values = means+factor @ rng.normal(size=len(names))
                if any(value < var.getMin() or value > var.getMax() for var, value in zip(variables, values)):
                    rejected += 1
                    continue
                for var, value in zip(variables, values):
                    var.setVal(float(value))
                draws[i] = evaluate()
                break
        diagnostics['draw_rejection_fraction'] = rejected/attempts
        diagnostics['interval_method'] = '16/84 percentiles of joint Gaussian fit-covariance draws conditional on parameter bounds; not profile-likelihood intervals'
        diagnostics['boundary_sensitive'] = rejected/attempts > .05 or any(v['within_two_sigma_of_boundary'] for v in beta_info.values())
        if diagnostics['boundary_sensitive']:
            print('[WARNING] Sideband constraints approach parameter boundaries. Bands are diagnostic; validate with profile scans/toys.')
        def summary(values, central_value):
            lo, hi = np.quantile(values, [.16, .84])
            return dict(value=float(central_value), low=float(lo), high=float(hi))
        result = copy.deepcopy(model)
        for index, row in enumerate(result['rows']):
            row['postfit'] = {p: summary(draws[:,index,j], central[index,j]) for j,p in enumerate(BACKGROUNDS)}
            row['postfit']['total_background'] = summary(draws[:,index,:].sum(axis=1), central[index].sum())
        sr_indices = [i for i,r in enumerate(model['rows']) if r['region'] == 'SR']
        sr_draws = draws[:,sr_indices,:].sum(axis=1)
        sr_central = central[sr_indices,:].sum(axis=0)
        result['combined_sr'] = {p: summary(sr_draws[:,j], sr_central[j]) for j,p in enumerate(BACKGROUNDS)}
        result['combined_sr']['total_background'] = summary(sr_draws.sum(axis=1), sr_central.sum())
        result['combined_sr']['asimov'] = sum(model['rows'][i]['asimov'] for i in sr_indices)
        result['fit'] = diagnostics
        result['note'] = 'SR Asimov points are predictions for illustration and were excluded from this fit. Same-model Asimov closure cannot establish modelling accuracy.'
        path = directory.parent/f'sideband_fit_{model["target"]}_M{base.mass_label(model["mass"])}.json'
        path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        print('[SAVED]', path)
        return result
    finally:
        wf.Close()
        ff.Close()


def model_directory(opt, mass):
    outer = 'full' if opt.outer_nsigma is None else 'outer'+base.mass_label(opt.outer_nsigma)
    tag = f'{opt.target}_exclude{base.mass_label(opt.exclude_nsigma)}sigma_{outer}_bin{base.mass_label(opt.bin_width)}_{opt.norm_scope}'
    return Path(opt.output_base).resolve()/tag/f'M{base.mass_label(mass)}'


def study_signature(opt):
    ignored = {'stage', 'samples', 'seed', 'r_max', 'masses', 'output_base'}
    return {k: v for k, v in vars(opt).items() if k not in ignored}


def invalidate_outputs(directory, target, mass):
    """Never leave an older successful result beside a newly built/failed model."""
    result = directory.parent/f'sideband_fit_{target}_M{base.mass_label(mass)}.json'
    result.unlink(missing_ok=True)
    for name in ('expected_limits.json', 'fit_failure.json', 'workspace.root',
                 'fitDiagnostics.Sideband.root'):
        (directory/name).unlink(missing_ok=True)


def main(argv=None):
    opt = make_parser().parse_args(argv)
    args = configuration(opt)
    reader = None
    try:
        signals = None
        if opt.stage in ('cards', 'all'):
            reader = BlindedReader()
            signals = base.scan_signal_files(args)
        for mass in opt.masses:
            directory = model_directory(opt, mass)
            directory.mkdir(parents=True, exist_ok=True)
            model_path = directory/'model.json'
            if signals is not None:
                invalidate_outputs(directory, opt.target, mass)
                # An unsuccessful rebuild must not permit a later --stage fit
                # to pick up an older model with the same CLI signature.
                model_path.unlink(missing_ok=True)
                (directory/'datacard.txt').unlink(missing_ok=True)
                found = next((key for key in signals if abs(key-mass) < 1e-6), None)
                if found is None:
                    raise ValueError(f'Missing simulated mass {mass}; interpolation is not enabled in this study.')
                model = build_model(reader, args, opt, mass, signals[found])
                text = card_text(model)
                nominal = nominal_predictions(text)
                for row in model['rows']:
                    if not math.isclose(nominal[row['channel']], row['asimov'], rel_tol=1e-10, abs_tol=1e-12):
                        raise ValueError('Serialized card nominal disagrees with the Asimov observation.')
                (directory/'datacard.txt').write_text(text)
                model['card_sha256'] = hashlib.sha256(text.encode()).hexdigest()
                model['signature'] = study_signature(opt)
                model_path.write_text(json.dumps(model, indent=2, allow_nan=False)+'\n')
            else:
                model = json.loads(model_path.read_text())
                if model.get('signature') != study_signature(opt) or model.get('blind') is not True:
                    raise ValueError('Configuration differs from the saved blinded model; rebuild cards.')
                if model.get('card_sha256') != hashlib.sha256((directory/'datacard.txt').read_bytes()).hexdigest():
                    raise ValueError('Datacard changed since blinded generation; rebuild cards.')
            commands = fit_commands(model, directory)
            (directory/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
            if opt.stage in ('fit', 'all'):
                # Remove stale published results before attempting a new fit.
                result_path = directory.parent/f'sideband_fit_{opt.target}_M{base.mass_label(mass)}.json'
                result_path.unlink(missing_ok=True)
                for name in ('workspace.root', 'fitDiagnostics.Sideband.root'):
                    (directory/name).unlink(missing_ok=True)
                for index, command in enumerate(commands):
                    base.run_command(command, directory, stdout_path=directory/f'fit_step{index}.log')
                extract_fit(model, directory, opt.samples, opt.seed)
            if opt.stage in ('limits', 'all'):
                (directory/'expected_limits.json').unlink(missing_ok=True)
                # Build the ORIGINAL, unmasked joint card each time. Never use
                # fitted beta values/errors as a second constraint on this model.
                outputs = base.run_asymptotic(args, directory/'datacard.txt', opt.target,
                                              base.mass_label(mass), mass, directory)
                import ROOT
                handle = ROOT.TFile.Open(str(outputs.expected))
                tree = handle.Get('limit') if handle else None
                if not tree:
                    raise ValueError('Missing expected-limit tree')
                expected = {f'{float(row.quantileExpected):.3f}': float(row.limit) for row in tree if row.quantileExpected >= 0}
                handle.Close()
                if len(expected) != 5:
                    raise ValueError('Expected five expected-limit quantiles')
                (directory/'expected_limits.json').write_text(json.dumps(dict(r=expected,
                    alpha={q: v*model['alpha_internal_unit'] for q,v in expected.items()},
                    model='joint SR+SB; original nuisance constraints; r>=0'), indent=2)+'\n')
            print('[DONE]', directory)
        return 0
    finally:
        if reader is not None:
            reader.close()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, base.WorkflowError, OSError) as error:
        print('ERROR:', error, file=sys.stderr)
        raise SystemExit(1)

