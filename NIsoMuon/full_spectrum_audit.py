#!/usr/bin/env python3
"""Audit completed blinded fits without minimising or rewriting batch products.

Read saved RooFitResults and SR bands; optionally evaluate SR background sums
in the saved Combine workspace using its full nuisance covariance. Sampling
uses a multivariate Gaussian covariance approximation, not new data toys.
All eras are reported separately; combined SR errors include cross-era terms.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import full_spectrum_fit as fs


def find_summaries(roots):
    paths = set()
    for root in roots:
        root = Path(root).resolve()
        if not root.exists():
            raise ValueError('Missing result path: '+str(root))
        candidates = [root] if root.is_file() else root.rglob('fit_summary.json')
        paths.update(p for p in candidates if p.name == 'fit_summary.json' and p.parent.name.startswith('fit_r'))
    if not paths:
        raise ValueError('No combined fit_r*/fit_summary.json found; check batch status first.')
    return sorted(paths)


def covariance_samples(values, central):
    """RMS about the fitted prediction, matching the saved-band convention."""
    values, central = np.asarray(values), np.asarray(central)
    rms = np.sqrt(np.mean((values-central)**2, axis=0))
    blocks = np.array_split(values, 4)
    block_rms = [np.sqrt(np.mean((block-central)**2, axis=0)).tolist() for block in blocks]
    return dict(rms_about_best_fit=rms.tolist(), mean=values.mean(axis=0).tolist(),
                sample_std=values.std(axis=0, ddof=1).tolist(),
                quantile16=np.quantile(values, .16, axis=0).tolist(),
                quantile84=np.quantile(values, .84, axis=0).tolist(),
                block_rms=block_rms)


def sample_workspace(ROOT, workspace, fit, rows, samples, seed):
    """Joint SR sampling retains process, bin and era correlations."""
    w = workspace
    w.saveSnapshot('audit_initial', w.allVars())
    parameters = fit.floatParsFinal()
    w.allVars().assignValueOnly(parameters)
    functions = []
    for row in rows:
        parts = []
        for process in ('QCD', 'DY', 'tt', 'ST', 'Others'):
            name = f'n_exp_bin{row["channel"]}_proc_{process}'
            function = w.function(name) or w.var(name)
            if not function:
                raise ValueError('Missing workspace expectation '+name)
            parts.append(function)
        functions.append(parts)
    def evaluate():
        values = [sum(float(f.getVal()) for f in parts) for parts in functions]
        if not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError('Invalid sampled total background expectation.')
        return values+[sum(values)]
    central = evaluate()
    # Background functions do not depend on r. Use the nuisance marginal
    # covariance from the joint fit, without imposing a boundary on sampled r.
    indices = [i for i, p in enumerate(parameters) if str(p.GetName()) != 'r']
    variables = [w.var(str(parameters[i].GetName())) for i in indices]
    if not all(variables):
        raise ValueError('Fit nuisance missing from workspace.')
    means = np.array([float(parameters[i].getVal()) for i in indices])
    matrix = fit.covarianceMatrix()
    covariance = np.array([[float(matrix[i][j]) for j in indices] for i in indices])
    factor = fs.shared.covariance_factor(covariance)
    rng = np.random.default_rng(seed)
    draws = rng.standard_normal((samples, len(indices))) @ factor.T + means
    ranges = [(float(v.getMin()), float(v.getMax())) for v in variables]
    outside = 0
    try:
        # Gaussian covariance propagation is untruncated. Expand only the
        # in-memory ranges when necessary; no workspace file is changed.
        for j, (var, (lo, hi)) in enumerate(zip(variables, ranges)):
            outside += int(np.count_nonzero((draws[:, j] < lo) | (draws[:, j] > hi)))
            step = max(1e-6, .01*math.sqrt(max(covariance[j, j], 0.)))
            var.setRange(min(lo, float(draws[:, j].min()), means[j]-step),
                         max(hi, float(draws[:, j].max()), means[j]+step))
        jacobians = []
        for scale in (1., .5):
            jacobian = np.zeros((len(central), len(indices)))
            for j, var in enumerate(variables):
                if not any(function.dependsOn(var) for parts in functions for function in parts):
                    continue
                step = scale*max(1e-6, .01*math.sqrt(max(covariance[j, j], 0.)))
                var.setVal(float(means[j]+step))
                up = np.asarray(evaluate())
                var.setVal(float(means[j]-step))
                down = np.asarray(evaluate())
                var.setVal(float(means[j]))
                jacobian[:, j] = (up-down)/(2*step)
            jacobians.append(jacobian)
        linear_covariances = [j@covariance@j.T for j in jacobians]
        values = []
        for draw in draws:
            for var, value in zip(variables, draw):
                var.setVal(float(value))
            values.append(evaluate())
        report = covariance_samples(values, central)
        report.update(central=central, labels=[r['era'] for r in rows]+['combined'],
                      samples=samples, seed=seed, draws_outside_original_ranges=outside,
                      linearised_error=np.sqrt(np.maximum(np.diag(linear_covariances[0]), 0)).tolist(),
                      linearised_error_half_step=np.sqrt(np.maximum(np.diag(linear_covariances[1]), 0)).tolist(),
                      linearised_covariance=linear_covariances[0].tolist(),
                      linearisation_note='Finite differences at 0.01 nuisance sigma and half step; '
                          'clipped/zero-yield responses can be non-differentiable. Compare with nonlinear sampling.',
                      method='joint Gaussian nuisance covariance; nonlinear background responses retained; not data toys')
        return report
    finally:
        for var, (lo, hi) in zip(variables, ranges):
            var.setRange(lo, hi)
        w.loadSnapshot('audit_initial')


def audit_summary(ROOT, path, samples=0, seed=123456):
    summary = json.loads(path.read_text())
    model_path = path.parent.parent/'model.json'
    model = json.loads(model_path.read_text())
    if summary.get('blind') is not True or model.get('blind') is not True:
        raise ValueError('This audit accepts blinded models only.')
    if summary['card_sha256'] != model['card_sha256']:
        raise ValueError('Summary/model card hashes differ.')
    sr = [row for row in model['rows'] if row['region'] == 'SR']
    if len(sr) != len(model['eras']) or len({r['era'] for r in sr}) != len(sr):
        raise ValueError('Require exactly one SR per era.')
    report = dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  mass=summary['mass'], inject_r=summary['inject_r'], blind=True,
                  sideband_binning=model.get('sideband_binning'), fits={})
    wh = None
    try:
        if samples:
            # Custom Combine classes must be available before deserialising w.
            if ROOT.gSystem.Load('libHiggsAnalysisCombinedLimit') < 0:
                raise ValueError('Workspace resampling requires the existing CMSSW/Combine environment; use --samples 0 for ROOT-only auditing.')
            wh = ROOT.TFile.Open(str(path.parent/'workspace.root'))
            if not wh or wh.IsZombie() or not wh.Get('w'):
                raise ValueError('Cannot read saved workspace.')
        for kind in ('fit_b', 'fit_s'):
            filename = path.parent/kind/'fitDiagnostics.FullSpectrum.root'
            if not filename.exists():
                filename = path.parent/'fitDiagnostics.FullSpectrum.root'
            handle = ROOT.TFile.Open(str(filename))
            try:
                fit = handle.Get(kind) if handle and not handle.IsZombie() else None
                if not fit or fit.status() != 0 or fit.covQual() != 3:
                    raise ValueError(str(filename)+': require status=0, covQual=3.')
                parameters = {str(p.GetName()): p for p in fit.floatParsFinal()}
                for name, value in summary['fits'][kind]['normalisations'].items():
                    if name not in parameters or not math.isclose(float(parameters[name].getVal()), value['theta'], abs_tol=1e-9, rel_tol=1e-8):
                        raise ValueError('Saved summary/ROOT normalisation mismatch: '+name)
                info = dict(status=int(fit.status()), covQual=int(fit.covQual()),
                            edm=float(fit.edm()), poi=fs.poi_diagnostics(fit), sr=[])
                for row in sr:
                    h = handle.Get(f'shapes_{kind}/{row["channel"]}/total_background')
                    if not h or h.GetNbinsX() != 1:
                        raise ValueError('Missing saved one-bin SR background band.')
                    info['sr'].append(dict(era=row['era'], low=row['low'], high=row['high'],
                        background=float(h.GetBinContent(1)), saved_error=float(h.GetBinError(1))))
                if samples:
                    info['resampled_sr'] = sample_workspace(ROOT, wh.Get('w'), fit, sr, samples, seed)
                info['normalisation_correlations'] = summary['fits'][kind]['normalisation_correlations']
                report['fits'][kind] = info
            finally:
                if handle:
                    handle.Close()
        return report
    finally:
        if wh:
            wh.Close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', nargs='+', required=True, help='Batch/result directories or combined summary files.')
    parser.add_argument('--output-base', default='full_spectrum_outputs/audit')
    parser.add_argument('--samples', type=int, default=0, help='0: read saved bands; >=200: joint workspace draws, without refitting.')
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--overwrite', action='store_true')
    opt = parser.parse_args(argv)
    if opt.samples != 0 and opt.samples < 200:
        raise ValueError('--samples must be zero or at least 200.')
    destination = Path(opt.output_base).resolve()
    if destination.exists() and any(destination.iterdir()) and not opt.overwrite:
        raise ValueError('Output directory is not empty; choose a new path or --overwrite.')
    paths = find_summaries(opt.results)
    import ROOT
    ROOT.gROOT.SetBatch(True)
    records = [audit_summary(ROOT, p, opt.samples, opt.seed) for p in paths]
    destination.mkdir(parents=True, exist_ok=True)
    output = destination/'fit_audit.json'
    output.write_text(json.dumps(dict(schema_version=1, blind=True, records=records), indent=2, allow_nan=False)+'\n')
    for record in records:
        poi = record['fits']['fit_s']['poi']
        print('[AUDIT]', record['source'])
        print(f'M{record["mass"]:g} inject_r={record["inject_r"]:g}: '
              f'r={poi["r"]:.6g}, covariance sigma={poi["r_covariance_error"]:.6g}, '
              f'profile upper error={poi["r_profile_error_high"]}')
    print('[SAVED]', output)
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print('ERROR:', error, file=sys.stderr)
        raise SystemExit(1)
