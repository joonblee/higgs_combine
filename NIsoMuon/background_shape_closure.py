#!/usr/bin/env python3
"""MC-only DY light-jet->b-jet and QCD SS->OS shape closure in saved fit bins.

Use completed window-model geometry, retaining one SR per era. Only nominal
NIsoMuon_DYJets_Inclusive.root and NIsoMuon_QCD_Inclusive.root are opened.
Each source shape is matched to its target integral in 11--80 GeV. The
delta-method covariance includes the fact that this factor uses the same MC
bins. Signed weighted yields are retained; undefined ratios are null.
This diagnoses shape transport, not SS-data fit/extrapolation or data closure.
No fit cards, production templates, anchors or nuisance models are changed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import limit_workflow as base

MC_FILES = {'DY': 'NIsoMuon_DYJets_Inclusive.root', 'QCD': 'NIsoMuon_QCD_Inclusive.root'}


def shape_closure(source, target, source_variance, target_variance):
    """Integral matching with full first-order normalisation covariance."""
    s, t, vs, vt = [np.asarray(x, dtype=float) for x in (source, target, source_variance, target_variance)]
    if (s.ndim != 1 or len(s) < 2 or any(x.shape != s.shape for x in (t, vs, vt))
            or not all(np.isfinite(x).all() for x in (s, t, vs, vt))
            or np.any(vs < 0) or np.any(vt < 0)):
        raise ValueError('Invalid weighted MC bin values/variances.')
    S, T = float(s.sum()), float(t.sum())
    if S <= 0 or T <= 0:
        return dict(status='inconclusive_nonpositive_total', source=s.tolist(), target=t.tolist(),
                    source_variance=vs.tolist(), target_variance=vt.tolist())
    a = T/S
    fraction = s/S
    jt = np.eye(len(s))-fraction[:, None]
    js = -a*jt
    covariance = (jt*vt)@jt.T+(js*vs)@js.T
    covariance = (covariance+covariance.T)/2
    prediction = a*s
    jp = a*jt
    prediction_covariance = np.outer(fraction, fraction)*vt.sum()+(jp*vs)@jp.T
    residual = t-prediction
    eigenvalues, vectors = np.linalg.eigh(covariance)
    threshold = max(float(eigenvalues.max()), 1e-300)*1e-10
    if eigenvalues.min() < -threshold:
        raise ValueError('Residual covariance is not positive semidefinite.')
    keep = eigenvalues > threshold
    projection = vectors.T@residual
    rank = int(keep.sum())
    unresolved = float(np.linalg.norm(projection[~keep]))
    chi2 = float(np.sum(projection[keep]**2/eigenvalues[keep])) if rank else None
    if unresolved > 1e-7*max(1., float(np.linalg.norm(residual))):
        chi2 = None
    ratios, ratio_errors, effective = [], [], []
    for i, (si, ti, pi) in enumerate(zip(s, t, prediction)):
        effective.append(dict(source=float(si*si/vs[i]) if vs[i] > 0 else None,
                              target=float(ti*ti/vt[i]) if vt[i] > 0 else None))
        if pi <= 0:
            ratios.append(None)
            ratio_errors.append(None)
            continue
        ratio = ti/pi
        dt = np.full(len(s), -ratio/T)
        dt[i] += 1/pi
        ds = np.full(len(s), ratio/S)
        ds[i] -= ratio/si
        ratios.append(float(ratio))
        ratio_errors.append(math.sqrt(max(0., float((dt*dt)@vt+(ds*ds)@vs))))
    return dict(status='diagnostic', normalisation=a,
        normalisation_error=math.sqrt(float(vt.sum()/S**2+T**2*vs.sum()/S**4)),
        source=s.tolist(), target=t.tolist(), source_variance=vs.tolist(), target_variance=vt.tolist(),
        prediction=prediction.tolist(), prediction_error=np.sqrt(np.maximum(np.diag(prediction_covariance), 0)).tolist(),
        residual=residual.tolist(), residual_covariance=covariance.tolist(),
        ratio=ratios, ratio_error=ratio_errors, effective_counts=effective,
        chi2=chi2, covariance_rank=rank, residual_in_zero_variance_subspace=unresolved,
        note='First-order weighted-MC diagnostic with source/target categories assumed disjoint. '
             'Sparse or cancellation-dominated bins require care; no automatic p-value, pass threshold or nuisance is assigned.')


def model_paths(roots):
    paths = set()
    for root in roots:
        root = Path(root).resolve()
        if not root.exists():
            raise ValueError('Missing model path: '+str(root))
        paths.update([root] if root.is_file() else root.rglob('model.json'))
    if not paths:
        raise ValueError('No model.json found.')
    return sorted(paths)


def validate_model(model):
    if model.get('blind') is not True or model.get('fit_kind') != 'window_count_with_sidebands':
        raise ValueError('Require a blinded window-count-with-sidebands model.')
    if model['settings']['region'] != base.DEFAULT_REGION:
        raise ValueError('Only the current OS POGMedium tight BJet selection is supported.')
    for era in model['eras']:
        rows = sorted((r for r in model['rows'] if r['era'] == era), key=lambda r: r['low'])
        if (not rows or rows[0]['low'] != 11. or rows[-1]['high'] != 80.
                or any(a['high'] != b['low'] for a, b in zip(rows, rows[1:]))
                or any(r['low'] >= r['high'] for r in rows)
                or sum(r['region'] == 'SR' for r in rows) != 1):
            raise ValueError('Invalid disjoint full-domain fit geometry for '+era)


class MCReader:
    def __init__(self, ROOT):
        self.ROOT, self.handles, self.inputs = ROOT, {}, {}

    def histogram(self, filename, region):
        filename = Path(filename).resolve()
        if filename.name not in MC_FILES.values():
            raise ValueError('Only the nominal DY/QCD MC filenames are allowed.')
        if filename not in self.handles:
            stat = filename.stat()
            handle = self.ROOT.TFile.Open(str(filename), 'READ')
            if not handle or handle.IsZombie():
                raise ValueError('Cannot read MC input '+str(filename))
            self.handles[filename] = handle
            self.inputs[str(filename)] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        path = base.hist_path(region)
        h = self.handles[filename].Get(path)
        if not h or h.GetDimension() != 1 or h.GetSumw2N() == 0:
            raise ValueError(f'Missing 1D weighted-MC histogram with Sumw2: {filename}:{path}')
        return h

    def integrate(self, hist, rows):
        import ctypes
        values, variances = [], []
        axis = hist.GetXaxis()
        for row in rows:
            lo, hi = row['low'], row['high']
            first, last = axis.FindFixBin(lo+1e-8), axis.FindFixBin(hi-1e-8)
            if (first < 1 or last > hist.GetNbinsX()
                    or not math.isclose(axis.GetBinLowEdge(first), lo, abs_tol=1e-7, rel_tol=0)
                    or not math.isclose(axis.GetBinUpEdge(last), hi, abs_tol=1e-7, rel_tol=0)):
                raise ValueError(f'MC native binning cannot reproduce saved window [{lo}, {hi}).')
            error = ctypes.c_double(0.)
            values.append(float(hist.IntegralAndError(first, last, error)))
            variances.append(error.value**2)
        return values, variances

    def close(self):
        for handle in self.handles.values():
            handle.Close()

    def check_unchanged(self):
        for path, expected in self.inputs.items():
            stat = Path(path).stat()
            if expected != dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns):
                raise ValueError('MC input changed during closure study: '+path)


def evaluate_model(reader, path, override_base=None):
    model = json.loads(path.read_text())
    validate_model(model)
    root = Path(override_base or model['settings']['base_dir']).expanduser().resolve()
    trigger = model['settings'].get('trigger', '')
    region = model['settings']['region']
    records = []
    for era in model['eras']:
        rows = sorted((r for r in model['rows'] if r['era'] == era), key=lambda r: r['low'])
        directory = root/era/trigger
        for process in ('DY', 'QCD'):
            filename = directory/MC_FILES[process]
            source_region = region.replace('_BJet_', '_LightJet_') if process == 'DY' else region.replace('OS_', 'SS_', 1)
            source, vs = reader.integrate(reader.histogram(filename, source_region), rows)
            target, vt = reader.integrate(reader.histogram(filename, region), rows)
            result = shape_closure(source, target, vs, vt)
            result.update(process=process, era=era, mass=model['mass'],
                source_region=source_region, target_region=region,
                bins=[dict(region=r['region'], low=r['low'], high=r['high']) for r in rows])
            records.append(result)
    return dict(model=str(path), model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                card_sha256=model['card_sha256'], target=model['target'], mass=model['mass'], records=records)


def plot_record(record, directory, geometry):
    if record['status'] != 'diagnostic':
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    bins = record['bins']
    edges = [bins[0]['low']]+[b['high'] for b in bins]
    centres = np.array([(b['low']+b['high'])/2 for b in bins])
    widths = np.diff(edges)
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(7, 6), sharex=True,
                                   gridspec_kw=dict(height_ratios=[3, 1]))
    ax.errorbar(centres, np.array(record['target'])/widths,
                yerr=np.sqrt(record['target_variance'])/widths, fmt='o', label='Target: OS b-jet MC')
    source_label = 'OS light-jet MC (scaled)' if record['process'] == 'DY' else 'SS b-jet MC (scaled)'
    ax.stairs(np.array(record['prediction'])/widths, edges, color='tab:orange', label=source_label)
    ax.errorbar(centres, np.array(record['prediction'])/widths,
                yerr=np.array(record['prediction_error'])/widths, fmt='none', color='tab:orange')
    valid = np.array([v is not None for v in record['ratio']])
    ratio.errorbar(centres[valid], np.array(record['ratio'], dtype=float)[valid],
                   yerr=np.array(record['ratio_error'], dtype=float)[valid], fmt='o')
    ratio.axhline(1, color='grey', linewidth=1)
    sr = next(b for b in bins if b['region'] == 'SR')
    for panel in (ax, ratio):
        panel.axvspan(sr['low'], sr['high'], color='grey', alpha=.12)
    ax.set_ylabel('Weighted events / GeV')
    ratio.set_ylabel('Target / pred.')
    ratio.set_xlabel('Dimuon mass [GeV]')
    ax.set_title(f'{record["process"]} MC shape closure, {record["era"]}, M={record["mass"]:g} GeV')
    ax.legend()
    fig.tight_layout()
    stem = f'{record["process"]}_{record["era"]}_M{record["mass"]:g}_{geometry}'
    for extension in ('pdf', 'png'):
        fig.savefig(directory/(stem+'.'+extension), dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-roots', nargs='+', required=True, help='Saved model files or result directories; their exact geometry is used.')
    parser.add_argument('--base-dir', help='Override the MC base directory recorded in saved models.')
    parser.add_argument('--output-base', default='full_spectrum_outputs/closure_coarse')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    opt = parser.parse_args(argv)
    paths = model_paths(opt.model_roots)
    output = Path(opt.output_base).resolve()
    if output.exists() and any(output.iterdir()) and not opt.overwrite:
        raise ValueError('Output directory is not empty; choose a new path or --overwrite.')
    import ROOT
    ROOT.gROOT.SetBatch(True)
    reader = MCReader(ROOT)
    try:
        results = [evaluate_model(reader, p, opt.base_dir) for p in paths]
        reader.check_unchanged()
        inputs = dict(reader.inputs)
    finally:
        reader.close()
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(schema_version=1, blind=True, mc_only=True, inputs=inputs, models=results,
                   note='Shape transport diagnostic only; no production nuisance or fit model changed.')
    (output/'closure_summary.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    if not opt.no_plots:
        for result in results:
            geometry = result['model_sha256'][:8]
            for record in result['records']:
                plot_record(record, output, geometry)
    for result in results:
        for record in result['records']:
            print(f'[CLOSURE] M{record["mass"]:g} {record["era"]} {record["process"]}: '
                  f'{record["status"]}, chi2={record.get("chi2")}, rank={record.get("covariance_rank")}')
    print('[SAVED]', output/'closure_summary.json')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print('ERROR:', error, file=sys.stderr)
        raise SystemExit(1)
