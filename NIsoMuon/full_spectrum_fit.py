#!/usr/bin/env python3
"""Step-1 blinded 11--80 GeV spectrum fit with in-situ QCD/DY/tt scales.

Run in the existing CMSSW/Combine environment; see FULL_SPECTRUM_FIT.md.
Consumes the current SKPlotMaker ROOT estimates without reading OS data.root.
Each era is retained and every disjoint mass bin enters the same likelihood.
Existing counting/sideband production commands and results are unchanged.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import limit_workflow as base
import limit_workflow_sideband_fit as shared

DOMAIN = (11., 80.)
SCHEMA = 'NPS26009_full_spectrum_step1_v1'
MAJOR = ('QCD', 'DY', 'tt')


def make_parser():
    import os
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--masses', type=base.parse_float_list, default=[12., 30., 70.])
    p.add_argument('--target', type=shared.single_target, default='Run2Run3')
    p.add_argument('--stage', choices=('cards', 'fit', 'all', 'status', 'resume', 'recover', 'set-domain', 'collect'), default='all')
    p.add_argument('--backend', choices=('local', 'condor'), default='condor',
                   help='Execution backend (default: condor); use local for foreground execution.')
    p.add_argument('--base-dir', default=base.DEFAULT_BASE_DIR)
    p.add_argument('--sigfit-dir', default='./sigfit_inputs')
    p.add_argument('--resolution-map')
    p.add_argument('--region', default=base.DEFAULT_REGION)
    p.add_argument('--trigger', default='')
    p.add_argument('--bin-sigma', type=shared.finite_positive, default=1.,
                   help='Target bin width / local sigma_m(m); snapped to common native edges.')
    p.add_argument('--norm-prior', type=shared.finite_positive, default=1.5,
                   help='lnN kappa for each independent process/era QCD, DY and tt scale (>1).')
    p.add_argument('--inject-r', type=float, default=0., help='Asimov signal strength, e.g. 0 or 1.')
    p.add_argument('--r-max', type=shared.finite_positive, default=100.)
    p.add_argument('--output-base', default='full_spectrum_outputs')
    p.add_argument('--overwrite', action='store_true', help='Rebuild an existing mass directory and invalidate its old fits.')
    p.add_argument('--dry-run', action='store_true', help='Print planned commands without ROOT, template reads or fitting.')
    batch = p.add_argument_group('Condor: independent era/mass-bin preparation, then joint fit')
    batch.add_argument('--bins-per-job', type=int, default=1,
                       help='Consecutive spectrum bins from one era per preparation job (default: 1).')
    batch.add_argument('--condor-dir', help='New batch directory; also identifies a batch for status/resume/recover/collect.')
    batch.add_argument('--condor-recovery-dir', help='New batch directory for --stage recover; the original batch is preserved.')
    batch.add_argument('--condor-dry-run', action='store_true', help='Prepare the real DAG and manifest without submitting (reads nominal axes).')
    batch.add_argument('--condor-maxjobs', type=int, default=0,
                       help='DAG submission cap (default: 0, no cap added by this workflow).')
    batch.add_argument('--condor-memory', type=int, default=4000, help='Memory request per job in MB.')
    batch.add_argument('--condor-retries', type=int, default=2)
    batch.add_argument('--condor-requirements', default='')
    batch.add_argument('--condor-filesystem-domain', default=os.environ.get('NPS26009_CONDOR_FILESYSTEM_DOMAIN'),
                       help='Shared filesystem domain advertised by eligible workers; requires shared inputs/CMSSW.')
    batch.add_argument('--condor-cmssw-base')
    batch.add_argument('--condor-scram-arch')
    batch.add_argument('--condor-worker', help=argparse.SUPPRESS)
    batch.add_argument('--condor-index', type=int, help=argparse.SUPPRESS)
    return p


def configuration(opt):
    if opt.norm_prior <= 1:
        raise ValueError('--norm-prior must exceed one.')
    if not math.isfinite(opt.inject_r) or not 0 <= opt.inject_r < opt.r_max:
        raise ValueError('Require 0 <= inject-r < r-max.')
    if not opt.masses or len(set(opt.masses)) != len(opt.masses) or any(
            not math.isfinite(m) or not 12 <= m <= 70 for m in opt.masses):
        raise ValueError('Use distinct simulated mass hypotheses in 12--70 GeV.')
    if opt.target not in ('Run2', 'Run3', 'Run2Run3', *base.YEARS):
        raise ValueError('Choose full, Run2, Run3, or one era.')
    if opt.region != base.DEFAULT_REGION:
        raise ValueError('This workflow is restricted to the current OS b-jet signal region.')
    args = base.build_parser().parse_args([])
    for name in ('base_dir', 'sigfit_dir', 'resolution_map', 'region', 'trigger', 'r_max'):
        setattr(args, name, getattr(opt, name))
    args.mode, args.backend, args.strict = 'blind', 'local', True
    args.target, args.masses = opt.target, opt.masses
    args.output_base = str(Path(opt.output_base).resolve())
    base.validate_args(args)
    return args


def resolution_bins(edges, resolution, bin_sigma=1.):
    """Partition the entire domain without splitting or double-counting bins.

    The grid depends on the era's resolution curve, never on the tested signal
    mass. Widths target bin_sigma * sigma at the candidate bin centre. A very
    small terminal remainder is merged with its neighbour.
    """
    grid = sorted(set(round(float(x), 8) for x in edges
                      if DOMAIN[0]-1e-7 <= x <= DOMAIN[1]+1e-7))
    if len(grid) < 3 or abs(grid[0]-DOMAIN[0]) > 1e-7 or abs(grid[-1]-DOMAIN[1]) > 1e-7:
        raise ValueError('Common native histogram edges must include both 11 and 80 GeV.')
    if not math.isfinite(bin_sigma) or bin_sigma <= 0:
        raise ValueError('bin-sigma must be finite and positive.')
    cuts, i = [grid[0]], 0
    while i < len(grid)-1:
        start = i
        i += 1
        while i < len(grid)-1:
            midpoint = (grid[start]+grid[i])/2
            sigma = resolution(midpoint)
            if not math.isfinite(sigma) or sigma <= 0:
                raise ValueError('Invalid local mass resolution.')
            if grid[i]-grid[start] >= bin_sigma*sigma:
                break
            i += 1
        cuts.append(grid[i])
    if len(cuts) > 2 and cuts[-1]-cuts[-2] < .5*bin_sigma*resolution((cuts[-1]+cuts[-2])/2):
        del cuts[-2]
    return [dict(region=f'B{i:04d}', low=a, high=b) for i, (a,b) in enumerate(zip(cuts, cuts[1:]))]


def normalisation_name(process, era):
    return f'CMS_NPS26009_inSituNorm_{process}_BJetOS_{era}'


def model_directory(opt, mass):
    tag = f'{opt.target}_binsigma{base.mass_label(opt.bin_sigma)}_lnN{base.mass_label(opt.norm_prior)}'
    return Path(opt.output_base).resolve()/tag/f'M{base.mass_label(mass)}'


def signature(opt):
    ignored = {'stage', 'masses', 'overwrite', 'dry_run', 'inject_r', 'r_max', 'output_base',
               'backend', 'bins_per_job'}
    return {k: v for k,v in vars(opt).items() if k not in ignored and not k.startswith('condor_')}


def build_model(reader, args, opt, mass, files, windows):
    # The shared builder requires these only for its original sideband mode.
    study = argparse.Namespace(**vars(opt), norm_scope='era')
    model = shared.build_model(reader, args, study, mass, files, windows_by_era=windows)
    return decorate_model(model, opt, windows)


def decorate_model(model, opt, windows):
    """Apply full-spectrum metadata after all raw bins are assembled."""
    model.update(schema_version=SCHEMA, fit_kind='full_spectrum', settings=vars(opt),
                 signal_normalisation_domain=list(DOMAIN),
                 uncertainty_model=('Step 1: broad QCD/DY/tt normalisations; inherited template '
                    'responses, SS covariance and envelope, DY source and MC statistics. '
                    'Additional DY closure/QCD transport shape studies remain separate.'))
    for era in model['eras']:
        info = model['eras'][era]
        info.pop('sidebands', None)
        info.pop('signal_sb_over_sr', None)
        info['bin_edges'] = [windows[era][0]['low']]+[w['high'] for w in windows[era]]
        info['zero_dy_bins'] = sum(r['rates']['DY'] == 0 for r in model['rows'] if r['era'] == era)
        print(f'[SPECTRUM] {era}: {len(windows[era])} bins, 11--80 GeV; zero-DY bins={info["zero_dy_bins"]}')
    model['normalisations'] = {
        normalisation_name(p,e): dict(process=p, era=e, kappa=opt.norm_prior)
        for e in model['eras'] for p in MAJOR}
    # A broad prior regularises degenerate shapes; do not claim the data alone
    # separated the processes. Report the three-component nominal information.
    model['separation'] = {}
    for era in model['eras']:
        rows = [r for r in model['rows'] if r['era'] == era]
        design = np.array([[r['rates'][p] for p in MAJOR] for r in rows])
        fisher = design.T @ (design/np.array([r['asimov'] for r in rows])[:,None])
        scale = np.sqrt(np.maximum(np.diag(fisher), 1e-300))
        gram = fisher/np.outer(scale,scale)
        condition = float(np.linalg.cond(gram))
        model['separation'][era] = dict(processes=list(MAJOR), template_cosines=gram.tolist(),
            condition_number=condition if math.isfinite(condition) else None,
            interpretation='Nominal Fisher diagnostic with shape nuisances fixed; priors are excluded.')
    return model


def fit_commands(opt, mass, directory):
    directory = Path(directory).resolve()
    fit_dir = directory/f'fit_r{base.mass_label(opt.inject_r)}'
    return [
        ['text2workspace.py', str(directory/'datacard.txt'), '-m', str(mass), '-o', str(fit_dir/'workspace.root')],
        ['combine', '-M', 'FitDiagnostics', str(fit_dir/'workspace.root'), '-m', str(mass),
         '-t', '-1', '--expectSignal', str(opt.inject_r), '--rMin', '0', '--rMax', str(opt.r_max),
         '--robustFit', '1', '--cminDefaultMinimizerStrategy', '1',
         '--saveShapes', '--saveWithUncertainties', '--saveNormalizations', '-n', '.FullSpectrum'],
    ]


def extract_summary(model, directory, opt):
    """Report both fits, all norm pulls/constraints and their correlations."""
    import ROOT
    fit_dir = directory/f'fit_r{base.mass_label(opt.inject_r)}'
    handle = ROOT.TFile.Open(str(fit_dir/'fitDiagnostics.FullSpectrum.root'))
    wf = ROOT.TFile.Open(str(fit_dir/'workspace.root'))
    if not handle or handle.IsZombie() or not wf or wf.IsZombie():
        raise ValueError('Missing FitDiagnostics/workspace output.')
    try:
        workspace = wf.Get('w')
        if not workspace:
            raise ValueError('Missing workspace w.')
        result = dict(schema_version=SCHEMA, blind=True, target=model['target'], mass=model['mass'],
                      inject_r=opt.inject_r, alpha_internal_unit=model['alpha_internal_unit'],
                      card_sha256=model['card_sha256'], fits={})
        for key in ('fit_b', 'fit_s'):
            fit = handle.Get(key)
            if not fit:
                raise ValueError('Missing '+key)
            parameters = fit.floatParsFinal()
            by_name = {str(p.GetName()):p for p in parameters}
            indices = {str(p.GetName()):i for i,p in enumerate(parameters)}
            info = dict(status=int(fit.status()), covQual=int(fit.covQual()), edm=float(fit.edm()),
                        minNll=float(fit.minNll()), normalisations={}, normalisation_correlations={})
            if not math.isfinite(info['edm']) or not math.isfinite(info['minNll']):
                raise ValueError('Nonfinite fit diagnostics.')
            result['fits'][key] = info
            if info['status'] != 0 or info['covQual'] != 3:
                (fit_dir/'fit_failure.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
                raise ValueError(f'{key} failed fit quality checks: status={info["status"]}, covQual={info["covQual"]}; see fit_failure.json.')
            for name, specification in model['normalisations'].items():
                if name not in by_name:
                    raise ValueError('Normalisation missing from fit: '+name)
                theta, error = float(by_name[name].getVal()), float(by_name[name].getError())
                kappa = specification['kappa']
                info['normalisations'][name] = dict(specification, theta=theta, theta_error=error,
                    scale=kappa**theta, scale_low=kappa**(theta-error), scale_high=kappa**(theta+error),
                    constraint=error, interval_method='transformed symmetric Hessian error; not a profile interval')
            covariance = fit.covarianceMatrix()
            for a in model['normalisations']:
                i = indices[a]
                info['normalisation_correlations'][a] = {}
                for b in model['normalisations']:
                    j = indices[b]
                    denom = math.sqrt(max(float(covariance[i][i])*float(covariance[j][j]), 1e-300))
                    info['normalisation_correlations'][a][b] = float(covariance[i][j])/denom
            # Set every floating parameter, then explicitly set the fixed POI.
            workspace.allVars().assignValueOnly(parameters)
            if key == 'fit_b':
                workspace.var('r').setVal(0.)
            info['r'] = float(workspace.var('r').getVal())
            if 'r' in by_name:
                info['r_error'] = float(by_name['r'].getError())
                info['r_at_boundary'] = abs(info['r']) < 1e-5 or abs(info['r']-opt.r_max) < 1e-5
            info['bins'] = []
            for row in model['rows']:
                yields = {}
                for p in base.PROCESSES:
                    process = 'Zprime' if p == 'sig' else p
                    name = f'n_exp_bin{row["channel"]}_proc_{process}'
                    function = workspace.function(name) or workspace.var(name)
                    if not function:
                        raise ValueError('Missing expectation '+name)
                    value = float(function.getVal())
                    if not math.isfinite(value) or value < -1e-10:
                        raise ValueError('Invalid expected yield '+name)
                    yields[p] = value
                info['bins'].append(dict(era=row['era'], low=row['low'], high=row['high'],
                                         channel=row['channel'], postfit=yields))
            print(f'[FIT] {key}: status={info["status"]}, covQual={info["covQual"]}, r={info["r"]:.6g}')
            for name, v in info['normalisations'].items():
                print(f'  {name}: scale={v["scale"]:.5g}, theta={v["theta"]:.4g} +/- {v["theta_error"]:.4g}')
        (fit_dir/'fit_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
        return result
    finally:
        handle.Close()
        wf.Close()


def load_model(directory, opt):
    missing = [name for name in ('model.json', 'datacard.txt') if not (directory/name).is_file()]
    if missing:
        raise ValueError(f'No complete saved model in {directory} (missing {", ".join(missing)}). '
                         'Card building must finish successfully before --stage fit; '
                         'rerun with --stage all --overwrite using the same configuration.')
    model = json.loads((directory/'model.json').read_text())
    if (model.get('schema_version') != SCHEMA or model.get('signature') != signature(opt)
            or model.get('blind') is not True):
        raise ValueError('Saved model configuration differs; rebuild cards.')
    if hashlib.sha256((directory/'datacard.txt').read_bytes()).hexdigest() != model.get('card_sha256'):
        raise ValueError('Datacard changed since generation; rebuild cards.')
    return model


def invalidate_fits(directory):
    """An old successful fit must not survive as a result of a rebuilt card."""
    for path in directory.glob('fit_r*/*'):
        if path.name in ('workspace.root', 'fitDiagnostics.FullSpectrum.root', 'fit_summary.json', 'fit_failure.json'):
            path.unlink()


def prepare_inputs(reader, args, opt):
    signals = base.scan_signal_files(args)
    base.prepare_resolution_coefficients(args)
    if not signals:
        raise ValueError('No simulated signal files found.')
    reference = signals[min(signals)]
    windows = {}
    for era in base.target_years(opt.target):
        if era not in reference:
            raise ValueError(f'Missing reference signal in {era}.')
        edges = shared.common_edges(reader, args, era, reference[era])
        windows[era] = resolution_bins(edges,
            lambda m, era=era: m*base.relative_resolution(args, era, m), opt.bin_sigma)
        print(f'[BINS] {era}: {len(windows[era])} bins over 11--80 GeV', flush=True)
    return signals, windows


def save_model(model, opt, mass, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    invalidate_fits(directory)
    text = shared.card_text(model, normalisation_kappa=opt.norm_prior, prefix='FS')
    nominal = shared.nominal_predictions(text)
    for row in model['rows']:
        if not math.isclose(nominal[row['channel']], row['asimov'], rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError('Serialized nominal prediction differs from background Asimov.')
    model['card_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    model['signature'] = signature(opt)
    (directory/'datacard.txt').write_text(text)
    (directory/'model.json').write_text(json.dumps(model, indent=2, allow_nan=False)+'\n')


def run_fit(model, directory, opt):
    directory = Path(directory)
    fit_dir = directory/f'fit_r{base.mass_label(opt.inject_r)}'
    fit_dir.mkdir(exist_ok=True)
    commands = fit_commands(opt, model['mass'], directory)
    (fit_dir/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    for name in ('workspace.root', 'fitDiagnostics.FullSpectrum.root', 'fit_summary.json', 'fit_failure.json'):
        (fit_dir/name).unlink(missing_ok=True)
    for i, command in enumerate(commands):
        base.run_command(command, fit_dir, stdout_path=fit_dir/f'step{i}.log')
    extract_summary(model, directory, opt)


def main(argv=None):
    opt = make_parser().parse_args(argv)
    if opt.condor_worker:
        import full_spectrum_condor as batch
        batch.run_worker(Path(opt.condor_worker), opt.condor_index)
        return 0
    if opt.stage in ('status', 'resume', 'recover', 'set-domain', 'collect'):
        import full_spectrum_condor as batch
        return batch.manage(opt)
    args = configuration(opt)
    if opt.dry_run:
        for mass in opt.masses:
            print('[PLAN] blinded full spectrum, era channels, QCD/DY/tt lnN', opt.norm_prior)
            print(json.dumps(fit_commands(opt, mass, model_directory(opt,mass)), indent=2))
        return 0
    if opt.backend == 'condor':
        import full_spectrum_condor as batch
        batch.submit(opt, args)
        return 0
    reader = None
    try:
        signals, windows = None, {}
        if opt.stage in ('cards', 'all'):
            reader = shared.BlindedReader()
            signals, windows = prepare_inputs(reader, args, opt)
        for mass in opt.masses:
            directory = model_directory(opt,mass)
            if signals is not None:
                if directory.exists() and any(directory.iterdir()) and not opt.overwrite:
                    raise ValueError(f'{directory} is not empty; choose a new --output-base or use --overwrite.')
                directory.mkdir(parents=True, exist_ok=True)
                invalidate_fits(directory)
                for name in ('model.json', 'datacard.txt'):
                    (directory/name).unlink(missing_ok=True)
                found = next((m for m in signals if abs(m-mass)<1e-6),None)
                if found is None:
                    raise ValueError(f'No simulated signal at M={mass}; intermediate-mass interpolation is a later step.')
                model = build_model(reader,args,opt,mass,signals[found],windows)
                save_model(model, opt, mass, directory)
            else:
                model = load_model(directory,opt)
            fit_dir = directory/f'fit_r{base.mass_label(opt.inject_r)}'
            fit_dir.mkdir(exist_ok=True)
            commands = fit_commands(opt,mass,directory)
            (fit_dir/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
            if opt.stage in ('fit', 'all'):
                run_fit(model, directory, opt)
            print('[SAVED]', directory)
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

