#!/usr/bin/env python3
"""Step-1 blinded 11--80 GeV spectrum fit with in-situ QCD/DY/tt scales.

Run in the existing CMSSW/Combine environment; see FULL_SPECTRUM_FIT.md.
Consumes the current SKPlotMaker ROOT estimates without reading OS data.root.
Each era is retained and every disjoint mass bin enters the same likelihood.
Existing counting/sideband production commands and results are unchanged.
--dy-shape-model linear adds the common AN linear DY closure shape with an
exact per-era nominal-yield normaliser for every nuisance value. It retains
lnN 1.5 normalisation and source statistics; regenerated DYAux inputs required.
--interpolate-signal uses the counting-yield interpolation from limit_workflow.py,
including SR systematic variations and anchor Sumw2. Sideband/spectrum channels
use corresponding whole-bin anchor-region counts; no whole-histogram signal
morph or fitted signal PDF is constructed. Window layout uses the same SR-only
25-event internal normalisation as the original counting cards, preserving
signal leakage in sidebands. Spectrum layout keeps its full-domain reference.
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
SIDEBAND_EDGES = tuple(float(x) for x in range(11,21)) + (22.,24.,26.,28.,30.,35.,40.,50.,60.,70.,80.)
SIDEBAND_PRESETS = {
    'central': SIDEBAND_EDGES,
    'fine': tuple(sorted((*SIDEBAND_EDGES,
                         *((a+b)/2 for a,b in zip(SIDEBAND_EDGES,SIDEBAND_EDGES[1:]))))),
    'coarse': tuple(float(x) for x in range(11,21)) + (22.,26.,30.,40.,60.,80.),
    'verycoarse': SIDEBAND_EDGES[::2],
}


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    # Stored in fresh manifests; absent in old batches, which keep their paths/reference.
    p.set_defaults(signal_counting_version=2)
    p.add_argument('--interpolate-signal', action='store_true',
                   help='Use limit_workflow counting-yield interpolation for bracketed masses, with corresponding sideband counts; no extrapolation')
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
    p.add_argument('--fit-layout', choices=('spectrum', 'window'), default='spectrum',
                   help='spectrum: existing resolution bins; window: one SR count plus coarse sidebands.')
    p.add_argument('--sr-nsigma', type=shared.finite_positive, default=5.)
    p.add_argument('--sideband-binning', choices=tuple(SIDEBAND_PRESETS), default='central',
                   help='Window-layout sidebands: central has 20 reference bins, fine 40, '
                        'coarse 15 (previous medium15), verycoarse 10 (previous coarse).')
    p.add_argument('--sideband-edges', type=base.parse_float_list,
                   help='Custom reference edges in GeV (overrides central); must span 11--80.')
    p.add_argument('--sideband-refine', type=int, default=1,
                   help='Subdivide each reference sideband interval (2 for binning stability checks).')
    p.add_argument('--sideband-fragment-fraction', type=float, default=.25,
                   help='Merge an SR-adjacent sideband fragment narrower than this fraction of its reference bin.')
    p.add_argument('--bin-sigma', type=shared.finite_positive, default=1.,
                   help='Target bin width / local sigma_m(m); snapped to common native edges.')
    p.add_argument('--fast-mode', action='store_true',
                   help='Test only: multiply the target bin width by four; retain all fit diagnostics and systematics.')
    p.add_argument('--norm-prior', type=shared.finite_positive, default=1.5,
                   help='lnN kappa for each independent process/era QCD, DY and tt scale (>1).')
    p.add_argument('--dy-shape-model', choices=('none','linear'), default='none',
                   help='linear: common yield-preserving AN DY closure shape (requires new SKPlotMaker DYAux); none preserves existing cards.')
    p.add_argument('--dy-shape-uncertainty', type=float, default=0.,
                   help='Explicit review study: independent era/bin DY lnN shifts (e.g. 0.30 gives 0.70/1.30); 0 preserves the existing model.')
    p.add_argument('--dy-normalisation', choices=('prior','free'), default='prior',
                   help='prior: existing DY lnN; free: one unconstrained DY rateParam per era.')
    p.add_argument('--dy-norm-max', type=shared.finite_positive, default=10.,
                   help='Upper numerical bound of a free DY scale; not a prior.')
    p.add_argument('--inject-r', type=float, default=0., help='Asimov signal strength, e.g. 0 or 1.')
    p.add_argument('--r-max', type=shared.finite_positive, default=100.)
    p.add_argument('--expected-limits', action='store_true',
                   help='Also compute blinded expected limits from each workspace; no impacts or toys. Condor writes limit/normalisation CSV tables automatically.')
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
    batch.add_argument('--condor-memory', type=int, default=16000, help='Memory in MB for fits, limits and ROOT summaries (default: 16000).')
    batch.add_argument('--condor-light-memory', type=int, default=2048,
                       help='Memory in MB for preparation, assembly and table jobs (default: 2048).')
    batch.add_argument('--condor-workspace-memory', type=int, default=4096,
                       help='Memory in MB for workspace compilation (default: 4096).')
    batch.add_argument('--condor-retries', type=int, default=2)
    batch.add_argument('--condor-requirements', default='')
    batch.add_argument('--condor-filesystem-domain', default=None,
                       help='Optional extra worker-domain filter; normally unnecessary with the standard transfer setup.')
    batch.add_argument('--condor-cmssw-base')
    batch.add_argument('--condor-scram-arch')
    batch.add_argument('--condor-worker', help=argparse.SUPPRESS)
    batch.add_argument('--condor-index', type=int, help=argparse.SUPPRESS)
    return p


def configuration(opt):
    if getattr(opt, 'expected_limits', False) and (opt.stage == 'cards' or opt.inject_r != 0 or getattr(opt, 'fast_mode', False)):
        raise ValueError('--expected-limits requires stage all/fit, inject-r 0 and a non-test model.')
    if getattr(opt, 'fit_layout', 'spectrum') == 'window':
        validate_sideband_options(opt)
        if getattr(opt, 'fast_mode', False) or opt.bin_sigma != 1.:
            raise ValueError('Window layout uses --sideband-edges/--sideband-refine, not --fast-mode/--bin-sigma.')
    if opt.norm_prior <= 1:
        raise ValueError('--norm-prior must exceed one.')
    dy_unc = getattr(opt, 'dy_shape_uncertainty', 0.)
    if dy_unc and getattr(opt, 'dy_shape_model', 'none') == 'linear':
        raise ValueError('Choose linear DY shape OR independent-bin --dy-shape-uncertainty, not both.')
    if not math.isfinite(dy_unc) or (dy_unc != 0 and not .30 <= dy_unc < 1.):
        raise ValueError('--dy-shape-uncertainty must be zero or in [0.30, 1).')
    if not math.isfinite(getattr(opt,'dy_norm_max',10.)) or getattr(opt, 'dy_norm_max', 10.) <= 1:
        raise ValueError('--dy-norm-max must exceed one.')
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
    if getattr(opt, 'fit_layout', 'spectrum') == 'window':
        args.n_sigma = opt.sr_nsigma
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


def validate_sideband_options(opt):
    edges = sideband_reference(opt)
    if (len(edges) < 2 or any(not math.isfinite(x) for x in edges)
            or edges[0] != DOMAIN[0] or edges[-1] != DOMAIN[1]
            or any(a >= b for a,b in zip(edges, edges[1:]))):
        raise ValueError('Sideband edges must increase strictly from 11 to 80 GeV.')
    if opt.sideband_refine < 1:
        raise ValueError('--sideband-refine must be at least 1.')
    if not math.isfinite(opt.sideband_fragment_fraction) or not 0 <= opt.sideband_fragment_fraction <= .5:
        raise ValueError('Require 0 <= sideband-fragment-fraction <= 0.5.')


def sideband_reference(opt):
    explicit = getattr(opt,'sideband_edges',None)
    preset = getattr(opt,'sideband_binning','central')
    if explicit is not None:
        if preset != 'central':
            raise ValueError('Use either --sideband-edges or --sideband-binning fine/coarse/verycoarse.')
        return list(explicit)
    return list(SIDEBAND_PRESETS[preset])


def window_bins(edges, low, high, reference=SIDEBAND_EDGES, refine=1, fragment_fraction=.25):
    """One whole-bin SR plus disjoint coarse sidebands, covering 11--80 exactly."""
    validate_sideband_options(argparse.Namespace(sideband_edges=reference,
        sideband_refine=refine, sideband_fragment_fraction=fragment_fraction))
    grid = sorted(set(round(float(x),8) for x in edges if DOMAIN[0]-1e-7 <= x <= DOMAIN[1]+1e-7))
    if len(grid) < 3 or grid[0] != DOMAIN[0] or grid[-1] != DOMAIN[1]:
        raise ValueError('Common native histogram edges must include both 11 and 80 GeV.')
    if not all(math.isfinite(x) for x in (low,high)) or not DOMAIN[0] <= low < high <= DOMAIN[1]:
        raise ValueError('Invalid signal window.')
    srlo = grid[max(0,int(np.searchsorted(grid,low+1e-8,side='right'))-1)]
    srhi = grid[min(len(grid)-1,int(np.searchsorted(grid,high-1e-8,side='left')))]
    anchors = sorted(set(min(grid,key=lambda g:abs(g-x))
        for a,b in zip(reference,reference[1:]) for x in np.linspace(a,b,refine+1)))
    left = sorted({DOMAIN[0],srlo,*[x for x in anchors if x < srlo]})
    right = sorted({srhi,DOMAIN[1],*[x for x in anchors if x > srhi]})
    def reference_width(x):
        i = min(len(anchors)-2,max(0,int(np.searchsorted(anchors,x,side='right'))-1))
        return anchors[i+1]-anchors[i]
    if len(left) > 2 and srlo not in anchors and left[-1]-left[-2] < fragment_fraction*reference_width(srlo):
        del left[-2]
    if len(right) > 2 and srhi not in anchors and right[1]-right[0] < fragment_fraction*reference_width(srhi):
        del right[1]
    rows = [dict(region=f'L{i:04d}',low=a,high=b) for i,(a,b) in enumerate(zip(left,left[1:]))]
    rows += [dict(region='SR',low=srlo,high=srhi)]
    rows += [dict(region=f'R{i:04d}',low=a,high=b) for i,(a,b) in enumerate(zip(right,right[1:]))]
    if len(rows) == 1:
        raise ValueError('Signal window leaves no sideband in 11--80 GeV.')
    return rows


def windows_for_mass(windows, mass, opt):
    return windows[base.mass_label(mass)] if getattr(opt,'fit_layout','spectrum') == 'window' else windows


def effective_bin_sigma(opt):
    return opt.bin_sigma * (4. if getattr(opt, 'fast_mode', False) else 1.)


def model_directory(opt, mass):
    if getattr(opt,'fit_layout','spectrum') == 'window':
        geometry = [sideband_reference(opt),opt.sideband_refine,opt.sideband_fragment_fraction]
        code = hashlib.sha256(json.dumps(geometry).encode()).hexdigest()[:10]
        tag = f'{opt.target}_window{base.mass_label(opt.sr_nsigma)}sigma_sb{code}_lnN{base.mass_label(opt.norm_prior)}'
        return Path(opt.output_base).resolve()/(tag+('_signalSR25_v1' if getattr(opt,'signal_counting_version',1)>=2 else '')+dy_study_tag(opt))/f'M{base.mass_label(mass)}'
    tag = f'{opt.target}_binsigma{base.mass_label(effective_bin_sigma(opt))}_lnN{base.mass_label(opt.norm_prior)}'
    if getattr(opt, 'fast_mode', False):
        tag += '_fast'
    return Path(opt.output_base).resolve()/(tag+dy_study_tag(opt))/f'M{base.mass_label(mass)}'


def dy_study_tag(opt):
    tag = '_DYlinearShape_v1' if getattr(opt, 'dy_shape_model', 'none') == 'linear' else ''
    if getattr(opt, 'dy_shape_uncertainty', 0.):
        tag += '_DYshape'+base.mass_label(opt.dy_shape_uncertainty)
    if getattr(opt, 'dy_normalisation', 'prior') == 'free':
        tag += '_DYfree'+base.mass_label(opt.dy_norm_max)
    if getattr(opt,'interpolate_signal',False) and getattr(opt,'signal_counting_version',1)>=2:
        tag += '_signalCounting_v2'
    return tag


def signature(opt):
    ignored = {'stage', 'masses', 'overwrite', 'dry_run', 'inject_r', 'r_max', 'output_base', 'expected_limits',
               'backend', 'bins_per_job', 'fast_mode', 'fit_layout', 'sr_nsigma',
               'sideband_edges', 'sideband_refine', 'sideband_fragment_fraction', 'sideband_binning',
               'dy_shape_uncertainty', 'dy_normalisation', 'dy_norm_max', 'dy_shape_model', 'interpolate_signal', 'signal_counting_version'}
    result = {k: v for k,v in vars(opt).items() if k not in ignored and not k.startswith('condor_')}
    if getattr(opt, 'interpolate_signal', False):
        result['interpolate_signal'] = True
        if getattr(opt,'signal_counting_version',1)>=2:
            result['signal_interpolation_method'] = 'counting_v2'
    # Preserve compatibility with existing full-resolution saved models.
    if getattr(opt, 'fast_mode', False):
        result['fast_mode'] = True
    if getattr(opt,'fit_layout','spectrum') == 'window':
        for name in ('fit_layout','sr_nsigma','sideband_edges','sideband_refine','sideband_fragment_fraction'):
            result[name] = getattr(opt,name)
        result['sideband_edges'] = sideband_reference(opt)
        if getattr(opt,'signal_counting_version',1)>=2:
            result['signal_normalisation_reference'] = 'SR'
    if getattr(opt, 'dy_shape_uncertainty', 0.):
        result['dy_shape_uncertainty'] = opt.dy_shape_uncertainty
    if getattr(opt, 'dy_normalisation', 'prior') == 'free':
        result.update(dy_normalisation='free', dy_norm_max=opt.dy_norm_max)
    if getattr(opt, 'dy_shape_model', 'none') == 'linear':
        result['dy_shape_model'] = 'linear'
    return result


def build_model(reader, args, opt, mass, files, windows):
    # The shared builder requires these only for its original sideband mode.
    study = argparse.Namespace(**vars(opt), norm_scope='era')
    model = shared.build_model(reader, args, study, mass, files, windows_by_era=windows)
    return decorate_model(model, opt, windows)


def decorate_model(model, opt, windows):
    """Apply full-spectrum metadata after all raw bins are assembled."""
    model.update(schema_version=SCHEMA, fit_kind='full_spectrum', settings=vars(opt),
                 test_only=bool(getattr(opt, 'fast_mode', False)),
                 effective_bin_sigma=effective_bin_sigma(opt),
                 signal_normalisation_domain=({era:[next(w['low'] for w in windows[era] if w['region']=='SR'),
                                                    next(w['high'] for w in windows[era] if w['region']=='SR')]
                                               for era in model['eras']}
                     if getattr(opt,'fit_layout','spectrum')=='window' and getattr(opt,'signal_counting_version',1)>=2 else list(DOMAIN)),
                 uncertainty_model=('Step 1: broad QCD/DY/tt normalisations; inherited template '
                    'responses, SS covariance and envelope, DY source and MC statistics. '
                    'Additional DY closure/QCD transport shape studies remain separate.'))
    if getattr(opt,'fit_layout','spectrum') == 'window':
        model['fit_kind'] = 'window_count_with_sidebands'
        model['effective_bin_sigma'] = None
        model['sideband_binning'] = dict(reference_edges=sideband_reference(opt),
            preset=getattr(opt,'sideband_binning','central') if getattr(opt,'sideband_edges',None) is None else 'custom',
            refine=opt.sideband_refine, fragment_fraction=opt.sideband_fragment_fraction,
            sr_nsigma=opt.sr_nsigma, status='proposal_requires_binning_stability_check')
    for era in model['eras']:
        info = model['eras'][era]
        if getattr(opt,'fit_layout','spectrum') == 'window':
            sr = [w for w in windows[era] if w['region'] == 'SR']
            if len(sr) != 1:
                raise ValueError(f'{era}: window layout requires exactly one SR count.')
            info['signal_window'] = [sr[0]['low'],sr[0]['high']]
        else:
            info.pop('sidebands', None)
            info.pop('signal_sb_over_sr', None)
        info['bin_edges'] = [windows[era][0]['low']]+[w['high'] for w in windows[era]]
        info['zero_dy_bins'] = sum(r['rates']['DY'] == 0 for r in model['rows'] if r['era'] == era)
        print(f'[SPECTRUM] {era}: {len(windows[era])} bins, 11--80 GeV; zero-DY bins={info["zero_dy_bins"]}')
    model['normalisations'] = {
        normalisation_name(p,e): dict(process=p, era=e, kappa=opt.norm_prior)
        for e in model['eras'] for p in MAJOR}
    dy_unc = getattr(opt, 'dy_shape_uncertainty', 0.)
    for row in model['rows']:
        row['nuisances'] = {k:v for k,v in row['nuisances'].items()
                            if not k.startswith('CMS_NPS26009_DYClosure_')}
        if dy_unc:
            name = f'CMS_NPS26009_DYClosure_{row["era"]}_{row["region"]}'
            row['nuisances'][name] = {'DY': f'{1-dy_unc:.17g}/{1+dy_unc:.17g}'}
    if getattr(opt, 'dy_normalisation', 'prior') == 'free':
        for era in model['eras']:
            model['normalisations'][normalisation_name('DY',era)] = dict(
                process='DY', era=era, kappa=None, kind='rateParam', bounds=[0.,opt.dy_norm_max])
    model['dy_review_study'] = dict(relative_bin_uncertainty=dy_unc,
        correlation='independent per era and fit bin; all 11--80 GeV bins',
        normalisation=getattr(opt,'dy_normalisation','prior'),
        zero_yield='existing additive DY source statistics retained; relative shifts do not create a nonzero nominal')
    if getattr(opt, 'dy_shape_model', 'none') == 'linear' and dy_unc:
        raise ValueError('Choose linear DY shape OR independent-bin uncertainty, not both')
    if getattr(opt, 'dy_shape_model', 'none') != 'linear':
        model.pop('dy_linear_shape', None)
    if getattr(opt, 'dy_shape_model', 'none') == 'linear':
        import dy_closure_shape as closure
        model['dy_linear_shape'] = dict(configuration=closure.configuration(), eras={})
        for era in model['eras']:
            selected = sorted([r for r in model['rows'] if r['era'] == era], key=lambda r:r['low'])
            if (abs(selected[0]['low']-11.) > 1e-7 or abs(selected[-1]['high']-80.) > 1e-7
                    or any(abs(a['high']-b['low']) > 1e-7 for a,b in zip(selected,selected[1:]))):
                raise ValueError('Linear DY shape requires the complete disjoint 11--80 GeV domain')
            if any('dy_closure_raw_up' not in r or 'dy_closure_metadata' not in r for r in selected):
                raise ValueError('Linear DY shape requires fresh preparation from updated DYAux inputs')
            if any(r['dy_closure_metadata'] != selected[0]['dy_closure_metadata'] for r in selected):
                raise ValueError('Mixed DY closure input versions across prepared bins')
            up, down, audit = closure.templates([r['rates']['DY'] for r in selected],
                                               [r['dy_closure_raw_up'] for r in selected])
            for row, u, d in zip(selected, up, down):
                row.update(dy_closure_up=u, dy_closure_down=d)
            model['dy_linear_shape']['eras'][era] = dict(audit, inputs=selected[0]['dy_closure_metadata'])
        model['uncertainty_model'] += ' Common integral-preserving linear DY closure shape; nominal-zero source statistics unchanged.'
    if dy_unc or getattr(opt,'dy_normalisation','prior')=='free':
        model['uncertainty_model'] += ' Explicit DY review: independent era/bin closure shifts; DY normalisation '+getattr(opt,'dy_normalisation','prior')+'.'
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


def fit_commands(opt, mass, directory, fit_kind=None):
    if fit_kind not in (None, 'fit_b', 'fit_s'):
        raise ValueError('Unknown fit kind.')
    directory = Path(directory).resolve()
    fit_dir = directory/f'fit_r{base.mass_label(opt.inject_r)}'
    commands = [
        ['text2workspace.py', str(fit_dir/'datacard.txt'), '-m', str(mass), '-o', str(fit_dir/'workspace.root')],
        ['combine', '-M', 'FitDiagnostics', str(fit_dir/'workspace.root'), '-m', str(mass),
         '-t', '-1', '--expectSignal', str(opt.inject_r), '--rMin', '0', '--rMax', str(opt.r_max),
         '--robustFit', '1', '--cminDefaultMinimizerStrategy', '1',
         '--saveShapes', '--saveWithUncertainties', '--saveNormalizations', '-n', '.FullSpectrum'],
    ]
    if fit_kind:
        commands[1].append('--skipSBFit' if fit_kind == 'fit_b' else '--skipBOnlyFit')
    return commands


def fit_card_text(text):
    """Use one-bin shapes so Combine evaluates the joint likelihood stably.

    CountingModelBuilder multiplies all Poisson probabilities in ordinary
    space. With a large spectrum this product can underflow even at nominal
    parameters. FAKE selects ShapeBuilder without requiring ROOT templates
    or changing the yields, observations or nuisance definitions.
    """
    shapes = [line.strip() for line in text.splitlines()
              if line.split()[:1] == ['shapes']]
    if shapes:
        if shapes != ['shapes * * FAKE']:
            raise ValueError('Unexpected shapes directive in full-spectrum card.')
        return text
    return 'shapes * * FAKE\n'+text


def poi_diagnostics(fit, name='r'):
    """Keep the reported error, covariance error and profile crossings distinct.

    Near a bounded POI, RooRealVar.getError() need not equal sqrt(V_ii).
    A zero asymmetric error does not identify a measured crossing.
    """
    parameters = fit.floatParsFinal()
    indices = {str(p.GetName()): i for i, p in enumerate(parameters)}
    if name not in indices:
        return {}
    index = indices[name]
    parameter = parameters[index]
    value = float(parameter.getVal())
    variance = float(fit.covarianceMatrix()[index][index])
    if not math.isfinite(variance) or variance < 0:
        raise ValueError('Invalid POI covariance variance: '+name)
    lower, upper = float(parameter.getMin()), float(parameter.getMax())
    lo = float(parameter.getAsymErrorLo()) if parameter.hasAsymError() else 0.
    hi = float(parameter.getAsymErrorHi()) if parameter.hasAsymError() else 0.
    return {
        name: value,
        name+'_error': float(parameter.getError()),  # legacy field, not a profile interval
        name+'_error_method': 'RooRealVar.getError(); use covariance/profile fields for interpretation',
        name+'_covariance_error': math.sqrt(variance),
        name+'_profile_error_low': lo if lo < 0 else None,
        name+'_profile_error_high': hi if hi > 0 else None,
        name+'_profile_crossing_low': value+lo if lo < 0 else None,
        name+'_profile_crossing_high': value+hi if hi > 0 else None,
        name+'_profile_note': 'Missing crossings are null; never interpret a stored zero error as zero uncertainty.',
        name+'_range': [lower, upper],
        name+'_at_boundary': abs(value-lower) < 1e-5 or abs(value-upper) < 1e-5,
    }


def normalisation_diagnostics(parameter, specification, covariance_error=None):
    value = float(parameter.getVal())
    error = float(parameter.getError()) if covariance_error is None else covariance_error
    if specification.get('kind') == 'rateParam':
        low, high = specification['bounds']
        return dict(specification, theta=value, theta_error=error, scale=value,
            scale_low=value-error, scale_high=value+error, constraint=None,
            parameterisation='free multiplicative scale; theta aliases the scale',
            at_boundary=min(abs(value-low),abs(value-high)) < 1e-4,
            hessian_interval_outside_bounds=value-error<low or value+error>high,
            nearest_bound_distance_over_sigma=min(abs(value-low),abs(value-high))/error if error>0 else None,
            interval_method='symmetric Hessian approximation; not a profile interval or a prior constraint')
    kappa = specification['kappa']
    return dict(specification, theta=value, theta_error=error,
        scale=kappa**value, scale_low=kappa**(value-error), scale_high=kappa**(value+error),
        constraint=error, interval_method='transformed symmetric Hessian error; not a profile interval')


def extract_summary(model, directory, opt, fit_kind=None):
    """Report both fits, all norm pulls/constraints and their correlations."""
    import ROOT
    root_dir = directory/f'fit_r{base.mass_label(opt.inject_r)}'
    fit_dir = root_dir/fit_kind if fit_kind else root_dir
    handle = ROOT.TFile.Open(str(fit_dir/'fitDiagnostics.FullSpectrum.root'))
    wf = ROOT.TFile.Open(str(root_dir/'workspace.root'))
    if not handle or handle.IsZombie() or not wf or wf.IsZombie():
        raise ValueError('Missing FitDiagnostics/workspace output.')
    try:
        workspace = wf.Get('w')
        if not workspace:
            raise ValueError('Missing workspace w.')
        result = dict(schema_version=SCHEMA, blind=True, target=model['target'], mass=model['mass'],
                      test_only=model.get('test_only', False),
                      inject_r=opt.inject_r, alpha_internal_unit=model['alpha_internal_unit'],
                      card_sha256=model['card_sha256'],
                      fit_card_sha256=hashlib.sha256((root_dir/'datacard.txt').read_bytes()).hexdigest(),
                      fits={})
        for key in ((fit_kind,) if fit_kind else ('fit_b', 'fit_s')):
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
                variance=float(fit.covarianceMatrix()[indices[name]][indices[name]])
                if not math.isfinite(variance) or variance<0:
                    raise ValueError('Invalid normalisation covariance: '+name)
                info['normalisations'][name] = normalisation_diagnostics(by_name[name], specification, math.sqrt(variance))
            covariance = fit.covarianceMatrix()
            for a in model['normalisations']:
                i = indices[a]
                info['normalisation_correlations'][a] = {}
                for b in model['normalisations']:
                    j = indices[b]
                    denom = math.sqrt(max(float(covariance[i][i])*float(covariance[j][j]), 1e-300))
                    info['normalisation_correlations'][a][b] = float(covariance[i][j])/denom
            if model.get('dy_linear_shape'):
                import dy_closure_shape
                name = dy_closure_shape.NUISANCE
                if name not in by_name:
                    raise ValueError('Missing DY closure shape in validated fit')
                i = indices[name]
                info['dy_linear_shape'] = dict(value=float(by_name[name].getVal()),
                    error=math.sqrt(max(float(covariance[i][i]),0.)), normalisation_correlations={})
                for norm in model['normalisations']:
                    if model['normalisations'][norm]['process'] == 'DY':
                        j = indices[norm]
                        denom = math.sqrt(max(float(covariance[i][i])*float(covariance[j][j]),1e-300))
                        info['dy_linear_shape']['normalisation_correlations'][norm] = float(covariance[i][j])/denom
            # Set every floating parameter, then explicitly set the fixed POI.
            workspace.allVars().assignValueOnly(parameters)
            if key == 'fit_b':
                workspace.var('r').setVal(0.)
            info['r'] = float(workspace.var('r').getVal())
            if 'r' in by_name:
                info.update(poi_diagnostics(fit))
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
    for path in directory.glob('fit_r*/**/*'):
        if path.name in ('workspace.root', 'fitDiagnostics.FullSpectrum.root', 'fit_summary.json', 'fit_failure.json', 'dy_shape_audit.json'):
            path.unlink()


def prepare_inputs(reader, args, opt):
    signals = base.scan_signal_files(args)
    base.prepare_resolution_coefficients(args)
    if not signals:
        raise ValueError('No simulated signal files found.')
    reference = signals[min(signals)]
    is_window = getattr(opt,'fit_layout','spectrum') == 'window'
    windows = {base.mass_label(m):{} for m in opt.masses} if is_window else {}
    for era in base.target_years(opt.target):
        if era not in reference:
            raise ValueError(f'Missing reference signal in {era}.')
        edges = shared.common_edges(reader, args, era, reference[era])
        if is_window:
            for mass in opt.masses:
                low,high = base.mass_window(args,era,mass)
                bins = window_bins(edges,low,high,sideband_reference(opt),opt.sideband_refine,opt.sideband_fragment_fraction)
                windows[base.mass_label(mass)][era] = bins
                print(f'[BINS] {era} M{mass:g}: 1 SR + {len(bins)-1} sideband bins over 11--80 GeV', flush=True)
        else:
            windows[era] = resolution_bins(edges,
                lambda m, era=era: m*base.relative_resolution(args, era, m), effective_bin_sigma(opt))
            print(f'[BINS] {era}: {len(windows[era])} bins over 11--80 GeV', flush=True)
    if getattr(opt, 'interpolate_signal', False):
        import signal_spectrum_interpolation as interpolation
        args.signal_counting_windows = windows if is_window else {}
        args.signal_interpolation_anchors = signals
        args.signal_interpolation_audit = interpolation.preflight(
            reader,args,signals,opt.masses,base.target_years(opt.target))
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


def audit_dy_shape_workspace(model, fit_dir):
    """Verify the compiled Combine expectations, not just the input templates."""
    if not model.get('dy_linear_shape'):
        return
    import ROOT
    import dy_closure_shape
    handle = ROOT.TFile.Open(str(Path(fit_dir)/'workspace.root'))
    if not handle or handle.IsZombie():
        raise ValueError('Cannot audit DY closure workspace')
    try:
        workspace = handle.Get('w')
        theta = workspace.var(dy_closure_shape.NUISANCE) if workspace else None
        if not theta:
            raise ValueError('Compiled DY closure shape parameter is missing')
        totals = {era:sum(r['rates']['DY'] for r in model['rows'] if r['era']==era) for era in model['eras']}
        audit = dict(nuisance=dy_closure_shape.NUISANCE, nominal_totals=totals, points=[])
        original = float(theta.getVal())
        for value in (-3., -1., -.5, 0., .5, 1., 3.):
            theta.setVal(value)
            sums = {era:0. for era in totals}
            for row in model['rows']:
                function = workspace.function(f'n_exp_bin{row["channel"]}_proc_DY')
                if not function:
                    raise ValueError('Missing compiled DY expectation')
                y = float(function.getVal())
                if not math.isfinite(y) or y < -1e-10:
                    raise ValueError('Invalid DY closure expectation')
                sums[row['era']] += y
            if any(not math.isclose(sums[e],totals[e],rel_tol=1e-8,abs_tol=1e-9) for e in totals):
                raise ValueError('Compiled DY closure changes total yield; aborting before fits')
            audit['points'].append(dict(theta=value, totals=sums))
        theta.setVal(original)
        (Path(fit_dir)/'dy_shape_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
    finally:
        handle.Close()


def prepare_workspace(model, directory, opt):
    """Compile once, before either independent hypothesis fit starts."""
    directory = Path(directory)
    fit_dir = directory/f'fit_r{base.mass_label(opt.inject_r)}'
    fit_dir.mkdir(exist_ok=True)
    source_card = (directory/'datacard.txt').read_bytes()
    if hashlib.sha256(source_card).hexdigest() != model['card_sha256']:
        raise ValueError('Datacard changed since generation; rebuild cards.')
    # Keep the assembled card and its provenance immutable, including when
    # reusing a completed batch for a new fit-only submission.
    (fit_dir/'datacard.txt').write_text(fit_card_text(source_card.decode()))
    commands = fit_commands(opt, model['mass'], directory)
    (fit_dir/'commands.json').write_text(json.dumps(commands[:1], indent=2)+'\n')
    for name in ('workspace.root', 'fitDiagnostics.FullSpectrum.root', 'fit_summary.json', 'fit_failure.json', 'dy_shape_audit.json'):
        (fit_dir/name).unlink(missing_ok=True)
    base.run_command(commands[0], fit_dir, stdout_path=fit_dir/'step0.log')
    audit_dy_shape_workspace(model, fit_dir)


def run_fit_component(model, directory, opt, fit_kind):
    if fit_kind not in ('fit_b', 'fit_s'):
        raise ValueError('Unknown fit kind.')
    root_dir = Path(directory)/f'fit_r{base.mass_label(opt.inject_r)}'
    expected_card = fit_card_text((Path(directory)/'datacard.txt').read_text())
    if (root_dir/'datacard.txt').read_text() != expected_card:
        raise ValueError('Compiled fit card differs from the assembled card.')
    fit_dir = root_dir/fit_kind
    fit_dir.mkdir(exist_ok=True)
    command = fit_commands(opt, model['mass'], directory, fit_kind)[1]
    (fit_dir/'commands.json').write_text(json.dumps([command], indent=2)+'\n')
    for name in ('fitDiagnostics.FullSpectrum.root', 'fit_summary.json', 'fit_failure.json'):
        (fit_dir/name).unlink(missing_ok=True)
    base.run_command(command, fit_dir, stdout_path=fit_dir/'step1.log')
    extract_summary(model, Path(directory), opt, fit_kind)


def combine_fit_summaries(model, directory, opt):
    """Collect independently validated results; never refit or average them."""
    root_dir = Path(directory)/f'fit_r{base.mass_label(opt.inject_r)}'
    (root_dir/'fit_summary.json').unlink(missing_ok=True)
    result = None
    fit_card_hash = hashlib.sha256((root_dir/'datacard.txt').read_bytes()).hexdigest()
    expected = dict(schema_version=SCHEMA, blind=True, target=model['target'], mass=model['mass'],
                    test_only=model.get('test_only', False), inject_r=opt.inject_r,
                    alpha_internal_unit=model['alpha_internal_unit'], card_sha256=model['card_sha256'],
                    fit_card_sha256=fit_card_hash)
    for kind in ('fit_b', 'fit_s'):
        part = json.loads((root_dir/kind/'fit_summary.json').read_text())
        if any(part.get(k) != v for k,v in expected.items()) or set(part.get('fits', {})) != {kind}:
            raise ValueError('Incompatible fit summary: '+kind)
        info = part['fits'][kind]
        if info['status'] != 0 or info['covQual'] != 3:
            raise ValueError('Failed fit quality in '+kind)
        if result is None:
            result = dict(expected, fits={}, fit_outputs={})
        result['fits'][kind] = info
        result['fit_outputs'][kind] = str(root_dir/kind/'fitDiagnostics.FullSpectrum.root')
    (root_dir/'fit_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def run_fit(model, directory, opt):
    """Foreground backend retains sequential diagnostics for compatibility."""
    directory = Path(directory)
    prepare_workspace(model, directory, opt)
    fit_dir = Path(directory)/f'fit_r{base.mass_label(opt.inject_r)}'
    command = fit_commands(opt, model['mass'], directory)[1]
    (fit_dir/'commands.json').write_text(json.dumps(fit_commands(opt, model['mass'], directory), indent=2)+'\n')
    base.run_command(command, fit_dir, stdout_path=fit_dir/'step1.log')
    extract_summary(model, directory, opt)


def run_expected_limits(model, directory, opt):
    """Reuse the audited workspace, keeping physical limits separate from diagnostics."""
    if model.get('blind') is not True or model.get('test_only') or opt.inject_r != 0:
        raise ValueError('Expected limits require a non-test background-only Asimov model.')
    fit_dir = Path(directory)/'fit_r0'
    card = fit_dir/'datacard.txt'
    expected_card = fit_card_text((Path(directory)/'datacard.txt').read_text())
    if card.read_text() != expected_card:
        raise ValueError('Compiled fit card differs from the saved model.')
    work = fit_dir/'limits'
    work.mkdir(exist_ok=True)
    destination = work/'expected_limits.json'
    destination.unlink(missing_ok=True)
    args = base.build_parser().parse_args([])
    vars(args).update(mode='blind', parameter='alpha', r_min=0., r_max=opt.r_max, strict=True, jobs=1)
    mass, target = model['mass'], model['target']
    label = base.mass_label(mass)
    result = base.run_asymptotic(args, fit_dir/'workspace.root', target, label, mass, work)
    values = base._read_single_limit_json(result.expected, work, 'full_spectrum_final')
    invalid, reason = base._limit_range_status(values, opt.r_max, .98, True)
    if invalid:
        raise ValueError('Invalid expected limits: '+reason)
    values = {key:float(values[key]) for key in base.EXPECTED_LIMIT_KEYS}
    unit = model['alpha_internal_unit']
    payload = dict(blind=True, mass=mass, target=target, card_sha256=model['card_sha256'],
        fit_card_sha256=hashlib.sha256(card.read_bytes()).hexdigest(),
        workspace_sha256=hashlib.sha256((fit_dir/'workspace.root').read_bytes()).hexdigest(),
        alpha_internal_unit=unit, r=values, alpha={key:value*unit for key,value in values.items()})
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print(f'[LIMIT] M{mass:g}: median alpha={payload["alpha"]["exp0"]:.6g}')
    return [destination, result.expected]


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
    if getattr(opt, 'fast_mode', False):
        print(f'[FAST] Test-only coarse spectrum: target bin width = {effective_bin_sigma(opt):g} sigma; '
              'systematics and fit diagnostics retained.', flush=True)
    if getattr(opt,'fit_layout','spectrum') == 'window':
        print('[LAYOUT] One signal-window count plus coarse sidebands; all regions fitted jointly. '
              f'SR=+/-{opt.sr_nsigma:g} sigma, reference edges={sideband_reference(opt)}, '
              f'refine={opt.sideband_refine}.', flush=True)
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
                import signal_spectrum_interpolation as interpolation
                files = interpolation.requested_files(signals,mass,base.target_years(opt.target),
                                                      getattr(opt,'interpolate_signal',False))
                model = build_model(reader,args,opt,mass,files,windows_for_mass(windows,mass,opt))
                save_model(model, opt, mass, directory)
            else:
                model = load_model(directory,opt)
            fit_dir = directory/f'fit_r{base.mass_label(opt.inject_r)}'
            fit_dir.mkdir(exist_ok=True)
            commands = fit_commands(opt,mass,directory)
            (fit_dir/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
            if opt.stage in ('fit', 'all'):
                run_fit(model, directory, opt)
                if getattr(opt, 'expected_limits', False):
                    run_expected_limits(model, directory, opt)
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

