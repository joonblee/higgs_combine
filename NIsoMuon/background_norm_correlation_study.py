#!/usr/bin/env python3
"""Blind NPS-26-009 QCD/DY/ttbar normalisation-correlation study.

Install alongside full_spectrum_fit.py in higgs_combine/NIsoMuon and activate
the production CMSSW/Combine environment. Start from one blinded full-spectrum
batch with completed model.json/datacard.txt pairs. No SKPlotMaker inputs are
needed. The batch and CMSSW must be accessible from the Condor workers.

  python3 background_norm_correlation_study.py submit \
      --source full_spectrum_outputs/condor/<existing-blind-batch> \
      --batch full_spectrum_outputs/condor/norm_correlation --masses 12,40,70
  python3 background_norm_correlation_study.py status --batch <new-batch>

For a larger limit scan, --masses all --fit-masses 12,40,70 runs the limits
for every saved hypothesis and FitDiagnostics only at those three masses.
Run 2 and Run 3 are each internally correlated in the default reference.
Asimov fits measure the model's expected constraints on the normalisations.
The scale intervals transform theta +/- one Hessian standard deviation.
Integrated yield errors use first-order propagation of the full fit covariance
and exclude fluctuation of future observed counts.
"""

import argparse
import csv
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

SCOPES = ('era', 'run', 'global')
MAJOR = ('QCD', 'DY', 'tt')
SOURCE_FILES = ('full_spectrum_fit.py', 'limit_workflow.py',
                'limit_workflow_sideband_fit.py', 'dy_closure_shape.py')
LIMIT_KEYS = ('exp-2', 'exp-1', 'exp0', 'exp+1', 'exp+2')


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate_card(model, card):
    from limit_workflow_sideband_fit import nominal_predictions

    lines = [line.split() for line in card.splitlines()]
    observations = [line[1:] for line in lines if line[:1] == ['observation']]
    bins = [line[1:] for line in lines if line[:1] == ['bin']]
    rows = model['rows']
    channels = [row['channel'] for row in rows]
    asimov = [row['asimov'] for row in rows]
    if len(set(channels)) != len(channels) or len(bins) != 2 or bins[0] != channels:
        raise ValueError('Saved model and card channels differ.')
    if len(observations) != 1 or list(map(float, observations[0])) != asimov:
        raise ValueError('Datacard observations must be the nominal background Asimov counts.')
    nominal = nominal_predictions(card)
    if any(not math.isclose(nominal[c], y, rel_tol=1e-10, abs_tol=1e-12)
           for c, y in zip(channels, asimov)):
        raise ValueError('Datacard nominal prediction differs from the background Asimov counts.')


def regroup_normalisations(model, card, scope):
    """Change only the normalisation nuisance lines named in the saved model."""
    from limit_workflow import run_group, spectrum_normalisation_name

    original = {
        name: dict(spec, eras=spec.get('eras', [spec.get('era')]))
        for name, spec in model['normalisations'].items()
    }
    lines = card.splitlines(keepends=True)
    by_name = {}
    columns, processes = [], []
    for line in lines:
        fields = line.split()
        if fields[:1] == ['bin']:
            columns = fields[1:]
        if fields[:1] == ['process'] and not processes:
            processes = fields[1:]
        if fields and fields[0] in original:
            by_name.setdefault(fields[0], []).append(fields[1:])
    row_era = {row['channel']: row['era'] for row in model['rows']}
    eras = list(model['eras'])
    if set(by_name) != set(original) or set(columns) != set(row_era) or len(columns) != len(processes):
        raise ValueError('Card columns or normalisation nuisances differ from the saved model.')

    grouped = {}
    for era in eras:
        group = {'era': era, 'run': run_group(era), 'global': 'Run2Run3'}[scope]
        grouped.setdefault(group, []).append(era)

    replacement, normalisations = [], {}
    for process in MAJOR:
        previous = {name: spec for name, spec in original.items() if spec['process'] == process}
        era_specs = {}
        parameterisation = {}
        for name, spec in previous.items():
            affected = set(spec['eras'])
            if not affected or not affected <= set(eras) or affected.intersection(era_specs):
                raise ValueError(f'Invalid era coverage for {name}.')
            expected = [p == process and row_era[c] in affected for c, p in zip(columns, processes)]
            records = by_name[name]
            if spec.get('kind', 'lnN') == 'lnN':
                kappa = spec['kappa']
                if not math.isfinite(kappa) or kappa <= 1 or len(records) != 1:
                    raise ValueError(f'Invalid lnN specification for {name}.')
                fields = records[0]
                if fields[0] != 'lnN' or len(fields) != len(expected) + 1 or any(
                        (float(value) != kappa if use else value != '-')
                        for value, use in zip(fields[1:], expected)):
                    raise ValueError(f'Unexpected lnN coefficients in {name}.')
                prescription = ('lnN', kappa)
            elif spec['kind'] == 'rateParam':
                bounds = tuple(spec['bounds'])
                actual = [r[1] for r in records if len(r) == 5 and r[0] == 'rateParam'
                          and r[2:4] == [process, '1']
                          and tuple(map(float, r[4].strip('[]').split(','))) == bounds]
                selected = [c for c, use in zip(columns, expected) if use]
                if len(records) != len(selected) or sorted(actual) != sorted(selected):
                    raise ValueError(f'Unexpected rateParam definition in {name}.')
                prescription = ('rateParam', *bounds)
            else:
                raise ValueError(f'Unsupported normalisation kind in {name}.')
            parameterisation[name] = prescription
            era_specs.update(dict.fromkeys(affected, name))
        if set(era_specs) != set(eras):
            raise ValueError(f'Incomplete {process} normalisation coverage.')

        for group, members in grouped.items():
            from_names = {era_specs[era] for era in members}
            prescriptions = {parameterisation[name] for name in from_names}
            if len(prescriptions) != 1:
                raise ValueError(f'{process} {group} merges different normalisation priors.')
            kind, *values = prescriptions.pop()
            matching = [name for name in from_names if set(previous[name]['eras']) == set(members)]
            name = matching[0] if matching else spectrum_normalisation_name(process, group)
            spec = dict(process=process, run=group, eras=members, kappa=values[0])
            if kind == 'rateParam':
                spec.update(kind=kind, kappa=None, bounds=values)
                lo, hi = values
                replacement.extend(f'{name} rateParam {c} {process} 1 [{lo:.17g},{hi:.17g}]\n'
                                   for c in row_era if row_era[c] in members)
            else:
                coefficients = [format(values[0], '.17g') if p == process and row_era[c] in members
                                else '-' for c, p in zip(columns, processes)]
                replacement.append(f"{name} lnN {' '.join(coefficients)}\n")
            normalisations[name] = spec

    if any(spec['process'] not in MAJOR for spec in original.values()):
        raise ValueError('Unexpected process in the saved normalisation metadata.')
    if set(normalisations) == set(original) and all(
            set(spec['eras']) == set(original[name]['eras']) for name, spec in normalisations.items()):
        return card, normalisations

    retained = [line for line in lines if not line.split() or line.split()[0] not in original]
    remaining = [line.split() for line in retained if line.split()]
    if any(words[0] in normalisations or any(name in token.split(',') for token in words[2:]
                                            for name in original)
           for words in remaining if not words[0].startswith('#')):
        raise ValueError('Normalisation name collision or external reference in the datacard.')
    return ''.join(retained + replacement), normalisations


def mass_list(text):
    if text == 'all':
        return None
    values = tuple(map(float, text.split(',')))
    if any(not math.isfinite(mass) or mass <= 0 for mass in values):
        raise argparse.ArgumentTypeError('Use positive finite masses or all.')
    return set(values)


def find_inputs(source, target, masses):
    import full_spectrum_fit as fs
    from limit_workflow import mass_label

    source = source.resolve(strict=True)
    manifest = source / 'manifest.json'
    if manifest.is_file():
        mode = read_json(manifest).get('options', {}).get('mode')
        if mode not in (None, 'blind'):
            raise ValueError('The source batch must be blinded.')
    results = source / 'results'
    if results.is_dir():
        source = results
    found = {}
    requested = None if masses is None else {mass_label(mass) for mass in masses}
    for path in sorted(source.rglob('model.json')):
        if re.fullmatch(r'M[0-9]+(?:p[0-9]+)?', path.parent.name) is None:
            continue
        label = path.parent.name[1:]
        if requested is not None and label not in requested:
            continue
        model = read_json(path)
        if model.get('target') != target:
            continue
        mass = float(model['mass'])
        if (model.get('blind') is not True or model.get('test_only')
                or model.get('schema_version') != fs.SCHEMA or mass_label(mass) != label):
            raise ValueError(f'Need a blinded production full-spectrum model: {path}.')
        if mass in found:
            raise ValueError(f'Duplicate M{mass:g} under {source}. Select a single fit layout.')
        found[mass] = (path.parent, model)
    if not found or (requested is not None and requested != {mass_label(mass) for mass in found}):
        raise ValueError(f'Missing input models for {target} under {source}. '
                         f'Found masses: {sorted(found)}.')
    reference = next(iter(found.values()))[1]
    for _, model in found.values():
        if (model.get('signature') != reference.get('signature')
                or model.get('fit_kind') != reference.get('fit_kind')
                or set(model['eras']) != set(reference['eras'])):
            raise ValueError('Input models use inconsistent fit configurations.')
    return found


def fit_options(config):
    return argparse.Namespace(mode='blind', inject_r=0, r_max=config['r_max'],
                              fit_components='b-only', fit_output='norms',
                              fit_strategy=config['fit_strategy'],
                              fit_max_calls=config['fit_max_calls'],
                              fit_robust_hesse=config['robust_hesse'])


def yield_summary(directory, model):
    """Propagate the full fit covariance to integrated background yields."""
    from array import array
    import numpy as np
    import ROOT
    from limit_workflow import PROCESSES

    root = directory / 'fit_r0'
    fit_file = ROOT.TFile.Open(str(root / 'fit_b/fitDiagnostics.FullSpectrum.root'))
    workspace_file = ROOT.TFile.Open(str(root / 'workspace.root'))
    try:
        fit = fit_file.Get('fit_b') if fit_file and not fit_file.IsZombie() else None
        workspace = workspace_file.Get('w') if workspace_file and not workspace_file.IsZombie() else None
        if not fit or not workspace:
            raise ValueError('Missing fit_b or workspace for yield-uncertainty extraction.')
        processes = [p for p in PROCESSES if p != 'sig']
        rows = model['rows']
        functions = []
        for row in rows:
            for process in processes:
                name = f'n_exp_bin{row["channel"]}_proc_{process}'
                function = workspace.function(name) or workspace.var(name)
                if not function:
                    raise ValueError(f'Missing yield function {name}.')
                functions.append(function)

        def evaluate():
            values = np.asarray(array('d', (float(f.getVal()) for f in functions)))
            return values.reshape(len(rows), len(processes))

        workspace.var('r').setVal(0.)
        prefit = evaluate()
        expected = [[row['rates'][p] for p in processes] for row in rows]
        if not np.allclose(prefit, expected, rtol=1e-9, atol=1e-9):
            raise ValueError('Prefit workspace yields disagree with the saved model.')
        parameters = fit.floatParsFinal()
        workspace.allVars().assignValueOnly(parameters)
        workspace.var('r').setVal(0.)
        postfit = evaluate()
        cov = fit.covarianceMatrix()
        variables = [workspace.var(par.GetName()) for par in parameters]
        size = len(variables)
        covariance = np.array([[float(cov[i][j]) for j in range(size)] for i in range(size)])
        if not np.isfinite(covariance).all() or np.any(covariance.diagonal() < 0):
            raise ValueError('Invalid fit_b covariance.')
        if not np.isfinite(postfit).all() or np.any(postfit < -1e-9):
            raise ValueError('Invalid fit_b background expectation.')
        gradients = np.zeros((*postfit.shape, size))
        for i, var in enumerate(variables):
            if not var:
                raise ValueError(f'Missing workspace nuisance {parameters.at(i).GetName()}.')
            value = float(var.getVal())
            step = 1e-3 * math.sqrt(covariance[i, i])
            if step == 0:
                continue
            upper = min(value + step, float(var.getMax()))
            lower = max(value - step, float(var.getMin()))
            if upper <= lower:
                raise ValueError(f'Cannot vary {var.GetName()} to propagate its fit uncertainty.')
            var.setVal(upper)
            plus = evaluate()
            var.setVal(lower)
            minus = evaluate()
            var.setVal(value)
            gradients[..., i] = (plus - minus) / (upper - lower)
        if not np.isfinite(gradients).all():
            raise ValueError('Nonfinite yield derivative.')

        summary = []
        domains = {'fit_domain': np.ones(len(rows), dtype=bool)}
        if any(row['region'] == 'SR' for row in rows):
            domains['SR'] = np.array([row['region'] == 'SR' for row in rows])
        eras = list(model['eras'])
        for region, mask in domains.items():
            for era in [*eras, 'combined']:
                selected = mask & np.array([era == 'combined' or row['era'] == era for row in rows])
                if not selected.any():
                    continue
                nominal = prefit[selected].sum(axis=0)
                fitted = postfit[selected].sum(axis=0)
                jacobian = gradients[selected].sum(axis=0)
                for name, before, after, grad in zip(
                        [*processes, 'TotalBkg'], [*nominal, nominal.sum()],
                        [*fitted, fitted.sum()], [*jacobian, jacobian.sum(axis=0)]):
                    variance = float(grad @ covariance @ grad)
                    if not math.isfinite(variance) or variance < -1e-10:
                        raise ValueError(f'Invalid propagated variance for {region}/{era}/{name}.')
                    sigma = math.sqrt(max(0., variance))
                    summary.append(dict(region=region, era=era, process=name,
                                        prefit_yield=float(before), postfit_yield=float(after),
                                        postfit_yield_sigma_linear_cov=sigma,
                                        relative_postfit_sigma=sigma/after if after > 0 else None))
        return summary
    finally:
        if fit_file:
            fit_file.Close()
        if workspace_file:
            workspace_file.Close()


def worker(batch, index, *, regenerate=False):
    import full_spectrum_fit as fs

    manifest = read_json(batch / 'manifest.json')
    job = manifest['jobs'][index]
    directory = batch / job['directory']
    with (batch / 'jobs' / f'{index}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = batch / 'jobs' / f'{index}.done'
        if marker.is_file() and not regenerate:
            return
        print(f"[WORKER] {job['kind']} {job['directory']}", flush=True)
        if job['kind'] == 'collect':
            collect(batch, manifest)
        else:
            model = read_json(directory / 'model.json')
            if model['blind'] is not True or model['card_sha256'] != digest(
                    (directory / 'datacard.txt').read_text()):
                raise ValueError('Worker requires an intact blinded card.')
            opt = fit_options(manifest['config'])
            if job['kind'] == 'workspace':
                fs.prepare_workspace(model, directory, opt)
            elif job['kind'] == 'fit_b':
                fs.run_fit_component(model, directory, opt, 'fit_b')
                write_json(directory / 'postfit_yields.json', yield_summary(directory, model))
            elif job['kind'] == 'limits':
                fs.run_expected_limits(model, directory, opt)
            else:
                raise ValueError(f"Unknown worker task {job['kind']}.")
        marker.touch()


def write_table(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_limits(batch, rows, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (upper, lower) = plt.subplots(2, 1, sharex=True, figsize=(7, 6),
                                       gridspec_kw={'height_ratios': [3, 1]})
    styles = {'era': ('tab:blue', 'Independent by era'),
              'run': ('black', 'Correlated within each Run'),
              'global': ('tab:red', 'One shared normalisation')}
    for scope, (colour, label) in styles.items():
        points = [row for row in rows if row['scope'] == scope]
        if not points:
            continue
        mass = [row['mass'] for row in points]
        if scope == reference:
            upper.fill_between(mass, [x['alpha_exp-1'] for x in points],
                               [x['alpha_exp+1'] for x in points], color='0.85',
                               label='Reference 68% expected band')
        upper.plot(mass, [x['alpha_exp0'] for x in points], 'o-', color=colour, label=label)
        lower.plot(mass, [x['ratio_exp0'] for x in points], 'o-', color=colour)
    upper.set_yscale('log')
    upper.set_ylabel(r'Expected 95% CL upper limit on $\alpha$')
    upper.legend(frameon=False)
    lower.axhline(1, color='0.6', linestyle='--')
    lower.set_ylabel(f'Median / {reference}')
    lower.set_xlabel(r'$m_{Z^\prime}$ (GeV)')
    fig.tight_layout()
    for extension in ('pdf', 'png'):
        fig.savefig(batch / f'expected_limit_comparison.{extension}', dpi=160)
    plt.close(fig)


def collect(batch, manifest):
    pending = [i for i, job in enumerate(manifest['jobs'])
               if job['kind'] != 'collect' and not (batch / 'jobs' / f'{i}.done').is_file()]
    if pending:
        raise ValueError(f'{len(pending)} required jobs are still incomplete.')
    reference = manifest['config']['reference']
    root_by_case = {(case['mass'], case['scope']): batch / case['directory']
                    for case in manifest['cases']}
    limits = {key: read_json(path / 'fit_r0/limits/expected_limits.json')
              for key, path in root_by_case.items()}
    limit_rows, norm_rows, cov_rows, yield_rows, fit_rows = [], [], [], [], []
    for case in manifest['cases']:
        mass, scope = case['mass'], case['scope']
        labels = dict(mass=mass, scope=scope, target=manifest['config']['target'])
        directory = root_by_case[mass, scope]
        limit = limits[mass, scope]
        baseline = limits[mass, reference]['alpha']
        if any(not math.isfinite(baseline[key]) or baseline[key] <= 0 for key in LIMIT_KEYS):
            raise ValueError(f'Invalid reference limits at M{mass:g}.')
        limit_rows.append(dict(labels, **{f'{unit}_{key}': limit[unit][key]
                                         for unit in ('r', 'alpha') for key in LIMIT_KEYS},
                               **{f'ratio_{key}': limit['alpha'][key]/baseline[key] for key in LIMIT_KEYS}))
        if not case['fit']:
            continue
        summary = read_json(directory / 'fit_r0/fit_b/fit_summary.json')['fits']['fit_b']
        fit_rows.append(dict(labels, status=summary['status'], covQual=summary['covQual'],
                             edm=summary['edm'], minNll=summary['minNll']))
        normalisations = summary['normalisations']
        for name, info in normalisations.items():
            for era in info['eras']:
                norm_rows.append(dict(labels, era=era, group=info['run'], process=info['process'],
                                      nuisance=name, kind=info.get('kind', 'lnN'), prior_kappa=info['kappa'],
                                      theta=info['theta'], theta_sigma=info['theta_error'],
                                      scale=info['scale'], scale_low=info['scale_low'],
                                      scale_high=info['scale_high'], constraint=info['constraint'],
                                      at_boundary=info.get('at_boundary', False)))
        for a, b in itertools.combinations_with_replacement(normalisations, 2):
            rho = summary['normalisation_correlations'][a][b]
            covariance = rho * normalisations[a]['theta_error'] * normalisations[b]['theta_error']
            cov_rows.append(dict(labels, nuisance_a=a, nuisance_b=b, correlation=rho,
                                 covariance=covariance))
        yield_rows.extend(dict(labels, **row) for row in read_json(directory / 'postfit_yields.json'))
    write_table(batch / 'expected_limit_comparison.csv', limit_rows)
    if fit_rows:
        write_table(batch / 'fit_quality.csv', fit_rows)
        write_table(batch / 'postfit_normalisations.csv', norm_rows)
        write_table(batch / 'normalisation_covariance.csv', cov_rows)
        write_table(batch / 'postfit_yields.csv', yield_rows)
    plot_limits(batch, limit_rows, reference)
    print(f'[COLLECT] {batch}', flush=True)


def stage_batch(args):
    from limit_workflow import mass_label

    batch = args.batch.resolve()
    if batch.exists():
        raise ValueError(f'Batch already exists: {batch}. Choose a new directory.')
    cmssw_base = args.cmssw_base or os.environ.get('CMSSW_BASE')
    arch = args.scram_arch or os.environ.get('SCRAM_ARCH')
    if not cmssw_base or not arch or not (Path(cmssw_base) / 'src').is_dir():
        raise ValueError('Activate CMSSW first, or provide --cmssw-base and --scram-arch.')
    cmssw = Path(cmssw_base).resolve()
    if not args.dry_run and not shutil.which('condor_submit_dag'):
        raise ValueError('condor_submit_dag is unavailable on this host.')

    scopes = tuple(dict.fromkeys(args.scopes.split(',')))
    if any(scope not in SCOPES for scope in scopes) or args.reference not in scopes:
        raise ValueError('Choose scopes from era,run,global and include the reference scope.')
    inputs = find_inputs(args.source, args.target, args.masses)
    fit_labels = None if args.fit_masses is None else {mass_label(m) for m in args.fit_masses}
    if fit_labels is not None and not fit_labels <= {mass_label(m) for m in inputs}:
        raise ValueError('--fit-masses must select a subset of the requested limit masses.')

    prepared = []
    for mass, (source, model) in sorted(inputs.items()):
        card = (source / 'datacard.txt').read_text()
        if digest(card) != model['card_sha256']:
            raise ValueError(f'Source card hash mismatch: {source}.')
        validate_card(model, card)
        for scope in scopes:
            variant, normalisations = regroup_normalisations(model, card, scope)
            if variant != card:
                validate_card(model, variant)
            case = dict(mass=mass, scope=scope, directory=f'results/{scope}/M{mass_label(mass)}',
                        fit=fit_labels is None or mass_label(mass) in fit_labels)
            changed = dict(model, normalisations=normalisations, card_sha256=digest(variant),
                           norm_correlation_study=dict(scope=scope, source=str(source),
                                                      source_card_sha256=model['card_sha256']))
            prepared.append((case, changed, variant))

    snapshot = batch / 'source'
    snapshot.mkdir(parents=True)
    (batch / 'jobs').mkdir()
    script = Path(__file__).resolve()
    for name in (script.name, *SOURCE_FILES):
        shutil.copy2(script.with_name(name), snapshot / name)

    config = dict(target=args.target, reference=args.reference, r_max=args.r_max,
                  fit_strategy=args.fit_strategy, fit_max_calls=args.fit_max_calls,
                  robust_hesse=args.robust_hesse)
    jobs = []

    def add_job(kind, directory, memory, parents=()):
        index = len(jobs)
        jobs.append(dict(kind=kind, directory=directory, memory=memory, parents=list(parents)))
        return index

    for case, model, card in prepared:
        directory = batch / case['directory']
        directory.mkdir(parents=True)
        (directory / 'datacard.txt').write_text(card)
        write_json(directory / 'model.json', model)
        parent = add_job('workspace', case['directory'], args.workspace_memory)
        if case['fit']:
            add_job('fit_b', case['directory'], args.fit_memory, (parent,))
        add_job('limits', case['directory'], args.limit_memory, (parent,))
    parents = [i for i, job in enumerate(jobs) if job['kind'] in ('fit_b', 'limits')]
    add_job('collect', '.', args.limit_memory, parents)
    write_json(batch / 'manifest.json', dict(config=config, jobs=jobs,
                                             cases=[case for case, _, _ in prepared]))

    wrapper = batch / 'run_job.sh'
    wrapper.write_text('\n'.join([
        '#!/bin/bash', 'set -eo pipefail', 'ulimit -s unlimited',
        'export LC_ALL=C ROOT_HIST=0',
        'source /cvmfs/cms.cern.ch/cmsset_default.sh',
        f'export SCRAM_ARCH={shlex.quote(arch)}',
        f'cd {shlex.quote(str(cmssw / "src"))}', 'eval "$(scram runtime -sh)"',
        'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1',
        f'exec python3 -u {shlex.quote(str(snapshot / script.name))} worker '
        f'--batch {shlex.quote(str(batch))} --index "$1"', '',
    ]))
    wrapper.chmod(0o755)
    requirements = f'requirements = {args.requirements}\n' if args.requirements else ''
    (batch / 'submit.jds').write_text(
        'universe = vanilla\nexecutable = run_job.sh\narguments = $(idx)\n'
        'should_transfer_files = YES\nwhen_to_transfer_output = ON_EXIT\n'
        'transfer_output_files = ""\nrequest_cpus = 1\nrequest_memory = $(memory)MB\n'
        'output = jobs/$(idx).out\nerror = jobs/$(idx).err\nlog = jobs/events.log\n'
        f'{requirements}queue 1\n')
    dag = []
    for i, job in enumerate(jobs):
        dag.extend([f'JOB J{i} submit.jds', f'VARS J{i} idx="{i}" memory="{job["memory"]}"'])
        if args.retries:
            dag.append(f'RETRY J{i} {args.retries}')
        if job['parents']:
            names = ' '.join(f'J{j}' for j in job['parents'])
            dag.append(f'PARENT {names} CHILD J{i}')
    (batch / 'workflow.dag').write_text('\n'.join(dag) + '\n')
    counts = {kind: sum(job['kind'] == kind for job in jobs)
              for kind in ('workspace', 'fit_b', 'limits', 'collect')}
    print(f'[PLAN] {batch} {counts}', flush=True)
    command = ['condor_submit_dag']
    if args.max_jobs:
        command += ['-maxjobs', str(args.max_jobs)]
    command.append('workflow.dag')
    if args.dry_run:
        print(f'[DRY RUN] cd {shlex.quote(str(batch))} && {shlex.join(command)}')
    else:
        subprocess.run(command, cwd=batch, check=True)


def positive_float(text):
    value = float(text)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('Use a positive finite number.')
    return value


def nonnegative_int(text):
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError('Use a nonnegative integer.')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    submit = commands.add_parser('submit', help='Stage and submit the comparison DAG.')
    submit.add_argument('--source', type=Path, required=True, help='Existing blind batch or results directory.')
    submit.add_argument('--target', choices=('Run2Run3', 'Run2', 'Run3'), default='Run2Run3')
    submit.add_argument('--masses', type=mass_list, default={12., 40., 70.})
    submit.add_argument('--fit-masses', type=mass_list, default=None)
    submit.add_argument('--scopes', default='era,run,global')
    submit.add_argument('--reference', choices=SCOPES, default='run')
    submit.add_argument('--r-max', type=positive_float, default=100.)
    submit.add_argument('--fit-strategy', type=int, choices=(0, 1, 2), default=1)
    submit.add_argument('--fit-max-calls', type=nonnegative_int)
    submit.add_argument('--robust-hesse', action='store_true', help='Apply the existing audited robustHesse workflow.')
    submit.add_argument('--cmssw-base')
    submit.add_argument('--scram-arch')
    submit.add_argument('--requirements', default='', help='Optional Condor ClassAd expression.')
    submit.add_argument('--workspace-memory', type=int, default=2048, metavar='MB')
    submit.add_argument('--fit-memory', type=int, default=4096, metavar='MB')
    submit.add_argument('--limit-memory', type=int, default=2048, metavar='MB')
    submit.add_argument('--max-jobs', type=nonnegative_int, default=0, help='DAG throttle; 0 leaves it unset.')
    submit.add_argument('--retries', type=nonnegative_int, default=1)
    submit.add_argument('--dry-run', action='store_true', help='Write the DAG without submitting it.')
    status = commands.add_parser('status', help='Show completed jobs and pending job IDs.')
    summary = commands.add_parser('collect', help='Regenerate tables and plots once all analysis jobs finish.')
    internal = commands.add_parser('worker', help='Run one DAG node.')
    internal.add_argument('--index', type=nonnegative_int, required=True)
    for command in (submit, status, summary, internal):
        command.add_argument('--batch', type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == 'submit':
        if min(args.workspace_memory, args.fit_memory, args.limit_memory) < 1:
            parser.error('Memory requests must be positive.')
        if any(char in args.requirements for char in '\r\n'):
            parser.error('--requirements must be a single line.')
        stage_batch(args)
        return

    batch = args.batch.resolve(strict=True)
    if args.command == 'worker':
        worker(batch, args.index)
    else:
        manifest = read_json(batch / 'manifest.json')
        if args.command == 'collect':
            index = next(i for i, job in enumerate(manifest['jobs']) if job['kind'] == 'collect')
            worker(batch, index, regenerate=True)
        else:
            for kind in ('workspace', 'fit_b', 'limits', 'collect'):
                ids = [i for i, job in enumerate(manifest['jobs']) if job['kind'] == kind]
                pending = [i for i in ids if not (batch / 'jobs' / f'{i}.done').is_file()]
                print(f'{kind}: {len(ids) - len(pending)}/{len(ids)} complete')
                for i in pending:
                    print(f"  J{i} {manifest['jobs'][i]['directory']}  jobs/{i}.err")


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f'Error: {error}')
