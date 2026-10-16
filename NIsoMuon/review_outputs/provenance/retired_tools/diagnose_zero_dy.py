#!/usr/bin/env python3
"""Isolated numerical tests of the NPS-26-009 zero-source DY rate parameter.

Run after enabling the SAME Combine/CMSSW environment as production.
No production card, workspace, plot, or Git reference is overwritten.

Examples (from NIsoMuon):
  python3 /path/to/diagnose_zero_dy.py --mass 70 --target Run2Run3
  python3 /path/to/diagnose_zero_dy.py --mass 55 --target Run2Run3
  python3 /path/to/diagnose_zero_dy.py --mass 70 --target Run3

The 'unit' copy performs q = sigma * theta with theta >= 0. It changes the
DY base rate from 1 to sigma, the Gaussian width from sigma to 1, and the
parameter upper bound from U to U/sigma. This is a change of coordinates,
NOT a different uncertainty size or a removal of the physical boundary.
The selected nuisance has the same name but different units in that copy.

S+B impacts use physical r >= 0, as in the production S+B diagnostic.
The nuisance scans retain its Gaussian constraint (-P, NOT promotion to POI).
Limits use pre-fit blinded expectations and matched minimiser/limit settings.
No claim of fit validity is made automatically from an exit code alone.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ERAS = {55: "2023BPix", 70: "2023"}


def prepare_unit_card(text: str, nuisance: str):
    """Accept only the current zero-source, unit-base-rate card form."""
    lines = text.splitlines()
    rows = [line.split('#', 1)[0].split() for line in lines]
    rp = [i for i, r in enumerate(rows) if r[:2] == [nuisance, "rateParam"]]
    gp = [i for i, r in enumerate(rows) if r[:2] == [nuisance, "param"]]
    if len(rp) != 1 or len(gp) != 1:
        raise ValueError(f"Expected exactly one rateParam and param for {nuisance}")
    rr, gg = rows[rp[0]], rows[gp[0]]
    if len(rr) != 6 or len(gg) != 4 or float(rr[4]) != 0 or float(gg[2]) != 0:
        raise ValueError("Not the expected zero-centred additive Gaussian card form")
    sigma = float(gg[3])
    match = re.fullmatch(r"\[([^,]+),([^\]]+)\]", rr[5])
    if not match or not (sigma > 0 and math.isfinite(sigma)):
        raise ValueError("Invalid parameter range or Gaussian width")
    low, upper = map(float, match.groups())
    if low != 0 or not (upper > sigma and math.isfinite(upper)):
        raise ValueError("Expected a nonnegative parameter with an upper bound above sigma")
    rate_rows = [i for i, r in enumerate(rows) if r and r[0] == "rate"]
    if len(rate_rows) != 1:
        raise ValueError("Expected one rate row")
    ir = rate_rows[0]
    bins = [r[1:] for r in rows[:ir] if r and r[0] == 'bin'][-1]
    names = [r[1:] for r in rows[:ir] if r and r[0] == 'process'
             and not all(re.fullmatch(r'[+-]?\d+', s) for s in r[1:])][-1]
    rates = rows[ir][1:]
    if not len(bins) == len(names) == len(rates):
        raise ValueError("Inconsistent rate columns")
    columns = [j for j, (b, p) in enumerate(zip(bins, names)) if (b, p) == (rr[2], rr[3])]
    if len(columns) != 1 or rr[3] != 'DY' or float(rates[columns[0]]) != 1.0:
        raise ValueError("Expected a single affected DY column with base rate exactly 1")
    rates[columns[0]] = f"{sigma:.17g}"
    lines[ir] = 'rate ' + ' '.join(rates)
    lines[rp[0]] = f"{nuisance} rateParam {rr[2]} DY 0 [0,{upper/sigma:.17g}]"
    lines[gp[0]] = f"{nuisance} param 0 1"
    result = '\n'.join(lines) + '\n'
    return result, {'nuisance': nuisance, 'sigma_events': sigma,
                    'original_range': [low, upper], 'unit_range': [0.0, upper/sigma],
                    'bin': rr[2], 'rate_column_zero_based': columns[0]}


def read_tree(path: Path, nuisance: str | None = None, factor: float = 1.0):
    import ROOT
    ROOT.gROOT.SetBatch(True)
    f = ROOT.TFile.Open(str(path))
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open {path}")
    try:
        t = f.Get('limit')
        if not t:
            raise RuntimeError(f"No limit tree in {path}")
        branches = {str(b.GetName()) for b in t.GetListOfBranches()}
        result = []
        for i in range(t.GetEntries()):
            t.GetEntry(i)
            entry = {k: float(getattr(t, k)) for k in
                     ('limit', 'limitErr', 'r', 'deltaNLL', 'quantileExpected') if k in branches}
            if nuisance and nuisance in branches:
                entry['nuisance_coordinate'] = float(getattr(t, nuisance))
                entry['nuisance_in_sigma_units'] = entry['nuisance_coordinate'] / factor
            result.append(entry)
        return result
    finally:
        f.Close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mass', type=int, choices=(55, 70), required=True)
    p.add_argument('--target', choices=('Run3', 'Run2Run3'), default='Run2Run3')
    p.add_argument('--source', type=Path, default=Path.cwd(), help='NIsoMuon directory')
    p.add_argument('--output', type=Path, help='NEW output directory; existing directories are refused')
    p.add_argument('--dry-run', action='store_true', help='Write isolated cards and commands, do not run Combine')
    args = p.parse_args()
    src = args.source.resolve()
    card = src / f'datacards_NIsoMuon/alpha/blind/combined/{args.target}/datacard_M-{args.mass}_{args.target}.txt'
    expected_file = src / f'limit_outputs/alpha/blind/{args.target}/limits_{args.target}_alpha_internal.json'
    expected = json.loads(expected_file.read_text())
    values = [v for k, v in expected.items() if float(k) == args.mass]
    if len(values) != 1 or float(values[0]['exp0']) <= 0:
        raise ValueError("Missing/invalid median expected internal-r limit")
    injection = float(values[0]['exp0'])
    nuisance = f'CMS_NPS26009_LightJetStat_DY_BJetOS_{ERAS[args.mass]}'
    text = card.read_text()
    unit_text, info = prepare_unit_card(text, nuisance)
    if not args.dry_run:
        for cmd in ('combine', 'text2workspace.py'):
            if not shutil.which(cmd):
                raise RuntimeError(f"{cmd} not found: enable your Combine environment first")
        import ROOT  # Test that the result-reader is available before starting fits.
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    output = (args.output or src / 'diagnostics_zero_dy' / stamp / args.target / f'M{args.mass}').resolve()
    if output.exists():
        raise ValueError(f"Refusing to reuse {output}; choose a NEW --output directory")
    output.mkdir(parents=True)
    report = {'source_card': str(card), 'source_card_sha256': hashlib.sha256(card.read_bytes()).hexdigest(),
              'expected_json': str(expected_file), 'target': args.target, 'mass': args.mass,
              'SplusB_injection_r': injection, 'dry_run': args.dry_run, **info, 'jobs': []}
    report_file = output / 'summary.json'
    def save():
        report_file.write_text(json.dumps(report, indent=2) + '\n')
    def run(cmd, cwd, label, method=None, unit=False):
        cmd = list(map(str, cmd))
        log = cwd / f'{label}.log'
        job = {'label': label, 'cwd': str(cwd), 'command': shlex.join(cmd), 'log': str(log)}
        report['jobs'].append(job)
        print(f"[{label}] {shlex.join(cmd)}", flush=True)
        if args.dry_run:
            job['status'] = 'not run'
            save()
            return job
        with log.open('w') as stream:
            completed = subprocess.run(cmd, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False)
        job['exit_code'] = completed.returncode
        log_text = log.read_text(errors='replace')
        job['closed_range_message_count'] = log_text.count('Closed range without finding crossing')
        if method:
            root = cwd / f'higgsCombine.{label}.{method}.mH{args.mass}.root'
            job['root_file'] = str(root)
            if root.is_file():
                try:
                    job['rows'] = read_tree(root, nuisance, 1.0 if unit else info['sigma_events'])
                except Exception as e:
                    job['read_error'] = str(e)
            else:
                job['read_error'] = 'No expected ROOT output'
        save()
        return job
    workspaces = {}
    for mode, content in [('original', text), ('unit', unit_text)]:
        directory = output / mode
        directory.mkdir()
        local_card = directory / 'card.txt'
        local_card.write_text(content)
        ws = directory / 'workspace.root'
        job = run(['text2workspace.py', local_card, '-m', args.mass, '-o', ws], directory, 'text2workspace')
        if not args.dry_run and (job['exit_code'] != 0 or not ws.is_file()):
            raise RuntimeError(f"Workspace creation failed; see {job['log']}")
        workspaces[mode] = ws
    # A: same S+B Asimov model, with default vs tighter interval settings.
    tight = ['--setCrossingTolerance', '1e-8', '--cminDefaultMinimizerStrategy', '1',
             '--cminDefaultMinimizerTolerance', '1e-4', '--setRobustFitStrategy', '1',
             '--setRobustFitTolerance', '1e-4']
    for mode, accuracy in [('original', 'default'), ('original', 'tight'), ('unit', 'default')]:
        ws = workspaces[mode]
        label = f'impact_{mode}_{accuracy}'
        cmd = ['combine', '-M', 'MultiDimFit', ws, '-m', args.mass,
               '--algo', 'impact', '--redefineSignalPOIs', 'r', '-P', nuisance,
               '--floatOtherPOIs', '1', '--saveInactivePOI', '1', '--robustFit', '1',
               '-t', '-1', '--expectSignal', f'{injection:.17g}',
               '--rMin', '0', '--rMax', '100', '-v', '2', '-n', f'.{label}']
        if accuracy == 'tight':
            cmd += tight
        run(cmd, ws.parent, label, 'MultiDimFit', mode == 'unit')
    # B: scan the UNIT nuisance, retaining its Gaussian constraint.
    ws = workspaces['unit']
    for dataset, rinj in [('Bonly', 0.0), ('SplusB', injection)]:
        label = f'scan_unit_{dataset}'
        cmd = ['combine', '-M', 'MultiDimFit', ws, '-m', args.mass, '--algo', 'grid',
               '-P', nuisance, '--floatOtherPOIs', '1', '--saveInactivePOI', '1',
               '--points', '61', '--setParameterRanges', f'{nuisance}=0,3',
               '-t', '-1', '--expectSignal', f'{rinj:.17g}', '--rMin', '0', '--rMax', '100',
               '--cminDefaultMinimizerStrategy', '1', '--cminDefaultMinimizerTolerance', '1e-4',
               '-n', f'.{label}']
        run(cmd, ws.parent, label, 'MultiDimFit', True)
    # C: effect on pre-fit expected limits. Identical settings in all three jobs.
    medians = {}
    for mode, frozen in [('original', False), ('unit', False), ('unit', True)]:
        ws = workspaces[mode]
        label = f'limit_{mode}' + ('_frozen' if frozen else '')
        cmd = ['combine', '-M', 'AsymptoticLimits', ws, '-m', args.mass,
               '--run', 'blind', '--rMin', '0', '--rMax', '100',
               '--rAbsAcc', '1e-6', '--rRelAcc', '1e-4',
               '--cminDefaultMinimizerStrategy', '1', '--cminDefaultMinimizerTolerance', '1e-4',
               '-n', f'.{label}']
        if frozen:
            cmd += ['--setParameters', f'{nuisance}=0', '--freezeParameters', nuisance]
        job = run(cmd, ws.parent, label, 'AsymptoticLimits', mode == 'unit')
        for row in job.get('rows', []):
            if abs(row.get('quantileExpected', -999)-0.5) < 1e-5:
                medians[label] = row['limit']
    report['median_expected_r95'] = medians
    original = medians.get('limit_original')
    unit = medians.get('limit_unit')
    frozen = medians.get('limit_unit_frozen')
    if original and unit:
        report['unit_minus_original_relative'] = (unit-original)/original
    if frozen and unit:
        report['frozen_minus_unit_relative'] = (frozen-unit)/unit
    report['interpretation_note'] = (
        'A missing lower crossing at theta=0 is a physical-boundary issue. '
        'Inspect the UPPER interval and deltaNLL scan separately. '
        'Finite output and zero exit codes are not automatic convergence certification. '
        'Small limit differences are only bounded by the achieved numerical stability; '
        'this test does not validate the underlying Gaussian control-sample model or coverage.'
    )
    save()
    print('\nExpected median r95:', json.dumps(medians, indent=2))
    print('Saved:', report_file)
    if not args.dry_run:
        failed = [j for j in report['jobs'] if j.get('exit_code') != 0 or j.get('read_error')]
        if failed:
            print('Some commands/outputs failed. Inspect summary.json and logs.', file=sys.stderr)
            return 1
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)

