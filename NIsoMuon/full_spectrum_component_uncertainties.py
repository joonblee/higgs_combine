#!/usr/bin/env python3
"""Plot relative yield uncertainties for total background, QCD, DY and ttbar.

Read a saved full_spectrum model.json and its compiled Combine workspace.
Move each independent constrained nuisance by +/- one prior sigma with all
others nominal, sum era yields first, then combine independent responses in
quadrature. Each process uses its own yield denominator. No observed data,
minimisation, covariance sampling or OS inputs are read. These are prefit prior
responses, not impacts or a postfit covariance decomposition. Free rateParams
have no prior width and are reported separately, never assigned an invented
uncertainty. Groups partition the nuisances rather than adding extra errors.
Combined spectra are coarsened to edges common to all selected eras; no coarse
fit bin is split. Statistics are taken from the existing likelihood model.

Use --impacts with the complete Combine impacts JSON for the same mass and
production. By default the top list alternates between the largest relative
prefit yield shift in any displayed bin and the largest profiled POI impact,
among nuisances that affect that spectrum. --postfit also evaluates the same
list at the fitted nuisance centres and profile endpoints from that JSON,
holding the remaining parameters fixed. Correlated postfit variations cannot
be added in quadrature. The JSON retains both signed yield shifts for each bin.

Example:
 python3 full_spectrum_component_uncertainties.py /path/to/M40/model.json \
     --periods Run2+3 --impacts /path/to/impacts_Run2Run3_alpha_M40.json --top 10
"""
import argparse
import hashlib
import re
import json
import math
from pathlib import Path

from limit_workflow import LUMI_FB, spectrum_channel_labels, spectrum_parameter_name

PROCESSES = ('QCD', 'DY', 'tt')
CATEGORIES = ('Statistics', 'Normalisation', 'Modelling', 'Renormalisation scale',
              'Factorisation scale', 'PDF', 'Strong coupling', 'Top mass', 'b tagging',
              'Jet energy scale', 'Jet energy resolution', 'Muon efficiencies',
              'Muon momentum scale', 'Pileup', 'L1 prefiring', 'Luminosity', 'Other')
RUN2 = ('2016preVFP','2016postVFP','2017','2018')
RUN3 = ('2022','2022EE','2023','2023BPix')


def nuisance_priors(text):
    result, free = {}, set()
    for line in text.splitlines():
        fields = line.split('#',1)[0].split()
        if len(fields) < 2: continue
        name, kind = fields[:2]
        if kind in ('lnN','shape','shapeN2'): result[name] = (0.,1.)
        elif kind == 'param' and len(fields) >= 4:
            result[name] = (float(fields[2]), float(fields[3]))
        elif kind == 'rateParam' and len(fields) >= 5:
            try: float(fields[4]); free.add(name)
            except ValueError: pass  # Derived formula is not an independent parameter.
    return result, free


def category(name, model):
    if name in model.get('normalisations', {}): return 'Normalisation'
    key = name.lower()
    if any(x in key for x in ('mcstat','mc_stat','_stat','stat_','fitmode','ssfitstat')): return 'Statistics'
    if any(x in key for x in ('qcdshape','bckgshape','closurelinear','dyclosure','dyshape','transportshape')): return 'Modelling'
    for token, group in (
        ('qcdscale_ren_', 'Renormalisation scale'), ('qcdscale_fac_', 'Factorisation scale'),
        ('pdf_alphas', 'Strong coupling'), ('pdf_', 'PDF'), ('topmass_', 'Top mass'),
        ('btag_', 'b tagging'), ('cms_scale_j_', 'Jet energy scale'), ('cms_res_j_', 'Jet energy resolution'),
        ('eff_m_', 'Muon efficiencies'), ('cms_scale_m_', 'Muon momentum scale'),
        ('cms_pileup_', 'Pileup'), ('l1_prefiring_', 'L1 prefiring'), ('lumi_', 'Luminosity')):
        if token in key:
            return group
    return 'Other'


def common_groups(rows, eras):
    """Only sum complete model bins; retain common-domain coverage."""
    selected = [r for r in rows if r['era'] in eras]
    edges = []
    for era in eras:
        own = [r for r in selected if r['era'] == era]
        if not own: raise ValueError('No saved model rows for '+era)
        edges.append({round(float(x),8) for r in own for x in (r['low'],r['high'])})
    common = sorted(set.intersection(*edges))
    if len(common) < 2 or common[0] != 11. or common[-1] != 80.:
        raise ValueError('Combined model has no common 11--80 GeV coverage')
    groups = [[i for i,r in enumerate(selected) if r['low'] >= low-1e-8 and r['high'] <= high+1e-8]
              for low,high in zip(common,common[1:])]
    if sorted(i for group in groups for i in group) != list(range(len(selected))):
        raise ValueError('Coarsening would split or omit a model bin')
    return selected, common, groups


def evaluate_workspace(workspace, model, priors, free, eras, *, include_total=False, fitted=None):
    rows, edges, groups = common_groups(model['rows'], eras)
    workspace.saveSnapshot('component_uncertainties_initial', workspace.allVars())
    functions = {}
    for process in ((*PROCESSES, 'ST', 'Others') if include_total else PROCESSES):
        functions[process] = []
        for row in rows:
            name=f'n_exp_bin{row["channel"]}_proc_{process}'
            function=workspace.function(name) or workspace.var(name)
            if not function: raise ValueError('Missing workspace yield '+name)
            functions[process].append(function)
    def yields():
        result={p:[sum(float(functions[p][i].getVal()) for i in group) for group in groups] for p in functions}
        if include_total:
            result['TotalBkg'] = list(map(sum, zip(*result.values())))
            del result['ST'], result['Others']
        if any(not math.isfinite(v) or v < -1e-8 for values in result.values() for v in values):
            raise ValueError('Invalid workspace component prediction')
        return result
    try:
        for name,(mean,width) in priors.items():
            var=workspace.var(name)
            if var: var.setVal(fitted[name]['fit'][1] if fitted is not None else mean)
        for name in free:
            var=workspace.var(name)
            if var: var.setVal(1.)
        central=yields()
        responses={p:{} for p in central}
        used=[]
        for name,(mean,width) in sorted(priors.items()):
            var=workspace.var(name)
            if not var or var.isConstant(): continue
            low, centre, high = fitted[name]['fit'] if fitted is not None else (mean-width, mean, mean+width)
            if width <= 0 or low < var.getMin() or high > var.getMax():
                raise ValueError('Invalid prior/range for '+name)
            var.setVal(high); up=yields()
            var.setVal(low); down=yields()
            var.setVal(centre)
            group=category(name,model); used.append(dict(name=name,category=group,mean=mean,sigma=width))
            for process in central:
                responses[process][name] = dict(category=group,
                    up=[v-n for v,n in zip(up[process],central[process])],
                    down=[v-n for v,n in zip(down[process],central[process])])
        output={}
        for process in central:
            series={}
            for group in ((*CATEGORIES,'Total') if fitted is None else ()):
                members=[r for r in responses[process].values() if group == 'Total' or r['category'] == group]
                up=[math.sqrt(sum(max(r['up'][i],r['down'][i],0.)**2 for r in members)) for i in range(len(groups))]
                down=[math.sqrt(sum(max(-r['up'][i],-r['down'][i],0.)**2 for r in members)) for i in range(len(groups))]
                series[group]=dict(up=[v/n if n > 0 else None for v,n in zip(up,central[process])],
                                   down=[v/n if n > 0 else None for v,n in zip(down,central[process])])
            output[process]=dict(nominal=central[process],relative=series,responses=responses[process])
        return dict(edges=edges,eras=list(eras),processes=output,nuisances=used,
                    unconstrained_parameters=sorted(free),
                    interpretation='Postfit one-at-a-time yield responses at the impact fit centre, other nuisances fixed' if fitted is not None else 'Prefit independent prior responses; era yields summed before shifts; no fit correlations or Poisson observation error; zero nominal => undefined relative error')
    finally:
        workspace.loadSnapshot('component_uncertainties_initial')


def label_payload(payload, names):
    """Publish the same names as the current cards after evaluating the original workspace."""
    for row in payload['nuisances']:
        raw = row['name']
        row.update(name=names[raw], workspace_name=raw)
    for result in payload['processes'].values():
        result['responses'] = {names[key]: value for key, value in result['responses'].items()}


def read_impacts(path, model, priors, free):
    data = json.loads(path.read_text())
    validation = path.parent/'validation.json'
    if validation.exists():
        info = json.loads(validation.read_text())
        if not info.get('passed') or info['mass'] != model['mass'] or info['target'] != model['target']:
            raise ValueError('Impact validation does not match this model')
    mass_tag = re.search(r'_M(\d+(?:p\d+)?)(?:_|\.|$)', path.name)
    if mass_tag and float(mass_tag[1].replace('p','.')) != model['mass']:
        raise ValueError('Impact file is for a different mass hypothesis')
    params = {row['name']:row for row in data['params']}
    if len(params) != len(data['params']) or set(params) != set(priors) or free:
        raise ValueError('Use the full impact JSON and its original constrained workspace/card: nuisance names differ or free rates are present')
    for name, row in params.items():
        lo, centre, hi = row['fit']
        if not all(map(math.isfinite, (lo, centre, hi, row['impact_r']))) or not lo <= centre <= hi:
            raise ValueError('Invalid fit interval for '+name)
        mean, width = priors[name]
        if not all(math.isclose(a,b,abs_tol=1e-6) for a,b in zip(row['prefit'], (mean-width,mean,mean+width))):
            raise ValueError('Prior mismatch for '+name)
    return params


def rank_sources(result, impacts, mode, count):
    records = []
    for name, shifts in result['responses'].items():
        peak = max((abs(delta)/n for side in ('up','down') for delta,n in
                    zip(shifts[side],result['nominal']) if n > 0), default=0.)
        if peak > 0:
            records.append(dict(name=name, peak_relative_prefit=peak,
                                poi_impact=abs(impacts[name]['impact_r']) if impacts else None))
    by_yield = sorted(records,key=lambda r:(-r['peak_relative_prefit'],r['name']))
    by_impact = sorted(records,key=lambda r:(-r['poi_impact'],r['name'])) if impacts else []
    for key, sequence in [('yield_rank',by_yield), ('impact_rank_among_affecting',by_impact)]:
        for i, row in enumerate(sequence,1):
            row[key] = i
    # Alternate the two ordered lists, skipping repeats. The limit applies per process.
    sequence = (by_impact if mode == 'impact' else
                [r for pair in zip(by_yield,by_impact) for r in pair] if by_impact and mode == 'both' else by_yield)
    selected = list(dict.fromkeys(r['name'] for r in sequence))[:count]
    for r in records:
        r['selected'] = r['name'] in selected
    return sorted(records,key=lambda r:(selected.index(r['name']) if r['selected'] else count,r['yield_rank']))


def draw_spectra(payload, directory, period, stage, selected=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    grouped = selected is None
    prefix = 'prefit_relative_uncertainties' if grouped else f'{stage}_ranked'
    for proc, result in payload['processes'].items():
        n = np.asarray(result['nominal'])
        edges = payload['edges']
        curves = []
        bounds = [0.]
        if grouped:
            names = [name for name, sides in result['relative'].items()
                     if name not in ('Total', 'Statistics') and
                     any(x for values in sides.values() for x in values if x is not None)]
        else:
            names = selected[proc]
        # Reserve space in inches for the legend so extra rows never cover the data.
        entries = len(names) + (2 if stage == 'prefit' else 0)
        qcd_zoom = proc == 'QCD'
        header = .80 + .19 * math.ceil(entries / 2) + (.25 if qcd_zoom else 0.)
        height = 5.15 + header
        fig, panels = plt.subplots(1, 2 if qcd_zoom else 1, squeeze=False, figsize=(10, height))
        axes = panels[0]
        ax = axes[-1]
        if qcd_zoom:
            fig.subplots_adjust(wspace=.32)
        fig.subplots_adjust(left=.12, right=.96, bottom=.73/height, top=1-header/height)

        if stage == 'prefit':
            for group, colour, label in [('Total', '0.88', 'Stat. + syst. unc.'),
                                         ('Statistics', '0.73', 'Stat. unc.')]:
                series = result['relative'][group]
                upper = 100*np.asarray(series['up'], dtype=float)
                lower = -100*np.asarray(series['down'], dtype=float)
                curves.extend((lower, upper))
                for panel in axes:
                    panel.stairs(upper, edges, baseline=lower, fill=True, color=colour, label=label)
        colours = ('#D62728', '#1F77B4', '#009E73', '#007D7E', '#B8860B',
                   '#9467BD', '#CC79A7', '#E66101', '#6B6B00', '#333333')
        for i, name in enumerate(names):
            colour = colours[i] if i < len(colours) else plt.get_cmap('tab20')(i % 20)
            for side in ('up','down'):
                if grouped:
                    magnitude = np.asarray(result['relative'][name][side],dtype=float)
                    relative = (100 if side == 'up' else -100) * magnitude
                else:
                    shifts = np.asarray(result['responses'][name][side])
                    relative = np.divide(100*shifts,n,out=np.full(n.shape,np.nan),where=n>0)
                curves.append(relative)
                for panel in axes:
                    panel.stairs(relative, edges, baseline=None, color=colour, linewidth=1.25,
                                label=name.removeprefix('CMS_NPS26009_') if side=='up' else None)
        for curve in curves:
            finite = curve[np.isfinite(curve)]
            if finite.size:
                bounds.extend((float(finite.min()), float(finite.max())))
        lo, hi = min(bounds), max(bounds)
        pad = .08 * (hi-lo) if hi > lo else 1.
        ax.set_ylim(lo-pad, hi+pad)
        for panel in axes:
            panel.axhline(0,color='0.45',linewidth=.7,zorder=0)
            panel.tick_params(which='both', direction='in', top=True, right=True, labelsize=12)
            panel.minorticks_on()
            panel.set_xlim(edges[0],edges[-1])
            panel.set_xlabel(r'$m_{\mu^{+}\mu^{-}}$ [GeV]', loc='right', fontsize=15)
        axes[0].set_ylabel('Relative yield shift [%]', fontsize=15)
        if qcd_zoom:
            zoom = axes[0]
            zoom.set(xlim=(11, 30), xticks=[11,15,20,25,30], ylim=(-50, 50), yticks=[-50,-25,0,25,50],
                     title='Low mass, linear zoom')
            ax.set_yscale('symlog', linthresh=50, linscale=1)
            # Pad in display coordinates to keep both signs visible for asymmetric tails.
            transform = ax.yaxis.get_transform()
            bottom, top = transform.transform([lo, hi])
            margin = .08 * (top-bottom) if top > bottom else 1.
            ax.set_ylim(*transform.inverted().transform([bottom-margin, top+margin]))
            ymin, ymax = ax.get_ylim()
            extent = max(abs(ymin), abs(ymax))
            if extent >= 100:
                decades = [10.**k for k in range(2, 1+math.ceil(math.log10(extent)))]
                ticks = [-t for t in reversed(decades) if -t >= ymin] + [0.] + [t for t in decades if t <= ymax]
                ax.set_yticks(ticks)
            ax.set_title('Full range, symmetric log scale')
        lumi = {'Run2+3': r'138 fb$^{-1}$ (13 TeV) + 62 fb$^{-1}$ (13.6 TeV)',
                'Run2': r'138 fb$^{-1}$ (13 TeV)', 'Run3': r'62 fb$^{-1}$ (13.6 TeV)'}.get(period)
        if lumi is None:
            energy = '13' if period in RUN2 else '13.6'
            lumi = rf'{LUMI_FB[period]:g} fb$^{{-1}}$ ({energy} TeV)'
        fig.text(.12, 1-.30/height, 'CMS', fontsize=20, fontweight='bold', va='baseline')
        fig.text(.205, 1-.30/height, 'Preliminary', fontsize=14, style='italic')
        fig.text(.96, 1-.30/height, lumi, ha='right', fontsize=13)
        process = {'TotalBkg':'Total background','tt':r'$t\bar{t}$'}.get(proc, proc)
        fig.text(.12, 1-.58/height,
                 f"{period}, OS, b-jet   {process}, {stage} uncertainties   M = {payload['mass']:g} GeV",
                 fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = fig.legend(handles, labels, loc='upper center',
                                bbox_to_anchor=(.54, 1-.65/height), ncol=2,
                                frameon=False, borderaxespad=0, fontsize=10,
                                columnspacing=1.6, handlelength=3, labelspacing=.32)
            # Account for the actual legend and QCD panel titles when setting the gap.
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            panel_top = max(panel.get_tightbbox(renderer).y1 for panel in axes)
            gap = .09 * fig.dpi
            offset = legend.get_window_extent(renderer).y0 - gap - panel_top
            fig.subplots_adjust(top=fig.subplotpars.top + offset / fig.bbox.height)
        for ext in ('pdf','png'):
            fig.savefig(directory/f'{prefix}_{period}_{proc}.{ext}')
        plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('model',type=Path,help='model.json saved by full_spectrum_fit.py')
    p.add_argument('--workspace',type=Path,help='Default: model directory/fit_r0/workspace.root')
    p.add_argument('--periods',default='Run2,Run3,Run2+3',help='Comma-separated Run2, Run3, Run2+3 or individual eras')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--impacts', type=Path, help='Combine impacts JSON from the same mass, card and fit production')
    p.add_argument('--top', type=int, default=10, help='Number of named variations per spectrum')
    p.add_argument('--rank-by', choices=('both','yield','impact'), default='both')
    p.add_argument('--postfit', action='store_true', help='Also draw the same variations at their fitted marginal endpoints')
    args=p.parse_args(argv)
    if args.top < 1 or (args.postfit and not args.impacts) or (args.rank_by == 'impact' and not args.impacts):
        p.error('--top must be positive and --postfit/impact ranking requires --impacts')
    model=json.loads(args.model.read_text())
    work=args.workspace or args.model.parent/'fit_r0/workspace.root'
    card=work.parent/'datacard.txt'
    if not card.exists(): raise ValueError('Missing compiled-workspace datacard '+str(card))
    card_source = card.read_text()
    priors,free=nuisance_priors(card_source)
    channels = spectrum_channel_labels(card_source)
    names = {raw: spectrum_parameter_name(raw, channels) for raw in priors}
    if len(set(names.values())) != len(names):
        raise ValueError('Distinct workspace parameters have the same display name')

    impacts = None
    if args.impacts:
        impacts = read_impacts(args.impacts, model, priors, free)
        summary = work.parent/'fit_summary.json'
        if summary.exists():
            info = json.loads(summary.read_text())
            if info['card_sha256'] != model['card_sha256'] or info['fit_card_sha256'] != hashlib.sha256(card.read_bytes()).hexdigest():
                raise ValueError('Workspace datacard and model do not match fit summary')
    import ROOT
    ROOT.gSystem.Load('libHiggsAnalysisCombinedLimit')
    ROOT.gROOT.SetBatch(True); ROOT.gStyle.SetOptStat(0)
    handle=ROOT.TFile.Open(str(work))
    if not handle or handle.IsZombie(): raise ValueError('Cannot open '+str(work))
    output=args.output_dir or args.model.parent/'component_uncertainties'
    output.mkdir(parents=True,exist_ok=True)
    try:
        workspace=handle.Get('w')
        if not workspace: raise ValueError('Missing Combine workspace w')
        for period in args.periods.split(','):
            eras=RUN2 if period == 'Run2' else RUN3 if period == 'Run3' else (*RUN2,*RUN3) if period == 'Run2+3' else (period,)
            if any(era not in model['eras'] for era in eras):
                raise ValueError('Requested eras are absent from this joint model: '+period)
            payload=evaluate_workspace(workspace,model,priors,free,eras,include_total=True)
            label_payload(payload, names)
            labelled_impacts = {names[key]: value for key,value in impacts.items()} if impacts else None
            chosen = {}
            for proc, result in payload['processes'].items():
                ranking = rank_sources(result, labelled_impacts, args.rank_by, args.top)
                chosen[proc] = [r['name'] for r in ranking if r['selected']]
                result['ranking'] = ranking
            payload['impact_file'] = str(args.impacts) if args.impacts else None
            payload['mass'] = model['mass']
            path=output/f'prefit_relative_uncertainties_{period}.json'
            path.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n'); print('[SAVED]',path)
            draw_spectra(payload, output, period, 'prefit')
            draw_spectra(payload, output, period, 'prefit', chosen)
            if args.postfit:
                post = evaluate_workspace(workspace,model,priors,free,eras,include_total=True,fitted=impacts)
                label_payload(post, names)
                post.update(mass=model['mass'], selected=chosen, impact_file=str(args.impacts))
                (output/f'postfit_responses_{period}.json').write_text(json.dumps(post,allow_nan=False)+'\n')
                draw_spectra(post, output, period, 'postfit', chosen)
    finally: handle.Close()
    return 0


if __name__ == '__main__': raise SystemExit(main())
