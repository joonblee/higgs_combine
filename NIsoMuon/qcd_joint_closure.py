#!/usr/bin/env python3
"""Joint QCD transport screening using saved models and MC closure results.

Profile the actual max(0,QCD+envelope+SS modes) response and its normalisation.
The covariance contains transported finite-MC errors plus nominal total-event
Poisson variances. This is a Gaussian expected-fit screening diagnostic with
other background responses fixed, not an exact Combine fit or a coverage
p-value. Source statistical errors are not converted into new shape nuisances.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import background_shape_closure as closure


def ratio_covariance(record, indices):
    s, t, vs, vt = [np.asarray(record[k], float) for k in
                    ('source','target','source_variance','target_variance')]
    S, T = s.sum(), t.sum()
    if S <= 0 or T <= 0:
        raise ValueError('Nonpositive MC full-domain integral.')
    ratio = np.array([record['ratio'][i] for i in indices])
    jt = np.full((len(indices),len(s)), 0.)
    js = np.full_like(jt, 0.)
    for j, i in enumerate(indices):
        jt[j,:] = -ratio[j]/T
        jt[j,i] += 1/(s[i]*T/S)
        js[j,:] = ratio[j]/S
        js[j,i] -= ratio[j]/s[i]
    return (jt*vt)@jt.T+(js*vs)@js.T


def profile_qcd(rows, record, kappa, mode):
    from scipy.optimize import least_squares
    indices = [i for i,v in enumerate(record['ratio']) if v is not None]
    if len(indices) < 2:
        return dict(status='inconclusive', reason='Fewer than two defined MC ratios.')
    selected = [rows[i] for i in indices]
    q = np.array([r['rates']['QCD'] for r in selected])
    target = q*np.array([record['ratio'][i] for i in indices])
    envelope = np.array([r['qcd_shape'] for r in selected])
    modes = np.asarray([r['qcd_modes'] for r in selected],float)
    total = np.array([r['asimov'] for r in selected])
    mc_covariance = q[:,None]*ratio_covariance(record,indices)*q[None,:]
    covariance = mc_covariance+np.diag(total)
    factor = np.linalg.cholesky((covariance+covariance.T)/2)
    response = np.column_stack((envelope,modes))
    def predict(parameters):
        beta = math.exp(math.log(kappa)*parameters[0]) if mode == 'prior' else parameters[0]
        return beta*np.maximum(0.,q+response@parameters[1:])
    def objective(parameters):
        delta = np.linalg.solve(factor, target-predict(parameters))
        prior = parameters if mode == 'prior' else parameters[1:]
        return np.r_[delta,prior]
    initial = np.zeros(1+response.shape[1])
    if mode == 'free':
        initial[0] = 1.
    lower, upper = np.full(len(initial),-np.inf),np.full(len(initial),np.inf)
    if mode == 'free':
        lower[0],upper[0] = 0.,10.
    fit = least_squares(objective,initial,bounds=(lower,upper),max_nfev=5000,
                        ftol=1e-9,xtol=1e-9,gtol=1e-9)
    residual = target-predict(fit.x)
    baseline = np.linalg.solve(factor,target-q)
    whitened = np.linalg.solve(factor,residual)
    penalty = fit.x if mode == 'prior' else fit.x[1:]
    return dict(status='diagnostic_converged' if fit.success else 'diagnostic_failed',
        normalisation_mode=mode, normalisation_scale=float(math.exp(math.log(kappa)*fit.x[0]) if mode=='prior' else fit.x[0]),
        normalisation_at_bound=bool(mode=='free' and min(abs(fit.x[0]),abs(fit.x[0]-10)) < 1e-4),
        envelope_pull=float(fit.x[1]), ss_mode_pulls=fit.x[2:].tolist(),
        normalisation_pull=float(fit.x[0]) if mode=='prior' else None,
        baseline_quadratic=float(baseline@baseline), residual_quadratic=float(whitened@whitened),
        prior_penalty=float(penalty@penalty), objective=float(fit.fun@fit.fun),
        largest_constrained_pull=float(np.max(np.abs(penalty))),
        included_bins=[r['region'] for r in selected],
        excluded_bins=[rows[i]['region'] for i in range(len(rows)) if i not in indices],
        gaussian_count_approximation_bins=[r['region'] for r in selected if r['asimov'] < 5],
        qcd_nominal=q.tolist(), transported_qcd_proxy=target.tolist(), fitted_qcd=predict(fit.x).tolist(),
        residual=residual.tolist(), covariance=covariance.tolist(),
        note='Screening only: MC ratio is a transport proxy; other backgrounds fixed; no pass threshold or coverage p-value.')


def grouped_closure(record, edges):
    groups = []
    for low,high in zip(edges,edges[1:]):
        indices = [i for i,b in enumerate(record['bins']) if b['low'] >= low-1e-8 and b['high'] <= high+1e-8]
        if not indices or not math.isclose(sum(record['bins'][i]['high']-record['bins'][i]['low'] for i in indices),high-low,abs_tol=1e-7):
            return None  # Never split an integrated fit bin.
        groups.append(indices)
    arrays = [[sum(record[key][i] for i in group) for group in groups]
              for key in ('source','target','source_variance','target_variance')]
    result = closure.shape_closure(*arrays)
    result.update(era=record['era'],process=record['process'],edges=edges)
    return result


def run(paths, closure_path):
    payload = json.loads(Path(closure_path).read_text())
    if payload.get('mc_only') is not True or payload.get('blind') is not True:
        raise ValueError('Require the blinded MC-only closure report.')
    output,wide = [],{}
    for path in paths:
        model = json.loads(path.read_text());closure.validate_model(model)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        matches = [m for m in payload['models'] if m['model_sha256']==digest]
        if len(matches) != 1:
            raise ValueError('Need exactly one matching MC closure model hash: '+str(path))
        for era in model['eras']:
            records = [r for r in matches[0]['records'] if r['era']==era and r['process']=='QCD']
            if len(records)!=1 or records[0]['status']!='diagnostic':
                raise ValueError('Missing usable QCD closure: '+era)
            record = records[0]
            by_region = {r['region']:r for r in model['rows'] if r['era']==era}
            rows = [by_region[b['region']] for b in record['bins']]
            if any((r['low'],r['high'])!=(b['low'],b['high']) for r,b in zip(rows,record['bins'])):
                raise ValueError('Closure/model bin geometry differs.')
            spec = next(v for v in model['normalisations'].values() if v['era']==era and v['process']=='QCD')
            output.append(dict(model=str(path),mass=model['mass'],era=era,
                prior=profile_qcd(rows,record,spec['kappa'],'prior'),
                free=profile_qcd(rows,record,spec['kappa'],'free')))
        for record in matches[0]['records']:
            key=record['era']+'/'+record['process']
            if key not in wide and record['status']=='diagnostic':
                grouped=grouped_closure(record,[11.,20.,30.,40.,60.,80.])
                if grouped is not None:
                    wide[key]=grouped
    return dict(schema_version=1,blind=True,closure_sha256=hashlib.sha256(Path(closure_path).read_bytes()).hexdigest(),
                qcd_joint_checks=output,wide_mc_checks=list(wide.values()))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-roots',nargs='+',required=True)
    parser.add_argument('--closure-json',required=True)
    parser.add_argument('--output',default='full_spectrum_outputs/qcd_joint_check.json')
    opt=parser.parse_args(argv)
    output=Path(opt.output)
    if output.exists():
        raise ValueError('Choose a new --output; existing reports are preserved.')
    result=run(closure.model_paths(opt.model_roots),opt.closure_json)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for row in result['qcd_joint_checks']:
        print(f'M{row["mass"]:g} {row["era"]}: prior={row["prior"]["status"]}, free={row["free"]["status"]}')
    print('[SAVED]',output)


if __name__=='__main__':
    try:
        main()
    except (ValueError,OSError) as error:
        print('ERROR:',error,file=sys.stderr)
        raise SystemExit(1)
