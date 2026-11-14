#!/usr/bin/env python3
"""Build a separate DY-review model from saved, audited raw bin inputs.

No ROOT input preparation or fitting is repeated. The nominal yields, QCD
responses and statistical inputs remain identical. The explicit study adds
independent 30% DY era/bin shifts and a free DY scale per era by default.
Use full_spectrum_fit.py --stage fit with the printed options afterwards.
"""
import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
import sys

import background_shape_closure as closure
import full_spectrum_fit as fs
import limit_workflow as base
import limit_workflow_sideband_fit as shared


def review_card(text,model,opt):
    """Edit the audited source card, preserving every unrelated nuisance line."""
    # A new review model may derive from a hash-verified card with the old names.
    text = re.sub(r'\b(?:FS_|CMS_NPS26009_inSituNorm_)\w+',
                  lambda match: base.spectrum_parameter_name(match.group()), text)
    lines=text.splitlines()
    tokenised=[line.split() for line in lines if line and not line.startswith('#')]
    channels=[line[1:] for line in tokenised if line[0]=='bin'][-1]
    processes=[line[1:] for line in tokenised if line[0]=='process'][0]
    if len(channels)!=len(processes):
        raise ValueError('Invalid source-card process columns.')
    if set(channels)!={r['channel'] for r in model['rows']}:
        raise ValueError('Source-card channels differ from saved model.')
    if opt.dy_normalisation=='free':
        for era in model['eras']:
            name=fs.normalisation_name('DY',era)
            matches=[i for i,line in enumerate(lines) if line.split()[:2]==[name,'lnN']]
            if len(matches)!=1:
                raise ValueError('Expected one original DY normalisation lnN: '+name)
            lines.pop(matches[0])
            for row in model['rows']:
                if row['era']==era:
                    lines.append(f'{name} rateParam {row["channel"]} DY 1 [0,{opt.dy_norm_max:.17g}]')
    if opt.dy_shape_uncertainty:
        for row in model['rows']:
            name=f'CMS_NPS26009_DYClosure_{row["era"]}_{row["region"]}'
            if any(line.split()[:1]==[name] for line in lines):
                raise ValueError('Source already has the requested closure nuisance.')
            response=f'{1-opt.dy_shape_uncertainty:.17g}/{1+opt.dy_shape_uncertainty:.17g}'
            values=[response if channel==row['channel'] and process=='DY' else '-'
                    for channel,process in zip(channels,processes)]
            lines.append(name+' lnN '+' '.join(values))
    return '\n'.join(lines)+'\n'


def prepare(path, output_base, uncertainty=.30, normalisation='free', norm_max=10.):
    original=json.loads(path.read_text());closure.validate_model(original)
    source=path.parent/'datacard.txt'
    old_card=source.read_text() if source.is_file() else shared.card_text(original,normalisation_kappa=original['settings']['norm_prior'],prefix='FS')
    if hashlib.sha256(old_card.encode()).hexdigest()!=original['card_sha256']:
        raise ValueError('Need the original hash-matched datacard.txt beside '+str(path)+'. Run against the original batch results, not a model-only archive.')
    opt=fs.make_parser().parse_args([])
    vars(opt).update(original['settings'])
    opt.output_base=str(Path(output_base).resolve())
    opt.dy_shape_uncertainty=uncertainty
    opt.dy_normalisation=normalisation
    opt.dy_norm_max=norm_max
    opt.inject_r=0.
    opt.stage='fit'
    fs.configuration(opt)
    windows={era:[dict(region=r['region'],low=r['low'],high=r['high']) for r in original['rows'] if r['era']==era]
             for era in original['eras']}
    model=fs.decorate_model(copy.deepcopy(original),opt,windows)
    model['review_parent']=dict(model=str(path.resolve()),
        model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),card_sha256=original['card_sha256'])
    directory=fs.model_directory(opt,original['mass'])
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Review model already exists: '+str(directory))
    text=review_card(old_card,model,opt)
    nominal=shared.nominal_predictions(text)
    if any(not math.isclose(nominal[r['channel']],r['asimov'],rel_tol=1e-10,abs_tol=1e-12) for r in model['rows']):
        raise ValueError('Review card changed a nominal background yield.')
    model['card_sha256']=hashlib.sha256(text.encode()).hexdigest()
    model['signature']=fs.signature(opt)
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'datacard.txt').write_text(text)
    (directory/'model.json').write_text(json.dumps(model,indent=2,allow_nan=False)+'\n')
    return dict(mass=original['mass'],directory=str(directory),
                card_sha256=model['card_sha256'],parent=model['review_parent'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-roots',nargs='+',required=True)
    parser.add_argument('--output-base',default='full_spectrum_outputs/review_dy30')
    parser.add_argument('--dy-shape-uncertainty',type=float,default=.30)
    parser.add_argument('--dy-normalisation',choices=('prior','free'),default='free')
    parser.add_argument('--dy-norm-max',type=float,default=10.)
    opt=parser.parse_args(argv)
    paths=closure.model_paths(opt.model_roots)
    records=[prepare(p,opt.output_base,opt.dy_shape_uncertainty,opt.dy_normalisation,opt.dy_norm_max) for p in paths]
    destination=Path(opt.output_base)/'review_models.json'
    destination.write_text(json.dumps(dict(schema_version=1,blind=True,models=records),indent=2)+'\n')
    print('[SAVED]',destination)
    print('Fit these saved cards with --stage fit and the same target, input and binning settings,')
    print(f'--dy-shape-uncertainty {opt.dy_shape_uncertainty:g} --dy-normalisation {opt.dy_normalisation} '
          f'--dy-norm-max {opt.dy_norm_max:g} --output-base {opt.output_base}')


if __name__=='__main__':
    try:
        main()
    except (ValueError,OSError) as error:
        print('ERROR:',error,file=sys.stderr)
        raise SystemExit(1)
