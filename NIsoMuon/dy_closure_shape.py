"""Run2+3 DY closure trend, signed ROOT inputs and yield-preserving morphing.

Identical protocol module in SKPlotMaker and higgs_combine/NIsoMuon.
The central fitted trend defines the modelling variation, not its fit errors.
"""
import hashlib
import json
import math

DOMAIN = (11., 80.)
SLOPE = .00583
INTERCEPT = .9038
NUISANCE = 'CMS_NPS26009_DYClosureLinearShape'
META_PATH = 'DYAux/ClosureLinearMetadata'
UP_PATH = 'DYAux/ClosureLinearSourceUp'
DOWN_PATH = 'DYAux/ClosureLinearSourceDown'
SOURCE_PATH = 'DYAux/LightJetSource'


def configuration():
    return dict(schema='NPS26009_DYClosureLinear_v1', slope_per_gev=SLOPE,
                intercept=INTERCEPT, domain=list(DOMAIN), fit_domain=[11., 70.],
                nuisance=NUISANCE, correlation='common across all eras and mass bins',
                prescription='normalised linear Up; nominal-reflected Down; physical bounds then renormalisation',
                source='AN eq:dy_mass_shape_linear_fit; central trend, not fit statistical errors')


def fingerprint(hist):
    values = [(float(hist.GetBinLowEdge(i)), float(hist.GetBinContent(i)),
               float(hist.GetBinError(i))) for i in range(1, hist.GetNbinsX()+1)]
    values.append(float(hist.GetXaxis().GetBinUpEdge(hist.GetNbinsX())))
    return hashlib.sha256(json.dumps(values, allow_nan=False).encode()).hexdigest()


def write_inputs(ROOT, directory, source):
    """Save signed weighted sources; do not change nominal or its Sumw2."""
    directory.cd()
    meta = dict(configuration(), source_sha256=fingerprint(source))
    for path, sign in ((UP_PATH, 1.), (DOWN_PATH, -1.)):
        out = source.Clone(path.split('/')[-1])
        out.SetDirectory(0)
        for i in range(out.GetNbinsX()+2):
            mass = float(source.GetBinCenter(i))
            delta = SLOPE*mass+INTERCEPT-1. if DOMAIN[0] < mass < DOMAIN[1] else 0.
            out.SetBinContent(i, float(source.GetBinContent(i))*(1.+sign*delta))
            out.SetBinError(i, 0.)  # Source statistics remain in LightJetSource once.
        out.Write()
    ROOT.TObjString(json.dumps(meta, sort_keys=True)).Write(META_PATH.split('/')[-1])
    return meta


def read_inputs(root_file):
    obj = root_file.Get(META_PATH)
    if not obj:
        raise ValueError('Missing DY closure metadata; regenerate DY estimates with updated SKPlotMaker')
    meta = json.loads(str(obj.GetString().Data()))
    if any(meta.get(k) != v for k, v in configuration().items()):
        raise ValueError('Stale/incompatible DY closure configuration; regenerate DY estimates')
    source, up, down = (root_file.Get(p) for p in (SOURCE_PATH, UP_PATH, DOWN_PATH))
    if not source or not up or not down or meta.get('source_sha256') != fingerprint(source):
        raise ValueError('Missing/stale DY closure source inputs')
    for variation, sign in ((up, 1.), (down, -1.)):
        if variation.GetNbinsX() != source.GetNbinsX():
            raise ValueError('DY closure input binning mismatch')
        for i in range(1, source.GetNbinsX()+1):
            mass = float(source.GetBinCenter(i))
            delta = SLOPE*mass+INTERCEPT-1. if DOMAIN[0] < mass < DOMAIN[1] else 0.
            expected = float(source.GetBinContent(i))*(1.+sign*delta)
            if (not math.isclose(float(variation.GetBinLowEdge(i)), float(source.GetBinLowEdge(i)), abs_tol=1e-9)
                    or not math.isclose(float(variation.GetBinContent(i)), expected, rel_tol=2e-6, abs_tol=1e-10)
                    or variation.GetBinError(i) != 0.):
                raise ValueError('DY closure input does not match signed source and fitted trend')
    return source, up, meta


def integrate(hist, low, high):
    axis = hist.GetXaxis()
    first, last = axis.FindFixBin(low+1e-7), axis.FindFixBin(high-1e-7)
    if not (math.isclose(float(axis.GetBinLowEdge(first)), low, abs_tol=1e-7)
            and math.isclose(float(axis.GetBinUpEdge(last)), high, abs_tol=1e-7)):
        raise ValueError('DY closure integration must use whole native bins')
    return float(hist.Integral(first, last))


def templates(nominal, weighted_up):
    """Use final nonnegative fit bins; keep nominal-zero bins zero in both shapes."""
    if len(nominal) != len(weighted_up) or any(not math.isfinite(x) for x in (*nominal, *weighted_up)):
        raise ValueError('Invalid DY closure yield arrays')
    if any(x < 0 for x in nominal):
        raise ValueError('DY closure nominal must use the existing physical integrated-bin prescription')
    total = sum(nominal)
    if total == 0.:
        return list(nominal), list(nominal), dict(total=0., clipped_up=[], clipped_down=[])
    up = [max(0., v) if n > 0 else 0. for n, v in zip(nominal, weighted_up)]
    if sum(up) <= 0.:
        raise ValueError('DY linear variation has no positive integrated yield')
    clipped_up = [i for i, (n, v) in enumerate(zip(nominal, weighted_up)) if n > 0 and v < 0]
    up = [v*total/sum(up) for v in up]
    reflected = [2.*n-v for n, v in zip(nominal, up)]
    down = [max(0., v) for v in reflected]
    down = [v*total/sum(down) for v in down]
    return up, down, dict(total=total, clipped_up=clipped_up,
                         clipped_down=[i for i,v in enumerate(reflected) if v < 0.])


def morph(nominal, up, down, theta):
    raw = [max(0., n+max(theta,0.)*(u-n)+max(-theta,0.)*(d-n))
           for n,u,d in zip(nominal, up, down)]
    total = sum(nominal)
    return [x*total/sum(raw) for x in raw] if total else list(raw)


def card_lines(rows, era):
    """One Gaussian shape plus a shared per-era normaliser, attached only to positive bins."""
    selected = [r for r in rows if r['era'] == era and r['rates']['DY'] > 0.]
    if not selected:
        return []
    f = lambda x: format(float(x), '.17g')
    factors = []
    for row in selected:
        n = row['rates']['DY']
        u, d = row['dy_closure_up']/n-1., row['dy_closure_down']/n-1.
        factors.append('max(0.,1.+max(@0,0.)*('+f(u)+')+max(-@0,0.)*('+f(d)+'))')
    denominator = '+'.join(f(r['rates']['DY'])+'*('+w+')' for r,w in zip(selected,factors))
    normaliser = f(sum(r['rates']['DY'] for r in selected))+'/('+denominator+')'
    lines = []
    for index, (row, weight) in enumerate(zip(selected, factors)):
        channel = row['channel']
        lines.append(f'DYClosureBin_{channel} rateParam {channel} DY {weight} {NUISANCE}')
        # Define the O(n-bin) denominator once; identities attach it to other
        # channels without an O(n-bin squared) card for fine spectrum layouts.
        if index == 0:
            lines.append(f'DYClosureNorm_{era} rateParam {channel} DY {normaliser} {NUISANCE}')
        else:
            lines.append(f'DYClosureNormUse_{channel} rateParam {channel} DY @0 DYClosureNorm_{era}')
    return lines
