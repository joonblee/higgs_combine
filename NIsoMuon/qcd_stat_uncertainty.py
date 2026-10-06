"""First-order QCD yield statistics shared by the producer and its consumers.

QCDStat/metadata stores the retained SS-fit covariance and primitive transfer
statistics. Integrate before propagating: fitted bins are correlated, so adding
their errors in quadrature is incorrect. The NF/fit cross-covariance is unknown;
sigma_nf + sigma_fit is used as an explicit conservative first-order bound.
This Gaussian approximation is not a coverage test or a control-region fit.
No ROOT or third-party Python package is required for the propagation itself.
"""

import json
import math

QCD_STAT_SCHEMA = "NPS26009_QCDStat_v1"
QCD_STAT_PATH = "QCDStat/metadata"
QCD_STAT_TREATMENT = "linear_bound_unknown_nf_fit_correlation"

# Positive abscissae and weights of 16-point Gauss-Legendre quadrature.
_GL16 = (
    (0.09501250983763744, 0.18945061045506850),
    (0.28160355077925891, 0.18260341504492359),
    (0.45801677765722739, 0.16915651939500254),
    (0.61787624440264375, 0.14959598881657673),
    (0.75540440835500303, 0.12462897125553387),
    (0.86563120238783174, 0.09515851168249278),
    (0.94457502307323258, 0.06225352393864789),
    (0.98940093499164993, 0.02715245941175410),
)


def validate_qcd_stat_metadata(metadata, era=None, template_path=None):
    if metadata.get("schema") != QCD_STAT_SCHEMA:
        raise ValueError("Missing/obsolete QCD statistical metadata; rerun qcd_bkg_estimation.py in ss-data mode.")
    if metadata.get("treatment") != QCD_STAT_TREATMENT:
        raise ValueError("Unsupported QCD statistical treatment.")
    if era is not None and metadata.get("era") != era:
        raise ValueError("QCD statistical metadata belong to a different era.")
    if template_path is not None and metadata.get("template_path") != template_path:
        raise ValueError("QCD statistical metadata belong to a different histogram.")
    fit = metadata["fit"]
    if (fit.get("model") != "power_exp_logistic" or fit.get("coordinates") != "log(A),shape"
            or not fit.get("reliable") or fit.get("covariance_status") != 3
            or fit.get("boundary_parameters")):
        raise ValueError("Reliable, interior SS central-fit covariance is required for QCD statistics.")
    p, c = fit["parameters"], fit["covariance"]
    if len(p) != 5 or len(c) != 5 or any(len(row) != 5 for row in c):
        raise ValueError("Invalid QCD fit covariance dimensions.")
    if not all(math.isfinite(v) for v in p + [v for row in c for v in row]):
        raise ValueError("Non-finite QCD fit parameters/covariance.")
    if p[4] <= 0 or any(c[i][i] < 0 for i in range(5)):
        raise ValueError("Invalid QCD fit width/variance.")
    if any(not math.isclose(c[i][j], c[j][i], rel_tol=1e-8, abs_tol=1e-15)
           for i in range(5) for j in range(5)):
        raise ValueError("Asymmetric QCD fit covariance.")
    transfer = metadata["transfer_statistics"]
    if not transfer.get("complete"):
        raise ValueError("Incomplete QCD transfer statistics; DYAux/NF_aMC is required.")
    for key in ("low_transfer", "mc_double_ratio", "high_transfer", "low_variance", "double_ratio_variance"):
        if not math.isfinite(transfer[key]) or transfer[key] < 0:
            raise ValueError("Invalid QCD transfer statistics.")
    if transfer["low_transfer"] <= 0 or transfer["mc_double_ratio"] <= 0:
        raise ValueError("Non-positive QCD transfer factor.")
    if not math.isclose(transfer["high_transfer"], transfer["low_transfer"] * transfer["mc_double_ratio"], rel_tol=1e-10):
        raise ValueError("Inconsistent low/high QCD transfer factors.")
    return metadata


def read_qcd_stat_metadata(root_file, era=None, template_path=None):
    obj = root_file.Get(QCD_STAT_PATH) if root_file else None
    if not obj or not hasattr(obj, "GetString"):
        raise ValueError("Missing QCDStat/metadata; regenerate this era's NIsoMuon_SS_fit.root.")
    return validate_qcd_stat_metadata(json.loads(str(obj.GetString())), era, template_path)


def whole_bin_window(hist, low, high):
    axis, n = hist.GetXaxis(), hist.GetNbinsX()
    low, high = max(low, axis.GetXmin()), min(high, axis.GetXmax())
    if high <= low:
        raise ValueError("Empty QCD statistical window.")
    eps = 1e-9 * max(1.0, high - low)
    first = max(1, min(n, axis.FindFixBin(low + eps)))
    last = max(1, min(n, axis.FindFixBin(high - eps)))
    return float(axis.GetBinLowEdge(first)), float(axis.GetBinUpEdge(last))


def _density_gradient(x, p):
    log_a, n, k, m0, w = p
    z = (x - m0) / w
    if z >= 0:
        e = math.exp(-z)
        turnon, complement = 1.0 / (1.0 + e), e / (1.0 + e)
        log_turnon = -math.log1p(e)
    else:
        e = math.exp(z)
        turnon, complement = e / (1.0 + e), 1.0 / (1.0 + e)
        log_turnon = z - math.log1p(e)
    log_f = log_a - n * math.log(x) - k * x + log_turnon
    # Match the producer's 1e-300 density floor, including its derivative.
    if log_f <= math.log(1e-300):
        return (1e-300, 0.0, 0.0, 0.0, 0.0, 0.0)
    f = math.exp(log_f)
    return (f, f, -math.log(x) * f, -x * f,
            -complement * f / w, -(x - m0) * complement * f / w ** 2)


def _integral_gradient(p, low, high, max_width):
    if high <= low:
        return [0.0] * 6
    parts = max(1, math.ceil((high - low) / max_width))
    contributions = [[] for _ in range(6)]
    for part in range(parts):
        a = low + (high - low) * part / parts
        b = low + (high - low) * (part + 1) / parts
        centre, half = (a + b) * 0.5, (b - a) * 0.5
        for node, weight in _GL16:
            for sign in (-1, 1):
                values = _density_gradient(centre + sign * half * node, p)
                for i, value in enumerate(values):
                    contributions[i].append(half * weight * value)
    return [math.fsum(values) for values in contributions]


def _window_statistics(metadata, low, high, max_width):
    fit, transfer = metadata["fit"], metadata["transfer_statistics"]
    p, covariance = fit["parameters"], fit["covariance"]
    integrals, gradients = [], []
    for a, b, factor in ((5.0, 9.0, transfer["low_transfer"]),
                         (11.0, 80.0, transfer["high_transfer"])):
        values = _integral_gradient(p, max(low, a), min(high, b), max_width)
        integrals.append(values[0])
        gradients.append([factor * g for g in values[1:]])
    gradient = [math.fsum(g[i] for g in gradients) for i in range(5)]
    terms = [gradient[i] * covariance[i][j] * gradient[j] for i in range(5) for j in range(5)]
    variance = math.fsum(terms)
    if not math.isfinite(variance) or variance < -1e-10 * max(math.fsum(map(abs, terms)), 1e-300):
        raise ValueError("Invalid propagated QCD fit variance.")
    t, k = transfer["low_transfer"], transfer["mc_double_ratio"]
    nf_variance = ((integrals[0] + k * integrals[1]) ** 2 * transfer["low_variance"]
                   + (t * integrals[1]) ** 2 * transfer["double_ratio_variance"])
    central = t * integrals[0] + transfer["high_transfer"] * integrals[1]
    return dict(central=central, sigma_nf_stat=math.sqrt(nf_variance),
                sigma_fit_stat=math.sqrt(max(variance, 0.0)), gradient=gradient)


def qcd_window_statistics(metadata, low, high, nominal=None, validate=True):
    """Statistics on the supplied effective bin edges, in event-yield units."""
    if validate:
        validate_qcd_stat_metadata(metadata)
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
        raise ValueError("Invalid QCD statistical window edges.")
    coarse = _window_statistics(metadata, low, high, 1.0)
    result = _window_statistics(metadata, low, high, 0.5)
    for key in ("central", "sigma_fit_stat", "sigma_nf_stat"):
        if not math.isclose(coarse[key], result[key], rel_tol=1e-5, abs_tol=1e-280):
            raise ValueError("QCD statistical integration did not converge.")
    if nominal is not None and not math.isclose(result["central"], nominal, rel_tol=1e-5, abs_tol=1e-280):
        raise ValueError("QCD statistical metadata disagree with the nominal template; regenerate matching files.")
    nf, fit = result["sigma_nf_stat"], result["sigma_fit_stat"]
    result.update(sigma_stat_bound=nf + fit,
                  stat_quadrature_assuming_independent=math.hypot(nf, fit),
                  treatment=QCD_STAT_TREATMENT, effective_low=low, effective_high=high)
    if not all(math.isfinite(result[k]) for k in ("central", "sigma_fit_stat", "sigma_nf_stat", "sigma_stat_bound")):
        raise ValueError("Non-finite propagated QCD statistics.")
    return result
