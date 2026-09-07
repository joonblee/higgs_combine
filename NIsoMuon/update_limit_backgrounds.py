#!/usr/bin/env python3
"""Apply a narrowly scoped DY/QCD background-interface update to limit_workflow.py.

Audited input:
  joonblee/higgs_combine c110a52d484cb121cb44b392cf87e28574de5e65
  NIsoMuon/limit_workflow.py (Git blob 227a889ca6c9d20530c21154d9418dd6f939dd8b)
Upstream producer:
  joonblee/SKPlotMaker a67efb7788123ee293be729aa18ccbe213005f71

No network, ROOT, external libraries or Git writes are required. The default
prints a unified diff without modifying anything. --apply backs up the original,
writes a patch and atomically replaces only this file. --output writes a separate
complete updated workflow. Unrecognised input revisions are refused.

Usage (from the higgs_combine/NIsoMuon directory):
  python3 update_limit_backgrounds.py limit_workflow.py --apply
  python3 -m py_compile limit_workflow.py
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import os
from pathlib import Path
import stat
import sys
import tempfile

EXPECTED_GIT_BLOB = "227a889ca6c9d20530c21154d9418dd6f939dd8b"
BASE_COMMIT = "c110a52d484cb121cb44b392cf87e28574de5e65"
PRODUCER_COMMIT = "a67efb7788123ee293be729aa18ccbe213005f71"
MARKER = 'BACKGROUND_CONTRACT = "SKPlotMaker-NF-SS-20260907-v1"'

REPLACEMENTS = (
    ('background revision',
     r'''Standalone NIsoMuon Run-2/Run-3 counting-limit workflow, revision timestamp 20260826_0541.''',
     r'''Standalone NIsoMuon Run-2/Run-3 counting-limit workflow, background-interface revision 20260907.'''),
    ('DYAux schema constants',
     r'''DY_AUX_NF_MG_PATH = "DYAux/NF_MG"
''',
     r'''DY_AUX_NF_MG_PATH = "DYAux/NF_MG"
DY_AUX_NF_MODEL_PATH = "DYAux/NFModelRel"
DY_AUX_NF_AMC_INPUTS_PATH = "DYAux/NFInputs_aMC"
DY_AUX_NF_MG_INPUTS_PATH = "DYAux/NFInputs_MG"
# Native SS/DY outputs exclude 9--11 GeV; adaptive SS bins are fit-only.
DATA_DRIVEN_MASS_WINDOW = (11.0, 80.0)
BACKGROUND_CONTRACT = "SKPlotMaker-NF-SS-20260907-v1"
'''),
    ('data-driven search support',
     r'''    low = max(SEARCH_MASS_MIN, mass - args.n_sigma * sigma)
    high = min(SEARCH_MASS_MAX, mass + args.n_sigma * sigma)
    if not high > low:
''',
     r'''    low = max(SEARCH_MASS_MIN, mass - args.n_sigma * sigma)
    high = min(SEARCH_MASS_MAX, mass + args.n_sigma * sigma)
    if args.dy_method == "data-driven" or args.qcd_method == "data-driven":
        # Use the same supported interval for data, signal and every background.
        # MC-only cross-checks retain the existing SEARCH_MASS_MIN setting.
        low = max(low, DATA_DRIVEN_MASS_WINDOW[0])
        high = min(high, DATA_DRIVEN_MASS_WINDOW[1])
    if not high > low:
'''),
    ('background helpers',
     r'''
def build_channels_for_mass(
''',
     r'''
# Background input contract: SKPlotMaker a67efb7 (NF DY and SS-data QCD).
def _background_check(args, audit, warnings, key, year, ok, detail):
    """Record a failed background-contract check; never hide it in strict mode."""
    audit.add(key, year, bool(ok))
    if not ok:
        message = f"{year}: {detail}"
        warnings.append(message)
        if args.strict:
            raise WorkflowError(message)
    return bool(ok)


def _background_close(value, reference):
    # Relative tolerance is important for the very small high-mass QCD tail.
    return (
        math.isfinite(value) and math.isfinite(reference)
        and math.isclose(value, reference, rel_tol=1.0e-6, abs_tol=1.0e-300)
    )


def _background_aux_values(reader, args, audit, warnings, year, filename, path, labels):
    """Read the labelled, one-dimensional DYAux schema written by the producer."""
    root_file = reader._file(filename)
    hist = root_file.Get(path) if root_file else None
    present = hist is not None and bool(hist)
    if not _background_check(
        args, audit, warnings, f"background/schema/{path}", year,
        present and hist.GetDimension() == 1 and hist.GetNbinsX() == len(labels),
        f"Missing or incompatible {filename}:{path}; regenerate NF DY templates.",
    ):
        return None
    actual_labels = tuple(str(hist.GetXaxis().GetBinLabel(i)) for i in range(1, len(labels) + 1))
    if not _background_check(
        args, audit, warnings, f"background/labels/{path}", year,
        actual_labels == tuple(labels),
        f"Unexpected labels in {path}: {actual_labels!r}, expected {tuple(labels)!r}.",
    ):
        return None
    values = tuple(float(hist.GetBinContent(i)) for i in range(1, len(labels) + 1))
    if not all(math.isfinite(value) for value in values):
        raise WorkflowError(f"{year}: non-finite DYAux values in {filename}:{path}.")
    return values


def _audit_dy_background_inputs(
    reader, args, audit, warnings, year, filename, low, high, source, nf_amc, nf_mg
):
    """Validate the NF factorisation before any numerical rate floor is applied.

    These are input-integrity checks, not additional nuisance parameters.
    NFStat is common to every mass window of one era; LightJetStat is the
    quadrature Sumw2 uncertainty of Data-QCD-Top-Others in the selected window.
    """
    for label, result in (("LightJetSource", source), ("NF_aMC", nf_amc), ("NF_MG", nf_mg)):
        if not (
            math.isfinite(result.value) and math.isfinite(result.error)
            and result.error >= 0.0
        ):
            raise WorkflowError(f"{year}: invalid value/error in DYAux/{label}.")
    if nf_amc.value <= 0.0 or nf_mg.value <= 0.0 or nf_amc.error <= 0.0:
        raise WorkflowError(f"{year}: DY requires positive aMC/MG NFs and a positive NFStat width.")
    _background_check(
        args, audit, warnings, "background/DY_source_positive", year, source.value > 0.0,
        f"DY LightJetSource integral is {source.value:g} in [{low:g},{high:g}] GeV. "
        "A non-positive subtracted source cannot define the current multiplicative "
        "LightJetStat model. Non-strict mode retains the historical rate floor.",
    )

    nominal = read_required(
        reader, audit, "nominal/DY_central", year,
        filename, hist_path(args.region), low, high,
    )
    _background_check(
        args, audit, warnings, "background/DY_factorisation", year,
        nominal is not None and _background_close(nominal.value, source.value * nf_amc.value),
        "DYAux/LightJetSource * NF_aMC does not reproduce the nominal DY yield "
        f"in [{low:g},{high:g}] GeV; regenerate the DY file instead of mixing inputs.",
    )
    _background_check(
        args, audit, warnings, "background/DY_source_error", year,
        nominal is not None and _background_close(nominal.error, source.error * nf_amc.value),
        "The nominal DY histogram error must contain LightJetStat only; "
        "NFStat must remain separate in DYAux/NF_aMC.",
    )

    for path, label in ((DY_AUX_NF_AMC_PATH, "NF_aMC"), (DY_AUX_NF_MG_PATH, "NF_MG")):
        _background_aux_values(reader, args, audit, warnings, year, filename, path, (label,))
    model = _background_aux_values(
        reader, args, audit, warnings, year, filename, DY_AUX_NF_MODEL_PATH, ("NFModelRel",),
    )
    expected_model = abs(nf_mg.value / nf_amc.value - 1.0)
    if model is not None:
        _background_check(
            args, audit, warnings, "background/DY_NFModel", year,
            model[0] >= 0.0 and _background_close(model[0], expected_model),
            f"DYAux/NFModelRel={model[0]:g}, but |NF_MG/NF_aMC-1|={expected_model:g}.",
        )

    labels = ("BJetYield", "BJetError", "LightJetYield", "LightJetError", "WindowLow", "WindowHigh")
    for tag, factor, path in (
        ("aMC", nf_amc, DY_AUX_NF_AMC_INPUTS_PATH),
        ("MG", nf_mg, DY_AUX_NF_MG_INPUTS_PATH),
    ):
        values = _background_aux_values(
            reader, args, audit, warnings, year, filename, path, labels,
        )
        if values is None:
            continue
        b, b_error, light, light_error, window_low, window_high = values
        _background_check(
            args, audit, warnings, f"background/DY_NFWindow_{tag}", year,
            _background_close(window_low, DATA_DRIVEN_MASS_WINDOW[0])
            and _background_close(window_high, DATA_DRIVEN_MASS_WINDOW[1]),
            f"DY {tag} NF was extracted in [{window_low:g},{window_high:g}] GeV, "
            "not the production interval [11,80] GeV.",
        )
        if not _background_check(
            args, audit, warnings, f"background/DY_NFInputs_{tag}", year,
            b > 0.0 and light > 0.0 and b_error >= 0.0 and light_error >= 0.0,
            f"Invalid primitive B/light yields or errors in {path}.",
        ):
            continue
        expected_nf = b / light
        expected_error = math.hypot(b_error / light, expected_nf * (light_error / light))
        _background_check(
            args, audit, warnings, f"background/DY_NFValue_{tag}", year,
            _background_close(factor.value, expected_nf),
            f"DY {tag} NF={factor.value:g} does not equal B/light={expected_nf:g}.",
        )
        _background_check(
            args, audit, warnings, f"background/DY_NFError_{tag}", year,
            _background_close(factor.error, expected_error),
            f"DY {tag} NFStat={factor.error:g} does not reproduce the full "
            f"finite-MC propagation from B/light inputs ({expected_error:g}).",
        )


def _audit_qcd_background_central(reader, args, audit, warnings, year, filename, low, high, nominal):
    syst_nominal = read_required(
        reader, audit, "background/QCD_RunSyst_central", year,
        filename, hist_path(args.region), low, high,
    )
    _background_check(
        args, audit, warnings, "background/QCD_central_consistency", year,
        syst_nominal is not None and _background_close(syst_nominal.value, nominal.value),
        "Nominal and RunSyst QCD central integrals disagree. Regenerate ss-data "
        "templates; do not combine a new central prediction with old variations.",
    )
    if not math.isfinite(nominal.value) or nominal.value < 0.0:
        raise WorkflowError(f"{year}: invalid fitted QCD central yield {nominal.value!r}.")
    _background_check(
        args, audit, warnings, "background/QCD_template_error", year,
        nominal.error == 0.0 and syst_nominal is not None and syst_nominal.error == 0.0,
        "SS fitted-template errors must be zero for the current QCD_norm/QCD_shape-only model.",
    )


def _audit_qcd_background_variation(args, audit, warnings, year, name, nominal, down, up):
    if down is None or up is None:
        return
    if not all(math.isfinite(value) and value >= 0.0 for value in (nominal, down.value, up.value)):
        raise WorkflowError(f"{year}: non-finite/negative {name} QCD template integral.")
    _background_check(
        args, audit, warnings, f"background/{name}_bracketing", year,
        (down.value <= nominal or _background_close(down.value, nominal))
        and (up.value >= nominal or _background_close(up.value, nominal)),
        f"{name} does not bracket QCD nominal: Down={down.value:g}, "
        f"central={nominal:g}, Up={up.value:g}.",
    )


def _qcd_background_norm_lnn(args, audit, warnings, year, nominal, down, up):
    """Preserve producer NormDown/Up ratios without clipping them with ratio_floor.

    The high-mass central already contains R_data(low)*R_MC(high)/R_MC(low).
    Do NOT apply that transfer factor again here, and do NOT add a QCD stat term.
    """
    if down is None or up is None:
        return "-"
    if nominal == 0.0:
        _background_check(
            args, audit, warnings, "background/QCD_zero_norm", year,
            down.value == 0.0 and up.value == 0.0,
            "A zero QCD nominal with non-zero Norm variations cannot be encoded as lnN.",
        )
        return "-"
    kd, ku = down.value / nominal, up.value / nominal
    if not (math.isfinite(kd) and math.isfinite(ku) and kd > 0.0 and ku > 0.0):
        raise WorkflowError(f"{year}: QCD_norm requires positive finite Down/Up ratios.")
    _background_check(
        args, audit, warnings, "background/QCD_norm_logsymmetry", year,
        abs(math.log(kd) + math.log(ku)) <= 1.0e-6,
        f"QCD NormDown/Up are not reciprocal ({kd:g}/{ku:g}); "
        "regenerate the latest log-symmetric ss-data templates.",
    )
    if max(abs(kd - 1.0), abs(ku - 1.0)) < args.ignore_rel_below:
        return "-"
    return f"{format_kappa(kd)}/{format_kappa(ku)}"


def _validate_background_card(args, card):
    """Reject pre-update data-driven cards before any old outputs are removed."""
    if args.dy_method == "mc" and args.qcd_method == "mc":
        return
    text = Path(card).read_text()
    required_lines = {
        f"# background_contract = {BACKGROUND_CONTRACT}",
        f"# background_methods = DY:{args.dy_method};QCD:{args.qcd_method}",
    }
    if not required_lines.issubset(set(text.splitlines())):
        raise WorkflowError(
            f"{card}: missing/mismatched current background contract. Rebuild "
            "cards with --stage cards (or --stage all) from the regenerated "
            "DY/QCD ROOT files before running Combine. Do not relabel old cards."
        )
    rows = [line.split() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if args.dy_method == "data-driven":
        for row in rows:
            if len(row) > 1 and row[1] == "lnN" and ("_NFStat_DY_" in row[0] or "_MCstat_DY_" in row[0]):
                raise WorkflowError(f"{card}: obsolete/duplicate DY statistical lnN row: {row[0]}.")
        bins = next((row[1:] for row in rows if row[0] == "bin"), [])
        if not bins:
            raise WorkflowError(f"{card}: no channel bins found.")
        for bin_name in bins:
            year = bin_name[4:] if bin_name.startswith("bin_") else bin_name
            if year not in YEARS:
                raise WorkflowError(f"{card}: unknown channel {bin_name}.")
            nf_name = nuisance_global_name("DY_NFStat", year)
            has_rate = any(len(row) >= 5 and row[:4] == [nf_name, "rateParam", bin_name, "DY"] for row in rows)
            has_constraint = any(len(row) >= 4 and row[:2] == [nf_name, "param"] for row in rows)
            if not (has_rate and has_constraint):
                raise WorkflowError(f"{card}: DY NFStat needs both rateParam and Gaussian param for {year}.")
    if args.qcd_method == "data-driven":
        if any(len(row) > 1 and row[1] == "lnN" and "_MCstat_QCD_" in row[0] for row in rows):
            raise WorkflowError(f"{card}: a fitted-template QCD MCstat nuisance is not part of the current model.")


def build_channels_for_mass(
'''),
    ('validate primitive DY before flooring',
     r'''                else:
                    dy_lightjet_yield = max(float(source.value), args.rate_floor)
''',
     r'''                else:
                    _audit_dy_background_inputs(
                        reader, args, audit, warnings, year, filename, low, high,
                        source, nf_amc_result, nf_mg_result,
                    )
                    dy_lightjet_yield = max(float(source.value), args.rate_floor)
'''),
    ('replace warning-only DY factorisation audit',
     r'''                    # Audit that the primitive factorization reproduces the
                    # historical final central histogram written by the producer.
                    legacy = reader.integral(filename, hist_path(args.region), low, high)
                    if legacy is not None:
                        denom = max(abs(dy_value), 1.0e-12)
                        rel_diff = abs(float(legacy.value) - dy_value) / denom
                        if rel_diff > 1.0e-6:
                            warnings.append(
                                "DYAux factorization does not reproduce the nominal "
                                f"DY histogram: primitive={dy_value:.6g}, "
                                f"nominal={legacy.value:.6g}, rel={rel_diff:.3g}."
                            )
''',
     r'''                    # Central/error closure and the full DYAux schema were
                    # audited above, using the signed source before flooring.
'''),
    ('QCD nominal/RunSyst consistency',
     r'''            qcd_file = file_for_process(args, year, "QCD", signal_file, "qcd")
            for syst_name, (down_suffix, up_suffix) in QCD_SYST.items():
''',
     r'''            qcd_file = file_for_process(args, year, "QCD", signal_file, "qcd")
            _audit_qcd_background_central(
                reader, args, audit, warnings, year, qcd_file, low, high,
                raw_nominal["QCD"],
            )
            for syst_name, (down_suffix, up_suffix) in QCD_SYST.items():
'''),
    ('QCD variation integrity',
     r'''                if syst_name == "QCD_shape":
                    if down is not None and up is not None:
''',
     r'''                _audit_qcd_background_variation(
                    args, audit, warnings, year, syst_name,
                    raw_nominal["QCD"].value, down, up,
                )
                if syst_name == "QCD_shape":
                    if down is not None and up is not None:
'''),
    ('unclipped log-symmetric QCD norm',
     r'''                nuis[syst_name]["QCD"] = lnn_from_down_up(
                    raw_nominal["QCD"].value,
                    down.value if down else None,
                    up.value if up else None,
                    args.ratio_floor,
                    args.ignore_rel_below,
                )
''',
     r'''                nuis[syst_name]["QCD"] = _qcd_background_norm_lnn(
                    args, audit, warnings, year, raw_nominal["QCD"].value,
                    down, up,
                )
'''),
    ('zero-yield QCD shape is well-defined',
     r'''                        relative_size = max(sigma_down, sigma_up) / max(
                            abs(nominal_qcd), args.rate_floor
                        )
''',
     r'''                        width = max(sigma_down, sigma_up)
                        reference = max(abs(nominal_qcd), args.rate_floor)
                        relative_size = (
                            width / reference if reference > 0.0
                            else (math.inf if width > 0.0 else 0.0)
                        )
'''),
    ('record card background contract',
     r'''        f"# limit_parameter = {args.parameter}",
    ]
''',
     r'''        f"# limit_parameter = {args.parameter}",
        f"# background_contract = {BACKGROUND_CONTRACT}",
        f"# background_methods = DY:{args.dy_method};QCD:{args.qcd_method}",
    ]
'''),
    ('correct symmetric QCD comment',
     r'''    # QCD event yield.  The same parameter receives an asymmetric Gaussian
    # constraint in event-yield units.
''',
     r'''    # QCD event yield.  The same parameter receives a symmetric Gaussian
    # constraint with sigma=max(sigma_down,sigma_up), in event-yield units.
'''),
    ('reject obsolete background cards before target execution',
     r'''    for card in cards:
        label, mass = extract_mass_from_card(card, target)
        clean_mass_outputs(outdir, tag, label, args.task)
''',
     r'''    # Preflight all cards for this target before cleaning or executing a mass.
    for card in cards:
        _validate_background_card(args, card)

    for card in cards:
        label, mass = extract_mass_from_card(card, target)
        clean_mass_outputs(outdir, tag, label, args.task)
'''),
)


class UpdateError(RuntimeError):
    pass


def git_blob_id(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def updated_source(source):
    """Exact anchored replacements; the CLI separately enforces the Git blob ID."""
    updated = source
    for label, before, after in REPLACEMENTS:
        count = updated.count(before)
        if count != 1:
            raise UpdateError(f"{label}: expected one source anchor, found {count}; no changes written.")
        updated = updated.replace(before, after, 1)
    compile(updated, "limit_workflow.py", "exec")
    return updated


def unified_diff(source, updated):
    return "".join(difflib.unified_diff(
        source.splitlines(keepends=True), updated.splitlines(keepends=True),
        fromfile="a/NIsoMuon/limit_workflow.py", tofile="b/NIsoMuon/limit_workflow.py",
    ))


def atomic_write(path, data, mode):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def preserve_file(path, content, mode):
    if path.exists():
        if path.read_bytes() != content:
            raise UpdateError(f"Refusing to overwrite an existing, different backup/patch: {path}")
        return
    with path.open("xb") as stream:
        stream.write(content)
    os.chmod(path, mode)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("workflow", type=Path, help="Existing NIsoMuon/limit_workflow.py")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Back up and update the original file")
    mode.add_argument("--output", type=Path, help="Write a separate complete updated workflow")
    args = parser.parse_args(argv)
    try:
        target = args.workflow.expanduser().resolve()
        original = target.read_bytes()
        source = original.decode("utf-8")
        actual = git_blob_id(original)
        if MARKER in source:
            print("The background-contract marker is already present; no changes made.")
            return 0
        if actual != EXPECTED_GIT_BLOB:
            raise UpdateError(
                f"Input revision mismatch: Git blob {actual}, expected {EXPECTED_GIT_BLOB}. "
                f"This updater was reviewed against commit {BASE_COMMIT}. "
                "Keep your local changes; do not overwrite them with an older file."
            )
        updated = updated_source(source)
        patch = unified_diff(source, updated)
        updated_bytes = updated.encode("utf-8")
        file_mode = stat.S_IMODE(target.stat().st_mode)
        if args.apply:
            backup = target.with_name(target.stem + ".before_background_20260907" + target.suffix)
            patch_path = target.with_name(target.stem + ".background_20260907.patch")
            for path, content in ((backup, original), (patch_path, patch.encode("utf-8"))):
                if path.exists() and path.read_bytes() != content:
                    raise UpdateError(f"Refusing to overwrite an existing, different artefact: {path}")
            preserve_file(backup, original, file_mode)
            preserve_file(patch_path, patch.encode("utf-8"), 0o644)
            if target.read_bytes() != original:
                raise UpdateError("The workflow changed while preparing the update; no replacement performed.")
            atomic_write(target, updated_bytes, file_mode)
            print(f"Updated: {target}")
            print(f"Backup:  {backup}")
            print(f"Patch:   {patch_path}")
        elif args.output:
            output = args.output.expanduser().resolve()
            if output == target or output.exists():
                raise UpdateError("--output must name a new file; use --apply to update the original safely.")
            if not output.parent.is_dir():
                raise UpdateError(f"Output directory does not exist: {output.parent}")
            preserve_file(output, updated_bytes, file_mode)
            print(f"Updated workflow copy: {output}")
        else:
            sys.stdout.write(patch)
        return 0
    except (OSError, UnicodeError, UpdateError, SyntaxError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
