#!/usr/bin/env python3
"""Patch an already background-updated limit_workflow.py for zero DY source bins.

Policy:
  * LightJetSource > 0: keep existing LightJetStat lnN, NFStat Gaussian rateParam,
    and NFModel lnN.
  * LightJetSource == 0 with finite source Sumw2 uncertainty: DY nominal stays 0,
    LightJetStat becomes an absolute-yield Gaussian-constrained positive rateParam
    with sigma = NF_aMC * sigma(LightJetSource), and NFStat/NFModel are disabled
    for that channel.

Run from NIsoMuon:
  python3 update_zero_dy_handling.py limit_workflow.py --apply
  python3 -m py_compile limit_workflow.py
"""
from __future__ import annotations

import argparse
import difflib
import os
from pathlib import Path
import stat
import sys
import tempfile

OLD_MARKER = 'BACKGROUND_CONTRACT = "SKPlotMaker-NF-SS-20260907-v1"'
NEW_MARKER = 'BACKGROUND_CONTRACT = "SKPlotMaker-NF-SS-20260907-v2-zeroDY"'

REPLACEMENTS = (
    (
        "background contract marker",
        OLD_MARKER,
        NEW_MARKER,
    ),
    (
        "allow zero DY source in strict audit",
        '''    _background_check(\n        args, audit, warnings, "background/DY_source_positive", year, source.value > 0.0,\n        f"DY LightJetSource integral is {source.value:g} in [{low:g},{high:g}] GeV. "\n        "A non-positive subtracted source cannot define the current multiplicative "\n        "LightJetStat model. Non-strict mode retains the historical rate floor.",\n    )\n''',
        '''    _background_check(\n        args, audit, warnings, "background/DY_source_nonnegative", year, source.value >= 0.0,\n        f"DY LightJetSource integral is {source.value:g} in [{low:g},{high:g}] GeV. "\n        "Negative source yields must be clamped to zero by the DY producer before "\n        "building the statistical model.",\n    )\n''',
    ),
    (
        "zero-DY nuisance construction",
        '''        if args.dy_method == "data-driven":\n            light_name = "DY_LightJetStat"\n            nuis[light_name] = {p: "-" for p in PROCESSES}\n            nuis[light_name]["DY"] = lnn_from_symmetric_error(\n                dy_lightjet_yield or 0.0,\n                dy_lightjet_error or 0.0,\n                args.ratio_floor,\n                args.ignore_rel_below,\n            )\n\n            model_name = "DY_NFModel"\n            nuis[model_name] = {p: "-" for p in PROCESSES}\n            rel_model = dy_nf_model_rel or 0.0\n            if rel_model >= args.ignore_rel_below and rel_model > 0.0:\n                kd = max(1.0 - rel_model, args.ratio_floor)\n                ku = 1.0 + rel_model\n                nuis[model_name]["DY"] = (\n                    f"{format_kappa(kd)}/{format_kappa(ku)}"\n                )\n''',
        '''        if args.dy_method == "data-driven":\n            light_name = "DY_LightJetStat"\n            nuis[light_name] = {p: "-" for p in PROCESSES}\n            source_yield = dy_lightjet_yield or 0.0\n            source_error = dy_lightjet_error or 0.0\n            # A zero source cannot support a multiplicative lnN.  Its finite\n            # Sumw2 uncertainty is emitted below as an additive absolute-yield\n            # Gaussian rateParam with sigma = NF_aMC * sigma(source).\n            if source_yield > 0.0:\n                nuis[light_name]["DY"] = lnn_from_symmetric_error(\n                    source_yield,\n                    source_error,\n                    args.ratio_floor,\n                    args.ignore_rel_below,\n                )\n\n            model_name = "DY_NFModel"\n            nuis[model_name] = {p: "-" for p in PROCESSES}\n            rel_model = dy_nf_model_rel or 0.0\n            # NFModel is multiplicative and has no first-order effect when the\n            # pre-NF source is exactly zero, so it is disabled in that channel.\n            if source_yield > 0.0 and rel_model >= args.ignore_rel_below and rel_model > 0.0:\n                kd = max(1.0 - rel_model, args.ratio_floor)\n                ku = 1.0 + rel_model\n                nuis[model_name]["DY"] = (\n                    f"{format_kappa(kd)}/{format_kappa(ku)}"\n                )\n''',
    ),
    (
        "zero-DY card base rate",
        '''        if args.dy_method == "data-driven" and process == "DY":\n            # The constrained NF rateParam below multiplies this pre-NF source.\n            return max(channel.dy_lightjet_yield or 0.0, args.rate_floor)\n''',
        '''        if args.dy_method == "data-driven" and process == "DY":\n            source_yield = channel.dy_lightjet_yield or 0.0\n            source_error = channel.dy_lightjet_error or 0.0\n            nf = channel.dy_nf_amc or 0.0\n            if source_yield == 0.0 and source_error > 0.0 and nf > 0.0:\n                # The zero-source LightJetStat rateParam below is the absolute DY\n                # yield, so the base process rate must be unity.\n                return 1.0\n            # Otherwise the constrained NF rateParam multiplies the pre-NF source.\n            return max(source_yield, args.rate_floor)\n''',
    ),
    (
        "skip NFStat in zero-source channels",
        '''        for channel in channels:\n            nf = channel.dy_nf_amc\n            nf_error = channel.dy_nf_amc_error\n            if nf is None or nf <= 0.0:\n                raise WorkflowError(f"Missing/non-positive DY aMC NF for {channel.year}.")\n            if nf_error is None or nf_error <= 0.0:\n                raise WorkflowError(f"Missing/non-positive DY NFStat for {channel.year}.")\n            upper = max(\n                nf + DY_NF_RATEPARAM_NSIGMA * nf_error,\n                2.0 * nf,\n                1.0e-6,\n            )\n''',
        '''        for channel in channels:\n            nf = channel.dy_nf_amc\n            nf_error = channel.dy_nf_amc_error\n            if nf is None or nf <= 0.0:\n                raise WorkflowError(f"Missing/non-positive DY aMC NF for {channel.year}.")\n            if nf_error is None or nf_error <= 0.0:\n                raise WorkflowError(f"Missing/non-positive DY NFStat for {channel.year}.")\n            if (channel.dy_lightjet_yield or 0.0) == 0.0:\n                # Multiplying a zero source by a varied NF still gives zero.\n                # The finite source uncertainty is handled additively below.\n                continue\n            upper = max(\n                nf + DY_NF_RATEPARAM_NSIGMA * nf_error,\n                2.0 * nf,\n                1.0e-6,\n            )\n''',
    ),
    (
        "add zero-source Gaussian block",
        '''    # Additive QCD functional-form uncertainty.  The QCD process has base\n''',
        '''    # Zero-source DY LightJetStat: absolute Gaussian uncertainty.  The DY\n    # process has base rate=1 only for these channels, so this rateParam is the\n    # absolute DY yield.  NFStat and NFModel are deliberately absent here.\n    if args.dy_method == "data-driven":\n        additive_dy_lines: List[str] = []\n        for channel in channels:\n            source_yield = channel.dy_lightjet_yield or 0.0\n            source_error = channel.dy_lightjet_error or 0.0\n            nf = channel.dy_nf_amc or 0.0\n            if source_yield != 0.0 or source_error <= 0.0 or nf <= 0.0:\n                continue\n            sigma = abs(nf) * source_error\n            if sigma <= 0.0 or not math.isfinite(sigma):\n                continue\n            upper = max(10.0 * sigma, 1.0e-6)\n            nuisance_name = nuisance_global_name("DY_LightJetStat", channel.year)\n            bin_name = f"bin_{channel.year}"\n            additive_dy_lines.append(\n                f"{nuisance_name} rateParam {bin_name} DY 0 [0,{format_number(upper)}]"\n            )\n            additive_dy_lines.append(\n                f"{nuisance_name} param 0 {format_number(sigma)}"\n            )\n        if additive_dy_lines:\n            lines.append("# Additive Gaussian DY LightJetStat for zero central source")\n            lines.extend(additive_dy_lines)\n\n    # Additive QCD functional-form uncertainty.  The QCD process has base\n''',
    ),
    (
        "card policy comment",
        '''        "# data-driven DY: light-jet data source x aMC NF; NFStat=Gaussian rateParam, LightJetStat=source Sumw2, NFModel=aMC-vs-MG; no generic DY_stat",\n''',
        '''        "# data-driven DY: light-jet data source x aMC NF; positive source: LightJetStat=lnN, NFStat=Gaussian rateParam, NFModel=lnN; zero source: LightJetStat=additive Gaussian, NFStat/NFModel disabled",\n''',
    ),
    (
        "card validator zero-source exception",
        '''            if not (has_rate and has_constraint):\n                raise WorkflowError(f"{card}: DY NFStat needs both rateParam and Gaussian param for {year}.")\n''',
        '''            if not (has_rate and has_constraint):\n                light_name = nuisance_global_name("DY_LightJetStat", year)\n                has_zero_rate = any(\n                    len(row) >= 5 and row[:4] == [light_name, "rateParam", bin_name, "DY"]\n                    for row in rows\n                )\n                has_zero_constraint = any(\n                    len(row) >= 4 and row[:2] == [light_name, "param"]\n                    for row in rows\n                )\n                if not (has_zero_rate and has_zero_constraint):\n                    raise WorkflowError(\n                        f"{card}: DY needs either NFStat rateParam+param or zero-source "\n                        f"additive LightJetStat rateParam+param for {year}."\n                    )\n''',
    ),
    (
        "card contract comment value",
        '''        f"# background_contract = {BACKGROUND_CONTRACT}",\n''',
        '''        f"# background_contract = {BACKGROUND_CONTRACT}",\n''',
    ),
)


def apply_replacements(text: str) -> str:
    if NEW_MARKER in text:
        raise RuntimeError("zero-DY update is already applied")
    if OLD_MARKER not in text:
        raise RuntimeError(
            "expected background-interface v1 marker not found; first apply "
            "update_limit_backgrounds.py to this workflow"
        )
    out = text
    for label, old, new in REPLACEMENTS:
        if old == new:
            continue
        count = out.count(old)
        if count != 1:
            raise RuntimeError(f"{label}: expected exactly one match, found {count}")
        out = out.replace(old, new, 1)
    return out


def atomic_write(path: Path, text: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.chmod(tmp_name, mode)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="limit_workflow.py")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    path = Path(args.path)
    before = path.read_text(encoding="utf-8")
    after = apply_replacements(before)
    diff = "".join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True),
        fromfile=str(path), tofile=str(path) + ".zeroDY",
    ))
    if not args.apply:
        sys.stdout.write(diff)
        return 0
    backup = path.with_name(path.name + ".before_zeroDY_20260907")
    backup.write_text(before, encoding="utf-8")
    atomic_write(path, after)
    print(f"[UPDATED] {path}")
    print(f"[BACKUP]  {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
