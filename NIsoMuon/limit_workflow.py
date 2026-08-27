#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone NIsoMuon Run-2/Run-3 counting-limit workflow, revision timestamp 20260826_0541.

This ONE file performs all stages:
  1. read ROOT histograms and audit all required inputs;
  2. build per-era, Run-2, Run-3, and combined Run-2+Run-3 datacards;
  3. apply a numerically stable internal parameterisation for alpha_qZ';
  4. run Combine limits, FitDiagnostics/pulls, and impacts;
  5. convert alpha limit/impact JSON files back to physical alpha_qZ';
  6. make the limit and impact plots.

Uncertainty policy in this revision
-----------------------------------
* Data-driven QCD has QCD_norm and QCD_shape only.  No QCD_stat is derived
  from the fitted functional template.  QCD_stat is retained only in QCD-MC mode.
* Data-driven DY uses the constant-NF estimate.  DY_NFStat and
  DY_LightJetStat are read from the explicitly separated templates; no generic
  DY_stat term is constructed from the nominal histogram error.
* Generic inclusive tt_xsec and ST_xsec nuisances are removed.  Generator PDF,
  alpha_s, and renormalisation/factorisation-scale weights are used for their
  corresponding theory sources, while a separate ttbar top-mass normalisation
  nuisance covers the cross-section dependence on the assumed top-quark mass.
* DeepJet fixed-WP uncertainties follow the BTV multi-era prescription.
* Run-2, Run-3, and combined Run-2+Run-3 cards are supported.
* Heavy-flavour (b/c) and light-flavour b-tag components are kept separate.
  Their correlated pieces are shared within Run 2 or within Run 3, while the
  uncorrelated pieces are decorrelated by era.
* A separate asymmetric ttbar top-mass cross-section uncertainty is included;
  generator scale/PDF/alpha_s terms remain separate.
* The PDF set is NNPDF31_nnlo_as_0118_mc_hessian_pdfas.  Its 100 PDF error
  members are a symmetric-Hessian basis.  Following the existing SKFlat input
  convention, PDFError0..99 are taken to correspond to LHAPDF members 1--100
  and are combined in quadrature with
  respect to the nominal yield.  They are not treated as Monte
  Carlo replicas, and no replica RMS or percentile prescription is used.
* Renormalisation- and factorisation-scale variations are retained as separate
  process-specific nuisances.  The simultaneous (muR,muF)=(0.5,0.5) and (2,2)
  variations are read only to validate the factorised response; they are not
  added as a third nuisance.  The two antipodal scale combinations are excluded.
* The Run-2 L1 ECAL prefiring correction is varied with the NanoAOD
  L1PreFiringWeight Up/Down branches for 2016preVFP, 2016postVFP, and 2017.
  The corresponding nuisance parameters are decorrelated between data-taking eras.

Planned Run-2, Run-3, and full-combination commands (sigFit files stored in ./sigfit_inputs):

  for target in run2 run3 full; do
    python3 limit_workflow.py \
      --stage all --target ${target} --parameter alpha --mode blind --task all \
      --sigfit-dir ./sigfit_inputs \
      --impact-parallel 12 --r-max 100 --strict --allow-negative-r --r-min -2 \
      &> limit_blind_${target}.log &
  done

The current production provides generator PDF/scale/alpha_s variations for tt and
single-top backgrounds. Signal theory variations are not requested by default; signal
experimental variations are retained.

For negative-r impact diagnostics, the lower POI range is automatically expanded
when the fitted POI reaches the requested lower boundary, while remaining inside
the nominal positive-yield domain of the counting model.

Physics interpretation for alpha mode:

  N_sig(alpha_qZ') = alpha_qZ' * N_sig(alpha_qZ'=1).

To avoid numerical underflow, Combine uses an internal POI r_internal with

  alpha_qZ' = r_internal * alpha_internal_unit.

By default alpha_internal_unit is chosen independently for each card so that
r_internal=1 corresponds to 25 selected signal events.  The default maximum
internal signal yield is 2500 events, therefore the default alpha-mode rMax is
2500/25 = 100.  Datacard comments, alpha_internal_scaling.csv/json, physical
limit JSON, and physical impact JSON record the conversion.  The final plotted
values are physical alpha_qZ'.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# -------------------------------------------------------------------------------------------------
# Analysis configuration
# -------------------------------------------------------------------------------------------------

RUN2_ERAS: Tuple[str, ...] = ("2016preVFP", "2016postVFP", "2017", "2018")
RUN3_ERAS: Tuple[str, ...] = ("2022", "2022EE", "2023", "2023BPix")
YEARS: Tuple[str, ...] = RUN2_ERAS + RUN3_ERAS
COMBINED_TARGETS = frozenset({"Run2", "Run3", "Run2Run3"})
PROCESSES: Tuple[str, ...] = ("sig", "QCD", "tt", "ST", "DY", "Others")
PROC_ID: Dict[str, int] = {"sig": 0, "QCD": 1, "tt": 2, "ST": 3, "DY": 4, "Others": 5}

DEFAULT_BASE_DIR = "/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon"
DEFAULT_REGION = "OS_POGMedium_tight_BJet_NIsoDimuon"
DEFAULT_PLOT_LIMITS_PY = (
    "/data6/Users/joonblee/higgs_combine/CMSSW_14_1_0_pre4/"
    "src/NIsoMuon/python/plotLimits.py"
)

LUMI_FB: Dict[str, float] = {
    "2016preVFP": 19.5,
    "2016postVFP": 16.8,
    "2017": 41.5,
    "2018": 59.8,
    "2022": 7.98,
    "2022EE": 26.67,
    "2023": 17.7,
    "2023BPix": 9.5,
}
LUMI_LNN: Dict[str, float] = {
    "2016preVFP": 1.012,
    "2016postVFP": 1.012,
    "2017": 1.023,
    "2018": 1.025,
    "2022": 1.014,
    "2022EE": 1.014,
    "2023": 1.013,
    "2023BPix": 1.013,
}

# The statistical interpretation is restricted to the blinded search interval.
SEARCH_MASS_MIN = 10.4
SEARCH_MASS_MAX = 80.0

# Signal-resolution inputs are generated by sigFit and are intentionally read
# from external JSON/CSV files rather than duplicated as hard-coded numbers.
# JSON is preferred when both representations are present.
DEFAULT_RESOLUTION: Dict[str, Tuple[float, float]] = {}
RESOLUTION_INPUT_NAMES: Tuple[str, ...] = (
    "resolution_coefficients.json",
    "resolution_coefficients.csv",
)
SIGFIT_RESULT_INPUT_NAMES: Tuple[str, ...] = (
    "sigFit_results.json",
    "sigFit_results.csv",
)

# Updated ttbar NNLO+NNLL reference cross sections supplied by the user.
# Only the top-mass dependence is used here; scale, PDF and alpha_s are already
# represented by the generator-theory nuisances.  One common tt_mass nuisance
# is used across Run 2 and Run 3, with an energy-dependent response.
TTBAR_MASS_XSEC: Dict[str, Tuple[float, float, float]] = {
    # run: (central pb, downward shift pb, upward shift pb)
    "Run2": (833.9, 22.5, 23.2),
    "Run3": (923.6, 24.6, 25.4),
}

EXP_SYST: Dict[str, Tuple[str, str]] = {
    "jer": ("JetResDown", "JetResUp"),
    "jes": ("JetEnDown", "JetEnUp"),
    "pu": ("PUDown", "PUUp"),
    "mu_trig_sf": ("MuonTriggerSFDown", "MuonTriggerSFUp"),
    "mu_id_sf": ("MuonIDSFDown", "MuonIDSFUp"),
    "mu_scale": ("MuonEnDown", "MuonEnUp"),
}

L1_PREFIRE_SYST: Dict[str, Tuple[str, str]] = {
    "l1prefire": ("L1PrefireDown", "L1PrefireUp"),
}
L1_PREFIRE_YEARS = frozenset({"2016preVFP", "2016postVFP", "2017", "2018"})

# BTV fixed-WP multi-era prescription.  Local names distinguish the four
# correctionlib source families.  nuisance_global_name() below keeps the
# correlated components common across Run-2 eras and decorrelates only the
# uncorrelated components.
BTAG_SYST: Dict[str, Tuple[str, str]] = {
    "btagSFbc_correlated": ("BTagHFCorrDown", "BTagHFCorrUp"),
    "btagSFbc_uncorrelated": ("BTagHFUncorrDown", "BTagHFUncorrUp"),
    "btagSFlight_correlated": ("BTagLFCorrDown", "BTagLFCorrUp"),
    "btagSFlight_uncorrelated": ("BTagLFUncorrDown", "BTagLFUncorrUp"),
}
QCD_SYST: Dict[str, Tuple[str, str]] = {
    "QCD_norm": ("NormDown", "NormUp"),
    "QCD_shape": ("ShapeDown", "ShapeUp"),
}
# Nominal data-driven DY treatment: constant normalisation factor (NF).
# The ROOT producer retains TFDown/TFUp directory names for compatibility,
# but these templates represent the NF numerator-statistics uncertainty.
DY_NF_SYST: Tuple[str, str, str] = ("DY_NFStat", "TFDown", "TFUp")
DY_LIGHTJET_STAT_SYST: Tuple[str, str, str] = (
    "DY_LightJetStat", "LightJetStatDown", "LightJetStatUp"
)
DATA_DRIVEN_DY_NUISANCES: Tuple[str, ...] = (
    "DY_NFStat", "DY_LightJetStat"
)

# PDF configuration used by the simulated tt and single-top samples.
# Input convention: PDFError0..99 preserve the order of LHAPDF members 1..100.
PDF_SET_NAME = "NNPDF31_nnlo_as_0118_mc_hessian_pdfas"
PDF_EIGENVECTOR_COUNT = 100
PDF_EIGENVECTOR_INDICES: Tuple[int, ...] = tuple(range(PDF_EIGENVECTOR_COUNT))
ALPHAS_VARIATION_INDICES: Tuple[int, int] = (0, 1)

# SKFlat stores the standard nine-point LHE scale grid as PDFScale0..8:
#   0=(muR,muF)=(1,1), 1=(1,2), 2=(1,0.5), 3=(2,1), 4=(2,2),
#   5=(2,0.5), 6=(0.5,1), 7=(0.5,2), 8=(0.5,0.5).
# The nominal model uses separate muF and muR nuisances.  The diagonal pair is
# validation-only because it is not an independent scale direction.
SCALE_CENTRAL_INDEX = 0
EXCLUDED_ANTIPODAL_SCALE_INDICES = frozenset({5, 7})
SCALE_VARIATION_PAIRS: Dict[str, Tuple[int, int]] = {
    # name: (Down member, Up member)
    "muF": (2, 1),
    "muR": (6, 3),
    "muRmuF": (8, 4),
}
SCALE_NUISANCE_DIRECTIONS: Tuple[str, ...] = ("muF", "muR")
SCALE_DIAGONAL_DIRECTION = "muRmuF"
THEORY_CENTRAL_TOLERANCE = 0.01
SCALE_DIAGONAL_LOG_TOLERANCE = 0.50

ROOT_FILES: Dict[str, str] = {
    "QCD_data": "NIsoMuon_SS_fit.root",
    "QCD_mc": "NIsoMuon_QCD_Inclusive.root",
    "tt": "NIsoMuon_tt.root",
    "ST": "NIsoMuon_ST.root",
    "DY_data": "NIsoMuon_DYJets_est.root",
    "DY_mc": "NIsoMuon_DYJets_Inclusive.root",
    "Others": "NIsoMuon_Others.root",
}

EXPECTED_LIMIT_KEYS: Tuple[str, ...] = ("exp-2", "exp-1", "exp0", "exp+1", "exp+2")


# -------------------------------------------------------------------------------------------------
# Data structures and errors
# -------------------------------------------------------------------------------------------------

class WorkflowError(RuntimeError):
    pass


@dataclass
class YieldResult:
    value: float
    error: float
    first_bin: int
    last_bin: int


@dataclass
class ChannelResult:
    year: str
    mass_label: str
    mass: float
    window_low: float
    window_high: float
    observation: int
    raw_rates: Dict[str, float]
    rates: Dict[str, float]
    nuisances: Dict[str, Dict[str, str]]
    warnings: List[str] = field(default_factory=list)


@dataclass
class CardInfo:
    path: Path
    target: str
    mass_label: str
    mass: float
    parameter: str
    reference_signal_yield: float
    signal_rate_scale: float
    alpha_internal_unit: Optional[float]
    internal_signal_yield: float
    reference_xsec_pb: Optional[float]


@dataclass
class LimitOutputs:
    observed: Path
    expected: Path


@dataclass
class AuditEntry:
    found: int = 0
    expected: int = 0
    missing: List[str] = field(default_factory=list)


class AuditLog:
    def __init__(self) -> None:
        self.entries: Dict[str, AuditEntry] = defaultdict(AuditEntry)

    def add(self, key: str, label: str, ok: bool) -> None:
        entry = self.entries[key]
        entry.expected += 1
        if ok:
            entry.found += 1
        else:
            entry.missing.append(label)

    def print_and_check(self, mass_label_in: str, strict: bool) -> None:
        print(f"[syst-check] M-{mass_label_in}")
        failures: List[str] = []
        for key in sorted(self.entries):
            entry = self.entries[key]
            if entry.found == entry.expected:
                suffix = f" ({entry.found}/{entry.expected})" if entry.expected > 2 else ""
                print(f"  - {key}: OK{suffix}")
            else:
                preview = ", ".join(entry.missing[:8])
                if len(entry.missing) > 8:
                    preview += f", ... +{len(entry.missing)-8}"
                print(f"  - {key}: MISSING ({entry.found}/{entry.expected}; {preview})")
                failures.append(
                    f"Required templates are missing for {key}: found {entry.found}/{entry.expected}; "
                    f"missing {preview}."
                )
        if strict and failures:
            raise WorkflowError(
                f"Strict systematic audit failed for M-{mass_label_in} with {len(failures)} "
                "missing requirement(s):\n  - " + "\n  - ".join(failures)
            )


# -------------------------------------------------------------------------------------------------
# Generic helpers
# -------------------------------------------------------------------------------------------------

def normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def canonical_target(value: str) -> str:
    key = normalise_key(value)
    if key in {"run2", "fullrun2"}:
        return "run2"
    if key in {"run3", "fullrun3"}:
        return "run3"
    if key in {"full", "run23", "run2run3", "combined", "run2plus3"}:
        return "full"
    if key in {"era", "eras", "perera", "peryear", "years"}:
        return "eras"
    if key in {"all"}:
        return "all"
    for year in YEARS:
        if value == year or key == normalise_key(year):
            return year
    raise argparse.ArgumentTypeError(
        "Use run2, run3, full/run2+3, eras, all, or one data-taking era."
    )


def canonical_parameter(value: str) -> str:
    key = normalise_key(value)
    if key in {"alpha", "alphaqz", "alphaqzp", "alphaqzprime", "coupling"}:
        return "alpha"
    if key in {"yield", "event", "events", "eventnumber", "nsig", "signalyield"}:
        return "yield"
    if key in {"xsec", "crosssection", "sigma"}:
        return "xsec"
    raise argparse.ArgumentTypeError("Use alpha, yield, or xsec.")


def canonical_process(value: str) -> str:
    key = normalise_key(value)
    mapping = {
        "sig": "sig", "signal": "sig",
        "qcd": "QCD",
        "tt": "tt", "ttbar": "tt",
        "st": "ST", "singletop": "ST", "tw": "ST",
        "dy": "DY", "drellyan": "DY",
        "others": "Others", "other": "Others",
    }
    if key not in mapping:
        raise argparse.ArgumentTypeError(f"Unknown process: {value}")
    return mapping[key]


def parse_process_list(value: str) -> Tuple[str, ...]:
    if not value.strip():
        return tuple()
    out: List[str] = []
    for token in re.split(r"[,;:\s]+", value.strip()):
        if not token:
            continue
        proc = canonical_process(token)
        if proc not in out:
            out.append(proc)
    return tuple(out)


def parse_float_list(value: str) -> List[float]:
    out: List[float] = []
    for token in re.split(r"[,;:\s]+", value.strip()):
        if not token:
            continue
        out.append(float(token.replace("M-", "").replace("p", ".")))
    if not out:
        raise argparse.ArgumentTypeError("At least one mass is required.")
    return out


def selected_targets(target: str) -> List[str]:
    if target == "run2":
        return ["Run2"]
    if target == "run3":
        return ["Run3"]
    if target == "full":
        return ["Run2Run3"]
    if target == "eras":
        return list(YEARS)
    if target == "all":
        return ["Run2", "Run3", "Run2Run3", *YEARS]
    return [target]


def target_years(target: str) -> Tuple[str, ...]:
    if target == "Run2":
        return RUN2_ERAS
    if target == "Run3":
        return RUN3_ERAS
    if target == "Run2Run3":
        return YEARS
    return (target,)


def years_needed_for_request(args: argparse.Namespace) -> Tuple[str, ...]:
    needed = set()
    for target in selected_targets(args.target):
        needed.update(target_years(target))
    return tuple(year for year in YEARS if year in needed)


def run_group(year: str) -> str:
    if year in RUN2_ERAS:
        return "Run2"
    if year in RUN3_ERAS:
        return "Run3"
    raise WorkflowError(f"Unknown data-taking era: {year}")


def lumi_group(year: str) -> str:
    if year.startswith("2016"):
        return "2016"
    if year in {"2022", "2022EE"}:
        return "2022"
    if year in {"2023", "2023BPix"}:
        return "2023"
    return year


def tt_mass_lnn(year: str) -> Tuple[float, float]:
    central, down_shift, up_shift = TTBAR_MASS_XSEC[run_group(year)]
    return (central - down_shift) / central, (central + up_shift) / central


def mass_label(mass: float) -> str:
    if abs(mass - round(mass)) < 1.0e-9:
        return str(int(round(mass)))
    return f"{mass:.8g}".replace(".", "p")


def format_mass_for_combine(mass: float) -> str:
    return str(int(round(mass))) if abs(mass - round(mass)) < 1.0e-9 else f"{mass:.8g}"


def format_number(value: float) -> str:
    if not math.isfinite(value):
        raise WorkflowError(f"Non-finite number: {value}")
    if value == 0.0:
        return "0"
    if abs(value) >= 1.0e5 or abs(value) < 1.0e-4:
        return f"{value:.9e}"
    return f"{value:.9f}"


def format_kappa(value: float) -> str:
    return f"{value:.6g}"


def pad_row(entries: Sequence[Any], width: int = 19) -> str:
    return " ".join(str(x).ljust(width) for x in entries).rstrip()


def require_command(command: str) -> None:
    if shutil.which(command) is None:
        raise WorkflowError(f"Required command is not available in PATH: {command}")


def resolve_path(path: str, description: str) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(path)))
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    if not p.exists():
        raise WorkflowError(f"{description} does not exist: {p}")
    return p


def run_command(
    command: Sequence[str],
    cwd: Path,
    *,
    allow_failure: bool = False,
    stdout_path: Optional[Path] = None,
) -> int:
    cwd.mkdir(parents=True, exist_ok=True)
    printable = " ".join(str(x) for x in command)
    print(f"[RUN] (cd {cwd} && {printable})", flush=True)
    if stdout_path is None:
        completed = subprocess.run(list(command), cwd=str(cwd), check=False)
    else:
        with stdout_path.open("w") as stream:
            completed = subprocess.run(
                list(command), cwd=str(cwd), stdout=stream,
                stderr=subprocess.STDOUT, check=False
            )
    if completed.returncode != 0 and not allow_failure:
        raise WorkflowError(f"Command failed with exit code {completed.returncode}: {printable}")
    return completed.returncode


def nonfatal_or_raise(args: argparse.Namespace, message: str) -> None:
    if args.strict:
        raise WorkflowError(message)
    print(f"[WARNING] {message}", file=sys.stderr)


# -------------------------------------------------------------------------------------------------
# ROOT input handling
# -------------------------------------------------------------------------------------------------

class RootReader:
    def __init__(self) -> None:
        try:
            import ROOT  # type: ignore
        except Exception as exc:
            raise WorkflowError(
                "Could not import PyROOT. Run after cmsenv in a ROOT/CMSSW environment. "
                f"Original error: {exc}"
            )
        self.ROOT = ROOT
        ROOT.gROOT.SetBatch(True)
        try:
            ROOT.TH1.AddDirectory(False)
        except Exception:
            pass
        self.files: Dict[str, Any] = {}
        self.integral_cache: Dict[Tuple[str, str, float, float], Optional[YieldResult]] = {}

    def close(self) -> None:
        for f in self.files.values():
            try:
                f.Close()
            except Exception:
                pass
        self.files.clear()

    def _file(self, filename: str):
        if filename in self.files:
            return self.files[filename]
        if not os.path.exists(filename):
            self.files[filename] = None
            return None
        f = self.ROOT.TFile.Open(filename, "READ")
        if not f or f.IsZombie():
            self.files[filename] = None
            return None
        self.files[filename] = f
        return f

    def integral(self, filename: str, hist_path_in: str, low: float, high: float) -> Optional[YieldResult]:
        key = (filename, hist_path_in, round(low, 12), round(high, 12))
        if key in self.integral_cache:
            return self.integral_cache[key]
        f = self._file(filename)
        if f is None:
            self.integral_cache[key] = None
            return None
        h = f.Get(hist_path_in)
        if not h:
            self.integral_cache[key] = None
            return None
        axis = h.GetXaxis()
        nbins = h.GetNbinsX()
        low_c = max(low, axis.GetXmin())
        high_c = min(high, axis.GetXmax())
        if high_c <= low_c:
            result = YieldResult(0.0, 0.0, 1, 0)
            self.integral_cache[key] = result
            return result
        eps = 1.0e-9 * max(1.0, abs(high_c - low_c))
        first = max(1, min(nbins, axis.FindFixBin(low_c + eps)))
        last = max(1, min(nbins, axis.FindFixBin(high_c - eps)))
        err = ctypes.c_double(0.0)
        value = float(h.IntegralAndError(first, last, err))
        result = YieldResult(value, float(err.value), first, last)
        self.integral_cache[key] = result
        return result


def root_dir(args: argparse.Namespace, year: str, collection: str = "") -> str:
    parts = [args.base_dir]
    if collection:
        parts.append(collection)
    parts.append(year)
    if args.trigger:
        parts.append(args.trigger)
    return os.path.join(*parts)


def hist_path(region: str) -> str:
    return f"{region}/Dilepton_Mass___{region}"


def syst_region(args: argparse.Namespace, suffix: str) -> str:
    prefix = args.region.replace("_NIsoDimuon", "")
    return f"{prefix}_Syst_{suffix}_NIsoDimuon"


def load_resolution_map_file(path_text: str) -> Dict[str, Tuple[float, float]]:
    path = resolve_path(path_text, "Signal-resolution map")
    out: Dict[str, Tuple[float, float]] = {}
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise WorkflowError("Resolution JSON must be an era-to-coefficients object.")
        for era, value in payload.items():
            if isinstance(value, Mapping):
                if "a" not in value or "b" not in value:
                    raise WorkflowError(f"Resolution JSON entry {era} needs keys a and b.")
                out[str(era)] = (float(value["a"]), float(value["b"]))
            elif isinstance(value, (list, tuple)) and len(value) == 2:
                out[str(era)] = (float(value[0]), float(value[1]))
            else:
                raise WorkflowError(
                    f"Resolution JSON entry {era} must be [a,b] or {{\"a\":...,\"b\":...}}."
                )
    else:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise WorkflowError("Resolution CSV is empty.")
            era_key = next((x for x in reader.fieldnames if normalise_key(x) in {"era", "year", "period"}), None)
            a_key = next(
                (x for x in reader.fieldnames if normalise_key(x) in {"a", "slope", "aslope"}),
                None,
            )
            b_key = next(
                (x for x in reader.fieldnames if normalise_key(x) in {"b", "intercept", "bintercept"}),
                None,
            )
            if era_key is None or a_key is None or b_key is None:
                raise WorkflowError("Resolution CSV needs era,a,b columns.")
            for row in reader:
                out[str(row[era_key]).strip()] = (float(row[a_key]), float(row[b_key]))
    bad = sorted(set(out) - set(YEARS))
    if bad:
        raise WorkflowError("Unknown era(s) in resolution map: " + ", ".join(bad))
    return out


def discover_sigfit_input(
    directory: Path,
    preferred_names: Sequence[str],
    description: str,
) -> Optional[Path]:
    """Find one sigFit JSON/CSV input, preferring the canonical JSON name."""
    if not directory.is_dir():
        raise WorkflowError(f"Signal-fit input directory does not exist: {directory}")

    for name in preferred_names:
        candidate = directory / name
        if candidate.is_file():
            return candidate.resolve()

    # Permit timestamped or otherwise decorated filenames while avoiding a
    # silent choice when several plausible inputs coexist.
    tokens = (
        ("resolution", "coeff")
        if "resolution" in preferred_names[0].lower()
        else ("sigfit", "result")
    )
    for suffix in (".json", ".csv"):
        matches = sorted(
            path.resolve()
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() == suffix
            and all(token in path.name.lower() for token in tokens)
        )
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise WorkflowError(
                f"Several candidate {description} files were found in {directory}: "
                + ", ".join(path.name for path in matches)
                + ". Use --resolution-map or keep only one canonical input."
            )
    return None


def _optional_number(value: Any, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_sigfit_results_file(path_text: str) -> List[Dict[str, Any]]:
    path = resolve_path(path_text, "Signal-fit result file")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if isinstance(payload, Mapping):
            payload = payload.get("results", payload.get("fits", payload))
        if not isinstance(payload, list):
            raise WorkflowError("Signal-fit JSON must contain a list of fit records.")
        rows = payload
    else:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise WorkflowError("Signal-fit CSV is empty.")
            rows = list(reader)

    records: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise WorkflowError(f"Signal-fit row {index} is not an object.")
        normalised = {normalise_key(str(key)): value for key, value in row.items()}
        era = str(
            normalised.get("era", normalised.get("year", normalised.get("period", "")))
        ).strip()
        if not era:
            raise WorkflowError(f"Signal-fit row {index} has no era field.")
        if era not in YEARS:
            raise WorkflowError(f"Signal-fit row {index} has unknown era {era!r}.")

        mass = _optional_number(normalised.get("mass", normalised.get("mz")))
        sigma = _optional_number(normalised.get("sigma"))
        status = int(_optional_number(normalised.get("fitstatus", normalised.get("status")), 999))
        covariance = int(
            _optional_number(
                normalised.get("covariancestatus", normalised.get("covstatus")),
                -1,
            )
        )
        accepted = (
            math.isfinite(mass)
            and math.isfinite(sigma)
            and sigma > 0.0
            and status == 0
            and covariance >= 2
        )
        records.append(
            {
                "era": era,
                "mass": mass,
                "sigma": sigma,
                "fit_status": status,
                "covariance_status": covariance,
                "accepted": accepted,
            }
        )
    return records


def audit_sigfit_results(
    records: Sequence[Mapping[str, Any]],
    needed_years: Sequence[str],
) -> Dict[str, Any]:
    per_era: Dict[str, Dict[str, Any]] = {
        year: {"total": 0, "accepted": 0, "masses": [], "rejected": []}
        for year in needed_years
    }
    for record in records:
        era = str(record["era"])
        if era not in per_era:
            continue
        summary = per_era[era]
        summary["total"] += 1
        mass = float(record["mass"])
        if bool(record["accepted"]):
            summary["accepted"] += 1
            summary["masses"].append(mass)
        else:
            summary["rejected"].append(
                {
                    "mass": mass,
                    "fit_status": int(record["fit_status"]),
                    "covariance_status": int(record["covariance_status"]),
                }
            )

    missing = [year for year, item in per_era.items() if item["total"] == 0]
    insufficient = [year for year, item in per_era.items() if item["accepted"] < 2]
    if missing:
        raise WorkflowError(
            "The signal-fit result file has no records for: " + ", ".join(missing)
        )
    if insufficient:
        raise WorkflowError(
            "Fewer than two accepted signal fits are available for: "
            + ", ".join(insufficient)
        )
    return {
        "total": sum(item["total"] for item in per_era.values()),
        "accepted": sum(item["accepted"] for item in per_era.values()),
        "per_era": per_era,
    }


def prepare_resolution_coefficients(args: argparse.Namespace) -> None:
    input_dir = resolve_path(args.sigfit_dir, "Signal-fit input directory")
    if not input_dir.is_dir():
        raise WorkflowError(f"Signal-fit input path is not a directory: {input_dir}")

    if args.resolution_map:
        resolution_path = resolve_path(args.resolution_map, "Signal-resolution map")
    else:
        resolution_path = discover_sigfit_input(
            input_dir,
            RESOLUTION_INPUT_NAMES,
            "resolution-coefficient",
        )
        if resolution_path is None:
            raise WorkflowError(
                f"No resolution coefficient JSON/CSV was found in {input_dir}. "
                "Expected resolution_coefficients.json or resolution_coefficients.csv."
            )

    coeffs = dict(DEFAULT_RESOLUTION)
    coeffs.update(load_resolution_map_file(str(resolution_path)))
    needed = years_needed_for_request(args)
    missing = [year for year in needed if year not in coeffs]
    if missing:
        raise WorkflowError(
            "Missing signal mass-resolution coefficients for " + ", ".join(missing)
            + ". The expected relation is sigma_m/m = a*m + b."
        )

    results_path = discover_sigfit_input(
        input_dir,
        SIGFIT_RESULT_INPUT_NAMES,
        "signal-fit result",
    )
    if results_path is None:
        raise WorkflowError(
            f"No sigFit result JSON/CSV was found in {input_dir}. "
            "Expected sigFit_results.json or sigFit_results.csv so that the "
            "resolution input can be audited."
        )
    audit = audit_sigfit_results(
        load_sigfit_results_file(str(results_path)),
        needed,
    )

    args.resolution_coefficients = coeffs
    args.resolution_source = resolution_path
    args.sigfit_results_source = results_path
    args.sigfit_audit = audit

    for year in needed:
        rejected = audit["per_era"][year]["rejected"]
        for item in rejected:
            print(
                f"[WARNING] Excluding invalid signal fit from resolution audit: "
                f"{year} M-{item['mass']:g}, status={item['fit_status']}, "
                f"cov={item['covariance_status']}",
                file=sys.stderr,
            )


def relative_resolution(args: argparse.Namespace, year: str, mass: float) -> float:
    try:
        a, b = args.resolution_coefficients[year]
    except (AttributeError, KeyError) as exc:
        raise WorkflowError(f"Signal-resolution coefficients are unavailable for {year}.") from exc
    return a * mass + b


def mass_window(args: argparse.Namespace, year: str, mass: float) -> Tuple[float, float]:
    resolution = relative_resolution(args, year, mass)
    sigma = resolution if args.absolute_resolution else mass * resolution
    low = max(SEARCH_MASS_MIN, mass - args.n_sigma * sigma)
    high = min(SEARCH_MASS_MAX, mass + args.n_sigma * sigma)
    if not high > low:
        raise WorkflowError(
            f"Empty signal window for {year}, M-{mass:g}: [{low:g},{high:g}] GeV"
        )
    return low, high


def signal_mass_from_filename(filename: str) -> Optional[Tuple[str, float]]:
    base = os.path.basename(filename)
    patterns = (
        r"NIsoMuon_Zp_M-([^/]+)\.root$",
        r"NIsoMuon_SkimTree_NIsoMuon_Zp_M-([^/]+)\.root$",
    )
    for pattern in patterns:
        match = re.search(pattern, base)
        if not match:
            continue
        label = match.group(1)
        try:
            return label, float(label.replace("p", "."))
        except ValueError:
            return None
    return None


def scan_signal_files(args: argparse.Namespace) -> Dict[float, Dict[str, str]]:
    found: Dict[float, Dict[str, str]] = defaultdict(dict)
    for year in YEARS:
        directory = root_dir(args, year)
        candidates = sorted(glob.glob(os.path.join(directory, "NIsoMuon_Zp_M-*.root")))
        candidates += sorted(glob.glob(os.path.join(directory, "NIsoMuon_SkimTree_NIsoMuon_Zp_M-*.root")))
        for path in candidates:
            parsed = signal_mass_from_filename(path)
            if parsed is None:
                continue
            _label, mass = parsed
            if year not in found[mass] or os.path.basename(path).startswith("NIsoMuon_Zp_M-"):
                found[mass][year] = path
    return dict(found)


def file_for_process(
    args: argparse.Namespace,
    year: str,
    process: str,
    signal_file: str,
    source: str,
) -> str:
    nominal_directory = root_dir(args, year)
    if process == "sig":
        filename = os.path.basename(signal_file)
    elif process == "QCD":
        filename = ROOT_FILES["QCD_mc" if args.qcd_method == "mc" else "QCD_data"]
    elif process == "DY":
        filename = ROOT_FILES["DY_mc" if args.dy_method == "mc" else "DY_data"]
    else:
        filename = ROOT_FILES[process]

    if source == "nominal":
        return os.path.join(nominal_directory, filename)
    if source in {"experimental", "qcd", "dy"}:
        return os.path.join(root_dir(args, year, "RunSyst"), filename)
    if source == "theory":
        return os.path.join(root_dir(args, year, "RunXSecSyst"), filename)
    raise WorkflowError(f"Unknown ROOT source: {source}")


def read_required(
    reader: RootReader,
    audit: AuditLog,
    key: str,
    label: str,
    filename: str,
    path: str,
    low: float,
    high: float,
) -> Optional[YieldResult]:
    result = reader.integral(filename, path, low, high)
    audit.add(key, label, result is not None)
    return result


# -------------------------------------------------------------------------------------------------
# Uncertainty calculations
# -------------------------------------------------------------------------------------------------

def safe_ratio(value: float, nominal: float, floor: float) -> Optional[float]:
    if nominal <= 0.0 or not math.isfinite(nominal) or not math.isfinite(value):
        return None
    return max(value / nominal, floor)


def lnn_from_down_up(
    nominal: float,
    down: Optional[float],
    up: Optional[float],
    ratio_floor: float,
    ignore_rel_below: float,
) -> str:
    if nominal <= 0.0 or down is None or up is None:
        return "-"
    kd = safe_ratio(down, nominal, ratio_floor)
    ku = safe_ratio(up, nominal, ratio_floor)
    if kd is None or ku is None:
        return "-"
    if max(abs(kd - 1.0), abs(ku - 1.0)) < ignore_rel_below:
        return "-"
    return f"{format_kappa(kd)}/{format_kappa(ku)}"


def lnn_from_single(kappa: float, ignore_rel_below: float) -> str:
    if not math.isfinite(kappa) or kappa <= 0.0 or abs(kappa - 1.0) < ignore_rel_below:
        return "-"
    return format_kappa(kappa)


def lnn_from_symmetric_error(
    nominal: float,
    error: float,
    ratio_floor: float,
    ignore_rel_below: float,
) -> str:
    if nominal <= 0.0 or error <= 0.0 or not math.isfinite(error):
        return "-"
    rel = error / nominal
    if rel < ignore_rel_below:
        return "-"
    return f"{format_kappa(max(1.0-rel, ratio_floor))}/{format_kappa(1.0+rel)}"


def lnn_from_symmetric_hessian(
    nominal: float,
    eigenvector_yields: Sequence[float],
    ratio_floor: float,
    ignore_rel_below: float,
) -> str:
    """Return the symmetric-Hessian PDF uncertainty around the nominal yield."""
    values = [value for value in eigenvector_yields if math.isfinite(value)]
    if nominal <= 0.0 or not values:
        return "-"
    error = math.sqrt(sum((value - nominal) ** 2 for value in values))
    return lnn_from_symmetric_error(
        nominal, error, ratio_floor, ignore_rel_below
    )


def scale_nuisance_name(process: str, direction: str) -> str:
    return f"PDF_scale_{process}_{direction}"


def scale_pair(direction: str) -> Tuple[int, int]:
    try:
        return SCALE_VARIATION_PAIRS[direction]
    except KeyError as exc:
        raise WorkflowError(f"Unknown scale direction: {direction}") from exc


def finite_template_stat_processes(args: argparse.Namespace) -> Tuple[str, ...]:
    """Processes whose nominal TH1 Sumw2 error is a genuine finite-template term."""
    processes: List[str] = ["sig", "tt", "ST", "Others"]
    if args.qcd_method == "mc":
        processes.append("QCD")
    if args.dy_method == "mc":
        processes.append("DY")
    return tuple(processes)


def effective_experimental_processes(args: argparse.Namespace) -> Tuple[str, ...]:
    """Return only processes for which detector-response templates are meaningful."""
    processes: List[str] = []
    for process in args.experimental_processes:
        if process == "QCD" and args.qcd_method == "data-driven":
            continue
        if process == "DY" and args.dy_method == "data-driven":
            continue
        if process not in processes:
            processes.append(process)
    if args.qcd_method == "mc" and "QCD" not in processes:
        processes.append("QCD")
    if args.dy_method == "mc" and "DY" not in processes:
        processes.append("DY")
    return tuple(processes)


def theory_nuisance_names(args: argparse.Namespace) -> List[str]:
    if not args.enable_generator_theory:
        return []
    names = ["PDF_error", "PDF_alphas"]
    for process in args.generator_theory_processes:
        names.extend(
            scale_nuisance_name(process, direction)
            for direction in SCALE_NUISANCE_DIRECTIONS
        )
    return names


def relative_difference(value: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if value == 0.0 else math.inf
    return abs(value / reference - 1.0)


def add_nuisance_quality_warnings(
    nuisances: Mapping[str, Mapping[str, str]],
    year: str,
    warnings: List[str],
    warning_factor: float,
) -> List[str]:
    """Flag duplicated, non-bracketing, or pathological lnN responses.

    The supplied templates are never silently symmetrised or clipped.  The
    returned issue list can optionally be promoted to a hard failure.
    """
    issues: List[str] = []
    signatures: Dict[Tuple[str, ...], str] = {}
    for name in (*EXP_SYST, *L1_PREFIRE_SYST, *BTAG_SYST):
        mapping = nuisances.get(name, {})
        signature = tuple(mapping.get(process, "-") for process in PROCESSES)
        if not any(value != "-" for value in signature):
            continue
        previous = signatures.get(signature)
        if previous is not None:
            issues.append(
                f"Experimental nuisances {previous} and {name} have identical "
                f"responses in {year}; inspect the upstream systematic histograms."
            )
        else:
            signatures[signature] = name

    for name, mapping in nuisances.items():
        for process, value in mapping.items():
            if "/" not in value or value == "-":
                continue
            try:
                down_text, up_text = value.split("/", 1)
                down = float(down_text)
                up = float(up_text)
            except ValueError:
                continue
            if (down - 1.0) * (up - 1.0) > 0.0:
                issues.append(
                    f"{name}/{process} has both Down and Up ratios on the same side "
                    f"of unity ({value}) in {year}; no automatic symmetrisation was applied."
                )
            if min(down, up) < 1.0 / warning_factor or max(down, up) > warning_factor:
                issues.append(
                    f"{name}/{process} has a pathological lnN factor ({value}) in {year}; "
                    "an additive or template-based treatment may be required for a nearly "
                    "zero nominal rate."
                )

    warnings.extend(issues)
    return issues


# -------------------------------------------------------------------------------------------------
# Build one mass point from ROOT inputs
# -------------------------------------------------------------------------------------------------

def build_channels_for_mass(
    reader: RootReader,
    args: argparse.Namespace,
    mass: float,
    files_by_year: Mapping[str, str],
    years: Sequence[str],
) -> Tuple[Dict[str, ChannelResult], AuditLog]:
    label = mass_label(mass)
    audit = AuditLog()
    channels: Dict[str, ChannelResult] = {}

    for year in years:
        warnings: List[str] = []
        low, high = mass_window(args, year, mass)
        signal_file = files_by_year.get(year, "")
        raw_nominal: Dict[str, YieldResult] = {}

        data_file = os.path.join(root_dir(args, year), "data.root")
        data_result = read_required(
            reader, audit, "nominal/data", year,
            data_file, hist_path(args.region), low, high
        )

        for process in PROCESSES:
            if process == "sig" and not signal_file:
                audit.add("nominal/sig", year, False)
                result = None
            else:
                filename = file_for_process(args, year, process, signal_file, "nominal")
                result = read_required(
                    reader, audit, f"nominal/{process}", year,
                    filename, hist_path(args.region), low, high
                )
            raw_nominal[process] = result or YieldResult(0.0, 0.0, 1, 0)

        raw_rates = {
            process: max(raw_nominal[process].value, args.rate_floor)
            for process in PROCESSES
        }
        bkg_sum = sum(raw_rates[p] for p in PROCESSES if p != "sig")
        if args.mode == "blind":
            observation = int(round(bkg_sum))
        elif data_result is None:
            observation = int(round(bkg_sum))
            warnings.append("Data is missing; background-only Asimov observation was used.")
        else:
            observation = int(round(data_result.value))
            if abs(data_result.value - observation) > 1.0e-6:
                warnings.append(
                    f"Weighted data integral {data_result.value:.6g} was rounded to {observation}."
                )

        nuis: Dict[str, Dict[str, str]] = {}

        # Luminosity affects simulation-normalised processes only.
        nuis["lumi"] = {p: "-" for p in PROCESSES}
        for process in ("sig", "tt", "ST", "Others"):
            nuis["lumi"][process] = lnn_from_single(LUMI_LNN[year], args.ignore_rel_below)
        if args.dy_method == "mc":
            nuis["lumi"]["DY"] = lnn_from_single(LUMI_LNN[year], args.ignore_rel_below)
        if args.qcd_method == "mc":
            nuis["lumi"]["QCD"] = lnn_from_single(LUMI_LNN[year], args.ignore_rel_below)

        # Inclusive ttbar normalisation dependence on the assumed top-quark mass.
        # Scale/PDF/alpha_s components of the reference cross section are not
        # included here because they are already represented by generator weights.
        tt_mass_down, tt_mass_up = tt_mass_lnn(year)
        nuis["tt_mass"] = {p: "-" for p in PROCESSES}
        nuis["tt_mass"]["tt"] = (
            f"{format_kappa(tt_mass_down)}/{format_kappa(tt_mass_up)}"
        )

        # Experimental variations.  MC QCD/DY are added automatically when used.
        for syst_name, (down_suffix, up_suffix) in EXP_SYST.items():
            nuis[syst_name] = {p: "-" for p in PROCESSES}
            for process in effective_experimental_processes(args):
                if process == "sig" and not signal_file:
                    continue
                filename = file_for_process(args, year, process, signal_file, "experimental")
                down = read_required(
                    reader, audit, f"{syst_name}/{process}", f"{year}:Down",
                    filename, hist_path(syst_region(args, down_suffix)), low, high
                )
                up = read_required(
                    reader, audit, f"{syst_name}/{process}", f"{year}:Up",
                    filename, hist_path(syst_region(args, up_suffix)), low, high
                )
                nuis[syst_name][process] = lnn_from_down_up(
                    raw_nominal[process].value,
                    down.value if down else None,
                    up.value if up else None,
                    args.ratio_floor,
                    args.ignore_rel_below,
                )

        # L1 ECAL prefiring uncertainty.  Only the affected Run-2 periods
        # have non-trivial NanoAOD prefiring weights.  The nuisance is kept
        # separate for each data-taking era.
        for syst_name, (down_suffix, up_suffix) in L1_PREFIRE_SYST.items():
            nuis[syst_name] = {p: "-" for p in PROCESSES}
            if year in L1_PREFIRE_YEARS:
                for process in effective_experimental_processes(args):
                    if process == "sig" and not signal_file:
                        continue
                    filename = file_for_process(args, year, process, signal_file, "experimental")
                    down = read_required(
                        reader, audit, f"{syst_name}/{process}", f"{year}:Down",
                        filename, hist_path(syst_region(args, down_suffix)), low, high
                    )
                    up = read_required(
                        reader, audit, f"{syst_name}/{process}", f"{year}:Up",
                        filename, hist_path(syst_region(args, up_suffix)), low, high
                    )
                    nuis[syst_name][process] = lnn_from_down_up(
                        raw_nominal[process].value,
                        down.value if down else None,
                        up.value if up else None,
                        args.ratio_floor,
                        args.ignore_rel_below,
                    )

        # BTV fixed-WP b-tagging variations.  HF means b/c jets and LF means
        # light-flavour jets.  The analyzer stores corr/uncorr separately so the
        # fit can implement the BTV-recommended multi-era correlations.
        for syst_name, (down_suffix, up_suffix) in BTAG_SYST.items():
            nuis[syst_name] = {p: "-" for p in PROCESSES}
            for process in effective_experimental_processes(args):
                if process == "sig" and not signal_file:
                    continue
                filename = file_for_process(args, year, process, signal_file, "experimental")
                down = read_required(
                    reader, audit, f"{syst_name}/{process}", f"{year}:Down",
                    filename, hist_path(syst_region(args, down_suffix)), low, high
                )
                up = read_required(
                    reader, audit, f"{syst_name}/{process}", f"{year}:Up",
                    filename, hist_path(syst_region(args, up_suffix)), low, high
                )
                nuis[syst_name][process] = lnn_from_down_up(
                    raw_nominal[process].value,
                    down.value if down else None,
                    up.value if up else None,
                    args.ratio_floor,
                    args.ignore_rel_below,
                )

        # Data-driven QCD: fitted central template plus explicit norm/shape terms.
        # No finite-template or fitted-function QCD_stat nuisance is constructed.
        if args.qcd_method == "data-driven":
            qcd_file = file_for_process(args, year, "QCD", signal_file, "qcd")
            for syst_name, (down_suffix, up_suffix) in QCD_SYST.items():
                nuis[syst_name] = {p: "-" for p in PROCESSES}
                down = read_required(
                    reader, audit, f"{syst_name}/QCD", f"{year}:Down",
                    qcd_file, hist_path(syst_region(args, down_suffix)), low, high
                )
                up = read_required(
                    reader, audit, f"{syst_name}/QCD", f"{year}:Up",
                    qcd_file, hist_path(syst_region(args, up_suffix)), low, high
                )
                nuis[syst_name]["QCD"] = lnn_from_down_up(
                    raw_nominal["QCD"].value,
                    down.value if down else None,
                    up.value if up else None,
                    args.ratio_floor,
                    args.ignore_rel_below,
                )

        # Data-driven DY: fixed constant-NF treatment.  The NF numerator and
        # light-jet source/denominator statistical terms are stored separately.
        if args.dy_method == "data-driven":
            dy_file = file_for_process(args, year, "DY", signal_file, "dy")

            nf_name, nf_down_suffix, nf_up_suffix = DY_NF_SYST
            nf_down = read_required(
                reader, audit, f"{nf_name}/DY", f"{year}:Down",
                dy_file, hist_path(syst_region(args, nf_down_suffix)), low, high
            )
            nf_up = read_required(
                reader, audit, f"{nf_name}/DY", f"{year}:Up",
                dy_file, hist_path(syst_region(args, nf_up_suffix)), low, high
            )
            nuis[nf_name] = {p: "-" for p in PROCESSES}
            nuis[nf_name]["DY"] = lnn_from_down_up(
                raw_nominal["DY"].value,
                nf_down.value if nf_down else None,
                nf_up.value if nf_up else None,
                args.ratio_floor,
                args.ignore_rel_below,
            )

            light_name, light_down_suffix, light_up_suffix = DY_LIGHTJET_STAT_SYST
            light_down = read_required(
                reader, audit, f"{light_name}/DY", f"{year}:Down",
                dy_file, hist_path(syst_region(args, light_down_suffix)), low, high
            )
            light_up = read_required(
                reader, audit, f"{light_name}/DY", f"{year}:Up",
                dy_file, hist_path(syst_region(args, light_up_suffix)), low, high
            )
            nuis[light_name] = {p: "-" for p in PROCESSES}
            nuis[light_name]["DY"] = lnn_from_down_up(
                raw_nominal["DY"].value,
                light_down.value if light_down else None,
                light_up.value if light_up else None,
                args.ratio_floor,
                args.ignore_rel_below,
            )

        # Finite-template statistics from stored TH1 Sumw2 errors.  Data-driven
        # QCD and DY are excluded by construction to avoid duplicate terms.
        for process in finite_template_stat_processes(args):
            name = f"{process}_stat"
            nuis[name] = {p: "-" for p in PROCESSES}
            nuis[name][process] = lnn_from_symmetric_error(
                raw_nominal[process].value,
                raw_nominal[process].error,
                args.ratio_floor,
                args.ignore_rel_below,
            )

        # Generator PDF/alpha_s/scale terms.  tt_xsec and ST_xsec are not added.
        if args.enable_generator_theory:
            nuis["PDF_error"] = {p: "-" for p in PROCESSES}
            nuis["PDF_alphas"] = {p: "-" for p in PROCESSES}
            for process in args.generator_theory_processes:
                for direction in SCALE_NUISANCE_DIRECTIONS:
                    nuis[scale_nuisance_name(process, direction)] = {
                        p: "-" for p in PROCESSES
                    }

            for process in args.generator_theory_processes:
                if process == "sig" and not signal_file:
                    continue
                theory_file = file_for_process(args, year, process, signal_file, "theory")
                nominal_value = raw_nominal[process].value

                # Central scale-member closure check.  PDFScale0 must reproduce
                # the nominal selected yield within tolerance.
                scale_central_suffix = f"PDFScale{SCALE_CENTRAL_INDEX}"
                scale_central = read_required(
                    reader, audit, f"PDF_scale_central/{process}",
                    f"{year}/{scale_central_suffix}", theory_file,
                    hist_path(syst_region(args, scale_central_suffix)), low, high
                )
                scale_closure_ok = (
                    scale_central is not None
                    and relative_difference(scale_central.value, nominal_value)
                    <= THEORY_CENTRAL_TOLERANCE
                )
                audit.add(
                    f"PDF_scale_central_closure/{process}", year, scale_closure_ok
                )
                if scale_central is not None and not scale_closure_ok:
                    warnings.append(
                        f"PDFScale0 differs from the nominal {process} yield by "
                        f"{relative_difference(scale_central.value, nominal_value):.3g} "
                        f"in {year}."
                    )

                # SKFlatMaker's PDF vector normally contains the 100 error
                # members only; there is no PDFError central member to remove.

                # alpha_s: preserve the supplied Down/Up member ordering directly.
                alpha_results: List[Optional[YieldResult]] = []
                for idx in ALPHAS_VARIATION_INDICES:
                    suffix = f"PDFAlphaS{idx}"
                    result = read_required(
                        reader, audit, f"PDF_alphas/{process}", f"{year}/{suffix}",
                        theory_file, hist_path(syst_region(args, suffix)), low, high
                    )
                    alpha_results.append(result)
                nuis["PDF_alphas"][process] = lnn_from_down_up(
                    nominal_value,
                    alpha_results[0].value if alpha_results[0] else None,
                    alpha_results[1].value if alpha_results[1] else None,
                    args.ratio_floor,
                    args.ignore_rel_below,
                )

                # Scale: use opposite members of the same physical scale direction.
                # No min/max envelope and no independent one-sided member nuisances
                # are constructed.  This avoids allowing all six members to float
                # simultaneously as unrelated parameters.
                scale_results: Dict[str, Tuple[Optional[YieldResult], Optional[YieldResult]]] = {}
                directions_to_read = [
                    *SCALE_NUISANCE_DIRECTIONS,
                    SCALE_DIAGONAL_DIRECTION,
                ]
                for direction in directions_to_read:
                    down_index, up_index = scale_pair(direction)
                    pair_results: List[Optional[YieldResult]] = []
                    for side, idx in (("Down", down_index), ("Up", up_index)):
                        suffix = f"PDFScale{idx}"
                        result = read_required(
                            reader, audit,
                            f"PDF_scale_{direction}/{process}",
                            f"{year}:{side}/member{idx}",
                            theory_file,
                            hist_path(syst_region(args, suffix)), low, high,
                        )
                        pair_results.append(result)
                    scale_results[direction] = (pair_results[0], pair_results[1])
                    if direction in SCALE_NUISANCE_DIRECTIONS:
                        nuisance_name = scale_nuisance_name(process, direction)
                        nuis[nuisance_name][process] = lnn_from_down_up(
                            nominal_value,
                            pair_results[0].value if pair_results[0] else None,
                            pair_results[1].value if pair_results[1] else None,
                            args.ratio_floor,
                            args.ignore_rel_below,
                        )

                # The diagonal pair probes the simultaneous muR/muF change.  When
                # it is validation-only, compare it with the product of the two
                # axis responses and report a large non-factorising residual.
                mu_f = scale_results.get("muF")
                mu_r = scale_results.get("muR")
                diagonal = scale_results.get(SCALE_DIAGONAL_DIRECTION)
                if mu_f and mu_r and diagonal and nominal_value > 0.0:
                    for side_index, side_name in ((0, "Down"), (1, "Up")):
                        f_result = mu_f[side_index]
                        r_result = mu_r[side_index]
                        d_result = diagonal[side_index]
                        if f_result and r_result and d_result:
                            predicted = (
                                f_result.value / nominal_value
                                * r_result.value / nominal_value
                            )
                            observed = d_result.value / nominal_value
                            log_residual = abs(
                                math.log(max(observed, args.ratio_floor))
                                - math.log(max(predicted, args.ratio_floor))
                            )
                            factorisation_ok = (
                                log_residual <= SCALE_DIAGONAL_LOG_TOLERANCE
                            )
                            audit.add(
                                f"PDF_scale_factorisation/{process}",
                                f"{year}:{side_name}",
                                factorisation_ok,
                            )
                            if not factorisation_ok:
                                warnings.append(
                                    f"The {process} diagonal scale variation ({side_name}) "
                                    f"does not factorise into the muF and muR responses in "
                                    f"{year}: observed ratio={observed:.6g}, product="
                                    f"{predicted:.6g}.  The diagonal pair remains a "
                                    "validation-only input and is not added as a third nuisance."
                                )

                # NNPDF31_nnlo_as_0118_mc_hessian_pdfas has one central
                # member and 100 symmetric-Hessian eigenvector members.  The
                # nominal histogram supplies the central yield; PDFError0..99
                # correspond to LHAPDF error members 1..100.
                pdf_values: List[float] = []
                for idx in PDF_EIGENVECTOR_INDICES:
                    suffix = f"PDFError{idx}"
                    result = read_required(
                        reader, audit, f"PDF_error/{process}", f"{year}/{suffix}",
                        theory_file, hist_path(syst_region(args, suffix)), low, high
                    )
                    if result:
                        pdf_values.append(result.value)
                nuis["PDF_error"][process] = lnn_from_symmetric_hessian(
                    nominal_value, pdf_values,
                    args.ratio_floor, args.ignore_rel_below
                )

        quality_issues = add_nuisance_quality_warnings(
            nuis, year, warnings, args.lnn_warning_factor
        )
        if args.fail_on_suspicious_systematics and quality_issues:
            raise WorkflowError(
                f"Suspicious systematic templates found in {year}, M-{label}:\n  - "
                + "\n  - ".join(quality_issues)
            )
        channels[year] = ChannelResult(
            year=year,
            mass_label=label,
            mass=mass,
            window_low=low,
            window_high=high,
            observation=observation,
            raw_rates=raw_rates,
            rates=dict(raw_rates),
            nuisances=nuis,
            warnings=warnings,
        )

    return channels, audit


# -------------------------------------------------------------------------------------------------
# Parameterisation and datacard writing
# -------------------------------------------------------------------------------------------------

def load_xsec_map(args: argparse.Namespace) -> Dict[float, float]:
    mapping: Dict[float, float] = {}
    if args.signal_xsec_map:
        path = resolve_path(args.signal_xsec_map, "Signal cross-section map")
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise WorkflowError("Signal xsec JSON must be a mass-to-xsec object.")
            for mass, xsec in payload.items():
                mapping[float(str(mass).replace("p", "."))] = float(xsec)
        else:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames:
                    raise WorkflowError("Signal xsec CSV is empty.")
                mass_key = next((x for x in reader.fieldnames if normalise_key(x) in {"mass", "massgev", "mzp"}), None)
                xsec_key = next((x for x in reader.fieldnames if normalise_key(x) in {"xsec", "xsecpb", "crosssectionpb", "sigma"}), None)
                if mass_key is None or xsec_key is None:
                    raise WorkflowError("Signal xsec CSV needs mass and xsec_pb columns.")
                for row in reader:
                    mapping[float(row[mass_key].replace("p", "."))] = float(row[xsec_key])
    return mapping


def lookup_xsec(args: argparse.Namespace, xsec_map: Mapping[float, float], mass: float) -> float:
    if args.signal_reference_xsec_pb is not None:
        xsec = args.signal_reference_xsec_pb
    else:
        matches = [(abs(key - mass), value) for key, value in xsec_map.items()]
        if not matches or min(matches)[0] > 1.0e-6:
            raise WorkflowError(
                f"No alpha=1 reference cross section was supplied for M-{mass_label(mass)}. "
                "Use --signal-xsec-map or --signal-reference-xsec-pb."
            )
        xsec = min(matches)[1]
    if xsec <= 0.0 or not math.isfinite(xsec):
        raise WorkflowError(f"Invalid reference cross section for mass {mass}: {xsec}")
    return xsec


def apply_parameterisation(
    args: argparse.Namespace,
    target: str,
    channels: Sequence[ChannelResult],
    xsec_map: Mapping[float, float],
) -> Tuple[float, Optional[float], Optional[float]]:
    raw_total = sum(ch.raw_rates["sig"] for ch in channels)
    if raw_total <= 0.0 or not math.isfinite(raw_total):
        raise WorkflowError(f"Non-positive signal yield for {target}, M-{channels[0].mass_label}.")

    alpha_unit: Optional[float] = None
    reference_xsec: Optional[float] = None
    if args.parameter == "alpha":
        alpha_unit = args.alpha_internal_unit or (args.alpha_card_yield / raw_total)
        scale = alpha_unit
    elif args.parameter == "yield":
        scale = 1.0 / raw_total
    else:
        reference_xsec = lookup_xsec(args, xsec_map, channels[0].mass)
        scale = 1.0 / reference_xsec

    for channel in channels:
        channel.rates = dict(channel.raw_rates)
        channel.rates["sig"] = channel.raw_rates["sig"] * scale
    return scale, alpha_unit, reference_xsec


def nuisance_global_name(local: str, year: str) -> str:
    """Return the CMS-style datacard nuisance name without changing correlations.

    Internal keys intentionally remain unchanged because they are also used to
    locate the existing ROOT templates.  Only the final datacard parameter name
    is translated here.  Common CMS sources use names from the current CMS
    systematics master list.  Analysis-specific sources use the CADI prefix
    CMS_NPS26009_.

    The current luminosity treatment is an aggregate uncertainty correlated in
    five year-groups (2016, 2017, 2018, 2022, 2023).  The official master list
    uses different Run-2 and Run-3 spelling conventions.  To keep one uniform
    year-suffix format without changing the existing correlation model, these
    aggregate luminosity nuisances are explicitly analysis-specific.
    """

    # Aggregate luminosity model: preserve the existing five correlation groups
    # while using one uniform naming format for Run 2 and Run 3.
    if local == "lumi":
        return f"CMS_NPS26009_lumi_{lumi_group(year)}"

    # One common top-mass dependence nuisance, with energy-dependent lnN values.
    if local == "tt_mass":
        return "CMS_NPS26009_topmass_ttbar_BJetOS"

    # BTV fixed-WP.  The official uncorrelated names can be used verbatim.
    # The correlated pieces are intentionally split between Run 2 and Run 3 in
    # this analysis, so they need analysis-specific names to avoid accidentally
    # correlating the two calibration campaigns in a full combination.
    if local == "btagSFbc_correlated":
        energy = "13TeV" if year in RUN2_ERAS else "13p6TeV"
        return f"CMS_NPS26009_btag_fixedWP_bc_correlated_{energy}"
    if local == "btagSFlight_correlated":
        energy = "13TeV" if year in RUN2_ERAS else "13p6TeV"
        return f"CMS_NPS26009_btag_fixedWP_light_correlated_{energy}"
    if local == "btagSFbc_uncorrelated":
        return f"CMS_btag_fixedWP_bc_uncorrelated_{year}"
    if local == "btagSFlight_uncorrelated":
        return f"CMS_btag_fixedWP_light_uncorrelated_{year}"

    # Common experimental sources.  The master list has no pre/post-VFP split
    # for the muon sources below.  Keep the existing 2016preVFP/2016postVFP
    # decorrelation by using analysis-specific names only for those two eras.
    common_exp = {
        "jer": "CMS_res_j",
        "jes": "CMS_scale_j",
        "pu": "CMS_pileup",
    }
    if local in common_exp:
        return f"{common_exp[local]}_{year}"

    muon_exp = {
        "mu_trig_sf": ("CMS_eff_m_trigger", "eff_m_trigger"),
        "mu_id_sf": ("CMS_eff_m_id", "eff_m_id"),
        "mu_scale": ("CMS_scale_m", "scale_m"),
    }
    if local in muon_exp:
        common_name, custom_stem = muon_exp[local]
        if year in {"2016preVFP", "2016postVFP"}:
            return f"CMS_NPS26009_{custom_stem}_{year}"
        return f"{common_name}_{year}"

    # The current L1 ECAL prefiring nuisances are decorrelated between
    # 2016preVFP, 2016postVFP and 2017.  The master common source has only a
    # combined 2016 entry, therefore the analysis-specific names preserve the
    # existing correlation model exactly.
    if local == "l1prefire":
        return f"CMS_l1_prefiring_{year}"

    # Analysis-specific data-driven background uncertainties.  Include the
    # affected process/category before the era, leaving the era as the suffix.
    if local == "QCD_norm":
        return f"CMS_NPS26009_bckgNorm_QCD_BJetOS_{year}"
    if local == "QCD_shape":
        return f"CMS_NPS26009_bckgShape_QCD_BJetOS_{year}"
    if local == "DY_NFStat":
        return f"CMS_NPS26009_NFStat_DY_BJetOS_{year}"
    if local == "DY_LightJetStat":
        return f"CMS_NPS26009_LightJetStat_DY_BJetOS_{year}"

    # Finite-template statistical terms remain decorrelated by process and era.
    if local.endswith("_stat"):
        process = local[:-5]
        process_name = {
            "sig": "signal",
            "tt": "ttbar",
            "ST": "singleTop",
            "DY": "DY",
            "QCD": "QCD",
            "Others": "other",
        }.get(process, process)
        return f"CMS_NPS26009_MCstat_{process_name}_BJetOS_{year}"

    # The total NNPDF31 symmetric-Hessian term is one analysis-defined nuisance
    # shared by the generator-theory processes.  alpha_s has a master name.
    if local == "PDF_error":
        return "CMS_NPS26009_pdf_NNPDF31"
    if local == "PDF_alphas":
        return "pdf_alphas"

    # Translate the internal PDF_scale_<process>_<direction> keys to the current
    # CMS QCD-scale convention.  ttbar has a common master name; single-top and
    # any optional non-master process use analysis-specific names.
    match = re.fullmatch(r"PDF_scale_(.+)_(muF|muR)", local)
    if match:
        process, direction = match.groups()
        scale_kind = "fac" if direction == "muF" else "ren"
        if process == "tt":
            return f"QCDscale_{scale_kind}_ttbar"
        process_name = {
            "sig": "signal",
            "ST": "singleTop",
            "DY": "DY",
            "QCD": "QCD",
            "Others": "other",
        }.get(process, process)
        return f"CMS_NPS26009_QCDscale_{scale_kind}_{process_name}"

    # Preserve any future/unknown internal source rather than silently changing
    # its correlation semantics.  New sources should be added explicitly above.
    return local


def nuisance_order(
    args: argparse.Namespace,
    channels: Optional[Sequence[ChannelResult]] = None,
) -> List[str]:
    order: List[str] = [
        "lumi",
        "tt_mass",
        *EXP_SYST.keys(),
        *L1_PREFIRE_SYST.keys(),
        *BTAG_SYST.keys(),
        *([] if args.qcd_method == "mc" else QCD_SYST.keys()),
    ]
    if args.dy_method == "data-driven":
        order.extend(DATA_DRIVEN_DY_NUISANCES)
    order.extend(f"{process}_stat" for process in finite_template_stat_processes(args))
    order.extend(theory_nuisance_names(args))

    # Auto-selected DY factor names and any future dynamically generated terms
    # are appended in their channel insertion order.
    if channels is not None:
        for channel in channels:
            for name in channel.nuisances:
                if name not in order:
                    order.append(name)
    return list(dict.fromkeys(order))


def datacard_path(args: argparse.Namespace, target: str, label: str) -> Path:
    root = Path(args.card_base) / args.parameter / args.mode
    if target in COMBINED_TARGETS:
        return root / "combined" / target / f"datacard_M-{label}_{target}.txt"
    return root / "per_year" / target / f"datacard_M-{label}_{target}.txt"


def write_datacard(
    args: argparse.Namespace,
    target: str,
    channels: Sequence[ChannelResult],
    signal_scale: float,
    alpha_unit: Optional[float],
    reference_xsec: Optional[float],
) -> CardInfo:
    label = channels[0].mass_label
    path = datacard_path(args, target, label)
    path.parent.mkdir(parents=True, exist_ok=True)

    bins = [f"bin_{ch.year}" for ch in channels]
    columns: List[Tuple[int, str]] = [
        (ich, process) for ich in range(len(channels)) for process in PROCESSES
    ]

    lines: List[str] = [
        f"imax {len(channels)} number of channels",
        "jmax * number of backgrounds",
        "kmax * number of nuisance parameters",
        "-" * 130,
        pad_row(["bin", *bins]),
        pad_row(["observation", *[ch.observation for ch in channels]]),
        "-" * 130,
        pad_row(["bin", *[bins[ich] for ich, _ in columns]]),
        pad_row(["process", *[process for _, process in columns]]),
        pad_row(["process", *[PROC_ID[process] for _, process in columns]]),
        pad_row(["rate", *[format_number(channels[ich].rates[process]) for ich, process in columns]]),
        "-" * 130,
        f"# mass M-{label}, target {target}",
        f"# limit_parameter = {args.parameter}",
    ]

    raw_total = sum(ch.raw_rates["sig"] for ch in channels)
    card_total = sum(ch.rates["sig"] for ch in channels)
    if args.parameter == "alpha":
        assert alpha_unit is not None
        lines.extend([
            "# Combine internal r = alpha_qZp / alpha_internal_unit.",
            "# alpha_qZp = r * alpha_internal_unit.",
            f"# alpha_internal_unit = {alpha_unit:.17g}",
            f"# alpha1_reference_signal_yield = {raw_total:.17g}",
            f"# alpha_internal_signal_yield = {card_total:.17g}",
            f"# signal_rate_scale = {signal_scale:.17g}",
        ])
    elif args.parameter == "yield":
        lines.extend([
            "# Combine r = total selected signal event yield.",
            f"# alpha1_reference_signal_yield = {raw_total:.17g}",
            f"# signal_rate_scale = {signal_scale:.17g}",
        ])
    else:
        assert reference_xsec is not None
        lines.extend([
            "# Combine r = signal cross section in pb.",
            f"# alpha1_reference_signal_yield = {raw_total:.17g}",
            f"# alpha1_reference_xsec_pb = {reference_xsec:.17g}",
            f"# signal_rate_scale = {signal_scale:.17g}",
        ])

    lines.extend([
        "# uncertainty_policy = explicit data-driven terms; no generic tt_xsec/ST_xsec",
        "# CMS_NPS26009_topmass_ttbar_BJetOS = asymmetric ttbar normalisation from top-mass dependence of the NNLO+NNLL reference cross section",
        "# b tagging = BTV fixed-WP HF/LF x correlated/uncorrelated multi-era scheme (correlated within Run 2 or Run 3)",
        "# data-driven QCD: analysis-specific normalization + shape nuisances; no fitted-template QCD_stat",
        "# data-driven DY: constant NF + analysis-specific NF/source-stat nuisances; no DY_stat",
        f"# PDF set = {PDF_SET_NAME}; PDFError0..99 use symmetric-Hessian quadrature",
        "# generator scale: separate paired muF/muR nuisances; no 7-point envelope",
        "# simultaneous muR/muF pair is validation-only; antipodal pairs are excluded",
    ])
    for channel in channels:
        lines.append(
            f"# {channel.year} counting window "
            f"[{channel.window_low:.8g}, {channel.window_high:.8g}] GeV"
        )

    local_nuisance_order = nuisance_order(args, channels)
    values_by_name: Dict[str, List[str]] = {}
    global_order: List[str] = []
    for local in local_nuisance_order:
        for channel in channels:
            global_name = nuisance_global_name(local, channel.year)
            if global_name not in values_by_name:
                values_by_name[global_name] = ["-"] * len(columns)
                global_order.append(global_name)

    for col_idx, (ich, process) in enumerate(columns):
        channel = channels[ich]
        for local in local_nuisance_order:
            global_name = nuisance_global_name(local, channel.year)
            values_by_name[global_name][col_idx] = channel.nuisances.get(local, {}).get(process, "-")

    for name in global_order:
        values = values_by_name[name]
        if any(value != "-" for value in values):
            lines.append(pad_row([name, "lnN", *values]))

    warnings = [f"{ch.year}: {warning}" for ch in channels for warning in ch.warnings]
    if warnings:
        lines.append("-" * 130)
        lines.append("# Warnings:")
        lines.extend(f"# - {warning}" for warning in warnings[:100])

    path.write_text("\n".join(lines) + "\n")
    return CardInfo(
        path=path.resolve(), target=target, mass_label=label, mass=channels[0].mass,
        parameter=args.parameter, reference_signal_yield=raw_total,
        signal_rate_scale=signal_scale, alpha_internal_unit=alpha_unit,
        internal_signal_yield=card_total, reference_xsec_pb=reference_xsec,
    )


def write_card_summaries(args: argparse.Namespace, cards: Sequence[CardInfo]) -> None:
    root = Path(args.card_base) / args.parameter / args.mode
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "card_summary.csv"
    json_path = root / "card_summary.json"
    fields = [
        "card", "target", "mass", "parameter", "alpha1_reference_signal_yield",
        "alpha1_reference_xsec_pb", "signal_rate_scale", "sum_card_signal_rate",
        "alpha_internal_unit",
    ]
    rows = []
    for card in cards:
        rows.append({
            "card": str(card.path),
            "target": card.target,
            "mass": card.mass,
            "parameter": card.parameter,
            "alpha1_reference_signal_yield": card.reference_signal_yield,
            "alpha1_reference_xsec_pb": card.reference_xsec_pb,
            "signal_rate_scale": card.signal_rate_scale,
            "sum_card_signal_rate": card.internal_signal_yield,
            "alpha_internal_unit": card.alpha_internal_unit,
        })
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")

    if args.parameter == "alpha":
        scaling_fields = [
            "target", "mass", "alpha1_reference_signal_yield",
            "alpha_internal_unit", "internal_signal_yield",
        ]
        scaling_rows = [{
            "target": c.target,
            "mass": c.mass,
            "alpha1_reference_signal_yield": c.reference_signal_yield,
            "alpha_internal_unit": c.alpha_internal_unit,
            "internal_signal_yield": c.internal_signal_yield,
        } for c in cards]
        with (root / "alpha_internal_scaling.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=scaling_fields)
            writer.writeheader()
            writer.writerows(scaling_rows)
        (root / "alpha_internal_scaling.json").write_text(
            json.dumps(scaling_rows, indent=2, sort_keys=True) + "\n"
        )


def clean_card_targets(args: argparse.Namespace) -> None:
    root = Path(args.card_base) / args.parameter / args.mode
    for target in selected_targets(args.target):
        directory = (
            root / "combined" / target
            if target in COMBINED_TARGETS
            else root / "per_year" / target
        )
        if directory.exists():
            shutil.rmtree(directory)


def build_cards(args: argparse.Namespace) -> List[CardInfo]:
    reader = RootReader()
    try:
        clean_card_targets(args)
        requested_years = years_needed_for_request(args)
        signal_files = scan_signal_files(args)
        masses = sorted(signal_files)
        if args.masses:
            masses = [m for m in masses if any(abs(m-r) < 1.0e-6 for r in args.masses)]
        if not masses:
            raise WorkflowError("No requested signal mass files were found.")
        xsec_map = load_xsec_map(args) if args.parameter == "xsec" else {}
        cards: List[CardInfo] = []

        for mass in masses:
            files_by_year = signal_files[mass]
            channels_by_year, audit = build_channels_for_mass(
                reader, args, mass, files_by_year, requested_years
            )
            audit.print_and_check(mass_label(mass), args.strict)

            for target in selected_targets(args.target):
                years = target_years(target)
                missing_signal_years = [year for year in years if not files_by_year.get(year)]
                if missing_signal_years:
                    message = (
                        f"M-{mass_label(mass)} has no signal file for target {target}: "
                        + ", ".join(missing_signal_years)
                    )
                    if args.strict or target in COMBINED_TARGETS or args.require_all_years:
                        raise WorkflowError(message)
                    print(f"[WARNING] {message}; skip card.", file=sys.stderr)
                    continue
                channels = [channels_by_year[year] for year in years]
                scale, alpha_unit, reference_xsec = apply_parameterisation(
                    args, target, channels, xsec_map
                )
                card = write_datacard(
                    args, target, channels, scale, alpha_unit, reference_xsec
                )
                cards.append(card)
                print(f"[CARD] {card.path}")

        write_card_summaries(args, cards)
        return cards
    finally:
        reader.close()


# -------------------------------------------------------------------------------------------------
# Datacard discovery and metadata
# -------------------------------------------------------------------------------------------------

def card_pattern(args: argparse.Namespace, target: str) -> Path:
    root = Path(args.card_base) / args.parameter / args.mode
    if target in COMBINED_TARGETS:
        return root / "combined" / target / f"datacard_M-*_{target}.txt"
    return root / "per_year" / target / f"datacard_M-*_{target}.txt"


def cards_for_target(args: argparse.Namespace, target: str) -> List[Path]:
    pattern = card_pattern(args, target)
    # Combine runs in the output directory.  Resolve here so datacard paths
    # remain valid after changing the subprocess working directory.
    cards = sorted(path.resolve() for path in pattern.parent.glob(pattern.name))
    if args.masses:
        selected: List[Path] = []
        for card in cards:
            _label, mass = extract_mass_from_card(card, target)
            if any(abs(mass-r) < 1.0e-6 for r in args.masses):
                selected.append(card)
        cards = selected
    return cards


def extract_mass_from_card(card: Path, target: str) -> Tuple[str, float]:
    suffix = target
    match = re.fullmatch(rf"datacard_M-(.+)_{re.escape(suffix)}\.txt", card.name)
    if not match:
        raise WorkflowError(f"Cannot extract mass from {card}")
    label = match.group(1)
    return label, float(label.replace("p", "."))


def parse_comment_float(card: Path, key: str) -> Optional[float]:
    pattern = re.compile(rf"^\s*#\s*{re.escape(key)}\s*=\s*([^\s#]+)")
    for line in card.read_text().splitlines():
        match = pattern.match(line)
        if match:
            return float(match.group(1))
    return None


def alpha_unit_from_card(card: Path) -> float:
    unit = parse_comment_float(card, "alpha_internal_unit")
    if unit is None or unit <= 0.0:
        raise WorkflowError(f"Missing or invalid alpha_internal_unit in {card}")
    return unit


def internal_signal_yield_from_card(card: Path) -> float:
    value = parse_comment_float(card, "alpha_internal_signal_yield")
    if value is not None and value > 0.0:
        return value
    processes: Optional[List[str]] = None
    rates: Optional[List[float]] = None
    for line in card.read_text().splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "process" and any(not re.fullmatch(r"[-+]?\d+", x) for x in tokens[1:]):
            processes = tokens[1:]
        elif tokens[0] == "rate":
            rates = [float(x) for x in tokens[1:]]
    if processes is None or rates is None:
        raise WorkflowError(f"Cannot parse signal rate from {card}")
    return sum(rate for process, rate in zip(processes, rates) if process == "sig")


# -------------------------------------------------------------------------------------------------
# Combine execution
# -------------------------------------------------------------------------------------------------

def effective_r_min(args: argparse.Namespace) -> float:
    if args.r_min is not None:
        return args.r_min
    return 0.0


def effective_r_max(args: argparse.Namespace, card: Path) -> float:
    if args.r_max is not None:
        return args.r_max
    if args.parameter == "alpha":
        internal_yield = internal_signal_yield_from_card(card)
        return args.max_internal_signal_yield / internal_yield
    if args.parameter == "yield":
        return args.default_yield_r_max
    return args.default_xsec_r_max


def explicit_range_args(rmin: float, rmax: float, card: Path) -> List[str]:
    if rmax <= rmin:
        raise WorkflowError(f"Invalid POI range [{rmin}, {rmax}] for {card}")
    return ["--rMin", f"{rmin:.12g}", "--rMax", f"{rmax:.12g}"]


def range_args(args: argparse.Namespace, card: Path) -> List[str]:
    rmin = effective_r_min(args)
    rmax = effective_r_max(args, card)
    if rmin < 0.0 and not args.allow_negative_r:
        raise WorkflowError(
            f"Negative rMin={rmin} requires --allow-negative-r. "
            "This is a diagnostic option only."
        )
    return explicit_range_args(rmin, rmax, card)


def counting_card_observation_signal_background(
    card: Path,
) -> Tuple[List[float], List[float], List[float]]:
    """Read observation, signal, and total background yields by channel.

    Datacards produced by this workflow list all processes channel by channel.
    This parser is intentionally independent of the fixed process count and uses
    the process-name row to identify signal columns.
    """
    observations: Optional[List[float]] = None
    process_names: Optional[List[str]] = None
    rates: Optional[List[float]] = None

    for line in card.read_text().splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "observation":
            observations = [float(x) for x in tokens[1:]]
        elif tokens[0] == "process" and any(
            not re.fullmatch(r"[-+]?\d+", x) for x in tokens[1:]
        ):
            process_names = tokens[1:]
        elif tokens[0] == "rate":
            rates = [float(x) for x in tokens[1:]]

    if observations is None or process_names is None or rates is None:
        raise WorkflowError(f"Cannot parse counting rates from {card}")
    if len(process_names) != len(rates):
        raise WorkflowError(f"Process/rate column mismatch in {card}")
    if not observations or len(rates) % len(observations) != 0:
        raise WorkflowError(f"Cannot map process columns to channels in {card}")

    nproc = len(rates) // len(observations)
    signals: List[float] = []
    backgrounds: List[float] = []
    for ich in range(len(observations)):
        start = ich * nproc
        stop = start + nproc
        sig = 0.0
        bkg = 0.0
        for process, rate in zip(process_names[start:stop], rates[start:stop]):
            if process == "sig":
                sig += rate
            else:
                bkg += rate
        signals.append(sig)
        backgrounds.append(bkg)
    return observations, signals, backgrounds


def nominal_negative_r_boundary(card: Path) -> float:
    """Closest-to-zero r for which a nominal channel expectation vanishes."""
    _obs, signals, backgrounds = counting_card_observation_signal_background(card)
    boundaries = [
        -bkg / sig for sig, bkg in zip(signals, backgrounds)
        if sig > 0.0 and bkg >= 0.0
    ]
    return max(boundaries) if boundaries else 0.0


def nominal_unconstrained_rhat(card: Path, lower: float, upper: float) -> float:
    """Nominal Poisson-only estimate of the unconstrained signal strength.

    This is used only to choose a safe diagnostic impact range; Combine still
    performs the full profiled likelihood fit with all nuisance parameters.
    """
    observations, signals, backgrounds = counting_card_observation_signal_background(card)

    def score(rvalue: float) -> float:
        total = 0.0
        for obs, sig, bkg in zip(observations, signals, backgrounds):
            expectation = bkg + rvalue * sig
            if sig <= 0.0:
                continue
            if expectation <= 0.0:
                return math.inf
            total += sig * (obs / expectation - 1.0)
        return total

    slo = score(lower)
    shi = score(upper)
    if not math.isfinite(slo) or slo > 0.0:
        if shi >= 0.0:
            return upper
        lo, hi = lower, upper
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            smid = score(mid)
            if smid > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    return lower


def initial_impact_range(args: argparse.Namespace, card: Path) -> Tuple[float, float, float]:
    """Return (rMin, rMax, nominal-safe-rMin) for an impact fit."""
    requested = effective_r_min(args)
    rmax = effective_r_max(args, card)
    if not args.allow_negative_r or requested >= 0.0:
        return requested, rmax, requested

    hard_boundary = nominal_negative_r_boundary(card)
    safe_boundary = hard_boundary * 0.90 if hard_boundary < 0.0 else requested
    safe_boundary = min(safe_boundary, -1.0e-6)

    if not args.auto_expand_impact_r_range:
        return max(requested, safe_boundary), rmax, safe_boundary

    rhat = nominal_unconstrained_rhat(card, safe_boundary, rmax)
    margin = max(2.0, 0.5 * abs(rhat))
    adaptive = max(safe_boundary, min(requested, rhat - margin))
    return adaptive, rmax, safe_boundary


def impact_fit_triplet(payload: Mapping[str, Any]) -> Optional[Tuple[float, float, float]]:
    for poi in payload.get("POIs", []):
        if not isinstance(poi, Mapping) or poi.get("name") != "r":
            continue
        fit = poi.get("fit")
        if isinstance(fit, list) and len(fit) >= 3 and all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in fit[:3]
        ):
            return float(fit[0]), float(fit[1]), float(fit[2])
    return None


def impact_fit_reaches_lower_boundary(
    payload: Mapping[str, Any], rmin: float
) -> Tuple[bool, Optional[Tuple[float, float, float]]]:
    fit = impact_fit_triplet(payload)
    if fit is None:
        return False, None
    low, central, _high = fit
    tolerance = max(1.0e-7, 1.0e-4 * max(1.0, abs(rmin)))
    return central <= rmin + tolerance or low <= rmin + tolerance, fit


def expanded_impact_rmin(current: float, safe_boundary: float) -> float:
    candidate = min(current - 2.0, current * 2.5)
    return max(safe_boundary, candidate)


def output_dir(args: argparse.Namespace, target: str) -> Path:
    return (Path(args.output_base) / args.parameter / args.mode / target).resolve()


def tag_for(args: argparse.Namespace, target: str) -> str:
    return f"{target}_{args.parameter}"


def clean_mass_outputs(outdir: Path, tag: str, label: str, task: str) -> None:
    """Remove only outputs belonging to the task being rerun."""
    patterns: List[str] = []

    if task in {"limits", "all"}:
        patterns.extend([
            f"higgsCombine.{tag}_M{label}.AsymptoticLimits.mH*.root",
            f"higgsCombine.{tag}_M{label.replace('p', '.')}.AsymptoticLimits.mH*.root",
            f"higgsCombine.{tag}_M{label}_BlindExpected.AsymptoticLimits.mH*.root",
            f"higgsCombine.{tag}_M{label.replace('p', '.')}_BlindExpected.AsymptoticLimits.mH*.root",
        ])

    if task in {"fitdiag", "checks", "all"}:
        patterns.extend([
            f"fitDiagnostics.FitDiag_{tag}_M{label}.root",
            f"higgsCombine.FitDiag_{tag}_M{label}.*.mH*.root",
            f"pulls_{tag}_M{label}.txt",
            f"pulls_{tag}_M{label}.root",
            f"pulls_{tag}_M{label}_with_graph.txt",
        ])

    if task in {"impacts", "all"}:
        patterns.extend([
            f"workspace_{tag}_M{label}.root",
            f"impacts_{tag}_M{label}*",
            f"*Impacts_{tag}_M{label}*",
        ])

    for pattern in patterns:
        for path in outdir.glob(pattern):
            if path.is_file() or path.is_symlink():
                path.unlink()


def run_asymptotic(
    args: argparse.Namespace,
    card: Path,
    target: str,
    label: str,
    mass: float,
    outdir: Path,
) -> LimitOutputs:
    tag = tag_for(args, target)

    def run_one(name_suffix: str, blind_expected: bool) -> Path:
        fit_tag = f"{tag}_M{label}{name_suffix}"
        command = [
            "combine", "-M", "AsymptoticLimits", str(card),
            "-m", format_mass_for_combine(mass),
            "--cminDefaultMinimizerStrategy", "0",
            *range_args(args, card),
            "-n", f".{fit_tag}",
        ]
        if blind_expected:
            command.extend(["--run", "blind"])
        run_command(command, outdir)
        matches = sorted(
            outdir.glob(f"higgsCombine.{fit_tag}.AsymptoticLimits.mH*.root")
        )
        if not matches:
            raise WorkflowError(
                f"AsymptoticLimits output is missing for {target} M-{label} ({fit_tag})"
            )
        return matches[-1]

    if args.mode == "blind":
        expected = run_one("", True)
        return LimitOutputs(observed=expected, expected=expected)

    observed = run_one("", False)
    expected = run_one("_BlindExpected", True)
    return LimitOutputs(observed=observed, expected=expected)


def run_fitdiagnostics(
    args: argparse.Namespace,
    card: Path,
    target: str,
    label: str,
    mass: float,
    outdir: Path,
) -> None:
    tag = tag_for(args, target)
    name = f".FitDiag_{tag}_M{label}"
    command = [
        "combine", "-M", "FitDiagnostics", str(card),
        "-m", format_mass_for_combine(mass),
        "--cminDefaultMinimizerStrategy", "0",
        "--setParameters", "r=0",
        *range_args(args, card),
        "--saveNormalizations", "--saveShapes", "--saveWithUncertainties",
        "-n", name,
    ]
    if args.mode == "blind":
        command.extend(["-t", "-1", "--expectSignal", "0"])
    if run_command(command, outdir, allow_failure=True) != 0:
        nonfatal_or_raise(args, f"FitDiagnostics failed for {target} M-{label}.")
        return

    fitfile = outdir / f"fitDiagnostics{name}.root"
    diff = Path(os.environ.get("CMSSW_BASE", "")) / "src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py"
    if not fitfile.exists() or not diff.exists():
        print(f"[WARNING] Pull extraction unavailable for {target} M-{label}.", file=sys.stderr)
        return
    run_command(
        [sys.executable, str(diff), str(fitfile), "--all", "--abs"],
        outdir, allow_failure=True,
        stdout_path=outdir / f"pulls_{tag}_M{label}.txt",
    )
    run_command(
        [
            sys.executable, str(diff), str(fitfile), "--all", "--abs",
            "-g", f"pulls_{tag}_M{label}.root",
        ],
        outdir, allow_failure=True,
        stdout_path=outdir / f"pulls_{tag}_M{label}_with_graph.txt",
    )


def should_run_impacts(args: argparse.Namespace, mass: float) -> bool:
    if args.impact_masses:
        return any(abs(mass-x) < 1.0e-6 for x in args.impact_masses)
    return args.run_impacts_all_masses


def scale_numeric_sequence(value: Any, factor: float) -> Any:
    if isinstance(value, list):
        return [scale_numeric_sequence(x, factor) for x in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value * factor
    return value


def rescale_impact_payload(payload: MutableMapping[str, Any], factor: float) -> None:
    for poi in payload.get("POIs", []):
        if isinstance(poi, MutableMapping) and poi.get("name") == "r":
            for key in ("fit", "prefit"):
                if key in poi:
                    poi[key] = scale_numeric_sequence(poi[key], factor)
    for nuisance in payload.get("params", []):
        if not isinstance(nuisance, MutableMapping):
            continue
        for key in ("r", "impact_r"):
            if key in nuisance:
                nuisance[key] = scale_numeric_sequence(nuisance[key], factor)


def nps_endpoint_safe_plot_impacts_script(plot_script: Path, outdir: Path) -> Path:
    # Create a local plotImpacts.py copy with plotting-only geometry fixes.
    #
    # ROOT draws the x-axis scientific-notation exponent (x10^N) just beyond
    # the right end of the impact axis.  In the standard impact canvas this can
    # be clipped.  Keep the exponent notation, but move it left and reserve
    # explicit right-side space.  The impact JSON and all fits are unchanged.

    source = plot_script.read_text()

    # Give the nuisance labels and the impact axis more horizontal room.
    old_width = "width=(900 if args.checkboxes else 700)"
    new_width = "width=(1000 if args.checkboxes else 860)"
    if old_width in source:
        source = source.replace(old_width, new_width, 1)

    # The non-checkbox impact panel is pads[1].  A larger right margin moves the
    # actual impact axis away from the canvas boundary.
    margin_line = (
        "    pads[1].SetRightMargin("
        "max(float(pads[1].GetRightMargin()), 0.05))"
    )
    if margin_line not in source:
        anchor = "    pads[0].SetGrid(1, 0)"
        if anchor not in source:
            raise WorkflowError(
                f"Could not patch {plot_script}: expected plotImpacts.py pad anchor "
                "was not found."
            )
        source = source.replace(anchor, margin_line + "\n" + anchor, 1)

    # Move ROOT's x10^N exponent to the left by 10% of the pad width.  This is
    # applied immediately before drawing the impact histogram, so the exponent
    # remains visible without changing the numerical axis representation.
    exponent_line = '    ROOT.TGaxis.SetExponentOffset(0.0, 0.0, "x")'
    if exponent_line not in source:
        anchor = "    h_impacts.Draw()"
        if anchor not in source:
            raise WorkflowError(
                f"Could not patch {plot_script}: expected h_impacts.Draw() anchor "
                "was not found."
            )
        source = source.replace(anchor, exponent_line + "\n" + anchor, 1)

    local_script = outdir / "_plotImpacts_NPS26009_top30.py"
    local_script.write_text(source)
    print(
        "[IMPACT PLOT] Using a local plotImpacts.py copy with top-30 display, "
        "a wider impact panel, and an inward-shifted x10^N exponent."
    )
    return local_script


def run_impacts(
    args: argparse.Namespace,
    card: Path,
    target: str,
    label: str,
    mass: float,
    outdir: Path,
) -> None:
    if not should_run_impacts(args, mass):
        print(f"[{target}] skip impacts for M-{label}")
        return

    tag = tag_for(args, target)
    name = f"Impacts_{tag}_M{label}"
    workspace = outdir / f"workspace_{tag}_M{label}.root"
    run_command(
        ["text2workspace.py", str(card), "-m", format_mass_for_combine(mass), "-o", workspace.name],
        outdir,
    )

    suffix = "_internal" if args.parameter == "alpha" else ""
    internal_json = outdir / f"impacts_{tag}_M{label}{suffix}.json"
    impact_rmin, impact_rmax, safe_boundary = initial_impact_range(args, card)

    if args.allow_negative_r:
        print(
            f"[IMPACT-RANGE] {target} M-{label}: requested rMin={effective_r_min(args):.6g}, "
            f"using rMin={impact_rmin:.6g}, rMax={impact_rmax:.6g}, "
            f"nominal safe lower bound={safe_boundary:.6g}"
        )

    payload_internal: Optional[MutableMapping[str, Any]] = None
    completed = False
    for attempt in range(args.impact_range_retries + 1):
        # Remove only MultiDimFit products from an earlier range attempt.
        for pattern in (
            f"higgsCombine_initialFit_*.MultiDimFit.mH*.root",
            f"higgsCombine_paramFit_*.MultiDimFit.mH*.root",
        ):
            for old_output in outdir.glob(pattern):
                if name in old_output.name:
                    old_output.unlink()
        if internal_json.exists():
            internal_json.unlink()

        common = [
            "combineTool.py", "-M", "Impacts", "-d", str(workspace),
            "-m", format_mass_for_combine(mass),
            "--robustFit", "1",
            "--cminDefaultMinimizerStrategy", "0",
            "--setParameters", "r=0",
            *explicit_range_args(impact_rmin, impact_rmax, card),
            "-n", f".{name}",
        ]
        if args.mode == "blind":
            common.extend(["-t", "-1", "--expectSignal", "0"])

        if run_command([*common, "--doInitialFit"], outdir, allow_failure=True) != 0:
            nonfatal_or_raise(args, f"Impact initial fit failed for {target} M-{label}.")
            return
        dofits = [*common, "--doFits"]
        if args.impact_parallel > 1:
            dofits.extend(["--parallel", str(args.impact_parallel)])
        if run_command(dofits, outdir, allow_failure=True) != 0:
            nonfatal_or_raise(args, f"Impact nuisance fits failed for {target} M-{label}.")
            return

        if run_command([
            "combineTool.py", "-M", "Impacts", "-d", str(workspace),
            "-m", format_mass_for_combine(mass), "-n", f".{name}",
            "-o", internal_json.name,
        ], outdir, allow_failure=True) != 0:
            nonfatal_or_raise(args, f"Impact JSON collection failed for {target} M-{label}.")
            return

        payload = json.loads(internal_json.read_text())
        if not isinstance(payload, MutableMapping):
            raise WorkflowError(f"Unexpected impact JSON structure: {internal_json}")
        payload_internal = payload

        at_boundary, fit = impact_fit_reaches_lower_boundary(payload, impact_rmin)
        can_retry = (
            at_boundary
            and args.allow_negative_r
            and args.auto_expand_impact_r_range
            and attempt < args.impact_range_retries
        )
        if can_retry:
            next_rmin = expanded_impact_rmin(impact_rmin, safe_boundary)
            if next_rmin < impact_rmin - 1.0e-9:
                fit_text = "unknown" if fit is None else "/".join(f"{x:.6g}" for x in fit)
                print(
                    f"[IMPACT-RANGE] {target} M-{label}: POI fit [{fit_text}] reached "
                    f"rMin={impact_rmin:.6g}; retry with rMin={next_rmin:.6g}."
                )
                impact_rmin = next_rmin
                continue

        if at_boundary and args.allow_negative_r:
            fit_text = "unknown" if fit is None else "/".join(f"{x:.6g}" for x in fit)
            nonfatal_or_raise(
                args,
                f"Impact POI fit for {target} M-{label} remains at the lower boundary "
                f"after range expansion: fit=[{fit_text}], rMin={impact_rmin:.6g}, "
                f"safe lower bound={safe_boundary:.6g}.",
            )
            if args.strict:
                return
        completed = True
        break

    if not completed or payload_internal is None:
        nonfatal_or_raise(args, f"Impact range retries were exhausted for {target} M-{label}.")
        return

    plot_json = internal_json
    if args.parameter == "alpha":
        unit = alpha_unit_from_card(card)
        rescale_impact_payload(payload_internal, unit)
        plot_json = outdir / f"impacts_{tag}_M{label}.json"
        plot_json.write_text(json.dumps(payload_internal, indent=2, sort_keys=True) + "\n")

    plot_impacts_executable = shutil.which("plotImpacts.py")
    if plot_impacts_executable is None:
        nonfatal_or_raise(args, f"plotImpacts.py is unavailable; retained {plot_json}.")
        return

    plot_impacts_script = nps_endpoint_safe_plot_impacts_script(
        Path(plot_impacts_executable), outdir
    )

    translation = outdir / f"impact_translation_{args.parameter}.json"
    poi_label = {
        "alpha": "#alpha_{qZ'}",
        "yield": "N_{sig}",
        "xsec": "#sigma [pb]",
    }[args.parameter]
    translation.write_text(json.dumps({"r": poi_label}, indent=2) + "\n")
    output = f"impacts_{tag}_M{label}"

    # The fit contains every nuisance. The figure intentionally displays only
    # the 30 largest absolute impacts, matching the ranking used in the AN.
    common_plot_args = [
        sys.executable, str(plot_impacts_script),
        "-i", plot_json.name,
        "-o", output,
        "--sort", "impact",
        "--per-page", "30",
        "--max-pages", "1",
    ]

    status = run_command(
        [*common_plot_args, "--translate", translation.name],
        outdir, allow_failure=True
    )
    if status != 0:
        status = run_command(
            common_plot_args,
            outdir, allow_failure=True
        )
    if status != 0:
        nonfatal_or_raise(args, f"Impact plotting failed for {target} M-{label}.")


def find_value_for_mass(mapping: Mapping[float, float], mass: float, description: str) -> float:
    matches = [(abs(key-mass), value) for key, value in mapping.items()]
    if not matches or min(matches)[0] > 1.0e-6:
        raise WorkflowError(f"No {description} matches mass {mass}.")
    return min(matches)[1]


def rescale_limit_json(
    args: argparse.Namespace,
    internal_json: Path,
    physical_json: Path,
    unit_map: Mapping[float, float],
    rmax_map: Mapping[float, float],
) -> None:
    payload = json.loads(internal_json.read_text())
    if not isinstance(payload, dict):
        raise WorkflowError(f"Unexpected limit JSON structure: {internal_json}")
    missing: List[str] = []
    boundaries: List[str] = []
    for mass_key, values in payload.items():
        if not isinstance(values, dict):
            continue
        try:
            mass = float(mass_key)
        except ValueError:
            continue
        unit = find_value_for_mass(unit_map, mass, "alpha unit")
        rmax = find_value_for_mass(rmax_map, mass, "rMax")
        for key in (*EXPECTED_LIMIT_KEYS, "obs"):
            value = values.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if value >= 0.98 * rmax:
                    boundaries.append(f"M-{mass_key} {key}={value:.6g} near rMax={rmax:.6g}")
                values[key] = value * unit
        required = list(EXPECTED_LIMIT_KEYS)
        if args.mode == "unblind":
            required.append("obs")
        absent = [key for key in required if key not in values]
        if absent:
            missing.append(f"M-{mass_key}: {', '.join(absent)}")
    physical_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if missing:
        raise WorkflowError(
            "Combine did not produce all required limit quantiles:\n  - " + "\n  - ".join(missing)
        )
    if boundaries:
        raise WorkflowError(
            "Limit quantiles reached the POI scan boundary:\n  - " + "\n  - ".join(boundaries)
            + "\nIncrease --r-max or --max-internal-signal-yield and rerun."
        )


def y_axis_title(parameter: str) -> str:
    return {
        "alpha": "95% CL upper limit on #alpha_{qZ'}",
        "yield": "95% CL upper limit on selected signal events",
        "xsec": "95% CL upper limit on #sigma [pb]",
    }[parameter]


def lumi_label(target: str) -> str:
    # Keep the CMS luminosity text identical to the NIsoMuon plotter.py setup.
    # Combined Run labels are intentionally rounded to 138 and 62 fb^-1.
    labels = {
        "2016preVFP": "19.5 fb^{-1} (13 TeV)",
        "2016postVFP": "16.8 fb^{-1} (13 TeV)",
        "2017": "41.5 fb^{-1} (13 TeV)",
        "2018": "59.8 fb^{-1} (13 TeV)",
        "2022": "7.98 fb^{-1} (13.6 TeV)",
        "2022EE": "26.67 fb^{-1} (13.6 TeV)",
        "2023": "17.7 fb^{-1} (13.6 TeV)",
        "2023BPix": "9.5 fb^{-1} (13.6 TeV)",
        "Run2": "138 fb^{-1} (13 TeV)",
        "Run3": "62 fb^{-1} (13.6 TeV)",
        "Run2Run3": "138 fb^{-1} (13 TeV) + 62 fb^{-1} (13.6 TeV)",
    }
    try:
        return labels[target]
    except KeyError as exc:
        raise WorkflowError(f"Unknown luminosity-label target: {target}") from exc

def collect_limits_json(
    outputs: Sequence[Path], output_json: Path, outdir: Path
) -> None:
    if not outputs:
        raise WorkflowError(f"No AsymptoticLimits outputs were supplied for {output_json.name}.")
    run_command([
        "combineTool.py", "-M", "CollectLimits", *[str(path) for path in outputs],
        "-o", output_json.name,
    ], outdir)


def merge_observed_with_blind_expected(
    observed_json: Path, expected_json: Path, merged_json: Path
) -> None:
    observed_payload = json.loads(observed_json.read_text())
    expected_payload = json.loads(expected_json.read_text())
    if not isinstance(observed_payload, dict) or not isinstance(expected_payload, dict):
        raise WorkflowError("Unexpected CollectLimits JSON structure.")

    merged: Dict[str, Dict[str, Any]] = {}
    for mass_key, expected_values in expected_payload.items():
        if not isinstance(expected_values, dict):
            continue
        values = dict(expected_values)
        observed_values = observed_payload.get(mass_key, {})
        if isinstance(observed_values, dict) and isinstance(observed_values.get("obs"), (int, float)):
            values["obs"] = observed_values["obs"]
        merged[mass_key] = values
    merged_json.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")


def collect_and_plot_limits(
    args: argparse.Namespace,
    target: str,
    cards: Sequence[Path],
    observed_outputs: Sequence[Path],
    expected_outputs: Sequence[Path],
    outdir: Path,
) -> None:
    if not observed_outputs or not expected_outputs:
        raise WorkflowError(f"No AsymptoticLimits outputs were produced for {target}.")

    suffix = "_internal" if args.parameter == "alpha" else ""
    internal_json = outdir / f"limits_{target}_{args.parameter}{suffix}.json"

    if args.mode == "blind":
        collect_limits_json(expected_outputs, internal_json, outdir)
    else:
        observed_json = outdir / f"limits_{target}_{args.parameter}_observed{suffix}.json"
        expected_json = outdir / f"limits_{target}_{args.parameter}_blindExpected{suffix}.json"
        collect_limits_json(observed_outputs, observed_json, outdir)
        collect_limits_json(expected_outputs, expected_json, outdir)
        merge_observed_with_blind_expected(observed_json, expected_json, internal_json)
        print(
            f"[LIMIT] {target}: observed quantiles are taken from data; expected bands "
            "are taken from a separate --run blind Asimov calculation."
        )

    plot_json = internal_json
    if args.parameter == "alpha":
        unit_map: Dict[float, float] = {}
        rmax_map: Dict[float, float] = {}
        for card in cards:
            _label, mass = extract_mass_from_card(card, target)
            unit_map[mass] = alpha_unit_from_card(card)
            rmax_map[mass] = effective_r_max(args, card)
        plot_json = outdir / f"limits_{target}_alpha.json"
        rescale_limit_json(args, internal_json, plot_json, unit_map, rmax_map)

    plot_script = resolve_path(args.plot_limits_py, "plotLimits.py")
    show = "exp" if args.mode == "blind" else "exp,obs"
    run_command([
        sys.executable, str(plot_script), plot_json.name,
        "--logy", "--show", show,
        "--x-title", "m_{Z'} (GeV)",
        "--y-title", y_axis_title(args.parameter),
        "--title-right", lumi_label(target),
        "--output", f"limits_{target}_{args.parameter}",
    ], outdir)


def run_target(args: argparse.Namespace, target: str) -> None:
    cards = cards_for_target(args, target)
    if not cards:
        raise WorkflowError(f"No datacards found for {target}: {card_pattern(args, target)}")
    outdir = output_dir(args, target)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = tag_for(args, target)
    observed_limit_outputs: List[Path] = []
    expected_limit_outputs: List[Path] = []

    for card in cards:
        label, mass = extract_mass_from_card(card, target)
        clean_mass_outputs(outdir, tag, label, args.task)
        rmax = effective_r_max(args, card)
        if args.parameter == "alpha":
            unit = alpha_unit_from_card(card)
            print(
                f"[{target}] M-{label}: alpha_unit={unit:.8g}, "
                f"internal r=[{effective_r_min(args):.4g},{rmax:.4g}], "
                f"physical alpha max={unit*rmax:.8g}"
            )
        else:
            print(f"[{target}] M-{label}: r=[{effective_r_min(args):.4g},{rmax:.4g}]")

        if args.task in {"limits", "all"}:
            result = run_asymptotic(args, card, target, label, mass, outdir)
            observed_limit_outputs.append(result.observed)
            expected_limit_outputs.append(result.expected)
        if args.task in {"fitdiag", "checks", "all"}:
            run_fitdiagnostics(args, card, target, label, mass, outdir)
        if args.task in {"impacts", "all"}:
            run_impacts(args, card, target, label, mass, outdir)

    if args.task in {"limits", "all"}:
        collect_and_plot_limits(
            args, target, cards, observed_limit_outputs, expected_limit_outputs, outdir
        )


def run_combine(args: argparse.Namespace) -> None:
    require_command("combine")
    require_command("combineTool.py")
    if args.task in {"impacts", "all"}:
        require_command("text2workspace.py")
    for target in selected_targets(args.target):
        run_target(args, target)


# -------------------------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="limit_workflow.py",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Standalone NIsoMuon Run-2/Run-3 datacard + Combine limit/impact workflow (20260826_0541).",
    )
    parser.add_argument("--stage", choices=("cards", "run", "all"), default="all")
    parser.add_argument("--target", type=canonical_target, default="run2")
    parser.add_argument("--parameter", type=canonical_parameter, default="alpha")
    parser.add_argument("--mode", choices=("blind", "unblind"), default="blind")
    parser.add_argument(
        "--task", choices=("limits", "fitdiag", "impacts", "checks", "all"),
        default="limits", help="all = limits + FitDiagnostics/pulls + impacts."
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--masses", type=parse_float_list, default=None)

    inputs = parser.add_argument_group("ROOT inputs and selection")
    inputs.add_argument("--base-dir", default=DEFAULT_BASE_DIR)
    inputs.add_argument("--trigger", default="")
    inputs.add_argument("--region", default=DEFAULT_REGION)
    inputs.add_argument(
        "--sigfit-dir", default=".",
        help=(
            "Directory containing resolution_coefficients.json/csv and "
            "sigFit_results.json/csv. JSON is preferred when both formats exist. "
            "Default: current working directory."
        ),
    )
    inputs.add_argument(
        "--resolution-map", default=None,
        help=(
            "Optional explicit JSON/CSV override for sigma_m/m = a*m + b. "
            "The sigFit result file is still read from --sigfit-dir for validation. "
            "JSON example: {\"2022\":[a,b], ...}; CSV columns: era,a,b."
        ),
    )
    inputs.add_argument("--qcd-method", choices=("data-driven", "mc"), default="data-driven")
    inputs.add_argument("--dy-method", choices=("data-driven", "mc"), default="data-driven")
    inputs.add_argument("--n-sigma", type=float, default=5.0)
    inputs.add_argument("--absolute-resolution", action="store_true")
    inputs.add_argument("--require-all-years", action="store_true", default=True)
    inputs.add_argument("--no-require-all-years", dest="require_all_years", action="store_false")

    theory = parser.add_argument_group("systematic inputs")
    theory.add_argument(
        "--generator-theory-processes", type=parse_process_list, default=("tt", "ST"),
        help=(
            "Processes requiring generator PDF/alpha_s/scale templates.\n"
            "Default: tt,ST.  Add sig when signal variations are available."
        ),
    )
    theory.add_argument(
        "--experimental-processes", type=parse_process_list,
        default=("sig", "tt", "ST", "Others"),
        help="MC processes requiring JER/JES/PU/BTV-btag/muon variations."
    )
    theory.add_argument("--no-generator-theory", dest="enable_generator_theory", action="store_false")
    theory.add_argument("--enable-generator-theory", dest="enable_generator_theory", action="store_true", default=True)
    theory.add_argument("--ratio-floor", type=float, default=1.0e-6)
    theory.add_argument("--rate-floor", type=float, default=0.0)
    theory.add_argument("--ignore-rel-below", type=float, default=0.0)
    theory.add_argument(
        "--lnn-warning-factor", type=float, default=1.0e3,
        help="Warn for lnN factors outside [1/factor, factor]; default: 1e3.",
    )
    theory.add_argument(
        "--fail-on-suspicious-systematics", action="store_true",
        help=(
            "Promote duplicated experimental responses, non-bracketing Up/Down "
            "pairs, and pathological lnN factors to a hard failure."
        ),
    )

    outputs = parser.add_argument_group("outputs")
    outputs.add_argument("--card-base", default="datacards_NIsoMuon")
    outputs.add_argument("--output-base", default="limit_outputs")
    outputs.add_argument("--plot-limits-py", default=DEFAULT_PLOT_LIMITS_PY)

    alpha = parser.add_argument_group("alpha internal scaling")
    alpha.add_argument("--alpha-card-yield", type=float, default=25.0)
    alpha.add_argument("--alpha-internal-unit", type=float, default=None)
    alpha.add_argument(
        "--max-internal-signal-yield", type=float, default=2500.0,
        help="Default 2500; with --alpha-card-yield 25 this gives alpha-mode rMax=100."
    )

    xsec = parser.add_argument_group("cross-section interpretation")
    xsec.add_argument("--signal-xsec-map", default=None)
    xsec.add_argument("--signal-reference-xsec-pb", type=float, default=None)

    combine = parser.add_argument_group("Combine controls")
    combine.add_argument("--r-min", type=float, default=None)
    combine.add_argument("--r-max", type=float, default=None)
    combine.add_argument(
        "--allow-negative-r",
        action="store_true",
        help=(
            "Diagnostic only: allow a negative internal signal strength. "
            "Do not use this option for the final physical upper limit."
        ),
    )
    combine.add_argument(
        "--auto-expand-impact-r-range",
        dest="auto_expand_impact_r_range",
        action="store_true",
        default=True,
        help=(
            "With --allow-negative-r, expand the impact-fit lower range when the "
            "nominal estimate or fitted POI reaches rMin. Enabled by default."
        ),
    )
    combine.add_argument(
        "--no-auto-expand-impact-r-range",
        dest="auto_expand_impact_r_range",
        action="store_false",
    )
    combine.add_argument(
        "--impact-range-retries",
        type=int,
        default=3,
        help="Maximum automatic impact-fit lower-range retries; default: 3.",
    )
    combine.add_argument("--default-yield-r-max", type=float, default=500.0)
    combine.add_argument("--default-xsec-r-max", type=float, default=10.0)
    combine.add_argument("--impact-masses", type=parse_float_list, default=None)
    combine.add_argument("--run-impacts-all-masses", dest="run_impacts_all_masses", action="store_true", default=True)
    combine.add_argument("--no-run-impacts-all-masses", dest="run_impacts_all_masses", action="store_false")
    combine.add_argument("--impact-parallel", type=int, default=1)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n_sigma <= 0.0:
        raise WorkflowError("--n-sigma must be positive.")
    if args.alpha_card_yield <= 0.0:
        raise WorkflowError("--alpha-card-yield must be positive.")
    if args.alpha_internal_unit is not None and args.alpha_internal_unit <= 0.0:
        raise WorkflowError("--alpha-internal-unit must be positive.")
    if args.max_internal_signal_yield <= 0.0:
        raise WorkflowError("--max-internal-signal-yield must be positive.")
    if args.lnn_warning_factor <= 1.0:
        raise WorkflowError("--lnn-warning-factor must be larger than 1.")
    used_scale_indices = {
        index
        for direction in (*SCALE_NUISANCE_DIRECTIONS, SCALE_DIAGONAL_DIRECTION)
        for index in scale_pair(direction)
    }
    if SCALE_CENTRAL_INDEX in used_scale_indices:
        raise WorkflowError("The central scale member cannot be a scale nuisance.")
    bad_scale = sorted(used_scale_indices & EXCLUDED_ANTIPODAL_SCALE_INDICES)
    if bad_scale:
        raise WorkflowError(
            "The fixed scale prescription unexpectedly uses excluded member(s): "
            + ",".join(map(str, bad_scale))
        )
    if args.impact_parallel < 1:
        raise WorkflowError("--impact-parallel must be at least 1.")
    if args.impact_range_retries < 0:
        raise WorkflowError("--impact-range-retries must be non-negative.")
    if args.r_min is not None and args.r_min < 0.0 and not args.allow_negative_r:
        raise WorkflowError(
            "Negative --r-min is diagnostic only. Add --allow-negative-r explicitly."
        )
    if args.r_max is not None and args.r_max <= effective_r_min(args):
        raise WorkflowError("--r-max must be larger than --r-min.")
    for process in args.generator_theory_processes:
        if process not in PROCESSES:
            raise WorkflowError(f"Invalid generator-theory process: {process}")
        if process == "QCD" and args.qcd_method == "data-driven":
            raise WorkflowError(
                "Generator theory templates cannot be assigned to data-driven QCD."
            )
        if process == "DY" and args.dy_method == "data-driven":
            raise WorkflowError(
                "Generator theory templates cannot be assigned to data-driven DY."
            )
    for process in args.experimental_processes:
        if process not in PROCESSES:
            raise WorkflowError(f"Invalid experimental process: {process}")
    if args.parameter == "xsec" and "Run2Run3" in selected_targets(args.target):
        raise WorkflowError(
            "A single xsec POI is not defined for the combined 13 and 13.6 TeV target. "
            "Use --parameter alpha/yield, or run the xsec interpretation separately for Run2 and Run3."
        )


def print_configuration(args: argparse.Namespace) -> None:
    print("[CONFIG] workflow_tag=20260826_0541")
    print(f"[CONFIG] stage={args.stage}, task={args.task}")
    print(f"[CONFIG] target={args.target}, parameter={args.parameter}, mode={args.mode}")
    print(f"[CONFIG] base_dir={args.base_dir}")
    print(f"[CONFIG] region={args.region}")
    print(f"[CONFIG] search_mass_window=[{SEARCH_MASS_MIN:g},{SEARCH_MASS_MAX:g}] GeV")
    print(f"[CONFIG] QCD={args.qcd_method}, DY={args.dy_method}")
    if args.dy_method == "data-driven":
        print("[CONFIG] DY=constant NF; DY_NFStat + DY_LightJetStat; DY_stat=disabled")
    if args.qcd_method == "data-driven":
        print("[CONFIG] data-driven QCD_stat=disabled")
    print(
        "[CONFIG] btag=BTV fixed-WP; HF/LF separated; "
        "corr shared within Run2 or within Run3 and independent between runs; "
        "uncorr decorrelated by era"
    )
    print("[CONFIG] tt_xsec/ST_xsec=disabled")
    print("[CONFIG] L1 prefiring syst=2016preVFP,2016postVFP,2017; decorrelated by era")
    d13, u13 = tt_mass_lnn("2018")
    d136, u136 = tt_mass_lnn("2023")
    print(
        "[CONFIG] tt_mass=enabled, one nuisance correlated across Run2+Run3; "
        f"13TeV={d13:.6g}/{u13:.6g}, 13.6TeV={d136:.6g}/{u136:.6g}"
    )
    if hasattr(args, "resolution_coefficients"):
        print(f"[CONFIG] resolution_source={args.resolution_source}")
        print(f"[CONFIG] sigfit_results_source={args.sigfit_results_source}")
        print(
            f"[CONFIG] sigfit_audit={args.sigfit_audit['accepted']}/"
            f"{args.sigfit_audit['total']} accepted"
        )
        print(
            "[CONFIG] resolution coefficients="
            + ",".join(
                f"{year}:({args.resolution_coefficients[year][0]:.8g},{args.resolution_coefficients[year][1]:.8g})"
                for year in years_needed_for_request(args)
            )
        )
    if args.enable_generator_theory:
        print(
            "[CONFIG] generator PDF/scale/alphaS=enabled for "
            + ",".join(args.generator_theory_processes)
        )
        print(
            "[CONFIG] scale directions="
            + ",".join(SCALE_NUISANCE_DIRECTIONS)
            + " (separate paired nuisances; diagonal validation-only)"
        )
        print(
            f"[CONFIG] PDF set={PDF_SET_NAME}; "
            f"stored PDFError0..{PDF_EIGENVECTOR_COUNT - 1} "
            "(LHAPDF members 1..100); method=symmetric-Hessian quadrature"
        )
    else:
        print("[CONFIG] generator PDF/scale/alphaS=disabled")
    print(f"[CONFIG] cards={args.card_base}/{args.parameter}/{args.mode}")
    print(f"[CONFIG] outputs={args.output_base}/{args.parameter}/{args.mode}")
    if args.allow_negative_r:
        print(
            "[CONFIG] WARNING: negative internal r is enabled for a diagnostic fit; "
            "do not use this setting for the final physical upper limit."
        )
    if args.parameter == "alpha":
        if args.alpha_internal_unit is None:
            default_rmax = args.max_internal_signal_yield / args.alpha_card_yield
            print(f"[CONFIG] alpha internal scaling: r=1 -> {args.alpha_card_yield:g} signal events/card")
            print(f"[CONFIG] default alpha rMax={default_rmax:g}")
        else:
            print(f"[CONFIG] fixed alpha_internal_unit={args.alpha_internal_unit:g}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
        if args.stage in {"cards", "all"}:
            prepare_resolution_coefficients(args)
        print_configuration(args)
        if args.stage in {"cards", "all"}:
            build_cards(args)
        if args.stage in {"run", "all"}:
            run_combine(args)
        print(
            f"[DONE] workflow_tag=20260826_0541, target={args.target}, parameter={args.parameter}, "
            f"mode={args.mode}, stage={args.stage}, task={args.task}"
        )
        return 0
    except (WorkflowError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
