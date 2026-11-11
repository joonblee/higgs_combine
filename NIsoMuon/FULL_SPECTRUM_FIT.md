# Step 1: blinded full-spectrum fit

Run `full_spectrum_fit.py` in **higgs_combine**, after activating the existing
CMSSW/Combine environment. SKPlotMaker supplies the current ROOT estimates;
there is no new SKPlotMaker production command for this step.

From the checked-out `NIsoMuon` directory:

```bash
python3 full_spectrum_fit.py --masses 30 --target full --dry-run
python3 full_spectrum_fit.py --masses 30 --target full
```

Start with one mass. Once that completes, build the other representative masses:

```bash
python3 full_spectrum_fit.py --masses 12,70 --target full
```

The default inputs are the existing per-era files under
`/data6/Users/joonblee/SKOutput/Run2UL_v3_Run3_v13/NIsoMuon`, with the same
nominal, `RunSyst` and `RunXSecSyst` layout used by `limit_workflow.py`.
Use `--base-dir`, `--sigfit-dir`, `--resolution-map` and `--trigger` if needed.
DYAux and QCDStat metadata/derivatives must match the current consumer contract;
missing or stale inputs fail rather than substituting MC backgrounds.

## Likelihood

The fit includes every bin in **11--80 GeV** and retains all eight eras as
separate channels. `--target Run2`, `Run3` or one era restricts that selection.
There is no excluded signal window and no preliminary sideband fit.

For process p in QCD, DY or ttbar and era e, the new factor is

    beta(p,e) = kappa ** theta(p,e),    theta(p,e) ~ Gaussian(0,1).

`--norm-prior 1.5` is the default: at theta = -1,+1 the scale is 1/1.5,1.5.
Each process/era has its own nuisance shared across all its mass bins. They
are constrained broad priors, not unrestricted rate parameters. ST and Others
retain their existing prescriptions. The QCD/DY nominal NF/transfer factors
provide the initial yields. DY NFStat/NFModel and pure QCD transfer uncertainties
are replaced by the broad normalisation prescription; they are not added twice.
The consumer verifies those replaced transfer/NF factors are mass independent.

The likelihood is serialized as one Poisson counting channel per disjoint mass
bin, all in a **single datacard**. This is a binned spectrum likelihood, not a
sequence of independent one-window limits. A nuisance's bin-dependent lnN
responses retain the inherited correlation names across bins/eras. QCD uses the
producer covariance modes and coherent additive functional envelope. Positive
DY-source bins retain source Sumw2 statistics; nonpositive integrated source
bins retain the existing zero-nominal/additive-error prescription. Their number
is reported per era; template smoothing or rebinning is not performed silently.
MC finite-template statistics remain the existing independent process/bin terms.
No `autoMCStats` line is added on top of those terms or the correlated QCD modes.

Nominal widths target `--bin-sigma 1` times the local mass resolution, using the
era's sigFit coefficients. Edges snap to the common native histogram grid and
must cover exactly 11--80 GeV. A short final bin is merged with its neighbour.
The resolution curve is evaluated over this domain for bin geometry only;
signal hypotheses remain in 12--70 GeV. The lowest available simulated mass
provides the reference native axis, independent of the requested hypothesis;
every integrated input/variation is checked against those bin edges.
No native histogram bin is split or used twice. Actual edges are saved in JSON.
The bin width is a configurable initial choice, to be checked for stability.

Signal templates are taken from the current generated-mass ROOT inputs, keeping
the full signal spectrum. Intermediate-mass interpolation is not enabled here.
For numerical conditioning, internal r=1 means **25 signal events over the full
11--80 GeV domain, summed over the selected eras**. This differs from the old
single-window convention. `alpha_internal_unit` records the conversion back to
the original signal-template normalisation. No limit curve is produced here.

## Fits and outputs

OS `data.root` is blocked by the reader; there is no unblinding option. Cards
contain nominal background Asimov observations. `FitDiagnostics -t -1` builds
the requested background-only or signal-injected Asimov dataset from the same
model and performs both the B-only and floating-signal fits.

```bash
# Build only, or reuse the protected card for a fit:
python3 full_spectrum_fit.py --masses 30 --target full --stage cards
python3 full_spectrum_fit.py --masses 30 --target full --stage fit

# Reuse the same card for a separate r=1 Asimov test:
python3 full_spectrum_fit.py --masses 30 --target full --stage fit --inject-r 1
```

The first `--stage cards` command is an alternative to `--stage all`, not a
required rerun after it. Nonempty card directories require `--overwrite` or a
new `--output-base`. Rebuilding invalidates old fit results. Fits verify the
card hash and model configuration before execution.

Example output directory:

    full_spectrum_outputs/Run2Run3_binsigma1_lnN1p5/M30/

- `datacard.txt`: full-spectrum likelihood.
- `model.json`: rates, bin edges, nuisance responses, input audit, normalisation
  mapping and signal scaling; nominal three-component separation diagnostics.
- `fit_r0/` or `fit_r1/`: `workspace.root`, `fitDiagnostics.FullSpectrum.root`,
  exact `commands.json` and execution logs.
- `fit_r*/fit_summary.json`: both fits' status, covariance quality, EDM, fitted
  r, QCD/DY/tt scale factors, standard-Gaussian nuisance pulls and constraints,
  their mutual correlations and postfit process yields per bin.

A fit must have status 0 and covariance quality 3 before a summary is written;
failed quality checks write `fit_failure.json`. The scale intervals are
transformed symmetric Hessian errors, not profile intervals. A constraint value
below one indicates shrinking relative to the unit Gaussian prior. Both norms
and shape responses can change a process's final integrated yield, so the broad
scale factor alone is not its total postfit/prefit yield ratio.

The default executes locally and sequentially; run it in an appropriate
interactive/compute allocation. It is not a new Condor backend.

## Scope and verification

This implements the first structural step. Existing experimental/theory
responses and statistical inputs are retained. Additional DY closure and QCD
transport shape uncertainties, alternative-background bias tests, final signal
interpolation and final limits are subsequent steps. Nominal Asimov closure
tests implementation and expected constraints, not background-model accuracy.
No analysis SR data are read and no validated physics-fit result is claimed.

Tests:

```bash
python3 -m unittest discover -s tests -p 'test_full_spectrum_fit.py' -v
python3 -m unittest discover -s tests -p 'test_sideband_fit.py' -v
```

These check full-domain native-bin coverage, prior/bin correlations, nominal
closure, zero-DY statistics, blinding, saved-card protection and CLI targets.
When PyROOT and Combine are available, the synthetic two-era test also runs
text2workspace and both r=0/r=1 Asimov fits. Real-input fit performance must be
checked in the analysis CMSSW environment.
