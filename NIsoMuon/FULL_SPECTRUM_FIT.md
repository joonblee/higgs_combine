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

The full-spectrum input audit distinguishes required templates and consistency
checks from the validation-only diagonal scale diagnostic. A bin's simultaneous
muR/muF response is compared with the product of its separate responses; a
large residual is printed as `[SCALE DIAGNOSTIC]` and retained with the observed
and predicted ratios in `model.json` (`input_audits[].diagnostic_failures` and
the row's `warnings`). It does not stop card building. The existing separate
muR and muF nuisance responses are retained. Missing scale members, a central
scale/nominal mismatch, and other required-input failures still stop execution.
The original counting and sideband audit policies are unchanged.

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

If card building stops, fix the reported error and rebuild before running a
fit-only injection test. For example:

```bash
python3 full_spectrum_fit.py --masses 30 --target full --overwrite && \
python3 full_spectrum_fit.py --masses 30 --target full --stage fit --inject-r 1
```

The `&&` runs the injected fit only after the first command succeeds. A failed
build has no usable `model.json`/datacard; the fit-only stage reports this and
the rebuild command instead of a generic file-not-found error.

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

The default executes locally and sequentially. The Condor backend below
distributes the expensive **input preparation for every era and mass bin**,
then constructs and fits the same combined likelihood.

## Condor preparation by era and mass bin

After activating the Combine CMSSW environment, submit from `NIsoMuon`:

```bash
python3 full_spectrum_fit.py --masses 30 --target full \
    --backend condor --bins-per-job 1 --condor-maxjobs 100 \
    --condor-dir full_spectrum_condor/M30_r0
```

This creates a new batch directory and submits a DAG automatically:

1. One preparation job per **(signal mass hypothesis, era, spectrum bin)**.
   Each reads nominal and varied inputs, performs the required audits and
   stores raw yields, SS covariance responses and source statistics. These
   are independent preparation jobs, not independent fits of the mass bins.
2. One assembly job per mass, after **all** its bins finish. It checks exact
   bin coverage, input consistency and mass independence of the replaced
   normalisations across the entire era. Signal rescaling to 25 events is
   applied once over the full selected spectrum, including all eras.
3. One joint `FitDiagnostics` job per mass, after assembly. It includes every
   selected era/bin and retains the correlations of the local model. Both
   B-only and floating-signal fits use the requested Asimov dataset.

For one mass with N total era/bin entries, the default creates **N preparation
jobs + 1 assembly job + 1 fit job**. `--stage cards` omits the last job.
`--masses 12,30,70` creates independent chains for all three hypotheses.
Submission reads nominal axes and audits the resolution inputs once; it does
not run the spectrum-bin preparation loop on the submitting machine.

`--bins-per-job 1` gives the finest distribution. Increase it, for example to
5 or 10, to amortise ROOT startup and file-reading overhead over several bins.
This changes job grouping only, never the binning or statistical model.
`--condor-maxjobs 100` caps simultaneously submitted DAG nodes; 0 leaves the
limit to pool scheduling. This is a concurrency cap, not a guaranteed number
of available CPUs. Jobs request one CPU and `--condor-memory 4000` MB each.
`--condor-retries 2` retries failed nodes twice before their dependants stop.
Optional `--condor-requirements` is passed to the submit file.

The worker wrapper activates the submitting user's `CMSSW_BASE` and
`SCRAM_ARCH`. Override them with `--condor-cmssw-base` and
`--condor-scram-arch` if needed. As with the existing analysis Condor backend,
the checkout's ROOT inputs, the batch directory, CMSSW and CVMFS must be
accessible on the worker nodes; files are not transferred by Condor.

To prepare a real DAG without submitting, add `--condor-dry-run`. This reads
nominal inputs and requires PyROOT/CMSSW. The separate `--dry-run` still prints
fit commands without reading ROOT or creating a batch.

Monitor, verify or resume with:

```bash
python3 full_spectrum_fit.py --stage status \
    --condor-dir full_spectrum_condor/M30_r0

python3 full_spectrum_fit.py --stage collect \
    --condor-dir full_spectrum_condor/M30_r0

python3 full_spectrum_fit.py --stage resume \
    --condor-dir full_spectrum_condor/M30_r0
```

`status` reports completed bins per era, job states, recorded failures and
assembly/fit elapsed times. `collect` verifies every expected completion and
product; assembly and fitting happen automatically in the DAG. `resume`
creates and submits a new DAG for incomplete or stale jobs and their affected
dependants, preserving verified bin results. Wait until the old DAG has exited,
or remove it and its queued node jobs before resuming; an active batch is
rejected to avoid duplicate workers. Add `--condor-dry-run` to inspect the
resume DAG without submitting. Resume uses the batch's original options.

Every batch has an immutable manifest and source snapshot. Worker inputs are
recorded with size/mtime fingerprints and checked for changes during preparation
and assembly. Missing/stale shards, changed products or incomplete coverage
stop the dependent jobs. Updating the checkout does not change an existing
batch. `--overwrite` does not replace a batch directory; choose a new directory
for a fresh run. Batch results are isolated from any ongoing local run.

For the example above, results are written under:

    full_spectrum_condor/M30_r0/results/Run2Run3_binsigma1_lnN1p5/M30/

The normal `model.json`, `datacard.txt` and `fit_r0/fit_summary.json` live there.
Per-job status, raw parts and elapsed time are in `jobs/job_00000/status.json`
and `part.json`; Condor stdout/stderr use `jobs/job_<index>.out` and `.err`.
`[PREP]` messages include the era, bin ID and actual mass interval, instead of
only repeating the tested signal mass. QCD derivative validation is cached
within each worker, while each bin's statistical propagation remains separate.

To reuse the assembled card for an r=1 Asimov test in another Condor batch,
without preparing the templates again:

```bash
python3 full_spectrum_fit.py --masses 30 --target full --stage fit \
    --backend condor --inject-r 1 \
    --output-base full_spectrum_condor/M30_r0/results \
    --condor-dir full_spectrum_condor/M30_r1
```

Use the same physics/input options as the original card. The new batch copies
the verified card and model and runs only the joint fit.

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
python3 -m unittest discover -s tests -p 'test_full_spectrum_condor.py' -v
```

These check full-domain native-bin coverage, prior/bin correlations, nominal
closure, zero-DY statistics, blinding, saved-card protection and CLI targets.
When PyROOT and Combine are available, the synthetic two-era test also runs
text2workspace and both r=0/r=1 Asimov fits. Real-input fit performance must be
checked in the analysis CMSSW environment.

The Condor tests additionally exercise planning, raw shard assembly, exact
local/batch datacard equivalence, zero-signal shards, grouping independence,
dependency failures/retries, stale-product rejection, resume and snapshot
integrity. They do not require an HTCondor pool. Actual queue submission and
real-input worker/fit performance require the analysis cluster.

