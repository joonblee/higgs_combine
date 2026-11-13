# Step 1: blinded full-spectrum fit

Run `full_spectrum_fit.py` in **higgs_combine**, after activating the existing
CMSSW/Combine environment. SKPlotMaker supplies the current ROOT estimates;
there is no new SKPlotMaker production command for this step.

Execution defaults to **Condor**, with **no submission cap added by the
workflow** (`--condor-maxjobs 0`). Every job requests **16000 MB** by default
(`--condor-memory 16000`). The default preparation grouping is one
spectrum bin per job (`--bins-per-job 1`). Use `--backend local` for foreground
execution. Condor results are isolated under each batch's `results/` directory.

## Signal-window counting with coarse sidebands

Use `--fit-layout window` for one integrated signal-window count per era plus
coarse lower/upper sideband bins. This is a simultaneous fit of disjoint
regions with shared background normalisations and systematic responses, not
a normalisation-only prefit. The default `--fit-layout spectrum` preserves the
existing full-spectrum geometry and saved batches.

```bash
python3 full_spectrum_fit.py --masses 30 --target full --fit-layout window \
    --sideband-binning central

# Independent stability comparisons (new batches):
python3 full_spectrum_fit.py --masses 30 --target full --fit-layout window \
    --sideband-binning fine
python3 full_spectrum_fit.py --masses 30 --target full --fit-layout window \
    --sideband-binning coarse
python3 full_spectrum_fit.py --masses 30 --target full --fit-layout window \
    --sideband-binning verycoarse
```

`--sideband-binning central` is the default for window layout. Motivated by
the supplied Run2+3 background plot, it resolves the changing low-mass
QCD/DY/tt mixture more finely. This is a starting proposal based on that plot,
not an optimisation on the original per-era templates.

| Preset | Reference bins over 11--80 before SR insertion | Definition |
| --- | ---: | --- |
| central | 20 | 1-GeV widths from 11--20, 2 GeV from 20--30, 5 GeV from 30--40, 10 GeV from 40--80 |
| fine | 40 | Bisect every central reference bin |
| coarse | 15 | Previous medium15 custom grid; retain 1-GeV low-mass bins |
| verycoarse | 10 | Previous coarse preset; merge adjacent central reference bins in pairs |

Central edges: **11,12,13,14,15,16,17,18,19,20,22,24,26,28,30,35,40,50,60,70,80 GeV**.
Coarse edges: **11,12,13,14,15,16,17,18,19,20,22,26,30,40,60,80 GeV**.
Verycoarse edges: **11,13,15,17,19,22,26,30,40,60,80 GeV**. Verycoarse bins can cross a
width-transition boundary such as 20 GeV to preserve the exact pair merging.
The presets are nested: verycoarse, coarse, central, fine. Reference counts
are before SR insertion, native-grid snapping and fragment merging; actual
sideband counts can differ. The SR boundaries are identical in all four cases.
The default remains central. `coarse` reproduces the earlier medium15 custom
edges exactly, including the geometry hash and saved-model signature;
`verycoarse` reproduces the previous coarse geometry. Existing batch snapshots
retain their original definitions. Use verycoarse to reproduce an old coarse
run with the updated source.

Override central with `--sideband-edges 11,15,20,25,30,40,50,60,80` to reproduce
the earlier proposal. Custom edges cannot be combined with fine/coarse/verycoarse presets.
Edges must be finite, strictly increasing and span 11--80.
`--sideband-refine 2` remains available to subdivide each selected reference
interval in two before snapping it to the common native grid.

For every requested signal mass and era, the existing +/-5-sigma window
(`--sr-nsigma 5`) is clipped to 11--80 and snapped outwards to whole native
bins. All reference edges inside it are removed, leaving exactly one `SR`
count. Sideband reference edges snap to the nearest common native edges.
An SR-adjacent sideband fragment narrower than 25% of its reference interval
is merged with its neighbour on the same side, when such a neighbour exists;
`--sideband-fragment-fraction` controls this threshold. SR boundaries never
move during this merging. Empty lower/upper sidebands at the domain edges
are omitted. The resulting regions cover 11--80 exactly, without overlap,
gaps or splitting native histogram bins. `[BINS]` reports actual counts.

The planning geometry is mass-specific, including in multi-mass Condor DAGs.
Yields, signed DY sources, template variations and statistical/covariance
responses are integrated afresh over each region from the original inputs.
Existing fine-bin error bars or nuisance values are not averaged. Signal
leakage into sidebands is included. The internal r=1 convention still means
25 signal events over the complete 11--80 domain, not 25 inside the SR.
Broad lnN 1.5 priors and all currently implemented uncertainty/correlation
prescriptions are retained. No new independent 30% DY modelling nuisance is
introduced by this binning change; that remains a separate modelling task.

Output model tags include `_window5sigma_sb<geometry hash>_lnN1p5`. Different
edges, refinement or fragment thresholds have separate paths and protected
model signatures. `model.json` stores the snapped per-era edges, signal
window and proposed sideband settings. Fine-binned saved cards cannot be
used as coarse-window cards: submit fresh preparation. `--fast-mode` and
nondefault `--bin-sigma` are rejected with this layout; its speed comes from
the coarse sideband geometry. Memory defaults to 16000 MB and B-only/S+B jobs
remain parallel, with no workflow concurrency cap.

These edges are a starting proposal, not a validated optimum. Compare nominal
and refined plans at representative masses (e.g. 12, 30, 70 GeV), examining
normalisations, errors/correlations and expected limits. A 5% limit change can
be a chosen study target, not a universal requirement. Keep the underlying
modelling-uncertainty prescription consistent: redefining independent shape
nuisances whenever the fit binning changes would also change the model.
This script runs blinded fit diagnostics; it does not yet implement a new
window-plus-sideband CLs limit calculation or an observed-data workflow.

From the checked-out `NIsoMuon` directory:

```bash
python3 full_spectrum_fit.py --masses 30 --target full --dry-run
python3 full_spectrum_fit.py --masses 30 --target full \
    --condor-dir full_spectrum_condor/M30_r0
```

Start with one mass. Once that completes, build the other representative masses:

```bash
python3 full_spectrum_fit.py --masses 12,70 --target full
```

For repeated development tests, `--fast-mode` multiplies the target bin width
by four (default: 4 sigma instead of 1 sigma). It covers the same 11--80 GeV
domain and all selected eras, but reduces the number of spectrum bins and
independent per-bin statistical nuisances. Each wider bin is integrated again
from the original inputs, with the same uncertainty/correlation prescriptions;
existing narrow-bin nuisance responses are not averaged together.

```bash
python3 full_spectrum_fit.py --masses 30 --target full --fast-mode
```

The broad normalisation priors, systematic uncertainties, B-only and S+B fits,
robust intervals and postfit uncertainty calculations are retained. This flag
changes binning only; it is distinct from the manual no-errors quick test.
Combining it with `--bin-sigma 2` targets 8 sigma. Coarse bins can lose narrow
signal sensitivity, so these are development results, not final limits or
full-resolution validation. `model.json` and `fit_summary.json` mark them as
`test_only: true`; model directories include the effective width and `_fast`.
Do not use an old 1-sigma saved model for a fast fit-only run: the coarse model
requires fresh preparation. Omit `--fast-mode` for the final 1-sigma run.
No speedup factor is guaranteed until timed on the production inputs.

Workers remove the stack-size limit with `ulimit -s unlimited` before loading
CMSSW. This addresses the observed local 8-MB stack-limit crash; it does not
increase the Condor memory allocation (`--condor-memory`, in MB).

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

Common-transfer reconstruction keeps the input ID, windows, estimator and
compatibility decisions exact. Derived floating-point entries in `rows` use
the same tolerance as the top-level common-fit numbers (`rel_tol=1e-10`,
`abs_tol=1e-14`). This permits last-bit differences between producer/worker
math libraries; it does not change the stored transfer factor, fit result or
uncertainty. A larger numerical discrepancy still stops with its field path
and saved/rebuilt values.

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

The Asimov observations use the central predictions, not their uncertainty
bands. The data term is Poisson, exactly as for observed counts; sqrt(n) is
only a familiar representation of its counting standard deviation. No AN
prediction uncertainty is added to an Asimov data error bar. Independently,
the prediction retains its SS-fit covariance, QCD functional envelope, DY
source and MC statistics and remaining detector/theory nuisance constraints.
Pure QCD NF/transport statistical responses and QCD_norm, DY_NFStat and
DY_NFModel are replaced by the broad normalisation treatment. Thus this is
not a fixed-shape, statistics-only fit, and the lnN 1.5 normalisation priors
mean the normalisations are not determined from data alone. More data bins
by themselves do not introduce nuisance parameters; the independent
template-statistical treatment of their source/MC inputs does.

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

For workspace construction, each fit writes a copy of the protected card to
`fit_r*/datacard.txt` with `shapes * * FAKE`. This selects Combine's simultaneous
one-bin-shape representation, avoiding underflow from multiplying hundreds of
Poisson probabilities in the counting-model builder (Combine issue #1075).
It requires no new ROOT templates and preserves all observations, rates and
nuisance definitions. The assembled card is unchanged; the summary records
both its hash and the hash of the card actually passed to `text2workspace.py`.

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

The commands in this subsection explicitly use the local backend. For Condor
execution and reuse of a batch's assembled card, see the next subsection.

```bash
# Build only, or reuse the protected card for a fit:
python3 full_spectrum_fit.py --backend local --masses 30 --target full --stage cards
python3 full_spectrum_fit.py --backend local --masses 30 --target full --stage fit

# Reuse the same card for a separate r=1 Asimov test:
python3 full_spectrum_fit.py --backend local --masses 30 --target full --stage fit --inject-r 1
```

The first `--stage cards` command is an alternative to `--stage all`, not a
required rerun after it. Nonempty card directories require `--overwrite` or a
new `--output-base`. Rebuilding invalidates old fit results. Fits verify the
card hash and model configuration before execution.

If card building stops, fix the reported error and rebuild before running a
fit-only injection test. For example:

```bash
python3 full_spectrum_fit.py --backend local --masses 30 --target full --overwrite && \
python3 full_spectrum_fit.py --backend local --masses 30 --target full --stage fit --inject-r 1
```

The `&&` runs the injected fit only after the first command succeeds. A failed
build has no usable `model.json`/datacard; the fit-only stage reports this and
the rebuild command instead of a generic file-not-found error.

Example local output directory:

    full_spectrum_outputs/Run2Run3_binsigma1_lnN1p5/M30/

- `datacard.txt`: full-spectrum likelihood.
- `model.json`: rates, bin edges, nuisance responses, input audit, normalisation
  mapping and signal scaling; nominal three-component separation diagnostics.
- `fit_r0/` or `fit_r1/`: `workspace.root`, `fitDiagnostics.FullSpectrum.root`,
  exact `commands.json` and execution logs.
- `fit_r*/fit_summary.json`: both fits' status, covariance quality, EDM, fitted
  r, QCD/DY/tt scale factors, standard-Gaussian nuisance pulls and constraints,
  their mutual correlations and postfit process yields per bin.

For new Condor batches, `fit_r*/workspace.root` is compiled once. Independent
`fit_b/` and `fit_s/` subdirectories each contain a FitDiagnostics ROOT file,
`commands.json`, `step1.log` and a validated partial `fit_summary.json`.
The final collector writes the combined `fit_r*/fit_summary.json`, with
`fit_outputs` pointing to the two ROOT files. It combines named fit results,
not their parameter values, covariance matrices or ROOT histograms. Local
foreground runs and original batch snapshots retain the sequential layout.

A fit must have status 0 and covariance quality 3 before a summary is written;
failed quality checks write `fit_failure.json`. The scale intervals are
transformed symmetric Hessian errors, not profile intervals. A constraint value
below one indicates shrinking relative to the unit Gaussian prior. Both norms
and shape responses can change a process's final integrated yield, so the broad
scale factor alone is not its total postfit/prefit yield ratio.

The default Condor backend distributes the expensive **input preparation for
every era and mass bin**, then constructs and fits the same combined likelihood.

If an older batch completed preparation and assembly but failed with
`modelObs_s = 0`, `function value is NAN` and `Missing fit_b`, pull the updated
code and submit only its saved model with a fresh source snapshot:

```bash
python3 full_spectrum_fit.py --stage fit --masses 30 --target full \
    --output-base full_spectrum_outputs/condor/OLD_BATCH/results \
    --condor-dir full_spectrum_outputs/condor/M30_pdf_fix
python3 full_spectrum_fit.py --stage collect \
    --condor-dir full_spectrum_outputs/condor/M30_pdf_fix
```

Replace `OLD_BATCH` with the original batch name. This submits workspace
compilation, two independent fit jobs and final summary collection. It
reuses the verified model/card and leaves the original batch untouched.
`--stage resume` on the original batch uses its old source snapshot and thus
does not apply this fix. Additional nondefault model options must match the
original configuration when loading its saved model.

## Condor preparation by era and mass bin

After activating the Combine CMSSW environment, submit from `NIsoMuon`:

```bash
python3 full_spectrum_fit.py --masses 30 --target full \
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
3. One workspace job per mass, after assembly. It compiles the verified card
   once for both fits.
4. Two independent joint `FitDiagnostics` jobs per mass, both after the
   workspace job: B-only (`--skipSBFit`) and S+B (`--skipBOnlyFit`). Each uses
   every selected era/bin, the same deterministic Asimov dataset and all
   nuisance correlations. Each performs its own uncertainty/postfit diagnostics.
   They write separate directories and may run concurrently.
5. A lightweight summary job, after both fits pass their quality checks.

For one mass with N total era/bin entries, the default creates **N preparation
jobs + 1 assembly + 1 workspace + 2 fits + 1 summary**. `--stage cards` stops
after assembly. `--stage fit` reuses a verified saved model and submits only
the workspace, two fits and summary. A failed B-only fit does not invalidate
an independently successful S+B fit, or vice versa; resume reruns the failed
fit and its collector, keeping valid products.
`--masses 12,30,70` creates independent chains for all three hypotheses.
Submission reads nominal axes and audits the resolution inputs once; it does
not run the spectrum-bin preparation loop on the submitting machine.

`--bins-per-job` is the number of consecutive spectrum bins assigned to one
Condor preparation job, within one era and signal mass hypothesis. The default
1 gives one bin per job; 5 gives up to five bins processed sequentially inside
each job (the final job in an era may contain fewer). Increase it to amortise
ROOT startup and file-reading overhead. This changes job grouping only, never
the spectrum binning, number of CPUs per job or statistical model.
`--condor-maxjobs 0` is the default: the workflow does not pass a `-maxjobs` cap
to DAGMan. Available resources and cluster scheduling still determine how many
jobs run. A positive value explicitly opts into a submission cap. Jobs request
one CPU and `--condor-memory 16000` MB each.
`--condor-retries 2` retries failed nodes twice before their dependants stop.
Optional `--condor-requirements` is passed to the submit file.

The worker wrapper activates the submitting user's `CMSSW_BASE` and
`SCRAM_ARCH`. Override them with `--condor-cmssw-base` and
`--condor-scram-arch` if needed. The ROOT inputs, batch directory, CMSSW and
CVMFS must remain accessible at their absolute paths on worker nodes.

Submission follows the existing SKFlatAnalyzer/SKNanoAnalyzer convention:

```text
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
transfer_output_files = ""
```

Condor transfers the small executable wrapper. The wrapper changes to the
shared batch directory and uses the shared absolute paths for source, ROOT
inputs and results. Explicitly empty `transfer_output_files` prevents automatic
transfer of result files from the execute sandbox; normal stdout/stderr logs
are still handled by Condor. The result files are already written to shared
storage. This enables normal pool scheduling without imposing an equality
between the submit host's and worker's `FileSystemDomain` labels.

No `--condor-filesystem-domain` argument or environment variable is required.
The earlier `NPS26009_CONDOR_FILESYSTEM_DOMAIN` workaround is no longer read,
so an old export does not silently restrict new submissions. The explicit
option remains available only as an optional additional worker filter.

The `--stage set-domain` operation is retained for legacy no-transfer batches;
it is not part of the normal workflow. For a fresh submission after cancelling
old jobs, simply run the normal command from either submit host:

```bash
python3 full_spectrum_fit.py --masses 30 --target full
```

The reference submission settings were checked in
[SKFlatAnalyzer](https://github.com/joonblee/SKFlatAnalyzer/blob/master/python/SKFlat.py)
and [SKNanoAnalyzer](https://github.com/joonblee/SKNanoAnalyzer/blob/main/templates/job_dict.py).

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

### Recovering an old batch after the common-transfer validation fix

An existing batch continues to run its original source snapshot after a Git
update. Therefore, ordinary `--stage resume` does not apply a new validator.
For the exact-equality QCD metadata failure, use:

```bash
git pull --ff-only
python3 full_spectrum_fit.py --stage recover \
    --condor-dir full_spectrum_condor/M30_r0 \
    --condor-recovery-dir full_spectrum_condor/M30_r0_fixed
```

This creates a new source snapshot and batch, copies verified successful **raw
preparation results**, and submits only unfinished preparation jobs plus the
assembly and joint fit. For 607 successful and 448 failed preparation jobs,
this means reusing 607 results and scheduling 448 preparation jobs, followed
by assembly and fit. All original files are preserved. The new part/status records retain
their parent batch, completion ID and original part checksum.

Recovery checks that all yield, nuisance, geometry and assembly calculations
are unchanged; it permits the common-transfer validator and CLI/management
changes. A changed calculation or input file requires a fresh batch. The
original bin grouping and physics options are retained. The new batch uses
the current `--condor-maxjobs` setting (default 0, no workflow cap), rather than
inheriting an older batch's cap. The old DAG must have exited before recovery.

The recovery directory must be new. If omitted, a unique sibling directory
is chosen automatically. Add `--condor-dry-run` to prepare without submitting.
Monitor the new batch using its new directory:

```bash
python3 full_spectrum_fit.py --stage status \
    --condor-dir full_spectrum_condor/M30_r0_fixed
```

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
    --inject-r 1 \
    --output-base full_spectrum_condor/M30_r0/results \
    --condor-dir full_spectrum_condor/M30_r1
```

Use the same physics/input options as the original card. The new batch copies
the verified card and model and runs only the joint fit.

## Scope and verification

### Continuing from the completed M30 medium15 batch

Use the new `coarse` preset as the working binning. The previous M30 medium15
custom-edge card is compatible, so its input preparation need not be repeated.
From `NIsoMuon`, after updating the source and activating CMSSW/Combine:

```bash
# Check the same geometry at the low/high representative signal masses.
python3 full_spectrum_fit.py --masses 12,70 --target full \
    --fit-layout window --sideband-binning coarse \
    --condor-dir full_spectrum_outputs/condor/M12_M70_coarse_r0

# Structural signal-recovery check; reuse the completed M30 medium15 card.
python3 full_spectrum_fit.py --masses 30 --target full --stage fit \
    --fit-layout window --sideband-binning coarse --inject-r 1 \
    --output-base full_spectrum_outputs/condor/M30_binning_medium15/results \
    --condor-dir full_spectrum_outputs/condor/M30_coarse_r1
```

The second command creates a separate fit-only batch; the existing r=0 results
are preserved. With the same input options, `fit_s` should recover the injected
r=1 within numerical precision. `fit_b` fixes r=0 and can shift background
parameters to accommodate the injected signal. Internal r=1 is 25 signal
events summed over all eras and the entire 11--80 GeV domain, including
sidebands. This same-model Asimov check does not measure toy bias or validate
background modelling.

Complete the binning comparison using normalisation correlations and total SR
background uncertainties (including covariance), then study DY/QCD closure.
Use the closure results to settle the shape-uncertainty model before running
alternative-background and signal-injection toys. Choose injection sizes
relative to sensitivity at each mass rather than treating 25 events as a
universal benchmark. Expected limits and impacts follow those checks. The
current full-spectrum CLI's `fit` stage runs Asimov FitDiagnostics only.
The separate MC-only shape-closure tool below provides the next diagnostic;
alternative-background toys and final limits remain subsequent work.

### Audit saved fits without refitting

`full_spectrum_fit.py` now records three distinct POI error definitions:

| Field | Meaning |
| --- | --- |
| `r_error` | Legacy `RooRealVar.getError()` value; not a profile interval |
| `r_covariance_error` | Square root of the r diagonal of the covariance matrix |
| `r_profile_error_low/high` | Stored signed lower / positive upper asymmetric errors |
| `r_profile_crossing_low/high` | Best-fit r plus the corresponding stored error |

Missing profile crossings are null. A stored asymmetric error of zero is not
reported as a measured zero uncertainty. None of these fields is a CLs limit.
Old batch snapshots and their summaries remain immutable. To audit their ROOT
results with the updated error definitions, and compare the central/coarse SR
bands without rerunning Combine minimisation:

```bash
python3 full_spectrum_audit.py --results \
    full_spectrum_outputs/condor/M30_binning_central/results \
    full_spectrum_outputs/condor/M30_binning_medium15/results \
    --samples 5000 --seed 123456 \
    --output-base full_spectrum_outputs/audit_M30_binning
```

This runs locally inside the existing CMSSW/Combine environment. It reads the
saved workspace and fit covariance, and writes a new `fit_audit.json`. Each
era's SR total background and the combined SR sum are sampled jointly. The
combined error includes cross-era correlations. This is Gaussian nuisance
covariance propagation through the original nonlinear yield functions, not
new pseudo-data, a fit, or a model-bias test. Draws are untruncated and any
draws outside original nuisance ranges are counted. Results also include
local linear propagation at two finite-difference steps and four block RMS
estimates to diagnose sampling precision. Clipped zero-yield responses need
the nonlinear result; local derivatives alone can be misleading.

`--samples 0` (default) only reads saved bands and RooFitResults, without
deserialising the Combine workspace. It requires PyROOT but not the Combine
library. Neither mode rewrites saved summaries, cards, or ROOT files.

### MC-only shape transport closure

Use the exact saved coarse geometry at all three representative masses:

```bash
python3 background_shape_closure.py --model-roots \
    full_spectrum_outputs/condor/M12_M70_coarse_r0/results \
    full_spectrum_outputs/condor/M30_binning_medium15/results \
    --output-base full_spectrum_outputs/closure_coarse
```

No signal templates or fit rerun are needed. The tool reads only the nominal
NLO DY and QCD inclusive MC files, in read-only mode. It produces separate
JSON results and PDF/PNG plots for every era, process and mass geometry:

- DY: OS light-jet mass shape, matched to OS b-jet DY MC over 11--80 GeV.
- QCD: SS b-jet mass shape, matched to OS b-jet QCD MC over 11--80 GeV.

The normalisation is removed separately for each era/process; the remaining
residual tests shape transport relevant to the in-situ normalisation model.
The covariance includes the correlation induced by using the same MC bins
in the integral-matching factor. It assumes the source/target categories are
disjoint. Weighted Sumw2 statistics, including signed bins, are preserved.
Nonpositive source bins have null ratio entries; nonpositive full-domain
integrals are reported as inconclusive rather than clipped or silently passed.
The covariance-rank chi-square is a first-order diagnostic, not an automatic
low-statistics p-value or a prescription for additional nuisance parameters.

Only native-bin-aligned intervals are accepted. `--base-dir` can override the
MC location stored in the model, and `--no-plots` writes JSON alone. The tool
does not open collision data, update anchors/templates, assign new shape
uncertainties, or test the SS-data analytic fit/extrapolation. Those are
separate checks. Inspect `closure_summary.json`, ratios and finite-MC errors
before settling a DY/QCD shape-uncertainty prescription and running toys.

### Reviewer studies: joint QCD response, DY30 with free normalisation, bias toys

The following are explicit studies; all existing commands retain their original
physics model unless the new DY options are supplied. No blinded OS data are
read. Start in `NIsoMuon` with CMSSW/Combine activated.

First, profile the coherent QCD envelope, correlated SS-fit modes and QCD
normalisation together against the MC transport residual:

```bash
python3 qcd_joint_closure.py --model-roots \
    full_spectrum_outputs/condor/M12_M70_coarse_r0/results \
    full_spectrum_outputs/condor/M30_binning_medium15/results \
    --closure-json full_spectrum_outputs/closure_coarse/closure_summary.json \
    --output full_spectrum_outputs/qcd_joint_check.json
```

This local SciPy calculation profiles the actual clipped QCD response in a
Gaussian approximation to an expected fit. Its covariance includes finite-MC
transport errors and nominal total-count Poisson variances. Other background
responses are fixed. It reports both the current lnN(1.5) normalisation and
an unconstrained-scale cross-check. The output is a screening diagnostic, not
an exact Poisson Combine fit, a coverage p-value or a background-model approval.
Undefined MC ratios are listed as excluded; sparse expected-count bins are
identified. Wider MC-only comparisons use 11--20, 20--30, 30--40, 40--60 and
60--80 GeV where the stored intervals can be summed without splitting bins.

Prepare a separate DY review model directly from the original saved cards:

```bash
python3 prepare_full_spectrum_review.py --model-roots \
    full_spectrum_outputs/condor/M12_M70_coarse_r0/results \
    full_spectrum_outputs/condor/M30_binning_medium15/results \
    --output-base full_spectrum_outputs/review_dy30
```

The source `datacard.txt` must be present and match its saved model hash. This
preserves every unrelated source nuisance line, including its existing naming
and correlations. The preparation changes only:

- DY normalisation: replace the broad lnN by one free scale per era, with
  numerical range [0,10]; proximity to the bounds is reported.
- DY shape: independent `0.70/1.30` lnN responses for every era/fit bin in the
  entire 11--80 GeV domain. This implements a 30% floor without choosing an
  arbitrary high-mass threshold. It is an explicit study prescription.

The nominal yields, QCD modes/envelope and all existing statistical inputs
are unchanged. DY_NFModel/NFStat remain excluded as in the existing in-situ
model. Zero nominal DY retains its original additive source-statistical term;
a relative closure shift does not create a nonzero central yield. Free-scale
summary entries have `constraint: null`, because they have no Gaussian prior.

Then submit **fit-only** Condor jobs (no raw input preparation):

```bash
python3 full_spectrum_fit.py --masses 12,30,70 --target full --stage fit \
    --fit-layout window --sideband-binning coarse \
    --dy-shape-uncertainty 0.30 --dy-normalisation free \
    --output-base full_spectrum_outputs/review_dy30 \
    --condor-dir full_spectrum_outputs/condor/M12_M30_M70_dy30_free
```

Model paths include `_DYshape0p3_DYfree10`; model signatures protect the chosen
prescription. The fit batch snapshots the prepared cards. Check both fit
qualities and free-DY bound proximity before using it as a toy fit model.

For a small runtime/convergence pilot **after those fits pass**, submit:

```bash
python3 full_spectrum_toys.py --results \
    full_spectrum_outputs/condor/M12_M30_M70_dy30_free/results \
    --toys 10 --toys-per-job 5 --multipliers 0,1,2 \
    --batch full_spectrum_outputs/condor/review_toy_pilot
```

This creates 18 independent Condor chunks and 90 toy fits for three masses and
three injection levels. Injections are 0,1,2 times each new model's positive
profile error (a sensitivity reference, not r95), so the signal event yields
are mass dependent. A negative fit-r range is chosen from that error and the
nominal Poisson-mean positivity bound. Generation uses nonnegative r and
fixed nominal truth nuisances (`--toysNoSystematics`); fitting profiles all
nuisances. Sources and fit/truth workspaces are snapshotted, checksummed and
transferred. `--dry-run` prepares submission files without queueing jobs.

Default toy fits use `--noErrors --minos none` to measure minimum-only empirical
bias efficiently. They do not test per-toy covariance quality, pulls or interval
coverage. `--toy-errors` also runs HESSE, but the bias summary deliberately uses
the empirical distribution rather than RooRealVar.getError(). After the pilot,
increase the toy count in a new batch only if fit quality and runtime permit.
For later `0,r95,2*r95` studies, `--injection-r-json` accepts a JSON mapping
from mass labels to explicit injection lists, overriding `--multipliers`.
Supply the blinded expected-r95 values of the new model, not old counting-only
limits. The profile error still determines the default negative fit-r range.
Collect with:

```bash
python3 full_spectrum_toys.py --stage collect \
    --batch full_spectrum_outputs/condor/review_toy_pilot
```

`toy_summary.json` preserves failed, missing and boundary toy counts. It reports
the mean fitted r, its empirical spread and the Monte Carlo error on the mean
bias. Any failed/missing/boundary fits mark the ensemble incomplete. No automatic
pass/fail bias threshold is imposed.

For alternative-background toys, `--truth-json` is supported for a single saved
model. The JSON must contain its `mass`, exact `fit_card_sha256`, and a `factors`
mapping from channel names to nonnegative background process factors. Example
structure (replace the hash and use a reviewed alternative shape):

```json
{"mass":30,"fit_card_sha256":"<exact saved fit-card hash>",
 "factors":{"bin_2018_SR":{"QCD":1.2,"DY":0.9}}}
```

Only the generator workspace gets these extra fixed-truth rate factors; toys
are fitted with the nominal study workspace. Unspecified factors remain one.
This interface does not turn sparse or negative weighted-MC closure ratios into
a physical truth shape automatically. Select and validate the alternative
before interpreting its toy result as a modelling-bias test.

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
python3 -m unittest discover -s tests -p 'test_qcd_common_transfer.py' -v
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

The Run-3 common-transfer regression fixture contains the MC records and
stored diagnostics reported on 2026-11-11. It checks last-bit tolerance,
unchanged saved values, and rejection of changed input identities, windows,
discrete decisions and meaningful numerical differences. Recovery tests check
reuse under the corrected validator, preservation of the original batch and
rejection of changed calculation code or inputs.



## Integral-preserving linear DY closure shape

After updating **both** repositories, regenerate the eight individual-era NF
estimates with SKPlotMaker (`dy_bkg_estimation.py --era ERA --blind`). NF output
now includes `DYAux/ClosureLinearMetadata` and signed pre-NF source weights
`ClosureLinearSourceUp/Down`. The nominal source, NF and source Sumw2 are
unchanged. Old DY files are rejected when the new shape is requested.

```bash
python3 full_spectrum_fit.py --masses 12,30,70 --target full \
    --fit-layout window --sideband-binning coarse \
    --norm-prior 1.5 --dy-shape-model linear \
    --condor-dir full_spectrum_outputs/condor/M12_M30_M70_coarse_DYlinear_r0
```

Submit a **fresh batch**: existing snapshots/cards/parts do not contain the new
shape inputs. The existing Condor DAG still prepares era/mass bins in parallel,
assembles a joint card, then runs B-only and S+B fits separately in parallel.
The new protocol module is included in worker snapshots and source hashes.
`--dy-shape-model none` (default) preserves existing model signatures and paths.
`linear` adds `_DYlinearShape_v1`; it cannot be combined with the independent
`--dy-shape-uncertainty 0.30` study. No free normalisation is enabled implicitly.

The adopted modelling variation uses the **central** Run2+3 fit
`R(m)=0.00583*m/GeV+0.9038`, measured over 11--70 GeV and extrapolated to 80 GeV.
Its fit statistical errors are not separate shape parameters. One nuisance,
`CMS_NPS26009_DYClosureLinearShape`, is shared across all eras and mass bins.
The existing independent per-era DY lnN 1.5 normalisations remain; NFStat/NFModel
remain excluded from these in-situ cards. Source statistics are retained once.
This is a correlated tilt, not a guarantee of independent >=30% freedom in
every high-mass bin.

Signed weighted sources are integrated over the **actual final fit bins**.
The existing nominal physical-boundary prescription is retained. Nominal-zero
bins stay zero in closure variations and retain their additive source error.
Up is normalised to each era's full 11--80 nominal DY integral; Down reflects
Up about nominal. Any negative variation bins are bounded at zero before a
final renormalisation and recorded explicitly in `model.json` (clipped_up and
clipped_down). No source bins are clipped in the producer. A complete disjoint
11--80 domain is required; do not normalise individual SR/sideband channels.

For arbitrary nuisance values, nonnegative piecewise-linear bin responses are
multiplied by a common per-era denominator. A single `DYClosureNorm_ERA`
rateParam denominator is shared through identity formula aliases by positive-DY
channels, avoiding quadratic card size for fine spectrum binning.
This preserves the full nominal DY yield for all shape values with other
nuisances at nominal, including beyond +/-1 sigma. The independent DY scale
still changes the total. Other statistical/detector nuisances retain their
existing responses.

After text2workspace, `fit_r0/dy_shape_audit.json` checks the **compiled** per-era
DY sums at theta=-3,-1,-0.5,0,0.5,1,3 and aborts before fitting if conservation
fails. `fit_summary.json` records the shape parameter, error and its correlation
with each DY normalisation. A nonzero post-fit correlation is allowed. Covariance
propagation and pre/postfit yields use the workspace expectations automatically.
