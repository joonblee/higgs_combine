# Blinded QCD/DY sideband study

`limit_workflow_sideband_fit.py` is a separate, opt-in study. The ordinary
`limit_workflow.py` and its existing cards/results retain their behaviour.
The only shared-code change is an optional explicit-window argument to the
existing template/systematic integration function.

## Statistical construction

For each mass and era, retain one signal-window count (default ±5 sigma), and
add non-overlapping mass bins in the sidebands. Sigma comes from the current
era-dependent `sigfit_inputs` resolution coefficients. All integrals use common
native histogram edges: SR/exclusion boundaries snap outwards, domain boundaries
inwards. Every histogram used for variations must respect those edges; otherwise
production fails rather than double-counting a partially overlapping native bin.
The nominal sideband bin width is 2 GeV; it is a study choice, not a CMS prescription.

Each era has independent, freely fitted nonnegative scales beta_QCD and beta_DY,
shared between its SR and every sideband bin. All eras of the requested target
are fitted simultaneously. The likelihood is a product of Poisson probabilities
and the nuisance constraints, with expected bin counts

    lambda_i = r s_i(theta) + beta_QCD q_i(theta)
               + beta_DY d_i(theta) + other backgrounds_i(theta).

Signal contributions in the sidebands are retained from the actual signal
histograms. The recorded S_SB/S_SR ratio is a leakage diagnostic, not an assumed
Gaussian tail fraction. The normalisation of r matches the original alpha
convention: r=1 gives 25 signal events in the combined SR, not in SR+SB.

- The observation in every channel is its nominal background-only expectation.
  The old single-window Asimov dataset has no sideband information; this workflow
  constructs the additional Asimov counts from the same background ROOT inputs.
  OS `data.root` is blocked by the reader. Existing SS/light-jet control-data
  estimates are still inputs. There is deliberately no unblinding option.
- **Background prediction:** `FitDiagnostics` masks all SR channels and runs the
  B-only fit with r=0 (`--skipSBFit`). The resulting functions predict QCD, DY and
  total background inside the unused SR as well as the fitted sidebands.
- **Expected limit:** the original SR+SB card is fitted jointly with r>=0. It
  includes all regions and signal tails. The sideband-fit central values/errors
  are NOT fed back as extra priors: that would use the same information twice.
- A mass scan comprises separate hypothesis-dependent fits. Its overlapping SR
  counts must not be interpreted as independent mass-spectrum bins or merged
  into one likelihood.

## Uncertainties and correlations

MC experimental/theory responses and cross-era names come from the existing
workflow. Finite-template statistics and DY light-jet source statistics are
independent between disjoint regions. Zero-source DY retains its additive
absolute source uncertainty, multiplied by the free DY scale.

Pure QCD transfer/normalisation and DY NFStat/NFModel constraints are replaced by
the free normalisations, rather than silently constraining those same overall
scales again. The workflow verifies that the QCD transfer derivative/nominal
ratios and DY NF are constant across the included regions. It refuses to absorb
a mass-dependent transport into a single scale. This is why the low-mass
calibration region is NOT appended as an extra sideband.

The SS-fit statistical covariance is propagated using the producer's full
parameter covariance and integrated derivative histograms. The same Gaussian
modes act on every region in an era, preserving shape correlations and
anticorrelations. Do not copy the old single-window QCD statistical error as an
independent nuisance into each new sideband bin.

QCD functional-form uncertainty retains the current symmetric absolute-envelope
prescription, with one coherent Gaussian nuisance per era and a nonnegative
final yield. Its cross-bin correlation is an explicit study assumption; this is
not discrete profiling of the alternative fitted functions. Alternative shapes
and pseudo-data bias tests are needed before treating a sideband constraint as
validated for the physics result. Similarly, inherited theory prescriptions
remain those of the existing workflow; this addition does not rederive them.

Postfit intervals are 16/84 percentiles of joint Gaussian fit-covariance draws,
conditioned on allowed parameter ranges. All yields are summed on each draw
before computing combined intervals. They are diagnostic propagation bands,
not profile-likelihood confidence intervals. Status, covariance quality, EDM,
normalisation correlations, proximity to bounds and rejected-draw fraction are
exported. A failed fit produces no new result JSON. Boundary-sensitive bands
are flagged on the plots; validate them with profile scans/toys, and check the
arbitrary beta upper bound (default 10) by increasing `--norm-max`.

## Sideband choice and the mass boundaries

Default: use the validated **11--80 GeV** range outside **±5 sigma**. Choosing
`--exclude-nsigma 7` or `10` keeps the SR at ±5 sigma and leaves the intervening
gaps unused. A wider exclusion removes nearby background information; it is not
automatically a more conservative or less biased result. Compare leakage,
normalisation correlations, SR uncertainty and expected limits before choosing.
`--outer-nsigma 15` restricts the fit to a local ±15 sigma window instead of the
full validated range. Also vary `--bin-width` (e.g. 1, 2, 4 GeV); sparse bins and
unstable signed-template responses can require coarser binning.

At 12 or 70 GeV, clip to 11 and 80 GeV. Keep both sidebands if present; otherwise
use the available one, labelled explicitly in `model.json`. Extrapolation into
the SR uses the existing background shape model and its correlated uncertainty,
not a polynomial chosen solely to extend the available sideband. At least two
sideband bins and nominal QCD/DY shape identifiability are required. A nominal
Fisher diagnostic is optimistic because it fixes other nuisances; inspect the
actual fitted beta covariance as well. No artificial prior is silently added
if the two normalisations cannot be separated.

Do not extend into <9 GeV to manufacture a lower sideband: composition and
transport there differ. Adding that region would require an explicit separate
control-region model and low-to-high transfer uncertainty. The current nominal
QCD estimate still uses its established low-mass calibration/MC transport; the
new fit does not add or reuse those counts as OS sideband observations.

Current committed resolution coefficients give these **unsnapped exclusion
boundaries**, showing why edge handling matters (ranges span all eight eras):

| Mass | Exclusion | Lower boundary [GeV] | Upper boundary [GeV] |
|---|---|---|---|
| 12 | 5 sigma | 11.29--11.36 | 12.64--12.71 |
| 12 | 7 sigma | 11.01--11.10 | 12.90--12.99 |
| 12 | 10 sigma | 10.59--10.72 | 13.28--13.41 |
| 70 | 5 sigma | 61.62--62.98 | 77.02--78.38 |
| 70 | 7 sigma | 58.27--60.17 | 79.83--81.73 |
| 70 | 10 sigma | 53.24--55.96 | 84.04--86.76 |

## Run in the configured CMSSW/Combine environment

From `higgs_combine/NIsoMuon` (with the usual `cmsenv`):

```bash
# Optional installation smoke test: runs a small synthetic fit/limit when Combine is available.
python3 -m unittest discover -s tests -p test_sideband_fit.py -v

# Start with representative masses. --stage all builds, fits, extracts, and calculates limits.
python3 limit_workflow_sideband_fit.py --masses 12,30,70 --target full

# All 13 simulated mass points:
python3 limit_workflow_sideband_fit.py --target full

# Explicit stages (same settings must accompany each invocation):
python3 limit_workflow_sideband_fit.py --masses 30 --stage cards
python3 limit_workflow_sideband_fit.py --masses 30 --stage fit
python3 limit_workflow_sideband_fit.py --masses 30 --stage limits

# Change the exclusion; outputs are kept separately.
python3 limit_workflow_sideband_fit.py --masses 12,30,70 --exclude-nsigma 7
python3 limit_workflow_sideband_fit.py --masses 12,30,70 --exclude-nsigma 10

# Compare a local fit range.
python3 limit_workflow_sideband_fit.py --masses 12,30,70 --outer-nsigma 15
```

`--base-dir`, `--sigfit-dir`, `--resolution-map`, `--trigger`, and `--region`
override the corresponding standard inputs. This study uses the simulated mass
points, not interpolated signals. Keep region selection consistent with the
producer's data-driven background contract. Negative/unstable template inputs
are governed by the existing input checks and must not be hidden by the plotter.

Outputs are separate from the existing datacards and limits:

```text
sideband_fit_outputs/Run2Run3_exclude5sigma_full_bin2_era/
  M30/model.json                    # geometry, rates, leakage, assumptions, audits
  M30/datacard.txt                  # joint SR+SB likelihood
  M30/commands.json                 # exact masked-fit commands
  M30/fit_step0.log, fit_step1.log
  M30/workspace.root, fitDiagnostics.Sideband.root
  M30/expected_limits.json          # five expected quantiles in r and alpha
  sideband_fit_Run2Run3_M30.json     # fitted backgrounds, correlated intervals
```

The output path encodes exclusion, outer window, bin width and normalisation
scope. Changing another modelling input replaces that configuration's files;
use a distinct `--output-base` to retain such comparisons. Saved models/cards
are hashed and configuration-checked before reuse. A new card build invalidates
old fitted results and expected-limit JSON for that mass.

In SKPlotMaker:

```bash
python3 plotter_sideband_fit.py \
  /path/to/higgs_combine/CMSSW_14_1_0_pre4/src/NIsoMuon/sideband_fit_outputs/Run2Run3_exclude5sigma_full_bin2_era/sideband_fit_Run2Run3_M*.json
```

The new plotter reads exported JSON only. It produces per-era mass distributions
with nominal and sideband-postfit backgrounds, uncertainty bands, Asimov points,
and a ratio panel. Unequal sideband/SR bins are drawn as **Events / GeV**.
The combined Run2+Run3 output is a **signal-window yield versus mass-hypothesis**
plot, using the covariance-aware summed SR intervals. It does not invent a
common fine-binned postfit mass spectrum from incompatible era windows. The
original `plotter.py` is untouched. Existing colour/order conventions are reused.

## Interpretation and required follow-up

Same-model background Asimov data should return beta approximately one. The
study quantifies expected constraints, degeneracies, and sensitivity; nominal
Asimov closure is not evidence that QCD/DY or ttbar modelling is accurate. Before
adopting the model, check alternative-background pseudo-data, signal injection,
local/full-range stability, binning and exclusion variations, bound stability,
and the behaviour of the sparse data-driven templates. A ttbar shape mismatch
can otherwise be absorbed partly by DY/QCD; a nominal Asimov fit cannot detect it.

Tests cover boundary clipping, non-overlap, covariance preservation, nominal
card closure, zero-source DY, nuisance correlations and target-data blocking.
The optional CMSSW integration test runs text2workspace, a masked FitDiagnostics
fit and an expected joint limit. The development host has no ROOT/Combine or
analysis ROOT inputs: that integration test and the physics fits must run in
the configured analysis environment. No claim of validated fit performance is
made from the pure-Python tests.

## Primary references

- [CMS EXO-21-005, section 6](https://arxiv.org/html/2309.16003v2): simultaneous
  S+B fits use a **total fit window** of ±5 resolution units, or ±8 for the high-pT
  category with fewer events. These numbers are NOT a universal rule that the
  excluded central interval or the sideband inner edge must be 5 or 8 sigma.
  The paper also tests background-function bias with pseudo-experiments.
- [Combine channel masking](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/v10.4.1/part3/nonstandard/#channel-masking):
  exclude SR information while retaining its prediction and nuisance response.
