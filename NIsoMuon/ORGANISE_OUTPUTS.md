# Organise the NIsoMuon production outputs

Run `organise_nisomuon.py` from the production NIsoMuon directory after all fits
finish. Python 3.9 or later and Git/SSH access to GitHub are sufficient; the
Combine/CMSSW environment is not needed. The helper does not execute Combine,
edit the AN or datacards, or commit/push any result.

```bash
python3 organise_nisomuon.py                  # read-only production audit and plan
python3 organise_nisomuon.py --apply          # consolidate, keep old trees outside NIsoMuon
```

To consolidate and delete the verified superseded trees in a single execution,
use this instead of the preceding `--apply` command:

```bash
python3 organise_nisomuon.py --apply --delete-obsolete
```

By default it clones `joonblee/combine_review` master over SSH into a temporary
directory. An existing **current** checkout can be supplied with
`--review-repo /absolute/path/to/combine_review`.

## Final paths

| Contents | NIsoMuon-relative path |
| --- | --- |
| Datacards and conversion inputs | `datacards_NIsoMuon/` (unchanged) |
| Existing limits, including any unblind outputs | `limit_outputs/` (preserved) |
| All 39 final B-only impact sets, matching workspaces, fits and repair directories | `limit_outputs/alpha/blind/{Run2,Run3,Run2Run3}/` |
| Impact selection record | `limit_outputs/final_impacts_manifest.json` |
| Updated M20/M70 S+B impacts, scans, correlations and HybridNew outputs | `review_outputs/Run2Run3/{M20,M70}/` |
| Final verified bias summary | `review_outputs/Run2Run3/M{mass}/bias/bias_summary_Run2Run3_M{mass}.json` |
| Final merged 500-toy ensembles and validation | `review_outputs/Run2Run3/M{mass}/bias/verified/` |
| Original pre-repair bias summary used as an input | `review_outputs/Run2Run3/M{mass}/bias/original/` |
| M20 reused fallback/poiseed/previous-profile fits; original M70 audit | `review_outputs/provenance/` |
| FitDiagnostics and zero-DY diagnostic runs | `review_outputs/diagnostics/` |
| Runs of unknown role, e.g. `bias_M20_verified.hul4v2hz` | `review_outputs/additional_runs/` |
| Exact JSONs before relocating references | `review_outputs/provenance/original_json/` |
| Old-to-new path lookup | `review_outputs/relocation_map.json` |
| Installed file hashes | `review_outputs/organisation_manifest.json` |

Source code, `tests/`, `python/`, `template/`, `sigfit_inputs/` and other input
directories are retained. The helper removes disposable Python caches from the
top level, `python/` and `tests/`. Unknown top-level files/directories are left
alone. Differing files from the previous `review_outputs` are retained under
`provenance/previous_review_outputs`; logs in `review_logs` are retained under
`logs/previous_driver_logs`.

## Deletion boundaries

The reviewed snapshot identifies exactly one source per target/mass. The helper
requires all 39 PDF/physical JSON/internal JSON sets to match the GitHub snapshot
and the production datacards to match its recorded hashes. It also requires the
matching workspace and original initial-fit ROOT file. It verifies the final
M20/M70 validation records, the three distinct 500-toy ensembles per mass, and
the original workspace/toy input hashes. These are file/structural checks using
existing numerical validation; they are not new minimisations or coverage tests.

The explicitly superseded probe/failed first retry directories
`bias_M20_probe.BmNJspFs`, `bias_M20_fallback.0xJKrZIc`,
`bias_repair_M20.WBgJ4pg8`, `review_outputs_old` and `review_logs_old` are retired.
If a retained JSON still references an existing file inside a disposable tree
without a preserved destination, the helper stops before modifying production.

Selected impact trees are consolidated. Extra raw M12 run directories retain
their unique logs/metadata under `additional_runs`; byte-identical ROOT/PDF
copies are omitted and mapped to their final files. Unrecognised extra runs
are retained rather than inferred to be obsolete.

Live JSON path strings and path keys are relocated, and `input_sha256` entries
are refreshed where a relocated input JSON changes. Exact original JSONs and
their original hashes remain in `provenance/original_json`. Command logs remain
historical records with their original command lines. The script does not
rewrite historical GitHub manifests or old shell snippets: use the new paths
in the table, the local final-impact manifest, and the relocation map.

## Installation and recovery

`--apply` copies the final trees into a sibling staging directory and checks
the copied bytes before renaming anything. It checks available disk space and
rejects symlinks/special files in trees it would retire. It rejects detected
active Combine/workflow processes using this production tree. Both final
directories are installed, then audited using a complete SHA-256 inventory.
Ordinary errors and Ctrl-C during installation restore the original trees.

Without `--delete-obsolete`, the old trees are retained in the reported sibling
`NIsoMuon-before-organise-*` directory, with a `rollback.json` recording the
renames. With the flag, that directory is removed **only after** the complete
installed-file audit passes. No rollback copy remains after successful deletion.
A machine/power failure during directory renames may require manual recovery
from the staging/backup directories; multi-directory replacement is not atomic.

Rerunning after successful organisation verifies the saved hashes and exits.
It does not delete newly appeared files. If results have subsequently changed,
it stops rather than treating the previous inventory as current.

The cleaned production tree contains ROOT files and full validation evidence;
it is not the compact GitLab submission package. Keep using the separate
`combine_review/scripts/export_gitlab_submission.py` for that package. Do not
run `git add -A` to publish this entire production tree.
