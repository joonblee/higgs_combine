#!/usr/bin/env bash
# Blind Run2Run3 review of the current production cards, M20 and M70.
set -e -o pipefail

production_dir=${REVIEW_PRODUCTION_DIR:-/data6/Users/joonblee/higgs_combine/CMSSW_14_1_0_pre4/src/NIsoMuon}
cd "$production_dir/.."
if [ ! -f setup_validation_v16.sh ]; then
  echo "[ERROR] Missing $PWD/setup_validation_v16.sh" >&2
  exit 1
fi
# The setup saves the current src directory and returns to its NIsoMuon child.
source ./setup_validation_v16.sh
cd "$production_dir"
if [ "${CMSSW_BASE##*/}" != CMSSW_16_0_0 ]; then
  echo "[ERROR] setup_validation_v16.sh did not initialise CMSSW_16_0_0." >&2
  exit 1
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
command -v combine
command -v text2workspace.py
command -v combineTool.py
combine --version
python3 -c 'import ROOT; print("[ROOT]", ROOT.gROOT.GetVersion())'

# A unique directory prevents reuse of workspaces or toys from the old model.
review_run_dir=$(mktemp -d "$PWD/review_outputs_updated.XXXXXXXX")
log_dir="$review_run_dir/logs"
mkdir -p "$log_dir"
echo "[OUTPUT] $review_run_dir"
echo "[LOGS] $log_dir"

common=(--target Run2Run3 --output-base "$review_run_dir"
        --card-base datacards_NIsoMuon/alpha/blind/combined
        --limit-output-base limit_outputs/alpha/blind)

python3 - "$review_run_dir" <<'PY'
import hashlib, json, os, pathlib, subprocess, sys
run_dir = pathlib.Path(sys.argv[1])
cards = {}
for mass in (20, 70):
    card = pathlib.Path(f"datacards_NIsoMuon/alpha/blind/combined/Run2Run3/datacard_M-{mass}_Run2Run3.txt")
    cards[str(mass)] = {"path": str(card.resolve()), "sha256": hashlib.sha256(card.read_bytes()).hexdigest()}
limits = pathlib.Path("limit_outputs/alpha/blind/Run2Run3/limits_Run2Run3_alpha_internal.json")
payload = json.loads(limits.read_text())
for mass in (20, 70):
    median = payload[str(float(mass))]["exp0"]
    if not median > 0:
        raise SystemExit(f"Invalid expected median for M{mass}: {median}")
manifest = {"mode": "blind", "target": "Run2Run3", "cards": cards,
            "limit_json": str(limits.resolve()), "limit_sha256": hashlib.sha256(limits.read_bytes()).hexdigest(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "cmssw_base": os.environ.get("CMSSW_BASE"), "scram_arch": os.environ.get("SCRAM_ARCH")}
(run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
PY

wait_pair() {
  local status20=0 status70=0
  wait "$pid20" || status20=$?
  wait "$pid70" || status70=$?
  if [ "$status20" -ne 0 ] || [ "$status70" -ne 0 ]; then
    echo "[ERROR] $1: M20 exit=$status20; M70 exit=$status70. See $log_dir" >&2
    exit 1
  fi
  echo "[OK] $1"
}

python3 -u combine_review.py 20 "${common[@]}" --task validate >"$log_dir/validate_M20.log" 2>&1 &
pid20=$!
python3 -u combine_review.py 70 "${common[@]}" --task validate >"$log_dir/validate_M70.log" 2>&1 &
pid70=$!
wait_pair validation

# Diagnostic signed-r scans: ranges are clipped by the helper's yield safety.
# M20 needs a wider negative range; M70 needs a finer grid near r=0.
python3 -u combine_review.py 20 "${common[@]}" --task scan,correlation \
  --scan-points 600 --scan-parallel 20 --r-min -50 --r-max 100 \
  >"$log_dir/scan_corr_M20.log" 2>&1 &
pid20=$!
python3 -u combine_review.py 70 "${common[@]}" --task scan,correlation \
  --scan-points 240 --scan-parallel 20 --r-min -2 --r-max 5 \
  >"$log_dir/scan_corr_M70.log" 2>&1 &
pid70=$!
wait_pair scan/correlation

# Independent HybridNew grid jobs, capped at 40 (29 grid points by default).
python3 -u combine_review.py 70 "${common[@]}" --task hybrid \
  --hybrid-toys 500 --hybrid-parallel 40 --r-max 100 \
  >"$log_dir/hybrid_M70.log" 2>&1
echo "[OK] HybridNew M70"

# Inject the current expected median r95, read from the production limit JSON.
python3 -u combine_review.py 20 "${common[@]}" --task sb-impacts \
  --impact-parallel 20 --r-max 100 >"$log_dir/sb_impacts_M20.log" 2>&1 &
pid20=$!
python3 -u combine_review.py 70 "${common[@]}" --task sb-impacts \
  --impact-parallel 20 --r-max 100 >"$log_dir/sb_impacts_M70.log" 2>&1 &
pid70=$!
wait_pair S+B-impacts

python3 -u combine_review.py 20 "${common[@]}" --task bias \
  --bias-toys 500 --bias-multipliers 0,1,2 --r-min -50 --r-max 100 \
  >"$log_dir/bias_M20.log" 2>&1 &
pid20=$!
python3 -u combine_review.py 70 "${common[@]}" --task bias \
  --bias-toys 500 --bias-multipliers 0,1,2 --r-min -2 --r-max 100 \
  >"$log_dir/bias_M70.log" 2>&1 &
pid70=$!
wait_pair bias

echo "[DONE] Blind review commands completed. Output: $review_run_dir"
