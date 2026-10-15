#!/usr/bin/env bash
set -euo pipefail

source /cvmfs/cms.cern.ch/cmsset_default.sh
cd /data6/Users/joonblee/higgs_combine/CMSSW_14_1_0_pre4/src
eval "$(scram runtime -sh)"
cd NIsoMuon
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

pids=()
for task in 'Run2 40' 'Run3 50' 'Run2Run3 50'; do
  read -r target mass <<< "$task"
  echo "[START] Initial POI repair: $target M$mass"
  python3 -u limit_workflow.py \
    --stage run --task impacts --target "$target" \
    --parameter alpha --mode blind \
    --masses "$mass" --impact-masses "$mass" \
    --repair-impacts --impact-parallel 1 \
    --sigfit-dir ./sigfit_inputs \
    --qcd-method data-driven --dy-method data-driven \
    --allow-negative-r --r-min -5 --r-max 10 \
    --output-base limit_outputs_impacts_updated \
    --strict > "poi_repair_${target}_M${mass}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

grep -H -E '\[IMPACT POI CHECK\]|\[IMPACT CHECK\]|\[ERROR\]' \
  poi_repair_Run2_M40.log poi_repair_Run3_M50.log poi_repair_Run2Run3_M50.log || true
if [ "$failed" -ne 0 ]; then
  echo '[ERROR] A POI repair failed; inspect the three logs above.'
  exit 1
fi
echo '[DONE] Three initial POI intervals repaired; existing nuisance scans retained.'
