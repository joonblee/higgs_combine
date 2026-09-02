# ============================================================
# NPS-26-009 Combine review diagnostics
# M20 + M70
#
# Assumption:
#   /data6/Users/joonblee/higgs_combine/NIsoMuon/combine_review.py
#   already exists.
#
# nproc = 48
# Use at most 40 logical CPUs, leaving some headroom.
# ============================================================

cd /data6/Users/joonblee/higgs_combine/CMSSW_14_1_0_pre4/src/NIsoMuon

CORES=40
HALF_CORES=20

mkdir -p review_logs


echo "============================================================"
echo "[0] Check environment"
echo "============================================================"

which combine
combine --version
which text2workspace.py
which combineTool.py

if which ValidateDatacards.py >/dev/null 2>&1; then
    which ValidateDatacards.py
else
    echo "ValidateDatacards.py will be taken from CombineHarvester source tree."
    ls "$CMSSW_BASE/src/CombineHarvester/CombineTools/scripts/ValidateDatacards.py"
fi

echo


echo "============================================================"
echo "[1] ValidateDatacards: M20 and M70"
echo "============================================================"

python3 combine_review.py 20 \
    --task validate \
    > review_logs/validate_M20.log 2>&1 &
PID1=$!

python3 combine_review.py 70 \
    --task validate \
    > review_logs/validate_M70.log 2>&1 &
PID2=$!

wait $PID1
STATUS1=$?

wait $PID2
STATUS2=$?

if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ]; then
    echo "[ERROR] ValidateDatacards failed."
    echo "Check:"
    echo "  review_logs/validate_M20.log"
    echo "  review_logs/validate_M70.log"
    return 1
fi

echo "[OK] ValidateDatacards M20/M70"
echo


echo "============================================================"
echo "[2] B-only likelihood scans + correlation matrices"
echo "    20 scan processes for M20"
echo "    20 scan processes for M70"
echo "============================================================"

python3 combine_review.py 20 \
    --task scan,correlation \
    --scan-points 120 \
    --scan-parallel ${HALF_CORES} \
    --r-min -2 \
    --r-max 100 \
    > review_logs/scan_corr_M20.log 2>&1 &
PID1=$!

python3 combine_review.py 70 \
    --task scan,correlation \
    --scan-points 120 \
    --scan-parallel ${HALF_CORES} \
    --r-min -2 \
    --r-max 100 \
    > review_logs/scan_corr_M70.log 2>&1 &
PID2=$!

wait $PID1
STATUS1=$?

wait $PID2
STATUS2=$?

if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ]; then
    echo "[ERROR] Likelihood scan/correlation failed."
    echo "Check:"
    echo "  review_logs/scan_corr_M20.log"
    echo "  review_logs/scan_corr_M70.log"
    return 1
fi

echo "[OK] Likelihood scans + correlations"
echo

echo "============================================================"
echo "[3] HybridNew vs existing AsymptoticLimits: M70"
echo "    HybridNew uses 40 forked processes"
echo "============================================================"

python3 combine_review.py 70 \
    --task hybrid \
    --hybrid-toys 500 \
    --hybrid-parallel 29 \
    --r-max 100 \
    --force \
    > review_logs/hybrid_M70.log 2>&1

if [ $? -ne 0 ]; then
    echo "[ERROR] HybridNew failed."
    echo "Check review_logs/hybrid_M70.log"
    return 1
fi

echo "[OK] HybridNew M70"
echo


echo "============================================================"
echo "[4] S+B Asimov impacts"
echo "    default injected r = existing median expected r95"
echo "    M20: 20 parallel nuisance fits"
echo "    M70: 20 parallel nuisance fits"
echo "============================================================"

python3 combine_review.py 20 \
    --task sb-impacts \
    --impact-parallel ${HALF_CORES} \
    --r-max 100 \
    > review_logs/sb_impacts_M20.log 2>&1 &
PID1=$!

python3 combine_review.py 70 \
    --task sb-impacts \
    --impact-parallel ${HALF_CORES} \
    --r-max 100 \
    > review_logs/sb_impacts_M70.log 2>&1 &
PID2=$!

wait $PID1
STATUS1=$?

wait $PID2
STATUS2=$?

if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ]; then
    echo "[ERROR] S+B impacts failed."
    echo "Check:"
    echo "  review_logs/sb_impacts_M20.log"
    echo "  review_logs/sb_impacts_M70.log"
    return 1
fi

echo "[OK] S+B impacts M20/M70"
echo

echo "============================================================"
echo "[5] Bias / signal-recovery tests"
echo "    injections = 0, 1, 2 x existing expected r95"
echo "    500 toys per injection"
echo "    M20 and M70 run simultaneously"
echo "============================================================"

python3 combine_review.py 20 \
    --task bias \
    --bias-toys 500 \
    --bias-multipliers 0,1,2 \
    --r-min -2 \
    --r-max 100 \
    > review_logs/bias_M20.log 2>&1 &
PID1=$!

python3 combine_review.py 70 \
    --task bias \
    --bias-toys 500 \
    --bias-multipliers 0,1,2 \
    --r-min -2 \
    --r-max 100 \
    > review_logs/bias_M70.log 2>&1 &
PID2=$!

wait $PID1
STATUS1=$?

wait $PID2
STATUS2=$?

if [ $STATUS1 -ne 0 ] || [ $STATUS2 -ne 0 ]; then
    echo "[ERROR] Bias tests failed."
    echo "Check:"
    echo "  review_logs/bias_M20.log"
    echo "  review_logs/bias_M70.log"
    return 1
fi

echo "[OK] Bias tests M20/M70"
echo


echo "============================================================"
echo "[DONE] All requested NPS Combine review diagnostics finished"
echo "============================================================"

find review_outputs/Run2Run3 \
    -maxdepth 3 \
    -type f \
    | sort

echo
echo "Logs:"
ls -lh review_logs/
