#!/usr/bin/env bash
# Dispatching-rule campaign across batch sizes (CPU only, one process at a time).
# Output: test_results/PPVC/<set>/Result_<rule>+pdr_<set>.npy and summary_pdr.md.
# A size whose result files all exist is skipped, so re-running resumes.
# On 10x25 only the lag-aware rules run; the classic rules there are the
# stored Result_<rule>+10x25+ppvc-mixed+smoke files.
set -u
cd "$(dirname "$0")"
PY=${PY:-python}
TAG=pdr
CORES=${CORES:-20-23}
CLASSIC="FIFO MOR SPT MWKR"
LAGAWARE="SPTL LLF MWKRL LRPTL"

run_set () {   # $1 = set name, rest = rules
    local set=$1; shift
    local rules="$*"
    local missing=0
    for r in $rules; do
        [ -f "test_results/PPVC/$set/Result_${r}+${TAG}_${set}.npy" ] || missing=1
    done
    if [ $missing -eq 0 ]; then
        echo "[$(date '+%F %T')] $set: all result files exist, skipped"
        return 0
    fi
    echo "[$(date '+%F %T')] $set: start rules=$rules"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c "$CORES" "$PY" eval_ppvc.py \
        --model_name 10x25+ppvc-mixed+full --out_tag "$TAG" \
        --data_path "data/PPVC/$set" --methods $rules
    local rc=$?
    echo "[$(date '+%F %T')] $set: exit code $rc"
    return $rc
}

run_set 10x25+ppvc-mixed $LAGAWARE || exit 1
for set in 20x25+ppvc-mixed 30x25+ppvc-mixed 50x25+ppvc-mixed 80x25+ppvc-mixed; do
    run_set "$set" $CLASSIC $LAGAWARE || exit 1
done
echo "[$(date '+%F %T')] campaign finished"
