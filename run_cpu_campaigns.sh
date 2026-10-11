#!/usr/bin/env bash
# run_cpu_campaigns.sh — CPU-only evaluation campaigns of the paper.
#   bash run_cpu_campaigns.sh A   # CP-SAT anytime campaign, pinned to CORES_A (4 workers)
#   bash run_cpu_campaigns.sh B   # memetic, flow-line, by-products, buffers, pinned to CORES_B
# Each queue waits until START_AT (HH:MM today, or empty) and until the rule
# campaign log reports "campaign finished", then runs its steps one at a time.
# Every step is logged to train_log/cpu_campaign_<queue>.log. Steps whose
# outputs are resume-safe JSONL files can be re-run to continue.
set -uo pipefail
QUEUE="${1:?queue A or B}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
PY="${PY:-python}"
CORES_A="${CORES_A:-18-21}"
CORES_B="${CORES_B:-22-23}"
START_AT="${START_AT:-}"
LOG="train_log/cpu_campaign_${QUEUE}.log"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# CPU-only queues must never open a CUDA context (the GPU stays free for training jobs)
export CUDA_VISIBLE_DEVICES=""
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

if [ -n "$START_AT" ]; then
    target=$(date -d "today $START_AT" +%s)
    now=$(date +%s)
    if [ "$now" -lt "$target" ]; then say "waiting until $START_AT"; sleep $((target - now)); fi
fi
while ! grep -q "campaign finished" train_log/pdr_scale.log 2>/dev/null; do
    say "rule campaign still running; waiting 5 min"; sleep 300
done

run() { # run <label> <cores> <cmd...>
    local label="$1" cores="$2"; shift 2
    say "START $label"
    taskset -c "$cores" "$@" >> "train_log/cpu_step_${label}.log" 2>&1
    say "END   $label (exit $?)"
}

if [ "$QUEUE" = "A" ]; then
    C="$CORES_A"
    # policy-independent modes first; warm-policy runs after the headline policy is retrained
    for ds in 50x25+ppvc-mixed 80x25+ppvc-mixed; do
        for mode in warm-spt cold; do
            run "anytime_${ds}_${mode}_600" "$C" $PY eval_cpsat_anytime.py --dataset $ds --mode $mode \
                --time_limit 600 --num_workers 4 --spt_seed_rule per-instance --quiet
        done
    done
    for ds in 20x25+ppvc-mixed 30x25+ppvc-mixed; do
        for mode in warm-spt cold; do
            run "anytime_${ds}_${mode}_300" "$C" $PY eval_cpsat_anytime.py --dataset $ds --mode $mode \
                --time_limit 300 --num_workers 4 --spt_seed_rule per-instance --quiet
        done
    done
    for ds in 50x25+ppvc-mixed 80x25+ppvc-mixed; do
        for mode in warm-spt cold; do
            run "anytime_${ds}_${mode}_1800" "$C" $PY eval_cpsat_anytime.py --dataset $ds --mode $mode \
                --time_limit 1800 --num_workers 4 --start 0 --end 20 --spt_seed_rule per-instance --quiet
        done
    done
    say "queue A finished"
elif [ "$QUEUE" = "B" ]; then
    C="$CORES_B"
    run "memetic_10x25_60"  "$C" $PY eval_memetic.py --data_path data/PPVC/10x25+ppvc-mixed --budget_s 60
    run "memetic_10x25_300" "$C" $PY eval_memetic.py --data_path data/PPVC/10x25+ppvc-mixed --budget_s 300
    for ds in 20x25 30x25 50x25 80x25; do
        run "memetic_${ds}_300" "$C" $PY eval_memetic.py --data_path data/PPVC/${ds}+ppvc-mixed --budget_s 300
    done
    for ds in 10x25 20x25 30x25 50x25 80x25; do
        run "byproducts_${ds}" "$C" $PY eval_byproducts.py --data_path data/PPVC/${ds}+ppvc-mixed --quiet
        run "buffers_occ_${ds}" "$C" $PY eval_buffers.py occupancy --data_path data/PPVC/${ds}+ppvc-mixed --methods greedy SPT
    done
    for ds in 10x25 20x25 50x25 80x25; do
        run "flowline_${ds}" "$C" $PY eval_flowline.py --data_path data/PPVC/${ds}+ppvc-mixed --cpsat_time_limit 300 --cpsat_workers 2 --quiet
    done
    run "buffers_masked_10x25" "$C" $PY eval_buffers.py masked-policy --data_path data/PPVC/10x25+ppvc-mixed --B 8 6 4 3 2
    run "buffers_cpsat_10x25"  "$C" $PY eval_buffers.py cpsat-capacity --data_path data/PPVC/10x25+ppvc-mixed --B 8 6 4 3 2 --time_limit 60 --num_workers 2
    run "buffers_masked_50x25" "$C" $PY eval_buffers.py masked-policy --data_path data/PPVC/50x25+ppvc-mixed --max_instances 20 --out_dir test_results/PPVC/50x25+ppvc-mixed/buffers20 --B 16 12 8 6 4
    run "buffers_cpsat_50x25"  "$C" $PY eval_buffers.py cpsat-capacity --data_path data/PPVC/50x25+ppvc-mixed --max_instances 20 --out_dir test_results/PPVC/50x25+ppvc-mixed/buffers20 --B 16 12 8 6 4 --time_limit 120 --num_workers 2
    say "queue B finished"
fi
