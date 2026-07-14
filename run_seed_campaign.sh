#!/usr/bin/env bash
# run_seed_campaign.sh — multi-seed ablation campaign.
# Trains the ablation arms with independent seeds so the A1<A2<A3 ordering
# can be shown to exceed run-to-run noise, foregrounding the contention-heavy
# M-tight class where the adaptations should separate cleanly.
#
# Design:
#   * M-tight (10x15, tight factory): A0/A1/A2/A3 x seeds {300,301,302}
#       (seed-300 'full'/A3 is produced by the running gpu-chain; others new)
#   * M-class (10x25, default):       A3 x seeds {301,302}  (300 already done)
#   * each model evaluated on its matching held-out test set
#       - A0 (lag-blind) via eval_a0_repair.py ; A1/A2/A3 via eval_ppvc.py greedy
#   * CHECKPOINT-SKIP: never retrains an existing model
#   * COLLISION-SAFE: before each training, wait until no OTHER train.py runs,
#       so this chain never trains concurrently with the gpu-chain (CPU guard),
#       yet fills idle GPU during the gpu-chain's sampling/eval phases.
#
#   Launch : tmux new-session -d -s seed-campaign 'bash run_seed_campaign.sh'
#   Log    : train_log/seed_campaign.log
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fjsp

# --- parallelism knobs (override at launch) ---
#   PIN_CORES="0-8"  -> taskset every python to that core set
#   OMP=9            -> cap BLAS thread count (match the pin width) to stop the
#                       24-core CPU from being thread-oversubscribed
#   NO_GUARD=1       -> skip the wait-for-train-free guard, so this campaign may
#                       run CONCURRENTLY with another pinned campaign on disjoint
#                       cores (pinning, not serialization, prevents contention)
PIN_CORES="${PIN_CORES:-}"
TASKSET=""; [ -n "$PIN_CORES" ] && TASKSET="taskset -c $PIN_CORES"
if [ -n "${OMP:-}" ]; then
    export OMP_NUM_THREADS="$OMP" MKL_NUM_THREADS="$OMP" \
           OPENBLAS_NUM_THREADS="$OMP" NUMEXPR_NUM_THREADS="$OMP"
fi

CLOG="${CLOG:-train_log/seed_campaign.log}"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$CLOG"; }

# Wait until no train.py OTHER than our own subtree is running. Because our
# trainings run in the foreground (this function only runs BETWEEN jobs), any
# match here belongs to the gpu-chain — yield to it to protect the CPU.
wait_for_train_free() {
    [ -n "${NO_GUARD:-}" ] && return 0   # parallel mode: pinning prevents contention
    # Match only a REAL python train.py process, NOT the shared tmux server whose
    # argv retains the original 'python -u train.py' launch string (that false
    # positive deadlocked this guard once all real trainings had finished).
    while pgrep -af "python -u train.py" | grep -v "tmux" | grep -q "train\.py"; do
        say "  ... another train.py active; waiting 90s before next training"
        sleep 90
    done
}

# train_arm <testset> <suffix> <seed> <extra train flags...>
train_arm() {
    local ds_factory_args="$1"; local data_name="$2"; local suffix="$3"; local seed="$4"; shift 4
    local extra="$*"
    local model="${data_name}+${suffix}"
    local ckpt="trained_network/PPVC/${model}.pth"
    if [ -f "$ckpt" ]; then
        say "SKIP train ${model} (checkpoint exists)"
        return 0
    fi
    wait_for_train_free
    say "TRAIN ${model} (seed ${seed})"
    # shellcheck disable=SC2086
    $TASKSET python -u train.py --data_source PPVC --n_j 10 ${ds_factory_args} \
        --vali_size 100 --seed_train "${seed}" --model_suffix "${suffix}" ${extra} \
        2>&1 | tee -a "train_log/seedrun_${model}.log"
    say "END train ${model} (exit ${PIPESTATUS[0]})"
}

# eval_arm <model_name> <dataset_dir> <arm_kind>
eval_arm() {
    local model="$1"; local dsdir="$2"; local kind="$3"
    [ -f "trained_network/PPVC/${model}.pth" ] || { say "SKIP eval ${model} (no checkpoint)"; return 0; }
    say "EVAL ${model} on ${dsdir} (${kind})"
    if [ "$kind" = "a0" ]; then
        $TASKSET python eval_a0_repair.py --model_name "${model}" --data_path "data/PPVC/${dsdir}" \
            2>&1 | tee -a "train_log/seedeval_${model}.log"
    else
        $TASKSET python eval_ppvc.py --model_name "${model}" --data_path "data/PPVC/${dsdir}" --methods greedy \
            2>&1 | tee -a "train_log/seedeval_${model}.log"
    fi
    say "END eval ${model} (exit ${PIPESTATUS[0]})"
}

# Knobs (override at launch): SEED_LIST controls which seeds; DO_MCLASS_SEEDS
# toggles the M-class A3 extra-seed loop. Probe = SEED_LIST="300" DO_MCLASS_SEEDS=0.
SEED_LIST="${SEED_LIST:-300 301 302}"
DO_MCLASS_SEEDS="${DO_MCLASS_SEEDS:-1}"
say "=== seed campaign START (SEED_LIST='${SEED_LIST}' DO_MCLASS_SEEDS=${DO_MCLASS_SEEDS}) ==="

MT_ARGS="--ppvc_factory tight"          # M-tight 10x15
MT_NAME="10x15+ppvc-mixed"
M_ARGS=""                                # M-class 10x25 default
M_NAME="10x25+ppvc-mixed"

for seed in $SEED_LIST; do
    sfx=""; [ "$seed" != "300" ] && sfx="-s${seed}"
    # --- M-tight ablation, all four arms ---
    train_arm "$MT_ARGS" "$MT_NAME" "a0-lagblind${sfx}" "$seed" --ppvc_lagblind True --use_lag_features False --use_type_embedding False
    eval_arm  "${MT_NAME}+a0-lagblind${sfx}" "$MT_NAME" a0
    train_arm "$MT_ARGS" "$MT_NAME" "a1-bare${sfx}"     "$seed" --use_lag_features False --use_type_embedding False
    eval_arm  "${MT_NAME}+a1-bare${sfx}" "$MT_NAME" greedy
    train_arm "$MT_ARGS" "$MT_NAME" "a2-lagfeat${sfx}"  "$seed" --use_lag_features True  --use_type_embedding False
    eval_arm  "${MT_NAME}+a2-lagfeat${sfx}" "$MT_NAME" greedy
    # M-tight A3 seed-300 ('full') is produced by the gpu-chain; skip to avoid
    # duplicate training. Train the extra-seed A3 here.
    if [ "$seed" != "300" ]; then
        train_arm "$MT_ARGS" "$MT_NAME" "full${sfx}"    "$seed" --use_lag_features True  --use_type_embedding True
    fi
    eval_arm  "${MT_NAME}+full${sfx}" "$MT_NAME" greedy
done

# --- M-class A3 extra seeds for headline robustness ---
if [ "$DO_MCLASS_SEEDS" = "1" ]; then
    for seed in 301 302; do
        train_arm "$M_ARGS" "$M_NAME" "full-s${seed}" "$seed" --use_lag_features True --use_type_embedding True
        eval_arm  "${M_NAME}+full-s${seed}" "$M_NAME" greedy
    done
fi

say "=== seed campaign COMPLETE — multi-seed ablation data ready ==="
