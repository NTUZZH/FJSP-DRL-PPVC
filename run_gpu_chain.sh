#!/usr/bin/env bash
# run_gpu_chain.sh — full GPU pipeline for the remaining paper experiments.
# Waits for the A1->A2 ablation queue to finish, then sequentially:
#   1. A0 lag-blind training            (arm A0, model 10x25+ppvc-mixed+a0-lagblind)
#   2. greedy evals of A1 / A2          (ablation table rows)
#   3. A0 repair eval                   (if eval_a0_repair.py exists)
#   4. A3 sampling-100 eval             (DRL-S column)
#   5. M-tight A3-arch training         (model 10x15+ppvc-mixed+full)
#   6. M-tight eval (greedy + PDRs)
#   7. generalization evals of the M-mixed A3 model on S / L / RC / Steel sets
#
#   Launch : tmux new-session -d -s gpu-chain 'bash run_gpu_chain.sh'
#   Watch  : tmux attach -t gpu-chain      Log: train_log/gpu_chain.log
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fjsp

CLOG="train_log/gpu_chain.log"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$CLOG"; }
run() { # run <label> <cmd...>  — log, execute, record exit status
    local label="$1"; shift
    say "START $label"
    "$@" 2>&1 | tee -a "train_log/chain_${label}.log"
    local st=${PIPESTATUS[0]}
    say "END   $label (exit $st)"
    return $st
}

say "gpu-chain armed: waiting for ablation queue (A1->A2) to finish"
while true; do
    if grep -q "ablation queue complete" train_log/ablation_queue.log 2>/dev/null; then
        say "ablation queue complete detected"; break
    fi
    if grep -q "A1 failed" train_log/ablation_queue.log 2>/dev/null; then
        say "FATAL: A1 failed — stopping chain, manual investigation needed"; exit 1
    fi
    sleep 120
done

# 1. A0 lag-blind training (policy never sees lags; eval repairs later)
run a0_train python -u train.py --data_source PPVC --n_j 10 \
    --ppvc_lagblind True --use_lag_features False --use_type_embedding False \
    --vali_size 100 --model_suffix a0-lagblind \
    || say "WARN: A0 training failed — its eval will be skipped"

# 2. ablation greedy evals (M-mixed test set)
run eval_a1 python eval_ppvc.py --model_name 10x25+ppvc-mixed+a1-bare \
    --data_path data/PPVC/10x25+ppvc-mixed --methods greedy
run eval_a2 python eval_ppvc.py --model_name 10x25+ppvc-mixed+a2-lagfeat \
    --data_path data/PPVC/10x25+ppvc-mixed --methods greedy

# 3. A0 repair eval (skip loudly if absent)
if [ -f eval_a0_repair.py ] && [ -f trained_network/PPVC/10x25+ppvc-mixed+a0-lagblind.pth ]; then
    run eval_a0 python eval_a0_repair.py --model_name 10x25+ppvc-mixed+a0-lagblind \
        --data_path data/PPVC/10x25+ppvc-mixed
else
    say "SKIP eval_a0 (script or checkpoint missing)"
fi

# 4. A3 sampling-100 (the DRL-S column; ~5-6 h)
run eval_a3_sampling python eval_ppvc.py --model_name 10x25+ppvc-mixed+full \
    --data_path data/PPVC/10x25+ppvc-mixed --methods sampling --sample_times 100

# 5. M-tight training (full method on the capacity-tight factory)
run mtight_train python -u train.py --data_source PPVC --n_j 10 --ppvc_factory tight \
    --use_lag_features True --use_type_embedding True \
    --vali_size 100 --model_suffix full \
    || say "WARN: M-tight training failed — its eval will be skipped"

# 6. M-tight eval (greedy + all PDRs)
if [ -f trained_network/PPVC/10x15+ppvc-mixed+full.pth ]; then
    run eval_mtight python eval_ppvc.py --model_name 10x15+ppvc-mixed+full \
        --data_path data/PPVC/10x15+ppvc-mixed
fi

# 7. generalization evals: M-mixed A3 model on other instance classes
for ds in 5x9+ppvc-mixed 20x25+ppvc-mixed 10x25+ppvc-rc_project 10x25+ppvc-steel_project; do
    run "eval_gen_${ds}" python eval_ppvc.py --model_name 10x25+ppvc-mixed+full \
        --data_path "data/PPVC/${ds}"
done

say "gpu-chain COMPLETE — all GPU-side experiment data produced"
