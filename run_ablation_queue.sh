#!/usr/bin/env bash
# run_ablation_queue.sh — waits for the running A3 production training to
# finish, then trains ablation arms A1 (bare backbone on lag env) and
# A2 (+lag features) sequentially on the freed GPU.
#
#   Launch : tmux new-session -d -s queue 'bash run_ablation_queue.sh'
#   Watch  : tmux attach -t queue        Stop: tmux kill-session -t queue
#
# Note: the A3 tmux pane stays open after training (run_train.sh ends in
# `read`), so we poll for the training PROCESS, not the tmux session.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fjsp

ts() { date '+%Y-%m-%d %H:%M:%S'; }
QLOG="train_log/ablation_queue.log"
echo "[$(ts)] queue armed: waiting for A3 (model_suffix full) to finish" | tee -a "$QLOG"

# Anchor to the python process itself: the tmux/bash wrapper processes embed
# the whole launch script in their argv and would match an unanchored pattern
# forever (they idle at `read` after training ends).
while pgrep -f "^python -u train.py.*model_suffix full" > /dev/null; do
    sleep 60
done
echo "[$(ts)] A3 process gone" | tee -a "$QLOG"

if [ -f trained_network/PPVC/10x25+ppvc-mixed+full.pth ]; then
    echo "[$(ts)] A3 checkpoint present" | tee -a "$QLOG"
else
    echo "[$(ts)] WARNING: A3 checkpoint missing (run may have crashed) — continuing with ablations anyway" | tee -a "$QLOG"
fi

echo "[$(ts)] starting A1 (a1-bare: lag dynamics only, 10 features, no embeddings)" | tee -a "$QLOG"
python -u train.py --data_source PPVC --n_j 10 \
    --use_lag_features False --use_type_embedding False \
    --vali_size 100 --model_suffix a1-bare 2>&1 | tee "train_log/run_a1_$(date +%Y%m%d_%H%M%S).log"
A1_STATUS=${PIPESTATUS[0]}
echo "[$(ts)] A1 exited with status $A1_STATUS" | tee -a "$QLOG"

if [ "$A1_STATUS" -ne 0 ]; then
    echo "[$(ts)] A1 failed — NOT starting A2; investigate first" | tee -a "$QLOG"
    exit 1
fi

echo "[$(ts)] starting A2 (a2-lagfeat: + anticipatory lag features, no embeddings)" | tee -a "$QLOG"
python -u train.py --data_source PPVC --n_j 10 \
    --use_lag_features True --use_type_embedding False \
    --vali_size 100 --model_suffix a2-lagfeat 2>&1 | tee "train_log/run_a2_$(date +%Y%m%d_%H%M%S).log"
A2_STATUS=${PIPESTATUS[0]}
echo "[$(ts)] A2 exited with status $A2_STATUS" | tee -a "$QLOG"
echo "[$(ts)] ablation queue complete (A1=$A1_STATUS A2=$A2_STATUS)" | tee -a "$QLOG"
