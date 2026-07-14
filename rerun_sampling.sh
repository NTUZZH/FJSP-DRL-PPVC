#!/usr/bin/env bash
# rerun_sampling.sh — re-run the A3 (full) sampling-100 eval that crashed earlier
# (eval_ppvc.py:238 bitwise-NOT on float done() — now fixed with .astype(bool)).
# Produces the DRL-S headline column (\FullSamplingMean) for the M-class table.
# Waits until no train.py is running so it never steals CPU from the env rollouts
# of an active training; CP-SAT (pinned to cores 0-5) may run alongside.
#   Launch: tmux new-session -d -s rerun-sampling 'bash rerun_sampling.sh'
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate fjsp
LOG="train_log/rerun_sampling.log"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

while pgrep -af "python -u train.py" | grep -v "tmux" | grep -q "train\.py"; do
    say "  ... train.py active; waiting 120s before sampling eval"
    sleep 120
done
say "=== sampling-100 eval START (A3 full, 10x25) ==="
python eval_ppvc.py --model_name 10x25+ppvc-mixed+full \
    --data_path data/PPVC/10x25+ppvc-mixed --methods sampling --sample_times 100 \
    2>&1 | tee -a "$LOG"
say "=== sampling-100 eval END (exit ${PIPESTATUS[0]}) ==="
