#!/usr/bin/env bash
# run_train.sh — launch FJSP training inside a detached tmux session so it keeps
# running after SSH / VSCode disconnects. Any arguments are passed to train.py.
#
#   bash run_train.sh                     # default 10x5 SD2 run
#   bash run_train.sh --n_j 10 --n_m 5    # with train.py args
#
#   Watch live : tmux attach -t train     (detach again: Ctrl-b then d   | alias: tt)
#   Tail log   : tail -f train_log/run_*.log
#   Stop       : tmux kill-session -t train
set -euo pipefail

SESSION=train
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "A training session named '$SESSION' is already running."
    echo "  Watch it : tmux attach -t $SESSION   (or: tt)"
    echo "  Stop it  : tmux kill-session -t $SESSION"
    exit 0
fi

mkdir -p train_log
LOG="train_log/run_$(date +%Y%m%d_%H%M%S).log"

# Command executed inside the tmux session:
#   load conda, activate the project env, run training unbuffered while mirroring
#   all output to the log, then keep the pane open so the final lines stay visible.
read -r -d '' CMD <<EOF || true
source "\$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fjsp
echo "[run_train] started \$(date)  env=fjsp  args: $*" | tee -a "$LOG"
python -u train.py $* 2>&1 | tee -a "$LOG"
status=\${PIPESTATUS[0]}
echo "[run_train] training exited (status \$status) — press Enter to close pane" | tee -a "$LOG"
read
EOF

tmux new-session -d -s "$SESSION" "$CMD"

echo "Training launched in tmux session '$SESSION' (conda env: fjsp)."
echo "  Log file : $REPO_DIR/$LOG"
echo "  Watch    : tmux attach -t $SESSION   (detach: Ctrl-b then d  | alias: tt)"
echo "  Tail log : tail -f $LOG"
echo "  Stop     : tmux kill-session -t $SESSION"
