#!/usr/bin/env bash
# run_reactive.sh — full reactive/online rescheduling experiment. Runs
# eval_reactive.py for both disruption types on the M-class test
# set with the trained A3 model. Opportunistic + polite: waits until NO heavy job
# (train.py OR run_cpsat_ppvc.py) is active, then runs nice'd, so it never
# competes with other heavy jobs for CPU.
# CP-SAT budgets {5,30}s = the *online* frontier (300s offline is already the
# main-results reference, so it is intentionally excluded here).
#   Launch: tmux new-session -d -s reactive 'bash run_reactive.sh'
#   Log   : train_log/reactive_run.log
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate fjsp
LOG="train_log/reactive_run.log"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

MODEL="${MODEL:-10x25+ppvc-mixed+full}"
DATA="${DATA:-data/PPVC/10x25+ppvc-mixed}"
BUDGETS="${BUDGETS:-5,30}"
CORES="${CORES:-}"; TS=""; [ -n "$CORES" ] && TS="taskset -c $CORES"

wait_idle() {
    # Yield only to real model TRAINING (CPU-env-bound, must not be perturbed).
    # CP-SAT may run alongside on DISJOINT cores (both are core-pinned), so we no
    # longer block on it. The tmux-server argv is excluded from the train match.
    while pgrep -af "python -u train.py" | grep -v "tmux" | grep -q "train\.py"; do
        say "  ... training active; waiting 180s"
        sleep 180
    done
}

say "=== reactive experiment START (model=${MODEL} budgets=${BUDGETS}) ==="
for dt in lag_perturb breakdown; do
    wait_idle
    say "RUN reactive ${dt}"
    $TS nice -n 19 python eval_reactive.py --model_name "${MODEL}" --data_path "${DATA}" \
        --disruption_type "${dt}" --cpsat_budgets "${BUDGETS}" \
        2>&1 | tee -a "$LOG"
    say "END reactive ${dt} (exit ${PIPESTATUS[0]})"
done
say "=== reactive experiment COMPLETE ==="
