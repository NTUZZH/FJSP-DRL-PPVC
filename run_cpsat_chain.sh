#!/usr/bin/env bash
# run_cpsat_chain.sh — CPU pipeline: after the running M-mixed CP-SAT batch
# finishes, solve the remaining test sets sequentially (all at nice +19 so
# GPU-training env stepping always wins the CPU).
#
#   Launch : tmux new-session -d -s cpsat-chain 'bash run_cpsat_chain.sh'
#   Log    : or_solution/PPVC/cpsat_chain.log
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate fjsp

CLOG="or_solution/PPVC/cpsat_chain.log"
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$CLOG"; }

say "cpsat-chain armed: waiting for the running M-mixed batch to finish"
# The live solver was launched with a full conda-path python, so match the
# script name instead of an anchored "python" prefix. Its bash wrapper also
# matches but exits together with the solver (no trailing `read`), so the
# wait ends exactly when the batch does. This wait runs BEFORE this chain
# spawns its own solver children, so self-matching is not a concern.
while pgrep -f "run_cpsat_ppvc.py" > /dev/null; do
    sleep 120
done
say "M-mixed batch process gone"

for ds in 10x15+ppvc-mixed 5x9+ppvc-mixed 20x25+ppvc-mixed \
          10x25+ppvc-rc_project 10x25+ppvc-steel_project; do
    say "START dataset $ds"
    nice -n 19 python -u run_cpsat_ppvc.py --dataset "$ds" 2>&1 \
        | tee -a "or_solution/PPVC/cpsat_run_${ds}.log"
    say "END   dataset $ds (exit ${PIPESTATUS[0]})"
done

say "cpsat-chain COMPLETE — exact references ready for every test set"
