#!/usr/bin/env bash
# run_ga_baseline.sh — GA metaheuristic peer baseline.
# Runs eval_ga.py on the M (10x25) and M-tight (10x15) test sets at a 60 s
# per-instance budget (the GA converges by ~10-60 s, per smoke tests). Output
# Result_GA60+*.npy is auto-discovered by analyze_results.py as method "GA60",
# yielding gap + paired Wilcoxon vs the DRL methods. Guarded: waits until no
# real training is active so it never steals CPU from the multi-seed campaign.
#   Launch: tmux new-session -d -s ga-baseline 'bash run_ga_baseline.sh'
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate fjsp
LOG="train_log/ga_baseline.log"
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

_train_active() { pgrep -af "python -u train.py" | grep -v "tmux" | grep -q "train\.py"; }
# Require SUSTAINED idle (5 consecutive 60s checks) so we don't start in the
# ~3.5-min eval gap BETWEEN ablation arms — only after the whole campaign ends.
wait_for_train_free() {
    while true; do
        if _train_active; then say "  ... training active; waiting 120s"; sleep 120; continue; fi
        local ok=1
        for _ in 1 2 3 4 5; do sleep 60; if _train_active; then ok=0; break; fi; done
        [ "$ok" = 1 ] && { say "  ... sustained idle confirmed; starting GA"; return 0; }
    done
}

BUDGET="${BUDGET:-60}"
say "=== GA baseline START (budget ${BUDGET}s) ==="
for ds in 10x25+ppvc-mixed 10x15+ppvc-mixed; do
    wait_for_train_free
    say "RUN GA on ${ds}"
    python eval_ga.py --data_path "data/PPVC/${ds}" --budget_s "${BUDGET}" --seed 0 \
        2>&1 | tee -a "$LOG"
    say "END GA on ${ds} (exit ${PIPESTATUS[0]})"
done
say "=== GA baseline COMPLETE — re-run analyze_results.py to integrate ==="
