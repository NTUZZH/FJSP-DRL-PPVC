#!/usr/bin/env bash
# Queue C: policy-independent arms of the uncertain-lag study (SPT closed loop, CP-SAT plan + repair,
# rolling residual CP-SAT), cores 16-17, stops automatically at the wall-clock deadline stop_at.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
PY="${PY:-python}"
LOG=train_log/cpu_campaign_C.log
ts() { date '+%Y-%m-%d %H:%M:%S'; }
stop_at=$(date -d "2026-10-03 11:55" +%s)
run() { local label="$1"; shift; local now=$(date +%s); local left=$((stop_at - now)); [ "$left" -le 60 ] && { echo "[$(ts)] SKIP $label (deadline reached)" >> $LOG; return; }
  echo "[$(ts)] START $label" >> $LOG; timeout "$left" taskset -c 16-17 "$@" >> "train_log/cpu_step_${label}.log" 2>&1; echo "[$(ts)] END   $label (exit $?)" >> $LOG; }
run stoch_10x25_indep $PY eval_stochastic_lags.py --data_path data/PPVC/10x25+ppvc-mixed --methods spt_reactive cpsat_repair cpsat_rolling --rolling_cpsat --cpsat_workers 2 --cv 0.1 0.2 0.3 --n_rep 3
run stoch_50x25_indep $PY eval_stochastic_lags.py --data_path data/PPVC/50x25+ppvc-mixed --max_instances 20 --save_dir test_results/PPVC/50x25+ppvc-mixed/stoch20 --methods spt_reactive cpsat_repair --cpsat_workers 2 --cv 0.1 0.2 0.3 --n_rep 3
echo "[$(ts)] queue C finished" >> $LOG
