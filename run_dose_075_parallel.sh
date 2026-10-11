#!/usr/bin/env bash
# Dose arm x0.75 in parallel with the x0.25 run (GPU 12% busy, each run CPU-bound on its own cores).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; unset CUDA_VISIBLE_DEVICES
PY="${PY:-python}"; ST=train_log/dose_075.status; SUF=a1lagx075-2k-s301; M=10x25+ppvc-mixed+$SUF
say() { echo "[$(date '+%F %T')] $*" >> "$ST"; }
taskset -c 22-23 $PY -B preflight_train.py --expect_params 28834 \
    --compare train_log/PPVC/config_10x25+ppvc-mixed+a1lagx05-2k-s301.json --allow_diff lag_scale,model_suffix --check_ckpt \
    --data_source PPVC --device cuda --ppvc_factory default --ppvc_mix mixed --max_updates 2000 --vali_size 100 \
    --ppvc_vali_sizes 10,20 --n_j 10 --use_lag_features False --use_type_embedding False --seed_train 301 \
    --lag_scale 0.75 --model_suffix $SUF > train_log/preflight_${M}.log 2>&1 || { say "PREFLIGHT FAIL $M"; exit 1; }
say "START train $M"
taskset -c 22-23 $PY -u train.py --data_source PPVC --device cuda --ppvc_factory default --ppvc_mix mixed \
    --max_updates 2000 --vali_size 100 --ppvc_vali_sizes 10,20 --n_j 10 --use_lag_features False \
    --use_type_embedding False --seed_train 301 --lag_scale 0.75 --model_suffix $SUF > "train_log/train_${M}.log" 2>&1
say "END   train $M (exit $?)"
for ds in 10x25 20x25 30x25 50x25 80x25 10x15; do
    taskset -c 22-23 $PY eval_ppvc.py --model_name $M --data_path data/PPVC/${ds}+ppvc-mixed --methods greedy --device cuda >> "train_log/eval_${M}.log" 2>&1
    say "  eval greedy $M $ds (exit $?)"
done
say "dose 0.75 finished"
