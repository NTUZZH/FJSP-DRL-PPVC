#!/usr/bin/env bash
# Training-wait dose arms: identical to 10x25+ppvc-mixed+a1lagx05-2k-s301 except --lag_scale (0.25, 0.75).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2; unset CUDA_VISIBLE_DEVICES
PY="${PY:-python}"; ST=train_log/dose_arms.status
say() { echo "[$(date '+%F %T')] $*" >> "$ST"; }
for pair in "0.25 a1lagx025-2k-s301" "0.75 a1lagx075-2k-s301"; do
    set -- $pair; SC=$1; SUF=$2; M=10x25+ppvc-mixed+${SUF}
    taskset -c 20-21 $PY -B preflight_train.py --expect_params 28834 \
        --compare train_log/PPVC/config_10x25+ppvc-mixed+a1lagx05-2k-s301.json --allow_diff lag_scale,model_suffix --check_ckpt \
        --data_source PPVC --device cuda --ppvc_factory default --ppvc_mix mixed --max_updates 2000 --vali_size 100 \
        --ppvc_vali_sizes 10,20 --n_j 10 --use_lag_features False --use_type_embedding False --seed_train 301 \
        --lag_scale $SC --model_suffix $SUF > train_log/preflight_${M}.log 2>&1 || { say "PREFLIGHT FAIL $M"; continue; }
    say "START train $M"
    taskset -c 20-21 $PY -u train.py --data_source PPVC --device cuda --ppvc_factory default --ppvc_mix mixed \
        --max_updates 2000 --vali_size 100 --ppvc_vali_sizes 10,20 --n_j 10 --use_lag_features False \
        --use_type_embedding False --seed_train 301 --lag_scale $SC --model_suffix $SUF > "train_log/train_${M}.log" 2>&1
    say "END   train $M (exit $?)"
    for ds in 10x25 20x25 30x25 50x25 80x25 10x15; do
        taskset -c 20-21 $PY eval_ppvc.py --model_name $M --data_path data/PPVC/${ds}+ppvc-mixed --methods greedy --device cuda >> "train_log/eval_${M}.log" 2>&1
        say "  eval greedy $M $ds (exit $?)"
    done
done
say "dose arms finished"
