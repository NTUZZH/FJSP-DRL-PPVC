#!/usr/bin/env bash
# After the dose arms leave the GPU: greedy evaluation of the SLAB policy (seed 301) and of the
# policy trained on variable routes, on the 50- and 80-module variable-route sets.
cd "$(dirname "$0")"
unset CUDA_VISIBLE_DEVICES; export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
PY="${PY:-python}"; ST=train_log/var_policies.status
until grep -q 'dose arms finished' train_log/dose_arms.status 2>/dev/null; do sleep 60; done
echo "[$(date '+%F %T')] GPU free, start" >> $ST
for M in 10x25+ppvc-mixed+a1lagx05-2k-s301 10x25+ppvc-mixed-var+full2k-s301; do
    for set in 50x25+ppvc-mixed-var 80x25+ppvc-mixed-var; do
        taskset -c 20-21 $PY eval_ppvc.py --model_name $M --data_path data/PPVC/$set --methods greedy --device cuda >> train_log/var_policies.log 2>&1
        echo "[$(date '+%F %T')] eval $M $set (exit $?)" >> $ST
    done
done
echo "[$(date '+%F %T')] var policies finished" >> $ST
