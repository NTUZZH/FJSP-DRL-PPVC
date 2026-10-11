#!/usr/bin/env bash
# Dispatching rules on the 50- and 80-module variable-route sets (CPU, deterministic, not timed).
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="${PY:-python}"; ST=train_log/var_rules.status
for set in 50x25+ppvc-mixed-var 80x25+ppvc-mixed-var; do
    echo "[$(date '+%F %T')] START $set" >> $ST
    taskset -c 18-19 $PY eval_ppvc.py --model_name 10x25+ppvc-mixed+full --out_tag pdr --data_path data/PPVC/$set \
        --methods FIFO MOR SPT MWKR SPTL LLF MWKRL LRPTL --device cpu > train_log/var_rules_$set.log 2>&1
    echo "[$(date '+%F %T')] END $set (exit $?)" >> $ST
done
