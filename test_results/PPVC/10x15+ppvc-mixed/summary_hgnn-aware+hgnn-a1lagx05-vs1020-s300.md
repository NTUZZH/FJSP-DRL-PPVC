# HGNN evaluation summary - hgnn-aware+hgnn-a1lagx05-vs1020-s300

- dataset: `10x15+ppvc-mixed` (100 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a1lagx05-vs1020-s300/best.pt`
- arm: aware, lag_features: False, bound_pt: min
- feasible (validator, true lags): 100/100

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 263.5 | 10.8 | 1.3778 |
