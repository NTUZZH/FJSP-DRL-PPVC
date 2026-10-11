# HGNN evaluation summary - hgnn-aware+hgnn-a1-s300

- dataset: `10x25+ppvc-mixed` (100 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a1-s300/best.pt`
- arm: aware, lag_features: False, bound_pt: min
- feasible (validator, true lags): 100/100

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 214.1 | 9.0 | 1.2919 |
