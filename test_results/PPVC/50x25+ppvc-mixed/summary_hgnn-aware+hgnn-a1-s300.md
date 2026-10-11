# HGNN evaluation summary - hgnn-aware+hgnn-a1-s300

- dataset: `50x25+ppvc-mixed` (50 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a1-s300/best.pt`
- arm: aware, lag_features: False, bound_pt: min
- feasible (validator, true lags): 50/50

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 466.8 | 8.7 | 7.8702 |
