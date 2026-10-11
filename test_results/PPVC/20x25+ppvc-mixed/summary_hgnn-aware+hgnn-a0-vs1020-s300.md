# HGNN evaluation summary - hgnn-aware+hgnn-a0-vs1020-s300

- dataset: `20x25+ppvc-mixed` (50 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a0-vs1020-s300/best.pt`
- arm: aware, lag_features: False, bound_pt: mean
- feasible (validator, true lags): 50/50

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 286.7 | 8.5 | 3.2424 |
