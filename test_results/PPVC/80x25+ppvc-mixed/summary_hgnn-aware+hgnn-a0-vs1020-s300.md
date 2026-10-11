# HGNN evaluation summary - hgnn-aware+hgnn-a0-vs1020-s300

- dataset: `80x25+ppvc-mixed` (50 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a0-vs1020-s300/best.pt`
- arm: aware, lag_features: False, bound_pt: mean
- feasible (validator, true lags): 50/50

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 680.1 | 10.2 | 15.3575 |
