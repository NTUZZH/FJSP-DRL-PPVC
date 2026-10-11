# HGNN evaluation summary - hgnn-aware+hgnn-a0-vs1020-s300

- dataset: `30x25+ppvc-mixed` (20 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a0-vs1020-s300/best.pt`
- arm: aware, lag_features: False, bound_pt: mean
- feasible (validator, true lags): 20/20

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 351.1 | 7.8 | 4.9599 |
