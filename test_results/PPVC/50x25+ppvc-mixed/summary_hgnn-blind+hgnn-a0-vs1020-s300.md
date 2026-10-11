# HGNN evaluation summary - hgnn-blind+hgnn-a0-vs1020-s300

- dataset: `50x25+ppvc-mixed` (50 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a0-vs1020-s300/best.pt`
- arm: blind, lag_features: False, bound_pt: mean
- feasible (validator, true lags): 50/50

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 895.3 | 37.5 | 7.9368 |
| lag-blind plan (before repair) | 389.3 | 7.3 | 7.9322 |
