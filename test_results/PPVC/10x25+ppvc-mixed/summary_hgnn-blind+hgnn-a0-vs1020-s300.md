# HGNN evaluation summary - hgnn-blind+hgnn-a0-vs1020-s300

- dataset: `10x25+ppvc-mixed` (100 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a0-vs1020-s300/best.pt`
- arm: blind, lag_features: False, bound_pt: mean
- feasible (validator, true lags): 100/100

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 318.0 | 22.8 | 1.5923 |
| lag-blind plan (before repair) | 136.1 | 4.8 | 1.5917 |
