# HGNN evaluation summary - hgnn-blind+hgnn-a0-s300

- dataset: `10x25+ppvc-mixed` (100 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a0-s300/best.pt`
- arm: blind, lag_features: False, bound_pt: mean
- feasible (validator, true lags): 100/100

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 315.9 | 22.0 | 1.2515 |
| lag-blind plan (before repair) | 136.0 | 4.6 | 1.2509 |
