# HGNN evaluation summary - hgnn-blind+hgnn-a0-s300

- dataset: `50x25+ppvc-mixed` (50 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a0-s300/best.pt`
- arm: blind, lag_features: False, bound_pt: mean
- feasible (validator, true lags): 50/50

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 927.6 | 39.8 | 8.1216 |
| lag-blind plan (before repair) | 397.2 | 9.2 | 8.1173 |
