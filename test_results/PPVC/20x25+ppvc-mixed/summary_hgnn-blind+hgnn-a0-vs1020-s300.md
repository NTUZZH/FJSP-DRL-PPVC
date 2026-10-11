# HGNN evaluation summary - hgnn-blind+hgnn-a0-vs1020-s300

- dataset: `20x25+ppvc-mixed` (50 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a0-vs1020-s300/best.pt`
- arm: blind, lag_features: False, bound_pt: mean
- feasible (validator, true lags): 50/50

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 477.0 | 29.0 | 3.3191 |
| lag-blind plan (before repair) | 197.4 | 6.0 | 3.3179 |
