# HGNN evaluation summary - hgnn-blind+hgnn-a0-vs1020-s300

- dataset: `80x25+ppvc-mixed` (50 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a0-vs1020-s300/best.pt`
- arm: blind, lag_features: False, bound_pt: mean
- feasible (validator, true lags): 50/50

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 1331.3 | 52.3 | 16.4051 |
| lag-blind plan (before repair) | 569.9 | 11.4 | 16.3985 |
