# HGNN evaluation summary - hgnn-blind+hgnn-a0-s300

- dataset: `80x25+ppvc-mixed` (50 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a0-s300/best.pt`
- arm: blind, lag_features: False, bound_pt: mean
- feasible (validator, true lags): 50/50

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 1329.2 | 55.4 | 13.3742 |
| lag-blind plan (before repair) | 581.4 | 12.7 | 13.3683 |
