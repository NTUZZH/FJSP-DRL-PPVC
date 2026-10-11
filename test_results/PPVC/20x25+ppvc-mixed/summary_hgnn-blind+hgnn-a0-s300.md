# HGNN evaluation summary - hgnn-blind+hgnn-a0-s300

- dataset: `20x25+ppvc-mixed` (50 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a0-s300/best.pt`
- arm: blind, lag_features: False, bound_pt: mean
- feasible (validator, true lags): 50/50

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 478.9 | 27.3 | 2.5227 |
| lag-blind plan (before repair) | 200.5 | 6.7 | 2.5216 |
