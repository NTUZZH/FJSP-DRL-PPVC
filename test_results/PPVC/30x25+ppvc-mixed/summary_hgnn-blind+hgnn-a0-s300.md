# HGNN evaluation summary - hgnn-blind+hgnn-a0-s300

- dataset: `30x25+ppvc-mixed` (20 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a0-s300/best.pt`
- arm: blind, lag_features: False, bound_pt: mean
- feasible (validator, true lags): 20/20

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 636.2 | 27.1 | 4.6624 |
| lag-blind plan (before repair) | 265.4 | 5.9 | 4.6607 |
