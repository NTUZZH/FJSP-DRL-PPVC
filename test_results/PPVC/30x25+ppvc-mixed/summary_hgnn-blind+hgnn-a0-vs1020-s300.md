# HGNN evaluation summary - hgnn-blind+hgnn-a0-vs1020-s300

- dataset: `30x25+ppvc-mixed` (20 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a0-vs1020-s300/best.pt`
- arm: blind, lag_features: False, bound_pt: mean
- feasible (validator, true lags): 20/20

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 620.0 | 29.9 | 4.5002 |
| lag-blind plan (before repair) | 261.1 | 5.6 | 4.4984 |
