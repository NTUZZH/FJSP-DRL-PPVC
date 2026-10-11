# HGNN evaluation summary - hgnn-aware+hgnn-a1-vs1020-s300

- dataset: `30x25+ppvc-mixed` (20 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a1-vs1020-s300/best.pt`
- arm: aware, lag_features: False, bound_pt: min
- feasible (validator, true lags): 20/20

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 329.7 | 5.1 | 4.7677 |
