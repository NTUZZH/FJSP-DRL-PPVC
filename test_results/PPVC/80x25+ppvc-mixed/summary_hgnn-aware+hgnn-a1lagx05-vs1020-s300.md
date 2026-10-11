# HGNN evaluation summary - hgnn-aware+hgnn-a1lagx05-vs1020-s300

- dataset: `80x25+ppvc-mixed` (50 instances)
- checkpoint: `backbones/hgnn/save/hgnn-a1lagx05-vs1020-s300/best.pt`
- arm: aware, lag_features: False, bound_pt: min
- feasible (validator, true lags): 50/50

| method | mean makespan | std | mean time (s) |
|---|---|---|---|
| greedy | 634.4 | 10.4 | 10.8739 |
