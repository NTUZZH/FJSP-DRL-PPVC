# PPVC residual CP-SAT -- breakdown -- 10x25+ppvc-mixed+full

- dataset: `10x25+ppvc-mixed` (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- disruption: **breakdown**  offset=0.4  lag_factor=2.0
- budgets (s): [1, 5, 30]
- mean ORIGINAL DRL makespan: 210.8 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | feasible |
|---|---|---|---|---|---|---|---|
| residual_cpsat_1s | 207.9 | -1.36 | 0.2316 | 76.2 | 84.3 | 160.6 | 100/100 |
| residual_cpsat_5s | 207.9 | -1.38 | 0.3538 | 76.3 | 85.7 | 162.0 | 100/100 |
| residual_cpsat_30s | 207.9 | -1.39 | 0.7294 | 76.2 | 84.6 | 160.8 | 100/100 |
| drl_reactive | 210.9 | 0.04 | 0.3413 | 10.1 | 6.1 | 16.2 | 100/100 |
| right_shift | 220.1 | 4.41 | 0.0005 | 0.0 | 59.8 | 59.8 | 100/100 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 1 | 96 | 4 | 0 |
| 5 | 99 | 1 | 0 |
| 30 | 99 | 1 | 0 |
