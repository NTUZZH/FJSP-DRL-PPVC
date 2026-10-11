# PPVC residual CP-SAT -- lag_perturb -- 10x25+ppvc-mixed+full

- dataset: `10x25+ppvc-mixed` (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- disruption: **lag_perturb**  offset=0.4  lag_factor=2.0
- budgets (s): [1, 5, 30]
- mean ORIGINAL DRL makespan: 210.8 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | feasible |
|---|---|---|---|---|---|---|---|
| residual_cpsat_1s | 218.2 | 3.53 | 0.1924 | 78.7 | 93.0 | 171.8 | 100/100 |
| residual_cpsat_5s | 218.2 | 3.52 | 0.3172 | 77.9 | 91.9 | 169.8 | 100/100 |
| residual_cpsat_30s | 218.2 | 3.51 | 0.5620 | 77.8 | 92.3 | 170.1 | 100/100 |
| drl_reactive | 220.8 | 4.76 | 0.3406 | 20.9 | 16.8 | 37.7 | 100/100 |
| right_shift | 232.9 | 10.51 | 0.0005 | 0.0 | 37.0 | 37.0 | 100/100 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 1 | 97 | 3 | 0 |
| 5 | 98 | 2 | 0 |
| 30 | 100 | 0 | 0 |
