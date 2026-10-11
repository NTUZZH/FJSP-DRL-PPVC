# PPVC residual CP-SAT -- pt_deviation -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed` (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **pt_deviation**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- mean ORIGINAL DRL makespan: 211.0 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 207.7 | -1.55 | 0.7072 | 75.5 | 84.8 | 160.4 | 3.88 | 100/100 |
| residual_cpsat_30s | 207.7 | -1.57 | 1.0188 | 74.8 | 86.8 | 161.7 | 3.94 | 100/100 |
| drl_reactive | 211.1 | 0.06 | 0.5370 | 28.5 | 21.8 | 50.3 | 0.42 | 100/100 |
| right_shift | 211.7 | 0.34 | 0.0005 | 0.0 | 23.4 | 23.4 | 0.34 | 100/100 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 97 | 3 | 0 |
| 30 | 99 | 1 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 100 | 100 | 3 | 0.991 | 0 | 0 |
| residual_cpsat_30s | 100 | 100 | 1 | 0.998 | 0 | 0 |
