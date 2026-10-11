# PPVC residual CP-SAT -- urgent_arrival -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed` (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **urgent_arrival**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- mean ORIGINAL DRL makespan: 411.6 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 603.6 | 46.65 | 5.0155 | 411.9 | 626.3 | 1038.2 | 62.81 | 20/20 |
| residual_cpsat_30s | 467.9 | 13.69 | 30.0231 | 402.9 | 620.0 | 1022.9 | 32.68 | 20/20 |
| drl_reactive | 415.6 | 0.97 | 4.9127 | 254.5 | 408.6 | 663.1 | 5.48 | 20/20 |
| right_shift | 545.4 | 32.52 | 0.0027 | 0.0 | 0.0 | 0.0 | 0.00 | 20/20 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 0 | 20 | 0 |
| 30 | 0 | 20 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20 | 20 | 20 | 0.954 | 0 | 0 |
| residual_cpsat_30s | 20 | 20 | 20 | 0.991 | 0 | 0 |
