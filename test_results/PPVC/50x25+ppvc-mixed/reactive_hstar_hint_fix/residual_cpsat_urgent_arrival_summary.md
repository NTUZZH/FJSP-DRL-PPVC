# PPVC residual CP-SAT -- urgent_arrival -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed` (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **urgent_arrival**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- warm start: right-shift plan as CP-SAT solution hint (instances solved cold: 0)
- mean ORIGINAL DRL makespan: 411.5 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 439.7 | 6.91 | 5.0120 | 1.4 | 7.4 | 8.8 | 0.29 | 20/20 |
| residual_cpsat_30s | 420.2 | 2.14 | 30.0186 | 25.4 | 93.9 | 119.3 | 2.05 | 20/20 |
| drl_reactive | 415.9 | 1.06 | 4.3024 | 253.2 | 395.2 | 648.5 | 5.21 | 20/20 |
| right_shift | 544.3 | 32.30 | 0.0023 | 0.0 | 0.0 | 0.0 | 0.00 | 20/20 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 0 | 20 | 0 |
| 30 | 0 | 20 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20 | 20 | 20 | 0.934 | 0 | 0 |
| residual_cpsat_30s | 20 | 20 | 20 | 0.987 | 0 | 0 |

Warm start: the hint is the right-shift plan retimed to the integer residual model; "above" counts solves whose makespan exceeds the right-shift makespan or the hint makespan.

| method | hint given | hint = right-shift times | mean right-shift (h) | mean hint (h) | mean CP-SAT (h) | above right-shift | above hint |
|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20/20 | 20/20 | 544.30 | 544.30 | 439.70 | 0 | 0 |
| residual_cpsat_30s | 20/20 | 20/20 | 544.30 | 544.30 | 420.25 | 0 | 0 |
