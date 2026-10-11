# PPVC residual CP-SAT -- urgent_arrival -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed` (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **urgent_arrival**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- warm start: right-shift plan as CP-SAT solution hint (instances solved cold: 0)
- mean ORIGINAL DRL makespan: 211.0 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 228.4 | 8.38 | 0.3345 | 10.9 | 25.0 | 35.9 | 1.21 | 100/100 |
| residual_cpsat_30s | 228.4 | 8.37 | 0.6132 | 10.9 | 25.3 | 36.2 | 1.26 | 100/100 |
| drl_reactive | 230.7 | 9.46 | 0.6778 | 31.9 | 17.9 | 49.9 | 0.39 | 100/100 |
| right_shift | 328.2 | 55.74 | 0.0006 | 0.0 | 0.0 | 0.0 | 0.00 | 100/100 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 97 | 3 | 0 |
| 30 | 100 | 0 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 100 | 100 | 3 | 0.986 | 0 | 0 |
| residual_cpsat_30s | 100 | 100 | 0 | n/a | 0 | 0 |

Warm start: the hint is the right-shift plan retimed to the integer residual model; "above" counts solves whose makespan exceeds the right-shift makespan or the hint makespan.

| method | hint given | hint = right-shift times | mean right-shift (h) | mean hint (h) | mean CP-SAT (h) | above right-shift | above hint |
|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 100/100 | 100/100 | 328.22 | 328.22 | 228.40 | 0 | 0 |
| residual_cpsat_30s | 100/100 | 100/100 | 328.22 | 328.22 | 228.39 | 0 | 0 |
