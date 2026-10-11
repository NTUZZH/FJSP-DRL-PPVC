# PPVC residual CP-SAT -- lag_perturb -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed` (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **lag_perturb**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- warm start: right-shift plan as CP-SAT solution hint (instances solved cold: 0)
- mean ORIGINAL DRL makespan: 211.0 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 218.0 | 3.33 | 0.2987 | 19.2 | 49.9 | 69.1 | 3.20 | 100/100 |
| residual_cpsat_30s | 218.0 | 3.32 | 0.3954 | 19.5 | 50.8 | 70.3 | 3.25 | 100/100 |
| drl_reactive | 221.1 | 4.78 | 0.5810 | 18.0 | 13.9 | 31.9 | 1.07 | 100/100 |
| right_shift | 232.9 | 10.40 | 0.0006 | 0.0 | 35.6 | 35.6 | 3.48 | 100/100 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 98 | 2 | 0 |
| 30 | 100 | 0 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 100 | 100 | 2 | 0.984 | 0 | 0 |
| residual_cpsat_30s | 100 | 100 | 0 | n/a | 0 | 0 |

Warm start: the hint is the right-shift plan retimed to the integer residual model; "above" counts solves whose makespan exceeds the right-shift makespan or the hint makespan.

| method | hint given | hint = right-shift times | mean right-shift (h) | mean hint (h) | mean CP-SAT (h) | above right-shift | above hint |
|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 100/100 | 100/100 | 232.91 | 232.91 | 217.99 | 0 | 0 |
| residual_cpsat_30s | 100/100 | 100/100 | 232.91 | 232.91 | 217.98 | 0 | 0 |
