# PPVC residual CP-SAT -- breakdown -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed` (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **breakdown**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- warm start: right-shift plan as CP-SAT solution hint (instances solved cold: 0)
- mean ORIGINAL DRL makespan: 411.5 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 411.4 | -0.03 | 5.0105 | 2.8 | 35.5 | 38.2 | 0.20 | 20/20 |
| residual_cpsat_30s | 410.8 | -0.18 | 30.0217 | 7.8 | 57.5 | 65.2 | 0.36 | 20/20 |
| drl_reactive | 412.3 | 0.20 | 4.0770 | 154.2 | 247.3 | 401.5 | 3.59 | 20/20 |
| right_shift | 504.0 | 22.47 | 0.0064 | 0.0 | 682.9 | 682.9 | 55.86 | 20/20 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 0 | 20 | 0 |
| 30 | 0 | 20 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20 | 20 | 20 | 0.938 | 0 | 0 |
| residual_cpsat_30s | 20 | 20 | 20 | 0.989 | 0 | 0 |

Warm start: the hint is the right-shift plan retimed to the integer residual model; "above" counts solves whose makespan exceeds the right-shift makespan or the hint makespan.

| method | hint given | hint = right-shift times | mean right-shift (h) | mean hint (h) | mean CP-SAT (h) | above right-shift | above hint |
|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20/20 | 0/20 | 503.99 | 411.90 | 411.35 | 0 | 0 |
| residual_cpsat_30s | 20/20 | 0/20 | 503.99 | 411.90 | 410.75 | 0 | 0 |
