# PPVC residual CP-SAT -- pt_deviation -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed` (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **pt_deviation**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- warm start: right-shift plan as CP-SAT solution hint (instances solved cold: 0)
- mean ORIGINAL DRL makespan: 411.6 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 415.4 | 0.91 | 5.0099 | 2.4 | 391.1 | 393.5 | 1.59 | 20/20 |
| residual_cpsat_30s | 413.2 | 0.39 | 30.0187 | 22.3 | 394.4 | 416.7 | 1.75 | 20/20 |
| drl_reactive | 412.6 | 0.26 | 4.2206 | 252.1 | 396.4 | 648.5 | 4.84 | 20/20 |
| right_shift | 415.9 | 1.06 | 0.0021 | 0.0 | 391.4 | 391.4 | 1.53 | 20/20 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 0 | 20 | 0 |
| 30 | 0 | 20 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20 | 20 | 20 | 0.962 | 0 | 0 |
| residual_cpsat_30s | 20 | 20 | 20 | 0.993 | 0 | 0 |

Warm start: the hint is the right-shift plan retimed to the integer residual model; "above" counts solves whose makespan exceeds the right-shift makespan or the hint makespan.

| method | hint given | hint = right-shift times | mean right-shift (h) | mean hint (h) | mean CP-SAT (h) | above right-shift | above hint |
|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20/20 | 20/20 | 415.95 | 415.95 | 415.35 | 0 | 0 |
| residual_cpsat_30s | 20/20 | 20/20 | 415.95 | 415.95 | 413.20 | 0 | 0 |
