# PPVC residual CP-SAT -- lag_perturb -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed` (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **lag_perturb**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- warm start: right-shift plan as CP-SAT solution hint (instances solved cold: 0)
- mean ORIGINAL DRL makespan: 411.6 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 421.6 | 2.42 | 5.0108 | 6.0 | 194.0 | 200.1 | 2.61 | 20/20 |
| residual_cpsat_30s | 413.9 | 0.57 | 30.0155 | 21.3 | 186.4 | 207.8 | 1.83 | 20/20 |
| drl_reactive | 411.6 | -0.00 | 4.7573 | 109.0 | 170.2 | 279.3 | 1.71 | 20/20 |
| right_shift | 423.9 | 2.98 | 0.0068 | 0.0 | 192.2 | 192.2 | 2.58 | 20/20 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 0 | 20 | 0 |
| 30 | 0 | 20 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20 | 20 | 20 | 0.958 | 0 | 0 |
| residual_cpsat_30s | 20 | 20 | 20 | 0.992 | 0 | 0 |

Warm start: the hint is the right-shift plan retimed to the integer residual model; "above" counts solves whose makespan exceeds the right-shift makespan or the hint makespan.

| method | hint given | hint = right-shift times | mean right-shift (h) | mean hint (h) | mean CP-SAT (h) | above right-shift | above hint |
|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20/20 | 20/20 | 423.90 | 423.90 | 421.55 | 0 | 0 |
| residual_cpsat_30s | 20/20 | 20/20 | 423.90 | 423.90 | 413.95 | 0 | 0 |
