# PPVC residual CP-SAT -- material_shortage -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed` (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **material_shortage**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x remaining_work
- budgets (s): [5, 30]  (num_workers=2)
- warm start: right-shift plan as CP-SAT solution hint (instances solved cold: 0)
- mean ORIGINAL DRL makespan: 411.5 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 438.3 | 6.52 | 5.0113 | 27.8 | 467.1 | 494.9 | 14.15 | 20/20 |
| residual_cpsat_30s | 422.6 | 2.69 | 30.0193 | 101.5 | 468.6 | 570.2 | 14.46 | 20/20 |
| drl_reactive | 409.7 | -0.44 | 4.2086 | 266.4 | 441.0 | 707.4 | 6.37 | 20/20 |
| right_shift | 442.3 | 7.48 | 0.0023 | 0.0 | 460.9 | 460.9 | 13.61 | 20/20 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 0 | 20 | 0 |
| 30 | 0 | 20 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20 | 20 | 20 | 0.937 | 0 | 0 |
| residual_cpsat_30s | 20 | 20 | 20 | 0.989 | 0 | 0 |

Warm start: the hint is the right-shift plan retimed to the integer residual model; "above" counts solves whose makespan exceeds the right-shift makespan or the hint makespan.

| method | hint given | hint = right-shift times | mean right-shift (h) | mean hint (h) | mean CP-SAT (h) | above right-shift | above hint |
|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20/20 | 2/20 | 442.27 | 442.75 | 438.30 | 6 | 0 |
| residual_cpsat_30s | 20/20 | 2/20 | 442.27 | 442.75 | 422.55 | 0 | 0 |
