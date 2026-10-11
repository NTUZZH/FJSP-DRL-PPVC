# PPVC residual CP-SAT -- material_shortage -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed` (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **material_shortage**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x remaining_work
- budgets (s): [5, 30]  (num_workers=2)
- warm start: right-shift plan as CP-SAT solution hint (instances solved cold: 0)
- mean ORIGINAL DRL makespan: 411.6 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 435.8 | 5.88 | 5.0102 | 29.9 | 458.0 | 487.9 | 13.54 | 20/20 |
| residual_cpsat_30s | 421.9 | 2.49 | 30.0209 | 111.3 | 484.8 | 596.1 | 13.78 | 20/20 |
| drl_reactive | 410.1 | -0.37 | 4.2020 | 268.1 | 438.6 | 706.6 | 6.20 | 20/20 |
| right_shift | 441.6 | 7.31 | 0.0020 | 0.0 | 457.1 | 457.1 | 13.07 | 20/20 |

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
| residual_cpsat_5s | 20/20 | 2/20 | 441.65 | 442.10 | 435.75 | 3 | 0 |
| residual_cpsat_30s | 20/20 | 2/20 | 441.65 | 442.10 | 421.85 | 0 | 0 |
