# PPVC residual CP-SAT -- cancellation -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed` (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **cancellation**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- warm start: right-shift plan as CP-SAT solution hint (instances solved cold: 0)
- mean ORIGINAL DRL makespan: 411.5 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 409.1 | -0.59 | 5.0107 | 1.6 | 49.6 | 51.2 | 0.32 | 20/20 |
| residual_cpsat_30s | 407.4 | -0.98 | 30.0187 | 7.6 | 78.6 | 86.2 | 0.55 | 20/20 |
| drl_reactive | 405.7 | -1.41 | 4.0640 | 264.1 | 434.3 | 698.4 | 6.52 | 20/20 |
| right_shift | 411.1 | -0.11 | 0.0022 | 0.0 | 45.8 | 45.8 | 0.27 | 20/20 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 0 | 20 | 0 |
| 30 | 0 | 20 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20 | 20 | 20 | 0.937 | 0 | 0 |
| residual_cpsat_30s | 20 | 20 | 20 | 0.990 | 0 | 0 |

Warm start: the hint is the right-shift plan retimed to the integer residual model; "above" counts solves whose makespan exceeds the right-shift makespan or the hint makespan.

| method | hint given | hint = right-shift times | mean right-shift (h) | mean hint (h) | mean CP-SAT (h) | above right-shift | above hint |
|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20/20 | 20/20 | 411.05 | 411.05 | 409.05 | 0 | 0 |
| residual_cpsat_30s | 20/20 | 20/20 | 411.05 | 411.05 | 407.45 | 0 | 0 |
