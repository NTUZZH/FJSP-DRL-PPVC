# PPVC residual CP-SAT -- cancellation -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed` (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **cancellation**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- warm start: right-shift plan as CP-SAT solution hint (instances solved cold: 0)
- mean ORIGINAL DRL makespan: 411.6 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 409.6 | -0.47 | 5.0095 | 1.1 | 51.5 | 52.6 | 0.31 | 20/20 |
| residual_cpsat_30s | 407.4 | -1.01 | 30.0240 | 15.5 | 97.0 | 112.5 | 0.84 | 20/20 |
| drl_reactive | 405.9 | -1.40 | 4.0009 | 264.4 | 435.4 | 699.9 | 6.41 | 20/20 |
| right_shift | 411.4 | -0.06 | 0.0020 | 0.0 | 49.3 | 49.3 | 0.28 | 20/20 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 0 | 20 | 0 |
| 30 | 0 | 20 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20 | 20 | 20 | 0.964 | 0 | 0 |
| residual_cpsat_30s | 20 | 20 | 20 | 0.992 | 0 | 0 |

Warm start: the hint is the right-shift plan retimed to the integer residual model; "above" counts solves whose makespan exceeds the right-shift makespan or the hint makespan.

| method | hint given | hint = right-shift times | mean right-shift (h) | mean hint (h) | mean CP-SAT (h) | above right-shift | above hint |
|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20/20 | 20/20 | 411.35 | 411.35 | 409.65 | 0 | 0 |
| residual_cpsat_30s | 20/20 | 20/20 | 411.35 | 411.35 | 407.45 | 0 | 0 |
