# PPVC residual CP-SAT -- cancellation -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed` (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **cancellation**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- warm start: right-shift plan as CP-SAT solution hint (instances solved cold: 0)
- mean ORIGINAL DRL makespan: 211.0 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 204.2 | -3.23 | 0.1516 | 15.9 | 28.8 | 44.7 | 1.11 | 100/100 |
| residual_cpsat_30s | 204.2 | -3.23 | 0.1542 | 16.2 | 29.1 | 45.3 | 1.10 | 100/100 |
| drl_reactive | 206.6 | -2.10 | 0.5057 | 36.6 | 26.5 | 63.1 | 0.62 | 100/100 |
| right_shift | 207.9 | -1.47 | 0.0005 | 0.0 | 1.9 | 1.9 | 0.04 | 100/100 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 100 | 0 | 0 |
| 30 | 100 | 0 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 100 | 100 | 0 | n/a | 0 | 0 |
| residual_cpsat_30s | 100 | 100 | 0 | n/a | 0 | 0 |

Warm start: the hint is the right-shift plan retimed to the integer residual model; "above" counts solves whose makespan exceeds the right-shift makespan or the hint makespan.

| method | hint given | hint = right-shift times | mean right-shift (h) | mean hint (h) | mean CP-SAT (h) | above right-shift | above hint |
|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 100/100 | 100/100 | 207.88 | 207.88 | 204.16 | 0 | 0 |
| residual_cpsat_30s | 100/100 | 100/100 | 207.88 | 207.88 | 204.16 | 0 | 0 |
