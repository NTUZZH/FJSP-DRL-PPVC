# PPVC residual CP-SAT -- breakdown -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed` (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **breakdown**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean  breakdown_hours=24.0
- budgets (s): [5, 30]  (num_workers=2)
- warm start: right-shift plan as CP-SAT solution hint (instances solved cold: 0)
- mean ORIGINAL DRL makespan: 210.9 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 207.8 | -1.45 | 0.4686 | 31.8 | 72.9 | 104.7 | 3.26 | 100/100 |
| residual_cpsat_30s | 207.8 | -1.46 | 1.0719 | 31.6 | 72.8 | 104.4 | 3.18 | 100/100 |
| drl_reactive | 211.4 | 0.22 | 0.4748 | 36.0 | 31.1 | 67.1 | 0.68 | 100/100 |
| right_shift | 239.2 | 13.46 | 0.0006 | 0.0 | 88.4 | 88.4 | 11.32 | 100/100 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 97 | 3 | 0 |
| 30 | 98 | 2 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 100 | 100 | 3 | 0.982 | 0 | 0 |
| residual_cpsat_30s | 100 | 100 | 2 | 0.997 | 0 | 0 |

Warm start: the hint is the right-shift plan retimed to the integer residual model; "above" counts solves whose makespan exceeds the right-shift makespan or the hint makespan.

| method | hint given | hint = right-shift times | mean right-shift (h) | mean hint (h) | mean CP-SAT (h) | above right-shift | above hint |
|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 100/100 | 28/100 | 239.16 | 223.31 | 207.84 | 0 | 0 |
| residual_cpsat_30s | 100/100 | 28/100 | 239.16 | 223.31 | 207.82 | 0 | 0 |
