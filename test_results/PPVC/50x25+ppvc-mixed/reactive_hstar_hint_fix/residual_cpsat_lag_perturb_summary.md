# PPVC residual CP-SAT -- lag_perturb -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed` (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **lag_perturb**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- warm start: right-shift plan as CP-SAT solution hint (instances solved cold: 0)
- mean ORIGINAL DRL makespan: 411.5 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 421.6 | 2.46 | 5.0111 | 8.4 | 192.8 | 201.2 | 2.76 | 20/20 |
| residual_cpsat_30s | 414.4 | 0.71 | 30.0173 | 27.1 | 217.4 | 244.6 | 2.38 | 20/20 |
| drl_reactive | 411.6 | 0.02 | 4.3077 | 109.2 | 164.6 | 273.8 | 1.67 | 20/20 |
| right_shift | 424.6 | 3.18 | 0.0068 | 0.0 | 191.3 | 191.3 | 2.73 | 20/20 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 0 | 20 | 0 |
| 30 | 0 | 20 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20 | 20 | 20 | 0.929 | 0 | 0 |
| residual_cpsat_30s | 20 | 20 | 20 | 0.987 | 0 | 0 |

Warm start: the hint is the right-shift plan retimed to the integer residual model; "above" counts solves whose makespan exceeds the right-shift makespan or the hint makespan.

| method | hint given | hint = right-shift times | mean right-shift (h) | mean hint (h) | mean CP-SAT (h) | above right-shift | above hint |
|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20/20 | 20/20 | 424.60 | 424.60 | 421.60 | 0 | 0 |
| residual_cpsat_30s | 20/20 | 20/20 | 424.60 | 424.60 | 414.45 | 0 | 0 |
