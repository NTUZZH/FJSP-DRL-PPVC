# PPVC residual CP-SAT -- lag_perturb -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed` (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **lag_perturb**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- mean ORIGINAL DRL makespan: 411.6 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 590.8 | 43.53 | 5.0139 | 413.6 | 624.5 | 1038.0 | 60.16 | 20/20 |
| residual_cpsat_30s | 462.4 | 12.36 | 30.0233 | 408.1 | 616.8 | 1024.8 | 31.91 | 20/20 |
| drl_reactive | 411.6 | -0.00 | 4.6041 | 109.0 | 170.2 | 279.3 | 1.71 | 20/20 |
| right_shift | 423.9 | 2.98 | 0.0031 | 0.0 | 192.2 | 192.2 | 2.58 | 20/20 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 0 | 20 | 0 |
| 30 | 0 | 20 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20 | 20 | 20 | 0.933 | 0 | 0 |
| residual_cpsat_30s | 20 | 20 | 20 | 0.988 | 0 | 0 |
