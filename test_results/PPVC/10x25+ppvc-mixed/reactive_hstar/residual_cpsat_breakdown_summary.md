# PPVC residual CP-SAT -- breakdown -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed` (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **breakdown**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- mean ORIGINAL DRL makespan: 211.0 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 207.6 | -1.62 | 0.6557 | 75.1 | 85.0 | 160.1 | 3.89 | 100/100 |
| residual_cpsat_30s | 207.6 | -1.63 | 1.1277 | 75.3 | 85.3 | 160.7 | 3.92 | 100/100 |
| drl_reactive | 211.0 | -0.02 | 0.5484 | 9.6 | 5.8 | 15.4 | 0.11 | 100/100 |
| right_shift | 221.1 | 4.82 | 0.0006 | 0.0 | 58.1 | 58.1 | 3.72 | 100/100 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 97 | 3 | 0 |
| 30 | 100 | 0 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 100 | 100 | 3 | 0.982 | 0 | 0 |
| residual_cpsat_30s | 100 | 100 | 1 | 0.999 | 0 | 0 |
