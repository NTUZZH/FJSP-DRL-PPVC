# PPVC residual CP-SAT -- cancellation -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed` (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **cancellation**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- mean ORIGINAL DRL makespan: 411.6 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 581.0 | 41.15 | 5.0137 | 397.7 | 603.5 | 1001.2 | 57.01 | 20/20 |
| residual_cpsat_30s | 448.7 | 9.02 | 30.0218 | 387.6 | 598.9 | 986.4 | 30.14 | 20/20 |
| drl_reactive | 405.9 | -1.40 | 4.4593 | 264.4 | 435.4 | 699.9 | 6.41 | 20/20 |
| right_shift | 411.4 | -0.06 | 0.0022 | 0.0 | 49.3 | 49.3 | 0.28 | 20/20 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 0 | 20 | 0 |
| 30 | 0 | 20 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20 | 20 | 20 | 0.936 | 0 | 0 |
| residual_cpsat_30s | 20 | 20 | 20 | 0.992 | 0 | 0 |
