# PPVC residual CP-SAT -- material_shortage -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed` (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **material_shortage**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x remaining_work
- budgets (s): [5, 30]  (num_workers=2)
- mean ORIGINAL DRL makespan: 411.6 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 589.8 | 43.27 | 5.0150 | 412.3 | 625.8 | 1038.0 | 60.24 | 20/20 |
| residual_cpsat_30s | 460.7 | 11.93 | 30.0236 | 403.9 | 618.4 | 1022.2 | 32.12 | 20/20 |
| drl_reactive | 410.1 | -0.37 | 4.7252 | 268.1 | 438.6 | 706.6 | 6.20 | 20/20 |
| right_shift | 441.6 | 7.31 | 0.0023 | 0.0 | 457.1 | 457.1 | 13.07 | 20/20 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 0 | 20 | 0 |
| 30 | 0 | 20 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 20 | 20 | 20 | 0.956 | 0 | 0 |
| residual_cpsat_30s | 20 | 20 | 20 | 0.992 | 0 | 0 |
