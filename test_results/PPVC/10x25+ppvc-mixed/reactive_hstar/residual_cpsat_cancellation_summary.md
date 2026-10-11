# PPVC residual CP-SAT -- cancellation -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed` (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **cancellation**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- budgets (s): [5, 30]  (num_workers=2)
- mean ORIGINAL DRL makespan: 211.0 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 204.2 | -3.23 | 0.3155 | 66.1 | 73.6 | 139.7 | 3.70 | 100/100 |
| residual_cpsat_30s | 204.2 | -3.23 | 0.3158 | 66.0 | 73.1 | 139.1 | 3.73 | 100/100 |
| drl_reactive | 206.6 | -2.10 | 0.4672 | 36.6 | 26.5 | 63.1 | 0.62 | 100/100 |
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
