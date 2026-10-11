# PPVC residual CP-SAT -- material_shortage -- 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed` (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **material_shortage**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x remaining_work
- budgets (s): [5, 30]  (num_workers=2)
- mean ORIGINAL DRL makespan: 211.0 h
- total validator-infeasibilities: 0
- total prefix violations: 0
- total breakdown-window violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| residual_cpsat_5s | 230.4 | 9.20 | 0.2637 | 80.0 | 105.7 | 185.7 | 9.44 | 100/100 |
| residual_cpsat_30s | 230.4 | 9.20 | 0.2672 | 80.3 | 106.2 | 186.5 | 9.41 | 100/100 |
| drl_reactive | 231.0 | 9.47 | 0.5371 | 45.3 | 43.5 | 88.8 | 3.11 | 100/100 |
| right_shift | 243.6 | 15.50 | 0.0005 | 0.0 | 91.2 | 91.2 | 12.60 | 100/100 |

| budget (s) | OPTIMAL | FEASIBLE-only | no-solution |
|---|---|---|---|
| 5 | 100 | 0 | 0 |
| 30 | 100 | 0 | 0 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| residual_cpsat_5s | 100 | 100 | 0 | n/a | 0 | 0 |
| residual_cpsat_30s | 100 | 100 | 0 | n/a | 0 | 0 |
