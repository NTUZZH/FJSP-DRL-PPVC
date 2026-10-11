# PPVC reactive rescheduling — breakdown — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **breakdown**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- ga_reactive: budget 5.0 s, population 100, seed 0, seeded with the residual right-shift plan
- cpsat budgets (s): [1, 5, 30]  (num_workers=2)
- seed_test: 50
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False
- mean ORIGINAL (pre-disruption) DRL makespan: 210.9 h
- mean realized op count: 220.0
- nervousness time threshold: > 1.0 h start shift
- total infeasibilities: 0; residual-contract violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| drl_reactive | 210.9 | -0.00 | 0.4723 | 9.1 | 5.8 | 14.9 | 0.11 | 100/100 |
| spt_reactive | 212.5 | 0.73 | 0.0848 | 65.1 | 36.2 | 101.3 | 0.83 | 100/100 |
| ga_reactive | 210.2 | -0.36 | 5.0041 | 1.4 | 17.2 | 18.6 | 0.43 | 100/100 |
| right_shift | 220.9 | 4.77 | 0.0006 | 0.0 | 58.8 | 58.8 | 3.71 | 100/100 |
| cpsat_1 | 216.7 | 2.81 | 0.9968 | n/a | n/a | n/a | n/a | 100/100 |
| cpsat_5 | 204.5 | -2.99 | 3.3970 | n/a | n/a | n/a | n/a | 100/100 |
| cpsat_30 | 203.5 | -3.49 | 11.3370 | n/a | n/a | n/a | n/a | 100/100 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: CP-SAT has no native machine-unavailability window and no solver changes are permitted, so its breakdown numbers re-solve the perturbed instance WITHOUT the downtime window — a lower-bound reference only. The lag_perturb frontier is the apples-to-apples CP-SAT comparison (perturbed time_lag fully captures it).

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| ga_reactive | 100 | 100 | 100 | 1.000 | 0 | 0 |
| cpsat_1 | 100 | 100 | 96 | 0.904 | 0 | 0 |
| cpsat_5 | 100 | 100 | 44 | 0.980 | 0 | 0 |
| cpsat_30 | 100 | 100 | 26 | 0.997 | 0 | 0 |
