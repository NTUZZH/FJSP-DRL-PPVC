# PPVC reactive rescheduling — breakdown — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **breakdown**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- ga_reactive: budget 5.0 s, population 100, seed 0, seeded with the residual right-shift plan
- cpsat budgets (s): [1, 5, 30]  (num_workers=2)
- seed_test: 50
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False
- mean ORIGINAL (pre-disruption) DRL makespan: 411.5 h
- mean realized op count: 1100.0
- nervousness time threshold: > 1.0 h start shift
- total infeasibilities: 0; residual-contract violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| drl_reactive | 412.3 | 0.20 | 3.9308 | 154.2 | 247.3 | 401.5 | 3.59 | 20/20 |
| spt_reactive | 426.5 | 3.66 | 0.7596 | 365.5 | 567.1 | 932.6 | 18.24 | 20/20 |
| ga_reactive | 411.6 | 0.04 | 5.0151 | 0.3 | 123.0 | 123.3 | 0.21 | 20/20 |
| right_shift | 412.0 | 0.12 | 0.0021 | 0.0 | 125.5 | 125.5 | 0.18 | 20/20 |
| cpsat_1 | 9161.9 | 2126.56 | 1.0330 | n/a | n/a | n/a | n/a | 0/20 |
| cpsat_5 | 752.9 | 82.91 | 5.0201 | n/a | n/a | n/a | n/a | 20/20 |
| cpsat_30 | 598.1 | 45.35 | 30.0369 | n/a | n/a | n/a | n/a | 20/20 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: CP-SAT has no native machine-unavailability window and no solver changes are permitted, so its breakdown numbers re-solve the perturbed instance WITHOUT the downtime window — a lower-bound reference only. The lag_perturb frontier is the apples-to-apples CP-SAT comparison (perturbed time_lag fully captures it).

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| ga_reactive | 20 | 20 | 20 | 0.991 | 0 | 0 |
| cpsat_1 | 20 | 20 | 20 | 0.729 | 20 | 0 |
| cpsat_5 | 20 | 20 | 20 | 0.947 | 0 | 0 |
| cpsat_30 | 20 | 20 | 20 | 0.989 | 0 | 0 |
