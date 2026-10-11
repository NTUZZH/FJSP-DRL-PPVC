# PPVC reactive rescheduling — pt_deviation — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **pt_deviation**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
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
| drl_reactive | 412.1 | 0.15 | 4.1479 | 255.3 | 394.3 | 649.6 | 4.84 | 20/20 |
| spt_reactive | 426.9 | 3.75 | 0.7729 | 364.8 | 567.4 | 932.2 | 18.51 | 20/20 |
| ga_reactive | 414.2 | 0.67 | 5.0195 | 1.4 | 385.6 | 387.0 | 1.33 | 20/20 |
| right_shift | 415.9 | 1.06 | 0.0021 | 0.0 | 396.4 | 396.4 | 1.57 | 20/20 |
| cpsat_1 | 9164.0 | 2127.08 | 1.0350 | n/a | n/a | n/a | n/a | 0/20 |
| cpsat_5 | 753.0 | 82.96 | 5.0202 | n/a | n/a | n/a | n/a | 20/20 |
| cpsat_30 | 603.5 | 46.65 | 30.0377 | n/a | n/a | n/a | n/a | 20/20 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: the whole-instance CP-SAT re-solves the realized instance from t=0 and does not fix the committed prefix — a reference only. The prefix-respecting residual CP-SAT is in eval_residual_cpsat.py.

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| ga_reactive | 20 | 20 | 20 | 1.000 | 0 | 0 |
| cpsat_1 | 20 | 20 | 20 | 0.727 | 20 | 0 |
| cpsat_5 | 20 | 20 | 20 | 0.946 | 0 | 0 |
| cpsat_30 | 20 | 20 | 20 | 0.989 | 0 | 0 |
