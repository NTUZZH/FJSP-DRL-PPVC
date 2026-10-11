# PPVC reactive rescheduling — urgent_arrival — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **urgent_arrival**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- ga_reactive: budget 5.0 s, population 100, seed 0, seeded with the residual right-shift plan
- cpsat budgets (s): [1, 5, 30]  (num_workers=2)
- seed_test: 50
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False
- mean ORIGINAL (pre-disruption) DRL makespan: 411.5 h
- mean realized op count: 1122.0
- nervousness time threshold: > 1.0 h start shift
- total infeasibilities: 0; residual-contract violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| drl_reactive | 415.9 | 1.06 | 4.3159 | 253.2 | 395.2 | 648.5 | 5.21 | 20/20 |
| spt_reactive | 432.0 | 4.98 | 0.8019 | 359.8 | 565.5 | 925.4 | 18.69 | 20/20 |
| ga_reactive | 455.6 | 10.70 | 5.0184 | 9.8 | 263.4 | 273.2 | 2.84 | 20/20 |
| right_shift | 544.3 | 32.30 | 0.0021 | 0.0 | 0.0 | 0.0 | 0.00 | 20/20 |
| cpsat_1 | 9340.8 | 2170.07 | 1.0330 | n/a | n/a | n/a | n/a | 0/20 |
| cpsat_5 | 765.1 | 85.90 | 5.0206 | n/a | n/a | n/a | n/a | 20/20 |
| cpsat_30 | 606.4 | 47.35 | 30.0381 | n/a | n/a | n/a | n/a | 20/20 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: the whole-instance CP-SAT re-solves the realized instance from t=0 and does not fix the committed prefix — a reference only. The prefix-respecting residual CP-SAT is in eval_residual_cpsat.py.

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| ga_reactive | 20 | 20 | 20 | 1.000 | 0 | 0 |
| cpsat_1 | 20 | 20 | 20 | 0.715 | 20 | 0 |
| cpsat_5 | 20 | 20 | 20 | 0.945 | 0 | 0 |
| cpsat_30 | 20 | 20 | 20 | 0.989 | 0 | 0 |
