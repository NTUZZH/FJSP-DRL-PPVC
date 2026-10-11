# PPVC reactive rescheduling — pt_deviation — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **pt_deviation**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
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
| drl_reactive | 211.1 | 0.07 | 0.4542 | 28.7 | 21.9 | 50.5 | 0.41 | 100/100 |
| spt_reactive | 212.6 | 0.77 | 0.0838 | 66.6 | 41.8 | 108.3 | 0.92 | 100/100 |
| ga_reactive | 210.3 | -0.27 | 5.0039 | 1.8 | 34.0 | 35.8 | 0.75 | 100/100 |
| right_shift | 211.6 | 0.33 | 0.0005 | 0.0 | 23.5 | 23.5 | 0.34 | 100/100 |
| cpsat_1 | 216.6 | 2.79 | 0.9978 | n/a | n/a | n/a | n/a | 100/100 |
| cpsat_5 | 204.7 | -2.91 | 3.4791 | n/a | n/a | n/a | n/a | 100/100 |
| cpsat_30 | 203.5 | -3.49 | 11.1095 | n/a | n/a | n/a | n/a | 100/100 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: the whole-instance CP-SAT re-solves the realized instance from t=0 and does not fix the committed prefix — a reference only. The prefix-respecting residual CP-SAT is in eval_residual_cpsat.py.

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| ga_reactive | 100 | 100 | 100 | 1.000 | 0 | 0 |
| cpsat_1 | 100 | 100 | 97 | 0.905 | 0 | 0 |
| cpsat_5 | 100 | 100 | 46 | 0.982 | 0 | 0 |
| cpsat_30 | 100 | 100 | 27 | 0.997 | 0 | 0 |
