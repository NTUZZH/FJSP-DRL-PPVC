# PPVC reactive rescheduling — material_shortage — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **material_shortage**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x remaining_work
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
| drl_reactive | 231.0 | 9.53 | 0.4607 | 45.4 | 43.9 | 89.4 | 3.11 | 100/100 |
| spt_reactive | 231.1 | 9.59 | 0.0839 | 66.9 | 51.8 | 118.7 | 3.28 | 100/100 |
| ga_reactive | 233.1 | 10.53 | 5.0044 | 8.5 | 90.5 | 99.0 | 8.75 | 100/100 |
| right_shift | 243.5 | 15.50 | 0.0005 | 0.0 | 91.6 | 91.6 | 12.64 | 100/100 |
| cpsat_1 | 217.2 | 3.04 | 0.9936 | n/a | n/a | n/a | n/a | 100/100 |
| cpsat_5 | 204.5 | -3.00 | 3.4192 | n/a | n/a | n/a | n/a | 100/100 |
| cpsat_30 | 203.4 | -3.53 | 11.0975 | n/a | n/a | n/a | n/a | 100/100 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: the whole-instance CP-SAT cannot encode the release time, so it re-solves the undisrupted instance — a lower-bound reference only. The prefix-respecting residual CP-SAT (eval_residual_cpsat.py) encodes it.

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| ga_reactive | 100 | 100 | 100 | 1.000 | 0 | 0 |
| cpsat_1 | 100 | 100 | 94 | 0.903 | 0 | 0 |
| cpsat_5 | 100 | 100 | 45 | 0.981 | 0 | 0 |
| cpsat_30 | 100 | 100 | 26 | 0.997 | 0 | 0 |
