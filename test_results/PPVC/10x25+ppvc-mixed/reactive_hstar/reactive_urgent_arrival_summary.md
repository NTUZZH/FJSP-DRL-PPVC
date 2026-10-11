# PPVC reactive rescheduling — urgent_arrival — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **urgent_arrival**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- ga_reactive: budget 5.0 s, population 100, seed 0, seeded with the residual right-shift plan
- cpsat budgets (s): [1, 5, 30]  (num_workers=2)
- seed_test: 50
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False
- mean ORIGINAL (pre-disruption) DRL makespan: 210.9 h
- mean realized op count: 242.0
- nervousness time threshold: > 1.0 h start shift
- total infeasibilities: 0; residual-contract violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| drl_reactive | 230.7 | 9.51 | 0.5634 | 32.0 | 17.6 | 49.6 | 0.39 | 100/100 |
| spt_reactive | 231.5 | 9.91 | 0.1002 | 64.9 | 37.6 | 102.5 | 0.85 | 100/100 |
| ga_reactive | 231.5 | 9.90 | 5.0050 | 22.1 | 74.9 | 97.1 | 4.84 | 100/100 |
| right_shift | 327.9 | 55.64 | 0.0005 | 0.0 | 0.0 | 0.0 | 0.00 | 100/100 |
| cpsat_1 | 236.7 | 12.32 | 1.0045 | n/a | n/a | n/a | n/a | 100/100 |
| cpsat_5 | 208.6 | -1.00 | 4.4983 | n/a | n/a | n/a | n/a | 100/100 |
| cpsat_30 | 205.6 | -2.45 | 17.8865 | n/a | n/a | n/a | n/a | 100/100 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: the whole-instance CP-SAT re-solves the realized instance from t=0 and does not fix the committed prefix — a reference only. The prefix-respecting residual CP-SAT is in eval_residual_cpsat.py.

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| ga_reactive | 100 | 100 | 100 | 1.000 | 0 | 0 |
| cpsat_1 | 100 | 100 | 99 | 0.904 | 0 | 0 |
| cpsat_5 | 100 | 100 | 76 | 0.980 | 0 | 0 |
| cpsat_30 | 100 | 100 | 49 | 0.997 | 0 | 0 |
