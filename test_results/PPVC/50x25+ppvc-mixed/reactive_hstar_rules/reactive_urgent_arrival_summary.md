# PPVC reactive rescheduling — urgent_arrival — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **urgent_arrival**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- ga_reactive: budget 5.0 s, population 100, seed 0, seeded with the residual right-shift plan
- cpsat budgets (s): []  (num_workers=1)
- seed_test: 50
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False
- mean ORIGINAL (pre-disruption) DRL makespan: 411.5 h
- mean realized op count: 1122.0
- nervousness time threshold: > 1.0 h start shift
- total infeasibilities: 0; residual-contract violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| drl_reactive | 415.9 | 1.06 | 4.1088 | 253.2 | 395.2 | 648.5 | 5.21 | 20/20 |
| spt_reactive | 432.0 | 4.98 | 0.8078 | 359.8 | 565.5 | 925.4 | 18.69 | 20/20 |
| right_shift | 544.3 | 32.30 | 0.0022 | 0.0 | 0.0 | 0.0 | 0.00 | 20/20 |
| rule_spt_reactive | 427.6 | 3.91 | 0.7991 | 352.9 | 567.9 | 920.8 | 15.98 | 20/20 |
| rule_llf_reactive | 428.1 | 4.03 | 0.8209 | 354.6 | 567.5 | 922.0 | 16.27 | 20/20 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: the whole-instance CP-SAT re-solves the realized instance from t=0 and does not fix the committed prefix — a reference only. The prefix-respecting residual CP-SAT is in eval_residual_cpsat.py.
