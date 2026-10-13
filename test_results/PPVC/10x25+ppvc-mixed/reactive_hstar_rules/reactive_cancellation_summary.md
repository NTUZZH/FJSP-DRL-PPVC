# PPVC reactive rescheduling — cancellation — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **cancellation**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- ga_reactive: budget 5.0 s, population 100, seed 0, seeded with the residual right-shift plan
- cpsat budgets (s): []  (num_workers=1)
- seed_test: 50
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False
- mean ORIGINAL (pre-disruption) DRL makespan: 210.9 h
- mean realized op count: 203.0
- nervousness time threshold: > 1.0 h start shift
- total infeasibilities: 0; residual-contract violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| drl_reactive | 206.5 | -2.07 | 0.4089 | 37.3 | 26.7 | 64.0 | 0.62 | 100/100 |
| spt_reactive | 207.1 | -1.78 | 0.0737 | 58.2 | 35.6 | 93.8 | 0.83 | 100/100 |
| right_shift | 207.8 | -1.47 | 0.0004 | 0.0 | 1.8 | 1.8 | 0.03 | 100/100 |
| rule_spt_reactive | 206.6 | -2.01 | 0.0727 | 58.0 | 35.2 | 93.2 | 0.81 | 100/100 |
| rule_llf_reactive | 206.7 | -1.97 | 0.0754 | 58.4 | 34.9 | 93.4 | 0.82 | 100/100 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: the whole-instance CP-SAT re-solves the realized instance from t=0 and does not fix the committed prefix — a reference only. The prefix-respecting residual CP-SAT is in eval_residual_cpsat.py.
