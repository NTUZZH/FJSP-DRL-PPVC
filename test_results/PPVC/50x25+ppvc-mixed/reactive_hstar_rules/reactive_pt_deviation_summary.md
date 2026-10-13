# PPVC reactive rescheduling — pt_deviation — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **pt_deviation**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- ga_reactive: budget 5.0 s, population 100, seed 0, seeded with the residual right-shift plan
- cpsat budgets (s): []  (num_workers=1)
- seed_test: 50
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False
- mean ORIGINAL (pre-disruption) DRL makespan: 411.5 h
- mean realized op count: 1100.0
- nervousness time threshold: > 1.0 h start shift
- total infeasibilities: 0; residual-contract violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| drl_reactive | 412.1 | 0.15 | 4.1566 | 255.3 | 394.3 | 649.6 | 4.84 | 20/20 |
| spt_reactive | 426.9 | 3.75 | 0.7936 | 364.8 | 567.4 | 932.2 | 18.51 | 20/20 |
| right_shift | 415.9 | 1.06 | 0.0022 | 0.0 | 396.4 | 396.4 | 1.57 | 20/20 |
| rule_spt_reactive | 422.4 | 2.64 | 0.7843 | 352.7 | 564.4 | 917.0 | 15.56 | 20/20 |
| rule_llf_reactive | 425.4 | 3.38 | 0.8013 | 358.8 | 569.7 | 928.5 | 16.51 | 20/20 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: the whole-instance CP-SAT re-solves the realized instance from t=0 and does not fix the committed prefix — a reference only. The prefix-respecting residual CP-SAT is in eval_residual_cpsat.py.
