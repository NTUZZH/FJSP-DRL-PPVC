# PPVC reactive rescheduling — cancellation — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **cancellation**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- ga_reactive: budget 5.0 s, population 100, seed 0, seeded with the residual right-shift plan
- cpsat budgets (s): []  (num_workers=1)
- seed_test: 50
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False
- mean ORIGINAL (pre-disruption) DRL makespan: 411.5 h
- mean realized op count: 1078.0
- nervousness time threshold: > 1.0 h start shift
- total infeasibilities: 0; residual-contract violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| drl_reactive | 405.7 | -1.41 | 4.0365 | 264.1 | 434.3 | 698.4 | 6.52 | 20/20 |
| spt_reactive | 420.8 | 2.27 | 0.7554 | 354.1 | 545.2 | 899.4 | 17.75 | 20/20 |
| right_shift | 411.1 | -0.11 | 0.0021 | 0.0 | 45.8 | 45.8 | 0.27 | 20/20 |
| rule_spt_reactive | 417.3 | 1.42 | 0.7492 | 340.6 | 545.0 | 885.6 | 15.64 | 20/20 |
| rule_llf_reactive | 418.3 | 1.66 | 0.7704 | 345.3 | 544.8 | 890.1 | 15.30 | 20/20 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: the whole-instance CP-SAT re-solves the realized instance from t=0 and does not fix the committed prefix — a reference only. The prefix-respecting residual CP-SAT is in eval_residual_cpsat.py.
