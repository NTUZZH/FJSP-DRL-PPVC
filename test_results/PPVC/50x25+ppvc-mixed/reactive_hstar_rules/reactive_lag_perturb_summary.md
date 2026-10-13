# PPVC reactive rescheduling — lag_perturb — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **lag_perturb**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
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
| drl_reactive | 411.6 | 0.02 | 4.1458 | 109.2 | 164.6 | 273.8 | 1.67 | 20/20 |
| spt_reactive | 429.4 | 4.38 | 0.8040 | 362.9 | 562.8 | 925.6 | 18.41 | 20/20 |
| right_shift | 424.6 | 3.18 | 0.0060 | 0.0 | 191.3 | 191.3 | 2.73 | 20/20 |
| rule_spt_reactive | 422.3 | 2.63 | 0.7902 | 353.7 | 561.0 | 914.6 | 15.66 | 20/20 |
| rule_llf_reactive | 421.9 | 2.53 | 0.8102 | 362.4 | 560.6 | 923.0 | 15.52 | 20/20 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
