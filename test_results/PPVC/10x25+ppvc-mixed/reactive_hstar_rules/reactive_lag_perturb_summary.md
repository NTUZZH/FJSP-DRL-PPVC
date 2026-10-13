# PPVC reactive rescheduling — lag_perturb — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **lag_perturb**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- ga_reactive: budget 5.0 s, population 100, seed 0, seeded with the residual right-shift plan
- cpsat budgets (s): []  (num_workers=1)
- seed_test: 50
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False
- mean ORIGINAL (pre-disruption) DRL makespan: 210.9 h
- mean realized op count: 220.0
- nervousness time threshold: > 1.0 h start shift
- total infeasibilities: 0; residual-contract violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| drl_reactive | 221.0 | 4.82 | 0.4863 | 17.7 | 13.8 | 31.5 | 1.06 | 100/100 |
| spt_reactive | 222.5 | 5.50 | 0.0856 | 64.6 | 40.1 | 104.7 | 1.71 | 100/100 |
| right_shift | 232.6 | 10.33 | 0.0006 | 0.0 | 36.4 | 36.4 | 3.54 | 100/100 |
| rule_spt_reactive | 221.2 | 4.90 | 0.0844 | 64.6 | 38.3 | 102.8 | 1.61 | 100/100 |
| rule_llf_reactive | 221.2 | 4.89 | 0.0875 | 64.9 | 38.5 | 103.4 | 1.62 | 100/100 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
