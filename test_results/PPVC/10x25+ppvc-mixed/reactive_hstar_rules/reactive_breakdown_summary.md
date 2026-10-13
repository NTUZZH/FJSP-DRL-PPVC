# PPVC reactive rescheduling — breakdown — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **breakdown**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
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
| drl_reactive | 210.9 | -0.00 | 0.4764 | 9.1 | 5.8 | 14.9 | 0.11 | 100/100 |
| spt_reactive | 212.5 | 0.73 | 0.0850 | 65.1 | 36.2 | 101.3 | 0.83 | 100/100 |
| right_shift | 210.9 | 0.01 | 0.0005 | 0.0 | 2.2 | 2.2 | 0.02 | 100/100 |
| rule_spt_reactive | 211.2 | 0.11 | 0.0840 | 65.3 | 33.5 | 98.8 | 0.74 | 100/100 |
| rule_llf_reactive | 211.1 | 0.09 | 0.0872 | 65.5 | 34.3 | 99.8 | 0.75 | 100/100 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: CP-SAT has no native machine-unavailability window and no solver changes are permitted, so its breakdown numbers re-solve the perturbed instance WITHOUT the downtime window — a lower-bound reference only. The lag_perturb frontier is the apples-to-apples CP-SAT comparison (perturbed time_lag fully captures it).
