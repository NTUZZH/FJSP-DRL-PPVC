# PPVC reactive rescheduling — breakdown — 10x25+ppvc-mixed+full

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- disruption: **breakdown**  offset=0.4  lag_factor=2.0
- cpsat budgets (s): [5, 30]
- seed_test: 50
- architecture (from snapshot): use_lag_features=True, use_type_embedding=True
- mean ORIGINAL (pre-disruption) DRL makespan: 210.9 h
- nervousness time threshold: > 1.0 h start shift
- total infeasibilities: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | feasible |
|---|---|---|---|---|---|---|---|
| drl_reactive | 211.0 | 0.04 | 1.2774 | 9.6 | 5.5 | 15.0 | 100/100 |
| right_shift | 220.1 | 4.39 | 0.0007 | 0.0 | 59.9 | 59.9 | 100/100 |
| cpsat_5 | 205.0 | -2.76 | 3.7043 | n/a | n/a | n/a | 100/100 |
| cpsat_30 | 203.4 | -3.51 | 11.8664 | n/a | n/a | n/a | 100/100 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum.
NOTE: CP-SAT has no native machine-unavailability window and no solver changes are permitted, so its breakdown numbers re-solve the perturbed instance WITHOUT the downtime window — a lower-bound reference only. The lag_perturb frontier is the apples-to-apples CP-SAT comparison (perturbed time_lag fully captures it).
