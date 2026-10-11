# PPVC reactive rescheduling — lag_perturb — 10x25+ppvc-mixed+full

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- disruption: **lag_perturb**  offset=0.4  lag_factor=2.0
- cpsat budgets (s): [5, 30]
- seed_test: 50
- architecture (from snapshot): use_lag_features=True, use_type_embedding=True
- mean ORIGINAL (pre-disruption) DRL makespan: 210.9 h
- nervousness time threshold: > 1.0 h start shift
- total infeasibilities: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | feasible |
|---|---|---|---|---|---|---|---|
| drl_reactive | 220.7 | 4.65 | 1.1862 | 20.7 | 16.8 | 37.5 | 100/100 |
| right_shift | 233.0 | 10.49 | 0.0006 | 0.0 | 37.2 | 37.2 | 100/100 |
| cpsat_5 | 213.1 | 1.11 | 2.6217 | n/a | n/a | n/a | 100/100 |
| cpsat_30 | 212.5 | 0.80 | 8.4088 | n/a | n/a | n/a | 100/100 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum.
