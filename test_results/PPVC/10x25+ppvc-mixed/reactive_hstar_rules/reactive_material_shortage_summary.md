# PPVC reactive rescheduling — material_shortage — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **material_shortage**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x remaining_work
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
| drl_reactive | 231.0 | 9.53 | 0.4521 | 45.4 | 43.9 | 89.4 | 3.11 | 100/100 |
| spt_reactive | 231.1 | 9.59 | 0.0846 | 66.9 | 51.8 | 118.7 | 3.28 | 100/100 |
| right_shift | 243.5 | 15.50 | 0.0004 | 0.0 | 91.6 | 91.6 | 12.64 | 100/100 |
| rule_spt_reactive | 231.0 | 9.51 | 0.0834 | 67.2 | 52.0 | 119.2 | 3.28 | 100/100 |
| rule_llf_reactive | 231.0 | 9.52 | 0.0865 | 65.9 | 52.0 | 117.9 | 3.29 | 100/100 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: the whole-instance CP-SAT cannot encode the release time, so it re-solves the undisrupted instance — a lower-bound reference only. The prefix-respecting residual CP-SAT (eval_residual_cpsat.py) encodes it.
