# PPVC reactive rescheduling — breakdown — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **breakdown**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean  breakdown_hours=24.0
- ga_reactive: budget 5.0 s, population 100, seed 0, seeded with the residual right-shift plan
- cpsat budgets (s): []  (num_workers=2)
- seed_test: 50
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False
- mean ORIGINAL (pre-disruption) DRL makespan: 210.9 h
- mean realized op count: 220.0
- nervousness time threshold: > 1.0 h start shift
- total infeasibilities: 0; residual-contract violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| drl_reactive | 211.4 | 0.22 | 0.4566 | 36.0 | 31.1 | 67.1 | 0.68 | 100/100 |
| spt_reactive | 215.0 | 1.95 | 0.0835 | 68.0 | 46.7 | 114.7 | 1.21 | 100/100 |
| ga_reactive | 211.8 | 0.45 | 5.0043 | 6.3 | 60.0 | 66.3 | 2.16 | 100/100 |
| right_shift | 223.3 | 5.92 | 0.0005 | 0.0 | 72.9 | 72.9 | 4.36 | 100/100 |
| rule_spt_reactive | 211.5 | 0.29 | 0.0827 | 66.5 | 44.4 | 110.8 | 1.02 | 100/100 |
| rule_llf_reactive | 211.8 | 0.42 | 0.0859 | 67.7 | 44.9 | 112.6 | 1.02 | 100/100 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: CP-SAT has no native machine-unavailability window and no solver changes are permitted, so its breakdown numbers re-solve the perturbed instance WITHOUT the downtime window — a lower-bound reference only. The lag_perturb frontier is the apples-to-apples CP-SAT comparison (perturbed time_lag fully captures it).

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| ga_reactive | 100 | 100 | 100 | 1.000 | 0 | 0 |
