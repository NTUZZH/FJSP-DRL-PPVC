# PPVC reactive rescheduling — material_shortage — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `50x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **material_shortage**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x remaining_work
- ga_reactive: budget 5.0 s, population 100, seed 0, seeded with the residual right-shift plan
- cpsat budgets (s): [1, 5, 30]  (num_workers=2)
- seed_test: 50
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False
- mean ORIGINAL (pre-disruption) DRL makespan: 411.5 h
- mean realized op count: 1100.0
- nervousness time threshold: > 1.0 h start shift
- total infeasibilities: 0; residual-contract violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| drl_reactive | 409.7 | -0.44 | 4.1579 | 266.4 | 441.0 | 707.4 | 6.37 | 20/20 |
| spt_reactive | 426.6 | 3.68 | 0.7758 | 363.4 | 565.0 | 928.4 | 18.46 | 20/20 |
| ga_reactive | 434.1 | 5.49 | 5.0207 | 3.8 | 488.3 | 492.1 | 10.33 | 20/20 |
| right_shift | 442.3 | 7.48 | 0.0021 | 0.0 | 460.9 | 460.9 | 13.61 | 20/20 |
| cpsat_1 | 9161.9 | 2126.56 | 1.0357 | n/a | n/a | n/a | n/a | 0/20 |
| cpsat_5 | 753.1 | 82.98 | 5.0197 | n/a | n/a | n/a | n/a | 20/20 |
| cpsat_30 | 600.0 | 45.80 | 30.0372 | n/a | n/a | n/a | n/a | 20/20 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.
NOTE: the whole-instance CP-SAT cannot encode the release time, so it re-solves the undisrupted instance — a lower-bound reference only. The prefix-respecting residual CP-SAT (eval_residual_cpsat.py) encodes it.

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| ga_reactive | 20 | 20 | 20 | 1.000 | 0 | 0 |
| cpsat_1 | 20 | 20 | 20 | 0.726 | 20 | 0 |
| cpsat_5 | 20 | 20 | 20 | 0.946 | 0 | 0 |
| cpsat_30 | 20 | 20 | 20 | 0.990 | 0 | 0 |
