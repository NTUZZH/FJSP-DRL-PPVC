# PPVC reactive rescheduling — lag_perturb — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- disruption: **lag_perturb**  offset=0.4  lag_factor=2.0  pt_factor=1.5  shortage=0.5 x op_mean
- ga_reactive: budget 5.0 s, population 100, seed 0, seeded with the residual right-shift plan
- cpsat budgets (s): [1, 5, 30]  (num_workers=2)
- seed_test: 50
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False
- mean ORIGINAL (pre-disruption) DRL makespan: 210.9 h
- mean realized op count: 220.0
- nervousness time threshold: > 1.0 h start shift
- total infeasibilities: 0; residual-contract violations: 0

| method | mean makespan (h) | mean inflation % | mean time (s) | mch-nerv | time-nerv | comb-nerv | abs-shift (h) | feasible |
|---|---|---|---|---|---|---|---|---|
| drl_reactive | 221.0 | 4.82 | 0.4743 | 17.7 | 13.8 | 31.5 | 1.06 | 100/100 |
| spt_reactive | 222.5 | 5.50 | 0.0853 | 64.6 | 40.1 | 104.7 | 1.71 | 100/100 |
| ga_reactive | 221.4 | 5.02 | 5.0047 | 5.9 | 54.5 | 60.4 | 3.37 | 100/100 |
| right_shift | 232.6 | 10.33 | 0.0006 | 0.0 | 36.4 | 36.4 | 3.54 | 100/100 |
| cpsat_1 | 221.2 | 4.97 | 0.9280 | n/a | n/a | n/a | n/a | 100/100 |
| cpsat_5 | 213.3 | 1.18 | 2.6005 | n/a | n/a | n/a | n/a | 100/100 |
| cpsat_30 | 212.7 | 0.88 | 8.3746 | n/a | n/a | n/a | n/a | 100/100 |

Inflation % = 100 * (realized makespan - original DRL plan makespan) / original DRL plan makespan.
Nervousness = #ops whose assigned machine changed (mch-nerv) + #ops whose start shifted > 1.0 h (time-nerv), vs the original plan; comb-nerv is their sum; abs-shift is the mean absolute start shift. All over the ops present in both plans.

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| ga_reactive | 100 | 100 | 100 | 1.000 | 0 | 0 |
| cpsat_1 | 100 | 100 | 73 | 0.907 | 0 | 0 |
| cpsat_5 | 100 | 100 | 31 | 0.981 | 0 | 0 |
| cpsat_30 | 100 | 100 | 21 | 0.997 | 0 | 0 |
