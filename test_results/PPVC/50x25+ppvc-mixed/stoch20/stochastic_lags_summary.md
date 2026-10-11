# Closed-loop execution under uncertain lags — 50x25+ppvc-mixed

- instances: 20; replications: 3; lag seed: 2027
- lag model: realized = round(L * X), X lognormal (mean 1, cv), truncated to [0.5, 2.0]; revealed at the op's completion
- policy: `10x25+ppvc-mixed+a1lagx05-2k-s301`; nominal CP-SAT budget 60 s; re-solve budget 5 s; CP-SAT workers 2

| cv | method | n | mean realized makespan | mean nominal plan | mean inflation % | gap to perfect info % | mean re-plans | mean re-plan s | mean wall s | feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | slab_reactive | 60 | 410.4 | 411.5 | -0.25 | -0.08 | 132.2 | 0.669 | 7.075 | 60/60 |
| 0.1 | slab_repair | 60 | 424.9 | 411.5 | 3.25 | 3.44 | 132.2 | 0.003 | 0.003 | 60/60 |
| 0.1 | slab_perfect_info | 60 | 410.8 | 411.5 | -0.17 | 0.00 | 0.0 | 0.000 | 6.939 | 60/60 |
| 0.2 | slab_reactive | 60 | 411.1 | 411.5 | -0.09 | 0.06 | 148.3 | 0.726 | 7.084 | 60/60 |
| 0.2 | slab_repair | 60 | 448.6 | 411.5 | 9.03 | 9.20 | 148.3 | 0.005 | 0.005 | 60/60 |
| 0.2 | slab_perfect_info | 60 | 410.9 | 411.5 | -0.15 | 0.00 | 0.0 | 0.000 | 6.922 | 60/60 |
| 0.3 | slab_reactive | 60 | 412.4 | 411.5 | 0.22 | 0.32 | 152.3 | 0.750 | 7.104 | 60/60 |
| 0.3 | slab_repair | 60 | 471.4 | 411.5 | 14.55 | 14.67 | 152.3 | 0.003 | 0.003 | 60/60 |
| 0.3 | slab_perfect_info | 60 | 411.1 | 411.5 | -0.10 | 0.00 | 0.0 | 0.000 | 6.923 | 60/60 |

Inflation % = 100 * (realized - nominal plan) / nominal plan, where the nominal plan is the method's own schedule under nominal lags (the policy plan for slab_perfect_info). Gap to perfect info is paired per (instance, cv, rep).
