# Closed-loop execution under uncertain lags — 10x25+ppvc-mixed

- instances: 100; replications: 3; lag seed: 2027
- lag model: realized = round(L * X), X lognormal (mean 1, cv), truncated to [0.5, 2.0]; revealed at the op's completion
- policy: `10x25+ppvc-mixed+a1lagx05-2k-s301`; nominal CP-SAT budget 60 s; re-solve budget 5 s; CP-SAT workers 2

| cv | method | n | mean realized makespan | mean nominal plan | mean inflation % | gap to perfect info % | mean re-plans | mean re-plan s | mean wall s | feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | slab_reactive | 300 | 212.2 | 210.9 | 0.61 | -0.04 | 26.1 | 0.076 | 0.798 | 300/300 |
| 0.1 | slab_repair | 300 | 217.8 | 210.9 | 3.27 | 2.63 | 26.1 | 0.001 | 0.001 | 300/300 |
| 0.1 | slab_perfect_info | 300 | 212.3 | 210.9 | 0.65 | 0.00 | 0.0 | 0.000 | 0.785 | 300/300 |
| 0.2 | slab_reactive | 300 | 215.2 | 210.9 | 2.04 | -0.02 | 29.0 | 0.083 | 0.801 | 300/300 |
| 0.2 | slab_repair | 300 | 227.4 | 210.9 | 7.79 | 5.72 | 29.0 | 0.001 | 0.001 | 300/300 |
| 0.2 | slab_perfect_info | 300 | 215.2 | 210.9 | 2.06 | 0.00 | 0.0 | 0.000 | 0.786 | 300/300 |
| 0.3 | slab_reactive | 300 | 218.8 | 210.9 | 3.75 | 0.02 | 29.8 | 0.085 | 0.802 | 300/300 |
| 0.3 | slab_repair | 300 | 236.1 | 210.9 | 11.92 | 8.07 | 29.8 | 0.001 | 0.001 | 300/300 |
| 0.3 | slab_perfect_info | 300 | 218.8 | 210.9 | 3.74 | 0.00 | 0.0 | 0.000 | 0.786 | 300/300 |

Inflation % = 100 * (realized - nominal plan) / nominal plan, where the nominal plan is the method's own schedule under nominal lags (the policy plan for slab_perfect_info). Gap to perfect info is paired per (instance, cv, rep).
