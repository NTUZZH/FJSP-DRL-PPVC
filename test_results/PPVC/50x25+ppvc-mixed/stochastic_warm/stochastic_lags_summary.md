# Closed-loop execution under uncertain lags — 50x25+ppvc-mixed

- instances: 20; replications: 3; lag seed: 2027
- lag model: realized = round(L * X), X lognormal (mean 1, cv), truncated to [0.5, 2.0]; revealed at the op's completion
- policy: `10x25+ppvc-mixed+full`; nominal CP-SAT budget 600 s; re-solve budget 5 s; CP-SAT workers 4; nominal CP-SAT plan started from the spt schedule (full hint)

| cv | method | n | mean realized makespan | mean nominal plan | mean inflation % | gap to perfect info % | mean re-plans | mean re-plan s | mean wall s | feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | cpsat_repair | 60 | 455.8 | 440.8 | 3.43 | n/a | 132.2 | 0.003 | 0.003 | 60/60 |
| 0.2 | cpsat_repair | 60 | 479.2 | 440.8 | 8.74 | n/a | 148.3 | 0.003 | 0.003 | 60/60 |
| 0.3 | cpsat_repair | 60 | 504.5 | 440.8 | 14.47 | n/a | 152.3 | 0.003 | 0.003 | 60/60 |

Inflation % = 100 * (realized - nominal plan) / nominal plan, where the nominal plan is the method's own schedule under nominal lags (the policy plan for slab_perfect_info). Gap to perfect info is paired per (instance, cv, rep).

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| cpsat_repair | 180 | 180 | 180 | 0.999 | 0 | 0 |
| nominal CP-SAT plan (one per instance) | 20 | 20 | 20 | 0.999 | 0 | 0 |
Each cpsat_repair and cpsat_rolling record counts the cached nominal plan solve among its timed solves.
