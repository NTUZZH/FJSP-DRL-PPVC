# Closed-loop execution under uncertain lags — 10x25+ppvc-mixed

- instances: 100; replications: 3; lag seed: 2027
- lag model: realized = round(L * X), X lognormal (mean 1, cv), truncated to [0.5, 2.0]; revealed at the op's completion
- policy: `10x25+ppvc-mixed+full`; nominal CP-SAT budget 60 s; re-solve budget 5 s; CP-SAT workers 2

| cv | method | n | mean realized makespan | mean nominal plan | mean inflation % | gap to perfect info % | mean re-plans | mean re-plan s | mean wall s | feasible |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.1 | cpsat_repair | 300 | 211.8 | 203.2 | 4.25 | n/a | 26.1 | 0.001 | 0.001 | 300/300 |
| 0.1 | cpsat_rolling | 300 | 206.8 | 203.2 | 1.77 | n/a | 24.4 | 2.326 | 2.326 | 300/300 |
| 0.2 | cpsat_repair | 300 | 222.5 | 203.2 | 9.51 | n/a | 29.0 | 0.001 | 0.001 | 300/300 |
| 0.2 | cpsat_rolling | 300 | 211.7 | 203.2 | 4.19 | n/a | 27.0 | 2.798 | 2.798 | 300/300 |
| 0.3 | cpsat_repair | 300 | 231.4 | 203.2 | 13.88 | n/a | 29.8 | 0.001 | 0.001 | 300/300 |
| 0.3 | cpsat_rolling | 300 | 217.1 | 203.2 | 6.85 | n/a | 27.8 | 3.268 | 3.268 | 300/300 |

Inflation % = 100 * (realized - nominal plan) / nominal plan, where the nominal plan is the method's own schedule under nominal lags (the policy plan for slab_perfect_info). Gap to perfect info is paired per (instance, cv, rep).

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| cpsat_repair | 900 | 900 | 117 | 0.999 | 0 | 0 |
| cpsat_rolling | 900 | 24669 | 257 | 0.983 | 0 | 0 |
| nominal CP-SAT plan (one per instance) | 100 | 100 | 13 | 0.999 | 0 | 0 |
Each cpsat_repair and cpsat_rolling record counts the cached nominal plan solve among its timed solves.
