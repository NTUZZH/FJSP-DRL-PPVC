# Flow-line restriction vs flexible job shop: `20x25+ppvc-mixed`

- instances: 50; every schedule validated against the true lags
- flow-line rule: every operation is fixed to one station of its type; module j uses the (j mod n_t)-th station of type t (n_t stations of that type), with the processing time it has there
- compatible stations per operation: flexible 3.12, flow line 1.00
- flow-line CP-SAT: 300 s, 2 workers, cold start; status counts {'OPTIMAL': 33, 'FEASIBLE': 17}
- flexible CP-SAT reference: `./or_solution/PPVC/20x25+ppvc-mixed.jsonl` (300.0 s; status counts {'FEASIBLE': 50})
- greedy policy: `10x25+ppvc-mixed+full`
- flexibility gain = 100 x (flow-line - flexible) / flow-line makespan, per instance; mean +/- std
- cross-checks: SPT: 50/50 instances equal to `test_results/PPVC/20x25+ppvc-mixed/Result_SPT+10x25+ppvc-mixed+full_20x25+ppvc-mixed.npy`; greedy: 42/50 instances equal to `test_results/PPVC/20x25+ppvc-mixed/Result_greedy+10x25+ppvc-mixed+full_20x25+ppvc-mixed.npy`

| comparison | flow-line makespan | flexible makespan | flexibility gain (%) | instances | flexible shorter |
|---|---|---|---|---|---|
| SPT (flow line) vs SPT (flexible) | 298.8 | 265.1 | 11.22 +/- 2.60 | 50/50 | 50/50 |
| CP-SAT (flow line) vs CP-SAT reference (flexible) | 261.9 | 255.0 | 2.59 +/- 2.01 | 50/50 | 45/50 |
| best flow-line schedule vs greedy policy (flexible) | 261.9 | 272.4 | -4.09 +/- 4.05 | 50/50 | 8/50 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| flow-line CP-SAT | 50 | 50 | 18 | 0.999 | 0 | 0 |
