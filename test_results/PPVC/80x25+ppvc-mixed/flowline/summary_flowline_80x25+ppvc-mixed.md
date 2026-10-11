# Flow-line restriction vs flexible job shop: `80x25+ppvc-mixed`

- instances: 50; every schedule validated against the true lags
- flow-line rule: every operation is fixed to one station of its type; module j uses the (j mod n_t)-th station of type t (n_t stations of that type), with the processing time it has there
- compatible stations per operation: flexible 3.12, flow line 1.00
- flow-line CP-SAT: 300 s, 2 workers, cold start; status counts {'FEASIBLE': 50}
- flexible CP-SAT reference: not available (`./or_solution/PPVC/80x25+ppvc-mixed.jsonl`)
- greedy policy: `10x25+ppvc-mixed+full`
- flexibility gain = 100 x (flow-line - flexible) / flow-line makespan, per instance; mean +/- std
- cross-checks: SPT: 50/50 instances equal to `test_results/PPVC/80x25+ppvc-mixed/Result_SPT+pdr_80x25+ppvc-mixed.npy`; greedy: no existing result file to compare

| comparison | flow-line makespan | flexible makespan | flexibility gain (%) | instances | flexible shorter |
|---|---|---|---|---|---|
| SPT (flow line) vs SPT (flexible) | 698.3 | 620.3 | 11.14 +/- 2.06 | 50/50 | 50/50 |
| CP-SAT (flow line) vs CP-SAT reference (flexible) | n/a | n/a | n/a | 0/50 | n/a |
| best flow-line schedule vs greedy policy (flexible) | 634.3 | 602.5 | 4.99 +/- 2.44 | 50/50 | 49/50 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| flow-line CP-SAT | 50 | 50 | 50 | 0.999 | 0 | 0 |
