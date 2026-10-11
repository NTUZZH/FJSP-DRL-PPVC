# Flow-line restriction vs flexible job shop: `10x25+ppvc-mixed`

- instances: 100; every schedule validated against the true lags
- flow-line rule: every operation is fixed to one station of its type; module j uses the (j mod n_t)-th station of type t (n_t stations of that type), with the processing time it has there
- compatible stations per operation: flexible 3.11, flow line 1.00
- flow-line CP-SAT: 300 s, 2 workers, cold start; status counts {'OPTIMAL': 100}
- flexible CP-SAT reference: `./or_solution/PPVC/10x25+ppvc-mixed.jsonl` (300.0 s; status counts {'FEASIBLE': 14, 'OPTIMAL': 86})
- greedy policy: `10x25+ppvc-mixed+full`
- flexibility gain = 100 x (flow-line - flexible) / flow-line makespan, per instance; mean +/- std
- cross-checks: SPT: 100/100 instances equal to `test_results/PPVC/10x25+ppvc-mixed/Result_SPT+10x25+ppvc-mixed+smoke_10x25+ppvc-mixed.npy`; greedy: 92/100 instances equal to `test_results/PPVC/10x25+ppvc-mixed/Result_greedy+10x25+ppvc-mixed+full_10x25+ppvc-mixed.npy`

| comparison | flow-line makespan | flexible makespan | flexibility gain (%) | instances | flexible shorter |
|---|---|---|---|---|---|
| SPT (flow line) vs SPT (flexible) | 238.4 | 216.0 | 9.34 +/- 3.38 | 100/100 | 100/100 |
| CP-SAT (flow line) vs CP-SAT reference (flexible) | 223.3 | 203.2 | 8.99 +/- 2.62 | 100/100 | 100/100 |
| best flow-line schedule vs greedy policy (flexible) | 223.3 | 210.9 | 5.53 +/- 3.08 | 100/100 | 97/100 |
