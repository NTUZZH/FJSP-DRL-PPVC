# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `10x25+ppvc-mixed-var`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/10x25+ppvc-mixed-var.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| SPT | 208.1 | 10.1 | 0.1329 | 100/100 | n/a |
| LLF | 207.8 | 10.3 | 0.1371 | 100/100 | n/a |
| MWKR | 214.9 | 10.6 | 0.1346 | 100/100 | n/a |
| MOR | 216.7 | 9.9 | 0.1346 | 100/100 | n/a |
| FIFO | 222.0 | 10.1 | 0.1340 | 100/100 | n/a |
| SPTL | 208.3 | 10.2 | 0.1333 | 100/100 | n/a |
| MWKRL | 209.4 | 11.0 | 0.1352 | 100/100 | n/a |
| LRPTL | 209.0 | 11.4 | 0.1348 | 100/100 | n/a |
