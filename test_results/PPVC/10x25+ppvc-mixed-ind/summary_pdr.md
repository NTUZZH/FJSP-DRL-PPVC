# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `10x25+ppvc-mixed-ind`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/10x25+ppvc-mixed-ind.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| SPT | 218.6 | 9.2 | 0.1399 | 100/100 | n/a |
| LLF | 218.7 | 9.1 | 0.1446 | 100/100 | n/a |
| MWKR | 228.3 | 9.3 | 0.1420 | 100/100 | n/a |
| MOR | 235.3 | 10.7 | 0.1422 | 100/100 | n/a |
| FIFO | 233.7 | 10.9 | 0.1417 | 100/100 | n/a |
| SPTL | 219.4 | 9.3 | 0.1410 | 100/100 | n/a |
| MWKRL | 222.4 | 10.2 | 0.1426 | 100/100 | n/a |
| LRPTL | 222.0 | 10.1 | 0.1427 | 100/100 | n/a |
