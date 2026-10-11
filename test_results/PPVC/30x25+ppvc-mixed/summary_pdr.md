# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `30x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/30x25+ppvc-mixed.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| FIFO | 401.2 | 20.4 | 1.0457 | 20/20 | n/a |
| MOR | 444.2 | 60.6 | 1.0392 | 20/20 | n/a |
| SPT | 324.9 | 6.3 | 1.0030 | 20/20 | n/a |
| MWKR | 389.5 | 24.7 | 1.2505 | 20/20 | n/a |
| SPTL | 339.8 | 8.1 | 1.1393 | 20/20 | n/a |
| LLF | 326.0 | 5.7 | 1.0142 | 20/20 | n/a |
| MWKRL | 374.4 | 17.3 | 1.0562 | 20/20 | n/a |
| LRPTL | 372.1 | 16.8 | 1.0397 | 20/20 | n/a |
