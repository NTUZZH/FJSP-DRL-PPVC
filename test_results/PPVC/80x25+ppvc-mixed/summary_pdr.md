# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `80x25+ppvc-mixed`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/80x25+ppvc-mixed.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| FIFO | 877.1 | 59.1 | 2.7305 | 50/50 | n/a |
| MOR | 1217.4 | 101.5 | 2.7296 | 50/50 | n/a |
| SPT | 620.3 | 10.6 | 2.7183 | 50/50 | n/a |
| MWKR | 835.2 | 63.4 | 2.7339 | 50/50 | n/a |
| SPTL | 635.3 | 14.8 | 2.7304 | 50/50 | n/a |
| LLF | 623.6 | 11.1 | 2.7634 | 50/50 | n/a |
| MWKRL | 805.4 | 43.6 | 2.7343 | 50/50 | n/a |
| LRPTL | 809.3 | 40.9 | 2.7333 | 50/50 | n/a |
