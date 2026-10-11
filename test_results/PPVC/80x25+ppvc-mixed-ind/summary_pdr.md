# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `80x25+ppvc-mixed-ind`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/80x25+ppvc-mixed-ind.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| SPT | 623.5 | 10.6 | 2.8356 | 50/50 | n/a |
| LLF | 625.5 | 10.9 | 2.8662 | 50/50 | n/a |
| MWKR | 832.9 | 61.6 | 2.8432 | 50/50 | n/a |
| MOR | 1237.4 | 93.0 | 2.8265 | 50/50 | n/a |
| FIFO | 892.3 | 62.7 | 2.8151 | 50/50 | n/a |
| SPTL | 635.5 | 13.1 | 2.8639 | 50/50 | n/a |
| MWKRL | 821.0 | 42.4 | 2.8473 | 50/50 | n/a |
| LRPTL | 815.2 | 39.5 | 2.8448 | 50/50 | n/a |
