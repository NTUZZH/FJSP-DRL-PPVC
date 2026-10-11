# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `20x25+ppvc-mixed-ind`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/20x25+ppvc-mixed-ind.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| SPT | 265.0 | 7.1 | 0.3370 | 50/50 | n/a |
| LLF | 266.9 | 7.2 | 0.3467 | 50/50 | n/a |
| MWKR | 310.4 | 18.1 | 0.3406 | 50/50 | n/a |
| MOR | 327.7 | 18.7 | 0.3410 | 50/50 | n/a |
| FIFO | 312.1 | 10.2 | 0.3398 | 50/50 | n/a |
| SPTL | 285.7 | 12.3 | 0.3390 | 50/50 | n/a |
| MWKRL | 297.5 | 10.8 | 0.3419 | 50/50 | n/a |
| LRPTL | 294.4 | 10.9 | 0.3422 | 50/50 | n/a |
