# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `50x25+ppvc-mixed-var`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/50x25+ppvc-mixed-var.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| FIFO | 539.4 | 28.4 | 1.1191 | 50/50 | n/a |
| MOR | 519.1 | 32.3 | 1.1229 | 50/50 | n/a |
| SPT | 428.1 | 8.9 | 1.1126 | 50/50 | n/a |
| MWKR | 509.0 | 34.4 | 1.1218 | 50/50 | n/a |
| SPTL | 451.9 | 15.0 | 1.1187 | 50/50 | n/a |
| LLF | 430.4 | 8.6 | 1.1373 | 50/50 | n/a |
| MWKRL | 492.3 | 21.2 | 1.1238 | 50/50 | n/a |
| LRPTL | 491.1 | 20.9 | 1.1236 | 50/50 | n/a |
