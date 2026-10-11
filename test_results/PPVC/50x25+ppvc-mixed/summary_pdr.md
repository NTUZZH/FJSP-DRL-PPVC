# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `50x25+ppvc-mixed`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/50x25+ppvc-mixed.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| FIFO | 592.5 | 38.0 | 2.2814 | 50/50 | n/a |
| MOR | 748.7 | 107.1 | 2.2106 | 50/50 | n/a |
| SPT | 449.0 | 7.7 | 2.2270 | 50/50 | n/a |
| MWKR | 582.4 | 54.2 | 2.1136 | 50/50 | n/a |
| SPTL | 465.5 | 10.8 | 2.0702 | 50/50 | n/a |
| LLF | 450.8 | 7.7 | 2.1399 | 50/50 | n/a |
| MWKRL | 551.4 | 26.6 | 2.1068 | 50/50 | n/a |
| LRPTL | 551.0 | 24.5 | 2.1962 | 50/50 | n/a |
