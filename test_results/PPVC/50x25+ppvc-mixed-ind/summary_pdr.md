# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `50x25+ppvc-mixed-ind`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/50x25+ppvc-mixed-ind.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| SPT | 448.8 | 10.1 | 1.2841 | 50/50 | n/a |
| LLF | 450.7 | 9.0 | 1.3144 | 50/50 | n/a |
| MWKR | 579.2 | 52.0 | 1.2932 | 50/50 | n/a |
| MOR | 713.7 | 115.4 | 1.2917 | 50/50 | n/a |
| FIFO | 592.7 | 39.5 | 1.2881 | 50/50 | n/a |
| SPTL | 463.7 | 11.2 | 1.2918 | 50/50 | n/a |
| MWKRL | 553.4 | 29.5 | 1.2937 | 50/50 | n/a |
| LRPTL | 551.3 | 28.7 | 1.2959 | 50/50 | n/a |
