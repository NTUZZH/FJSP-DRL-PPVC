# PPVC evaluation summary — 10x25+ppvc-mixed+full2k-s302

- dataset: `50x25+ppvc-mixed-ind`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full2k-s302.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/50x25+ppvc-mixed-ind.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 443.5 | 16.1 | 8.9266 | 50/50 | n/a |
