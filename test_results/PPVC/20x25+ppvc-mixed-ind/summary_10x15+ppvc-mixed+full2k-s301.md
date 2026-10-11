# PPVC evaluation summary — 10x15+ppvc-mixed+full2k-s301

- dataset: `20x25+ppvc-mixed-ind`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x15+ppvc-mixed+full2k-s301.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/20x25+ppvc-mixed-ind.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 254.8 | 5.1 | 4.3986 | 50/50 | n/a |
