# PPVC evaluation summary — 10x25+ppvc-mixed+a0-2k-s303

- dataset: `30x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a0-2k-s303.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=False, use_type_embedding=False, fea_j_input_dim=10, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/30x25+ppvc-mixed.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 314.5 | 9.1 | 3.9876 | 20/20 | n/a |
