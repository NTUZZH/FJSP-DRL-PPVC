# PPVC evaluation summary — mix10-15-20x25+ppvc-mixed+full2k-s301

- dataset: `30x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/mix10-15-20x25+ppvc-mixed+full2k-s301.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/30x25+ppvc-mixed.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 314.9 | 8.5 | 5.2602 | 20/20 | n/a |
