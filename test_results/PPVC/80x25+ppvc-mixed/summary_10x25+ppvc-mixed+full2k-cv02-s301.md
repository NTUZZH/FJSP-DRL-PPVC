# PPVC evaluation summary — 10x25+ppvc-mixed+full2k-cv02-s301

- dataset: `80x25+ppvc-mixed`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full2k-cv02-s301.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/80x25+ppvc-mixed.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 589.0 | 8.4 | 15.4907 | 50/50 | n/a |
