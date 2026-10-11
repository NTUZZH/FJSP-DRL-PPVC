# PPVC evaluation summary — 10x25+ppvc-mixed+full2k-s301

- dataset: `10x25+ppvc-mixed-var`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full2k-s301.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/10x25+ppvc-mixed-var.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 202.7 | 11.5 | 1.7116 | 100/100 | n/a |
