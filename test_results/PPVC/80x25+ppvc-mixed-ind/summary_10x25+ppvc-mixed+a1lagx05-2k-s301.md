# PPVC evaluation summary — 10x25+ppvc-mixed+a1lagx05-2k-s301

- dataset: `80x25+ppvc-mixed-ind`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=False, use_type_embedding=False, fea_j_input_dim=10, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/80x25+ppvc-mixed-ind.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 567.8 | 8.7 | 16.9953 | 50/50 | n/a |
