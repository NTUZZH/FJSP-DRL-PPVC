# PPVC evaluation summary — 10x25+ppvc-mixed+a0-2k-s302

- dataset: `50x25+ppvc-mixed-ind`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a0-2k-s302.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=False, use_type_embedding=False, fea_j_input_dim=10, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/50x25+ppvc-mixed-ind.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 414.4 | 7.9 | 10.2515 | 50/50 | n/a |
