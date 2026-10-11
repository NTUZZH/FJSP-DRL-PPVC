# PPVC evaluation summary — 10x25+ppvc-mixed+a1nolagbound-2k-s302

- dataset: `50x25+ppvc-mixed`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1nolagbound-2k-s302.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=False, use_type_embedding=False, fea_j_input_dim=10, n_op_types=5, n_mch_types=9, bound_lag_free=True (model rollouts)
- CP-SAT reference: not available (`./or_solution/PPVC/50x25+ppvc-mixed.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 430.0 | 11.3 | 6.0310 | 50/50 | n/a |
