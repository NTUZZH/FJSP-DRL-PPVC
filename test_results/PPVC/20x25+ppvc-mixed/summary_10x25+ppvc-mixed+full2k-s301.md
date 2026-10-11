# PPVC evaluation summary — 10x25+ppvc-mixed+full2k-s301

- dataset: `20x25+ppvc-mixed`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full2k-s301.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/20x25+ppvc-mixed.jsonl` (coverage 50/50 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 258.0 | 6.8 | 3.3469 | 50/50 | 1.18% (50) |
