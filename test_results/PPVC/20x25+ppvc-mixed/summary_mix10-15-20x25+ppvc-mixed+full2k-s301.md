# PPVC evaluation summary — mix10-15-20x25+ppvc-mixed+full2k-s301

- dataset: `20x25+ppvc-mixed`  (50 instances)
- checkpoint: `./trained_network/PPVC/mix10-15-20x25+ppvc-mixed+full2k-s301.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/20x25+ppvc-mixed.jsonl` (coverage 50/50 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 256.7 | 7.0 | 3.2589 | 50/50 | 0.69% (50) |
