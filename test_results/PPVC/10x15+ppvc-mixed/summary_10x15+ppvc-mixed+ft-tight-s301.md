# PPVC evaluation summary — 10x15+ppvc-mixed+ft-tight-s301

- dataset: `10x15+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x15+ppvc-mixed+ft-tight-s301.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/10x15+ppvc-mixed.jsonl` (coverage 100/100 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 253.8 | 13.0 | 1.3331 | 100/100 | 6.47% (100) |
