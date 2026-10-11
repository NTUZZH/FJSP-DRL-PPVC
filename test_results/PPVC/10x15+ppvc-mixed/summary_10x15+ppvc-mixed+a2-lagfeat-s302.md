# PPVC evaluation summary — 10x15+ppvc-mixed+a2-lagfeat-s302

- dataset: `10x15+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x15+ppvc-mixed+a2-lagfeat-s302.pth`
- seed_test: 50  
- architecture (from snapshot): use_lag_features=True, use_type_embedding=False, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/10x15+ppvc-mixed.jsonl` (coverage 100/100 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 256.1 | 13.8 | 2.2882 | 100/100 | 7.45% (100) |
