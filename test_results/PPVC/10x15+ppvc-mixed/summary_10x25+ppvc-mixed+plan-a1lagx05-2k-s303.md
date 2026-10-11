# PPVC evaluation summary — 10x25+ppvc-mixed+plan-a1lagx05-2k-s303

- dataset: `10x15+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+plan-a1lagx05-2k-s303.pth`
- seed_test: 50  
- architecture (from snapshot): framework=PLAN, use_lag_features=False, use_type_embedding=False, fea_j_input_dim=10, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/10x15+ppvc-mixed.jsonl` (coverage 100/100 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 259.8 | 13.6 | 0.4695 | 100/100 | 9.04% (100) |
