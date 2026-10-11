# PPVC evaluation summary — 10x25+ppvc-mixed+plan-a1lagx05-2k-s303

- dataset: `20x25+ppvc-mixed`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+plan-a1lagx05-2k-s303.pth`
- seed_test: 50  
- architecture (from snapshot): framework=PLAN, use_lag_features=False, use_type_embedding=False, fea_j_input_dim=10, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/20x25+ppvc-mixed.jsonl` (coverage 50/50 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 257.6 | 9.0 | 1.0345 | 50/50 | 1.03% (50) |
