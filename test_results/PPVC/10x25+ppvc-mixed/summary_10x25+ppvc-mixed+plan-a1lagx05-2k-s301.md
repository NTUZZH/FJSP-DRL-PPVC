# PPVC evaluation summary — 10x25+ppvc-mixed+plan-a1lagx05-2k-s301

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+plan-a1lagx05-2k-s301.pth`
- seed_test: 50  
- architecture (from snapshot): framework=PLAN, use_lag_features=False, use_type_embedding=False, fea_j_input_dim=10, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/10x25+ppvc-mixed.jsonl` (coverage 100/100 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 211.6 | 9.7 | 0.6604 | 100/100 | 4.19% (100) |
