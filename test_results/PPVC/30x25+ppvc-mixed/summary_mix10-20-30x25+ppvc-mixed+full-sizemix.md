# PPVC evaluation summary — mix10-20-30x25+ppvc-mixed+full-sizemix

- dataset: `30x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/mix10-20-30x25+ppvc-mixed+full-sizemix.pth`
- seed_test: 50  
- architecture (from snapshot): use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/30x25+ppvc-mixed.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 326.2 | 10.4 | 5.6350 | 20/20 | n/a |
| SPT | 324.9 | 6.3 | 3.3747 | 20/20 | n/a |
