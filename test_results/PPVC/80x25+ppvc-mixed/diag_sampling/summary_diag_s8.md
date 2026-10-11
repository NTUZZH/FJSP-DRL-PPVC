# PPVC evaluation summary — mix10-15-20x25+ppvc-mixed+full2k-s301

- dataset: `80x25+ppvc-mixed`  (10 instances)
- checkpoint: `./trained_network/PPVC/mix10-15-20x25+ppvc-mixed+full2k-s301.pth`
- seed_test: 50  |  sample_times: 8
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/80x25+ppvc-mixed.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 639.7 | 9.5 | 11.5932 | 10/10 | n/a |
| sampling | 636.8 | 14.5 | 73.5861 | 10/10 | n/a |
