# PPVC evaluation summary — mix10-20-30x25+ppvc-mixed+full-sizemix

- dataset: `20x25+ppvc-mixed`  (50 instances)
- checkpoint: `./trained_network/PPVC/mix10-20-30x25+ppvc-mixed+full-sizemix.pth`
- seed_test: 50  |  sample_times: 100
- architecture (from snapshot): use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/20x25+ppvc-mixed.jsonl` (coverage 50/50 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| sampling | 253.9 | 5.9 | 54.1539 | 50/50 | -0.42% (50) |
