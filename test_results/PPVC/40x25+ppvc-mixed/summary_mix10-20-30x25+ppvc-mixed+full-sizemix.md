# PPVC evaluation summary — mix10-20-30x25+ppvc-mixed+full-sizemix

- dataset: `40x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/mix10-20-30x25+ppvc-mixed+full-sizemix.pth`
- seed_test: 50  |  sample_times: 100
- architecture (from snapshot): use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/40x25+ppvc-mixed.jsonl` (coverage 12/20 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| sampling | 390.1 | 10.4 | 146.0941 | 20/20 | 0.66% (12) |
