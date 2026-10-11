# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `40x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  |  sample_times: 100
- architecture (from snapshot): use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/40x25+ppvc-mixed.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 402.9 | 21.2 | 19.2184 | 20/20 | n/a |
| sampling | 389.2 | 12.0 | 323.5502 | 20/20 | n/a |
| SPT | 388.2 | 5.8 | 11.1074 | 20/20 | n/a |
| FIFO | 510.1 | 31.2 | 7.3966 | 20/20 | n/a |
| MOR | 606.4 | 82.3 | 8.2583 | 20/20 | n/a |
| MWKR | 499.9 | 46.1 | 8.9020 | 20/20 | n/a |
