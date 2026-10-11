# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `20x25+ppvc-mixed`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/20x25+ppvc-mixed.jsonl` (coverage 17/50 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 272.6 | 10.3 | 3.8053 | 50/50 | 6.73% (17) |
| FIFO | 314.5 | 10.7 | 1.4804 | 50/50 | 23.58% (17) |
| MOR | 329.0 | 19.7 | 1.4528 | 50/50 | 29.12% (17) |
| SPT | 265.1 | 7.0 | 1.4322 | 50/50 | 4.23% (17) |
| MWKR | 313.1 | 20.1 | 1.5170 | 50/50 | 22.43% (17) |
