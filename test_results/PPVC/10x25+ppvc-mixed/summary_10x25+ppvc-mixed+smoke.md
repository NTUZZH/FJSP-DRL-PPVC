# PPVC evaluation summary — 10x25+ppvc-mixed+smoke

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+smoke.pth`
- seed_test: 50  
- architecture (from snapshot): use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/10x25+ppvc-mixed.jsonl` (coverage 8/100 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 228.4 | 10.5 | 2.8527 | 100/100 | 10.56% (8) |
| FIFO | 231.2 | 9.3 | 1.3801 | 100/100 | 14.99% (8) |
| MOR | 233.8 | 11.7 | 1.3279 | 100/100 | 17.16% (8) |
| SPT | 216.0 | 8.5 | 1.3436 | 100/100 | 6.92% (8) |
| MWKR | 224.7 | 9.6 | 1.3536 | 100/100 | 10.47% (8) |
