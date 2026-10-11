# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `5x9+ppvc-mixed`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/5x9+ppvc-mixed.jsonl` (coverage 50/50 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 271.0 | 18.5 | 0.9750 | 50/50 | 14.18% (50) |
| FIFO | 295.4 | 21.1 | 0.3260 | 50/50 | 24.42% (50) |
| MOR | 310.2 | 25.6 | 0.3177 | 50/50 | 30.66% (50) |
| SPT | 256.2 | 12.5 | 0.3114 | 50/50 | 7.92% (50) |
| MWKR | 291.9 | 21.8 | 0.3132 | 50/50 | 22.99% (50) |
