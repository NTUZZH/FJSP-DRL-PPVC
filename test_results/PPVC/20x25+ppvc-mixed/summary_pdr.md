# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `20x25+ppvc-mixed`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: `./or_solution/PPVC/20x25+ppvc-mixed.jsonl` (coverage 50/50 instances)

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| FIFO | 314.5 | 10.7 | 0.6207 | 50/50 | 23.39% (50) |
| MOR | 329.0 | 19.7 | 0.6092 | 50/50 | 29.05% (50) |
| SPT | 265.1 | 7.0 | 0.6019 | 50/50 | 3.99% (50) |
| MWKR | 313.1 | 20.1 | 0.6136 | 50/50 | 22.82% (50) |
| SPTL | 285.4 | 11.2 | 0.6320 | 50/50 | 11.95% (50) |
| LLF | 265.0 | 6.6 | 0.6511 | 50/50 | 3.93% (50) |
| MWKRL | 298.2 | 11.7 | 0.6221 | 50/50 | 16.96% (50) |
| LRPTL | 298.7 | 12.9 | 0.6107 | 50/50 | 17.13% (50) |
