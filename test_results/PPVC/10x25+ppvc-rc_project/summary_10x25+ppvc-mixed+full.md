# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `10x25+ppvc-rc_project`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/10x25+ppvc-rc_project.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 231.5 | 8.6 | 2.0060 | 50/50 | n/a |
| FIFO | 258.7 | 6.3 | 0.7003 | 50/50 | n/a |
| MOR | 263.1 | 9.8 | 0.6691 | 50/50 | n/a |
| SPT | 235.8 | 7.0 | 0.6688 | 50/50 | n/a |
| MWKR | 247.7 | 10.3 | 0.6477 | 50/50 | n/a |
