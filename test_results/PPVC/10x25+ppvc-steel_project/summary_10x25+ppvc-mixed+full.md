# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `10x25+ppvc-steel_project`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/10x25+ppvc-steel_project.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| greedy | 221.7 | 11.2 | 1.9471 | 50/50 | n/a |
| FIFO | 259.6 | 6.4 | 0.6673 | 50/50 | n/a |
| MOR | 266.5 | 10.8 | 0.6607 | 50/50 | n/a |
| SPT | 235.9 | 9.1 | 0.6265 | 50/50 | n/a |
| MWKR | 242.8 | 9.8 | 0.6019 | 50/50 | n/a |
