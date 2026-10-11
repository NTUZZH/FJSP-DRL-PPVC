# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `10x25+ppvc-mixed-lagx1.5`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/10x25+ppvc-mixed-lagx1.5.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| SPT | 269.8 | 14.4 | 0.1373 | 100/100 | n/a |
| LLF | 269.7 | 14.5 | 0.1418 | 100/100 | n/a |
