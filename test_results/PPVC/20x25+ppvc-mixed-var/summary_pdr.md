# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `20x25+ppvc-mixed-var`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/20x25+ppvc-mixed-var.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| SPT | 252.8 | 7.2 | 0.3163 | 50/50 | n/a |
| LLF | 253.1 | 8.5 | 0.3244 | 50/50 | n/a |
| MWKR | 285.9 | 12.5 | 0.3198 | 50/50 | n/a |
| MOR | 284.4 | 12.3 | 0.3205 | 50/50 | n/a |
| FIFO | 296.2 | 12.0 | 0.3192 | 50/50 | n/a |
| SPTL | 276.9 | 14.3 | 0.3179 | 50/50 | n/a |
| MWKRL | 268.1 | 11.7 | 0.3210 | 50/50 | n/a |
| LRPTL | 267.2 | 11.1 | 0.3209 | 50/50 | n/a |
