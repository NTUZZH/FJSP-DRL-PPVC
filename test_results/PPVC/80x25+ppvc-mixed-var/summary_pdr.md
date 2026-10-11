# PPVC evaluation summary — 10x25+ppvc-mixed+full

- dataset: `80x25+ppvc-mixed-var`  (50 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+full.pth`
- seed_test: 50  
- architecture (from snapshot): framework=DANIEL, use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- CP-SAT reference: not available (`./or_solution/PPVC/80x25+ppvc-mixed-var.jsonl` missing/empty) — gap% omitted

| method | mean makespan | std | mean time (s) | feasible | mean gap% vs CP-SAT (cov) |
|---|---|---|---|---|---|
| FIFO | 823.4 | 48.0 | 2.3927 | 50/50 | n/a |
| MOR | 771.3 | 30.9 | 2.3980 | 50/50 | n/a |
| SPT | 594.5 | 10.8 | 2.3851 | 50/50 | n/a |
| MWKR | 739.1 | 26.3 | 2.3981 | 50/50 | n/a |
| SPTL | 622.4 | 17.8 | 2.3967 | 50/50 | n/a |
| LLF | 597.6 | 10.9 | 2.4270 | 50/50 | n/a |
| MWKRL | 747.2 | 40.6 | 2.3996 | 50/50 | n/a |
| LRPTL | 743.2 | 35.7 | 2.3997 | 50/50 | n/a |
