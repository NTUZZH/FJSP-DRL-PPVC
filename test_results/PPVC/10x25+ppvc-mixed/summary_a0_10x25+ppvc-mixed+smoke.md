# PPVC ablation A0 (lag-blind + right-shift repair) — 10x25+ppvc-mixed+smoke

- dataset: `10x25+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+smoke.pth`
- seed_test: 50
- architecture (from snapshot): use_lag_features=True, use_type_embedding=True, fea_j_input_dim=12, n_op_types=5, n_mch_types=9
- method: lag-blind greedy plan (env time_lag_list=None) -> right-shift repair vs TRUE lags -> validated vs TRUE lags

| metric | value |
|---|---|
| mean repaired makespan (A0 score) | 339.9 |
| std repaired makespan | 23.4 |
| mean lag-blind plan makespan | 162.5 |
| std lag-blind plan makespan | 13.4 |
| mean plan-vs-reality inflation % | 110.25% |
| feasible (repaired vs TRUE lags) | 100/100 |
| mean time (rollout + repair, s) | 2.1196 |

Inflation % = 100 * (repaired - lag-blind) / lag-blind, the cost of ignoring lags at plan time then repairing.
