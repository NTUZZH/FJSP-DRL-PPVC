# PPVC ablation A0 (lag-blind + right-shift repair) — 10x25+ppvc-mixed+a0-2k-s301

- dataset: `30x25+ppvc-mixed`  (20 instances)
- checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a0-2k-s301.pth`
- seed_test: 50
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False, fea_j_input_dim=10, n_op_types=5, n_mch_types=9
- method: lag-blind greedy plan (env time_lag_list=None) -> right-shift repair vs TRUE lags -> validated vs TRUE lags

| metric | value |
|---|---|
| mean repaired makespan (A0 score) | 582.1 |
| std repaired makespan | 32.1 |
| mean lag-blind plan makespan | 233.2 |
| std lag-blind plan makespan | 8.0 |
| mean plan-vs-reality inflation % | 149.51% |
| feasible (repaired vs TRUE lags) | 20/20 |
| mean time (rollout + repair, s) | 3.3015 |

Inflation % = 100 * (repaired - lag-blind) / lag-blind, the cost of ignoring lags at plan time then repairing.
