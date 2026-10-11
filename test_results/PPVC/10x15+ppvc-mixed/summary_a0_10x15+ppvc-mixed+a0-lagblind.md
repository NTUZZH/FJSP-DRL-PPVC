# PPVC ablation A0 (lag-blind + right-shift repair) — 10x15+ppvc-mixed+a0-lagblind

- dataset: `10x15+ppvc-mixed`  (100 instances)
- checkpoint: `./trained_network/PPVC/10x15+ppvc-mixed+a0-lagblind.pth`
- seed_test: 50
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False, fea_j_input_dim=10, n_op_types=5, n_mch_types=9
- method: lag-blind greedy plan (env time_lag_list=None) -> right-shift repair vs TRUE lags -> validated vs TRUE lags

| metric | value |
|---|---|
| mean repaired makespan (A0 score) | 421.4 |
| std repaired makespan | 43.3 |
| mean lag-blind plan makespan | 187.2 |
| std lag-blind plan makespan | 12.0 |
| mean plan-vs-reality inflation % | 124.94% |
| feasible (repaired vs TRUE lags) | 100/100 |
| mean time (rollout + repair, s) | 2.0699 |

Inflation % = 100 * (repaired - lag-blind) / lag-blind, the cost of ignoring lags at plan time then repairing.
