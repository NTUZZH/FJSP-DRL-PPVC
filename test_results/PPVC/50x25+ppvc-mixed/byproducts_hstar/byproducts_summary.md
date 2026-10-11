# Secondary objectives of makespan schedules: `50x25+ppvc-mixed`

- instances: 50; every schedule re-validated against the true lags
- greedy policy: `10x25+ppvc-mixed+a1lagx05-2k-s301`; dispatching rules seeded as eval_ppvc (seed_test=50 + index)
- due date d_j = 1.5 x CP_j (own lag-inclusive critical path, min processing time; release 0)
- values are means over instances; times in hours

| method | makespan | mean flow | max flow | mean WIP | util. (mean) | lag hold | queue wait | total tard. | max tard. | tardy modules |
|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 412.5 | 237.0 | 387.9 | 28.72 | 0.422 | 4085.6 | 3410.3 | 3732.0 | 201.8 | 37.38 |
| SPT | 449.0 | 309.2 | 439.8 | 34.43 | 0.392 | 4085.6 | 6979.4 | 4712.9 | 208.7 | 41.88 |
| LLF | 450.8 | 311.0 | 442.3 | 34.48 | 0.390 | 4085.6 | 7064.2 | 4801.5 | 211.6 | 42.04 |

## Station utilization by type (mean over instances)

| method | A | B | C | D | E | F | G | H | Q |
|---|---|---|---|---|---|---|---|---|---|
| greedy | 0.321 | 0.076 | 0.436 | 0.620 | 0.333 | 0.596 | 0.303 | 0.518 | 0.363 |
| SPT | 0.289 | 0.070 | 0.407 | 0.590 | 0.307 | 0.551 | 0.278 | 0.465 | 0.335 |
| LLF | 0.288 | 0.070 | 0.405 | 0.592 | 0.306 | 0.549 | 0.277 | 0.463 | 0.334 |
