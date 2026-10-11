# Secondary objectives of makespan schedules: `80x25+ppvc-mixed`

- instances: 50; every schedule re-validated against the true lags
- greedy policy: `10x25+ppvc-mixed+a1lagx05-2k-s301`; dispatching rules seeded as eval_ppvc (seed_test=50 + index)
- due date d_j = 1.5 x CP_j (own lag-inclusive critical path, min processing time; release 0)
- values are means over instances; times in hours

| method | makespan | mean flow | max flow | mean WIP | util. (mean) | lag hold | queue wait | total tard. | max tard. | tardy modules |
|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 566.8 | 278.3 | 502.4 | 39.28 | 0.488 | 6524.6 | 8824.9 | 12836.3 | 369.9 | 66.42 |
| SPT | 620.3 | 390.9 | 609.7 | 50.41 | 0.453 | 6524.6 | 17717.0 | 15436.4 | 384.4 | 70.28 |
| LLF | 623.6 | 393.6 | 612.4 | 50.49 | 0.451 | 6524.6 | 17929.7 | 15651.2 | 386.5 | 70.56 |

## Station utilization by type (mean over instances)

| method | A | B | C | D | E | F | G | H | Q |
|---|---|---|---|---|---|---|---|---|---|
| greedy | 0.375 | 0.088 | 0.506 | 0.709 | 0.388 | 0.688 | 0.353 | 0.600 | 0.423 |
| SPT | 0.338 | 0.081 | 0.474 | 0.680 | 0.356 | 0.636 | 0.323 | 0.539 | 0.389 |
| LLF | 0.336 | 0.081 | 0.470 | 0.681 | 0.355 | 0.632 | 0.321 | 0.535 | 0.387 |
