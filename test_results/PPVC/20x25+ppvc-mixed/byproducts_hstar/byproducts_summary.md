# Secondary objectives of makespan schedules: `20x25+ppvc-mixed`

- instances: 50; every schedule re-validated against the true lags
- greedy policy: `10x25+ppvc-mixed+a1lagx05-2k-s301`; dispatching rules seeded as eval_ppvc (seed_test=50 + index)
- due date d_j = 1.5 x CP_j (own lag-inclusive critical path, min processing time; release 0)
- values are means over instances; times in hours

| method | makespan | mean flow | max flow | mean WIP | util. (mean) | lag hold | queue wait | total tard. | max tard. | tardy modules |
|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 254.5 | 188.6 | 239.8 | 14.82 | 0.276 | 1632.0 | 384.4 | 104.2 | 43.5 | 4.12 |
| SPT | 265.1 | 207.3 | 257.1 | 15.64 | 0.266 | 1632.0 | 753.9 | 61.3 | 21.6 | 3.70 |
| LLF | 265.0 | 207.8 | 257.6 | 15.69 | 0.266 | 1632.0 | 766.0 | 63.5 | 23.4 | 3.76 |

## Station utilization by type (mean over instances)

| method | A | B | C | D | E | F | G | H | Q |
|---|---|---|---|---|---|---|---|---|---|
| greedy | 0.215 | 0.051 | 0.286 | 0.413 | 0.212 | 0.385 | 0.198 | 0.332 | 0.235 |
| SPT | 0.204 | 0.049 | 0.278 | 0.402 | 0.202 | 0.372 | 0.190 | 0.312 | 0.225 |
| LLF | 0.203 | 0.049 | 0.278 | 0.403 | 0.202 | 0.370 | 0.190 | 0.312 | 0.225 |
