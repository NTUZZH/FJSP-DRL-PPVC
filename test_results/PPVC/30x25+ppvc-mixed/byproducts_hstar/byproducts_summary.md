# Secondary objectives of makespan schedules: `30x25+ppvc-mixed`

- instances: 20; every schedule re-validated against the true lags
- greedy policy: `10x25+ppvc-mixed+a1lagx05-2k-s301`; dispatching rules seeded as eval_ppvc (seed_test=50 + index)
- due date d_j = 1.5 x CP_j (own lag-inclusive critical path, min processing time; release 0)
- values are means over instances; times in hours

| method | makespan | mean flow | max flow | mean WIP | util. (mean) | lag hold | queue wait | total tard. | max tard. | tardy modules |
|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 310.4 | 202.6 | 289.1 | 19.59 | 0.338 | 2386.9 | 1071.2 | 685.6 | 104.8 | 15.00 |
| SPT | 324.9 | 242.5 | 318.4 | 22.40 | 0.325 | 2386.9 | 2250.9 | 784.9 | 90.4 | 17.45 |
| LLF | 326.0 | 243.7 | 320.1 | 22.42 | 0.325 | 2386.9 | 2279.9 | 815.3 | 90.8 | 17.90 |

## Station utilization by type (mean over instances)

| method | A | B | C | D | E | F | G | H | Q |
|---|---|---|---|---|---|---|---|---|---|
| greedy | 0.239 | 0.055 | 0.348 | 0.501 | 0.268 | 0.475 | 0.240 | 0.440 | 0.291 |
| SPT | 0.223 | 0.053 | 0.338 | 0.491 | 0.256 | 0.458 | 0.229 | 0.413 | 0.278 |
| LLF | 0.222 | 0.053 | 0.338 | 0.491 | 0.255 | 0.459 | 0.228 | 0.411 | 0.278 |
