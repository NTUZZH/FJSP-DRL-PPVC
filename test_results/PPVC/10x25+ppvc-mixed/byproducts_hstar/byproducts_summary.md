# Secondary objectives of makespan schedules: `10x25+ppvc-mixed`

- instances: 100; every schedule re-validated against the true lags
- greedy policy: `10x25+ppvc-mixed+a1lagx05-2k-s301`; dispatching rules seeded as eval_ppvc (seed_test=50 + index)
- due date d_j = 1.5 x CP_j (own lag-inclusive critical path, min processing time; release 0)
- values are means over instances; times in hours

| method | makespan | mean flow | max flow | mean WIP | util. (mean) | lag hold | queue wait | total tard. | max tard. | tardy modules |
|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 210.9 | 169.3 | 206.8 | 8.03 | 0.166 | 790.6 | 28.8 | 1.0 | 1.0 | 0.14 |
| SPT | 216.0 | 177.6 | 211.4 | 8.23 | 0.161 | 790.6 | 115.3 | 0.2 | 0.2 | 0.04 |
| LLF | 215.7 | 177.7 | 211.2 | 8.24 | 0.162 | 790.6 | 115.8 | 0.2 | 0.2 | 0.04 |

## Station utilization by type (mean over instances)

| method | A | B | C | D | E | F | G | H | Q |
|---|---|---|---|---|---|---|---|---|---|
| greedy | 0.121 | 0.028 | 0.169 | 0.243 | 0.130 | 0.237 | 0.118 | 0.215 | 0.144 |
| SPT | 0.118 | 0.028 | 0.168 | 0.237 | 0.125 | 0.228 | 0.115 | 0.207 | 0.139 |
| LLF | 0.118 | 0.028 | 0.168 | 0.237 | 0.125 | 0.229 | 0.116 | 0.207 | 0.139 |
