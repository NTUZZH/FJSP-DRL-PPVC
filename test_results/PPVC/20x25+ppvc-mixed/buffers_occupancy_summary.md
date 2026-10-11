# Lag-buffer peak occupancy: 20x25+ppvc-mixed

- window: [c_i, c_i + lag_i) for every non-terminal op with lag > 0; peak = most windows sharing one instant
- every schedule passed schedule_validator


| method | n | peak mean | median | max | curing max | ponding max | drying max | mean makespan |
|---|---|---|---|---|---|---|---|---|
| greedy+10x25+ppvc-mixed+full | 50 | 15.38 | 15.0 | 18 | 11 | 13 | 18 | 272.4 |
| SPT | 50 | 13.52 | 13.0 | 16 | 12 | 11 | 13 | 265.1 |

| method | lag type | peak mean | median | max |
|---|---|---|---|---|
| greedy+10x25+ppvc-mixed+full | curing | 8.58 | 9.0 | 11 |
| greedy+10x25+ppvc-mixed+full | ponding | 9.22 | 9.0 | 13 |
| greedy+10x25+ppvc-mixed+full | drying | 15.02 | 15.0 | 18 |
| SPT | curing | 8.92 | 9.0 | 12 |
| SPT | ponding | 8.52 | 8.0 | 11 |
| SPT | drying | 10.52 | 11.0 | 13 |
