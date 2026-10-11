# Lag-buffer peak occupancy: 10x25+ppvc-mixed

- window: [c_i, c_i + lag_i) for every non-terminal op with lag > 0; peak = most windows sharing one instant
- every schedule passed schedule_validator


| method | n | peak mean | median | max | curing max | ponding max | drying max | mean makespan |
|---|---|---|---|---|---|---|---|---|
| greedy+10x25+ppvc-mixed+full | 100 | 8.94 | 9.0 | 10 | 8 | 8 | 10 | 210.9 |
| SPT | 100 | 8.90 | 9.0 | 10 | 8 | 9 | 9 | 216.0 |

| method | lag type | peak mean | median | max |
|---|---|---|---|---|
| greedy+10x25+ppvc-mixed+full | curing | 4.86 | 5.0 | 8 |
| greedy+10x25+ppvc-mixed+full | ponding | 5.49 | 6.0 | 8 |
| greedy+10x25+ppvc-mixed+full | drying | 7.31 | 7.0 | 10 |
| SPT | curing | 4.88 | 5.0 | 8 |
| SPT | ponding | 5.90 | 6.0 | 9 |
| SPT | drying | 6.83 | 7.0 | 9 |
