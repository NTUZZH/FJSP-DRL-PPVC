# Lag-buffer peak occupancy: 30x25+ppvc-mixed

- window: [c_i, c_i + lag_i) for every non-terminal op with lag > 0; peak = most windows sharing one instant
- every schedule passed schedule_validator


| method | n | peak mean | median | max | curing max | ponding max | drying max | mean makespan |
|---|---|---|---|---|---|---|---|---|
| greedy+10x25+ppvc-mixed+full | 20 | 17.70 | 18.5 | 21 | 11 | 12 | 21 | 342.6 |
| SPT | 20 | 15.20 | 15.0 | 17 | 12 | 11 | 14 | 324.9 |

| method | lag type | peak mean | median | max |
|---|---|---|---|---|
| greedy+10x25+ppvc-mixed+full | curing | 9.35 | 10.0 | 11 |
| greedy+10x25+ppvc-mixed+full | ponding | 10.05 | 10.0 | 12 |
| greedy+10x25+ppvc-mixed+full | drying | 17.50 | 18.5 | 21 |
| SPT | curing | 10.10 | 10.0 | 12 |
| SPT | ponding | 9.05 | 9.0 | 11 |
| SPT | drying | 12.05 | 12.0 | 14 |
