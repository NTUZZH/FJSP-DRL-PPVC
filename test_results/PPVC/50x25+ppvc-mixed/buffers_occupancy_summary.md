# Lag-buffer peak occupancy: 50x25+ppvc-mixed

- window: [c_i, c_i + lag_i) for every non-terminal op with lag > 0; peak = most windows sharing one instant
- every schedule passed schedule_validator


| method | n | peak mean | median | max | curing max | ponding max | drying max | mean makespan |
|---|---|---|---|---|---|---|---|---|
| greedy+10x25+ppvc-mixed+full | 50 | 20.20 | 20.0 | 24 | 12 | 13 | 20 | 445.4 |
| SPT | 50 | 18.08 | 17.0 | 24 | 13 | 13 | 16 | 449.0 |

| method | lag type | peak mean | median | max |
|---|---|---|---|---|
| greedy+10x25+ppvc-mixed+full | curing | 9.70 | 10.0 | 12 |
| greedy+10x25+ppvc-mixed+full | ponding | 11.58 | 12.0 | 13 |
| greedy+10x25+ppvc-mixed+full | drying | 16.80 | 17.0 | 20 |
| SPT | curing | 10.74 | 11.0 | 13 |
| SPT | ponding | 10.72 | 11.0 | 13 |
| SPT | drying | 13.42 | 13.0 | 16 |
