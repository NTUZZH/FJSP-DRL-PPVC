# Lag-buffer peak occupancy: 80x25+ppvc-mixed

- window: [c_i, c_i + lag_i) for every non-terminal op with lag > 0; peak = most windows sharing one instant
- every schedule passed schedule_validator


| method | n | peak mean | median | max | curing max | ponding max | drying max | mean makespan |
|---|---|---|---|---|---|---|---|---|
| greedy+10x25+ppvc-mixed+full | 50 | 23.22 | 23.0 | 25 | 11 | 14 | 19 | 602.5 |
| SPT | 50 | 22.70 | 23.0 | 27 | 13 | 14 | 16 | 620.3 |

| method | lag type | peak mean | median | max |
|---|---|---|---|---|
| greedy+10x25+ppvc-mixed+full | curing | 9.72 | 10.0 | 11 |
| greedy+10x25+ppvc-mixed+full | ponding | 12.10 | 12.0 | 14 |
| greedy+10x25+ppvc-mixed+full | drying | 15.08 | 15.0 | 19 |
| SPT | curing | 11.04 | 11.0 | 13 |
| SPT | ponding | 11.28 | 11.0 | 14 |
| SPT | drying | 13.68 | 14.0 | 16 |
