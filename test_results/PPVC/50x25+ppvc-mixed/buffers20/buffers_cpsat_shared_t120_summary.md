# Makespan vs lag-buffer capacity (shared buffer)

- source: `buffers_cpsat_shared_t120.jsonl`; value: CP-SAT objective
- every schedule with a value passed schedule_validator and the occupancy check (peak <= B)
- inflation = value / unconstrained value - 1

| instance | B | status | value | bound | peak | inflation % |
|---|---|---|---|---|---|---|
| instance_000 | free | FEASIBLE | 436 | 248 | 17 | 0.00 |
| instance_000 | 16 | FEASIBLE | 441 | 249 | 16 | 1.15 |
| instance_000 | 12 | FEASIBLE | 456 | 248 | 12 | 4.59 |
| instance_000 | 8 | FEASIBLE | 588 | 249 | 8 | 34.86 |
| instance_000 | 6 | FEASIBLE | 756 | 248 | 6 | 73.39 |
| instance_000 | 4 | FEASIBLE | 1076 | 251 | 4 | 146.79 |
| instance_001 | free | FEASIBLE | 432 | 241 | 18 | 0.00 |
| instance_001 | 16 | FEASIBLE | 431 | 241 | 16 | -0.23 |
| instance_001 | 12 | FEASIBLE | 457 | 242 | 12 | 5.79 |
| instance_001 | 8 | FEASIBLE | 585 | 242 | 8 | 35.42 |
| instance_001 | 6 | FEASIBLE | 727 | 242 | 6 | 68.29 |
| instance_001 | 4 | FEASIBLE | 1061 | 243 | 4 | 145.60 |
| instance_002 | free | FEASIBLE | 453 | 255 | 21 | 0.00 |
| instance_002 | 16 | FEASIBLE | 445 | 254 | 16 | -1.77 |
| instance_002 | 12 | FEASIBLE | 472 | 254 | 12 | 4.19 |
| instance_002 | 8 | FEASIBLE | 593 | 254 | 8 | 30.91 |
| instance_002 | 6 | FEASIBLE | 760 | 254 | 6 | 67.77 |
| instance_002 | 4 | FEASIBLE | 1088 | 255 | 4 | 140.18 |
| instance_003 | free | FEASIBLE | 460 | 261 | 17 | 0.00 |
| instance_003 | 16 | FEASIBLE | 462 | 261 | 16 | 0.43 |
| instance_003 | 12 | FEASIBLE | 495 | 261 | 12 | 7.61 |
| instance_003 | 8 | FEASIBLE | 611 | 261 | 8 | 32.83 |
| instance_003 | 6 | FEASIBLE | 772 | 261 | 6 | 67.83 |
| instance_003 | 4 | FEASIBLE | 1121 | 263 | 4 | 143.70 |
| instance_004 | free | FEASIBLE | 434 | 241 | 19 | 0.00 |
| instance_004 | 16 | FEASIBLE | 437 | 241 | 16 | 0.69 |
| instance_004 | 12 | FEASIBLE | 462 | 241 | 12 | 6.45 |
| instance_004 | 8 | FEASIBLE | 586 | 243 | 8 | 35.02 |
| instance_004 | 6 | FEASIBLE | 737 | 245 | 6 | 69.82 |
| instance_004 | 4 | FEASIBLE | 1068 | 244 | 4 | 146.08 |
| instance_005 | free | FEASIBLE | 453 | 245 | 16 | 0.00 |
| instance_005 | 16 | FEASIBLE | 452 | 246 | 16 | -0.22 |
| instance_005 | 12 | FEASIBLE | 463 | 245 | 12 | 2.21 |
| instance_005 | 8 | FEASIBLE | 586 | 247 | 8 | 29.36 |
| instance_005 | 6 | FEASIBLE | 753 | 245 | 6 | 66.23 |
| instance_005 | 4 | FEASIBLE | 1070 | 251 | 4 | 136.20 |
| instance_006 | free | FEASIBLE | 447 | 254 | 17 | 0.00 |
| instance_006 | 16 | FEASIBLE | 448 | 254 | 16 | 0.22 |
| instance_006 | 12 | FEASIBLE | 470 | 254 | 12 | 5.15 |
| instance_006 | 8 | FEASIBLE | 595 | 253 | 8 | 33.11 |
| instance_006 | 6 | FEASIBLE | 755 | 253 | 6 | 68.90 |
| instance_006 | 4 | FEASIBLE | 1089 | 254 | 4 | 143.62 |
| instance_007 | free | FEASIBLE | 432 | 242 | 16 | 0.00 |
| instance_007 | 16 | FEASIBLE | 432 | 242 | 16 | 0.00 |
| instance_007 | 12 | FEASIBLE | 436 | 242 | 12 | 0.93 |
| instance_007 | 8 | FEASIBLE | 562 | 242 | 8 | 30.09 |
| instance_007 | 6 | FEASIBLE | 711 | 242 | 6 | 64.58 |
| instance_007 | 4 | FEASIBLE | 1017 | 243 | 4 | 135.42 |
| instance_008 | free | FEASIBLE | 446 | 251 | 22 | 0.00 |
| instance_008 | 16 | FEASIBLE | 437 | 251 | 16 | -2.02 |
| instance_008 | 12 | FEASIBLE | 479 | 252 | 12 | 7.40 |
| instance_008 | 8 | FEASIBLE | 602 | 251 | 8 | 34.98 |
| instance_008 | 6 | FEASIBLE | 767 | 251 | 6 | 71.97 |
| instance_008 | 4 | FEASIBLE | 1100 | 253 | 4 | 146.64 |
| instance_009 | free | FEASIBLE | 452 | 243 | 17 | 0.00 |
| instance_009 | 16 | FEASIBLE | 448 | 242 | 16 | -0.88 |
| instance_009 | 12 | FEASIBLE | 457 | 242 | 12 | 1.11 |
| instance_009 | 8 | FEASIBLE | 590 | 243 | 8 | 30.53 |
| instance_009 | 6 | FEASIBLE | 766 | 245 | 6 | 69.47 |
| instance_009 | 4 | FEASIBLE | 1097 | 245 | 4 | 142.70 |
| instance_010 | free | FEASIBLE | 458 | 253 | 18 | 0.00 |
| instance_010 | 16 | FEASIBLE | 461 | 253 | 16 | 0.66 |
| instance_010 | 12 | FEASIBLE | 480 | 253 | 12 | 4.80 |
| instance_010 | 8 | FEASIBLE | 619 | 253 | 8 | 35.15 |
| instance_010 | 6 | FEASIBLE | 773 | 253 | 6 | 68.78 |
| instance_010 | 4 | FEASIBLE | 1112 | 253 | 4 | 142.79 |
| instance_011 | free | FEASIBLE | 436 | 235 | 16 | 0.00 |
| instance_011 | 16 | FEASIBLE | 436 | 236 | 16 | 0.00 |
| instance_011 | 12 | FEASIBLE | 446 | 236 | 12 | 2.29 |
| instance_011 | 8 | FEASIBLE | 567 | 235 | 8 | 30.05 |
| instance_011 | 6 | FEASIBLE | 723 | 237 | 6 | 65.83 |
| instance_011 | 4 | FEASIBLE | 1039 | 240 | 4 | 138.30 |
| instance_012 | free | FEASIBLE | 440 | 249 | 19 | 0.00 |
| instance_012 | 16 | FEASIBLE | 435 | 249 | 16 | -1.14 |
| instance_012 | 12 | FEASIBLE | 479 | 249 | 12 | 8.86 |
| instance_012 | 8 | FEASIBLE | 600 | 249 | 8 | 36.36 |
| instance_012 | 6 | FEASIBLE | 757 | 249 | 6 | 72.05 |
| instance_012 | 4 | FEASIBLE | 1100 | 249 | 4 | 150.00 |
| instance_013 | free | FEASIBLE | 445 | 248 | 23 | 0.00 |
| instance_013 | 16 | FEASIBLE | 442 | 249 | 16 | -0.67 |
| instance_013 | 12 | FEASIBLE | 484 | 248 | 12 | 8.76 |
| instance_013 | 8 | FEASIBLE | 633 | 249 | 8 | 42.25 |
| instance_013 | 6 | FEASIBLE | 810 | 247 | 6 | 82.02 |
| instance_013 | 4 | FEASIBLE | 1167 | 251 | 4 | 162.25 |
| instance_014 | free | FEASIBLE | 442 | 240 | 19 | 0.00 |
| instance_014 | 16 | FEASIBLE | 442 | 240 | 16 | 0.00 |
| instance_014 | 12 | FEASIBLE | 471 | 242 | 12 | 6.56 |
| instance_014 | 8 | FEASIBLE | 592 | 243 | 8 | 33.94 |
| instance_014 | 6 | FEASIBLE | 752 | 242 | 6 | 70.14 |
| instance_014 | 4 | FEASIBLE | 1078 | 243 | 4 | 143.89 |
| instance_015 | free | FEASIBLE | 443 | 253 | 19 | 0.00 |
| instance_015 | 16 | FEASIBLE | 451 | 253 | 16 | 1.81 |
| instance_015 | 12 | FEASIBLE | 476 | 253 | 12 | 7.45 |
| instance_015 | 8 | FEASIBLE | 593 | 253 | 8 | 33.86 |
| instance_015 | 6 | FEASIBLE | 762 | 253 | 6 | 72.01 |
| instance_015 | 4 | FEASIBLE | 1102 | 253 | 4 | 148.76 |
| instance_016 | free | FEASIBLE | 437 | 237 | 17 | 0.00 |
| instance_016 | 16 | FEASIBLE | 437 | 237 | 16 | 0.00 |
| instance_016 | 12 | FEASIBLE | 448 | 237 | 12 | 2.52 |
| instance_016 | 8 | FEASIBLE | 576 | 237 | 8 | 31.81 |
| instance_016 | 6 | FEASIBLE | 732 | 238 | 6 | 67.51 |
| instance_016 | 4 | FEASIBLE | 1049 | 240 | 4 | 140.05 |
| instance_017 | free | FEASIBLE | 439 | 248 | 16 | 0.00 |
| instance_017 | 16 | FEASIBLE | 439 | 248 | 16 | 0.00 |
| instance_017 | 12 | FEASIBLE | 460 | 248 | 12 | 4.78 |
| instance_017 | 8 | FEASIBLE | 585 | 248 | 8 | 33.26 |
| instance_017 | 6 | FEASIBLE | 735 | 248 | 6 | 67.43 |
| instance_017 | 4 | FEASIBLE | 1057 | 250 | 4 | 140.77 |
| instance_018 | free | FEASIBLE | 452 | 258 | 17 | 0.00 |
| instance_018 | 16 | FEASIBLE | 449 | 258 | 16 | -0.66 |
| instance_018 | 12 | FEASIBLE | 471 | 258 | 12 | 4.20 |
| instance_018 | 8 | FEASIBLE | 595 | 258 | 8 | 31.64 |
| instance_018 | 6 | FEASIBLE | 763 | 258 | 6 | 68.81 |
| instance_018 | 4 | FEASIBLE | 1086 | 258 | 4 | 140.27 |
| instance_019 | free | FEASIBLE | 447 | 244 | 20 | 0.00 |
| instance_019 | 16 | FEASIBLE | 449 | 244 | 16 | 0.45 |
| instance_019 | 12 | FEASIBLE | 468 | 244 | 12 | 4.70 |
| instance_019 | 8 | FEASIBLE | 601 | 245 | 8 | 34.45 |
| instance_019 | 6 | FEASIBLE | 753 | 244 | 6 | 68.46 |
| instance_019 | 4 | FEASIBLE | 1093 | 246 | 4 | 144.52 |

| B | n | filled (slack) | missing | mean value | mean inflation % |
|---|---|---|---|---|---|
| 16 | 20 | 0 | 0 | 443.7 | -0.11 |
| 12 | 20 | 0 | 0 | 466.5 | 5.02 |
| 8 | 20 | 0 | 0 | 593.0 | 33.49 |
| 6 | 20 | 0 | 0 | 753.2 | 69.56 |
| 4 | 20 | 0 | 0 | 1083.5 | 143.93 |

Contention evidence: cpu_share = CPU s / (wall s x workers) of each timed solve; a solve is budget-limited when it used >= 95% of its limit, and a budget-limited solve with cpu_share < 0.9 is invalid.

| method | records | timed solves | budget-limited | min cpu_share | cpu_share < 0.9 | no evidence |
|---|---|---|---|---|---|---|
| CP-SAT | 120 | 120 | 120 | 0.993 | 0 | 0 |
