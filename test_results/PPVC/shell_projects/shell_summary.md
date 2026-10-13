# Single-shell project orders: SLAB against the dispatching rules

All schedules built on the CPU and re-validated against the true waits (schedule_validator.validate_schedule). SLAB = DANIEL policy `10x25+ppvc-mixed+a1lagx05-2k-s{301,302,303}`, greedy decoding. Rules in the SLAB transition, tie-break seed 50 + order index. Differences are SLAB minus rival (h), 95% percentile bootstrap CI from 2000 resamples (seed 0), two-sided Wilcoxon signed-rank p with zero differences dropped, rank-biserial r (r < 0: SLAB shorter). Counts are SLAB shorter / equal / longer.

## 10x25+ppvc-rc_project-ind (50 orders, 10 modules, rc_project, seeds 740000-740049)

| Method | Mean makespan (h) |
|---|---:|
| SLAB s301 | 229.60 |
| SLAB s302 | 231.70 |
| SLAB s303 | 230.36 |
| SLAB seed mean | 230.55 |
| SLAB seed std (population, of the seed means) | 0.87 |
| FIFO | 258.44 |
| MOR | 263.46 |
| SPT | 237.60 |
| MWKR | 249.96 |
| SPTL | 237.82 |
| LLF | 237.36 |
| MWKRL | 241.48 |
| LRPTL | 240.30 |

| Comparison | SLAB mean | Rival mean | Diff (h) | 95% CI | % | shorter/equal/longer | Wilcoxon p | r |
|---|---:|---:|---:|---|---:|:-:|---:|---:|
| seed mean vs SPT | 230.55 | 237.60 | -7.05 | [-9.20, -5.17] | -2.97 | 41/0/9 | 2.87e-07 | -0.83 |
| seed mean vs LLF | 230.55 | 237.36 | -6.81 | [-8.95, -4.86] | -2.87 | 39/0/11 | 7.85e-07 | -0.80 |
| s301 vs SPT | 229.60 | 237.60 | -8.00 | [-10.10, -5.96] | -3.37 | 44/1/5 | 6.68e-08 | -0.89 |
| s301 vs LLF | 229.60 | 237.36 | -7.76 | [-9.90, -5.64] | -3.27 | 42/2/6 | 1.45e-07 | -0.87 |
| s302 vs SPT | 231.70 | 237.60 | -5.90 | [-8.16, -3.90] | -2.48 | 36/5/9 | 2.34e-05 | -0.72 |
| s302 vs LLF | 231.70 | 237.36 | -5.66 | [-7.98, -3.58] | -2.38 | 35/3/12 | 5.43e-05 | -0.68 |
| s303 vs SPT | 230.36 | 237.60 | -7.24 | [-9.50, -5.10] | -3.05 | 40/2/8 | 8.10e-07 | -0.82 |
| s303 vs LLF | 230.36 | 237.36 | -7.00 | [-9.30, -4.86] | -2.95 | 40/1/9 | 1.75e-06 | -0.78 |
| seed mean vs best rule (LLF) | 230.55 | 237.36 | -6.81 | [-8.95, -4.86] | -2.87 | 39/0/11 | 7.85e-07 | -0.80 |

## 10x25+ppvc-steel_project-ind (50 orders, 10 modules, steel_project, seeds 750000-750049)

| Method | Mean makespan (h) |
|---|---:|
| SLAB s301 | 225.76 |
| SLAB s302 | 221.88 |
| SLAB s303 | 225.80 |
| SLAB seed mean | 224.48 |
| SLAB seed std (population, of the seed means) | 1.84 |
| FIFO | 259.90 |
| MOR | 268.86 |
| SPT | 234.78 |
| MWKR | 245.92 |
| SPTL | 235.18 |
| LLF | 235.08 |
| MWKRL | 230.58 |
| LRPTL | 229.36 |

| Comparison | SLAB mean | Rival mean | Diff (h) | 95% CI | % | shorter/equal/longer | Wilcoxon p | r |
|---|---:|---:|---:|---|---:|:-:|---:|---:|
| seed mean vs SPT | 224.48 | 234.78 | -10.30 | [-13.24, -7.38] | -4.39 | 42/2/6 | 4.98e-08 | -0.90 |
| seed mean vs LLF | 224.48 | 235.08 | -10.60 | [-13.51, -7.73] | -4.51 | 42/1/7 | 3.18e-08 | -0.91 |
| s301 vs SPT | 225.76 | 234.78 | -9.02 | [-12.62, -5.56] | -3.84 | 37/1/12 | 1.27e-05 | -0.72 |
| s301 vs LLF | 225.76 | 235.08 | -9.32 | [-12.98, -5.90] | -3.96 | 37/1/12 | 5.42e-06 | -0.75 |
| s302 vs SPT | 221.88 | 234.78 | -12.90 | [-16.44, -9.48] | -5.49 | 43/0/7 | 4.37e-08 | -0.89 |
| s302 vs LLF | 221.88 | 235.08 | -13.20 | [-16.74, -9.76] | -5.62 | 43/0/7 | 3.51e-08 | -0.90 |
| s303 vs SPT | 225.80 | 234.78 | -8.98 | [-11.78, -5.98] | -3.82 | 39/1/10 | 1.70e-06 | -0.79 |
| s303 vs LLF | 225.80 | 235.08 | -9.28 | [-12.08, -6.32] | -3.95 | 39/3/8 | 1.00e-06 | -0.82 |
| seed mean vs best rule (LRPTL) | 224.48 | 229.36 | -4.88 | [-7.50, -2.33] | -2.13 | 33/3/14 | 5.70e-04 | -0.58 |

## 50x25+ppvc-rc_project (50 orders, 50 modules, rc_project, seeds 760000-760049)

| Method | Mean makespan (h) |
|---|---:|
| SLAB s301 | 443.38 |
| SLAB s302 | 458.54 |
| SLAB s303 | 445.46 |
| SLAB seed mean | 449.13 |
| SLAB seed std (population, of the seed means) | 6.71 |
| FIFO | 751.70 |
| MOR | 930.84 |
| SPT | 458.30 |
| MWKR | 695.62 |
| SPTL | 478.12 |
| LLF | 459.74 |
| MWKRL | 676.56 |
| LRPTL | 669.74 |

| Comparison | SLAB mean | Rival mean | Diff (h) | 95% CI | % | shorter/equal/longer | Wilcoxon p | r |
|---|---:|---:|---:|---|---:|:-:|---:|---:|
| seed mean vs SPT | 449.13 | 458.30 | -9.17 | [-11.03, -7.30] | -2.00 | 43/1/6 | 1.73e-08 | -0.92 |
| seed mean vs LLF | 449.13 | 459.74 | -10.61 | [-12.63, -8.55] | -2.31 | 47/0/3 | 8.98e-09 | -0.93 |
| s301 vs SPT | 443.38 | 458.30 | -14.92 | [-17.86, -12.00] | -3.26 | 47/0/3 | 1.09e-08 | -0.93 |
| s301 vs LLF | 443.38 | 459.74 | -16.36 | [-19.58, -13.06] | -3.56 | 47/0/3 | 1.81e-08 | -0.91 |
| s302 vs SPT | 458.54 | 458.30 | +0.24 | [-2.06, +2.56] | +0.05 | 24/1/25 | 9.33e-01 | +0.01 |
| s302 vs LLF | 458.54 | 459.74 | -1.20 | [-3.46, +1.08] | -0.26 | 26/0/24 | 3.95e-01 | -0.14 |
| s303 vs SPT | 445.46 | 458.30 | -12.84 | [-15.30, -10.28] | -2.80 | 45/1/4 | 2.81e-08 | -0.91 |
| s303 vs LLF | 445.46 | 459.74 | -14.28 | [-16.68, -11.62] | -3.11 | 46/0/4 | 1.02e-08 | -0.93 |
| seed mean vs best rule (SPT) | 449.13 | 458.30 | -9.17 | [-11.03, -7.30] | -2.00 | 43/1/6 | 1.73e-08 | -0.92 |

## 50x25+ppvc-steel_project (50 orders, 50 modules, steel_project, seeds 770000-770049)

| Method | Mean makespan (h) |
|---|---:|
| SLAB s301 | 570.70 |
| SLAB s302 | 554.32 |
| SLAB s303 | 563.88 |
| SLAB seed mean | 562.97 |
| SLAB seed std (population, of the seed means) | 6.72 |
| FIFO | 768.92 |
| MOR | 893.56 |
| SPT | 579.78 |
| MWKR | 668.62 |
| SPTL | 587.22 |
| LLF | 579.28 |
| MWKRL | 668.64 |
| LRPTL | 662.94 |

| Comparison | SLAB mean | Rival mean | Diff (h) | 95% CI | % | shorter/equal/longer | Wilcoxon p | r |
|---|---:|---:|---:|---|---:|:-:|---:|---:|
| seed mean vs SPT | 562.97 | 579.78 | -16.81 | [-20.15, -13.49] | -2.90 | 46/0/4 | 3.17e-09 | -0.96 |
| seed mean vs LLF | 562.97 | 579.28 | -16.31 | [-19.22, -13.33] | -2.82 | 48/0/2 | 1.46e-09 | -0.98 |
| s301 vs SPT | 570.70 | 579.78 | -9.08 | [-12.68, -5.86] | -1.57 | 37/1/12 | 1.88e-05 | -0.70 |
| s301 vs LLF | 570.70 | 579.28 | -8.58 | [-11.98, -5.32] | -1.48 | 35/1/14 | 2.73e-05 | -0.69 |
| s302 vs SPT | 554.32 | 579.78 | -25.46 | [-29.62, -21.40] | -4.39 | 48/0/2 | 1.11e-09 | -0.99 |
| s302 vs LLF | 554.32 | 579.28 | -24.96 | [-28.64, -21.22] | -4.31 | 50/0/0 | 7.52e-10 | -1.00 |
| s303 vs SPT | 563.88 | 579.78 | -15.90 | [-20.58, -10.92] | -2.74 | 39/0/11 | 1.67e-06 | -0.78 |
| s303 vs LLF | 563.88 | 579.28 | -15.40 | [-19.80, -10.66] | -2.66 | 41/0/9 | 4.99e-07 | -0.82 |
| seed mean vs best rule (LLF) | 562.97 | 579.28 | -16.31 | [-19.22, -13.33] | -2.82 | 48/0/2 | 1.46e-09 | -0.98 |

## 10x25+ppvc-rc_project (50 orders, 10 modules, rc_project, seeds 90000-90049)

| Method | Mean makespan (h) |
|---|---:|
| SLAB s301 | 229.34 |
| SLAB s302 | 231.34 |
| SLAB s303 | 230.68 |
| SLAB seed mean | 230.45 |
| SLAB seed std (population, of the seed means) | 0.83 |
| FIFO | 258.68 |
| MOR | 263.06 |
| SPT | 235.80 |
| MWKR | 247.74 |
| SPTL | 236.16 |
| LLF | 235.80 |
| MWKRL | 239.60 |
| LRPTL | 239.26 |

| Comparison | SLAB mean | Rival mean | Diff (h) | 95% CI | % | shorter/equal/longer | Wilcoxon p | r |
|---|---:|---:|---:|---|---:|:-:|---:|---:|
| seed mean vs SPT | 230.45 | 235.80 | -5.35 | [-7.30, -3.43] | -2.27 | 40/1/9 | 1.05e-05 | -0.72 |
| seed mean vs LLF | 230.45 | 235.80 | -5.35 | [-7.33, -3.36] | -2.27 | 41/0/9 | 1.00e-05 | -0.72 |
| s301 vs SPT | 229.34 | 235.80 | -6.46 | [-8.32, -4.64] | -2.74 | 43/1/6 | 3.60e-07 | -0.83 |
| s301 vs LLF | 229.34 | 235.80 | -6.46 | [-8.36, -4.56] | -2.74 | 43/0/7 | 4.12e-07 | -0.82 |
| s302 vs SPT | 231.34 | 235.80 | -4.46 | [-6.62, -2.36] | -1.89 | 37/2/11 | 5.84e-04 | -0.57 |
| s302 vs LLF | 231.34 | 235.80 | -4.46 | [-6.64, -2.26] | -1.89 | 35/3/12 | 5.64e-04 | -0.58 |
| s303 vs SPT | 230.68 | 235.80 | -5.12 | [-7.30, -3.08] | -2.17 | 39/2/9 | 6.54e-05 | -0.66 |
| s303 vs LLF | 230.68 | 235.80 | -5.12 | [-7.34, -2.96] | -2.17 | 41/1/8 | 9.33e-05 | -0.64 |
| seed mean vs best rule (SPT) | 230.45 | 235.80 | -5.35 | [-7.30, -3.43] | -2.27 | 40/1/9 | 1.05e-05 | -0.72 |

## 10x25+ppvc-steel_project (50 orders, 10 modules, steel_project, seeds 95000-95049)

| Method | Mean makespan (h) |
|---|---:|
| SLAB s301 | 223.80 |
| SLAB s302 | 221.46 |
| SLAB s303 | 227.80 |
| SLAB seed mean | 224.35 |
| SLAB seed std (population, of the seed means) | 2.62 |
| FIFO | 259.56 |
| MOR | 266.54 |
| SPT | 235.88 |
| MWKR | 242.78 |
| SPTL | 236.12 |
| LLF | 235.78 |
| MWKRL | 227.76 |
| LRPTL | 226.38 |

| Comparison | SLAB mean | Rival mean | Diff (h) | 95% CI | % | shorter/equal/longer | Wilcoxon p | r |
|---|---:|---:|---:|---|---:|:-:|---:|---:|
| seed mean vs SPT | 224.35 | 235.88 | -11.53 | [-14.33, -8.51] | -4.89 | 42/0/8 | 9.62e-08 | -0.87 |
| seed mean vs LLF | 224.35 | 235.78 | -11.43 | [-14.27, -8.36] | -4.85 | 42/0/8 | 9.36e-08 | -0.87 |
| s301 vs SPT | 223.80 | 235.88 | -12.08 | [-15.34, -8.44] | -5.12 | 42/0/8 | 5.40e-07 | -0.81 |
| s301 vs LLF | 223.80 | 235.78 | -11.98 | [-15.22, -8.36] | -5.08 | 42/0/8 | 6.23e-07 | -0.81 |
| s302 vs SPT | 221.46 | 235.88 | -14.42 | [-17.68, -11.08] | -6.11 | 44/0/6 | 2.82e-08 | -0.90 |
| s302 vs LLF | 221.46 | 235.78 | -14.32 | [-17.64, -10.88] | -6.07 | 44/0/6 | 3.92e-08 | -0.89 |
| s303 vs SPT | 227.80 | 235.88 | -8.08 | [-11.38, -4.66] | -3.43 | 34/4/12 | 9.31e-05 | -0.66 |
| s303 vs LLF | 227.80 | 235.78 | -7.98 | [-11.34, -4.54] | -3.38 | 36/3/11 | 1.35e-04 | -0.64 |
| seed mean vs best rule (LRPTL) | 224.35 | 226.38 | -2.03 | [-4.61, +0.64] | -0.90 | 30/2/18 | 1.12e-01 | -0.26 |

