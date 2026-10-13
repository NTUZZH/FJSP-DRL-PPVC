# CP-SAT time to reach the SLAB makespan (anytime logs, four workers)

Target (a): per-order mean of SLAB seeds 301-303; target (b): SLAB seed 301 alone (stored greedy results). Time = first solver time at which the incumbent (running minimum of the logged trajectory; SPT-started runs log the SPT hint at t = 0) is at most the target. Median over all orders counts an order never reached as +inf ("> budget").

## (a) SLAB seed mean

| Modules | Mode | Budget (s) | n | <=1 s | <=5 s | <=60 s | <=300 s | <=600 s | <=1800 s | reached | median, all (s) | median, reached (s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | cold | 300 | 100 | 36% (36) | 100% (100) | 100% (100) | 100% (100) | -- | -- | 100/100 | 1.2 | 1.2 |
| 10 | warm-spt | 300 | 100 | 89% (89) | 100% (100) | 100% (100) | 100% (100) | -- | -- | 100/100 | 0.2 | 0.2 |
| 20 | cold | 300 | 50 | 0% (0) | 0% (0) | 12% (6) | 64% (32) | -- | -- | 32/50 | 186.6 | 145.3 |
| 20 | warm-spt | 300 | 50 | 12% (6) | 12% (6) | 28% (14) | 64% (32) | -- | -- | 32/50 | 198.6 | 73.7 |
| 30 | cold | 300 | 20 | 0% (0) | 0% (0) | 0% (0) | 5% (1) | -- | -- | 1/20 | >budget | 122.4 |
| 30 | warm-spt | 300 | 20 | 5% (1) | 5% (1) | 5% (1) | 10% (2) | -- | -- | 2/20 | >budget | 48.4 |
| 50 | cold | 600 | 50 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | -- | 0/50 | >budget | >budget |
| 50 | warm-spt | 600 | 50 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | -- | 0/50 | >budget | >budget |
| 80 | cold | 600 | 50 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | -- | 0/50 | >budget | >budget |
| 80 | warm-spt | 600 | 50 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | -- | 0/50 | >budget | >budget |
| 50 | cold | 1800 | 20 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0/20 | >budget | >budget |
| 50 | warm-spt | 1800 | 20 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0/20 | >budget | >budget |
| 80 | cold | 1800 | 20 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0/20 | >budget | >budget |
| 80 | warm-spt | 1800 | 20 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0/20 | >budget | >budget |

## (b) SLAB seed 301

| Modules | Mode | Budget (s) | n | <=1 s | <=5 s | <=60 s | <=300 s | <=600 s | <=1800 s | reached | median, all (s) | median, reached (s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | cold | 300 | 100 | 34% (34) | 99% (99) | 100% (100) | 100% (100) | -- | -- | 100/100 | 1.2 | 1.2 |
| 10 | warm-spt | 300 | 100 | 85% (85) | 100% (100) | 100% (100) | 100% (100) | -- | -- | 100/100 | 0.2 | 0.2 |
| 20 | cold | 300 | 50 | 0% (0) | 0% (0) | 12% (6) | 58% (29) | -- | -- | 29/50 | 218.7 | 107.2 |
| 20 | warm-spt | 300 | 50 | 14% (7) | 14% (7) | 30% (15) | 52% (26) | -- | -- | 26/50 | 229.5 | 26.5 |
| 30 | cold | 300 | 20 | 0% (0) | 0% (0) | 5% (1) | 20% (4) | -- | -- | 4/20 | >budget | 192.5 |
| 30 | warm-spt | 300 | 20 | 20% (4) | 20% (4) | 25% (5) | 25% (5) | -- | -- | 5/20 | >budget | 0.0 |
| 50 | cold | 600 | 50 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | -- | 0/50 | >budget | >budget |
| 50 | warm-spt | 600 | 50 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | -- | 0/50 | >budget | >budget |
| 80 | cold | 600 | 50 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | -- | 0/50 | >budget | >budget |
| 80 | warm-spt | 600 | 50 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | -- | 0/50 | >budget | >budget |
| 50 | cold | 1800 | 20 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0/20 | >budget | >budget |
| 50 | warm-spt | 1800 | 20 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0/20 | >budget | >budget |
| 80 | cold | 1800 | 20 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0/20 | >budget | >budget |
| 80 | warm-spt | 1800 | 20 | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0% (0) | 0/20 | >budget | >budget |

## Solver status

| Modules | Mode | Budget (s) | n | proven optimal | median solve end (s) | median end, optimal runs (s) | mean CP-SAT | mean SLAB seed mean | final <= SLAB seed mean | final < SLAB seed mean | SPT hint <= SLAB at t=0 | min cpu_share (budget-limited) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | cold | 300 | 100 | 83 | 3.6 | 2.3 | 203.20 | 211.30 | 100 | 100 | -- | 0.993 |
| 10 | warm-spt | 300 | 100 | 85 | 1.9 | 1.1 | 203.23 | 211.30 | 100 | 100 | 25 | 0.993 |
| 20 | cold | 300 | 50 | 0 | 300.0 | >budget | 253.74 | 255.01 | 32 | 30 | -- | 0.959 |
| 20 | warm-spt | 300 | 50 | 0 | 300.0 | >budget | 253.36 | 255.01 | 32 | 31 | 4 | 0.998 |
| 30 | cold | 300 | 20 | 0 | 300.0 | >budget | 318.85 | 308.67 | 1 | 1 | -- | 0.999 |
| 30 | warm-spt | 300 | 20 | 0 | 300.0 | >budget | 318.50 | 308.67 | 2 | 2 | 0 | 0.998 |
| 50 | cold | 600 | 50 | 0 | 600.1 | >budget | 464.24 | 414.85 | 0 | 0 | -- | 0.997 |
| 50 | warm-spt | 600 | 50 | 0 | 600.1 | >budget | 444.08 | 414.85 | 0 | 0 | 0 | 0.998 |
| 80 | cold | 600 | 50 | 0 | 600.1 | >budget | 724.54 | 570.26 | 0 | 0 | -- | 0.905 |
| 80 | warm-spt | 600 | 50 | 0 | 600.1 | >budget | 617.12 | 570.26 | 0 | 0 | 0 | 0.993 |
| 50 | cold | 1800 | 20 | 0 | 1800.1 | >budget | 444.60 | 414.47 | 0 | 0 | -- | 0.971 |
| 50 | warm-spt | 1800 | 20 | 0 | 1800.1 | >budget | 439.65 | 414.47 | 0 | 0 | 0 | 0.995 |
| 80 | cold | 1800 | 20 | 0 | 1800.1 | >budget | 672.05 | 568.58 | 0 | 0 | -- | 0.996 |
| 80 | warm-spt | 1800 | 20 | 0 | 1800.1 | >budget | 616.00 | 568.58 | 0 | 0 | 0 | 0.977 |

Sources: or_solution/PPVC/anytime/10x25+ppvc-mixed_cold_300s_w4.jsonl, or_solution/PPVC/anytime/10x25+ppvc-mixed_warm-spt_300s_w4.jsonl, or_solution/PPVC/anytime/20x25+ppvc-mixed_cold_300s_w4.jsonl, or_solution/PPVC/anytime/20x25+ppvc-mixed_warm-spt_300s_w4.jsonl, or_solution/PPVC/anytime/30x25+ppvc-mixed_cold_300s_w4.jsonl, or_solution/PPVC/anytime/30x25+ppvc-mixed_warm-spt_300s_w4.jsonl, or_solution/PPVC/anytime/50x25+ppvc-mixed_cold_1800s_w4.jsonl, or_solution/PPVC/anytime/50x25+ppvc-mixed_cold_600s_w4.jsonl, or_solution/PPVC/anytime/50x25+ppvc-mixed_warm-spt_1800s_w4.jsonl, or_solution/PPVC/anytime/50x25+ppvc-mixed_warm-spt_600s_w4.jsonl, or_solution/PPVC/anytime/80x25+ppvc-mixed_cold_1800s_w4.jsonl, or_solution/PPVC/anytime/80x25+ppvc-mixed_cold_600s_w4.jsonl, or_solution/PPVC/anytime/80x25+ppvc-mixed_warm-spt_1800s_w4.jsonl, or_solution/PPVC/anytime/80x25+ppvc-mixed_warm-spt_600s_w4.jsonl
