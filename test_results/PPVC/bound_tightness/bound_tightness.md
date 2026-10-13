# Tightness of the two bounds along SLAB seed-301 greedy trajectories (CPU)

Model `10x25+ppvc-mixed+a1lagx05-2k-s301`, greedy decoding on the CPU, true waits, every order of each test set. Ratios are Chat/Cmax of the same order, averaged over orders; the step average runs over the T decision states s_0..s_{T-1}. "aug" = lag-augmented bound of the environment (bound_lag_free=False), "free" = lag-free bound (replay env with bound_lag_free=True, identical schedule). "aug+t" / "free+t" = Eq. (4) as printed, with max(., t), which the environment does not compute.

| Modules | Orders | aug s0/Cmax | free s0/Cmax | aug step-avg | free step-avg | (aug-free)/Cmax step-avg | aug+t s0/Cmax | aug+t step-avg | free+t step-avg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 100 | 0.9522 | 0.4324 | 0.9829 | 0.6548 | 0.3281 | 0.9522 | 0.9837 | 0.6687 |
| 20 | 50 | 0.8250 | 0.3619 | 0.9268 | 0.6472 | 0.2796 | 0.8250 | 0.9312 | 0.6594 |
| 30 | 20 | 0.6681 | 0.2998 | 0.8561 | 0.6260 | 0.2302 | 0.6681 | 0.8668 | 0.6394 |
| 50 | 50 | 0.5167 | 0.2267 | 0.7801 | 0.5919 | 0.1883 | 0.5167 | 0.7941 | 0.6047 |
| 80 | 50 | 0.3814 | 0.1669 | 0.7040 | 0.5596 | 0.1443 | 0.3814 | 0.7206 | 0.5709 |

## Checks

| Modules | steps | free > aug | aug > Cmax | free > Cmax | aug+t > Cmax | orders aug(s_T) != Cmax | orders free(s_T) != Cmax | max abs env - direct (h) | same schedule | validator | orders CPU != stored GPU | max abs diff (h) | mean diff (h) | orders CPU != S5 CPU run |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|---:|---:|---:|---:|
| 10 | 22100 | 0 | 0 | 0 | 0 | 0 | 0 | 1.42e-13 | True | True | 11 | 5 | -0.11 | 0 |
| 20 | 22050 | 0 | 0 | 0 | 0 | 0 | 0 | 1.99e-13 | True | True | 15 | 17 | -0.46 | 0 |
| 30 | 13220 | 0 | 0 | 0 | 0 | 0 | 0 | 2.27e-13 | True | True | 11 | 14 | -0.75 | 0 |
| 50 | 55050 | 0 | 0 | 0 | 0 | 0 | 0 | 2.84e-13 | True | True | 24 | 6 | -0.14 | 0 |
| 80 | 88050 | 0 | 0 | 0 | 0 | 0 | 0 | 6.82e-13 | True | True | 35 | 15 | +0.64 | 0 |

Mean makespan: CPU rerun / stored GPU result / S5 CPU run

- 10x25+ppvc-mixed: 210.91 / 211.02 / 210.91 h
- 20x25+ppvc-mixed: 254.48 / 254.94 / 254.48 h
- 30x25+ppvc-mixed: 310.40 / 311.15 / 310.40 h
- 50x25+ppvc-mixed: 412.48 / 412.62 / 412.48 h
- 80x25+ppvc-mixed: 566.76 / 566.12 / 566.76 h
