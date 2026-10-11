# Duration-sensitivity robustness summary

- dataset: `10x25+ppvc-mixed`  (100 instances)
- A3 checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False
- perturbation: every compatible processing time scaled by i.i.d. U[1-delta, 1+delta], rounded to >=1 integer hours; time-lags UNCHANGED
- master seed: 0  (per-instance RNG = f(seed, delta, instance))  |  seed_test: 50
- stat: two-sided paired Wilcoxon signed-rank (A3 vs SPT)
- headline claim verified at each delta: A3 wins the majority of pairs AND p < 0.05

| delta | A3 mean (std) | SPT mean (std) | A3 W/T/L vs SPT | Wilcoxon p | A3 < SPT? |
|---|---|---|---|---|---|
| 0 | 210.9 (9.2) | 216.0 (8.5) | 76/3/21 | 1.1e-09 | YES |
| 0.1 | 210.8 (9.6) | 215.0 (8.3) | 69/9/22 | 2.7e-08 | YES |
| 0.2 | 207.9 (9.5) | 213.1 (10.0) | 74/3/23 | 5.6e-09 | YES |
| 0.3 | 204.5 (9.5) | 209.4 (9.2) | 74/4/22 | 1.7e-09 | YES |

**Conclusion:** the A3 < SPT ranking HOLDS at every tested delta (see the A3 < SPT column). The central relative ranking is robust to multiplicative duration perturbations of up to +/-0.3.

---

## Copy-paste \newcommand macro block (for macros.tex)

```latex
% --- duration-sensitivity robustness (eval_sensitivity.py) ---
\newcommand{\SensDeltaZeroA3}{210.9}   % A3 greedy mean makespan, delta=0 (h)
\newcommand{\SensDeltaZeroSpt}{216.0}  % SPT mean makespan, delta=0 (h)
\newcommand{\SensDeltaZeroP}{1.1\times 10^{-9}}  % Wilcoxon p, A3 vs SPT, delta=0
\newcommand{\SensDeltaTenA3}{210.8}   % A3 greedy mean makespan, delta=0.1 (h)
\newcommand{\SensDeltaTenSpt}{215.0}  % SPT mean makespan, delta=0.1 (h)
\newcommand{\SensDeltaTenP}{2.7\times 10^{-8}}  % Wilcoxon p, A3 vs SPT, delta=0.1
\newcommand{\SensDeltaTwentyA3}{207.9}   % A3 greedy mean makespan, delta=0.2 (h)
\newcommand{\SensDeltaTwentySpt}{213.1}  % SPT mean makespan, delta=0.2 (h)
\newcommand{\SensDeltaTwentyP}{5.6\times 10^{-9}}  % Wilcoxon p, A3 vs SPT, delta=0.2
\newcommand{\SensDeltaThirtyA3}{204.5}   % A3 greedy mean makespan, delta=0.3 (h)
\newcommand{\SensDeltaThirtySpt}{209.4}  % SPT mean makespan, delta=0.3 (h)
\newcommand{\SensDeltaThirtyP}{1.7\times 10^{-9}}  % Wilcoxon p, A3 vs SPT, delta=0.3
\newcommand{\SensMaxDelta}{0.3}  % largest delta at which A3 < SPT still holds
```

