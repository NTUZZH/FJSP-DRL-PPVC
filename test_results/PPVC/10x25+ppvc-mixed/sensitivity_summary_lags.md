# Time-lag-sensitivity robustness summary

- dataset: `10x25+ppvc-mixed`  (100 instances)
- A3 checkpoint: `./trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth`
- architecture (from snapshot): use_lag_features=False, use_type_embedding=False
- perturbation (--perturb lags): every nonzero time-lag scaled by i.i.d. U[1-delta, 1+delta], rounded to >=1 integer hours; zero lags stay zero; processing times UNCHANGED
- master seed: 0  (per-instance RNG = f(seed, delta, instance); lag draws use the separate child stream 1 of that seed)  |  seed_test: 50
- every A3 and SPT schedule validated against the perturbed lags and processing times
- stat: two-sided paired Wilcoxon signed-rank (A3 vs SPT)
- headline claim verified at each delta: A3 wins the majority of pairs AND p < 0.05

| delta | A3 mean (std) | SPT mean (std) | A3 W/T/L vs SPT | Wilcoxon p | A3 < SPT? |
|---|---|---|---|---|---|
| 0 | 210.9 (9.2) | 216.0 (8.5) | 76/3/21 | 1.1e-09 | YES |
| 0.1 | 211.4 (10.0) | 216.6 (8.8) | 77/9/14 | 1.3e-10 | YES |
| 0.2 | 212.5 (11.2) | 217.0 (9.9) | 71/4/25 | 7.3e-08 | YES |
| 0.3 | 213.6 (12.8) | 217.8 (11.4) | 73/5/22 | 7.3e-08 | YES |

**Conclusion:** the A3 < SPT ranking HOLDS at every tested delta (see the A3 < SPT column). The central relative ranking is robust to multiplicative time-lag perturbations of up to +/-0.3.

---

## Copy-paste \newcommand macro block (for macros.tex)

```latex
% --- time-lag-sensitivity robustness (eval_sensitivity.py --perturb lags) ---
\newcommand{\SensLagDeltaZeroA3}{210.9}   % A3 greedy mean makespan, delta=0 (h)
\newcommand{\SensLagDeltaZeroSpt}{216.0}  % SPT mean makespan, delta=0 (h)
\newcommand{\SensLagDeltaZeroP}{1.1\times 10^{-9}}  % Wilcoxon p, A3 vs SPT, delta=0
\newcommand{\SensLagDeltaTenA3}{211.4}   % A3 greedy mean makespan, delta=0.1 (h)
\newcommand{\SensLagDeltaTenSpt}{216.6}  % SPT mean makespan, delta=0.1 (h)
\newcommand{\SensLagDeltaTenP}{1.3\times 10^{-10}}  % Wilcoxon p, A3 vs SPT, delta=0.1
\newcommand{\SensLagDeltaTwentyA3}{212.5}   % A3 greedy mean makespan, delta=0.2 (h)
\newcommand{\SensLagDeltaTwentySpt}{217.0}  % SPT mean makespan, delta=0.2 (h)
\newcommand{\SensLagDeltaTwentyP}{7.3\times 10^{-8}}  % Wilcoxon p, A3 vs SPT, delta=0.2
\newcommand{\SensLagDeltaThirtyA3}{213.6}   % A3 greedy mean makespan, delta=0.3 (h)
\newcommand{\SensLagDeltaThirtySpt}{217.8}  % SPT mean makespan, delta=0.3 (h)
\newcommand{\SensLagDeltaThirtyP}{7.3\times 10^{-8}}  % Wilcoxon p, A3 vs SPT, delta=0.3
\newcommand{\SensLagMaxDelta}{0.3}  % largest delta at which A3 < SPT still holds
```

