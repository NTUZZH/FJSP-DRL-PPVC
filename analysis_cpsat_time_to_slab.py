#!/usr/bin/env python3
"""analysis_cpsat_time_to_slab.py -- how long CP-SAT needs to reach the SLAB
makespan, from the anytime logs of the paper's CP-SAT runs (analysis only).

For the cold and the SPT-started CP-SAT runs (four workers) at 10, 20 and 30
modules (300 s) and at 50 and 80 modules (600 s, and the 1800-s runs on the
first 20 orders), and for each order, the script finds the first solver time
at which the incumbent makespan is at most
  (a) the per-order mean of the three SLAB seeds (the comparison unit of the
      paper), and
  (b) the makespan of SLAB seed 301,
both taken from the stored greedy results of make_paper_macros.HSTAR.
The incumbent is the running minimum of the logged trajectory
([time, objective, bound] per improving solution); an SPT-started run logs
its hint at t = 0, and a point logged after the time limit (solver shutdown)
is clamped to the limit, as in Fig. 4. Records are read with
make_paper_macros.anytime(), which checks instance ids, validator
feasibility and the CPU share of every budget-limited run.

Per size and mode it reports the share of orders reached within 1, 5, 60,
300 s (and 600, 1800 s where the budget allows), the share reached at all,
the median time to reach over all orders (an order never reached counts as
+inf, so the median is +inf when half or more are never reached) and over the
reached orders, the number of runs proven optimal, and the median solve end
time (solve_elapsed_s).

Writes test_results/PPVC/cpsat_time_to_slab/time_to_slab.json and .md.
usage: python analysis_cpsat_time_to_slab.py
"""
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.dont_write_bytecode = True
import make_paper_macros as M  # noqa: E402

OUT = os.path.join(REPO, 'test_results', 'PPVC', 'cpsat_time_to_slab')
RUNS = (  # size label, dataset, budget (s), orders
    ('Ten', '10x25+ppvc-mixed', 300, 100),
    ('Twenty', '20x25+ppvc-mixed', 300, 50),
    ('Thirty', '30x25+ppvc-mixed', 300, 20),
    ('Fifty', '50x25+ppvc-mixed', 600, 50),
    ('Eighty', '80x25+ppvc-mixed', 600, 50),
    ('Fifty', '50x25+ppvc-mixed', 1800, 20),
    ('Eighty', '80x25+ppvc-mixed', 1800, 20),
)
MODES = ('cold', 'warm-spt')
CHECKPOINTS = (1, 5, 60, 300, 600, 1800)


def first_reach(rec, target, limit):
    pts = sorted((min(float(t), float(limit)), float(o))
                 for t, o, *_ in rec['trajectory'] if o is not None)
    M.check(abs(min(o for _, o in pts) - float(rec['objective'])) < 1e-9,
            f"{rec['instance']}: trajectory minimum != objective")
    best = np.inf
    for t, o in pts:
        best = min(best, o)
        if best <= target + 1e-9:
            return t
    return np.inf


def summarize(times, limit):
    times = np.asarray(times, float)
    n = times.size
    d = {'n': int(n)}
    for c in CHECKPOINTS:
        if c <= limit:
            d[f'share_within_{c}s'] = float(np.mean(times <= c))
            d[f'n_within_{c}s'] = int(np.sum(times <= c))
    d['n_reached'] = int(np.sum(np.isfinite(times)))
    d['median_time_all_s'] = float(np.median(times))   # inf if >= half unreached
    fin = times[np.isfinite(times)]
    d['median_time_reached_s'] = float(np.median(fin)) if fin.size else None
    return d


def main():
    res = []
    for S, ds, limit, n in RUNS:
        mean3 = M.inst_of(M.HSTAR, ds)[:n]
        s301 = np.asarray(M.result(M.HSTAR.format(s=301), ds))[:n]
        for mode in MODES:
            recs = M.anytime(ds, mode, limit, mode, n=n)
            src = M.any_rel(ds, mode, limit)
            t_mean = [first_reach(r, mean3[i], limit) for i, r in enumerate(recs)]
            t_301 = [first_reach(r, s301[i], limit) for i, r in enumerate(recs)]
            obj = np.array([float(r['objective']) for r in recs])
            row = {
                'size': S, 'dataset': ds, 'mode': mode, 'budget_s': limit, 'n': n,
                'source': src,
                'target_seed_mean': summarize(t_mean, limit),
                'target_seed301': summarize(t_301, limit),
                'n_optimal': int(sum(r['status'] == 'OPTIMAL' for r in recs)),
                'status_counts': {s: int(sum(r['status'] == s for r in recs))
                                  for s in sorted({r['status'] for r in recs})},
                'median_solve_end_s': float(np.median([float(r['solve_elapsed_s']) for r in recs])),
                'median_solve_end_optimal_s': (float(np.median(
                    [float(r['solve_elapsed_s']) for r in recs if r['status'] == 'OPTIMAL']))
                    if any(r['status'] == 'OPTIMAL' for r in recs) else None),
                'mean_objective': float(obj.mean()),
                'mean_slab_seed_mean': float(mean3.mean()),
                'mean_slab_s301': float(s301.mean()),
                'final_obj_le_seed_mean': int(np.sum(obj <= mean3 + 1e-9)),
                'final_obj_lt_seed_mean': int(np.sum(obj < mean3 - 1e-9)),
                'min_cpu_share': float(np.nanmin([float(r.get('cpu_share', np.nan)) for r in recs])),
                'min_cpu_share_budget_limited': float(min(
                    [float(r['cpu_share']) for r in recs
                     if float(r['solve_elapsed_s']) > M.ANYTIME_TIMED_S] or [np.nan])),
            }
            if mode == 'warm-spt':
                hint = np.array([float(r['hint_makespan']) for r in recs])
                row['hint_le_seed_mean_at_t0'] = int(np.sum(hint <= mean3 + 1e-9))
                row['hint_le_s301_at_t0'] = int(np.sum(hint <= s301 + 1e-9))
            res.append(row)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'time_to_slab.json'), 'w') as f:
        json.dump({'script': 'analysis_cpsat_time_to_slab.py',
                   'slab_sources': {ds: [M.src(M.HSTAR, ds)] for _, ds, _, _ in RUNS},
                   'runs': res}, f, indent=1)

    def fmt_t(x):
        return '>budget' if x is None or not np.isfinite(x) else f'{x:.1f}'

    L = ['# CP-SAT time to reach the SLAB makespan (anytime logs, four workers)', '',
         'Target (a): per-order mean of SLAB seeds 301-303; target (b): SLAB seed 301 alone (stored greedy results). '
         'Time = first solver time at which the incumbent (running minimum of the logged trajectory; '
         'SPT-started runs log the SPT hint at t = 0) is at most the target. Median over all orders counts an '
         'order never reached as +inf ("> budget").', '']
    for tgt, lab in (('target_seed_mean', '(a) SLAB seed mean'), ('target_seed301', '(b) SLAB seed 301')):
        L += [f'## {lab}', '',
              '| Modules | Mode | Budget (s) | n | <=1 s | <=5 s | <=60 s | <=300 s | <=600 s | <=1800 s | reached | median, all (s) | median, reached (s) |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for r in res:
            d = r[tgt]

            def sh(c):
                k = f'share_within_{c}s'
                return f"{100 * d[k]:.0f}% ({d[f'n_within_{c}s']})" if k in d else '--'
            L.append(f"| {r['dataset'].split('x')[0]} | {r['mode']} | {r['budget_s']} | {r['n']} | "
                     f"{sh(1)} | {sh(5)} | {sh(60)} | {sh(300)} | {sh(600)} | {sh(1800)} | "
                     f"{d['n_reached']}/{d['n']} | {fmt_t(d['median_time_all_s'])} | "
                     f"{fmt_t(d['median_time_reached_s'])} |")
        L.append('')
    L += ['## Solver status', '',
          '| Modules | Mode | Budget (s) | n | proven optimal | median solve end (s) | median end, optimal runs (s) | mean CP-SAT | mean SLAB seed mean | final <= SLAB seed mean | final < SLAB seed mean | SPT hint <= SLAB at t=0 | min cpu_share (budget-limited) |',
          '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in res:
        L.append(f"| {r['dataset'].split('x')[0]} | {r['mode']} | {r['budget_s']} | {r['n']} | "
                 f"{r['n_optimal']} | {r['median_solve_end_s']:.1f} | "
                 f"{fmt_t(r['median_solve_end_optimal_s'])} | {r['mean_objective']:.2f} | "
                 f"{r['mean_slab_seed_mean']:.2f} | {r['final_obj_le_seed_mean']} | {r['final_obj_lt_seed_mean']} | "
                 f"{r.get('hint_le_seed_mean_at_t0', '--')} | {r['min_cpu_share_budget_limited']:.3f} |")
    L += ['', 'Sources: ' + ', '.join(sorted({r['source'] for r in res}))]
    with open(os.path.join(OUT, 'time_to_slab.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    os.chdir(REPO)
    main()
