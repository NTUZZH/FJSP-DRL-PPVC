"""
Station contention of PPVC orders as a function of order size and lag scale.

For each test set and lag scale, every instance is scheduled by the SPT rule
(the strongest rule at every size) with all lags multiplied by the scale and
rounded, exactly as train.py --lag_scale does for training orders. The
validated schedule then gives:
  util_mean      busy time of a station / makespan, averaged over stations;
  queue_share    time modules wait for a station / (that time + lag time),
                 i.e. how much of a module's idle time is station contention
                 and how much is the mandatory wait;
  lag_share_cp   share of the lags in the longest lag-weighted route.
The comparison asks whether a 10-module training order with shortened lags
shows the station contention of a 50- or 80-module production batch at the
true lags.

usage: python analyze_training_load.py [--max_instances N] [--out PATH]
"""
import argparse
import json
import os

import numpy as np

from eval_byproducts import (compute_byproducts, critical_path_lengths,
                             import_eval_ppvc, run_validated)

CASES = [  # (dataset, lag scale)
    ('10x25+ppvc-mixed', 0.0), ('10x25+ppvc-mixed', 0.25), ('10x25+ppvc-mixed', 0.5),
    ('10x25+ppvc-mixed', 0.75), ('10x25+ppvc-mixed', 1.0),
    ('20x25+ppvc-mixed', 1.0), ('30x25+ppvc-mixed', 1.0),
    ('50x25+ppvc-mixed', 1.0), ('80x25+ppvc-mixed', 1.0),
    ('10x15+ppvc-mixed', 1.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max_instances', type=int, default=20)
    ap.add_argument('--out', default='test_results/PPVC/training_load.json')
    args = ap.parse_args()
    ep = import_eval_ppvc()
    from ppvc_instance_generator import load_instance
    rows = []
    for ds, scale in CASES:
        stems = ep.list_instances(f'data/PPVC/{ds}')[:args.max_instances]
        vals = {'util_mean': [], 'queue_share': [], 'lag_share_cp': [], 'makespan': []}
        for i, stem in enumerate(stems):
            jl, pt, meta = load_instance(stem)
            meta = dict(meta)
            meta['time_lag'] = np.rint(np.asarray(meta['time_lag'], dtype=float) * scale).astype(int)
            ms, _, asg, ct = run_validated(ep, 'SPT', jl, pt, meta, i, 50)
            cp = critical_path_lengths(jl, pt, meta['time_lag'])
            r = compute_byproducts(jl, pt, meta['time_lag'], asg, ct, meta['mch_type'],
                                   np.full(len(jl), 1e9))
            q, lag_hold = float(r['queue_wait']), float(r['lag_hold'])
            vals['util_mean'].append(r['util_mean'])
            vals['makespan'].append(ms)
            vals['queue_share'].append(q / (q + lag_hold) if q + lag_hold > 0 else 0.0)
            # lag share on the longest lag-weighted route of the order
            j = int(np.argmax(cp))
            s = int(np.sum(jl[:j]))
            e = s + int(jl[j])
            lag_route = float(np.asarray(meta['time_lag'], dtype=float)[s:e - 1].sum())
            vals['lag_share_cp'].append(lag_route / float(cp[j]) if cp[j] > 0 else 0.0)
        row = {'dataset': ds, 'lag_scale': scale, 'n': len(stems)}
        row.update({k: (float(np.mean(v)) if v else None) for k, v in vals.items()})
        rows.append(row)
        print(f"{ds:20s} lag x{scale:<4g} n={len(stems):3d} util={row['util_mean']:.3f} "
              f"queue_share={row['queue_share']:.3f} "
              f"lag_share_cp={row['lag_share_cp']:.3f} SPT makespan={row['makespan']:.1f}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(rows, f, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()
