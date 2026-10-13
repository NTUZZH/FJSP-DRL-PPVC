"""
merge_memetic_parts.py
======================
Merge the per-range result files of eval_memetic.py --instances LO:HI into one
result file per dataset, with the schema of a full run.

  python merge_memetic_parts.py --parts_dir <dir> --dataset 50x25+ppvc-mixed \
      --tag MA300 --out_dir <dir> [--n_instances 50]

Inputs: <parts_dir>/Result_<tag>+memetic_<dataset>.iLO-HI.json/.npy. The
base parts must cover the instances 0..N-1 exactly once. Files under
<parts_dir>/rerun/ with the same naming replace the records (and .npy rows)
of their instances; the merged JSON lists them under "reruns" with the
replaced record's cpu_share. All parts must share every run-level setting
(budget, seeds, parameters, fast_eval, numba version).

Output: <out_dir>/Result_<tag>+memetic_<dataset>.npy ([N, 2] = makespan,
seconds, row = instance id) and .json (run-level fields as in a full run,
"instances" in instance order, "contention" recomputed over the merged
records, "merged_from" naming the part files).
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

from eval_stochastic_lags import contention_summary, LOW_CPU_SHARE

RUN_KEYS = ('method', 'dataset', 'route_variant', 'budget_s',
            'budget_includes_seeding', 'max_gens', 'seed', 'seed_test',
            'params', 'fast_eval', 'numba')


def _load_parts(pattern):
    parts = []
    for js in sorted(glob.glob(pattern)):
        m = re.search(r'\.i(\d+)-(\d+)\.json$', js)
        if not m:
            continue
        lo, hi = int(m.group(1)), int(m.group(2))
        with open(js) as f:
            d = json.load(f)
        arr = np.load(js[:-len('.json')] + '.npy')
        assert d.get('instance_range') == [lo, hi], js
        assert len(d['instances']) == hi - lo == arr.shape[0], js
        parts.append((lo, hi, js, d, arr))
    return parts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts_dir', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--n_instances', type=int, default=None)
    args = ap.parse_args(argv)

    stem = f'Result_{args.tag}+memetic_{args.dataset}'
    base = _load_parts(os.path.join(args.parts_dir, stem + '.i*-*.json'))
    reruns = _load_parts(os.path.join(args.parts_dir, 'rerun',
                                      stem + '.i*-*.json'))
    if not base:
        sys.exit(f'no parts {stem}.i*-*.json under {args.parts_dir}')
    base.sort(key=lambda p: p[0])
    n = args.n_instances or base[-1][1]
    covered = []
    for lo, hi, *_ in base:
        covered.extend(range(lo, hi))
    if covered != list(range(n)):
        sys.exit(f'base parts do not cover 0..{n - 1} exactly once: '
                 f'{[(p[0], p[1]) for p in base]}')
    ref = base[0][3]
    for lo, hi, js, d, _ in base + reruns:
        for k in RUN_KEYS:
            if d.get(k) != ref.get(k):
                sys.exit(f'{js}: run-level field {k} differs '
                         f'({d.get(k)} vs {ref.get(k)})')

    records = [None] * n
    results = np.zeros((n, 2))
    for lo, hi, js, d, arr in base:
        for k in range(hi - lo):
            records[lo + k] = d['instances'][k]
            results[lo + k] = arr[k]
    replaced = []
    for lo, hi, js, d, arr in sorted(reruns, key=lambda p: os.path.getmtime(
            p[2])):
        for k in range(hi - lo):
            i = lo + k
            new = d['instances'][k]
            assert new['instance'] == records[i]['instance'], js
            replaced.append(dict(index=i, instance=new['instance'],
                                 replaced_cpu_share=records[i]['cpu_share'],
                                 replaced_best_makespan=records[i]
                                 ['best_makespan'],
                                 new_cpu_share=new['cpu_share'],
                                 file=os.path.basename(js)))
            records[i] = new
            results[i] = arr[k]
    for i, r in enumerate(records):
        assert r['instance'] == f'instance_{i:03d}', (i, r['instance'])
        assert r['validator_violations'] == 0, r['instance']
        assert results[i, 0] == r['best_makespan'], r['instance']

    out = {k: ref[k] for k in ref if k not in ('instances', 'contention',
                                                'instance_range',
                                                'n_instances')}
    merged = dict(method=ref['method'], dataset=ref['dataset'],
                  route_variant=ref['route_variant'], n_instances=n)
    merged.update({k: v for k, v in out.items() if k not in merged})
    merged['merged_from'] = [os.path.basename(p[2]) for p in base]
    merged['reruns'] = replaced
    merged['instances'] = records
    merged['contention'] = contention_summary(records)
    os.makedirs(args.out_dir, exist_ok=True)
    stem_out = os.path.join(args.out_dir, stem)
    np.save(stem_out + '.npy', results)
    with open(stem_out + '.json', 'w') as f:
        json.dump(merged, f, indent=1)
    c = merged['contention']
    low = [r['instance'] for r in records
           if r.get('hit_limit') and r['cpu_share'] < LOW_CPU_SHARE]
    print(f'{stem}: {n} instances, mean makespan {results[:, 0].mean():.3f}, '
          f'min cpu_share {c["cpu_share_min"]}, low-share {low}, '
          f'reruns {len(replaced)} -> {stem_out}.json')
    return merged


if __name__ == '__main__':
    main()
