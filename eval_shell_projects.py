#!/usr/bin/env python3
"""eval_shell_projects.py -- single-shell project orders (all-RC or all-steel)
at 10 and 50 modules: seed audit, set generation, evaluation, statistics.

Stages
  audit  Lists the generator-seed range of every instance set on disk
         (data/PPVC/*, data/data_train_vali/PPVC/*), identifies the
         unrecorded seed of 40x25+ppvc-mixed by regeneration, replays the
         training-instance seed stream of the SLAB runs (seed_train 301, 302,
         303: np.random.seed(seed_train), then one np.random.randint(0,
         2**31 - 1) per training order, 20 orders every 20 updates for 2000
         updates, as train.py does), checks that the existing single-shell
         sets regenerate from their recorded seeds, and reports every
         overlap of the single-shell seed ranges with any other range.
         Writes test_results/PPVC/shell_projects/seed_audit.json.
  make   Generates the new single-shell sets with the repo's generator
         (ppvc_instance_generator, DEFAULT_FACTORY, fixed routes, one order
         per seed seed0 + i), exactly like make_scale_testsets.py, and with
         seed ranges that the audit shows to be disjoint from every other
         range. Existing instance files are never overwritten.
  eval   Rolls out one method on one set on the CPU, re-validates every
         schedule with schedule_validator.validate_schedule against the true
         waits, and writes test_results/PPVC/<set>/shell_<method>.json and
         shell_Result_<method>_<set>.npy ([n, 2]: makespan, seconds). Methods:
         greedy-s301, greedy-s302, greedy-s303 (SLAB policy of that training
         seed, greedy decoding) and rules (the eight dispatching rules FIFO,
         MOR, SPT, MWKR, SPTL, LLF, MWKRL, LRPTL in the SLAB transition,
         tie-break seed 50 + order index, as eval_ppvc.py).
  stats  Mean makespan of every method; the SLAB seed mean (per-order mean of
         the three seeds) and each seed alone against SPT and LLF: mean paired
         difference with a 95% percentile bootstrap interval (2000 resamples,
         numpy default_rng(0), stats_tests.paired_mean_diff_ci), shorter /
         equal / longer counts, two-sided Wilcoxon signed-rank p (zeros
         dropped) and the matched-pairs rank-biserial r (r < 0: SLAB
         shorter). Writes test_results/PPVC/shell_projects/shell_stats.json
         and shell_summary.md.

usage
  python eval_shell_projects.py --stage audit
  python eval_shell_projects.py --stage make
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0-5 \
      python eval_shell_projects.py --stage eval --set 50x25+ppvc-rc_project \
      --method greedy-s301
  python eval_shell_projects.py --stage stats
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, 'data', 'PPVC')
TRAIN_VALI = os.path.join(REPO, 'data', 'data_train_vali', 'PPVC')
TR = os.path.join(REPO, 'test_results', 'PPVC')
OUT = os.path.join(TR, 'shell_projects')
MODEL = '10x25+ppvc-mixed+a1lagx05-2k-s{s}'
SEEDS = (301, 302, 303)
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR', 'SPTL', 'LLF', 'MWKRL', 'LRPTL')
SEED_TEST = 50
N_BOOT, BOOT_SEED = 2000, 0

OLD_SETS = ('10x25+ppvc-rc_project', '10x25+ppvc-steel_project')
# new sets: name -> (n_modules, mix, seed0, n_orders); seeds chosen above the
# independent confirmation sets (700,000-730,049) and checked by the audit
NEW_SETS = {
    '10x25+ppvc-rc_project-ind': (10, 'rc_project', 740000, 50),
    '10x25+ppvc-steel_project-ind': (10, 'steel_project', 750000, 50),
    '50x25+ppvc-rc_project': (50, 'rc_project', 760000, 50),
    '50x25+ppvc-steel_project': (50, 'steel_project', 770000, 50),
}
PURPOSE = {
    10: 'single-shell project test set at 10 modules (eval-only), seeds disjoint from all other sets',
    50: 'single-shell project test set at 50 modules (eval-only), seeds disjoint from all other sets',
}
EVAL_SETS = tuple(NEW_SETS) + OLD_SETS


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #
def set_ranges():
    out = []
    for root, kind in ((DATA, 'test'), (TRAIN_VALI, 'validation')):
        for d in sorted(glob.glob(os.path.join(root, '*'))):
            mp = os.path.join(d, 'dataset_meta.json')
            n_files = len(glob.glob(os.path.join(d, 'instance_*.fjs')))
            if not os.path.exists(mp):
                out.append({'set': os.path.relpath(d, REPO), 'kind': kind,
                            'seed0': None, 'n': n_files, 'meta': None})
                continue
            m = json.load(open(mp))
            out.append({'set': os.path.relpath(d, REPO), 'kind': kind,
                        'seed0': int(m['seed0']), 'n': int(m['n_instances']),
                        'n_files': n_files, 'meta': m})
    return out


def identify_seed(set_dir, n_modules, mix='mixed'):
    """seed0 of a set without dataset_meta.json, by regenerating order 0 and
    order 1 for candidate seeds (multiples of 1000 up to 1,000,000, then of
    100 up to 200,000)."""
    from ppvc_instance_generator import (DEFAULT_FACTORY, load_instance,
                                         ppvc_instance_generator)
    jl0, pt0, m0 = load_instance(os.path.join(set_dir, 'instance_000'))
    cands = list(range(0, 1_000_001, 1000)) + [
        s for s in range(0, 200_001, 100) if s % 1000]
    for s in cands:
        jl, pt, m = ppvc_instance_generator(n_modules=n_modules, class_mix=mix,
                                            station_counts=DEFAULT_FACTORY,
                                            seed=s)
        if pt.shape == pt0.shape and np.array_equal(pt, pt0) and \
                np.array_equal(m['time_lag'], m0['time_lag']):
            jl1, pt1, m1 = load_instance(os.path.join(set_dir, 'instance_001'))
            _, ptb, mb = ppvc_instance_generator(
                n_modules=n_modules, class_mix=mix,
                station_counts=DEFAULT_FACTORY, seed=s + 1)
            return s, bool(np.array_equal(ptb, pt1)
                           and np.array_equal(mb['time_lag'], m1['time_lag']))
    return None, False


def training_seed_draws(seed_train, n_updates=2000, reset_every=20, n_envs=20):
    """Replay of train.py's training-order seeds for one seed_train."""
    np.random.seed(seed_train)
    n = (n_updates // reset_every) * n_envs
    return [int(np.random.randint(0, 2 ** 31 - 1)) for _ in range(n)]


def regen_matches(set_dir, seed0, n, n_modules, mix):
    from ppvc_instance_generator import (DEFAULT_FACTORY, load_instance,
                                         ppvc_instance_generator)
    bad = []
    for i in range(n):
        jl0, pt0, m0 = load_instance(os.path.join(set_dir, f'instance_{i:03d}'))
        jl, pt, m = ppvc_instance_generator(n_modules=n_modules, class_mix=mix,
                                            station_counts=DEFAULT_FACTORY,
                                            seed=seed0 + i)
        ok = (np.array_equal(jl, jl0) and np.array_equal(pt, pt0)
              and np.array_equal(m['time_lag'], m0['time_lag'])
              and np.array_equal(m['routing_class'], m0['routing_class']))
        if not ok:
            bad.append(i)
    return bad


def overlap(a0, an, b0, bn):
    lo, hi = max(a0, b0), min(a0 + an, b0 + bn)
    return max(0, hi - lo)


def audit():
    sets = set_ranges()
    notes = []
    for s in sets:
        if s['seed0'] is None and s['n'] > 0:
            nm = int(os.path.basename(s['set']).split('x')[0])
            seed0, both = identify_seed(os.path.join(REPO, s['set']), nm)
            s['seed0_identified'] = seed0
            s['identified_orders_0_and_1'] = both
            if seed0 is not None:
                s['seed0'] = seed0
            notes.append(f"{s['set']}: no dataset_meta.json; seed0 identified "
                         f"by regeneration = {seed0} (orders 0 and 1 match: {both})")
    # training runs of the SLAB recipe
    cfgs = {}
    train = {}
    for st in SEEDS:
        cp = os.path.join(REPO, 'train_log', 'PPVC',
                          f'config_{MODEL.format(s=st)}.json')
        c = json.load(open(cp))
        cfgs[st] = {k: c.get(k) for k in (
            'seed_train', 'num_envs', 'reset_env_timestep', 'max_updates',
            'ppvc_route_variant', 'ppvc_mix', 'ppvc_factory', 'lag_scale',
            'seed_train_vali_datagen', 'vali_size', 'ppvc_vali_sizes',
            'ppvc_lag_cv', 'size_mix', 'init_from')}
        assert c['seed_train'] == st and c['num_envs'] == 20 and \
            c['reset_env_timestep'] == 20 and c['max_updates'] == 2000
        train[st] = training_seed_draws(st)
    # the single-shell ranges to check: the two existing sets and the new ones
    targets = {}
    for name in OLD_SETS:
        m = json.load(open(os.path.join(DATA, name, 'dataset_meta.json')))
        targets[name] = (int(m['seed0']), int(m['n_instances']), int(m['n_modules']),
                         m['ppvc_mix'], 'existing')
    for name, (nm, mix, s0, n) in NEW_SETS.items():
        targets[name] = (s0, n, nm, mix, 'new')
    res = {}
    for name, (s0, n, nm, mix, kind) in targets.items():
        hits = []
        for s in sets:
            if s['seed0'] is None or os.path.basename(s['set']) == name:
                continue
            k = overlap(s0, n, s['seed0'], s['n'])
            if k:
                hits.append({'with': s['set'], 'kind': s['kind'],
                             'range': [s['seed0'], s['seed0'] + s['n'] - 1],
                             'n_shared_seeds': k})
        for other, (o0, on, _, _, _) in targets.items():
            if other != name and overlap(s0, n, o0, on):
                hits.append({'with': other, 'kind': 'single-shell',
                             'range': [o0, o0 + on - 1],
                             'n_shared_seeds': overlap(s0, n, o0, on)})
        tr_hits = {st: [x for x in train[st] if s0 <= x < s0 + n] for st in SEEDS}
        tr_min_dist = {st: int(min(min(abs(x - s0), abs(x - (s0 + n - 1)))
                                   for x in train[st])) for st in SEEDS}
        res[name] = {'seed0': s0, 'n': n, 'seed_range': [s0, s0 + n - 1],
                     'n_modules': nm, 'mix': mix, 'status': kind,
                     'overlaps_with_sets': hits,
                     'training_draws_in_range': {str(k): v for k, v in tr_hits.items()},
                     'training_draws_min_distance': {str(k): v for k, v in tr_min_dist.items()}}
        if kind == 'existing':
            bad = regen_matches(os.path.join(DATA, name), s0, n, nm, mix)
            res[name]['orders_not_regenerated_from_recorded_seed'] = bad
    out = {
        'script': 'eval_shell_projects.py --stage audit',
        'sets_on_disk': [{k: v for k, v in s.items() if k != 'meta'} for s in sets],
        'slab_training_configs': cfgs,
        'training_seed_stream': ('train.py: setup_seed(seed_train) at the start of '
                                 'train(); one np.random.randint(0, 2**31-1) per '
                                 'training order; 20 orders every 20 updates over '
                                 '2000 updates = 2000 draws per seed'),
        'training_draws_range': {str(st): [min(train[st]), max(train[st])]
                                 for st in SEEDS},
        'n_training_draws': {str(st): len(train[st]) for st in SEEDS},
        'validation_seeds': '100-199 at 10 and 20 modules (seed_train_vali_datagen=100, vali_size=100)',
        'single_shell': res,
        'notes': notes,
    }
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'seed_audit.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=1)
    for name, r in res.items():
        print(name, r['seed_range'], 'overlaps:', r['overlaps_with_sets'],
              'train hits:', r['training_draws_in_range'],
              'regen bad:', r.get('orders_not_regenerated_from_recorded_seed'))
    for n in notes:
        print(n)
    print('wrote', p)


# --------------------------------------------------------------------------- #
# make
# --------------------------------------------------------------------------- #
def make():
    from ppvc_instance_generator import (DEFAULT_FACTORY, ppvc_instance_generator,
                                         save_instance)
    au = json.load(open(os.path.join(OUT, 'seed_audit.json')))
    for name, (nm, mix, s0, n) in NEW_SETS.items():
        r = au['single_shell'][name]
        assert r['seed0'] == s0 and not r['overlaps_with_sets'] and \
            not any(r['training_draws_in_range'].values()), f'{name}: seeds overlap'
        d = os.path.join(DATA, name)
        os.makedirs(d, exist_ok=True)
        mp = os.path.join(d, 'dataset_meta.json')
        if os.path.exists(mp):
            assert json.load(open(mp))['seed0'] == s0, f'{d}: stale seed0'
        k = 0
        for i in range(n):
            stem = os.path.join(d, f'instance_{i:03d}')
            if os.path.exists(stem + '.fjs'):
                continue
            jl, pt, meta = ppvc_instance_generator(
                n_modules=nm, class_mix=mix, station_counts=DEFAULT_FACTORY,
                seed=s0 + i)
            save_instance(stem, jl, pt, meta)
            k += 1
        with open(mp, 'w') as f:
            json.dump({'seed0': s0, 'n_instances': n, 'n_modules': nm,
                       'ppvc_mix': mix, 'ppvc_factory': 'default',
                       'purpose': PURPOSE[nm]}, f, indent=1)
        print(f'{d}: {k} new instance(s), {n} total')


# --------------------------------------------------------------------------- #
# eval
# --------------------------------------------------------------------------- #
def evaluate(ds, method):
    from eval_byproducts import import_eval_ppvc, load_policy, run_validated
    ep = import_eval_ppvc()
    import torch
    from ppvc_instance_generator import load_instance
    from schedule_validator import validate_schedule
    stems = ep.list_instances(os.path.join(DATA, ds))
    meta = json.load(open(os.path.join(DATA, ds, 'dataset_meta.json')))
    assert len(stems) == int(meta['n_instances']), f'{ds}: instance count'
    if method.startswith('greedy-s'):
        st = int(method.split('-s')[1])
        model = MODEL.format(s=st)
        ppo, use_lag, use_type, ckpt = load_policy(ep, model, SEED_TEST)
        policy = (ppo, use_lag, use_type)
        methods = [('greedy', method)]
    elif method == 'rules':
        model, ckpt, policy = None, None, None
        methods = [(r, r) for r in RULES]
    else:
        raise ValueError(method)
    t0 = time.time()
    for m, label in methods:
        recs, arr = [], np.zeros((len(stems), 2))
        for i, stem in enumerate(stems):
            jl, pt, mt = load_instance(stem)
            ms, sec, asg, ct = run_validated(ep, m, jl, pt, mt, i, SEED_TEST,
                                             policy=policy)
            v = validate_schedule(jl, pt, mt['time_lag'], asg, ct)
            assert v['feasible'] and abs(v['makespan'] - ms) < 1e-6
            arr[i] = [ms, sec]
            recs.append({'instance': os.path.basename(stem), 'makespan': ms,
                         'seconds': sec, 'validator_feasible': bool(v['feasible']),
                         'validator_n_violations': len(v['violations']),
                         'validator_makespan': float(v['makespan'])})
        od = os.path.join(TR, ds)
        os.makedirs(od, exist_ok=True)
        pj = os.path.join(od, f'shell_{label}.json')
        pn = os.path.join(od, f'shell_Result_{label}_{ds}.npy')
        for p in (pj, pn):
            assert not os.path.exists(p), f'{p} exists; not overwritten'
        np.save(pn, arr)
        with open(pj, 'w') as f:
            json.dump({'dataset': ds, 'method': label, 'model': model,
                       'checkpoint': ckpt,
                       'checkpoint_sha256': sha256(os.path.join(REPO, ckpt)) if ckpt else None,
                       'device': 'cpu', 'decoding': 'greedy' if model else 'rule',
                       'seed_test': SEED_TEST,
                       'tie_break_seed': 'seed_test + order index' if not model else None,
                       'n_instances': len(stems), 'all_validated': True,
                       'torch_threads': torch.get_num_threads(),
                       'affinity': sorted(os.sched_getaffinity(0)),
                       'mean_makespan': float(arr[:, 0].mean()),
                       'per_instance': recs}, f, indent=1)
        print(f'[{ds}] {label}: mean {arr[:, 0].mean():.2f} h  '
              f'({time.time() - t0:.0f}s)  -> {os.path.relpath(pj, REPO)}', flush=True)


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #
def load_ms(ds, label):
    d = json.load(open(os.path.join(TR, ds, f'shell_{label}.json')))
    assert d['dataset'] == ds and d['all_validated']
    return np.array([r['makespan'] for r in d['per_instance']], float)


def compare(a, b):
    from stats_tests import (paired_mean_diff_ci, rank_biserial,
                             wilcoxon_paired)
    d, (lo, hi) = paired_mean_diff_ci(a, b, n_boot=N_BOOT, seed=BOOT_SEED)
    return {'mean_a': float(a.mean()), 'mean_b': float(b.mean()),
            'mean_diff': d, 'ci95': [lo, hi],
            'pct_diff': 100.0 * (a.mean() / b.mean() - 1.0),
            'shorter': int(np.sum(a < b)), 'equal': int(np.sum(a == b)),
            'longer': int(np.sum(a > b)),
            'p_wilcoxon': wilcoxon_paired(a, b), 'r_rank_biserial': rank_biserial(a, b)}


def stats():
    out = {'script': 'eval_shell_projects.py --stage stats',
           'conventions': ('mean paired difference SLAB minus rival with 95% percentile '
                           'bootstrap CI (2000 resamples, default_rng(0)); two-sided '
                           'Wilcoxon signed-rank, zero differences dropped; matched-pairs '
                           'rank-biserial r, r < 0 = SLAB shorter; SLAB seed mean = '
                           'per-order mean of seeds 301-303, all on the CPU'),
           'sets': {}}
    L = ['# Single-shell project orders: SLAB against the dispatching rules', '',
         'All schedules built on the CPU and re-validated against the true waits '
         '(schedule_validator.validate_schedule). SLAB = DANIEL policy `10x25+ppvc-mixed+a1lagx05-2k-s{301,302,303}`, '
         'greedy decoding. Rules in the SLAB transition, tie-break seed 50 + order index. '
         'Differences are SLAB minus rival (h), 95% percentile bootstrap CI from 2000 resamples (seed 0), '
         'two-sided Wilcoxon signed-rank p with zero differences dropped, rank-biserial r (r < 0: SLAB shorter). '
         'Counts are SLAB shorter / equal / longer.', '']
    for ds in EVAL_SETS:
        if not os.path.exists(os.path.join(TR, ds, 'shell_SPT.json')):
            continue
        meta = json.load(open(os.path.join(DATA, ds, 'dataset_meta.json')))
        means, files = {}, {}
        seeds = {st: load_ms(ds, f'greedy-s{st}') for st in SEEDS}
        sm = np.mean([seeds[st] for st in SEEDS], axis=0)
        for st in SEEDS:
            means[f'SLAB s{st}'] = float(seeds[st].mean())
        means['SLAB seed mean'] = float(np.mean([seeds[st].mean() for st in SEEDS]))
        means['SLAB seed std (population, of the seed means)'] = float(
            np.std([seeds[st].mean() for st in SEEDS]))
        rules = {r: load_ms(ds, r) for r in RULES}
        for r in RULES:
            means[r] = float(rules[r].mean())
        cmp = {}
        for rv in ('SPT', 'LLF'):
            cmp[f'seed mean vs {rv}'] = compare(sm, rules[rv])
        for st in SEEDS:
            cmp[f's{st} vs SPT'] = compare(seeds[st], rules['SPT'])
            cmp[f's{st} vs LLF'] = compare(seeds[st], rules['LLF'])
        best_rule = min(RULES, key=lambda r: rules[r].mean())
        cmp[f'seed mean vs best rule ({best_rule})'] = compare(sm, rules[best_rule])
        out['sets'][ds] = {'seed0': meta['seed0'], 'n_orders': meta['n_instances'],
                           'n_modules': meta['n_modules'], 'mix': meta['ppvc_mix'],
                           'mean_makespan': means, 'comparisons': cmp,
                           'best_rule': best_rule,
                           'sources': [os.path.relpath(os.path.join(TR, ds, f'shell_{x}.json'), REPO)
                                       for x in [f'greedy-s{s}' for s in SEEDS] + list(RULES)]}
        L += [f'## {ds} ({meta["n_instances"]} orders, {meta["n_modules"]} modules, '
              f'{meta["ppvc_mix"]}, seeds {meta["seed0"]}-{meta["seed0"] + meta["n_instances"] - 1})', '',
              '| Method | Mean makespan (h) |', '|---|---:|']
        for k, v in means.items():
            L.append(f'| {k} | {v:.2f} |')
        L += ['', '| Comparison | SLAB mean | Rival mean | Diff (h) | 95% CI | % | shorter/equal/longer | Wilcoxon p | r |',
              '|---|---:|---:|---:|---|---:|:-:|---:|---:|']
        for k, c in cmp.items():
            L.append(f"| {k} | {c['mean_a']:.2f} | {c['mean_b']:.2f} | {c['mean_diff']:+.2f} | "
                     f"[{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] | {c['pct_diff']:+.2f} | "
                     f"{c['shorter']}/{c['equal']}/{c['longer']} | {c['p_wilcoxon']:.2e} | "
                     f"{c['r_rank_biserial']:+.2f} |")
        L.append('')
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'shell_stats.json'), 'w') as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(OUT, 'shell_summary.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--stage', required=True, choices=('audit', 'make', 'eval', 'stats'))
    ap.add_argument('--set', choices=EVAL_SETS)
    ap.add_argument('--method', choices=[f'greedy-s{s}' for s in SEEDS] + ['rules'])
    a = ap.parse_args()
    # params.py parses sys.argv at import time (via data_utils / eval_ppvc)
    sys.argv = [sys.argv[0]]
    if a.stage == 'audit':
        audit()
    elif a.stage == 'make':
        make()
    elif a.stage == 'eval':
        if not (a.set and a.method):
            ap.error('eval needs --set and --method')
        evaluate(a.set, a.method)
    else:
        stats()


if __name__ == '__main__':
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    main()
