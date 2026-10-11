"""
Write independent confirmation test sets for the PPVC ladder to data/PPVC/.

A configuration chosen after looking at the main test sets (10/20/30/50/80
modules) is confirmed on these fresh instances, generated exactly like the
main sets but from generator seeds that no other set or training run uses.
Existing instance files are never overwritten.

usage: python make_independent_testsets.py
"""
import json
import os

from ppvc_instance_generator import (DEFAULT_FACTORY, ppvc_instance_generator,
                                     save_instance)

PURPOSE = 'independent confirmation test set (eval-only), seeds disjoint from all other sets'
SETS = [  # (n_modules, seed0, n_instances)
    (10, 720000, 100),
    (20, 730000, 50),
    (50, 700000, 50),
    (80, 710000, 50),
]


def write_set(n_modules, seed0, n_instances, mix='mixed'):
    save_dir = os.path.join('data', 'PPVC', f'{n_modules}x25+ppvc-{mix}-ind')
    os.makedirs(save_dir, exist_ok=True)
    meta_path = os.path.join(save_dir, 'dataset_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            old = json.load(f)
        assert old['seed0'] == seed0, f'{save_dir}: stale seed0 {old["seed0"]}'
    n_new = 0
    for i in range(n_instances):
        stem = os.path.join(save_dir, f'instance_{i:03d}')
        if os.path.exists(stem + '.fjs'):
            continue
        jl, pt, meta = ppvc_instance_generator(
            n_modules=n_modules, class_mix=mix,
            station_counts=DEFAULT_FACTORY, seed=seed0 + i)
        save_instance(stem, jl, pt, meta)
        n_new += 1
    with open(meta_path, 'w') as f:
        json.dump({'seed0': seed0, 'n_instances': n_instances, 'n_modules': n_modules,
                   'ppvc_mix': mix, 'ppvc_factory': 'default', 'purpose': PURPOSE}, f, indent=1)
    print(f'{save_dir}: {n_new} new instances, {n_instances} total')


if __name__ == '__main__':
    for n, s0, k in SETS:
        write_set(n, s0, k)
