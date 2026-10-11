"""
Write variable-route PPVC test sets at production-batch sizes (50 and 80
modules) to data/PPVC/<n>x25+ppvc-mixed-var/.

Each instance i is generated as the 10- and 20-module variable-route sets were:
ppvc_instance_generator(n_modules, 'mixed', DEFAULT_FACTORY, seed=seed0 + i,
route_variant='variable'), followed by save_instance. The seed ranges are used
by no other set (Table S19 of the Supplementary Material lists every set).
Existing instance files are never overwritten.

usage: python make_var_testsets.py
"""
import json
import os

from ppvc_instance_generator import (DEFAULT_FACTORY, ppvc_instance_generator,
                                     save_instance)

PURPOSE = 'variable-route test set at production-batch size (modules of 18-22 operations)'
SETS = [  # (n_modules, seed0, n_instances)
    (50, 140000, 50),
    (80, 150000, 50),
]


def write_set(n_modules, seed0, n_instances, mix='mixed', factory='default'):
    save_dir = os.path.join('data', 'PPVC', f'{n_modules}x25+ppvc-{mix}-var')
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
            n_modules=n_modules, class_mix=mix, station_counts=DEFAULT_FACTORY,
            seed=seed0 + i, route_variant='variable')
        save_instance(stem, jl, pt, meta)
        n_new += 1
    ds_meta = {'seed0': seed0, 'n_instances': n_instances, 'n_modules': n_modules,
               'ppvc_mix': mix, 'ppvc_factory': factory,
               'route_variant': 'variable', 'purpose': PURPOSE}
    with open(meta_path, 'w') as f:
        json.dump(ds_meta, f, indent=1)
    print(f'{save_dir}: {n_new} new instance(s), {n_instances} total')


if __name__ == '__main__':
    for n_modules, seed0, n_instances in SETS:
        write_set(n_modules, seed0, n_instances)
