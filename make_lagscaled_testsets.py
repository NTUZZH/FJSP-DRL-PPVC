"""
Write test sets with every time-lag scaled, for the shifted-lag transfer study.

Each instance of a source test set is copied with its lags multiplied by the
scale and rounded (np.rint), exactly as train.py --lag_scale scales training
orders. Routes, processing times and stations are unchanged, so results on
the scaled set are paired with results on the source set instance by
instance. Existing files are never overwritten.

usage: python make_lagscaled_testsets.py
"""
import json
import os

import numpy as np

from ppvc_instance_generator import load_instance, save_instance

SETS = [  # (source set, lag scale)
    ('10x25+ppvc-mixed', 1.5),
    ('50x25+ppvc-mixed', 1.5),
]


def write_set(src, scale):
    src_dir = os.path.join('data', 'PPVC', src)
    dst_dir = os.path.join('data', 'PPVC', f'{src}-lagx{scale:g}')
    os.makedirs(dst_dir, exist_ok=True)
    with open(os.path.join(src_dir, 'dataset_meta.json')) as f:
        src_meta = json.load(f)
    stems = sorted(f[:-4] for f in os.listdir(src_dir) if f.startswith('instance_') and f.endswith('.fjs'))
    n_new = 0
    for stem in stems:
        out = os.path.join(dst_dir, stem)
        if os.path.exists(out + '.fjs'):
            continue
        jl, pt, meta = load_instance(os.path.join(src_dir, stem))
        meta = dict(meta)
        meta['time_lag'] = np.rint(np.asarray(meta['time_lag'], dtype=float) * scale).astype(int)
        save_instance(out, jl, pt, meta)
        n_new += 1
    dst_meta = dict(src_meta, lag_scale=scale, source_set=src,
                    purpose=f'shifted-lag transfer test set: {src} with every lag x{scale:g} (rounded)')
    with open(os.path.join(dst_dir, 'dataset_meta.json'), 'w') as f:
        json.dump(dst_meta, f, indent=1)
    print(f'{dst_dir}: {n_new} new instances, {len(stems)} total')


if __name__ == '__main__':
    for s, k in SETS:
        write_set(s, k)
