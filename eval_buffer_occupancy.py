"""
eval_buffer_occupancy.py
========================
Lag-buffer occupancy analysis for the PPVC paper.

The lag-aware env models curing / ponding / paint-drying as a post-operation
TIME-LAG: after op O_ij completes at c_ij, the MODULE is held for lag_ij hours
before its job successor becomes eligible (the machine is free during the lag).
A natural question is whether this implicitly assumes UNBOUNDED lag-buffer
capacity (curing yards / ponding tanks / drying racks). This script quantifies how many
modules are SIMULTANEOUSLY sitting in a lag-buffer under the learned policy's
schedules vs the best PDR baseline, so we can state empirically how many buffer
slots a finite-capacity factory would actually need.

DEFINITION
  A module occupies a lag buffer over the half-open interval
        [c_ij, c_ij + lag_ij)
  for every NON-TERMINAL op O_ij with lag_ij > 0 (curing/ponding/drying).
  PEAK occupancy of a schedule = the maximum number of these intervals that
  overlap at any single instant (a sweep-line over interval endpoints). This is
  exactly the minimum number of buffer slots under which that schedule is
  feasible.

We reuse the VALIDATED rollout + feasibility machinery from eval_ppvc.py
verbatim (rollout_greedy / rollout_heuristic / validate_schedule) so the
schedules are identical to the paper's main table; every schedule is
re-validated against the TRUE lags before its occupancy is computed.

METHODS
  (a) greedy  : the trained DANIEL model 10x25+ppvc-mixed+full
  (b) SPT     : best PDR baseline

OUTPUT
  - console table: method x {mean / median / max peak occupancy (pooled),
    mean makespan}
  - per-lag-type peaks (curing / ponding / drying), mapped via op_name
  - 2 spot-check instances printed in full (interval set + computed max overlap)
  - a copy-paste LaTeX \newcommand macro block
  - raw per-instance arrays saved to
        test_results/PPVC/10x25+ppvc-mixed/buffer_occupancy.npy

Run:
  python eval_buffer_occupancy.py
"""
import glob
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# params.py parses sys.argv at import time. Scrub argv to just the program name
# BEFORE importing anything that pulls in params (mirrors eval_ppvc.parse_cli).
# We take no CLI flags here -- everything is fixed for this analysis.
# ---------------------------------------------------------------------------
sys.argv = [sys.argv[0]]

MODEL_NAME = '10x25+ppvc-mixed+full'
DATA_PATH = 'data/PPVC/10x25+ppvc-mixed'
SEED_TEST = 50

from params import configs

# CRITICAL: force CPU so we do not contend with another GPU job. Set this on the
# configs singleton BEFORE the network is built (PPO_initialize reads it) and
# before any env constructs an EnvState (EnvState.device reads it at class
# definition time, but we re-point torch's default device below regardless).
configs.device = 'cpu'

# Architecture-critical keys copied from the training-config snapshot (identical
# list to eval_ppvc.ARCH_KEYS) so the checkpoint loads into a matching DANIEL.
ARCH_KEYS = (
    'use_lag_features', 'use_type_embedding',
    'fea_j_input_dim', 'fea_m_input_dim',
    'type_emb_dim', 'n_op_types', 'n_mch_types',
    'n_j', 'n_m', 'n_op',
    'num_heads_OAB', 'num_heads_MAB', 'layer_fea_output_dim',
    'num_mlp_layers_actor', 'hidden_dim_actor',
    'num_mlp_layers_critic', 'hidden_dim_critic',
    'dropout_prob',
)


def load_train_config(model_name):
    path = f'./train_log/PPVC/config_{model_name}.json'
    if not os.path.exists(path):
        sys.exit(f'[buffer] training-config snapshot not found: {path}')
    with open(path) as f:
        snap = json.load(f)
    for k in ARCH_KEYS:
        if k in snap:
            setattr(configs, k, snap[k])
    # keep CPU even though the snapshot says cuda
    configs.device = 'cpu'
    return snap


def list_instances(data_path):
    fjs = sorted(glob.glob(os.path.join(data_path, 'instance_*.fjs')))
    if not fjs:
        sys.exit(f'[buffer] no instance_*.fjs found under {data_path}')
    return [p[:-len('.fjs')] for p in fjs]


# ---------------------------------------------------------------------------
# Lag-type mapping. The three physical lag mechanisms map CLEANLY to operation
# names (op_type alone does NOT separate ponding from drying -- both are
# op_type=2 "finishing"). See ppvc_instance_generator ROUTES:
#   curing  : concrete_pour            (24-48h, [N] G 4.1.2)
#   ponding : waterproofing            (24-48h, [N] G 4.1.3/4.2.2)
#   drying  : paint_coat_1, paint_coat_2 (12-24h, [N] G 4.2.2/10.2)
# Every other op has lag==0. We map each op to its category from meta['op_name']
# (the 'M<j>:<base_name>' convention) and assert the mapping covers EVERY lag>0
# op so the per-type split is exhaustive (never silently drops a buffered op).
# ---------------------------------------------------------------------------
LAG_TYPE_BY_OPNAME = {
    'concrete_pour': 'curing',
    'waterproofing': 'ponding',
    'paint_coat_1': 'drying',
    'paint_coat_2': 'drying',
}
LAG_TYPES = ('curing', 'ponding', 'drying')


def op_lag_categories(meta):
    """Return a category string per op ('curing'/'ponding'/'drying'/None)."""
    cats = []
    for nm in meta['op_name']:
        base = nm.split(':', 1)[1] if ':' in nm else nm
        cats.append(LAG_TYPE_BY_OPNAME.get(base))
    return cats


def buffer_intervals(jl, meta, true_op_ct):
    """Build the lag-buffer occupancy intervals [c_ij, c_ij + lag_ij).

    Only NON-TERMINAL ops with lag>0 occupy a buffer (the terminal op of a job
    has no successor to wait on; the generator already sets terminal lags to 0,
    but we exclude job-last ops defensively). Returns:
        intervals : list of (start, end, op_idx, category)
    """
    jl = np.asarray(jl, dtype=int)
    time_lag = np.asarray(meta['time_lag'], dtype=float)
    cats = op_lag_categories(meta)
    n_ops = len(time_lag)

    # job-last op ids (terminal ops): cumulative end of each job - 1
    job_first = np.concatenate([[0], np.cumsum(jl)[:-1]]).astype(int)
    job_last = job_first + jl - 1
    is_terminal = np.zeros(n_ops, dtype=bool)
    is_terminal[job_last] = True

    intervals = []
    for i in range(n_ops):
        lag = time_lag[i]
        if lag > 0 and not is_terminal[i]:
            c = float(true_op_ct[i])
            intervals.append((c, c + lag, i, cats[i]))
    return intervals


def peak_overlap(intervals):
    """Max number of half-open intervals [s,e) overlapping at any instant.

    Clean sweep-line: a +1 event at each start, a -1 event at each end. Because
    intervals are half-open [s, e), an interval ENDING at time t does NOT
    overlap one STARTING at t, so we must process end (-1) events BEFORE start
    (+1) events at the same coordinate. We sort by (coord, delta) with delta
    ascending so -1 precedes +1.
    """
    if not intervals:
        return 0
    events = []
    for (s, e, _i, _c) in intervals:
        events.append((s, +1))
        events.append((e, -1))
    # at a tie, process -1 (ends) before +1 (starts): delta ascending
    events.sort(key=lambda x: (x[0], x[1]))
    cur = 0
    peak = 0
    for (_t, d) in events:
        cur += d
        if cur > peak:
            peak = cur
    return peak


def peak_overlap_verbose(intervals):
    """Same as peak_overlap but also returns the instant where the peak occurs
    (for the spot-check printout)."""
    if not intervals:
        return 0, None
    events = []
    for (s, e, _i, _c) in intervals:
        events.append((s, +1))
        events.append((e, -1))
    events.sort(key=lambda x: (x[0], x[1]))
    cur = 0
    peak = 0
    peak_t = None
    for (t, d) in events:
        cur += d
        if cur > peak:
            peak = cur
            peak_t = t
    return peak, peak_t


def main():
    snap = load_train_config(MODEL_NAME)
    use_lag_features = bool(getattr(configs, 'use_lag_features', False))
    use_type_embedding = bool(getattr(configs, 'use_type_embedding', False))

    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # hard-hide the GPU
    import torch
    from common_utils import setup_seed
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')
    setup_seed(SEED_TEST)

    # import the VALIDATED rollouts + validator from eval_ppvc (no reimplementation)
    from eval_ppvc import rollout_greedy, rollout_heuristic
    from schedule_validator import validate_schedule
    from ppvc_instance_generator import load_instance
    from model.PPO import PPO_initialize

    ckpt_path = f'./trained_network/PPVC/{MODEL_NAME}.pth'
    if not os.path.exists(ckpt_path):
        sys.exit(f'[buffer] checkpoint not found: {ckpt_path}')
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    ppo.policy.eval()

    stems = list_instances(DATA_PATH)
    n = len(stems)
    dataset = os.path.basename(os.path.normpath(DATA_PATH))

    print('=' * 74)
    print(f'eval_buffer_occupancy  model={MODEL_NAME}  dataset={dataset}  '
          f'instances={n}')
    print(f'  device=cpu  use_lag_features={use_lag_features}  '
          f'use_type_embedding={use_type_embedding}  seed_test={SEED_TEST}')
    print('  lag-buffer interval = [c_ij, c_ij + lag_ij) for non-terminal '
          'lag>0 ops')
    print('=' * 74)

    methods = ['A3', 'SPT']  # A3 == greedy DANIEL; SPT == best PDR
    # per-instance peak (pooled) and makespan
    peak_pooled = {m: np.zeros(n) for m in methods}
    makespan = {m: np.zeros(n) for m in methods}
    # per-type peak [n_types]
    peak_by_type = {m: {t: np.zeros(n) for t in LAG_TYPES} for m in methods}
    feas = {m: 0 for m in methods}
    infeasible = {m: [] for m in methods}

    # sanity: confirm the op_name lag-mapping is exhaustive on instance 0
    jl0, pt0, meta0 = load_instance(stems[0])
    cats0 = op_lag_categories(meta0)
    tl0 = np.asarray(meta0['time_lag'])
    unmapped = [i for i in range(len(tl0)) if tl0[i] > 0 and cats0[i] is None]
    if unmapped:
        sys.exit(f'[buffer] FATAL: {len(unmapped)} lag>0 ops on instance_000 '
                 f'are not mapped to a lag category by op_name -> per-type '
                 f'split would be unreliable. ops={unmapped[:10]}')
    print(f'  lag-type mapping check on instance_000: all '
          f'{int((tl0 > 0).sum())} lag>0 ops mapped '
          f'(curing/ponding/drying via op_name) OK')
    print('=' * 74)

    spot_printed = 0

    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        base = os.path.basename(stem)
        time_lag = np.asarray(meta['time_lag'], dtype=float)

        for method in methods:
            if method == 'A3':
                ms, sec, amch, oct_ = rollout_greedy(
                    ppo, jl, pt, meta, use_lag_features, use_type_embedding)
            else:  # SPT, per-instance reseed matching eval_ppvc convention
                ms, sec, amch, oct_ = rollout_heuristic(
                    'SPT', jl, pt, meta, seed=SEED_TEST + i)

            # re-validate against the TRUE lags BEFORE computing occupancy
            res = validate_schedule(jl, pt, time_lag, amch, oct_)
            if not res['feasible']:
                infeasible[method].append((base, res['violations'][:5]))
                print(f'  !!! INFEASIBLE {method} on {base}:')
                for v in res['violations'][:5]:
                    print(f'        {v}')
                continue
            assert abs(res['makespan'] - ms) < 1e-6, (
                f'{method} {base}: env makespan {ms} != validator '
                f'{res["makespan"]}')
            feas[method] += 1

            intervals = buffer_intervals(jl, meta, oct_)
            pk = peak_overlap(intervals)
            peak_pooled[method][i] = pk
            makespan[method][i] = ms

            for t in LAG_TYPES:
                sub = [iv for iv in intervals if iv[3] == t]
                peak_by_type[method][t][i] = peak_overlap(sub)

            # ----- spot-check printout for the first 2 instances (A3 only,
            # then SPT on the same instance) -----
            if i < 2:
                print(f'\n  --- SPOT-CHECK {method} on {base} '
                      f'(makespan={ms:.1f}h) ---')
                ivs = sorted(intervals, key=lambda x: x[0])
                print(f'    {len(ivs)} lag-buffer intervals '
                      f'[start, end)  op  category :')
                for (s, e, opi, cat) in ivs:
                    print(f'      [{s:7.2f}, {e:7.2f})  op{opi:>4}  {cat}')
                pk_v, pk_t = peak_overlap_verbose(intervals)
                # explicit list of intervals live at the peak instant
                if pk_t is not None:
                    live = [opi for (s, e, opi, cat) in ivs
                            if s <= pk_t < e]
                    print(f'    => PEAK overlap = {pk_v} modules at t={pk_t:.2f}h'
                          f'  (live ops: {live})')
                else:
                    print(f'    => PEAK overlap = {pk_v} (no lag intervals)')

        if (i + 1) % 10 == 0 or i == n - 1:
            print(f'  [{i + 1:3d}/{n}]  '
                  f'A3 peak={peak_pooled["A3"][i]:.0f} ms={makespan["A3"][i]:.0f}h'
                  f'   SPT peak={peak_pooled["SPT"][i]:.0f} '
                  f'ms={makespan["SPT"][i]:.0f}h')

    # ---- feasibility gate ----
    print('\n' + '=' * 74)
    all_feasible = True
    for m in methods:
        if feas[m] != n:
            all_feasible = False
            print(f'  *** {m}: ONLY {feas[m]}/{n} schedules feasible '
                  f'-- {len(infeasible[m])} FAILURES. Aggregates below EXCLUDE '
                  f'the failures and MUST NOT go in the paper as-is.')
        else:
            print(f'  {m}: {feas[m]}/{n} schedules feasible (validated against '
                  f'true lags) OK')
    print('=' * 74)

    # ---- aggregate (only over feasible instances) ----
    def agg(method):
        ok = makespan[method] >= 0  # all entries; failures left at 0 ms & 0 peak
        # restrict to feasible instances explicitly
        mask = np.array([True] * n)
        if feas[method] != n:
            # mark infeasible-instance slots: they were `continue`d, so peak/ms
            # stayed 0; safest is to recompute the feasible mask. We tracked
            # only basenames, so reconstruct from makespan>0 (every feasible
            # PPVC schedule has makespan > 0).
            mask = makespan[method] > 0
        pk = peak_pooled[method][mask]
        ms = makespan[method][mask]
        return pk, ms

    print('\nLAG-BUFFER PEAK CONCURRENT OCCUPANCY (pooled across all lag types)')
    print(f'{"method":<8}{"mean":>9}{"median":>9}{"max":>7}{"min":>7}'
          f'{"mean_makespan":>16}')
    rows = {}
    for m in methods:
        pk, ms = agg(m)
        rows[m] = dict(mean=pk.mean(), median=float(np.median(pk)),
                       max=int(pk.max()), min=int(pk.min()),
                       mean_ms=ms.mean())
        print(f'{m:<8}{rows[m]["mean"]:>9.2f}{rows[m]["median"]:>9.1f}'
              f'{rows[m]["max"]:>7d}{rows[m]["min"]:>7d}'
              f'{rows[m]["mean_ms"]:>16.1f}')

    # ---- per-type peaks ----
    print('\nPER-LAG-TYPE PEAK CONCURRENT OCCUPANCY '
          '(mapped via op_name -- reliable split)')
    print(f'{"method":<8}{"type":<10}{"mean":>9}{"median":>9}{"max":>7}')
    type_rows = {m: {} for m in methods}
    for m in methods:
        mask = makespan[m] > 0 if feas[m] != n else np.array([True] * n)
        for t in LAG_TYPES:
            arr = peak_by_type[m][t][mask]
            type_rows[m][t] = dict(mean=arr.mean(),
                                   median=float(np.median(arr)),
                                   max=int(arr.max()))
            print(f'{m:<8}{t:<10}{arr.mean():>9.2f}'
                  f'{float(np.median(arr)):>9.1f}{int(arr.max()):>7d}')

    # ---- sanity vs paper ----
    print('\nMAKESPAN SANITY CHECK vs paper (A3 greedy ~210.9h, SPT ~216.0h):')
    print(f'  A3  mean makespan = {rows["A3"]["mean_ms"]:.1f}h')
    print(f'  SPT mean makespan = {rows["SPT"]["mean_ms"]:.1f}h')

    # ---- save raw arrays ----
    save_dir = f'./test_results/PPVC/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'buffer_occupancy.npy')
    blob = {
        'methods': methods,
        'peak_pooled': {m: peak_pooled[m] for m in methods},
        'makespan': {m: makespan[m] for m in methods},
        'peak_by_type': {m: {t: peak_by_type[m][t] for t in LAG_TYPES}
                         for m in methods},
        'lag_types': LAG_TYPES,
        'feasible_counts': feas,
        'n_instances': n,
    }
    np.save(out_path, blob, allow_pickle=True)
    print(f'\nsaved raw per-instance arrays -> {out_path}')

    # ---- LaTeX macro block ----
    print('\n' + '=' * 74)
    print('LaTeX macro block (copy-paste into macros.tex):')
    print('=' * 74)

    def fmt_mean(x):
        return f'{x:.1f}'

    macros = []
    macros.append(f'\\newcommand{{\\BufNumInstances}}{{{n}}}')
    macros.append(f'\\newcommand{{\\BufA3PeakMax}}{{{rows["A3"]["max"]}}}')
    macros.append(f'\\newcommand{{\\BufA3PeakMean}}{{{fmt_mean(rows["A3"]["mean"])}}}')
    macros.append(f'\\newcommand{{\\BufA3PeakMedian}}{{{rows["A3"]["median"]:.0f}}}')
    macros.append(f'\\newcommand{{\\BufA3PeakMin}}{{{rows["A3"]["min"]}}}')
    macros.append(f'\\newcommand{{\\BufSptPeakMax}}{{{rows["SPT"]["max"]}}}')
    macros.append(f'\\newcommand{{\\BufSptPeakMean}}{{{fmt_mean(rows["SPT"]["mean"])}}}')
    macros.append(f'\\newcommand{{\\BufSptPeakMedian}}{{{rows["SPT"]["median"]:.0f}}}')
    macros.append(f'\\newcommand{{\\BufSptPeakMin}}{{{rows["SPT"]["min"]}}}')
    macros.append(f'\\newcommand{{\\BufA3MakespanMean}}{{{rows["A3"]["mean_ms"]:.1f}}}')
    macros.append(f'\\newcommand{{\\BufSptMakespanMean}}{{{rows["SPT"]["mean_ms"]:.1f}}}')
    # per-type maxima (the binding numbers per buffer kind)
    for m, tag in (('A3', 'A3'), ('SPT', 'Spt')):
        for t in LAG_TYPES:
            tt = t.capitalize()
            macros.append(
                f'\\newcommand{{\\Buf{tag}{tt}PeakMax}}'
                f'{{{type_rows[m][t]["max"]}}}')
    for line in macros:
        print(line)
    print('=' * 74)

    # ---- prose summary with real numbers ----
    a3max, sptmax = rows['A3']['max'], rows['SPT']['max']
    a3mean, sptmean = rows['A3']['mean'], rows['SPT']['mean']
    verdict = ('better (smaller peak)' if a3max < sptmax else
               'comparable' if a3max == sptmax else
               'worse (larger peak)')
    modest = 'modest' if a3max <= 15 else 'non-trivial'
    print('\nPROSE SUMMARY')
    print('-' * 74)
    print(
        f'Across the {n} M-class test instances, the LARGEST number of modules '
        f'ever simultaneously\noccupying a lag buffer under the learned A3 '
        f'policy is {a3max} (mean {a3mean:.1f} per instance);\nfor SPT it is '
        f'{sptmax} (mean {sptmean:.1f}). A3 is {verdict} than SPT on this '
        f'metric.\nThis peak is {modest} relative to the {jl0.shape[0]} '
        f'modules in flight, so a factory provisioning\nabout {a3max} buffer '
        f'slots would satisfy EVERY A3 schedule in the test set; the '
        f'unbounded-\nbuffer assumption is therefore empirically benign at this '
        f'scale rather than a hidden\ninfeasibility. Per-type maxima: '
        f'curing={type_rows["A3"]["curing"]["max"]}, '
        f'ponding={type_rows["A3"]["ponding"]["max"]}, '
        f'drying={type_rows["A3"]["drying"]["max"]} (A3).')
    print('-' * 74)

    if not all_feasible:
        print('\n*** WARNING: not all schedules were feasible -- DO NOT use '
              'these aggregates in the paper until resolved. ***')
        sys.exit(2)


if __name__ == '__main__':
    main()
