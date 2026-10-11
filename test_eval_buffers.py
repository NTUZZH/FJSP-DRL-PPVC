"""
test_eval_buffers.py
====================
Checks for eval_buffers.py. Run from the repository root:

    python test_eval_buffers.py            # all tests
    python test_eval_buffers.py occupancy  # tests whose name contains a word

The sub-commands run as subprocesses on a few instances and write into a
temporary directory, so test_results/ is never touched. The reference values
come from eval_buffer_occupancy.py (interval builder and sweep) and from the
rollouts of eval_ppvc.py.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

_WORDS = sys.argv[1:] if __name__ == '__main__' else []
sys.argv = sys.argv[:1]

import numpy as np  # noqa: E402

PY = sys.executable
DATA_10 = 'data/PPVC/10x25+ppvc-mixed'
DATA_5 = 'data/PPVC/5x9+ppvc-mixed'
MODEL = '10x25+ppvc-mixed+full'
SEED_TEST = 50
TMP = tempfile.mkdtemp(prefix='test_eval_buffers_')

_REF = {}


def run_cli(*args, timeout=900, out_dir=None):
    cmd = [PY, 'eval_buffers.py', *args, '--out_dir', out_dir or TMP]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    print(f'    $ {" ".join(cmd[1:])}   ({time.time() - t:.1f}s)')
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        raise AssertionError(f'command failed with code {p.returncode}')
    return p.stdout


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def reference_schedules(n):
    """A3 greedy and SPT schedules from eval_ppvc's own rollouts, with their
    occupancy computed by eval_buffer_occupancy's interval builder and sweep."""
    if n in _REF:
        return _REF[n]
    import eval_buffer_occupancy as ebo  # scrubs argv, imports params
    ebo.configs.device = 'cpu'
    ebo.load_train_config(MODEL)
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu')
    from common_utils import setup_seed
    setup_seed(SEED_TEST)
    from eval_ppvc import rollout_greedy, rollout_heuristic
    from ppvc_instance_generator import load_instance
    from model.PPO import PPO_initialize
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(
        f'./trained_network/PPVC/{MODEL}.pth', map_location='cpu'))
    ppo.policy.eval()
    ulf = bool(ebo.configs.use_lag_features)
    ute = bool(ebo.configs.use_type_embedding)
    out = {'greedy': [], 'SPT': []}
    for i, stem in enumerate(ebo.list_instances(DATA_10)[:n]):
        jl, pt, meta = load_instance(stem)
        for method in ('greedy', 'SPT'):
            if method == 'greedy':
                ms, _s, amch, oct_ = rollout_greedy(ppo, jl, pt, meta, ulf, ute)
            else:
                ms, _s, amch, oct_ = rollout_heuristic('SPT', jl, pt, meta,
                                                       seed=SEED_TEST + i)
            ivs = ebo.buffer_intervals(jl, meta, oct_)
            out[method].append({
                'instance': os.path.basename(stem), 'makespan': ms,
                'peak': ebo.peak_overlap(ivs),
                'peak_by_type': {t: ebo.peak_overlap(
                    [v for v in ivs if v[3] == t]) for t in ebo.LAG_TYPES}})
    _REF[n] = out
    return out


def independent_check(data_path, rec, B, mode='shared'):
    """Validator plus eval_buffer_occupancy's sweep on a stored schedule."""
    import eval_buffer_occupancy as ebo
    from ppvc_instance_generator import load_instance
    from schedule_validator import validate_schedule
    jl, pt, meta = load_instance(os.path.join(data_path, rec['instance']))
    res = validate_schedule(jl, pt, meta['time_lag'], rec['assigned_mch'],
                            rec['op_ct'])
    ivs = ebo.buffer_intervals(jl, meta, np.asarray(rec['op_ct'], float))
    if mode == 'shared':
        peak = ebo.peak_overlap(ivs)
    else:
        peak = max(ebo.peak_overlap([v for v in ivs if v[3] == t])
                   for t in ebo.LAG_TYPES)
    assert res['feasible'], res['violations'][:5]
    assert peak <= B, f'{rec["instance"]}: peak {peak} > B={B}'
    return res['makespan'], peak


# ---------------------------------------------------------------------------

def test_tracker_against_brute_force():
    """BufferTracker.max_occ / earliest_fit against a dense time grid."""
    import eval_buffers as eb
    rng = np.random.default_rng(0)
    for trial in range(200):
        n = int(rng.integers(0, 12))
        B = int(rng.integers(1, 4))
        win = np.array([float(rng.integers(1, 6))])
        tr = eb.BufferTracker(win, ['drying'], B, 'shared')
        tr.s['shared'] = [float(x) for x in rng.integers(0, 30, n)]
        tr.e['shared'] = [s + float(rng.integers(1, 8)) for s in tr.s['shared']]
        s = np.array(tr.s['shared'])
        e = np.array(tr.e['shared'])
        c0 = float(rng.integers(0, 30))
        L = float(win[0])
        grid = np.arange(0, 80, 0.5)
        occ = ((s[None, :] <= grid[:, None]) & (e[None, :] > grid[:, None])).sum(1)
        in_win = (grid >= c0) & (grid < c0 + L)
        want = int(occ[in_win].max()) if n else 0
        got = tr.max_occ('shared', c0, L)
        assert got == want, (trial, got, want)
        c = tr.earliest_fit(0, c0)
        # brute force over half-hour steps (all data are integers)
        for cc in np.arange(c0, 200, 0.5):
            m = (grid >= cc) & (grid < cc + L)
            if n == 0 or occ[m].max() < B:
                assert abs(cc - c) < 1e-9, (trial, cc, c)
                break
    print('    200 random window sets: max_occ and earliest_fit match a '
          'dense grid')


def test_occupancy_matches_eval_buffer_occupancy():
    run_cli('occupancy', '--data_path', DATA_10, '--max_instances', '3',
            '--methods', 'greedy', 'SPT', '--model_name', MODEL)
    ref = reference_schedules(3)
    saved = None
    npy = f'./test_results/PPVC/{os.path.basename(DATA_10)}/buffer_occupancy.npy'
    if os.path.exists(npy):
        saved = np.load(npy, allow_pickle=True).item()
    for method, label, saved_key in (('greedy', f'greedy+{MODEL}', 'A3'),
                                     ('SPT', 'SPT', 'SPT')):
        with open(os.path.join(TMP, f'buffers_occupancy_{label}.json')) as f:
            got = json.load(f)['per_instance']
        assert len(got) == 3
        for k, (g, r) in enumerate(zip(got, ref[method])):
            assert g['instance'] == r['instance']
            assert g['makespan'] == r['makespan'], (g['makespan'], r['makespan'])
            assert g['peak'] == r['peak'], (method, k, g['peak'], r['peak'])
            assert g['peak_by_type'] == r['peak_by_type'], (
                method, k, g['peak_by_type'], r['peak_by_type'])
            if saved is not None:
                assert g['peak'] == saved['peak_pooled'][saved_key][k]
                assert g['makespan'] == saved['makespan'][saved_key][k]
                for t in ('curing', 'ponding', 'drying'):
                    assert g['peak_by_type'][t] == \
                        saved['peak_by_type'][saved_key][t][k]
            print(f'    {method:<6} {g["instance"]}: makespan={g["makespan"]:.0f} '
                  f'peak={g["peak"]} by_type={g["peak_by_type"]}  == reference'
                  + ('  == buffer_occupancy.npy' if saved is not None else ''))


def test_masked_policy_b99_and_b2():
    run_cli('masked-policy', '--data_path', DATA_10, '--max_instances', '3',
            '--methods', 'greedy', 'SPT', '--B', '99', '2',
            '--model_name', MODEL, '--save_schedules')
    ref = reference_schedules(3)
    for method, label in (('greedy', f'greedy+{MODEL}'), ('SPT', 'SPT')):
        recs = read_jsonl(os.path.join(TMP,
                                       f'buffers_masked_{label}_shared.jsonl'))
        by = {(r['instance'], r['B']): r for r in recs}
        for r in ref[method]:
            inst = r['instance']
            free, b99, b2 = by[(inst, None)], by[(inst, 99)], by[(inst, 2)]
            assert free['objective'] == r['makespan']
            assert b99['objective'] == r['makespan'], (
                method, inst, b99['objective'], r['makespan'])
            assert b99['n_masked_decisions'] == 0
            assert np.array_equal(b99['op_ct'], free['op_ct'])
            ms2, peak2 = independent_check(DATA_10, b2, 2)
            assert ms2 == b2['objective'] and b2['within_capacity']
            print(f'    {method:<6} {inst}: unmasked={r["makespan"]:.0f} '
                  f'B=99 -> {b99["objective"]:.0f}  B=2 -> '
                  f'{b2["objective"]:.0f} (peak {peak2}, '
                  f'{b2["n_wait_decisions"]} wait decisions, '
                  f'{b2["n_delayed_ops"]} delayed ops)')


def test_masked_policy_per_type():
    run_cli('masked-policy', '--data_path', DATA_10, '--max_instances', '2',
            '--methods', 'SPT', '--buffer', 'per-type', '--B', '1',
            '--save_schedules')
    recs = read_jsonl(os.path.join(TMP, 'buffers_masked_SPT_per-type.jsonl'))
    for r in recs:
        if r['B'] == 1:
            ms, peak = independent_check(DATA_10, r, 1, mode='per-type')
            print(f'    SPT per-type B=1 {r["instance"]}: makespan={ms:.0f} '
                  f'max per-type peak={peak} '
                  f'(shared peak {r["occupancy"]["peak"]})')


def _check_cpsat(data_path, n, limit, exact):
    tl = f'{float(limit):g}'
    ds = os.path.basename(data_path)
    out = os.path.join(TMP, ds)
    run_cli('cpsat-capacity', '--data_path', data_path, '--max_instances',
            str(n), '--B', '99', '1', '--time_limit', tl, '--num_workers', '2',
            out_dir=out)
    recs = read_jsonl(os.path.join(out, f'buffers_cpsat_shared_t{tl}.jsonl'))
    ref = {}
    ref_path = f'./or_solution/PPVC/{ds}.jsonl'
    if os.path.exists(ref_path):
        ref = {r['instance']: r for r in read_jsonl(ref_path)}
    by = {(r['instance'], r['B']): r for r in recs}
    insts = sorted({r['instance'] for r in recs})
    assert len(insts) == n
    for inst in insts:
        free, b99, b1 = by[(inst, None)], by[(inst, 99)], by[(inst, 1)]
        for r in (free, b99, b1):
            assert r['feasible'] and r['within_capacity'], r['status']
        independent_check(data_path, b1, 1)
        independent_check(data_path, b99, 99)
        # B=99 exceeds the number of windows, so both models share one optimum
        assert b99['objective'] >= free['bound'] - 1e-9
        assert free['objective'] >= b99['bound'] - 1e-9
        if exact:
            assert free['status'] == b99['status'] == 'OPTIMAL'
            assert b99['objective'] == free['objective']
            if inst in ref and ref[inst].get('status_lag') == 'OPTIMAL':
                assert free['objective'] == ref[inst]['makespan_lag']
        note = 'equal' if b99['objective'] == free['objective'] else 'DIFFERENT'
        print(f'    {ds} {inst}: free={free["objective"]:.0f} '
              f'[{free["status"]}, bound {free["bound"]:.0f}]  '
              f'B=99={b99["objective"]:.0f} [{b99["status"]}, bound '
              f'{b99["bound"]:.0f}] ({note})  B=1={b1["objective"]:.0f} '
              f'[{b1["status"]}] peak={b1["occupancy"]["peak"]}  '
              f'or_solution ref={ref.get(inst, {}).get("makespan_lag")}')
    return by, insts


def test_cpsat_capacity_small_exact():
    """5x9: CP-SAT proves optimality, so B=99 must equal the unconstrained
    optimum exactly and match or_solution."""
    _check_cpsat(DATA_5, 2, 60, exact=True)


def test_cpsat_capacity_10x25():
    """10x25 at 60 s: B=99 reproduces the unconstrained makespan; B=1 is
    validator-clean with peak <= 1."""
    by, insts = _check_cpsat(DATA_10, 2, 60, exact=False)
    for inst in insts:
        assert by[(inst, 99)]['objective'] == by[(inst, None)]['objective'], (
            inst, by[(inst, 99)]['objective'], by[(inst, None)]['objective'])


TESTS = [
    test_tracker_against_brute_force,
    test_occupancy_matches_eval_buffer_occupancy,
    test_masked_policy_b99_and_b2,
    test_masked_policy_per_type,
    test_cpsat_capacity_small_exact,
    test_cpsat_capacity_10x25,
]


if __name__ == '__main__':
    words = _WORDS
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f'output dir: {TMP}')
    failed = 0
    for t in TESTS:
        if words and not any(w in t.__name__ for w in words):
            continue
        t0 = time.time()
        print(f'[RUN ] {t.__name__}')
        try:
            t()
            print(f'[PASS] {t.__name__}  ({time.time() - t0:.1f}s)')
        except Exception as ex:  # report and continue with the next test
            failed += 1
            import traceback
            traceback.print_exc()
            print(f'[FAIL] {t.__name__}: {ex!r}')
    print(f'{failed} failed')
    if not failed:
        import shutil
        shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(1 if failed else 0)
