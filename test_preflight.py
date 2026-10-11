"""
test_preflight.py
=================
Checks for the revision training queue. CPU only.

  1. preflight_train.py on queue arms 1, 2, 6 and 9, with the exact flags the
     queue passes (parsed from its --dry_run listing): parameter counts
     (32,530 / 32,530 / 28,834 / 13,506), resolved model names, and the
     comparator diffs of arms 1 and 2. Arms 6 and 9 are resolved for both
     headlines; their comparators are snapshots the queue creates first, so
     their --compare is dropped while those files do not exist.
  2. --bound_lag_free: default and False keep the lag-augmented bound
     (checked against an independent formula), True on zero-lag instances
     changes nothing, True on lagged instances gives a smaller initial bound
     while the dynamics (schedule, completion times) are unchanged and the
     schedule is validator-clean; eval_ppvc.py applies the flag from a
     snapshot.
  3. bash -n on run_revision_gpu_queue.sh and its --dry_run listing (printed).
  4. (--train_smoke, about 6 minutes) train.py with --max_updates 1 on the
     CPU for a DANIEL and a PLAN arm in a temporary directory: the checkpoint
     name and the config snapshot equal what preflight_train.py resolves.

Run:  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 22-23 \
          python test_preflight.py [--train_smoke]
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
QUEUE = os.path.join(REPO, 'run_revision_gpu_queue.sh')
CHILD_ENV = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')

RESULTS = []


def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok)))
    print(f'[{"PASS" if ok else "FAIL"}] {name}' + (f'  {detail}' if detail else ''))


# ---------------------------------------------------------------------------
# queue listing -> preflight commands
# ---------------------------------------------------------------------------

def dry_run(*args):
    out = subprocess.run(['bash', QUEUE, '--dry_run', *args], cwd=REPO, env=CHILD_ENV,
                         capture_output=True, text=True)
    return out.returncode, out.stdout + out.stderr


def preflight_commands(listing):
    """{arm id: (model name, preflight argv)} from a --dry_run listing."""
    arms, current = {}, None
    for line in listing.splitlines():
        if '=== ARM ' in line and ': ' in line.split('=== ARM ', 1)[1]:
            head = line.split('=== ARM ', 1)[1]
            arm_id = head.split()[0]
            current = (arm_id, head.split(': ', 1)[1].strip())
        elif current and 'preflight_train.py' in line and line.strip().startswith('+'):
            toks = line.strip()[1:].split()
            argv = toks[toks.index('preflight_train.py') + 1:]
            arms[current[0]] = (current[1], argv)
    return arms


def strip_opt(argv, opt, has_value=True):
    out, skip = [], False
    for i, t in enumerate(argv):
        if skip:
            skip = False
            continue
        if t == opt:
            skip = has_value
            continue
        out.append(t)
    return out


def run_preflight(argv):
    out = subprocess.run([PY, os.path.join(REPO, 'preflight_train.py'), *argv], cwd=REPO,
                         env=CHILD_ENV, capture_output=True, text=True)
    return out.returncode, out.stdout, out.stderr


def summary_lines(stdout):
    return [ln for ln in stdout.splitlines()
            if ln.startswith(('PREFLIGHT', 'DIFF', '  [data]', '  [test]', '  absent', '  allowed'))
            or ('[allowed]' in ln or '[NOT ALLOWED]' in ln)]


def test_preflight_arms():
    expected = {'1': 32530, '2': 32530, '6': 28834, '9': 13506}
    names = {('1', None): 'mix10-15-20x25+ppvc-mixed+full2k-s301',
             ('2', None): '10x25+ppvc-mixed+full2k-s301',
             ('6', 'single'): '10x25+ppvc-mixed+a1-2k-s301',
             ('6', 'mix'): 'mix10-15-20x25+ppvc-mixed+a1-2k-s301',
             ('9', 'single'): '10x25+ppvc-mixed+plan-a1-2k-s301',
             ('9', 'mix'): 'mix10-15-20x25+ppvc-mixed+plan-a1-2k-s301'}
    listings = {None: dry_run(), 'single': dry_run('--from', '3', '--headline', 'single'),
                'mix': dry_run('--from', '3', '--headline', 'mix')}
    for (arm, head), want_name in names.items():
        rc, text = listings[head]
        cmds = preflight_commands(text)
        label = f'arm {arm}' + (f' (headline {head})' if head else '')
        if arm not in cmds:
            check(f'{label}: preflight command in queue listing', False, f'rc={rc}')
            continue
        queue_name, argv = cmds[arm]
        argv = strip_opt(argv, '--out')
        tmp = tempfile.NamedTemporaryFile(suffix='.json', delete=False).name
        argv = ['--out', tmp] + argv
        dropped = ''
        if '--compare' in argv:
            cmp_path = argv[argv.index('--compare') + 1]
            if not os.path.exists(os.path.join(REPO, cmp_path)):
                argv = strip_opt(strip_opt(argv, '--compare'), '--allow_diff')
                dropped = f' (comparator {cmp_path} not created yet: diff skipped)'
        rc, out, err = run_preflight(argv)
        print(f'--- preflight {label}: {queue_name}{dropped}')
        print('\n'.join('    ' + ln for ln in summary_lines(out)))
        if rc != 0:
            print(err[-2000:])
        with open(tmp) as f:
            resolved = json.load(f)
        os.unlink(tmp)
        check(f'{label}: preflight exit 0', rc == 0, f'rc={rc}')
        check(f'{label}: parameter count {expected[arm]}',
              resolved['param_count'] == expected[arm], str(resolved['param_count']))
        check(f'{label}: model name {want_name}',
              resolved['derived']['model_name'] == want_name == queue_name,
              resolved['derived']['model_name'])
        if arm in ('1', '2'):
            check(f'{label}: comparator diff printed and allowed',
                  'PREFLIGHT ok: every difference is in --allow_diff' in out)

    # a mismatch must fail: wrong expected count, and a diff outside the allow list
    rc, text = listings[None]
    _, argv = preflight_commands(text)['2']
    argv = strip_opt(argv, '--out')
    bad = list(argv)
    bad[bad.index('--expect_params') + 1] = '12345'
    rc, out, _ = run_preflight(bad)
    check('wrong --expect_params exits 2', rc == 2, f'rc={rc}')
    bad = list(argv)
    i = bad.index('--allow_diff')
    bad[i + 1] = 'max_updates,model_suffix'
    rc, out, _ = run_preflight(bad)
    check('diff outside --allow_diff exits 3 (seed_train not allowed)',
          rc == 3 and 'unintended differences: seed_train' in out, f'rc={rc}')
    bad = strip_opt(argv, '--model_suffix') + ['--model_suffix', 'full']
    bad = strip_opt(bad, '--compare')
    bad = strip_opt(bad, '--allow_diff')
    rc, out, _ = run_preflight(bad)
    check('--check_ckpt on an existing checkpoint exits 4', rc == 4, f'rc={rc}')


# ---------------------------------------------------------------------------
# --bound_lag_free
# ---------------------------------------------------------------------------

def _env_module():
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    import fjsp_env_same_op_nums as E
    return E


def _rollout(E, jl, pt, lag, meta, rule='SPT', seed=7, **kw):
    from common_utils import heuristic_select_action
    env = E.FJSPEnvForSameOpNums(jl.shape[0], pt.shape[1], use_lag_features=True, **kw)
    env.set_initial_data([jl], [pt], time_lag_list=[lag],
                         op_type_list=[meta['op_type']], mch_type_list=[meta['mch_type']])
    init = (np.array(env.op_ct_lb), np.array(env.init_quality), np.array(env.fea_j))
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    amch = np.full(pt.shape[0], -1, dtype=int)
    feas, rews = [np.array(env.fea_j)], []
    M = pt.shape[1]
    while not env.done().all():
        if rule == 'random':
            legal = np.where(~np.array(env.dynamic_pair_mask[0]).reshape(-1))[0]
            a = int(rng.choice(legal))
        else:
            a = int(heuristic_select_action(rule, env))
        amch[env.candidate[0, a // M]] = a % M
        _, r, _ = env.step(np.array([a]))
        feas.append(np.array(env.fea_j))
        rews.append(np.array(r))
    return env, init, feas, rews, amch


def test_bound_flag():
    E = _env_module()
    from ppvc_instance_generator import load_instance
    from schedule_validator import validate_schedule
    stems = [os.path.join(REPO, 'data', 'PPVC', '10x25+ppvc-mixed', f'instance_{i:03d}')
             for i in range(5)]
    ok_default = ok_formula = ok_zero = ok_smaller = ok_dyn = ok_valid = True
    for stem in stems:
        jl, pt, meta = load_instance(stem)
        lag = np.asarray(meta['time_lag'], dtype=float)
        # default constructor == bound_lag_free=False (features, rewards, makespan)
        e0, i0, f0, r0, _ = _rollout(E, jl, pt, lag, meta)
        e1, i1, f1, r1, _ = _rollout(E, jl, pt, lag, meta, bound_lag_free=False)
        ok_default &= (all(np.array_equal(a, b) for a, b in zip(f0, f1))
                       and all(np.array_equal(a, b) for a, b in zip(r0, r1))
                       and np.array_equal(i0[0], i1[0])
                       and e0.current_makespan[0] == e1.current_makespan[0])
        # independent formula: normalized min pt, predecessor lag added (or not)
        scale = pt.max() - pt.min() + 1e-8
        mpt = np.where(pt > 0, (pt - pt.min()) / scale, np.inf).min(axis=1)
        nlag = lag / scale
        lb_aware, lb_free = np.zeros_like(mpt), np.zeros_like(mpt)
        first = np.concatenate([[0], np.cumsum(jl)[:-1]])
        for f_, n_ in zip(first, jl):
            for k in range(f_, f_ + n_):
                prev_a = lb_aware[k - 1] + nlag[k - 1] if k > f_ else 0.0
                prev_f = lb_free[k - 1] if k > f_ else 0.0
                lb_aware[k], lb_free[k] = prev_a + mpt[k], prev_f + mpt[k]
        e2, i2, f2, r2, amch2 = _rollout(E, jl, pt, lag, meta, bound_lag_free=True)
        ok_formula &= (np.allclose(i0[0][0], lb_aware, atol=1e-9)
                       and np.allclose(i2[0][0], lb_free, atol=1e-9))
        # lagged instance: smaller initial bound, same dynamics, validator-clean
        ok_smaller &= bool(i2[1][0] < i0[1][0]) and bool((i2[0] <= i0[0] + 1e-12).all())
        ok_dyn &= (e2.current_makespan[0] == e0.current_makespan[0]
                   and np.array_equal(e2.true_op_ct, e0.true_op_ct))
        res = validate_schedule(jl, pt, lag, amch2, e2.true_op_ct[0])
        ok_valid &= bool(res['feasible']) and abs(res['makespan'] - e2.current_makespan[0]) < 1e-6
        # random actions: identical dynamics under both bounds
        er0 = _rollout(E, jl, pt, lag, meta, rule='random', seed=11)[0]
        er1, _, _, _, amr = _rollout(E, jl, pt, lag, meta, rule='random', seed=11, bound_lag_free=True)
        ok_dyn &= np.array_equal(er0.true_op_ct, er1.true_op_ct)
        ok_valid &= bool(validate_schedule(jl, pt, lag, amr, er1.true_op_ct[0])['feasible'])
        # zero lags: the flag changes nothing
        z = np.zeros_like(lag)
        ez0, iz0, fz0, rz0, _ = _rollout(E, jl, pt, z, meta, bound_lag_free=False)
        ez1, iz1, fz1, rz1, _ = _rollout(E, jl, pt, z, meta, bound_lag_free=True)
        ok_zero &= (all(np.array_equal(a, b) for a, b in zip(fz0, fz1))
                    and all(np.array_equal(a, b) for a, b in zip(rz0, rz1))
                    and np.array_equal(iz0[0], iz1[0])
                    and ez0.current_makespan[0] == ez1.current_makespan[0])
        print(f'    {os.path.basename(stem)}: initial bound (max op_ct_lb, normalized) '
              f'lag-aware {i0[1][0]:.4f} vs lag-free {i2[1][0]:.4f}; SPT makespan '
              f'{e0.current_makespan[0]:.0f} under both')
    check('bound: default constructor == bound_lag_free=False (features, rewards, bound)', ok_default)
    check('bound: both bounds equal the independent per-job formula', ok_formula)
    check('bound: zero-lag instances unchanged by the flag', ok_zero)
    check('bound: lagged instances have a smaller initial bound with the flag', ok_smaller)
    check('bound: dynamics unchanged (SPT and random actions: same completion times)', ok_dyn)
    check('bound: schedules with the flag are validator-clean against the true lags', ok_valid)

    code = ("import sys; sys.argv=[sys.argv[0]]; import eval_ppvc as ev; "
            "assert 'bound_lag_free' in ev.ARCH_KEYS; "
            "e = ev._make_env(10, 25, False, True); assert e.bound_lag_free is True; "
            "e = ev._make_env(10, 25, False); assert e.bound_lag_free is False; "
            "assert ev._model_bound_lag_free() is False; "
            "ev.configs.bound_lag_free = True; assert ev._model_bound_lag_free() is True; "
            "print('eval_ppvc ok')")
    out = subprocess.run([PY, '-c', code], cwd=REPO, env=CHILD_ENV, capture_output=True, text=True)
    check('eval_ppvc applies bound_lag_free from the snapshot keys', out.returncode == 0,
          (out.stdout + out.stderr).strip()[-300:])


# ---------------------------------------------------------------------------
# queue script
# ---------------------------------------------------------------------------

def test_queue_script():
    out = subprocess.run(['bash', '-n', QUEUE], capture_output=True, text=True)
    check('bash -n run_revision_gpu_queue.sh', out.returncode == 0, out.stderr.strip())
    for args in ((), ('--from', '3', '--headline', 'mix')):
        rc, text = dry_run(*args)
        print(f'--- dry run {" ".join(args) or "(pilot phase)"} ---')
        print(text)
        check(f'dry run {" ".join(args) or "(pilot)"} exits 0', rc == 0, f'rc={rc}')
    rc, _ = dry_run('--from', '3')
    check('arms >= 3 without --headline are refused', rc != 0, f'rc={rc}')


# ---------------------------------------------------------------------------
# train.py equivalence (slow)
# ---------------------------------------------------------------------------

def test_train_smoke():
    smoke = ['--device', 'cpu', '--max_updates', '1', '--validate_timestep', '1',
             '--minibatch_size', '256', '--vali_size', '10']
    arms = {
        'DANIEL size-mix A3': ['--data_source', 'PPVC', '--ppvc_factory', 'default',
                               '--ppvc_mix', 'mixed', '--use_lag_features', 'True',
                               '--use_type_embedding', 'True', '--ppvc_size_mix', '10,15,20',
                               '--ppvc_vali_sizes', '10,20', '--seed_train', '301',
                               '--model_suffix', 'full2k-s301'],
        'PLAN single A1': ['--data_source', 'PPVC', '--framework', 'PLAN', '--n_j', '10',
                           '--ppvc_factory', 'default', '--ppvc_mix', 'mixed',
                           '--use_lag_features', 'False', '--use_type_embedding', 'False',
                           '--ppvc_vali_sizes', '10,20', '--seed_train', '301',
                           '--model_suffix', 'a1-2k-s301'],
    }
    for label, flags in arms.items():
        flags = flags + smoke
        with tempfile.TemporaryDirectory() as work:
            env = dict(CHILD_ENV, PYTHONPATH=REPO)
            tr = subprocess.run([PY, '-u', os.path.join(REPO, 'train.py'), *flags], cwd=work,
                                env=env, capture_output=True, text=True)
            ckpts = os.listdir(os.path.join(work, 'trained_network', 'PPVC')) \
                if tr.returncode == 0 else []
            tmp = os.path.join(work, 'resolved.json')
            rc, out, err = run_preflight(['--out', tmp, *flags])
            with open(tmp) as f:
                resolved = json.load(f)
            name = resolved['derived']['model_name']
            snap_path = os.path.join(work, 'train_log', 'PPVC', f'config_{name}.json')
            snap = json.load(open(snap_path)) if os.path.exists(snap_path) else None
            check(f'train smoke {label}: train.py exit 0', tr.returncode == 0, tr.stderr[-500:])
            check(f'train smoke {label}: checkpoint name == preflight name ({name})',
                  ckpts == [f'{name}.pth'], str(ckpts))
            diff = [k for k in set(resolved['snapshot']) | set(snap or {})
                    if resolved['snapshot'].get(k) != (snap or {}).get(k)]
            check(f'train smoke {label}: snapshot identical to preflight resolution',
                  snap is not None and not diff, f'differing keys: {diff}')


def main():
    train_smoke = '--train_smoke' in sys.argv   # read before a test scrubs sys.argv
    test_preflight_arms()
    test_bound_flag()
    test_queue_script()
    if train_smoke:
        test_train_smoke()
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f'\n{len(RESULTS) - n_fail} passed, {n_fail} failed')
    return 1 if n_fail else 0


if __name__ == '__main__':
    sys.exit(main())
