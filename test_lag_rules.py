"""
test_lag_rules.py
=================
Checks for the lag-aware dispatching rules SPTL, LLF, MWKRL and LRPTL in
common_utils.heuristic_select_action, run through eval_ppvc's rollout.

  T1  every rule gives a validator-clean schedule on 5 instances of
      10x25+ppvc-mixed, and the env makespan equals the validator makespan
  T2  on the same instances with all lags set to zero, SPTL reproduces SPT
      exactly (same machine assignment, same completion times)
  T3  the remaining-route sums used by MWKRL / LRPTL equal a brute-force sum
      over the unscheduled operations of each job
  T4  the classic rules still reproduce the stored 10x25 results
      (Result_<rule>+10x25+ppvc-mixed+smoke) on the same 5 instances

usage: python test_lag_rules.py
"""
import os
import sys

import numpy as np

sys.argv = [sys.argv[0]]          # eval_ppvc and params parse argv at import
import eval_ppvc                   # noqa: E402
from common_utils import _lag_rule_route_suffixes  # noqa: E402
from ppvc_instance_generator import load_instance  # noqa: E402
from schedule_validator import validate_schedule   # noqa: E402

DATA = os.path.join('data', 'PPVC', '10x25+ppvc-mixed')
N_INST = 5
SEED_TEST = 50                     # eval_ppvc default; instance i uses 50 + i
NEW_RULES = ('SPTL', 'LLF', 'MWKRL', 'LRPTL')


def main():
    stems = eval_ppvc.list_instances(DATA)[:N_INST]
    insts = [load_instance(s) for s in stems]
    ok = True

    print('T1  validator-clean schedules, env makespan == validator makespan')
    ms_table = {}
    for rule in NEW_RULES:
        row, rule_ok = [], True
        for i, (jl, pt, meta) in enumerate(insts):
            ms, _, amch, oct_ = eval_ppvc.rollout_heuristic(
                rule, jl, pt, meta, seed=SEED_TEST + i)
            res = validate_schedule(jl, pt, meta['time_lag'], amch, oct_)
            good = res['feasible'] and abs(res['makespan'] - ms) < 1e-6
            ok &= good
            rule_ok &= good
            if not good:
                print(f'    FAIL {rule} {os.path.basename(stems[i])}: '
                      f'env={ms} validator={res["makespan"]} '
                      f'{res["violations"][:3]}')
            row.append(ms)
        ms_table[rule] = row
        print(f'    {rule:<6} makespans {row}  -> '
              f'{"OK" if rule_ok else "FAIL"}')

    print('T2  zero lags: SPTL == SPT')
    for i, (jl, pt, meta) in enumerate(insts):
        meta0 = dict(meta, time_lag=np.zeros_like(meta['time_lag']))
        a = eval_ppvc.rollout_heuristic('SPT', jl, pt, meta0, seed=SEED_TEST + i)
        b = eval_ppvc.rollout_heuristic('SPTL', jl, pt, meta0, seed=SEED_TEST + i)
        same = (a[0] == b[0] and np.array_equal(a[2], b[2])
                and np.array_equal(a[3], b[3]))
        ok &= same
        print(f'    {os.path.basename(stems[i])}: SPT={a[0]:.0f}  '
              f'SPTL={b[0]:.0f}  identical schedule={same}')

    print('T3  remaining-route sums vs brute force')
    jl, pt, meta = insts[0]
    env, _ = eval_ppvc._new_env(jl, pt, meta, use_lag_features=False,
                                use_type_embedding=False)
    lag = meta['time_lag'].astype(float)
    for stat, fn in (('mean', lambda r: r[r > 0].mean()),
                     ('min', lambda r: r[r > 0].min())):
        work, lagsum = _lag_rule_route_suffixes(env, stat)
        max_err = 0.0
        first = 0
        for L in jl:
            last = first + L - 1
            for o in range(first, last + 1):
                w = sum(fn(pt[k].astype(float)) for k in range(o, last + 1))
                g = lag[o:last].sum()
                max_err = max(max_err, abs(work[o] - w), abs(lagsum[o] - g))
            first = last + 1
        ok &= max_err < 1e-9
        print(f'    pt={stat}: max abs error {max_err:.2e}')
    print(f'    MWKRL vs LRPTL makespans differ on '
          f'{sum(x != y for x, y in zip(ms_table["MWKRL"], ms_table["LRPTL"]))}'
          f'/{N_INST} instances')

    print('T4  classic rules reproduce the stored 10x25 results')
    rdir = os.path.join('test_results', 'PPVC', '10x25+ppvc-mixed')
    for rule in ('FIFO', 'MOR', 'SPT', 'MWKR'):
        path = os.path.join(
            rdir, f'Result_{rule}+10x25+ppvc-mixed+smoke_10x25+ppvc-mixed.npy')
        if not os.path.exists(path):
            print(f'    {rule}: stored file missing, skipped')
            continue
        stored = np.load(path)[:N_INST, 0]
        now = [eval_ppvc.rollout_heuristic(rule, *inst, seed=SEED_TEST + i)[0]
               for i, inst in enumerate(insts)]
        same = np.allclose(stored, now)
        ok &= same
        print(f'    {rule:<5} stored={stored.tolist()} now={now} match={same}')

    print('ALL PASS' if ok else 'SOME CHECKS FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
