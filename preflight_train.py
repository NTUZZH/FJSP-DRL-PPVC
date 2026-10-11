"""
preflight_train.py
==================
Resolve a training configuration exactly as the trainer would, WITHOUT
training, and check it before a run is launched.

DANIEL / PLAN (train.py)
  The train.py flags are injected into sys.argv, params.configs is parsed from
  them, and the real train.Trainer is constructed on the CPU. Its __init__ is
  the single source of the resolved configuration: model and data names, the
  factory's machine count, fea_j_input_dim for lag features, size mix,
  validation sizes and sets, the --init_from architecture check and weight
  load. The only interception: validation sets are generated in memory and
  never written to disk (the calls are recorded for --check_data). The
  printed snapshot is what train.py writes to train_log/PPVC/config_<model>.json.

HGNN (backbones/hgnn/train_ppvc.py, with --hgnn)
  The HGNN flags are parsed by train_ppvc.parse_args(); the run_config.json
  snapshot that train_ppvc.py would write is rebuilt and the policy is built
  on the CPU to count its parameters.

CLI
  python preflight_train.py [preflight options] <train.py flags>
  python preflight_train.py --hgnn [preflight options] <train_ppvc.py flags>

  --expect_params N     exit 2 unless the policy has exactly N parameters
  --compare PATH        key-by-key diff against a config snapshot (train.py's
                        config_<model>.json, or an HGNN run_config.json);
                        exit 3 unless --allow_diff covers every differing key
  --allow_diff K1,K2    keys allowed to differ from the comparator. A key that
                        the comparator lacks because it predates the flag is
                        not a difference when this run keeps the flag's
                        default; it is listed separately.
  --check_ckpt          exit 4 if the target checkpoint already exists
  --resume_ok           ... unless this is set
  --check_data          print the validation sets the run uses (seed0, count,
                        module count, on disk or in memory); exit 5 on a
                        stale on-disk record or a regenerated-vs-disk mismatch
  --testsets D1,D2      also verify test sets under data/PPVC/ (instance
                        count equals dataset_meta.json n_instances); exit 5
  --out PATH            write the resolved JSON to PATH
Exit codes: 0 pass, 1 error, 2 parameter count, 3 config diff, 4 checkpoint
exists, 5 data. All checks run; the first failing code is returned.
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
MISSING = '<absent>'
# keys train.py adds to its config snapshot (not command-line flags); a
# comparator written before train.py recorded them simply lacks them
RECORD_KEYS = ('vali_sizes_resolved',)


def parse_own(argv):
    ap = argparse.ArgumentParser(description='Preflight check of a training run',
                                 allow_abbrev=False)
    ap.add_argument('--hgnn', action='store_true')
    ap.add_argument('--expect_params', type=int, default=None)
    ap.add_argument('--compare', type=str, default=None)
    ap.add_argument('--allow_diff', type=str, default='')
    ap.add_argument('--check_ckpt', action='store_true')
    ap.add_argument('--resume_ok', action='store_true')
    ap.add_argument('--check_data', action='store_true')
    ap.add_argument('--testsets', type=str, default='')
    ap.add_argument('--out', type=str, default=None)
    return ap.parse_known_args(argv)


def jsonable(obj):
    """Round-trip through JSON exactly as train.py's snapshot writer does."""
    return json.loads(json.dumps(obj, default=str))


def import_with_retry(name, tries=3, wait=20):
    """Import a module that another process may be editing right now."""
    import importlib
    for k in range(tries):
        try:
            return importlib.import_module(name)
        except (SyntaxError, ImportError, IndentationError) as e:
            if k == tries - 1:
                raise
            print(f'[preflight] import of {name} failed ({e!r}); retry in {wait}s',
                  file=sys.stderr)
            sys.modules.pop(name, None)
            time.sleep(wait)


# ---------------------------------------------------------------------------
# DANIEL / PLAN
# ---------------------------------------------------------------------------

def resolve_train(train_argv):
    os.chdir(REPO)
    sys.path.insert(0, REPO)
    sys.argv = ['train.py'] + list(train_argv)
    from params import configs, parser
    requested_device = configs.device
    # build on the CPU; the recorded device stays the requested one
    configs.device = 'cpu'
    T = import_with_retry('train')
    import torch
    from model.PPO import build_policy

    calls = []
    orig_make = T.Trainer.make_ppvc_dataset

    def recording_make(self, n_instances, seed0, save_dir=None, n_modules=None):
        out = orig_make(self, n_instances, seed0, save_dir=None, n_modules=n_modules)
        calls.append({'n_instances': n_instances, 'seed0': seed0,
                      'n_modules': self.n_j if n_modules is None else n_modules,
                      'save_dir': save_dir,
                      'first_instance': (out[0][0], out[1][0], out[2][0])})
        return out

    T.Trainer.make_ppvc_dataset = recording_make
    try:
        trainer = T.Trainer(configs)
    finally:
        T.Trainer.make_ppvc_dataset = orig_make
    n_params = sum(p.numel() for p in trainer.ppo.policy.parameters())
    # independent cross-check: a fresh build_policy on the resolved config
    n_params_fresh = sum(p.numel() for p in build_policy(configs).parameters())
    configs.device = requested_device

    snapshot = jsonable(vars(configs))
    ds = configs.data_source
    derived = {
        'model_name': trainer.model_name,
        'data_name': trainer.data_name,
        'framework': configs.framework,
        'checkpoint': f'./trained_network/{ds}/{trainer.model_name}.pth',
        'snapshot_path': f'./train_log/{ds}/config_{trainer.model_name}.json',
        'reward_log': f'./train_log/{ds}/reward_{trainer.model_name}.txt',
        'n_j': trainer.n_j, 'n_m': trainer.n_m,
        'size_mix': trainer.size_mix,
        'vali_sizes': getattr(trainer, 'vali_sizes', [trainer.n_j]),
        'route_variant': configs.ppvc_route_variant,
        'train_env': type(trainer.env).__name__,
        'vali_data_path': trainer.vali_data_path,
        'vali_sets': [{k: v for k, v in c.items() if k != 'first_instance'} for c in calls],
    }
    defaults = jsonable({a.dest: a.default for a in parser._actions
                         if a.dest != 'help'})
    return {'kind': 'train.py', 'param_count': n_params,
            'param_count_fresh_build': n_params_fresh,
            'derived': derived, 'snapshot': snapshot}, defaults, calls, configs


def check_train_data(calls, configs):
    """Validation sets: in memory (seeded) or on disk (provenance record)."""
    import numpy as np
    from ppvc_instance_generator import load_instance
    ok = True
    for c in calls:
        seeds = f'{c["seed0"]}..{c["seed0"] + c["n_instances"] - 1}'
        if c['save_dir'] is None:
            print(f'  [data] vali {c["n_modules"]} modules: {c["n_instances"]} instances '
                  f'generated in memory, seed0={c["seed0"]} (seeds {seeds})')
            continue
        d = c['save_dir']
        meta_path = os.path.join(d, 'dataset_meta.json')
        if not os.path.exists(meta_path):
            print(f'  [data] vali {d}: absent; train.py will write {c["n_instances"]} '
                  f'instances with seed0={c["seed0"]} (seeds {seeds})')
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        n_disk = len(glob.glob(os.path.join(d, 'instance_*.fjs')))
        bad = []
        if meta.get('seed0') != c['seed0']:
            bad.append(f'seed0 on disk {meta.get("seed0")} != run {c["seed0"]}')
        if meta.get('n_modules', c['n_modules']) != c['n_modules']:
            bad.append(f'n_modules on disk {meta.get("n_modules")} != run {c["n_modules"]}')
        if meta.get('route_variant', 'fixed') != configs.ppvc_route_variant:
            bad.append(f'route_variant on disk {meta.get("route_variant", "fixed")}')
        if meta.get('lag_scale', 1.0) != getattr(configs, 'lag_scale', 1.0):
            bad.append(f'lag_scale on disk {meta.get("lag_scale", 1.0)}')
        if n_disk < c['n_instances']:
            print(f'  [data] note: {d} holds {n_disk} < {c["n_instances"]} instances; '
                  f'train.py writes the missing ones (same seeds)')
        # spot check: the regenerated first instance equals the file on disk
        stem = os.path.join(d, 'instance_000')
        if os.path.exists(stem + '.fjs'):
            jl, pt, m = load_instance(stem)
            jl0, pt0, lag0 = c['first_instance']
            same = np.array_equal(jl, jl0) and np.array_equal(pt, pt0)
            if not configs.ppvc_lagblind:
                same = same and np.array_equal(np.asarray(m['time_lag']), np.asarray(lag0))
            if not same:
                bad.append('instance_000 on disk differs from the regenerated instance')
        status = 'OK' if not bad else 'STALE: ' + '; '.join(bad)
        print(f'  [data] vali {d}: seed0={meta.get("seed0")} n_instances(meta)='
              f'{meta.get("n_instances")} files={n_disk} n_modules={meta.get("n_modules")} '
              f'-> {status}')
        ok = ok and not bad
    return ok


# ---------------------------------------------------------------------------
# HGNN
# ---------------------------------------------------------------------------

def resolve_hgnn(hgnn_argv):
    hg = os.path.join(REPO, 'backbones', 'hgnn')
    os.chdir(REPO)
    sys.path.insert(0, hg)
    sys.argv = ['train_ppvc.py'] + list(hgnn_argv)
    H = import_with_retry('train_ppvc')
    P = H.P
    import torch
    args, cfg = H.parse_args()
    device = torch.device('cpu')
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(device)
    num_mas = sum(P.FACTORIES[args.factory].values())
    train_paras = copy.deepcopy(cfg['train_paras'])
    train_paras.update(max_iterations=args.max_iterations, parallel_iter=args.parallel_iter,
                       update_timestep=args.update_timestep, save_timestep=args.save_timestep)
    env_paras, model_paras = P.make_paras(cfg, device, args.lag_mode, args.lag_features,
                                          args.batch_size, args.n_modules, num_mas,
                                          bound_pt=args.bound_pt)
    snapshot = jsonable({'args': vars(args),
                         'env_paras': {k: v for k, v in env_paras.items() if k != 'device'},
                         'model_paras': {k: v for k, v in model_paras.items() if k != 'device'},
                         'train_paras': train_paras, 'upstream_config': cfg})
    model = P.build_model(model_paras, train_paras, num_envs=args.batch_size)
    n_params = sum(p.numel() for p in model.policy.parameters())
    save_dir = os.path.join(args.save_root, args.run_name)
    derived = {'run_name': args.run_name,
               'save_dir': os.path.relpath(save_dir, REPO),
               'checkpoint': os.path.relpath(os.path.join(save_dir, 'best.pt'), REPO),
               'snapshot_path': os.path.relpath(os.path.join(save_dir, 'run_config.json'), REPO),
               'machines': num_mas,
               'vali_sets': [{'n_instances': args.vali_size, 'seed0': args.vali_seed0,
                              'n_modules': args.n_modules, 'save_dir': None}]}
    return {'kind': 'hgnn', 'param_count': n_params, 'derived': derived,
            'snapshot': snapshot}, args


def flatten(d, prefix=''):
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------------------
# shared checks
# ---------------------------------------------------------------------------

def config_diff(mine, ref, defaults, absent_record=None):
    """Return (differences, absent-at-default keys). mine/ref are flat dicts.
    Record keys (RECORD_KEYS) absent from the comparator are appended to
    absent_record, when given, instead of counting as differences."""
    diffs, absent_default = [], []
    for k in sorted(set(mine) | set(ref)):
        a, b = mine.get(k, MISSING), ref.get(k, MISSING)
        if a == b:
            continue
        if b == MISSING and k in defaults and a == defaults[k]:
            absent_default.append(k)
            continue
        if b == MISSING and k in RECORD_KEYS and absent_record is not None:
            absent_record.append((k, a))
            continue
        diffs.append((k, b, a))
    return diffs, absent_default


def check_testsets(names):
    ok = True
    for ds in names:
        d = os.path.join('data', 'PPVC', ds)
        meta_path = os.path.join(d, 'dataset_meta.json')
        n_files = len(glob.glob(os.path.join(d, 'instance_*.fjs')))
        if not os.path.exists(meta_path):
            print(f'  [test] {ds}: MISSING dataset_meta.json (files={n_files})')
            ok = False
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        good = n_files == meta.get('n_instances') and n_files > 0
        print(f'  [test] {ds}: seed0={meta.get("seed0")} n_instances(meta)='
              f'{meta.get("n_instances")} files={n_files} n_modules={meta.get("n_modules")} '
              f'route={meta.get("route_variant", "fixed")} -> {"OK" if good else "MISMATCH"}')
        ok = ok and good
    return ok


def main():
    own, rest = parse_own(sys.argv[1:])
    if rest and rest[0] == '--':
        rest = rest[1:]
    calls, configs = [], None
    if own.hgnn:
        resolved, _ = resolve_hgnn(rest)
        defaults = {}
        mine_flat = flatten(resolved['snapshot'])
    else:
        resolved, defaults, calls, configs = resolve_train(rest)
        mine_flat = resolved['snapshot']

    text = json.dumps(resolved, indent=1)
    print(text)
    if own.out:
        with open(own.out, 'w') as f:
            f.write(text + '\n')

    code = 0

    def fail(c):
        nonlocal code
        if code == 0:
            code = c

    name = resolved['derived'].get('model_name', resolved['derived'].get('run_name'))
    print(f'PREFLIGHT {name}: parameters={resolved["param_count"]}')
    if resolved.get('param_count_fresh_build', resolved['param_count']) != resolved['param_count']:
        print('PREFLIGHT FAIL: trainer policy and fresh build_policy differ in size')
        fail(2)
    if own.expect_params is not None:
        if resolved['param_count'] == own.expect_params:
            print(f'PREFLIGHT ok: parameter count {resolved["param_count"]} == expected')
        else:
            print(f'PREFLIGHT FAIL: parameter count {resolved["param_count"]} != expected '
                  f'{own.expect_params}')
            fail(2)

    if own.compare:
        if not os.path.exists(own.compare):
            print(f'PREFLIGHT FAIL: comparator snapshot not found: {own.compare}')
            fail(3)
        else:
            with open(own.compare) as f:
                ref = json.load(f)
            ref_flat = flatten(ref) if own.hgnn else ref
            absent_record = []
            diffs, absent_default = config_diff(mine_flat, ref_flat, defaults,
                                                absent_record)
            allowed = {k.strip() for k in own.allow_diff.split(',') if k.strip()}
            print(f'DIFF vs {own.compare}  (comparator -> this run)')
            for k, b, a in diffs:
                tag = 'allowed' if k in allowed else 'NOT ALLOWED'
                print(f'  {k:28s} {json.dumps(b):>28s} -> {json.dumps(a):<28s} [{tag}]')
            if not diffs:
                print('  (no differences)')
            if absent_default:
                print('  absent in comparator (it predates these flags), this run at '
                      'the default: ' + ', '.join(absent_default))
            for k, a in absent_record:
                print(f'  absent in comparator (recorded by train.py since; the '
                      f'flags above decide it): {k} = {json.dumps(a)}')
            unused = sorted(allowed - {k for k, _, _ in diffs})
            if unused:
                print('  allowed but equal: ' + ', '.join(unused))
            bad = [k for k, _, _ in diffs if k not in allowed]
            if bad:
                print('PREFLIGHT FAIL: unintended differences: ' + ', '.join(bad))
                fail(3)
            else:
                print('PREFLIGHT ok: every difference is in --allow_diff')

    if own.check_ckpt:
        ck = resolved['derived']['checkpoint']
        if os.path.exists(ck):
            if own.resume_ok:
                print(f'PREFLIGHT note: checkpoint exists ({ck}); --resume_ok given')
            else:
                print(f'PREFLIGHT FAIL: checkpoint already exists: {ck}')
                fail(4)
        else:
            print(f'PREFLIGHT ok: no checkpoint at {ck}')
            snap_path = resolved['derived']['snapshot_path']
            if os.path.exists(snap_path):
                print(f'PREFLIGHT note: a snapshot without a checkpoint exists '
                      f'({snap_path}); an earlier run of this name stopped early and '
                      f'its logs will be overwritten')

    if own.check_data:
        if own.hgnn:
            v = resolved['derived']['vali_sets'][0]
            print(f'  [data] vali {v["n_modules"]} modules: {v["n_instances"]} instances '
                  f'generated in memory, seed0={v["seed0"]} (seeds {v["seed0"]}..'
                  f'{v["seed0"] + v["n_instances"] - 1}); training seeds from '
                  f'RandomState({resolved["snapshot"]["args"]["seed"]})')
        elif not check_train_data(calls, configs):
            print('PREFLIGHT FAIL: validation data record is stale')
            fail(5)
    if own.testsets:
        if not check_testsets([s for s in own.testsets.split(',') if s.strip()]):
            print('PREFLIGHT FAIL: test set check')
            fail(5)

    print(f'PREFLIGHT {"PASS" if code == 0 else "FAIL"} {name} (exit {code})')
    return code


if __name__ == '__main__':
    sys.exit(main())
