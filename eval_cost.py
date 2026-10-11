"""
eval_cost.py
============
Training and deployment cost of trained PPVC policies: parameter count,
checkpoint size, greedy decision latency on CPU (one torch thread) and, when
the GPU is available, on CUDA; peak process memory (CPU) and peak allocated
GPU memory; offline training wall-clock read from train_time.txt.

MEASUREMENT PROTOCOL
  * Latency = wall-clock of one full greedy schedule of one instance (all N
    dispatching decisions, batch of 1), measured from after the environment is
    initialised to the last decision, as in eval_ppvc.rollout_greedy. It is
    reported per instance and per decision; the policy forward pass alone is
    also timed (per decision). Mean over the first --n_inst instances of each
    dataset; one untimed warm-up rollout per model precedes the timed ones.
  * CPU: torch.set_num_threads(1) and set_num_interop_threads(1). Run under
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 and a taskset core pin.
  * GPU: CUDA with torch.cuda.synchronize() around every timed region; the
    whole GPU probe is capped at --gpu_budget_s seconds. GPU utilisation by
    other processes is recorded before the probe (contended if > 5 %).
  * Peak memory: CPU = peak resident set size of the measuring process during
    the timed rollouts (VmHWM, reset per model and dataset via clear_refs);
    GPU = torch.cuda.max_memory_allocated during the timed rollouts.
  * Training cost: train_time.txt records, per run, the start time and the
    wall-clock seconds from before the first update to after the last (it
    includes the validation passes). A run is flagged contended when another
    run in the same file overlapped it in time. Jobs of other projects on the
    same GPU are not in that file, so "no overlap" is not proof of a quiet GPU.
  * The machine load average is recorded at the start and end of every phase.

Each measurement phase runs in its own worker process so that CPU and GPU
memory figures do not contaminate each other.

CLI
  python eval_cost.py [--models M1 M2 ...] [--datasets 10x25 20x25 ...]
      [--n_inst 3] [--cpu on|off] [--gpu auto|off] [--n_inst_gpu 1]
      [--gpu_budget_s 60] [--out <json path>] [--append]
  --append merges a run into an existing JSON, so a long measurement can be
  split into several short runs (e.g. one per model, then one GPU probe).
"""
import argparse
import datetime
import glob
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

DEFAULT_OUT = os.path.join('paper_inputs', 'cost_table.json')
GPU_MARKER = 'GPU_SHARE_OK'
DEFAULT_DATASETS = ['10x25', '20x25', '30x25', '50x25', '80x25']
A3_MODEL = '10x25+ppvc-mixed+full'

# network-shape flags copied from the training-config snapshot (as eval_ppvc)
ARCH_KEYS = (
    'use_lag_features', 'use_type_embedding',
    'fea_j_input_dim', 'fea_m_input_dim',
    'type_emb_dim', 'n_op_types', 'n_mch_types',
    'num_heads_OAB', 'num_heads_MAB', 'layer_fea_output_dim',
    'num_mlp_layers_actor', 'hidden_dim_actor',
    'num_mlp_layers_critic', 'hidden_dim_critic',
    'dropout_prob', 'framework',
)


def parse_cli():
    ap = argparse.ArgumentParser(description='Training and deployment cost of PPVC policies')
    ap.add_argument('--models', nargs='+', default=None,
                    help='model names under trained_network/PPVC/ (default: the A3 '
                         f'checkpoint {A3_MODEL} and every +plan checkpoint present)')
    ap.add_argument('--datasets', nargs='+', default=DEFAULT_DATASETS,
                    help="dataset sizes 'JxM' (data/PPVC/<JxM>+ppvc-mixed) or full dataset dirs")
    ap.add_argument('--n_inst', type=int, default=3, help='timed instances per dataset (CPU)')
    ap.add_argument('--gpu', type=str, default='auto', choices=['auto', 'off'],
                    help="'auto' = run the GPU probe only if the GPU marker file exists")
    ap.add_argument('--gpu_marker', type=str, default=GPU_MARKER,
                    help='file whose presence permits the GPU probe (relative to the repo)')
    ap.add_argument('--n_inst_gpu', type=int, default=1, help='timed instances per dataset (GPU)')
    ap.add_argument('--gpu_budget_s', type=float, default=60.0,
                    help='wall-clock cap of the whole GPU probe (seconds)')
    ap.add_argument('--cpu', type=str, default='on', choices=['on', 'off'],
                    help="'off' skips the CPU latency phase (e.g. a GPU-only probe run)")
    ap.add_argument('--append', action='store_true',
                    help='merge this run into an existing --out JSON (new entries replace '
                         'old ones of the same model and test set; runs are listed)')
    ap.add_argument('--out', type=str, default=DEFAULT_OUT,
                    help='JSON output; a markdown table is written next to it (.md)')
    # internal: worker mode
    ap.add_argument('--phase', type=str, default='main', choices=['main', 'cpu', 'gpu'],
                    help=argparse.SUPPRESS)
    ap.add_argument('--worker_out', type=str, default='', help=argparse.SUPPRESS)
    args = ap.parse_args()
    # params.py parses sys.argv at import time: hand it a clean argv
    sys.argv = [sys.argv[0]]
    return args


def load_avg():
    return [round(x, 2) for x in os.getloadavg()]


def default_models():
    models = [A3_MODEL] if os.path.exists(f'./trained_network/PPVC/{A3_MODEL}.pth') else []
    for p in sorted(glob.glob('./trained_network/PPVC/*.pth')):
        name = os.path.basename(p)[:-len('.pth')]
        last = name.split('+')[-1]
        if (last == 'plan' or last.startswith('plan-')) and name not in models:
            models.append(name)
    return models


def dataset_dir(d):
    if os.path.isdir(d):
        return d
    return os.path.join('data', 'PPVC', f'{d}+ppvc-mixed')


def read_snapshot(model):
    path = f'./train_log/PPVC/config_{model}.json'
    with open(path) as f:
        snap = json.load(f)
    snap.setdefault('framework', 'DANIEL')
    return snap


# ---------------------------------------------------------------------------
# worker: latency and memory on one device
# ---------------------------------------------------------------------------

def _vm_hwm_kb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmHWM:'):
                return int(line.split()[1])
    return -1


def _vm_rss_kb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])
    return -1


def _reset_peak_rss():
    """Reset VmHWM to the current RSS (Linux >= 4.0); False if unsupported."""
    try:
        with open('/proc/self/clear_refs', 'w') as f:
            f.write('5')
        return True
    except OSError:
        return False


def run_worker(args):
    device_name = args.phase
    from params import configs
    configs.device = 'cuda' if device_name == 'gpu' else 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = configs.device_id
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda' if device_name == 'gpu' else 'cpu')
    dev = torch.device(configs.device)
    is_gpu = dev.type == 'cuda'

    from model.PPO import build_policy
    from common_utils import greedy_select_action
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    from ppvc_instance_generator import load_instance

    def sync():
        if is_gpu:
            torch.cuda.synchronize()

    def rollout(policy, jl, pt, meta, lag_feat, type_emb):
        env = FJSPEnvForSameOpNums(n_j=jl.shape[0], n_m=pt.shape[1],
                                   use_lag_features=lag_feat)
        kwargs = dict(time_lag_list=[meta['time_lag']])
        if type_emb:
            kwargs['op_type_list'] = [meta['op_type']]
            kwargs['mch_type_list'] = [meta['mch_type']]
        state = env.set_initial_data([jl], [pt], **kwargs)
        n_dec, t_fwd = 0, 0.0
        sync()
        t0 = time.perf_counter()
        while True:
            with torch.no_grad():
                sync()
                f0 = time.perf_counter()
                pi, _ = policy(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                               candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                               mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                               dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                               fea_pairs=state.fea_pairs_tensor,
                               op_type=state.op_type_tensor, mch_type=state.mch_type_tensor)
                sync()
                t_fwd += time.perf_counter() - f0
            action = greedy_select_action(pi)
            state, _, done = env.step(actions=action.cpu().numpy())
            n_dec += 1
            if done.all():
                break
        sync()
        t_total = time.perf_counter() - t0
        return float(env.current_makespan[0]), t_total, t_fwd, n_dec

    n_inst = args.n_inst_gpu if is_gpu else args.n_inst
    budget = args.gpu_budget_s if is_gpu else float('inf')
    t_phase0 = time.perf_counter()
    out = {'device': device_name, 'load_avg_start': load_avg(),
           'torch_threads': torch.get_num_threads(),
           'torch_interop_threads': torch.get_num_interop_threads(),
           'torch_version': torch.__version__, 'results': {}, 'budget_exhausted': False}
    if is_gpu:
        out['gpu_name'] = torch.cuda.get_device_name(0)
    models = args.models or default_models()
    for model in models:
        snap = read_snapshot(model)
        for k in ARCH_KEYS:
            if k in snap:
                setattr(configs, k, snap[k])
        lag_feat = bool(snap.get('use_lag_features', False))
        type_emb = bool(snap.get('use_type_embedding', False))
        policy = build_policy(configs)
        ckpt = f'./trained_network/PPVC/{model}.pth'
        policy.load_state_dict(torch.load(ckpt, map_location=dev), strict=True)
        policy.eval()
        res_m = {'_params_module': int(sum(p.numel() for p in policy.parameters()))}
        # warm-up (untimed): first instance of the first dataset
        stems0 = sorted(glob.glob(os.path.join(dataset_dir(args.datasets[0]), 'instance_*.fjs')))
        jl, pt, meta = load_instance(stems0[0][:-len('.fjs')])
        _, t_w, _, n_w = rollout(policy, jl, pt, meta, lag_feat, type_emb)
        sec_per_dec = t_w / n_w
        for d in args.datasets:
            ddir = dataset_dir(d)
            stems = sorted(glob.glob(os.path.join(ddir, 'instance_*.fjs')))[:n_inst]
            if not stems:
                res_m[d] = {'error': f'no instances under {ddir}'}
                continue
            peak_ok = _reset_peak_rss()
            rss_before = _vm_rss_kb()
            if is_gpu:
                torch.cuda.reset_peak_memory_stats()
                gpu_before = torch.cuda.memory_allocated()
            rows = []
            for s in stems:
                jl, pt, meta = load_instance(s[:-len('.fjs')])
                # predicted duration: seconds per decision seen so far, times this
                # instance's decisions, with a margin for superlinear growth
                if time.perf_counter() - t_phase0 + 1.5 * sec_per_dec * pt.shape[0] > budget:
                    out['budget_exhausted'] = True
                    break
                ms, t_total, t_fwd, n_dec = rollout(policy, jl, pt, meta, lag_feat, type_emb)
                sec_per_dec = max(sec_per_dec, t_total / n_dec)
                print(f'[{device_name}] {model} {d} {os.path.basename(s)}: {t_total:.2f} s, '
                      f'{n_dec} decisions, makespan {ms:.0f}', flush=True)
                rows.append({'instance': os.path.basename(s)[:-len('.fjs')], 'makespan': ms,
                             'seconds': t_total, 'forward_seconds': t_fwd, 'decisions': n_dec})
            if not rows:
                res_m[d] = {'skipped': 'GPU probe budget exhausted'}
                continue
            entry = {
                'dataset': os.path.basename(os.path.normpath(ddir)),
                'n_instances': len(rows),
                'decisions_per_instance': rows[0]['decisions'],
                'latency_s_per_instance_mean': float(np.mean([r['seconds'] for r in rows])),
                'latency_ms_per_decision_mean': float(np.mean(
                    [1e3 * r['seconds'] / r['decisions'] for r in rows])),
                'latency_ms_per_decision_min': float(np.min(
                    [1e3 * r['seconds'] / r['decisions'] for r in rows])),
                'forward_ms_per_decision_mean': float(np.mean(
                    [1e3 * r['forward_seconds'] / r['decisions'] for r in rows])),
                'instances': rows,
                'load_avg': load_avg(),
            }
            if is_gpu:
                entry['peak_gpu_allocated_MB'] = torch.cuda.max_memory_allocated() / 2 ** 20
                entry['gpu_allocated_before_MB'] = gpu_before / 2 ** 20
            else:
                entry['peak_rss_MB'] = _vm_hwm_kb() / 1024 if peak_ok else None
                entry['rss_before_rollouts_MB'] = rss_before / 1024
            res_m[d] = entry
        out['results'][model] = res_m
        del policy
    out['load_avg_end'] = load_avg()
    out['phase_seconds'] = time.perf_counter() - t_phase0
    with open(args.worker_out, 'w') as f:
        json.dump(out, f, indent=1)


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------

def static_model_info(model):
    """Parameter count (actor, critic, total) and checkpoint size, from the file."""
    import torch
    ckpt = f'./trained_network/PPVC/{model}.pth'
    sd = torch.load(ckpt, map_location='cpu')
    total = int(sum(v.numel() for v in sd.values()))
    critic = int(sum(v.numel() for k, v in sd.items() if k.startswith('critic')))
    snap = read_snapshot(model)
    return {'framework': snap['framework'], 'params_total': total,
            'params_critic': critic, 'params_without_critic': total - critic,
            'checkpoint_bytes': os.path.getsize(ckpt),
            'trained_updates': snap.get('max_updates'), 'train_n_j': snap.get('n_j'),
            'train_factory': snap.get('ppvc_factory', 'default'),
            'train_device': snap.get('device')}


def parse_train_time(path='./train_time.txt'):
    """[(model name, start datetime, end datetime, seconds)] from train_time.txt."""
    runs = []
    if not os.path.exists(path):
        return runs
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.split('\t') if p.strip()]
            try:
                name = parts[0].split('/')[-1]
                sec = float(parts[1].split(':')[1])
                start = datetime.datetime.strptime(parts[2].split(':')[1].strip(),
                                                   '%Y%m%d_%H%M%S')
            except (IndexError, ValueError):
                continue
            runs.append((name, start, start + datetime.timedelta(seconds=sec), sec))
    return runs


def training_cost(model, runs):
    """Training cost of a model from train_time.txt (the last run of that name)."""
    mine = [r for r in runs if r[0] == model]
    if not mine:
        return {'recorded': False,
                'note': 'no entry in train_time.txt (run not completed or not logged)'}
    name, st, et, sec = mine[-1]
    overlaps = sorted({r[0] for r in runs if r is not mine[-1] and r[1] < et and r[2] > st})
    return {'recorded': True, 'start': st.isoformat(), 'wall_seconds': sec,
            'wall_hours': sec / 3600.0,
            'quantity': 'wall-clock hours of the training process on one GPU '
                        '(train_time.txt: first to last update, validation included)',
            'overlapping_runs_in_log': overlaps,
            'contended': bool(overlaps) or None,
            'contention_note': ('overlapped other logged runs' if overlaps else
                                'no overlap with runs logged in train_time.txt; other '
                                'projects sharing the GPU are not logged, so a quiet GPU '
                                'is not established')}


def seed_group_cost(model, runs):
    """Wall-clock of the seed replicates of the same configuration (-sNNN)."""
    names = {r[0] for r in runs}
    group = [n for n in names if n == model or (n.startswith(model + '-s')
                                                 and n[len(model) + 2:].isdigit())]
    secs = {n: [r[3] for r in runs if r[0] == n][-1] for n in group}
    if len(secs) < 2:
        return None
    v = np.array(list(secs.values())) / 3600.0
    return {'runs': {k: round(s / 3600.0, 3) for k, s in sorted(secs.items())},
            'fastest_hours': float(v.min()), 'median_hours': float(np.median(v)),
            'slowest_hours': float(v.max()),
            'note': 'same configuration and update count; spread across replicates '
                    'reflects machine load, the fastest is the least-contended estimate'}


def gpu_state():
    try:
        q = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,memory.total',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True,
                           timeout=20).stdout.strip().split('\n')[0]
        name, util, used, total = [x.strip() for x in q.split(',')]
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=20).stdout.split()
        return {'name': name, 'utilization_pct': float(util), 'memory_used_MB': float(used),
                'memory_total_MB': float(total), 'compute_processes': len(apps)}
    except Exception as e:  # nvidia-smi missing or failed
        return {'error': str(e)}


def cpu_name():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def spawn_worker(args, phase, out_path):
    cmd = [sys.executable, os.path.abspath(__file__), '--phase', phase,
           '--worker_out', out_path, '--n_inst', str(args.n_inst),
           '--n_inst_gpu', str(args.n_inst_gpu), '--gpu_budget_s', str(args.gpu_budget_s),
           '--datasets', *args.datasets, '--models', *args.models]
    r = subprocess.run(cmd)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f'{phase} worker failed (exit {r.returncode})')
    with open(out_path) as f:
        return json.load(f)


def fmt(x, nd=1):
    return '-' if x is None else f'{x:.{nd}f}'


def core_busy(cores, seconds=2.0):
    """Busy fraction of each given core over a short window (/proc/stat)."""
    def snap():
        out = {}
        with open('/proc/stat') as f:
            for line in f:
                if line.startswith('cpu') and line[3].isdigit():
                    k, *v = line.split()
                    v = list(map(int, v))
                    out[int(k[3:])] = (sum(v), v[3] + v[4])   # total, idle + iowait
        return out
    a = snap()
    time.sleep(seconds)
    b = snap()
    return {c: round(1.0 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0]), 3)
            for c in cores if c in a and c in b}


def write_markdown(res, md_path):
    L = ['# Training and deployment cost of the PPVC policies', '',
         f"- machine: {res['machine']['cpu']} ({res['machine']['n_cpus']} logical CPUs); "
         f"GPU {res['machine'].get('gpu', {}).get('name', '-')}; torch {res['machine']['torch']}",
         '- CPU latency: torch 1 thread (intra-op and inter-op), greedy decoding, batch of 1, '
         'mean over the first instances of each test set; the one-instance timing '
         'excludes environment initialisation.', '']
    for k, r in enumerate(res['runs']):
        L.append(f"- run {k} ({r['measured_at']}): models {', '.join(r['models'])}; phases "
                 f"{', '.join(r['phases']) or 'none'}; load average (1, 5, 15 min) "
                 f"{r['load_avg_start']} at start, {r['load_avg_end']} at end; pinned cores "
                 f"{r['affinity']} busy before the run {r['pinned_cores_busy_before']}"
                 + (f"; GPU probe: {r['gpu_status']}" if 'gpu_status' in r else ''))
    L += ['', '## Model size and training cost', '',
          '| model | backbone | parameters (actor+critic) | without critic | checkpoint (KB) '
          '| training wall-clock (h) | overlapped other logged runs |',
          '|---|---|---|---|---|---|---|']
    for m, s in res['static'].items():
        tc = res['training'][m]
        wall = fmt(tc.get('wall_hours'), 2) if tc.get('recorded') else 'not logged'
        cont = ('yes' if tc.get('contended') else 'no') if tc.get('recorded') else '-'
        L.append(f"| {m} | {s['framework']} | {s['params_total']:,} | "
                 f"{s['params_without_critic']:,} | {s['checkpoint_bytes'] / 1024:.1f} | "
                 f"{wall} | {cont} |")
    for m, g in res.get('training_seed_groups', {}).items():
        if g:
            L.append('')
            L.append(f"Seed replicates of {m}: " + ', '.join(
                f'{k} {v:.2f} h' for k, v in g['runs'].items())
                     + f"; fastest {g['fastest_hours']:.2f} h.")
    L += ['', '## Greedy decision latency and peak memory', '',
          '| model | test set | decisions | n (CPU) | CPU s/instance | CPU ms/decision '
          '| CPU forward ms/decision | CPU peak RSS (MB) | n (GPU) | GPU s/instance '
          '| GPU ms/decision | GPU peak allocated (MB) |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for m in res['static']:
        for d in res['datasets']:
            c = res['cpu'].get(m, {}).get(d, {})
            g = res['gpu'].get(m, {}).get(d, {})
            if not c and not g:
                continue
            L.append(f"| {m} | {c.get('dataset', g.get('dataset', d))} | "
                     f"{c.get('decisions_per_instance', g.get('decisions_per_instance', '-'))} | "
                     f"{c.get('n_instances', '-')} | "
                     f"{fmt(c.get('latency_s_per_instance_mean'), 3)} | "
                     f"{fmt(c.get('latency_ms_per_decision_mean'), 2)} | "
                     f"{fmt(c.get('forward_ms_per_decision_mean'), 2)} | "
                     f"{fmt(c.get('peak_rss_MB'), 0)} | {g.get('n_instances', '-')} | "
                     f"{fmt(g.get('latency_s_per_instance_mean'), 3)} | "
                     f"{fmt(g.get('latency_ms_per_decision_mean'), 2)} | "
                     f"{fmt(g.get('peak_gpu_allocated_MB'), 1)} |")
    L.append('')
    with open(md_path, 'w') as f:
        f.write('\n'.join(L))
    return '\n'.join(L)


NOTES = [
    'Latency = one full greedy schedule of one instance (all decisions), batch of 1, '
    'environment initialisation excluded; forward = policy network only.',
    'Parameter counts are read from the checkpoint state dict (actor, critic, feature '
    'extractor, type embeddings); the critic is not needed for deployment.',
    'Training wall-clock is from train_time.txt; see training[*].quantity and '
    'contention_note.',
    'Each run records the load average and the busy fraction of its pinned cores; '
    'latencies measured on busy cores or a busy GPU are upper bounds.',
]


def main_orchestrator(args):
    args.models = args.models or default_models()
    if not args.models:
        sys.exit('[eval_cost] no model checkpoints found')
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    stem = os.path.splitext(args.out)[0]
    import torch
    if args.append and os.path.exists(args.out):
        with open(args.out) as f:
            res = json.load(f)
    else:
        res = {'machine': {}, 'runs': [], 'models': [], 'datasets': [], 'static': {},
               'training': {}, 'training_seed_groups': {}, 'cpu': {}, 'gpu': {}}
    res['machine'].update({'cpu': cpu_name(), 'n_cpus': os.cpu_count(),
                           'torch': torch.__version__, 'python': platform.python_version()})
    res['notes'] = NOTES
    affinity = sorted(os.sched_getaffinity(0))
    run = {'measured_at': datetime.datetime.now().isoformat(timespec='seconds'),
           'models': args.models, 'datasets': args.datasets, 'phases': [],
           'load_avg_start': load_avg(), 'affinity': affinity,
           'env_threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
           'pinned_cores_busy_before': core_busy(affinity)}
    run_id = len(res['runs'])
    for m in args.models:
        if m not in res['models']:
            res['models'].append(m)
    for d in args.datasets:
        if d not in res['datasets']:
            res['datasets'].append(d)
    runs = parse_train_time()
    for m in args.models:
        res['static'][m] = static_model_info(m)
        res['training'][m] = training_cost(m, runs)
        res['training_seed_groups'][m] = seed_group_cost(m, runs)

    def merge(phase, worker):
        for m, per_d in worker.pop('results').items():
            for d, entry in per_d.items():
                if isinstance(entry, dict):
                    entry['run'] = run_id
                res[phase].setdefault(m, {})[d] = entry
        run[f'{phase}_worker'] = worker
        run['phases'].append(phase)

    if args.cpu != 'off':
        merge('cpu', spawn_worker(args, 'cpu', stem + '.cpu_worker.json'))
        os.remove(stem + '.cpu_worker.json')

    if args.gpu == 'off':
        run['gpu_status'] = 'not run (--gpu off)'
    elif not os.path.exists(args.gpu_marker):
        run['gpu_status'] = f'not run: GPU marker {args.gpu_marker} absent at measurement time'
    else:
        before = gpu_state()
        res['machine']['gpu'] = before
        run['gpu_state_before_probe'] = before
        util = before.get('utilization_pct')
        contended = util is None or util > 5.0
        worker = spawn_worker(args, 'gpu', stem + '.gpu_worker.json')
        os.remove(stem + '.gpu_worker.json')
        run['gpu_state_after_probe'] = gpu_state()
        run['gpu_contended'] = contended
        run['gpu_status'] = (
            f"{args.n_inst_gpu} instance(s) per test set, budget {args.gpu_budget_s:.0f} s "
            f"({worker['phase_seconds']:.0f} s used"
            f"{', budget exhausted' if worker['budget_exhausted'] else ''}); GPU utilisation "
            f"{util} % with {before.get('compute_processes')} compute processes of other jobs "
            f"before the probe: "
            f"{'CONTENDED, GPU latencies are upper bounds' if contended else 'quiet'}")
        merge('gpu', worker)
    if 'gpu' not in res['machine']:
        res['machine']['gpu'] = gpu_state()
    run['load_avg_end'] = load_avg()
    res['runs'].append(run)
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=1, default=str)
    md = write_markdown(res, stem + '.md')
    print(md)
    print(f'wrote {args.out} and {stem}.md')


if __name__ == '__main__':
    _args = parse_cli()
    if _args.phase == 'main':
        main_orchestrator(_args)
    else:
        run_worker(_args)
