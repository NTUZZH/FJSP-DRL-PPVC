"""
make_gantt.py
=============
Gantt chart of one capacity-tight PPVC instance scheduled twice: by the
lag-aware policy (A3) and by the lag-blind policy (A0) whose plan is then
right-shift repaired against the true lags. Post-operation lag windows
(curing, ponding test, paint drying) are drawn on separate rows at the bottom
of each panel, so the reader sees stations idle while modules wait.

Representative instance: with --instance auto (default) both policies are
rolled out on every instance of --data_path, the per-instance makespan gap
(A0 repaired minus A3) is computed, and the instance whose gap is closest to
the class median gap is plotted (ties broken by the lower instance id).

Pipeline (per instance, batch of 1, greedy decoding):
  * lag-aware model: rollout on the lag-aware env (as in eval_ppvc.py).
  * lag-blind model: rollout on a lag-free env, then right_shift_repair with
    the true lags, keeping the lag-blind machine order (as in
    eval_a0_repair.py).
  * both schedules are checked by schedule_validator.validate_schedule with
    the true lags.

Outputs (under --out, default "IEEE Paper/draft/figures_src"):
  fig_gantt.pdf / .png        double-column figure (7.16 in wide)
  fig_gantt_single.pdf / .png single-column variant (3.5 in wide)
  fig_gantt_data.json         all makespans, the chosen instance, and the two
                              plotted schedules (the figure's source data)

CLI
  python make_gantt.py --data_path data/PPVC/10x15+ppvc-mixed \
      --models 10x15+ppvc-mixed+full 10x15+ppvc-mixed+a0-lagblind \
      --instance auto --device cpu
  python make_gantt.py --plot_only      # re-plot from fig_gantt_data.json
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def parse_cli():
    """Parse our flags, then scrub argv before params.py parses it at import."""
    ap = argparse.ArgumentParser(description='Gantt chart: lag-aware vs '
                                             'lag-blind + repair')
    ap.add_argument('--data_path', type=str,
                    default='data/PPVC/10x15+ppvc-mixed')
    ap.add_argument('--instance', type=str, default='auto',
                    help="instance id (e.g. 37) or 'auto' = gap closest to "
                         "the class median gap")
    ap.add_argument('--models', nargs=2,
                    default=['10x15+ppvc-mixed+full',
                             '10x15+ppvc-mixed+a0-lagblind'],
                    metavar=('LAG_AWARE', 'LAG_BLIND'),
                    help='lag-aware checkpoint stem, lag-blind checkpoint stem')
    ap.add_argument('--out', type=str,
                    default=os.path.join('IEEE Paper', 'draft', 'figures_src'))
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--seed_test', type=int, default=50)
    ap.add_argument('--plot_only', action='store_true',
                    help='skip the rollouts and plot from fig_gantt_data.json')
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]
    return args


_ARGS = parse_cli()

ARCH_KEYS = (
    'use_lag_features', 'use_type_embedding',
    'fea_j_input_dim', 'fea_m_input_dim',
    'type_emb_dim', 'n_op_types', 'n_mch_types',
    'n_j', 'n_m', 'n_op',
    'num_heads_OAB', 'num_heads_MAB', 'layer_fea_output_dim',
    'num_mlp_layers_actor', 'hidden_dim_actor',
    'num_mlp_layers_critic', 'hidden_dim_critic',
    'dropout_prob',
    'framework',
    'bound_lag_free',
)

# Lag type of each lag-carrying operation (all other operations have no lag).
LAG_TYPE_OF_OP = {'concrete_pour': 'curing', 'waterproofing': 'ponding',
                  'paint_coat_1': 'drying', 'paint_coat_2': 'drying'}
LAG_ROWS = [('curing', 'Concrete curing'), ('ponding', 'Ponding test'),
            ('drying', 'Paint drying')]

# Station types in process order (top to bottom) with readable names.
STATION_ORDER = [('H', 'Steel fabrication'), ('A', 'Formwork'),
                 ('B', 'Concrete casting'), ('C', 'MEP fit-out'),
                 ('D', 'Finishing'), ('E', 'Painting'), ('F', 'Assembly'),
                 ('Q', 'Quality check'), ('G', 'Dispatch')]


# ---------------------------------------------------------------------------
# Rollouts
# ---------------------------------------------------------------------------

def load_model(model_name, device):
    """Apply the snapshot's architecture flags, build the policy, load it."""
    import torch
    from params import configs
    path = f'./train_log/PPVC/config_{model_name}.json'
    with open(path) as f:
        snap = json.load(f)
    configs.bound_lag_free = False
    for k in ARCH_KEYS:
        if k in snap and snap[k] is not None:
            setattr(configs, k, snap[k])
    configs.framework = snap.get('framework') or 'DANIEL'
    configs.device = device
    from common_utils import check_backbone_tag
    check_backbone_tag(model_name, configs.framework)
    from model.PPO import PPO_initialize
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(
        f'./trained_network/PPVC/{model_name}.pth',
        map_location=torch.device(device)))
    ppo.policy.eval()
    flags = dict(use_lag_features=bool(configs.use_lag_features),
                 use_type_embedding=bool(configs.use_type_embedding),
                 bound_lag_free=bool(getattr(configs, 'bound_lag_free',
                                             False)))
    return ppo, flags


def greedy_rollout(ppo, flags, jl, pt, meta, lag_aware):
    """Greedy rollout on the fixed-route env; lags zeroed if not lag_aware.

    Returns (makespan, assigned_mch[N], op_ct[N]).
    """
    import torch
    from common_utils import greedy_select_action
    from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    M = pt.shape[1]
    kw_env = dict(n_j=jl.shape[0], n_m=M,
                  use_lag_features=flags['use_lag_features'])
    if lag_aware and flags['bound_lag_free']:
        kw_env['bound_lag_free'] = True
    env = FJSPEnvForSameOpNums(**kw_env)
    kwargs = dict(time_lag_list=[meta['time_lag']] if lag_aware else None)
    if flags['use_type_embedding']:
        kwargs['op_type_list'] = [meta['op_type']]
        kwargs['mch_type_list'] = [meta['mch_type']]
    state = env.set_initial_data([jl], [pt], **kwargs)
    assigned = np.full(pt.shape[0], -1, dtype=int)
    while True:
        with torch.no_grad():
            pi, _ = ppo.policy(fea_j=state.fea_j_tensor,
                               op_mask=state.op_mask_tensor,
                               candidate=state.candidate_tensor,
                               fea_m=state.fea_m_tensor,
                               mch_mask=state.mch_mask_tensor,
                               comp_idx=state.comp_idx_tensor,
                               dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                               fea_pairs=state.fea_pairs_tensor,
                               op_type=state.op_type_tensor,
                               mch_type=state.mch_type_tensor)
        action = greedy_select_action(pi)
        a = int(action.cpu().numpy()[0])
        assigned[env.candidate[0, a // M]] = a % M
        state, _, done = env.step(actions=action.cpu().numpy())
        if done.all():
            break
    assert (assigned >= 0).all()
    return float(env.current_makespan[0]), assigned, env.true_op_ct[0].copy()


def schedule_record(jl, pt, meta, assigned, op_ct):
    """Plain-list schedule record: per op module, name, machine, start, end,
    lag."""
    job_of_op = np.repeat(np.arange(len(jl)), jl)
    start = [float(op_ct[i] - pt[i, assigned[i]]) for i in range(len(op_ct))]
    return dict(module=job_of_op.tolist(),
                op_name=[n.split(':', 1)[1] for n in meta['op_name']],
                machine=[int(m) for m in assigned],
                start=start, end=[float(c) for c in op_ct],
                lag=[int(x) for x in meta['time_lag']])


def run_rollouts(args):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cpu' if args.device == 'cpu' else 'cuda')
    from common_utils import setup_seed
    from ppvc_instance_generator import load_instance
    from right_shift_repair import right_shift_repair
    from schedule_validator import validate_schedule

    stems = sorted(p[:-4] for p in
                   glob.glob(os.path.join(args.data_path, 'instance_*.fjs')))
    assert stems, f'no instances under {args.data_path}'
    ids = [int(os.path.basename(s).split('_')[1]) for s in stems]
    aware_name, blind_name = args.models

    # All instances are needed only for the median-gap selection.
    if args.instance == 'auto':
        todo = list(range(len(stems)))
    else:
        todo = [ids.index(int(args.instance))]

    ms_aware, ms_blind_plan, ms_repair = {}, {}, {}
    sched = {}
    for role, name in (('aware', aware_name), ('blind', blind_name)):
        setup_seed(args.seed_test)
        ppo, flags = load_model(name, args.device)
        t0 = time.time()
        for k in todo:
            jl, pt, meta = load_instance(stems[k])
            iid = ids[k]
            if role == 'aware':
                ms, asg, ct = greedy_rollout(ppo, flags, jl, pt, meta, True)
                ms_aware[iid] = ms
            else:
                ms0, asg, ct0 = greedy_rollout(ppo, flags, jl, pt, meta, False)
                ct, ms = right_shift_repair(jl, pt, meta['time_lag'], asg,
                                            op_ct_lagblind=ct0)
                ms_blind_plan[iid] = ms0
                ms_repair[iid] = float(ms)
            res = validate_schedule(jl, pt, meta['time_lag'], asg, ct)
            assert res['feasible'], (name, iid, res['violations'][:5])
            assert abs(res['makespan'] - ms) < 1e-6
            sched[(role, iid)] = (asg, np.asarray(ct, dtype=float))
        print(f'[gantt] {name}: {len(todo)} instances in '
              f'{time.time() - t0:.1f} s')

    done_ids = sorted(ms_aware)
    gap = {i: ms_repair[i] - ms_aware[i] for i in done_ids}
    med = float(np.median([gap[i] for i in done_ids]))
    if args.instance == 'auto':
        chosen = min(done_ids, key=lambda i: (abs(gap[i] - med), i))
    else:
        chosen = int(args.instance)
    k = ids.index(chosen)
    jl, pt, meta = load_instance(stems[k])
    data = dict(
        data_path=args.data_path.replace(os.sep, '/'),
        models=dict(lag_aware=aware_name, lag_blind=blind_name),
        selection=('instance whose (lag-blind repaired - lag-aware) makespan '
                   'gap is closest to the class median gap'
                   if args.instance == 'auto' else 'user-specified'),
        n_instances=len(done_ids),
        median_gap=med,
        mean_makespan_lag_aware=float(np.mean(list(ms_aware.values()))),
        mean_makespan_lag_blind_repaired=float(
            np.mean(list(ms_repair.values()))),
        makespans={str(i): dict(lag_aware=ms_aware[i],
                                lag_blind_plan=ms_blind_plan[i],
                                lag_blind_repaired=ms_repair[i])
                   for i in done_ids},
        instance=chosen,
        instance_makespan_lag_aware=ms_aware[chosen],
        instance_makespan_lag_blind_repaired=ms_repair[chosen],
        mch_type=[int(x) for x in meta['mch_type']],
        schedules=dict(
            lag_aware=schedule_record(jl, pt, meta, *sched[('aware', chosen)]),
            lag_blind_repaired=schedule_record(jl, pt, meta,
                                               *sched[('blind', chosen)])),
    )
    return data


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

# Pale module fills (no yellow or amber), dark thin edges.
MODULE_FILLS = ['#a8c6e3', '#e3b5b9', '#b5d8a8', '#c9b8dc', '#9fd6dc',
                '#e2d6c6', '#d2b9a6', '#f4bfd9', '#c3cbd3', '#a3c9bb']
EDGE = '#333333'


def pack_lanes(intervals):
    """Greedy interval partitioning: lane index per interval (start order)."""
    order = sorted(range(len(intervals)), key=lambda i: intervals[i][0])
    lane_end, lane_of = [], [0] * len(intervals)
    for i in order:
        s, e = intervals[i]
        for L, end in enumerate(lane_end):
            if end <= s + 1e-9:
                lane_end[L] = e
                lane_of[i] = L
                break
        else:
            lane_end.append(e)
            lane_of[i] = len(lane_end) - 1
    return lane_of, len(lane_end)


def build_rows(mch_type):
    """Station rows in process order: list of (machine index, group), and the
    group names (one per station type)."""
    from ppvc_instance_generator import STATION_TYPES
    rows, names = [], []
    for g, (t, name) in enumerate(STATION_ORDER):
        names.append(name)
        for m, ty in enumerate(mch_type):
            if STATION_TYPES[ty] == t:
                rows.append((m, g))
    return rows, names


def draw_panel(ax, rec, mch_type, xmax, makespan, lanes_per_row, style):
    import matplotlib.patches as mpatches
    rows, names = build_rows(mch_type)
    row_h = 1.0
    bar_h = 0.72
    lag_h = style['lag_row_h']   # height of one lane in a lag row
    # y layout, top to bottom: station rows (one label per station type,
    # centred on its rows), a gap, then the lag rows
    y_of_mch, yt, yl = {}, [], []
    y = 0.0
    sep = []
    g_span = {}
    for m, g in rows:
        if g_span and g not in g_span:
            sep.append(y)
        y_of_mch[m] = y + 0.5 * row_h
        lo, hi = g_span.get(g, (y, y))
        g_span[g] = (lo, y + row_h)
        y += row_h
    for g, (lo, hi) in g_span.items():
        yt.append(0.5 * (lo + hi))
        yl.append(names[g])
    y += style['gap']
    lag_box = {}
    for r, (key, lab) in enumerate(LAG_ROWS):
        h = lanes_per_row[key] * lag_h + 0.3
        if r > 0:
            sep.append(y)
        lag_box[key] = (y + 0.15, y + h - 0.15)
        yt.append(y + 0.5 * h)
        yl.append(lab)
        y += h
    y_total = y

    # group separators and the station / lag boundary
    for s in sep:
        ax.axhline(s, color='#BFBFBF', lw=0.35, zorder=0)
    ax.axhline(len(rows) * row_h + 0.5 * style['gap'], color='#808080',
               lw=0.5, zorder=0)

    fs_in = style['fs_in']
    min_w = style['label_min_w']
    n = len(rec['start'])
    for i in range(n):
        s, e = rec['start'][i], rec['end'][i]
        mod = rec['module'][i]
        yc = y_of_mch[rec['machine'][i]]
        ax.add_patch(mpatches.Rectangle(
            (s, yc - bar_h / 2), e - s, bar_h, fc=MODULE_FILLS[mod],
            ec=EDGE, lw=style['edge_lw'], zorder=2))
    # module id: one label per run of back-to-back ops of one module on one
    # station, centred on the run's widest op (so no op edge crosses it) and
    # only where that op is wide enough to hold it
    for m in y_of_mch:
        ops = sorted((i for i in range(n) if rec['machine'][i] == m),
                     key=lambda i: rec['start'][i])
        runs = []
        for i in ops:
            w = rec['end'][i] - rec['start'][i]
            if (runs and rec['module'][i] == runs[-1][0]
                    and abs(rec['start'][i] - runs[-1][3]) < 1e-6):
                runs[-1][3] = rec['end'][i]
                if w > runs[-1][2] - runs[-1][1]:
                    runs[-1][1:3] = [rec['start'][i], rec['end'][i]]
            else:
                runs.append([rec['module'][i], rec['start'][i],
                             rec['end'][i], rec['end'][i]])
        for mod, s, e, _ in runs:
            if e - s >= min_w:
                ax.text((s + e) / 2, y_of_mch[m], str(mod + 1), ha='center',
                        va='center_baseline', fontsize=fs_in, color='black',
                        zorder=3)

    # lag windows: [end of op, end + lag], lane-packed within each lag row
    for key, _ in LAG_ROWS:
        idx = [i for i in range(n) if rec['lag'][i] > 0
               and LAG_TYPE_OF_OP.get(rec['op_name'][i]) == key]
        iv = [(rec['end'][i], rec['end'][i] + rec['lag'][i]) for i in idx]
        lane_of, _ = pack_lanes(iv)
        y0, y1 = lag_box[key]
        lh = (y1 - y0) / lanes_per_row[key]
        for (s, e), i, ln in zip(iv, idx, lane_of):
            ax.add_patch(mpatches.Rectangle(
                (s, y0 + ln * lh + 0.08 * lh), e - s, 0.84 * lh,
                fc=MODULE_FILLS[rec['module'][i]], ec=EDGE,
                lw=style['edge_lw'], hatch=style['hatch'], zorder=2))

    ax.axvline(makespan, color='black', ls=(0, (4, 2)), lw=0.8, zorder=4)
    ax.set_xlim(0, xmax)
    ax.set_ylim(y_total + 0.15, -0.15)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    ax.tick_params(axis='y', length=0, pad=3)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.grid(axis='x', color='#D9D9D9', lw=0.35, zorder=0)
    ax.set_axisbelow(True)


def plot(data, out_stem, width, height, style):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Liberation Serif', 'Nimbus Roman', 'TeX Gyre Termes',
                       'Times New Roman'],
        'mathtext.fontset': 'stix',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'font.size': style['fs'], 'axes.labelsize': style['fs'],
        'xtick.labelsize': style['fs_tick'],
        'ytick.labelsize': style['fs_tick'],
        'legend.fontsize': style['fs_tick'],
        'hatch.linewidth': 0.35, 'hatch.color': EDGE,
        'axes.edgecolor': '#1A1A1A', 'text.color': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'xtick.major.width': 0.5, 'xtick.major.size': 2.5,
    })
    recs = [data['schedules']['lag_aware'],
            data['schedules']['lag_blind_repaired']]
    mss = [data['instance_makespan_lag_aware'],
           data['instance_makespan_lag_blind_repaired']]
    # one lane count per lag row, shared by both panels so rows align
    lanes = {}
    for key, _ in LAG_ROWS:
        need = 1
        for rec in recs:
            iv = [(rec['end'][i], rec['end'][i] + rec['lag'][i])
                  for i in range(len(rec['start'])) if rec['lag'][i] > 0
                  and LAG_TYPE_OF_OP.get(rec['op_name'][i]) == key]
            need = max(need, pack_lanes(iv)[1] if iv else 1)
        lanes[key] = need
    print('[gantt] lanes per lag row:', lanes)
    step = style['xstep']
    xmax = 50 * np.ceil(1.02 * max(mss) / 50)

    fig, axes = plt.subplots(2, 1, figsize=(width, height), sharex=True)
    titles = [f'(a) Lag-aware policy, makespan {mss[0]:.0f} h',
              f'(b) Lag-blind policy after right-shift repair, '
              f'makespan {mss[1]:.0f} h']
    for ax, rec, ms, t in zip(axes, recs, mss, titles):
        draw_panel(ax, rec, data['mch_type'], xmax, ms, lanes, style)
        ax.set_title(t, fontsize=style['fs'], pad=style['title_pad'])
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
    axes[1].set_xlabel('Time (h)')
    axes[0].tick_params(axis='x', labelbottom=True)

    handles = [mpatches.Patch(fc=MODULE_FILLS[j], ec=EDGE, lw=0.4,
                              label=f'Module {j + 1}') for j in range(10)]
    handles.append(mpatches.Patch(fc='white', ec=EDGE, lw=0.4,
                                  hatch=style['hatch'],
                                  label='Module in lag window'))
    handles.append(Line2D([0], [0], color='black', ls=(0, (4, 2)), lw=0.8,
                          label='Makespan'))
    leg = fig.legend(handles=handles, loc='upper center',
                     ncol=style['leg_ncol'], frameon=True, fancybox=False,
                     edgecolor='black', framealpha=1.0,
                     bbox_to_anchor=(0.5 * (style['adjust']['left'] + style['adjust']['right']) + style['leg_dx'], 1.0),
                     handlelength=1.6, handleheight=0.9,
                     columnspacing=style['leg_colsp'], borderpad=0.4,
                     labelspacing=0.35, handletextpad=0.5)
    leg.get_frame().set_linewidth(0.5)
    fig.subplots_adjust(**style['adjust'])
    fig.savefig(out_stem + '.pdf')
    fig.savefig(out_stem + '.png', dpi=300)
    plt.close(fig)
    print(f'[gantt] wrote {out_stem}.pdf/.png')


STYLE_DOUBLE = dict(fs=7.5, fs_tick=7, fs_in=5.5, label_min_w=6.5,
                    edge_lw=0.3, hatch='//////', lag_row_h=0.42, gap=0.4,
                    xstep=50, title_pad=3, leg_ncol=6, leg_dx=0.0,
                    leg_colsp=1.2,
                    adjust=dict(left=0.14, right=0.985, top=0.89,
                                bottom=0.075, hspace=0.2))
STYLE_SINGLE = dict(fs=7, fs_tick=6.5, fs_in=4.5, label_min_w=np.inf,
                    edge_lw=0.25, hatch='//////', lag_row_h=0.42, gap=0.4,
                    xstep=100, title_pad=3, leg_ncol=3, leg_dx=-0.04,
                    leg_colsp=0.8,
                    adjust=dict(left=0.27, right=0.965, top=0.855,
                                bottom=0.055, hspace=0.2))


def main():
    args = _ARGS
    os.makedirs(args.out, exist_ok=True)
    data_file = os.path.join(args.out, 'fig_gantt_data.json')
    if args.plot_only:
        with open(data_file) as f:
            data = json.load(f)
    else:
        data = run_rollouts(args)
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=1)
        print(f'[gantt] wrote {data_file}')
    print(f"[gantt] class median gap {data['median_gap']:.1f} h over "
          f"{data['n_instances']} instances; instance {data['instance']}: "
          f"lag-aware {data['instance_makespan_lag_aware']:.0f} h, lag-blind "
          f"repaired {data['instance_makespan_lag_blind_repaired']:.0f} h")
    plot(data, os.path.join(args.out, 'fig_gantt'), 7.16, 5.2, STYLE_DOUBLE)
    plot(data, os.path.join(args.out, 'fig_gantt_single'), 3.5, 5.6,
         STYLE_SINGLE)


if __name__ == '__main__':
    main()
