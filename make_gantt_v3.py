"""
make_gantt_v3.py
================
Gantt charts of one PPVC order scheduled by ONE network, the network trained
without the post-operation waits, executed in two ways:

  (a) dispatched in the lag-aware transition: the policy acts on the
      environment that carries the true waits (eval_ppvc.rollout_greedy), so a
      module waits after curing, the ponding test or paint drying while its
      station serves other modules;
  (b) plan then repair: the same policy plans on a wait-free environment
      (eval_a0_repair.rollout_greedy_lagblind), and the plan is right-shift
      repaired against the true waits with its station sequences kept
      (right_shift_repair.right_shift_repair).

Both schedules of every order are re-validated against the true waits with
schedule_validator.validate_schedule (feasibility and makespan asserted), and
every per-order makespan is compared with the result files that the
evaluation scripts wrote for the paper, where such a file exists (read only;
those files come from GPU rollouts, so a few orders may differ by a near-tie
broken differently on the CPU, and the differences are recorded).

Order selection (--instance auto, the default): for every order of the test
set, repair penalty = 100 x (makespan of (b) / makespan of (a) - 1). The order
whose penalty is closest to the median penalty over all orders is plotted;
ties are broken by the lower order id. With an even number of orders the
median is the mean of the two middle penalties, so these two orders tie by
construction and the lower id is plotted.

Test sets: the capacity-tight 15-station set (default,
data/PPVC/10x15+ppvc-mixed) and the default 25-station set
(data/PPVC/10x25+ppvc-mixed), 100 orders of 10 modules each. The network is
the same in both cases.

Outputs (under --out, default "IEEE Paper/draft/figures_src"):
  <stem>.pdf / .png          two stacked panels sharing the time axis
  <stem>_data.json           per-order makespans, the selection rule, the
                             chosen order and its two plotted schedules

CLI
  python make_gantt_v3.py                                  # 15 stations
  python make_gantt_v3.py --data_path data/PPVC/10x25+ppvc-mixed \
      --stem fig_gantt_v3_default                          # 25 stations
  python make_gantt_v3.py --plot_only [--stem ...]         # re-plot

Run on the CPU with one thread, for example
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      taskset -c 7 python make_gantt_v3.py
"""
import argparse
import json
import os
import sys
import time

import numpy as np


def parse_cli():
    """Parse our flags, then scrub argv before params.py parses it at import."""
    ap = argparse.ArgumentParser(
        description='Gantt charts: one wait-free-trained network, dispatched '
                    'with the waits vs planned and right-shift repaired')
    ap.add_argument('--data_path', default='data/PPVC/10x15+ppvc-mixed')
    ap.add_argument('--model_name', default='10x25+ppvc-mixed+a0-2k-s301')
    ap.add_argument('--instance', default='auto',
                    help="order id, or 'auto' = penalty closest to the median")
    ap.add_argument('--seed_test', type=int, default=50)
    ap.add_argument('--out', default=os.path.join('IEEE Paper', 'draft',
                                                  'figures_src'))
    ap.add_argument('--stem', default='fig_gantt_v3',
                    help='output stem: <stem>.pdf/.png and <stem>_data.json')
    ap.add_argument('--plot_only', action='store_true')
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]
    return args


_ARGS = parse_cli()

# Wait type of each wait-carrying operation (no other operation has a wait).
WAIT_TYPE_OF_OP = {'concrete_pour': 'curing', 'waterproofing': 'ponding',
                   'paint_coat_1': 'drying', 'paint_coat_2': 'drying'}
WAIT_ROWS = [('curing', 'Concrete curing'), ('ponding', 'Ponding test'),
             ('drying', 'Paint drying')]

# Station types in process order (top to bottom) with readable names.
STATION_ORDER = [('H', 'Steel fabrication'), ('A', 'Formwork'),
                 ('B', 'Concrete casting'), ('C', 'MEP fit-out'),
                 ('D', 'Finishing'), ('E', 'Painting'), ('F', 'Assembly'),
                 ('Q', 'Quality check'), ('G', 'Dispatch')]


# ---------------------------------------------------------------------------
# Rollouts (repository code only)
# ---------------------------------------------------------------------------

def _import_scrubbed(name):
    """Import a module that parses sys.argv at import time, with a clean argv."""
    saved = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        mod = __import__(name)
    finally:
        sys.argv = saved
    return mod


def schedule_record(jl, pt, meta, assigned, op_ct):
    """Plain-list schedule: per operation its module, name, station, start,
    end and wait (h)."""
    job_of_op = np.repeat(np.arange(len(jl)), jl)
    start = [float(op_ct[i] - pt[i, assigned[i]]) for i in range(len(op_ct))]
    return dict(module=job_of_op.tolist(),
                op_name=[n.split(':', 1)[1] for n in meta['op_name']],
                station=[int(m) for m in assigned],
                start=start, end=[float(c) for c in op_ct],
                wait=[float(x) for x in meta['time_lag']])


def run_rollouts(args):
    import eval_byproducts as eb
    ep = eb.import_eval_ppvc()            # CPU, one torch thread
    a0 = _import_scrubbed('eval_a0_repair')
    from ppvc_instance_generator import (load_instance, DEFAULT_FACTORY,
                                         TIGHT_FACTORY)
    from right_shift_repair import right_shift_repair
    from schedule_validator import validate_schedule

    ep.ROUTE_VARIANT = ep.detect_route_variant(args.data_path, 'auto')
    assert ep.ROUTE_VARIANT == 'fixed', 'the wait-free rollout needs fixed routes'
    ppo, use_lag, use_type, ckpt = eb.load_policy(ep, args.model_name,
                                                  args.seed_test)
    from params import configs
    assert configs.framework == 'DANIEL', configs.framework
    assert not use_lag and not use_type, 'expected a network without aids'
    assert not bool(getattr(configs, 'bound_lag_free', False))

    stems = ep.list_instances(args.data_path)
    dataset = os.path.basename(os.path.normpath(args.data_path))
    ids = [int(os.path.basename(s).split('_')[1]) for s in stems]
    assert ids == list(range(len(ids))), 'order ids must be 0..n-1'

    ms_disp, ms_plan, ms_rep, sched = [], [], [], {}
    t0 = time.time()
    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        # (a) dispatched in the lag-aware transition, validated in run_validated
        ms_a, _, asg_a, ct_a = eb.run_validated(
            ep, 'greedy', jl, pt, meta, i, args.seed_test,
            (ppo, use_lag, use_type))
        # (b) wait-free plan, then right-shift repair against the true waits
        ms_p, _, asg_b, ct_p = a0.rollout_greedy_lagblind(
            ppo, jl, pt, meta, use_lag, use_type)
        ct_b, ms_b = right_shift_repair(jl, pt, meta['time_lag'], asg_b,
                                        op_ct_lagblind=ct_p)
        res = validate_schedule(jl, pt, meta['time_lag'], asg_b, ct_b)
        assert res['feasible'], (i, res['violations'][:5])
        assert abs(res['makespan'] - ms_b) < 1e-6, (i, res['makespan'], ms_b)
        assert ms_b >= ms_p - 1e-6
        ms_disp.append(float(ms_a))
        ms_plan.append(float(ms_p))
        ms_rep.append(float(ms_b))
        sched[i] = ((asg_a, np.asarray(ct_a, float)),
                    (asg_b, np.asarray(ct_b, float)))
        if (i + 1) % 20 == 0:
            print(f'[gantt v3] {i + 1}/{len(stems)} orders, '
                  f'{time.time() - t0:.0f} s', flush=True)
    ms_disp, ms_plan, ms_rep = map(np.array, (ms_disp, ms_plan, ms_rep))

    # Compare with the paper's result files (read only). Those files were
    # written by GPU rollouts; a CPU greedy rollout can break a near-tie of
    # the policy scores differently, so a few orders may differ. The
    # differences are recorded, not hidden.
    checks, stored = {}, {}
    for key, prefix, mine in (('dispatched', 'greedy', ms_disp),
                              ('plan', 'A0BLIND', ms_plan),
                              ('repaired', 'A0REPAIR', ms_rep)):
        path = os.path.join('test_results', 'PPVC', dataset,
                            f'Result_{prefix}+{args.model_name}_{dataset}.npy')
        rel = path.replace(os.sep, '/')
        if not os.path.exists(path):
            checks[key] = dict(result_file=rel, note='no stored result file')
            continue
        ref = np.load(path)[:, 0]
        assert ref.shape == mine.shape, (path, ref.shape)
        stored[key] = ref
        d = mine - ref
        checks[key] = dict(
            result_file=rel,
            n_orders_equal=int((np.abs(d) < 1e-6).sum()),
            max_abs_difference_h=float(np.abs(d).max()),
            mean_difference_h=float(d.mean()),
            mean_stored_h=float(ref.mean()), mean_this_run_h=float(mine.mean()))
    if 'dispatched' in stored and 'repaired' in stored:
        sp = 100.0 * (stored['repaired'] / stored['dispatched'] - 1.0)
        checks['median_penalty_pct_from_stored_files'] = float(np.median(sp))
    print('[gantt v3] stored-file comparison:', json.dumps(checks, indent=1))

    pen = 100.0 * (ms_rep / ms_disp - 1.0)
    med = float(np.median(pen))
    if args.instance == 'auto':
        chosen = min(ids, key=lambda k: (round(abs(pen[k] - med), 9), k))
    else:
        chosen = int(args.instance)
    order = np.argsort(pen, kind='stable')
    mid = [int(order[len(pen) // 2 - 1]), int(order[len(pen) // 2])]

    jl, pt, meta = load_instance(stems[chosen])
    (asg_a, ct_a), (asg_b, ct_b) = sched[chosen]
    for asg, ct, ms in ((asg_a, ct_a, ms_disp[chosen]),
                        (asg_b, ct_b, ms_rep[chosen])):
        res = validate_schedule(jl, pt, meta['time_lag'], asg, ct)
        assert res['feasible'] and abs(res['makespan'] - ms) < 1e-6
    from ppvc_instance_generator import STATION_TYPES
    counts = {t: int(sum(1 for m in meta['mch_type']
                         if STATION_TYPES[m] == t)) for t in STATION_TYPES}
    factory = [name for name, fac in (('default', DEFAULT_FACTORY),
                                      ('capacity-tight', TIGHT_FACTORY))
               if counts == {k: int(v) for k, v in fac.items()}]
    assert len(factory) == 1, counts

    return dict(
        data_path=args.data_path.replace(os.sep, '/'),
        model=args.model_name,
        checkpoint=ckpt.replace('./', '').replace(os.sep, '/'),
        backbone='DANIEL', input_aids='none', trained_with_waits=False,
        seed_test=args.seed_test,
        execution=dict(
            a='greedy rollout in the lag-aware transition '
              '(eval_ppvc.rollout_greedy, via eval_byproducts.run_validated)',
            b='greedy rollout on a wait-free environment '
              '(eval_a0_repair.rollout_greedy_lagblind), then '
              'right_shift_repair against the true waits'),
        validation='every schedule of both executions on every order '
                   'validated against the true waits; validator makespan '
                   'equals the reported makespan',
        device='cpu',
        stored_result_comparison=checks,
        selection=dict(
            penalty='100 x (repaired makespan / dispatched makespan - 1), '
                    'per order',
            rule=('order whose penalty is closest to the median penalty over '
                  'all orders; ties broken by the lower order id'
                  if args.instance == 'auto' else 'user-specified order'),
            n_orders=len(ids),
            median_penalty_pct=med,
            two_middle_orders=mid,
            two_middle_penalties_pct=[float(pen[k]) for k in mid],
            median_difference_h=float(np.median(ms_rep - ms_disp)),
            mean_penalty_pct=float(pen.mean()),
            penalty_of_means_pct=float(100.0 * (ms_rep.mean()
                                               / ms_disp.mean() - 1.0))),
        mean_makespan_h=dict(dispatched=float(ms_disp.mean()),
                             plan=float(ms_plan.mean()),
                             repaired=float(ms_rep.mean())),
        per_order=[dict(order=k, dispatched=float(ms_disp[k]),
                        plan=float(ms_plan[k]), repaired=float(ms_rep[k]),
                        penalty_pct=float(pen[k]),
                        **{f'stored_{key}': float(v[k])
                           for key, v in stored.items()})
                   for k in ids],
        instance=chosen,
        instance_file=os.path.basename(stems[chosen]) + '.fjs',
        instance_makespan_h=dict(dispatched=float(ms_disp[chosen]),
                                 plan=float(ms_plan[chosen]),
                                 repaired=float(ms_rep[chosen])),
        instance_penalty_pct=float(pen[chosen]),
        factory=factory[0], n_stations=int(sum(counts.values())),
        station_counts=counts,
        mch_type=[int(x) for x in meta['mch_type']],
        schedules=dict(
            dispatched=schedule_record(jl, pt, meta, asg_a, ct_a),
            repaired=schedule_record(jl, pt, meta, asg_b, ct_b)),
    )


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

# Pale module fills (no yellow or amber) with thin dark edges.
MODULE_FILLS = ['#a8c6e3', '#e3b5b9', '#b5d8a8', '#c9b8dc', '#9fd6dc',
                '#e2d6c6', '#d2b9a6', '#f4bfd9', '#c3cbd3', '#a3c9bb']
EDGE = '#333333'
TERMES = os.path.expanduser(
    '~/.TinyTeX/texmf-dist/fonts/opentype/public/tex-gyre')


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


def wait_intervals(rec, key):
    idx = [i for i in range(len(rec['start'])) if rec['wait'][i] > 0
           and WAIT_TYPE_OF_OP.get(rec['op_name'][i]) == key]
    return idx, [(rec['end'][i], rec['end'][i] + rec['wait'][i]) for i in idx]


def build_rows(mch_type):
    """Station rows in process order: [(station index, group)], group names."""
    from ppvc_instance_generator import STATION_TYPES
    rows, names = [], []
    for g, (t, name) in enumerate(STATION_ORDER):
        names.append(name)
        for m, ty in enumerate(mch_type):
            if STATION_TYPES[ty] == t:
                rows.append((m, g))
    return rows, names


def draw_panel(ax, rec, mch_type, xmax, makespan, lanes, st):
    import matplotlib.patches as mpatches
    rows, names = build_rows(mch_type)
    bar_h = 0.72
    y_of, yt, yl, sep, span = {}, [], [], [], {}
    y = 0.0
    for m, g in rows:
        if span and g not in span:
            sep.append(y)
        y_of[m] = y + 0.5
        lo, _ = span.get(g, (y, y))
        span[g] = (lo, y + 1.0)
        y += 1.0
    for g, (lo, hi) in span.items():
        yt.append(0.5 * (lo + hi))
        yl.append(names[g])
    boundary = y + 0.5 * st['gap']
    y += st['gap']
    box = {}
    for r, (key, lab) in enumerate(WAIT_ROWS):
        h = lanes[key] * st['lane_h'] + 0.3
        if r > 0:
            sep.append(y)
        box[key] = (y + 0.15, y + h - 0.15)
        yt.append(y + 0.5 * h)
        yl.append(lab)
        y += h
    y_total = y

    for s in sep:
        ax.axhline(s, color='#BFBFBF', lw=0.35, zorder=0)
    ax.axhline(boundary, color='#808080', lw=0.6, zorder=0)

    n = len(rec['start'])
    for i in range(n):
        s, e = rec['start'][i], rec['end'][i]
        yc = y_of[rec['station'][i]]
        ax.add_patch(mpatches.Rectangle(
            (s, yc - bar_h / 2), e - s, bar_h,
            fc=MODULE_FILLS[rec['module'][i]], ec=EDGE, lw=st['edge_lw'],
            zorder=2))
    # module number: one per run of back-to-back operations of one module on
    # one station, centred on the run's widest operation, only if it fits
    for m in y_of:
        ops = sorted((i for i in range(n) if rec['station'][i] == m),
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
            need = st['min_w1'] if mod < 9 else st['min_w2']
            if e - s >= need:
                ax.text((s + e) / 2, y_of[m], str(mod + 1), ha='center',
                        va='center_baseline', fontsize=st['fs_in'],
                        color='black', zorder=3)

    # waits: [end of operation, end + wait], lane-packed within each row
    for key, _ in WAIT_ROWS:
        idx, iv = wait_intervals(rec, key)
        lane_of, _ = pack_lanes(iv)
        y0, y1 = box[key]
        lh = (y1 - y0) / lanes[key]
        for (s, e), i, ln in zip(iv, idx, lane_of):
            ax.add_patch(mpatches.Rectangle(
                (s, y0 + ln * lh + 0.1 * lh), e - s, 0.8 * lh,
                fc=MODULE_FILLS[rec['module'][i]], ec=EDGE,
                lw=st['edge_lw'], hatch=st['hatch'], zorder=2))

    ax.axvline(makespan, color='black', ls=(0, (4, 2)), lw=0.9, zorder=4)
    ax.set_xlim(0, xmax)
    ax.set_ylim(y_total + 0.15, -0.15)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl)
    ax.tick_params(axis='y', length=0, pad=4)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.grid(axis='x', color='#D9D9D9', lw=0.4, zorder=0)
    ax.set_axisbelow(True)


STYLE = dict(width=6.5, height=8.0, fs=9, fs_tick=8.5, fs_in=5.5,
             min_w1=5.0, min_w2=7.4, edge_lw=0.3, hatch='//////',
             lane_h=0.45, gap=0.5, xstep=50, leg_ncol=6, leg_colsp=0.9,
             adjust=dict(left=0.165, right=0.975, top=0.905, bottom=0.062,
                         hspace=0.16))


def otf_to_ttf(src, dst):
    """Convert a CFF-flavoured OpenType font to TrueType outlines.

    matplotlib embeds fonts as Type 42 (TrueType); a CFF font would land in a
    TrueType font stream, which the PDF specification does not allow. The
    quadratic conversion keeps the Termes name and metrics."""
    from fontTools.ttLib import TTFont, newTable
    from fontTools.pens.cu2quPen import Cu2QuPen
    from fontTools.pens.ttGlyphPen import TTGlyphPen
    font = TTFont(src)
    assert font.sfntVersion == 'OTTO' and 'CFF ' in font
    order = font.getGlyphOrder()
    gset = font.getGlyphSet()
    glyphs = {}
    for name in gset.keys():
        tt_pen = TTGlyphPen(gset)
        gset[name].draw(Cu2QuPen(tt_pen, 1.0, reverse_direction=True))
        glyphs[name] = tt_pen.glyph()
    font['loca'] = newTable('loca')
    font['glyf'] = glyf = newTable('glyf')
    glyf.glyphOrder = order
    glyf.glyphs = glyphs
    del font['CFF ']
    if 'VORG' in font:
        del font['VORG']
    glyf.compile(font)
    hmtx = font['hmtx']
    for name, g in glyf.glyphs.items():
        if hasattr(g, 'xMin'):
            hmtx[name] = (hmtx[name][0], g.xMin)
    font['maxp'] = maxp = newTable('maxp')
    maxp.tableVersion = 0x00010000
    for k in ('maxZones',):
        setattr(maxp, k, 1)
    for k in ('maxTwilightPoints', 'maxStorage', 'maxFunctionDefs',
              'maxInstructionDefs', 'maxStackElements',
              'maxSizeOfInstructions', 'maxComponentElements'):
        setattr(maxp, k, 0)
    maxp.compile(font)
    post = font['post']
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = order
    font.sfntVersion = '\x00\x01\x00\x00'
    font.save(dst)


def setup_fonts(tmpdir):
    """Register TeX Gyre Termes (the Times clone of the manuscript's
    newtxtext) as TrueType, so every figure font embeds as valid Type 42."""
    import matplotlib
    from matplotlib import font_manager
    for f in ('regular', 'bold'):
        dst = os.path.join(tmpdir, f'texgyretermes-{f}.ttf')
        otf_to_ttf(os.path.join(TERMES, f'texgyretermes-{f}.otf'), dst)
        font_manager.fontManager.addfont(dst)
    matplotlib.rcParams.update({
        'font.family': 'serif', 'font.serif': ['TeX Gyre Termes'],
        'mathtext.fontset': 'custom', 'mathtext.rm': 'TeX Gyre Termes',
        'mathtext.bf': 'TeX Gyre Termes:bold',
        'axes.unicode_minus': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def plot(data, out_stem, st=STYLE):
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:   # fonts live until saved
        _plot(data, out_stem, st, tmpdir)


def _plot(data, out_stem, st, tmpdir):
    import matplotlib
    matplotlib.use('Agg')
    setup_fonts(tmpdir)
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.ticker as mticker
    from matplotlib.lines import Line2D
    plt.rcParams.update({
        'font.size': st['fs'], 'axes.labelsize': st['fs'],
        'xtick.labelsize': st['fs_tick'], 'ytick.labelsize': st['fs'],
        'legend.fontsize': st['fs'],
        'hatch.linewidth': 0.35, 'hatch.color': EDGE,
        'axes.edgecolor': '#1A1A1A', 'text.color': 'black',
        'xtick.color': 'black', 'ytick.color': 'black',
        'xtick.major.width': 0.5, 'xtick.major.size': 2.5,
    })
    recs = [data['schedules']['dispatched'], data['schedules']['repaired']]
    mss = [data['instance_makespan_h']['dispatched'],
           data['instance_makespan_h']['repaired']]
    for rec, ms in zip(recs, mss):
        assert abs(max(rec['end']) - ms) < 1e-6
    lanes = {}
    for key, _ in WAIT_ROWS:
        lanes[key] = max([1] + [pack_lanes(wait_intervals(r, key)[1])[1]
                                for r in recs if wait_intervals(r, key)[1]])
    print('[gantt v3] lanes per wait row:', lanes)
    xmax = st['xstep'] * np.ceil(1.02 * max(mss) / st['xstep'])

    fig, axes = plt.subplots(2, 1, figsize=(st['width'], st['height']),
                             sharex=True)
    fig.subplots_adjust(**st['adjust'])
    for ax, rec, ms, tag in zip(axes, recs, mss, ('(a)', '(b)')):
        draw_panel(ax, rec, data['mch_type'], xmax, ms, lanes, st)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(st['xstep']))
        ax.tick_params(axis='x', labelbottom=True)
        bb = ax.get_position()
        fig.text(0.012, bb.y1 + 0.006, tag, ha='left', va='bottom',
                 fontsize=st['fs'], fontweight='bold')
    axes[1].set_xlabel('Time (h)')

    handles = [mpatches.Patch(fc=MODULE_FILLS[j], ec=EDGE, lw=0.4,
                              label=f'Module {j + 1}') for j in range(10)]
    handles.append(mpatches.Patch(fc='white', ec=EDGE, lw=0.4,
                                  hatch=st['hatch'],
                                  label='Module during a wait'))
    handles.append(Line2D([0], [0], color='black', ls=(0, (4, 2)), lw=0.9,
                          label='Makespan'))
    # the legend fills column by column: interleave so that the top row reads
    # Modules 1-5 and the wait swatch, the bottom row Modules 6-10 and the
    # makespan line
    order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9, 10, 11]
    leg = fig.legend(handles=[handles[k] for k in order], loc='upper center',
                     ncol=st['leg_ncol'], frameon=True, fancybox=False,
                     edgecolor='black', framealpha=1.0,
                     bbox_to_anchor=(0.5, 0.997),
                     handlelength=1.6, handleheight=0.9,
                     columnspacing=st['leg_colsp'], borderpad=0.45,
                     labelspacing=0.4, handletextpad=0.45)
    leg.get_frame().set_linewidth(0.5)
    fig.savefig(out_stem + '.pdf', metadata={'Title': 'Gantt charts'})
    fig.savefig(out_stem + '.png', dpi=300)
    plt.close(fig)
    print(f'[gantt v3] wrote {out_stem}.pdf/.png')


def main():
    args = _ARGS
    os.makedirs(args.out, exist_ok=True)
    data_file = os.path.join(args.out, f'{args.stem}_data.json')
    if args.plot_only:
        with open(data_file) as f:
            data = json.load(f)
    else:
        data = run_rollouts(args)
        with open(data_file, 'w') as f:
            json.dump(data, f, indent=1)
        print(f'[gantt v3] wrote {data_file}')
    sel = data['selection']
    print(f"[gantt v3] median penalty {sel['median_penalty_pct']:.2f}% over "
          f"{sel['n_orders']} orders (middle orders {sel['two_middle_orders']},"
          f" {sel['two_middle_penalties_pct']}); order {data['instance']}: "
          f"dispatched {data['instance_makespan_h']['dispatched']:.0f} h, "
          f"plan {data['instance_makespan_h']['plan']:.0f} h, repaired "
          f"{data['instance_makespan_h']['repaired']:.0f} h "
          f"(+{data['instance_penalty_pct']:.2f}%)")
    plot(data, os.path.join(args.out, args.stem))


if __name__ == '__main__':
    main()
