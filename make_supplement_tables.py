#!/usr/bin/env python3
"""make_supplement_tables.py -- the tables and the supplement-only numbers of
the Supplementary Material of the paper, computed from the
result files.

Writes to paper_tables/supplement/ (or $SUPP_OUT_DIR, or --out_dir):
  s1_size<N>.tex      per-size results at N modules (rules, metaheuristics,
                      CP-SAT modes and budgets, SLAB per seed and mean), with
                      the paired difference to SPT, its 95% bootstrap CI and
                      the shorter/equal/longer counts against SPT
  s1_long.tex         CP-SAT given 1800 s on the first 20 orders
  s1_ind.tex          independent test sets
  s1_mix.tex          size-mixed training pilot and its single-size match
  s2_stats.tex        Wilcoxon p, Holm-adjusted p, rank-biserial r
  s3_hgnn.tex         HGNN training options and checkpoint-selection rules
  s4_bound.tex        full waits, lag-augmented versus lag-free bound
  s4_aids.tex         full waits, with and without the input aids
  s5_byproducts.tex   flow time, work in progress, utilization, tardiness
  s6_flowline.tex     flow-line routing versus flexible routing
  s7_sensitivity.tex  processing-time and wait perturbations
  s8_stability.tex    stability-aware residual CP-SAT frontier
  s10_routes.tex      operations, stations, durations and waits per route
  s10_stations.tex    station counts of the three factories
  s11_hyper.tex       PPO and network hyperparameters
  s11_protocols.tex   disruption and uncertain-wait parameters
  supp_macros.tex     supplement-only scalars, one \\newcommand per line with
                      its source and computation; this includes the numbers
                      of the S4 validation curves and of the S9 Gantt chart,
                      read from the data files that make_fig_bound_curves.py
                      and make_gantt_v3.py write next to their figures
Each table file is one tabular environment; supplement.tex holds the float,
the caption and the label.

Conventions are those of make_paper_macros.py (imported): its loaders, its
evidence assertions, its seed handling and its rounding rule. A cell whose
number also appears in the main paper is written as the paper's macro
(macros_results.tex), after the value computed here is asserted equal to the
value recorded in macros_results.json. A macro that is pending there makes
the cell, and every cell derived from the same source, \\pend{...}. A
missing source gives \\pend{...}; files in no_evidence_backup/ are never
read. Paired differences carry a 95% percentile bootstrap CI with 2000
resamples and numpy default_rng(0) (stats_tests.paired_mean_diff_ci).

Every source is only read.

usage: python make_supplement_tables.py [--strict] [--out_dir DIR]
"""
import argparse
import ast
import glob
import json
import os
import re
import subprocess
import sys
from collections import OrderedDict

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
import make_paper_macros as M  # noqa: E402
from make_paper_macros import Missing, check, dec, fmt, fmt_p  # noqa: E402
from stats_tests import paired_mean_diff_ci, wilcoxon_paired  # noqa: E402

PAPER_INPUTS = os.path.join(REPO, 'paper_inputs')
OUT_DIR = os.environ.get('SUPP_OUT_DIR', os.path.join(M.OUT_DIR, 'supplement'))
MAC_JSON = os.path.join(M.OUT_DIR, 'macros_results.json')
PAPER_MACRO_FILES = ('macros.tex', 'macros_pending.tex', 'macros_results.tex')
FIG_SRC = os.environ.get('FIG_SRC_DIR', PAPER_INPUTS)
GANTT_JSON = os.path.join(FIG_SRC, 'fig_gantt_v3_data.json')
CURVES_JSON = os.path.join(FIG_SRC, 'fig_bound_curves_data.json')
N_BOOT, BOOT_SEED = 2000, 0
CP_EARLY_S = 60          # incumbent time reported for every CP-SAT run
DS10 = '10x25+ppvc-mixed'
SIZE_NUM = {'Ten': 10, 'Twenty': 20, 'Thirty': 30, 'Fifty': 50, 'Eighty': 80,
            'Tight': 10}
RULES = ('FIFO', 'MOR', 'SPT', 'MWKR', 'SPTL', 'LLF', 'MWKRL', 'LRPTL')

MAC = {}                 # macros_results.json
PENDING = OrderedDict()  # pend label -> reason
WARN = []
SUPP = OrderedDict()     # supplement macro -> (text, source, computation)
SOURCES = []             # sources of the table being built
CPU_GPU = OrderedDict()  # dataset -> (orders that differ, orders, mean diff)
SEED_HI = []             # upper CI ends of SLAB seed minus SPT, all sizes
BP_BLOCKS = []           # by-product values per size block (S5 checks)


# --------------------------------------------------------------------------- #
# Cells
# --------------------------------------------------------------------------- #
class Cell:
    """One table cell: LaTeX text, the number it shows (None if pending or
    not a number) and the displayed rounding used for ties."""

    def __init__(self, text, value=None, pending=False, key=None):
        self.text, self.value, self.pending, self.key = text, value, pending, key

    def __str__(self):
        return self.text


DASH = '--'


def stext(s):
    """Decimal text with a typographic minus sign."""
    return '$-$' + s[1:] if s.startswith('-') else s


def pend(label, reason):
    PENDING.setdefault(label, reason)
    return Cell('\\pend{%s}' % label, None, True)


def num(fn, kind, label):
    """A number computed here (no counterpart in the main paper)."""
    try:
        v = float(fn())
    except Missing as err:
        return pend(label, str(err))
    f = fmt(kind, v)
    return Cell(stext(f), v, key=f)


def mac(name, fn, kind):
    """A number of the main paper, written as its macro after the value
    computed here is asserted equal to the macro's recorded value."""
    e = MAC.get(name)
    check(e is not None, f'{name}: not defined in macros_results.json')
    try:
        v, why = float(fn()), None
    except Missing as err:
        v, why = None, str(err)
    if e['pending']:
        if v is not None:
            WARN.append(f'{name}: the source is present but the macro is '
                        f'pending; re-run make_paper_macros.py')
        PENDING.setdefault(name, e['reason'])
        return Cell('\\' + name, None, True)
    check(v is not None, f'{name}: the macro has a value but its source is '
                         f'missing ({why})')
    f = fmt(kind, v)
    check(f == e['value'] and abs(v - e['raw']) < 1e-9,
          f'{name}: computed {f} ({v}) differs from macros_results.json '
          f'{e["value"]} ({e["raw"]})')
    t = '\\' + name
    if f.startswith('-'):
        t = '$' + t + '$'
    return Cell(t, v, key=f)


def mac_text(name):
    """The macro of an already checked cell, for a derived cell."""
    return '\\' + name


def ci_text(lo, hi):
    return '[%s, %s]' % (stext(dec(lo, 1)), stext(dec(hi, 1)))


def wtl(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    check(a.shape == b.shape, f'pairing shapes {a.shape} {b.shape}')
    return '%d/%d/%d' % (int(np.sum(a < b)), int(np.sum(a == b)),
                         int(np.sum(a > b)))


def bold_best(cells, maximize=False):
    """Bold every cell whose displayed value is the best of the list."""
    present = [c for c in cells if isinstance(c, Cell) and c.value is not None]
    if not present:
        return
    keys = [float(c.key) for c in present]
    best = max(keys) if maximize else min(keys)
    for c, k in zip(present, keys):
        if k == best:
            c.text = '\\textbf{%s}' % c.text


def src(path_rel):
    SOURCES.append(path_rel)
    return path_rel


def supp(name, value_text, source, comp):
    check(re.match(r'^Su[A-Za-z]+$', name) is not None, f'bad name {name}')
    check(name not in SUPP, f'supplement macro defined twice: {name}')
    SUPP[name] = (value_text, source, comp)


def supp_pend(name, source, why):
    PENDING.setdefault(name, why)
    SUPP[name] = ('\\pend{%s}' % name, source, 'PENDING: ' + why)


# --------------------------------------------------------------------------- #
# Writer
# --------------------------------------------------------------------------- #
def write_table(out_dir, fname, colspec, head, body, ncols):
    lines = [f'% {fname}: generated by make_supplement_tables.py from the '
             'result files; do not edit by hand.']
    for s in sorted(set(SOURCES)):
        lines.append(f'% source: {s}')
    lines += [f'\\begin{{tabular}}{{{colspec}}}', '\\toprule'] + head
    lines.append('\\midrule')
    for r in body:
        if r == 'MID':
            lines.append('\\midrule')
        elif isinstance(r, tuple) and r[0] == 'BLOCK':
            lines.append('\\multicolumn{%d}{@{}l}{\\emph{%s}} \\\\'
                         % (ncols, r[1]))
        else:
            check(len(r) == ncols, f'{fname}: row of {len(r)} cells, '
                                   f'expected {ncols}: {[str(x) for x in r]}')
            lines.append(' & '.join(str(x) for x in r) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    with open(os.path.join(out_dir, fname), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    del SOURCES[:]
    return fname


# --------------------------------------------------------------------------- #
# Readers that extend make_paper_macros
# --------------------------------------------------------------------------- #
def rule_values(r, ds):
    """Per-instance makespans of rule r. At 10x25 the FIFO, MOR and MWKR runs
    were stored under a model tag; such a file is asserted equal, instance by
    instance, to the independent re-run in byproducts_<rule>.json."""
    tag = f'{r}+pdr'
    if os.path.exists(os.path.join(REPO, M.res_rel(tag, ds))):
        src(M.res_rel(tag, ds))
        return M.result(tag, ds)
    if r == 'SPT':
        src(M.rule_src('SPT', ds))
        return M.rule_col('SPT', ds)
    tag = f'{r}+{ds}+smoke'
    col = M.result(tag, ds)
    bp = os.path.join(M.TR, ds, f'byproducts_{r}.json')
    d = M.read_json(bp)
    check(d['dataset'] == ds and d['method'] == r and d['seed_test'] == 50
          and d['validated'] is True, f'{M.rel(bp)}: header')
    ms = np.array([x['makespan'] for x in d['per_instance']], dtype=float)
    check(np.array_equal(ms, col), f'{M.res_rel(tag, ds)} differs from '
                                   f'{M.rel(bp)}')
    src(M.res_rel(tag, ds))
    src(M.rel(bp))
    return col


def boot(a, b):
    d, (lo, hi) = paired_mean_diff_ci(np.asarray(a, dtype=float),
                                      np.asarray(b, dtype=float),
                                      n_boot=N_BOOT, seed=BOOT_SEED)
    return d, lo, hi


def at_time(rec, T):
    """Best objective found within T seconds of one anytime CP-SAT run."""
    traj = rec.get('trajectory')
    check(bool(traj), f'{rec["instance"]}: no trajectory')
    check(abs(float(traj[-1][1]) - float(rec['objective'])) < 1e-9,
          f'{rec["instance"]}: trajectory ends off the objective')
    v = [float(p[1]) for p in traj if float(p[0]) <= T]
    check(len(v) > 0, f'{rec["instance"]}: no schedule within {T} s')
    return min(v)


def cp_recs(ds, S, mode):
    if mode == 'warm-policy':
        src(M.cp_src(ds, S, 'warm-policy'))
        return M.anytime(ds, 'warm-policy-halflag-s301', 600, 'warm-policy',
                         model=M.HSTAR_MODEL)
    src(M.cp_src(ds, S, mode))
    return M.anytime(ds, mode, M.cp_budget(S), mode)


def py_constants(path, names):
    """Module-level constants of a source file, read without importing it."""
    with open(path) as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in names:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    check(set(out) == set(names), f'{M.rel(path)}: constants {set(names) - set(out)} not found')
    return out


def paper_macro_paths(files):
    """The manuscript macro files present in M.OUT_DIR, then the design
    constants shipped in paper_inputs/design_constants.tex."""
    out = [os.path.join(M.OUT_DIR, fn) for fn in files
           if os.path.exists(os.path.join(M.OUT_DIR, fn))]
    return out + [os.path.join(PAPER_INPUTS, 'design_constants.tex')]


def paper_constants():
    vals = {}
    for p in paper_macro_paths(PAPER_MACRO_FILES[:2]):
        with open(p) as f:
            for k, v in re.findall(r'\\newcommand\{\\([A-Za-z]+)\}\{([^{}]*)\}',
                                   f.read()):
                vals.setdefault(k, v)
    return vals


# --------------------------------------------------------------------------- #
# S1 per-size results
# --------------------------------------------------------------------------- #
S1_HEAD = ['Method & Mean (h) & \\makecell{Difference\\\\to SPT (h)} & '
           '95\\% CI (h) & \\makecell{Shorter/equal/\\\\longer than SPT} \\\\']


def s1_row(label, mean_cell, vals_fn, spt, label_key):
    if mean_cell.pending:
        p = mean_cell.text if mean_cell.text.startswith('\\pend') \
            else '\\pend{%s}' % label_key
        return [label, mean_cell, p, p, p]
    a = vals_fn()
    d, lo, hi = boot(a, spt)
    if label.startswith('Seed '):
        SEED_HI.append(hi)
    return [label, mean_cell, stext(dec(d, 1)), ci_text(lo, hi), wtl(a, spt)]


def s1_size(S, ds, out_dir):
    spt = rule_values('SPT', ds)
    rows, means = [('BLOCK', 'Dispatching rules')], []
    for r in RULES:
        vals = rule_values(r, ds)
        if r == 'SPT':
            c = mac(f'Spt{S}', lambda v=vals: v.mean(), 'h')
            rows.append(['SPT', c, DASH, DASH, DASH])
        else:
            c = (mac(f'Llf{S}', lambda v=vals: v.mean(), 'h') if r == 'LLF'
                 else num(lambda v=vals: v.mean(), 'h', f'{r}{S}'))
            rows.append(s1_row(r, c, lambda v=vals: v, spt, f'{r}{S}'))
        means.append(c)

    rows.append(('BLOCK', 'Metaheuristics, one core'))
    if S == 'Ten':
        src(M.ma_rel(ds, 60))
        c = num(lambda: M.memetic(ds, 60).mean(), 'h', 'MaSixtyTen')
        rows.append(s1_row('Memetic, 60\\,s', c, lambda: M.memetic(ds, 60),
                           spt, 'MaSixtyTen'))
        means.append(c)
    src(M.ma_rel(ds))
    c = mac(f'Ma{S}', lambda: M.memetic(ds).mean(), 'h')
    rows.append(s1_row('Memetic, 300\\,s', c, lambda: M.memetic(ds), spt,
                       f'Ma{S}'))
    means.append(c)
    if S == 'Ten':
        for name, b in (('GaSixtyTen', 60), ('GaThreeHundredTen', 300)):
            src(M.ga_rel(ds, b))
            c = mac(name, lambda b=b: M.ga(ds, b).mean(), 'h')
            rows.append(s1_row(f'GA, {b}\\,s', c, lambda b=b: M.ga(ds, b),
                               spt, name))
            means.append(c)

    rows.append(('BLOCK', 'CP-SAT, four workers'))
    tb = M.cp_budget(S)
    times = (CP_EARLY_S, 300) + ((600,) if tb == 600 else ())
    modes = [('cold', 'Cold start', 'CpCold'),
             ('warm-spt', 'From SPT', 'CpWarmSpt')]
    if S in ('Fifty', 'Eighty'):
        modes.append(('warm-policy', 'From SLAB', 'CpWarmPol'))
    for mode, lab, stem in modes:
        recs = cp_recs(ds, S, mode)
        for T in times:
            label = f'{lab}, {T}\\,s'
            if T == tb:
                vals = (lambda m=mode: M.cp_col(ds, S, m))
                c = mac(f'{stem}{S}', lambda v=vals: v().mean(), 'h')
            else:
                vals = (lambda r=recs, T=T: np.array([at_time(x, T)
                                                       for x in r]))
                c = num(lambda v=vals: v().mean(), 'h', f'{stem}{T}{S}')
            rows.append(s1_row(label, c, vals, spt, f'{stem}{S}'))
            means.append(c)

    rows.append(('BLOCK', 'SLAB policy'))
    for s in M.SEEDS:
        tag = M.HSTAR.format(s=s)
        src(M.res_rel(tag, ds))
        c = num(lambda t=tag: M.result(t, ds).mean(), 'h', f'SlabSeed{S}')
        rows.append(s1_row(f'Seed {s}', c, lambda t=tag: M.result(t, ds),
                           spt, f'SlabSeed{S}'))   # seeds are not bolded
    c = mac(f'Hstar{S}', lambda: M.mean_of(M.HSTAR, ds), 'h')
    inst = M.inst_of(M.HSTAR, ds)
    e = MAC[f'HstarWinsSpt{S}']
    if not e['pending']:
        check(int(np.sum(inst < spt)) == int(e['raw']),
              f'HstarWinsSpt{S}: wins differ')
    rows.append(s1_row('Mean of three seeds', c, lambda: inst, spt,
                       f'Hstar{S}'))
    means.append(c)
    bold_best(means)
    n = SIZE_NUM[S]
    return write_table(out_dir, f's1_size{n}.tex', '@{}lcccc@{}', S1_HEAD,
                       rows, 5)


def s1_long(out_dir):
    head = ['Method & 50 modules & 80 modules \\\\']
    cols = {}
    for S, ds in (('Fifty', '50x25+ppvc-mixed'),
                  ('Eighty', '80x25+ppvc-mixed')):
        n20 = int(MAC['CpLongN']['raw'])
        spt = rule_values('SPT', ds)[:n20]
        c = []
        c.append(num(lambda: spt.mean(), 'h', f'SptFirst{S}'))
        for mode, stem in (('cold', 'Cold'), ('warm-spt', 'WarmSpt')):
            src(M.cp_src(ds, S, mode))
            c.append(num(lambda m=mode: M.cp_col(ds, S, m)[:n20].mean(), 'h',
                         f'Cp{stem}SixHundredFirst{S}'))
            src(M.any_rel(ds, mode, 1800))
            if mode == 'warm-spt':
                c.append(mac(f'CpLongWarm{S}', lambda: np.mean(
                    [r['objective'] for r in M.anytime(
                        ds, 'warm-spt', 1800, 'warm-spt', n=n20)]), 'h'))
            else:
                c.append(num(lambda: np.mean([r['objective'] for r in M.anytime(
                    ds, 'cold', 1800, 'cold', n=n20)]), 'h', f'CpColdLong{S}'))
        src(M.cp_src(ds, S, 'warm-policy'))
        c.append(num(lambda: M.cp_col(ds, S, 'warm-policy')[:n20].mean(), 'h',
                     f'CpWarmPolFirst{S}'))
        src(M.src(M.HSTAR, ds))
        c.append(mac(f'LongSlab{S}',
                     lambda: float(np.mean(M.inst_of(M.HSTAR, ds)[:n20])), 'h'))
        # Section S1 states that 1800 s of CP-SAT stays longer than SLAB
        check(c[2].value > c[-1].value and c[4].value > c[-1].value,
              f'{S}: CP-SAT at 1800 s is not longer than SLAB')
        bold_best(c)
        cols[S] = c
    labels = ['SPT rule', 'CP-SAT, cold start, 600\\,s',
              'CP-SAT, cold start, 1800\\,s', 'CP-SAT, from SPT, 600\\,s',
              'CP-SAT, from SPT, 1800\\,s', 'CP-SAT, from SLAB, 600\\,s',
              'SLAB, mean of three seeds']
    rows = [[lab, cols['Fifty'][i], cols['Eighty'][i]]
            for i, lab in enumerate(labels)]
    return write_table(out_dir, 's1_long.tex', '@{}lcc@{}', head, rows, 3)


def s1_ind(out_dir):
    head = ['Method & 10 & 20 & 50 & 80 \\\\']
    names = ['FIFO', 'MOR', 'SPT', 'MWKR', 'SPTL', 'LLF', 'MWKRL', 'LRPTL',
             'Policy trained without waits', 'SLAB, seed 301',
             'SLAB, seed 302', 'SLAB, seed 303', 'SLAB, mean of three seeds']
    grid = {k: [] for k in names}
    diff, cir, wl = [], [], []
    for S, ds in M.SIZES_IND:
        spt = rule_values('SPT', ds)
        col = []
        for r in RULES:
            v = rule_values(r, ds)
            c = (mac(f'SptInd{S}', lambda v=v: v.mean(), 'h') if r == 'SPT'
                 else num(lambda v=v: v.mean(), 'h', f'{r}Ind{S}'))
            grid[r].append(c)
            col.append(c)
        src(M.src(M.LAGFREE, ds))
        c = mac(f'LagFreeInd{S}', lambda: M.mean_of(M.LAGFREE, ds), 'h')
        grid['Policy trained without waits'].append(c)
        col.append(c)
        for s in M.SEEDS:
            tag = M.HSTAR.format(s=s)
            src(M.res_rel(tag, ds))
            c = num(lambda t=tag: M.result(t, ds).mean(), 'h', f'SlabInd{S}')
            grid[f'SLAB, seed {s}'].append(c)   # seeds are not bolded
        c = mac(f'HstarInd{S}', lambda: M.mean_of(M.HSTAR, ds), 'h')
        grid['SLAB, mean of three seeds'].append(c)
        col.append(c)
        bold_best(col)
        inst = M.inst_of(M.HSTAR, ds)
        d, lo, hi = boot(inst, spt)
        diff.append(stext(dec(d, 1)))
        cir.append(ci_text(lo, hi))
        wl.append(wtl(inst, spt))
        if S in ('Fifty', 'Eighty'):
            check(bool(np.all(inst < spt)), f'Section S1 states that SLAB is '
                  f'shorter than SPT on every independent order at {S}')
        e = MAC[f'HstarIndWinsSpt{S}']
        check(e['pending'] or int(np.sum(inst < spt)) == int(e['raw']),
              f'HstarIndWinsSpt{S}: wins differ')
    rows = [[k] + grid[k] for k in names]
    rows += ['MID', ['SLAB minus SPT (h)'] + diff, ['95\\% CI (h)'] + cir,
             ['SLAB shorter/equal/longer'] + wl]
    return write_table(out_dir, 's1_ind.tex', '@{}lcccc@{}', head, rows, 5)


MIX_MODEL = 'mix10-15-20x25+ppvc-mixed+full2k-s301'
MIX_SINGLE = '10x25+ppvc-mixed+full2k-s301'


def s1_mix(out_dir):
    """Size-mixed training pilot (one seed) and the single-size run that
    differs from it only in the training order sizes (asserted on the saved
    configurations), greedy, at every size of the default factory."""
    cfg = {}
    for m in (MIX_MODEL, MIX_SINGLE):
        p = f'train_log/PPVC/config_{m}.json'
        cfg[m] = M.read_json(os.path.join(REPO, p))
        src(p)
    a, b = cfg[MIX_MODEL], cfg[MIX_SINGLE]
    diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    check(diff == ['ppvc_size_mix'], f'size-mix pilot: configs differ in {diff}')
    mix = [int(x) for x in a['ppvc_size_mix'].split(',')]
    check(mix == [10, 15, 20] and b['ppvc_size_mix'] == '' and b['n_j'] == 10,
          f'size-mix pilot: training sizes {mix}')
    check(a['use_lag_features'] is True and a['use_type_embedding'] is True
          and float(a['lag_scale']) == 1.0 and a['bound_lag_free'] is False
          and int(a['seed_train']) == 301 and a['ppvc_factory'] == 'default',
          'size-mix pilot: full waits, both aids, lag-augmented bound, seed 301')
    head = [' & \\multicolumn{5}{c}{Modules per order} \\\\',
            '\\cmidrule(l){2-6}',
            'Method & 10 & 20 & 30 & 50 & 80 \\\\']
    sizes = ', '.join(str(x) for x in mix[:-1]) + f', and {mix[-1]}'
    labels = ('SPT rule', 'Trained on 10 modules', f'Trained on {sizes} modules')
    rows = [[lab] for lab in labels]
    vals = {}
    for S, ds in M.SIZES:
        col = [mac(f'Spt{S}', lambda d=ds: rule_values('SPT', d).mean(), 'h')]
        for m, key in ((MIX_SINGLE, 'Single'), (MIX_MODEL, 'Mix')):
            tag = 'greedy+' + m
            src(M.res_rel(tag, ds))
            col.append(num(lambda t=tag, d=ds: M.result(t, d).mean(), 'h',
                           f'{key}{S}'))
            vals[(key, S)] = col[-1].value
        bold_best(col)
        for r, c in zip(rows, col):
            r.append(c)
    # Section S1 states that mixing the sizes does not improve the 80-module
    # result of this run
    if vals[('Mix', 'Eighty')] is not None and vals[('Single', 'Eighty')] is not None:
        check(vals[('Mix', 'Eighty')] >= vals[('Single', 'Eighty')],
              'size-mix pilot is shorter than the single-size run at 80 modules')
    s = f'test_results/PPVC/80x25+ppvc-mixed/Result_greedy+<model>_80x25+ppvc-mixed.npy'
    for key, m in (('Mix', MIX_MODEL), ('Single', MIX_SINGLE)):
        v = vals[(key, 'Eighty')]
        if v is None:
            supp_pend(f'Su{key}Eighty', s, 'result file absent')
        else:
            supp(f'Su{key}Eighty', fmt('h', v), s.replace('<model>', m),
                 'mean makespan at 80 modules (h)')
    supp('SuMixSizes', sizes,
         f'train_log/PPVC/config_{MIX_MODEL}.json', 'ppvc_size_mix')
    return write_table(out_dir, 's1_mix.tex', '@{}lccccc@{}', head, rows, 6)


def s1_dose(out_dir):
    """Training-wait factor: the DANIEL network without aids trained with every
    training and validation wait multiplied by 1, 0.75, 0.5, 0.25 or 0, greedy
    at every size of the default factory and on the 15-station factory. The
    single-seed arms are asserted to differ from the SLAB run of seed 301 only
    in lag_scale."""
    ref = M.read_json(os.path.join(REPO, src(
        'train_log/PPVC/config_10x25+ppvc-mixed+a1lagx05-2k-s301.json')))
    for m, k in (('10x25+ppvc-mixed+a1lagx075-2k-s301', 0.75),
                 ('10x25+ppvc-mixed+a1lagx025-2k-s301', 0.25)):
        c = M.read_json(os.path.join(REPO, src(f'train_log/PPVC/config_{m}.json')))
        diff = sorted(x for x in set(c) | set(ref)
                      if c.get(x) != ref.get(x) and x != 'model_suffix')
        check(diff == ['lag_scale'] and float(c['lag_scale']) == k,
              f'dose arm {m}: configs differ in {diff}')
    arms = (('1 (full waits)', 'Three', 'TrainFull', M.FULL, 'ContTenLagFull'),
            ('0.75', 'One', 'TrainThreeQuarter', M.DOSE_TQ, 'ContTenLagThreeQuarter'),
            ('0.5 (SLAB)', 'Three', 'TrainHalf', M.HSTAR, 'ContTenLagHalf'),
            ('0.25', 'One', 'TrainQuarter', M.DOSE_Q, 'ContTenLagQuarter'),
            ('0 (no waits)', 'Three', 'TrainLagFree', M.LAGFREE, 'ContTenLagZero'))
    head = [' & & & \\multicolumn{5}{c}{Default factory, modules per order} & '
            '\\FactoryTight{} stations \\\\',
            '\\cmidrule(lr){4-8}\\cmidrule(l){9-9}',
            'Factor & Seeds & $\\sigma$ & 10 & 20 & 30 & 50 & 80 & 10 \\\\']
    rows = [[lab, seeds.lower(), '\\' + sig] for lab, seeds, _, _, sig in arms]
    for S, ds in M.SIZES + (M.TIGHT,):
        col = [mac(f'{stem}{S}', lambda t=tpl, d=ds: M.mean_of(t, d), 'h')
               for _, _, stem, tpl, _ in arms]
        bold_best(col)
        for r, c in zip(rows, col):
            r.append(c)
    return write_table(out_dir, 's1_dose.tex', '@{}lcccccccc@{}', head, rows, 9)


# --------------------------------------------------------------------------- #
# S2 statistics
# --------------------------------------------------------------------------- #
def s2_stats(out_dir):
    head = ['Modules & Orders & \\makecell{SLAB minus\\\\rival (h)} & 95\\% CI (h) & '
            '$p$ & \\makecell{Holm-adjusted\\\\$p$} & $r$ \\\\']
    rows = []
    titles = {'Spt': 'Rival: SPT rule',
              'Ma': 'Rival: memetic algorithm, 300\\,s',
              'CpWarm': 'Rival: CP-SAT from SPT, time limits of '
                        'Table~\\ref{M-tab:main} of the paper'}
    for R, _what in M.RIVALS:
        rows.append(('BLOCK', titles[R]))
        for k, (S, ds) in enumerate(M.SIZES):
            src(M.src(M.HSTAR, ds))
            src(M.rival_src(R, S, ds))
            n = mac(f'nInst{S}', lambda d=ds: len(M.result(
                M.HSTAR.format(s=301), d)), 'int')
            p = mac(f'Pval{R}{S}', lambda r=R, s=S, d=ds: M.pval(r, s, d), 'p')
            ph = mac(f'Pval{R}Holm{S}', lambda r=R, i=k: M.holm_family(r)[i],
                     'p')
            rb = mac(f'Rbis{R}{S}', lambda r=R, s=S, d=ds: M.rank_biserial(
                M.inst_of(M.HSTAR, d), M.rival_inst(r, s, d)), 'r')
            if p.pending:
                lab = '\\pend{%s%s}' % (R, S)
                rows.append([str(SIZE_NUM[S]), n, lab, lab, p, ph, rb])
                continue
            d, lo, hi = boot(M.inst_of(M.HSTAR, ds), M.rival_inst(R, S, ds))
            rows.append([str(SIZE_NUM[S]), n, stext(dec(d, 1)),
                         ci_text(lo, hi), p, ph, rb])
    # Section S2 states: the largest Holm-adjusted p against SPT is at 30
    # modules, r runs from the 10-module to the 80-module value, and the
    # difference to warm CP-SAT changes sign between 20 and 30 modules
    hs = M.holm_family('Spt')
    check(max(hs) == hs[2], 'largest Holm p against SPT is not at 30 modules')
    rs = [M.rank_biserial(M.inst_of(M.HSTAR, d), M.rival_inst('Spt', S, d))
          for S, d in M.SIZES]
    check(max(rs) == rs[0] and min(rs) == rs[-1], f'r against SPT {rs}')
    ds_ = [float(np.mean(M.inst_of(M.HSTAR, d) - M.rival_inst('CpWarm', S, d)))
           for S, d in M.SIZES]
    check(ds_[0] > 0 and ds_[1] > 0 and all(x < 0 for x in ds_[2:]),
          f'difference to warm CP-SAT {ds_}')
    return write_table(out_dir, 's2_stats.tex', '@{}rrccccc@{}', head, rows, 7)


# --------------------------------------------------------------------------- #
# S3 HGNN
# --------------------------------------------------------------------------- #
HGNN_SAVE = os.path.join(REPO, 'backbones', 'hgnn', 'save')
HGNN_ROWS = (  # training orders, selection, execution, run, result tag, macro
    ('Without waits', 'Published', 'SLAB dispatch', 'hgnn-a0-s300',
     M.HGNN_A0, 'HgnnLagFree'),
    ('Without waits', 'Published', 'Repaired plan', 'hgnn-a0-s300',
     M.HGNN_REPAIR, 'RepairHgnn'),
    ('Without waits', '10 and 20', 'SLAB dispatch', 'hgnn-a0-vs1020-s300',
     'greedy+hgnn-aware+hgnn-a0-vs1020-s300', None),
    ('Without waits', '10 and 20', 'Repaired plan', 'hgnn-a0-vs1020-s300',
     'greedy+hgnn-blind+hgnn-a0-vs1020-s300', None),
    ('Full waits', 'Published', 'SLAB dispatch', 'hgnn-a1-s300',
     M.HGNN_A1, 'HgnnFull'),
    ('Full waits', '10 and 20', 'SLAB dispatch', 'hgnn-a1-vs1020-s300',
     M.HGNN_A1VS, 'HgnnFullVs'),
    ('Halved waits', '10 and 20', 'SLAB dispatch',
     'hgnn-a1lagx05-vs1020-s300', M.HGNN_HALF, 'HgnnHalf'))


def hgnn_run(run):
    cfg_p = os.path.join(HGNN_SAVE, run, 'run_config.json')
    log_p = os.path.join(HGNN_SAVE, run, 'train_log.json')
    cfg, log = M.read_json(cfg_p), M.read_json(log_p)
    src(M.rel(cfg_p))
    src(M.rel(log_p))
    a = cfg['args']
    blind = '-a0-' in run
    check(a['lag_mode'] == ('blind' if blind else 'aware'), f'{run}: lag_mode')
    check(a['bound_pt'] == ('mean' if blind else 'min'), f'{run}: bound_pt')
    check(a.get('lag_scale') == (0.5 if 'lagx05' in run else None),
          f'{run}: lag_scale')
    check(a.get('vali_sizes') == ('10,20' if 'vs1020' in run else None),
          f'{run}: vali_sizes')
    check(a['max_iterations'] == 1000 and a['seed'] == 300
          and a['n_modules'] == 10 and a['factory'] == 'default'
          and a['vali_size'] == 100 and a['save_timestep'] == 10
          and a['lag_features'] is False, f'{run}: recipe')
    vm = np.asarray(log['vali_makespan'], dtype=float)
    it = np.asarray(log['vali_iteration'])
    check(len(vm) == len(it) == 100, f'{run}: {len(vm)} validations')
    return int(it[int(np.argmin(vm))]), a


def s3_hgnn(out_dir):
    head = ['Training orders & Selection & Execution & Kept & 10 & 20 & 30 & '
            '50 & 80 \\\\']
    rows, cols = [], {S: [] for S, _ in M.SIZES}
    for orders, sel, exe, run, tag, stem in HGNN_ROWS:
        kept, _a = hgnn_run(run)
        row = [orders, sel, exe, str(kept)]
        for S, ds in M.SIZES:
            src(M.src(tag, ds))
            if stem is None:
                c = num(lambda t=tag, d=ds: M.mean_of(t, d), 'h', f'Hgnn{S}')
            else:
                c = mac(f'{stem}{S}', lambda t=tag, d=ds: M.mean_of(t, d), 'h')
            row.append(c)
            cols[S].append(c)
        rows.append(row)
    # Section S3 states: every run dispatched in the SLAB transition is
    # shorter than SPT at 10 modules and longer at every larger size
    for orders, sel, exe, run, tag, stem in HGNN_ROWS:
        if exe != 'SLAB dispatch':
            continue
        for S, ds in M.SIZES:
            v, sp = M.mean_of(tag, ds), rule_values('SPT', ds).mean()
            check(v < sp if S == 'Ten' else v > sp,
                  f'HGNN {run} at {S}: {v:.1f} against SPT {sp:.1f}')
    rows.append('MID')
    row = ['SPT rule', DASH, DASH, DASH]
    for S, ds in M.SIZES:
        c = mac(f'Spt{S}', lambda d=ds: rule_values('SPT', d).mean(), 'h')
        row.append(c)
        cols[S].append(c)
    rows.append(row)
    for S, _ in M.SIZES:
        bold_best(cols[S])
    # selection normalizers of the two-size rule
    with open(os.path.join(REPO, 'backbones', 'hgnn', 'train_ppvc.py')) as f:
        m = re.search(r'SPT_SCALE = \{10: ([0-9.]+), 20: ([0-9.]+)', f.read())
    check(m is not None, 'train_ppvc.py: SPT_SCALE not found')
    supp('SuHgnnNormTen', dec(float(m.group(1)), 0),
         'backbones/hgnn/train_ppvc.py', 'SPT_SCALE[10]')
    supp('SuHgnnNormTwenty', dec(float(m.group(2)), 0),
         'backbones/hgnn/train_ppvc.py', 'SPT_SCALE[20]')
    return write_table(out_dir, 's3_hgnn.tex', '@{}lllrccccc@{}', head, rows, 9)


# --------------------------------------------------------------------------- #
# S4 bound and aids
# --------------------------------------------------------------------------- #
S4_COLS = M.SIZES + (M.TIGHT,)
S4_HEAD = [' & \\multicolumn{5}{c}{Default factory, modules per order} & '
           '\\FactoryTight{} stations \\\\',
           '\\cmidrule(lr){2-6}\\cmidrule(l){7-7}',
           ' & 10 & 20 & 30 & 50 & 80 & 10 \\\\']


def seed_block(title, tpl, mean_macro, short):
    rows = [('BLOCK', title)]
    for s in M.SEEDS:
        row = [f'Seed {s}']
        for S, ds in S4_COLS:
            t = tpl.format(s=s)
            src(M.res_rel(t, ds))
            row.append(num(lambda t=t, d=ds: M.result(t, d).mean(), 'h',
                           f'{short}{S}'))
        rows.append(row)
    row, means = ['Mean of three seeds'], []
    for S, ds in S4_COLS:
        if mean_macro:
            c = mac(f'{mean_macro}{S}', lambda d=ds: M.mean_of(tpl, d), 'h')
        else:
            c = num(lambda d=ds: M.mean_of(tpl, d), 'h', f'{short}{S}')
        row.append(c)
        means.append(c)
    rows.append(row)
    return rows, means


def s4_bound(out_dir):
    ra, ma = seed_block('Lag-augmented bound', M.FULL, 'TrainFull', 'Full')
    rf, mf = seed_block('Lag-free bound', M.LFBOUND, None, 'LfBound')
    for a, b in zip(ma, mf):
        if not (a.pending or b.pending):
            bold_best([a, b])
    dif, ci = (['\\makecell[l]{Lag-free minus\\\\lag-augmented (h)}'],
                ['95\\% CI (h)'])
    for S, ds in S4_COLS:
        def bt(d=ds):
            return boot(M.inst_of(M.LFBOUND, d), M.inst_of(M.FULL, d))
        c = mac(f'BoundDiff{S}', lambda b=bt: b()[0], 'h')
        lo = mac(f'BoundCiLo{S}', lambda b=bt: b()[1], 'h')
        hi = mac(f'BoundCiHi{S}', lambda b=bt: b()[2], 'h')
        dif.append('$\\BoundDiff%s$' % S if not c.pending else c.text)
        ci.append('[$\\BoundCiLo%s$, $\\BoundCiHi%s$]' % (S, S)
                  if not lo.pending else lo.text)
        if not lo.pending:
            # Section S4 states: zero inside the interval at 10, 50 and 80
            # modules, interval below zero at 20 and 30 modules
            if S in ('Ten', 'Fifty', 'Eighty'):
                check(lo.value <= 0 <= hi.value, f'bound CI at {S} excludes 0')
            elif S in ('Twenty', 'Thirty'):
                check(hi.value < 0, f'bound CI at {S} reaches 0')
    rows = ra + rf + ['MID', dif, ci]
    return write_table(out_dir, 's4_bound.tex', '@{}lcccccc@{}', S4_HEAD,
                       rows, 7)


def s4_aids(out_dir):
    ra, ma = seed_block('Without the aids', M.FULL, 'TrainFull', 'Full')
    rf, mf = seed_block('With both aids', M.FULLAIDS, 'TrainFullAids',
                        'FullAids')
    for a, b in zip(ma, mf):
        if not (a.pending or b.pending):
            bold_best([a, b])
    dif, ci = (['\\makecell[l]{With minus\\\\without aids (h)}'],
                ['95\\% CI (h)'])
    for S, ds in S4_COLS:
        try:
            d, lo, hi = boot(M.inst_of(M.FULLAIDS, ds), M.inst_of(M.FULL, ds))
            dif.append(stext(dec(d, 1)))
            ci.append(ci_text(lo, hi))
            # Section S4 states this sign pattern of the aids comparison
            if S in ('Ten', 'Thirty'):
                check(lo <= 0 <= hi, f'aids CI at {S} excludes 0')
            elif S in ('Twenty', 'Fifty', 'Eighty'):
                check(lo > 0, f'aids CI at {S} reaches 0')
            else:
                check(hi < 0, f'aids CI at {S} reaches 0')
        except Missing as err:
            dif.append(pend(f'AidsDiff{S}', str(err)).text)
            ci.append(pend(f'AidsDiff{S}', str(err)).text)
    rows = ra + rf + ['MID', dif, ci]
    return write_table(out_dir, 's4_aids.tex', '@{}lcccccc@{}', S4_HEAD,
                       rows, 7)


CURVE_TAIL = 50     # validations averaged for the final level (every 10 updates)
CURVE_BAND = 1.0    # band around the final level (%)
CURVE_ARMS = (('lag_augmented', 'Aug', M.FULL), ('lag_free', 'Free', M.LFBOUND))


def settle_update(upd, mean, tail=CURVE_TAIL, band=CURVE_BAND):
    """First update from which the curve stays within band % of its mean
    over the last tail validations."""
    mean = np.asarray(mean, dtype=float)
    fin = float(mean[-tail:].mean())
    inside = np.abs(mean - fin) <= band / 100.0 * fin
    outside = np.where(~inside)[0]
    i = 0 if len(outside) == 0 else int(outside[-1]) + 1
    check(i < len(mean), 'curve ends outside its band')
    return int(upd[i])


def s4_curves():
    """Numbers of the S4 validation-curve figure, read from the data file of
    make_fig_bound_curves.py. Each curve is checked against its log in
    train_log/PPVC where that log is present, and each checkpoint against the
    file in trained_network/PPVC that the evaluation files come from."""
    s = M.rel(CURVES_JSON)
    try:
        d = M.read_json(CURVES_JSON)
    except Missing as err:
        for k in ('ValiN', 'Tail', 'Band', 'SettleAug', 'SettleFree',
                  'TailAugTwenty', 'TailFreeTwenty', 'SpreadAug', 'SpreadFree'):
            supp_pend(f'SuCurve{k}', s, str(err))
        return
    upd = np.asarray(d['updates'])
    check(list(upd) == list(range(10, 2001, 10)) and d['vali_sizes'] == [10, 20],
          f'{s}: updates or validation sizes')
    out, n_vali = {}, set()
    for key, short, tpl in CURVE_ARMS:
        runs = d['arms'][key]['runs']
        check([r['seed'] for r in runs] == list(M.SEEDS), f'{s}: {key} seeds')
        check(d['arms'][key]['bound_lag_free'] is (key == 'lag_free'),
              f'{s}: {key} bound')
        for r in runs:
            model = tpl.format(s=r['seed'])[len('greedy+'):]
            check(r['model'] == model, f'{s}: {key} model {r["model"]}')
            ck = os.path.join(REPO, 'trained_network', 'PPVC', f'{model}.pth')
            check(file_sha256(ck) == r['checkpoint_sha256'],
                  f'{s}: checkpoint of {model} differs')
            c = M.read_json(os.path.join(REPO, 'train_log', 'PPVC',
                                         f'config_{model}.json'))
            check(float(c['lag_scale']) == 1.0
                  and c['vali_sizes_resolved'] == d['vali_sizes'],
                  f'config_{model}.json: validation at the full waits')
            n_vali.add(int(c['vali_size']))
            log = os.path.join(REPO, 'train_log', 'PPVC', f'valicurve_{model}.json')
            if os.path.exists(log):
                v = M.read_json(log)
                check([e[3]['mean_makespan'][0] for e in v] == r['makespan_10']
                      and [e[3]['mean_makespan'][1] for e in v] == r['makespan_20']
                      and [float(e[1]) for e in v] == r['score'],
                      f'{s}: curve of {model} differs from its log')
            else:
                WARN.append(f'{M.rel(log)} absent; the curve of {model} is read '
                            f'from {s} only')
        m10 = np.array([r['makespan_10'] for r in runs])
        m20 = np.array([r['makespan_20'] for r in runs])
        tail20 = m20[:, -CURVE_TAIL:].mean(axis=1)
        out[short] = dict(settle=settle_update(upd, m10.mean(axis=0)),
                          tail20=float(m20[:, -CURVE_TAIL:].mean()),
                          spread20=float(np.ptp(tail20)))
    a, f = out['Aug'], out['Free']
    # Section S4 states: at 10 modules the lag-augmented bound settles first;
    # at 20 modules its runs end longer and further apart
    check(a['settle'] < f['settle'], f'S4 curves: settle {a} {f}')
    check(a['tail20'] > f['tail20'] and a['spread20'] > f['spread20'],
          f'S4 curves: 20-module tail {a} {f}')
    check(len(n_vali) == 1, f'S4 curves: validation set sizes {n_vali}')
    supp('SuCurveValiN', str(n_vali.pop()),
         'train_log/PPVC/config_<model>.json', 'vali_size of the six runs')
    supp('SuCurveTail', str(CURVE_TAIL * 10), 'make_supplement_tables.py',
         'updates averaged for the final level (CURVE_TAIL validations x 10)')
    supp('SuCurveBand', dec(CURVE_BAND, 0), 'make_supplement_tables.py',
         'band around the final level (%)')
    for short in ('Aug', 'Free'):
        supp(f'SuCurveSettle{short}', str(out[short]['settle']), s,
             'first update from which the three-seed mean at 10 modules stays '
             'within the band of its final level')
        supp(f'SuCurveTail{short}Twenty', fmt('h', out[short]['tail20']), s,
             'mean over seeds and the last validations at 20 modules (h)')
        supp(f'SuCurveSpread{short}', fmt('h', out[short]['spread20']), s,
             'max minus min over seeds of the per-seed tail mean at 20 modules (h)')


def file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# S5 by-product objectives
# --------------------------------------------------------------------------- #
BP_COLS = (('makespan', 'h', False), ('flow_mean', 'h', False),
           ('wip_mean', 'h', False), ('util_mean', 'util', True),
           ('tard_total', 'h', False), ('n_tardy', 'pct2', False))


def byproducts(ds, method):
    rel_p = f'test_results/PPVC/{ds}/byproducts_hstar/byproducts_{method}.json'
    d = M.read_json(os.path.join(REPO, rel_p))
    src(rel_p)
    n = M.N_INST[ds]
    check(d['dataset'] == ds and d['method'] == method and d['seed_test'] == 50
          and d['validated'] is True and d['n_instances'] == n
          and len(d['per_instance']) == n, f'{rel_p}: header')
    check(float(d['due_jitter']) == 0.0, f'{rel_p}: due jitter')
    for i, r in enumerate(d['per_instance']):
        check(r['instance'] == f'instance_{i:03d}', f'{rel_p}: order at {i}')
    ms = np.array([r['makespan'] for r in d['per_instance']], dtype=float)
    if method == 'greedy':
        # The by-product runs decode the policy on the CPU, the evaluation
        # files on the GPU; floating-point differences change some greedy
        # choices. The model is asserted by name, and the two runs must agree
        # in mean within 1 h and on at least a quarter of the orders.
        check(d['model'] == M.HSTAR_MODEL, f'{rel_p}: model {d["model"]}')
        ref = M.result(M.HSTAR1, ds)
        src(M.res_rel(M.HSTAR1, ds))
        nd = int(np.sum(ms != ref))
        diff = float(ms.mean() - ref.mean())
        check(abs(diff) < 1.0 and nd <= 0.75 * n,
              f'{rel_p}: CPU run differs from the evaluation file on {nd} '
              f'of {n} orders, mean difference {diff:.2f} h')
        CPU_GPU[ds] = (nd, n, diff)
    else:
        ref = rule_values(method, ds)
        check(np.array_equal(ms, ref), f'{rel_p}: makespans differ from the '
                                       f'evaluation file')
    return d


def s5_byproducts(out_dir):
    head = ['Method & \\makecell{Makespan\\\\(h)} & \\makecell{Flow\\\\time (h)} & '
            '\\makecell{Work in progress\\\\(modules)} & '
            '\\makecell{Utilization\\\\(\\%)} & \\makecell{Total\\\\tardiness (h)} & '
            '\\makecell{Tardy\\\\modules} \\\\']
    rows, due = [], set()
    del BP_BLOCKS[:]
    for S, ds in M.SIZES:
        rows.append(('BLOCK', f'{SIZE_NUM[S]} modules, {M.N_INST[ds]} orders'))
        block = []
        for method, label in (('greedy', 'SLAB, seed 301'), ('SPT', 'SPT'),
                              ('LLF', 'LLF')):
            try:
                d = byproducts(ds, method)
            except Missing as err:
                block.append([label] + [pend(f'Byp{S}', str(err))
                                        for _ in BP_COLS])
                continue
            due.add(float(d['due_factor']))
            row = [label]
            for key, kind, _mx in BP_COLS:
                v = np.mean([float(r[key]) for r in d['per_instance']])
                if kind == 'util':
                    f = dec(100.0 * v, 1)
                    row.append(Cell(f, 100.0 * v, key=f))
                else:
                    f = fmt(kind, v)
                    row.append(Cell(stext(f), v, key=f))
            block.append(row)
        BP_BLOCKS.append([[None] + [c.value for c in r[1:]] for r in block])
        for j, (_k, _kind, mx) in enumerate(BP_COLS):
            bold_best([r[j + 1] for r in block], maximize=mx)
        rows += block
    check(len(due) == 1, f'due-date factors {due}')
    # Section S5 states: SLAB best in flow time, WIP and utilization at every
    # size, lowest tardiness from 30 modules on, SPT lower at 10 and 20
    for (S, _ds), b in zip(M.SIZES, BP_BLOCKS):
        sl, sp, ll = b
        # columns: 1 makespan, 2 flow time, 3 WIP, 4 utilization, 5 tardiness
        check(sl[2] < min(sp[2], ll[2]) and sl[3] < min(sp[3], ll[3])
              and sl[4] > max(sp[4], ll[4]), f'S5 {S}: flow, WIP, utilization')
        if S in ('Ten', 'Twenty'):
            check(sp[5] < sl[5], f'S5 {S}: SPT tardiness not lower')
        else:
            check(sl[5] < min(sp[5], ll[5]), f'S5 {S}: SLAB tardiness not lowest')
    if CPU_GPU:
        s = 'test_results/PPVC/<ds>/byproducts_hstar/byproducts_greedy.json'
        supp('SuCpuGpuMaxDiff', dec(max(abs(v[2]) for v in CPU_GPU.values()), 1),
             s, 'max over sizes of |mean CPU makespan - mean evaluation-file '
             'makespan| of the SLAB policy, seed 301 (h)')
        for S, ds in (('Ten', '10x25+ppvc-mixed'), ('Eighty', '80x25+ppvc-mixed')):
            if ds in CPU_GPU:
                supp(f'SuCpuGpuOrders{S}', str(CPU_GPU[ds][0]), s,
                     f'orders of {ds} on which the CPU and GPU schedules differ')
    supp('SuDueFactor', dec(due.pop(), 1),
         'test_results/PPVC/<ds>/byproducts_hstar/*.json', 'due_factor')
    return write_table(out_dir, 's5_byproducts.tex', '@{}lcccccc@{}', head,
                       rows, 7)


# --------------------------------------------------------------------------- #
# S6 flow-line routing
# --------------------------------------------------------------------------- #
FLOW_SIZES = (('Ten', '10x25+ppvc-mixed'), ('Twenty', '20x25+ppvc-mixed'),
              ('Fifty', '50x25+ppvc-mixed'), ('Eighty', '80x25+ppvc-mixed'))


def flowline(S, ds):
    rel_p = f'test_results/PPVC/{ds}/flowline/flowline_per_instance_{ds}.json'
    d = M.read_json(M.guard(os.path.join(REPO, rel_p)))
    src(rel_p)
    n = M.N_INST[ds]
    pi = d['per_instance']
    check(d['dataset'] == ds and d['n_instances'] == n and len(pi) == n,
          f'{rel_p}: header')
    for i, r in enumerate(pi):
        check(r['instance'] == f'instance_{i:03d}', f'{rel_p}: order at {i}')
    if 'contention' in d:
        M.check_contention_summary(d['contention'], rel_p)
        for i, r in enumerate(pi):
            M.check_evidence(r, f'{rel_p} instance {i}')
        proven = None
    else:
        # A run without contention fields counts only if every flow-line
        # solve is a proven optimum, which does not depend on the CPU share.
        check(all(r['cpsat_flowline_status'] == 'OPTIMAL'
                  and float(r['cpsat_flowline_bound'])
                  == float(r['cpsat_flowline']) for r in pi),
              f'{rel_p}: no contention evidence and not every solve optimal')
        proven = max(float(r['cpsat_flowline_s']) for r in pi)
    spt_flow = np.array([r['spt_flowline'] for r in pi], dtype=float)
    spt_flex = np.array([r['spt_flexible'] for r in pi], dtype=float)
    check(np.array_equal(spt_flex, rule_values('SPT', ds)),
          f'{rel_p}: flexible SPT differs from the evaluation file')
    cp_flow = np.array([r['cpsat_flowline'] for r in pi], dtype=float)
    check(float(d['cpsat_time_limit_s']) == 300.0
          and int(d['cpsat_workers']) == 2 and d['cpsat_start'] == 'cold',
          f'{rel_p}: CP-SAT settings')
    return spt_flow, spt_flex, cp_flow, proven, d


def s6_flowline(out_dir):
    head = [' & \\multicolumn{3}{c}{SPT rule} & \\multicolumn{3}{c}{CP-SAT} \\\\',
            '\\cmidrule(lr){2-4}\\cmidrule(l){5-7}',
            'Modules & Flow line (h) & Flexible (h) & Gain (\\%) & '
            'Flow line (h) & Flexible (h) & Gain (\\%) \\\\']
    rows, proven_s, compat = [], [], []
    for S, ds in FLOW_SIZES:
        try:
            spt_flow, spt_flex, cp_flow, proven, d = flowline(S, ds)
        except Missing as err:
            p = pend(f'Flow{S}', str(err))
            rows.append([str(SIZE_NUM[S])] + [p.text] * 6)
            continue
        if proven is not None:
            proven_s.append(proven)
        compat.append((np.mean([r['compat_flexible'] for r in d['per_instance']]),
                       np.mean([r['compat_flowline'] for r in d['per_instance']])))
        recs = cp_recs(ds, S, 'cold')
        cp_flex = np.array([at_time(r, 300) for r in recs])
        g_spt = np.mean(100.0 * (spt_flow - spt_flex) / spt_flow)
        g_cp = np.mean(100.0 * (cp_flow - cp_flex) / cp_flow)
        if M.cp_budget(S) == 300:
            cpf = mac(f'CpCold{S}', lambda: M.cp_col(ds, S, 'cold').mean(), 'h')
        else:
            cpf = num(lambda: cp_flex.mean(), 'h', f'CpColdThreeHundred{S}')
        if S == 'Ten':
            supp('SuFlowGainSptTen', dec(g_spt, 1), M.rel(os.path.join(
                M.TR, ds, 'flowline')), 'mean per-order gain of flexible SPT (%)')
            supp('SuFlowGainCpTen', dec(g_cp, 1), M.rel(os.path.join(
                M.TR, ds, 'flowline')), 'mean per-order gain of flexible CP-SAT (%)')
        rows.append([str(SIZE_NUM[S]), num(lambda: spt_flow.mean(), 'h', 'x'),
                     mac(f'Spt{S}', lambda: spt_flex.mean(), 'h'),
                     num(lambda: g_spt, 'pct', 'x'),
                     num(lambda: cp_flow.mean(), 'h', 'x'), cpf,
                     num(lambda: g_cp, 'pct', 'x')])
    supp('SuFlowLimit', '300', 'flowline_per_instance_<ds>.json',
         'cpsat_time_limit_s (asserted)')
    supp('SuFlowWorkers', '2', 'flowline_per_instance_<ds>.json',
         'cpsat_workers (asserted)')
    if proven_s:
        supp('SuFlowCpMaxS', dec(max(proven_s), 2),
             'test_results/PPVC/10x25+ppvc-mixed/flowline/'
             'flowline_per_instance_10x25+ppvc-mixed.json',
             'max cpsat_flowline_s over the orders (every solve OPTIMAL)')
    if compat:
        supp('SuFlowCompatFlex', dec(compat[0][0], 1),
             'flowline_per_instance_10x25+ppvc-mixed.json',
             'mean compatible stations per operation, flexible routing')
    return write_table(out_dir, 's6_flowline.tex', '@{}rcccccc@{}', head,
                       rows, 7)


# --------------------------------------------------------------------------- #
# S7 sensitivity
# --------------------------------------------------------------------------- #
SENS_ROW = re.compile(
    r'^\| ([0-9.]+) \| ([0-9.]+) \(([0-9.]+)\) \| ([0-9.]+) \(([0-9.]+)\) '
    r'\| (\d+)/(\d+)/(\d+) \| ([0-9.eE+-]+) \| (YES|NO) \|$', re.M)


def parse_sens(fname, what):
    rel_p = f'test_results/PPVC/{DS10}/{fname}'
    path = M.guard(os.path.join(REPO, rel_p))
    if not os.path.exists(path):
        raise Missing(rel_p, 'file absent')
    with open(path) as f:
        txt = f.read()
    src(rel_p)
    check(f'`{DS10}`  (100 instances)' in txt, f'{rel_p}: dataset')
    check(f'./trained_network/PPVC/{M.HSTAR_MODEL}.pth' in txt,
          f'{rel_p}: checkpoint')
    check('seed_test: 50' in txt and what in txt, f'{rel_p}: protocol')
    rows = SENS_ROW.findall(txt)
    check([r[0] for r in rows] == ['0', '0.1', '0.2', '0.3'],
          f'{rel_p}: deltas {[r[0] for r in rows]}')
    return rows


def s7_sensitivity(out_dir):
    head = [' & \\multicolumn{4}{c}{Processing times perturbed} & '
            '\\multicolumn{4}{c}{Waits perturbed} \\\\',
            '\\cmidrule(lr){2-5}\\cmidrule(l){6-9}',
            '$\\delta$ & SLAB (h) & SPT (h) & \\makecell{Shorter/equal/\\\\longer} '
            '& $p$ & SLAB (h) & SPT (h) & \\makecell{Shorter/equal/\\\\longer} & '
            '$p$ \\\\']
    parts = []
    spt = rule_values('SPT', DS10)
    # the sensitivity runs decode on the CPU, like the by-product runs, so
    # delta 0 is checked against the CPU schedules of byproducts_greedy.json
    bp = byproducts(DS10, 'greedy')
    h1 = np.array([r['makespan'] for r in bp['per_instance']], dtype=float)
    for fname, what in (('sensitivity_summary.md', 'processing time'),
                        ('sensitivity_summary_lags.md', 'time-lag')):
        try:
            rows = parse_sens(fname, what)
        except Missing as err:
            parts.append(None)
            PENDING.setdefault('Sens', str(err))
            continue
        r0 = rows[0]   # delta 0 must equal the evaluation files
        check(r0[1] == fmt('h', h1.mean()) and r0[3] == fmt('h', spt.mean()),
              f'{fname}: delta-0 means differ from the CPU schedules and the '
              f'SPT evaluation file')
        check('/'.join(r0[5:8]) == wtl(h1, spt), f'{fname}: delta-0 counts')
        check(fmt_p(float(r0[8])) == fmt_p(wilcoxon_paired(h1, spt)),
              f'{fname}: delta-0 p-value')
        parts.append(rows)
    ps = [float(r[8]) for rows in parts if rows for r in rows]
    if ps and all(rows is not None for rows in parts):
        check(all(float(r[1]) < float(r[3]) for rows in parts for r in rows),
              'Section S7 states that SLAB is shorter than SPT at every delta')
        supp('SuSensMaxP', fmt_p(max(ps)), 'test_results/PPVC/10x25+ppvc-mixed/'
             'sensitivity_summary*.md', 'largest Wilcoxon p over both studies')
        supp('SuSensMaxDelta', parts[0][-1][0], 'sensitivity_summary*.md',
             'largest delta')
    out = []
    for k in range(4):
        row = None
        for rows in parts:
            if rows is None:
                cells = ['\\pend{Sens}'] * 4
                delta = ('0', '0.1', '0.2', '0.3')[k]
            else:
                r = rows[k]
                delta = r[0]
                a = Cell(r[1], float(r[1]), key=r[1])
                b = Cell(r[3], float(r[3]), key=r[3])
                bold_best([a, b])
                cells = [a, b, '/'.join(r[5:8]), fmt_p(float(r[8]))]
            row = (row or [delta]) + cells
        out.append(row)
    return write_table(out_dir, 's7_sensitivity.tex', '@{}lcccccccc@{}', head,
                       out, 9)


# --------------------------------------------------------------------------- #
# S8 stability-aware residual CP-SAT
# --------------------------------------------------------------------------- #
def sweep(dtype):
    rel_p = (f'test_results/PPVC/{DS10}/reactive_hstar/'
             f'stability_sweep_{dtype}.npy')
    path = M.guard(os.path.join(REPO, rel_p))
    if not os.path.exists(path):
        raise Missing(rel_p, 'file absent')
    d = np.load(path, allow_pickle=True).item()
    src(rel_p)
    n = int(d['n'])
    check(d['disruption_type'] == dtype and n == 30 and int(d['budget']) == 30,
          f'{rel_p}: header')
    m = d['meta']
    check(float(m['offset']) == 0.4 and float(m['lag_factor']) == 2.0
          and int(m['seed_test']) == 50 and int(m['cpsat_workers']) == 2,
          f'{rel_p}: meta')
    check([float(w) for w in d['weights']] == [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
          f'{rel_p}: weights')
    for w in d['weights']:
        c = d['contention'][w]
        check(c['n_records'] == n and c['n_low_share'] == 0
              and c['n_no_evidence'] == 0
              and c.get('n_records_without_fields', 0) == 0,
              f'{rel_p}: contention at w={w}')
        fr = d['frontier'][w]
        check(fr['feas'] == n and len(fr['infl']) == n, f'{rel_p}: w={w} feasible')
    check(d['drl']['feas'] == n and len(d['drl']['nerv']) == n,
          f'{rel_p}: SLAB feasible')
    return d


def s8_stability(out_dir):
    head = ['Re-planner & $w$ (h) & \\makecell{Makespan\\\\increase (\\%)} & '
            '\\makecell{Changed\\\\operations} & \\makecell{Station\\\\changes} & '
            '\\makecell{Start\\\\moves} & \\makecell{Mean\\\\time (s)} & '
            '\\makecell{Max.\\\\time (s)} & \\makecell{Proven\\\\optimal} \\\\']
    rows, max_s, n_opt, n_all = [], [], [], []
    for dtype, title in (('lag_perturb', 'One wait doubled'),
                         ('breakdown', 'Station breakdown')):
        rows.append(('BLOCK', title))
        try:
            d = sweep(dtype)
        except Missing as err:
            rows.append(['SLAB'] + [pend('Stab', str(err)).text] * 8)
            continue
        n = int(d['n'])
        dr = d['drl']
        if dtype == 'lag_perturb':
            infl = mac('SlabInflTen', lambda: np.mean(dr['infl']), 'pct')
            nerv = mac('SlabNervTen', lambda: np.mean(dr['nerv']), 'h')
        else:
            infl = num(lambda: np.mean(dr['infl']), 'pct', 'x')
            nerv = num(lambda: np.mean(dr['nerv']), 'h', 'x')
        rows.append(['SLAB re-planning', DASH, infl, nerv,
                     dec(np.mean(dr['mch_nerv']), 1),
                     dec(np.mean(dr['time_nerv']), 1),
                     fmt('s', np.mean(dr['sec'])), fmt('s', np.max(dr['sec'])),
                     DASH])
        for w in d['weights']:
            fr = d['frontier'][w]
            if dtype == 'lag_perturb' and float(w) == 0.5:
                infl = mac('StabInflTen', lambda f=fr: np.mean(f['infl']), 'pct')
                nerv = mac('StabNervTen', lambda f=fr: np.mean(f['nerv']), 'h')
            else:
                infl = num(lambda f=fr: np.mean(f['infl']), 'pct', 'x')
                nerv = num(lambda f=fr: np.mean(f['nerv']), 'h', 'x')
            max_s.append(float(np.max(fr['sec'])))
            n_opt.append(int(fr['optimal']))
            n_all.append(n)
            rows.append(['Residual CP-SAT', dec(float(w), 1), infl, nerv,
                         dec(np.mean(fr['mch_nerv']), 1),
                         dec(np.mean(fr['time_nerv']), 1),
                         fmt('s', np.mean(fr['sec'])),
                         fmt('s', np.max(fr['sec'])),
                         f'{int(fr["optimal"])}/{n}'])
    s = 'test_results/PPVC/10x25+ppvc-mixed/reactive_hstar/stability_sweep_*.npy'
    if max_s:
        # Section S8 states: at w = 0.5 the doubled-wait makespan increase
        # is at most that of SLAB with fewer changes, and after a breakdown every
        # positive weight changes under one operation without a longer plan
        lp, bd = sweep('lag_perturb'), sweep('breakdown')
        check(np.mean(lp['frontier'][0.5]['infl'])
              <= np.mean(lp['drl']['infl'])
              and np.mean(lp['frontier'][0.5]['nerv'])
              < np.mean(lp['drl']['nerv']), 'S8 w=0.5 against SLAB')
        for w in bd['weights'][1:]:
            check(np.mean(bd['frontier'][w]['nerv']) < 1.0
                  and np.mean(bd['frontier'][w]['infl']) <= 1e-9,
                  f'S8 breakdown w={w}')
        del SOURCES[:2]
        supp('SuStabMaxS', fmt('s', max(max_s)), s,
             'max solve seconds over both disruption types and all weights')
        supp('SuStabN', str(n_all[0]), s, 'disrupted orders per type')
        supp('SuStabBudget', str(int(d['budget'])), s, 'CP-SAT time limit (s)')
        supp('SuStabWorkers', str(int(d['meta']['cpsat_workers'])), s,
             'CP-SAT workers')
        supp('SuStabWLow', dec(float(d['weights'][1]), 1), s,
             'smallest positive weight (h per changed operation)')
        supp('SuStabWMax', dec(float(d['weights'][-1]), 0), s, 'largest weight')
        supp('SuStabAllOptimal', 'yes' if n_opt == n_all else 'no', s,
             'every penalized solve proven optimal')
        check(n_opt == n_all, 'not every stability solve is optimal; the '
                              'S8 text states that every solve is optimal')
    return write_table(out_dir, 's8_stability.tex', '@{}lrccccccc@{}', head,
                       rows, 9)


# --------------------------------------------------------------------------- #
# S9 Gantt chart (numbers of the caption)
# --------------------------------------------------------------------------- #
def s9_gantt():
    """Numbers of the S9 Gantt chart, read from the data file of
    make_gantt_v3.py: one network trained without waits (DANIEL, seed 301,
    the DANIEL network of the necessity table of the paper) schedules the
    100 orders of the 15-station test set in two ways, (a) dispatched with
    the waits and (b) planned without the waits and right-shift repaired.
    The plotted order is the one whose repair penalty lies closest to the
    median penalty of the set (lowest id on ties)."""
    s = M.rel(GANTT_JSON)
    ds = M.TIGHT[1]
    d = M.read_json(GANTT_JSON)
    model = M.LAGFREE.format(s=301)[len('greedy+'):]
    check(d['data_path'] == f'data/PPVC/{ds}' and d['model'] == model
          and d['backbone'] == 'DANIEL' and d['trained_with_waits'] is False
          and d['input_aids'] == 'none' and int(d['seed_test']) == 50,
          f'{s}: header')
    p = f'train_log/PPVC/config_{model}.json'
    c = M.read_json(os.path.join(REPO, p))
    check(c['ppvc_lagblind'] is True and c['use_lag_features'] is False
          and c['use_type_embedding'] is False and int(c['seed_train']) == 301,
          f'{p}: not the network trained without waits')
    pc = paper_constants()
    check(int(d['n_stations']) == sum(d['station_counts'].values())
          == int(pc['FactoryTight']), f'{s}: station count')
    po = d['per_order']
    n = M.N_INST[ds]
    check([int(r['order']) for r in po] == list(range(n)), f'{s}: orders')
    disp = np.array([r['dispatched'] for r in po], dtype=float)
    rep = np.array([r['repaired'] for r in po], dtype=float)
    pen = 100.0 * (rep / disp - 1.0)
    check(np.allclose(pen, [r['penalty_pct'] for r in po], atol=1e-9),
          f'{s}: per-order penalties')
    med = float(np.median(pen))
    sel = d['selection']
    check(abs(med - float(sel['median_penalty_pct'])) < 1e-9
          and int(sel['n_orders']) == n, f'{s}: median penalty')
    # with an even count the median is the mean of the two middle penalties,
    # which then tie up to rounding; the lower order id is plotted
    dist = np.abs(pen - med)
    pick = int(np.flatnonzero(dist <= dist.min() + 1e-9)[0])
    check(pick == int(d['instance'])
          and d['instance_file'] == f'instance_{pick:03d}.fjs', f'{s}: order')
    one = d['instance_makespan_h']
    check(float(one['dispatched']) == disp[pick]
          and float(one['repaired']) == rep[pick], f'{s}: panel values')
    for k, v in (('dispatched', disp[pick]), ('repaired', rep[pick])):
        sch = d['schedules'][k]
        check(float(max(sch['end'])) == v, f'{s}: {k} schedule ends off its makespan')
    # the dispatched schedules (CPU rollout) against the evaluation file of
    # the paper (GPU rollout): equal on most orders and on the plotted order
    stored = M.result(M.LAGFREE.format(s=301), ds)
    n_eq = int(np.sum(stored == disp))
    check(n_eq >= 0.9 * n and stored[pick] == disp[pick]
          and abs(stored.mean() - disp.mean()) < 1.0,
          f'{s}: dispatched makespans differ from {M.res_rel(M.LAGFREE.format(s=301), ds)} '
          f'on {n - n_eq} orders')
    supp('SuGanttN', str(n), s, f'test orders of {ds}')
    supp('SuGanttDispatch', dec(disp[pick], 0), s,
         'panel (a) makespan, dispatched with the waits (h)')
    supp('SuGanttRepair', dec(rep[pick], 0), s,
         'panel (b) makespan, planned without waits and repaired (h)')
    supp('SuGanttPenalty', dec(pen[pick], 1), s,
         'repair penalty of the plotted order (%)')
    supp('SuGanttPenaltyMedian', dec(med, 1), s,
         f'median over the {n} orders of 100 x (repaired / dispatched - 1) (%)')


# --------------------------------------------------------------------------- #
# S10 generator tables
# --------------------------------------------------------------------------- #
OP_NAMES = {
    'mould_setup': 'Mould set-up', 'rebar_cage': 'Rebar cage',
    'mep_cast_in': 'Cast-in MEP items', 'pre_pour_QC': 'Pre-pour check',
    'concrete_pour': 'Casting', 'demould_inspect': 'Demoulding and inspection',
    'trial_assembly': 'Trial assembly', 'material_prep': 'Material preparation',
    'steel_cutting': 'Steel cutting', 'frame_welding_2d': 'Frame welding',
    'shell_welding_3d': 'Shell welding', 'weld_QC': 'Weld check',
    'fire_corrosion_protect': 'Fire and corrosion coating',
    'trial_stack': 'Trial stacking', 'plumbing': 'Plumbing',
    'electrical': 'Electrical', 'HVAC_ducting': 'Air-conditioning ducts',
    'mep_test_QC': 'MEP test', 'internal_partitions': 'Internal partitions',
    'waterproofing': 'Waterproofing', 'screed_tiling': 'Screed and tiling',
    'plaster_skim': 'Plaster and skim coat', 'paint_coat_1': 'Paint coat 1',
    'paint_coat_2': 'Paint coat 2', 'finishing_QC': 'Finishing check',
    'floor_screed': 'Floor screed', 'floor_finishes': 'Floor finishes',
    'doors_windows': 'Doors and windows', 'fitout_fixtures': 'Fixtures',
    'final_QC': 'Final check', 'wrap_protect_ship': 'Wrapping and dispatch'}
STATION_LABEL = {'A': 'Formwork', 'B': 'Concrete casting', 'C': 'MEP fit-out',
                 'D': 'Finishing', 'E': 'Painting', 'F': 'Assembly',
                 'G': 'Dispatch', 'H': 'Steel fabrication', 'Q': 'Quality check'}


def rng_text(lo, hi):
    return f'{lo}' if lo == hi else f'{lo}--{hi}'


def s10_tables(out_dir):
    import ppvc_instance_generator as G
    src('ppvc_instance_generator.py')
    check(set(STATION_LABEL) == set(G.STATION_TYPES), 'station types')
    opt = {}
    for names, p in G.OPTIONAL_OPS:
        for nm in names:
            opt[nm] = p
    segs = (('RC shell', G._RC_SHELL), ('Steel shell', G._STEEL_SHELL),
            ('MEP fit-out, every module', G._MEP),
            ('Wet finishing (wet modules)', G._WET_FINISH),
            ('Dry finishing (dry modules)', G._DRY_FINISH),
            ('Assembly and dispatch, every module', G._ASSEMBLY))
    head = ['Operation & Station type & Processing time (h) & Wait (h) & '
            'Omission probability \\\\']
    rows, waits = [], {}
    for title, seg in segs:
        rows.append(('BLOCK', title))
        for name, st, _ot, (dlo, dhi), (llo, lhi) in seg:
            check(name in OP_NAMES, f'no display name for {name}')
            rows.append([OP_NAMES[name], STATION_LABEL[st], rng_text(dlo, dhi),
                         rng_text(llo, lhi) if lhi > 0 else DASH,
                         ('%s\\%%' % dec(100.0 * opt[name], 0)) if name in opt
                         else DASH])
            if lhi > 0:
                waits.setdefault(name, set()).add((llo, lhi))
    for r in G.ROUTES.values():
        check(len(r) == G.OPS_PER_MODULE, 'route length')
    pc = paper_constants()
    for name, lo, hi in (('concrete_pour', 'CureLo', 'CureHi'),
                         ('waterproofing', 'PondLo', 'PondHi'),
                         ('paint_coat_1', 'DryLo', 'DryHi'),
                         ('paint_coat_2', 'DryLo', 'DryHi')):
        check(waits[name] == {(int(pc[lo]), int(pc[hi]))},
              f'{name}: generator waits {waits[name]} differ from '
              f'macros.tex {pc[lo]}-{pc[hi]}')
    f1 = write_table(out_dir, 's10_routes.tex', '@{}llccc@{}', head, rows, 5)

    src('ppvc_instance_generator.py')
    head = ['Station type & Default & \\FactoryTight{} stations & '
            '\\FactorySmall{} stations \\\\']
    rows = []
    for t in G.STATION_TYPES:
        rows.append([STATION_LABEL[t], str(G.DEFAULT_FACTORY[t]),
                     str(G.TIGHT_FACTORY[t]), str(G.SMALL_FACTORY[t])])
    check(all(G.TIGHT_FACTORY[t] == -(-G.DEFAULT_FACTORY[t] // 2)
              for t in G.STATION_TYPES), 'tight factory is not the default '
          'halved and rounded up')
    tot = [sum(f.values()) for f in (G.DEFAULT_FACTORY, G.TIGHT_FACTORY,
                                     G.SMALL_FACTORY)]
    check(tot == [int(pc['FactoryDefault']), int(pc['FactoryTight']),
                  int(pc['FactorySmall'])], f'station totals {tot}')
    rows += ['MID', ['Total', '\\FactoryDefault', '\\FactoryTight',
                     '\\FactorySmall']]
    f2 = write_table(out_dir, 's10_stations.tex', '@{}lccc@{}', head, rows, 4)

    mix = G.PRESET_MIXES['mixed']
    for nm, v in zip(('SuMixRcWet', 'SuMixRcDry', 'SuMixSteelWet',
                      'SuMixSteelDry'), mix):
        supp(nm, dec(v, 3), 'ppvc_instance_generator.py',
             "PRESET_MIXES['mixed']")
    rc = G.PRESET_MIXES['rc_project']
    supp('SuWetShare', dec(100.0 * rc[0] / (rc[0] + rc[1]), 0),
         'ppvc_instance_generator.py', "wet share of PRESET_MIXES['rc_project'] (%)")
    supp('SuPtDev', dec(100.0 * G.PT_DEVIATION, 0), 'ppvc_instance_generator.py',
         'PT_DEVIATION (%)')
    supp('SuOpsPerModule', str(G.OPS_PER_MODULE), 'ppvc_instance_generator.py',
         'OPS_PER_MODULE')
    supp('SuOpsMin', str(G.OPS_PER_MODULE - len(G.OPTIONAL_OPS)),
         'ppvc_instance_generator.py', 'OPS_PER_MODULE - len(OPTIONAL_OPS)')
    return f1, f2


# --------------------------------------------------------------------------- #
# S11 hyperparameters and protocols
# --------------------------------------------------------------------------- #
def n_params(path):
    import torch
    torch.set_num_threads(1)
    sd = torch.load(path, map_location='cpu', weights_only=True)
    check(isinstance(sd, dict) and all(hasattr(v, 'numel') for v in sd.values()),
          f'{path}: not a state dict')
    return int(sum(v.numel() for v in sd.values()))


def thousands(k):
    return f'{k:,}'


def sci(x):
    m, e = f'{x:.0e}'.split('e')
    return '$%s\\times10^{%d}$' % (m, int(e))


def s11_hyper(out_dir):
    cfgs = {}
    for fw, stem in (('DANIEL', '10x25+ppvc-mixed+a1lagx05-2k-s{s}'),
                     ('PLAN', '10x25+ppvc-mixed+plan-a1lagx05-2k-s{s}')):
        per = []
        for s in M.SEEDS:
            p = f'train_log/PPVC/config_{stem.format(s=s)}.json'
            per.append(M.read_json(os.path.join(REPO, p)))
            src(p)
        base = {k: v for k, v in per[0].items()
                if k not in ('seed_train', 'model_suffix')}
        for c in per[1:]:
            check({k: v for k, v in c.items()
                   if k not in ('seed_train', 'model_suffix')} == base,
                  f'{fw}: seeds differ in configuration')
        check([c['seed_train'] for c in per] == list(M.SEEDS), f'{fw}: seeds')
        check(per[0]['framework'] == fw, f'{fw}: framework')
        cfgs[fw] = per[0]
    hp = os.path.join(HGNN_SAVE, 'hgnn-a1lagx05-vs1020-s300', 'run_config.json')
    h = M.read_json(hp)
    src(M.rel(hp))
    ht, hm, ha = h['train_paras'], h['model_paras'], h['args']
    ck = {'DANIEL': 'trained_network/PPVC/10x25+ppvc-mixed+a1lagx05-2k-s301.pth',
          'PLAN': 'trained_network/PPVC/10x25+ppvc-mixed+plan-a1lagx05-2k-s301.pth',
          'HGNN': 'backbones/hgnn/save/hgnn-a1lagx05-vs1020-s300/best.pt'}
    npar = {}
    for k, p in ck.items():
        src(p)
        npar[k] = n_params(os.path.join(REPO, p))
    d, pl = cfgs['DANIEL'], cfgs['PLAN']

    def both(key, f=str):
        return [f(d[key]), f(pl[key])]
    vs = ', '.join(str(x) for x in d['vali_sizes_resolved'])
    check(pl['vali_sizes_resolved'] == d['vali_sizes_resolved'], 'vali sizes')
    hvs = ha['vali_sizes'].replace(',', ', ')
    rows = [
        ['Trainable parameters', thousands(npar['DANIEL']),
         thousands(npar['PLAN']), thousands(npar['HGNN'])],
        ['PPO updates'] + both('max_updates') + [str(ht['max_iterations'])],
        ['Training orders per update'] + both('num_envs')
        + [str(h['env_paras']['batch_size'])],
        ['New training orders every (updates)'] + both('reset_env_timestep')
        + [str(ht['parallel_iter'])],
        ['Learning rate'] + both('lr', sci) + [sci(ht['lr'])],
        ['Discount factor $\\gamma$'] + both('gamma', lambda x: dec(x, 0))
        + [dec(ht['gamma'], 0)],
        ['GAE parameter $\\lambda$'] + both('gae_lambda') + [DASH],
        ['Clipping parameter'] + both('eps_clip') + [str(ht['eps_clip'])],
        ['Epochs per update'] + both('k_epochs') + [str(ht['K_epochs'])],
        ['Minibatch size'] + both('minibatch_size') + [str(ht['minibatch_size'])],
        ['Policy-loss weight'] + both('ploss_coef', lambda x: dec(x, 1))
        + [dec(ht['A_coeff'], 1)],
        ['Value-loss weight'] + both('vloss_coef') + [str(ht['vf_coeff'])],
        ['Entropy weight'] + both('entloss_coef') + [str(ht['entropy_coeff'])],
        ['Validation every (updates)'] + both('validate_timestep')
        + [str(ha['save_timestep'])],
        ['Validation orders per size'] + both('vali_size') + [str(ha['vali_size'])],
        ['Validation sizes (modules)', vs, vs, hvs],
        ['Wait factor of training orders'] + both('lag_scale')
        + [str(ha['lag_scale'])],
        ['Training seeds', ', '.join(str(s) for s in M.SEEDS),
         ', '.join(str(s) for s in M.SEEDS), str(ha['seed'])],
        'MID',
        ['Attention heads per layer'] + both(
            'num_heads_OAB', lambda x: ', '.join(str(v) for v in x))
        + [', '.join(str(v) for v in hm['num_heads'])],
        ['Embedding sizes per layer'] + both(
            'layer_fea_output_dim', lambda x: ', '.join(str(v) for v in x))
        + [f'{hm["out_size_ope"]} (hidden {hm["hidden_size_ope"]})'],
        ['Actor layers $\\times$ width', f'{d["num_mlp_layers_actor"]} $\\times$ '
         f'{d["hidden_dim_actor"]}', f'{pl["num_mlp_layers_actor"]} $\\times$ '
         f'{pl["hidden_dim_actor"]}', f'{hm["n_hidden_actor"]} $\\times$ '
         f'{hm["n_latent_actor"]}'],
        ['Critic layers $\\times$ width', f'{d["num_mlp_layers_critic"]} $\\times$ '
         f'{d["hidden_dim_critic"]}', f'{pl["num_mlp_layers_critic"]} $\\times$ '
         f'{pl["hidden_dim_critic"]}', f'{hm["n_hidden_critic"]} $\\times$ '
         f'{hm["n_latent_critic"]}'],
    ]
    check(d['num_heads_OAB'] == d['num_heads_MAB'], 'DANIEL heads')
    check(d['use_lag_features'] is False and d['use_type_embedding'] is False,
          'SLAB policy without the aids')
    head = ['Setting & DANIEL (SLAB) & PLAN & HGNN \\\\']
    supp('SuTypeEmbDim', str(int(d['type_emb_dim'])),
         'train_log/PPVC/config_10x25+ppvc-mixed+a1lagx05-2k-s301.json',
         'type_emb_dim')
    supp('SuParamsDaniel', thousands(npar['DANIEL']), ck['DANIEL'],
         'sum of numel over the state dict')
    supp('SuParamsPlan', thousands(npar['PLAN']), ck['PLAN'],
         'sum of numel over the state dict')
    supp('SuParamsHgnn', thousands(npar['HGNN']), ck['HGNN'],
         'sum of numel over the state dict')
    return write_table(out_dir, 's11_hyper.tex', '@{}lccc@{}', head, rows, 4)


def s11_protocols(out_dir):
    head = ['Setting & Value \\\\']
    metas, aff = [], set()
    for _S, ds, n in M.DIS_SIZES:
        for _T, dtype in M.DIS_TYPES:
            rel_p = M.dis_rel(ds, dtype)
            try:
                d = M.reactive(ds, dtype, n)
            except Missing as err:
                PENDING.setdefault('Dis', str(err))
                continue
            src(rel_p)
            metas.append(d['meta'])
            for key in ('drl_reactive', 'ga_reactive', 'cpsat_5', 'cpsat_30'):
                for a in d['records'][key]['affinity']:
                    aff.update(int(x) for x in a)
    check(len(metas) == 12, f'{len(metas)} disruption files')
    keys = ('disruption_offset', 'lag_factor', 'pt_factor', 'shortage_factor',
            'ga_budget', 'ga_pop', 'cpsat_workers', 'seed_test',
            'nervousness_time_threshold', 'low_cpu_share', 'hit_limit_frac')
    for k in keys:
        check(len({json.dumps(m[k]) for m in metas}) == 1, f'meta {k} differs')
    m0 = metas[0]
    basis = {m['shortage_basis'] for m in metas
             if m['disruption_type'] == 'material_shortage'}
    check(basis == {'remaining_work'}, f'shortage basis {basis}')
    budgets = {tuple(m['cpsat_budgets']) for m in metas}
    check(len(budgets) == 1, 'CP-SAT budgets differ between files')
    # the 1-s arm is never read (make_paper_macros.DIS_METHODS)
    cp_b = sorted(set(budgets.pop()) - {1})
    check(cp_b == [5, 30], f'CP-SAT budgets {cp_b}')
    # uncertain waits
    recs = M.stoch('Ten', 'cpsat_rolling')
    src(M.stoch_rel('Ten', 'cpsat_rolling'))
    sig = {r['sig'] for r in recs}
    check(len(sig) == 1, f'rolling sigs {sig}')
    mm = re.match(r'^cpsat(\d+)s-w(\d+)-re(\d+)s\|lagseed(\d+)$', sig.pop())
    check(mm is not None, 'rolling sig format')
    nominal_s, workers, resolve_s, lagseed = (int(x) for x in mm.groups())
    for r in recs:
        check(float(r['nominal_solve_timing']['time_limit_s']) == nominal_s
              and int(r['nominal_solve_timing']['workers']) == workers,
              'nominal plan budget')
        for t in r['resolve_timings']:
            check(float(t['time_limit_s']) == resolve_s, 're-solve budget')
        aff.update(int(x) for x in r['affinity'])
    # 50-module nominal plan: CP-SAT started from the SPT schedule
    recs50 = M.stoch('Fifty', 'cpsat_repair')
    src(M.stoch_rel('Fifty', 'cpsat_repair'))
    sig50 = {r['sig'] for r in recs50}
    check(len(sig50) == 1, f'50-module repair sigs {sig50}')
    mm50 = re.match(r'^cpsat(\d+)s-w(\d+)-hintspt\|lagseed(\d+)$', sig50.pop())
    check(mm50 is not None, '50-module repair sig format (SPT hint expected)')
    nominal50_s, workers50, lagseed50 = (int(x) for x in mm50.groups())
    check(lagseed50 == lagseed, 'lag seed differs between sizes')
    for r in recs50:
        check(float(r['nominal_solve_timing']['time_limit_s']) == nominal50_s
              and int(r['nominal_solve_timing']['workers']) == workers50
              and r.get('cpsat_hint') == 'spt', '50-module nominal plan settings')
    # residual CP-SAT hint, from the residual result files of both sizes
    hints = set()
    for ds in ('10x25+ppvc-mixed', '50x25+ppvc-mixed'):
        for _T, dtype in M.DIS_TYPES:
            rp = f'test_results/PPVC/{ds}/reactive_hstar_hint/residual_cpsat_{dtype}.npy'
            hints.add(np.load(os.path.join(REPO, rp), allow_pickle=True).item()['meta'].get('hint'))
    check(hints == {'right_shift'}, f'residual CP-SAT hints {hints}')
    cvs = sorted({round(float(r['cv']), 3) for r in recs})
    reps = sorted({int(r['rep']) for r in recs})
    trunc = py_constants(os.path.join(REPO, 'eval_stochastic_lags.py'),
                         ('LAG_TRUNC', 'LOW_CPU_SHARE', 'HIT_LIMIT_FRAC'))
    src('eval_stochastic_lags.py')
    check(trunc['LOW_CPU_SHARE'] == M.LOW_CPU_SHARE == m0['low_cpu_share'],
          'CPU-share threshold')
    rows = [
        ('BLOCK', 'Uncertain waits (Table~\\ref{M-tab:uncertain} of the paper)'),
        ['Wait factor', 'lognormal, mean 1, truncated to [%s, %s]'
         % (dec(trunc['LAG_TRUNC'][0], 1), dec(trunc['LAG_TRUNC'][1], 0))],
        ['Coefficients of variation', ', '.join(dec(c, 1) for c in cvs)],
        ['Realizations per order and level', str(len(reps))],
        ['Seed of the wait realizations', str(lagseed)],
        ['Nominal CP-SAT plan', '%d\\,s, %d workers at 10 modules; from the SPT '
         'schedule, %d\\,s, %d workers at 50 modules'
         % (nominal_s, workers, nominal50_s, workers50)],
        ['Rolling CP-SAT re-solve', 'up to %d\\,s' % resolve_s],
        ('BLOCK', 'Disruptions (Table~\\ref{M-tab:disrupt} of the paper)'),
        ['Operations committed before the disruption',
         '%s\\%% of the SLAB plan' % dec(100 * m0['disruption_offset'], 0)],
        ['Doubled wait', 'wait $\\times$ %s' % dec(m0['lag_factor'], 0)],
        ['Longer operation', 'processing time $\\times$ %s' % dec(m0['pt_factor'], 1)],
        ['Material delay', '%s $\\times$ remaining work of the module'
         % dec(m0['shortage_factor'], 1)],
        ['Genetic algorithm', 'population %d, %s\\,s per re-plan'
         % (int(m0['ga_pop']), dec(m0['ga_budget'], 0))],
        ['Residual CP-SAT', '%d\\,s and %d\\,s, %d workers, started from the '
         'right-shifted plan' % (cp_b[0], cp_b[1], int(m0['cpsat_workers']))],
        ['Changed operation', 'station changed or start moved by more than '
         '%s\\,h' % dec(m0['nervousness_time_threshold'], 0)],
        ('BLOCK', 'Time-limited runs'),
        ['Minimum CPU-time ratio', dec(trunc['LOW_CPU_SHARE'], 1)],
        ['Budget-limited run', 'uses at least %s\\%% of its limit'
         % dec(100 * trunc['HIT_LIMIT_FRAC'], 0)],
    ]
    supp('SuLowShare', dec(trunc['LOW_CPU_SHARE'], 1), 'eval_stochastic_lags.py',
         'LOW_CPU_SHARE')
    supp('SuHitLimit', dec(100 * trunc['HIT_LIMIT_FRAC'], 0),
         'eval_stochastic_lags.py', 'HIT_LIMIT_FRAC (%)')
    supp('SuCpEarlyS', str(CP_EARLY_S), 'make_supplement_tables.py',
         'incumbent time reported for every CP-SAT run (s)')
    supp('SuStochSeed', str(lagseed), M.stoch_rel('Ten', 'cpsat_rolling'), 'sig')
    check(M.STOCH['Fifty'][3] == [n for S, _d, n in M.DIS_SIZES if S == 'Fifty'][0],
          'the two studies use different 50-module subsets')
    supp('SuStudyNFifty', str(M.STOCH['Fifty'][3]), 'make_paper_macros.STOCH',
         'first orders of the 50-module set in the uncertain-wait and '
         'disruption studies')
    supp('SuStochReps', str(len(reps)), M.stoch_rel('Ten', 'cpsat_rolling'),
         'replications per order and cv')
    supp('SuStochNominalS', str(nominal_s), M.stoch_rel('Ten', 'cpsat_rolling'),
         'nominal CP-SAT time limit (s)')
    supp('SuDisGaPop', str(int(m0['ga_pop'])), 'reactive_hstar meta', 'ga_pop')
    supp('SuNervThresh', dec(m0['nervousness_time_threshold'], 0),
         'reactive_hstar meta', 'nervousness_time_threshold (h)')
    supp('SuDisOffset', dec(100 * m0['disruption_offset'], 0),
         'reactive_hstar meta', 'disruption_offset (%)')
    supp('SuDisGaSeed', str(int(m0['ga_seed'])), 'reactive_hstar meta', 'ga_seed')
    f = write_table(out_dir, 's11_protocols.tex', '@{}ll@{}', head, rows, 2)
    return f, aff


TEST_SETS = (  # block, label, dataset
    ('Test sets', 'Default factory', '10x25+ppvc-mixed'),
    ('Test sets', 'Default factory', '20x25+ppvc-mixed'),
    ('Test sets', 'Default factory', '30x25+ppvc-mixed'),
    ('Test sets', 'Default factory', '50x25+ppvc-mixed'),
    ('Test sets', 'Default factory', '80x25+ppvc-mixed'),
    ('Test sets', '\\FactoryTight-station factory', '10x15+ppvc-mixed'),
    ('Test sets', 'Variable routes', '10x25+ppvc-mixed-var'),
    ('Test sets', 'Variable routes', '20x25+ppvc-mixed-var'),
    ('Test sets', 'Variable routes', '50x25+ppvc-mixed-var'),
    ('Test sets', 'Variable routes', '80x25+ppvc-mixed-var'),
    ('Test sets', 'Waits $\\times$\\FtLagScale', '10x25+ppvc-mixed-lagx1.5'),
    ('Test sets', 'Waits $\\times$\\FtLagScale', '50x25+ppvc-mixed-lagx1.5'),
    ('Independent test sets', 'Default factory', '10x25+ppvc-mixed-ind'),
    ('Independent test sets', 'Default factory', '20x25+ppvc-mixed-ind'),
    ('Independent test sets', 'Default factory', '50x25+ppvc-mixed-ind'),
    ('Independent test sets', 'Default factory', '80x25+ppvc-mixed-ind'),
    ('Validation sets, halved waits', 'Default factory',
     'vali:10x25+ppvc-mixed-lagx0.5'),
    ('Validation sets, halved waits', 'Default factory',
     'vali:20x25+ppvc-mixed-lagx0.5'))


def s11_testsets(out_dir):
    """Every test, independent and validation set with its size and the seed
    of its first order; asserts that the seed ranges of the three kinds of
    set do not overlap (the paper states it)."""
    head = ['Set & Modules & Stations & Orders & Seed of the first order \\\\']
    rows, last, ranges = [], None, {}
    for block, label, key in TEST_SETS:
        vali = key.startswith('vali:')
        ds = key[5:] if vali else key
        base = (os.path.join(REPO, 'data', 'data_train_vali', 'PPVC', ds)
                if vali else os.path.join(M.DATA, ds))
        meta_p = os.path.join(base, 'dataset_meta.json')
        meta = M.read_json(meta_p)
        src(M.rel(meta_p))
        n = len(glob.glob(os.path.join(base, 'instance_*.fjs')))
        check(int(meta['n_instances']) == n, f'{ds}: {n} instance files')
        if not vali:
            check(n == M.N_INST[ds], f'{ds}: {n} orders')
        m = re.match(r'^(\d+)x(\d+)\+', ds)
        modules, stations = int(m.group(1)), int(m.group(2))
        check(int(meta['n_modules']) == modules, f'{ds}: modules')
        if 'lagx1.5' in ds:
            check(float(meta['lag_scale']) == 1.5
                  and meta['source_set'] == ds.replace('-lagx1.5', ''),
                  f'{ds}: source set')
        if vali:
            check(float(meta['lag_scale']) == 0.5 and int(meta['seed0']) == 100,
                  f'{ds}: validation meta')
        kind = 'vali' if vali else ('ind' if ds.endswith('-ind') else 'test')
        if 'lagx1.5' not in ds:   # these reuse the orders of their source set
            ranges.setdefault(kind, []).append(
                (int(meta['seed0']), int(meta['seed0']) + n))
        if block != last:
            rows.append(('BLOCK', block))
            last = block
        rows.append([label, str(modules), str(stations), str(n),
                     f'{int(meta["seed0"]):,}'])
    kinds = sorted(ranges)
    for i, a in enumerate(kinds):
        for b in kinds[i + 1:]:
            for lo1, hi1 in ranges[a]:
                for lo2, hi2 in ranges[b]:
                    check(hi1 <= lo2 or hi2 <= lo1,
                          f'seed ranges of {a} and {b} overlap')
    return write_table(out_dir, 's11_testsets.tex', '@{}lcccr@{}', head, rows, 5)


# --------------------------------------------------------------------------- #
# Memetic and GA parameters, hardware, minimum CPU share
# --------------------------------------------------------------------------- #
def metaheuristic_constants():
    c = py_constants(os.path.join(REPO, 'eval_memetic.py'),
                     ('NC', 'NP_RESTART', 'PM', 'NM', 'SEED_RULES'))
    supp('SuMaPop', str(c['NC']), 'eval_memetic.py', 'NC')
    supp('SuMaShortS', '60', M.ma_rel(DS10, 60),
         'budget of the shorter memetic run at 10 modules (s)')
    supp('SuMaRestart', thousands(c['NP_RESTART']), 'eval_memetic.py',
         'NP_RESTART')
    supp('SuMaLsProb', dec(100 * c['PM'], 0), 'eval_memetic.py', 'PM (%)')
    supp('SuMaLsIter', str(c['NM']), 'eval_memetic.py', 'NM')
    supp('SuMaSeedRules', str(len(c['SEED_RULES'])), 'eval_memetic.py',
         'len(SEED_RULES)')
    g = py_constants(os.path.join(REPO, 'eval_ga.py'), ('SEED_RULES',))
    with open(os.path.join(REPO, 'eval_ga.py')) as f:
        txt = f.read()
    pop = re.search(r"'--pop', type=int, default=(\d+)", txt)
    tour = re.search(r'tour_k=(\d+)', txt)
    elite = re.search(r'elite=(\d+), tour_k', txt)
    check(pop and tour and elite, 'eval_ga.py: defaults not found')
    supp('SuGaPop', pop.group(1), 'eval_ga.py', '--pop default')
    supp('SuGaTour', tour.group(1), 'eval_ga.py', 'tour_k default')
    supp('SuGaElite', elite.group(1), 'eval_ga.py', 'elite default')
    supp('SuGaSeedRules', str(len(g['SEED_RULES'])), 'eval_ga.py',
         'len(SEED_RULES)')


def timed_cores_and_share():
    """Cores and minimum CPU share of the budget-limited CP-SAT runs of the
    per-size tables (anytime files)."""
    cores, shares = set(), []
    for S, ds in M.SIZES:
        modes = ['cold', 'warm-spt'] + (['warm-policy'] if S in ('Fifty', 'Eighty')
                                        else [])
        for mode in modes:
            for r in cp_recs(ds, S, mode):
                cores.update(int(x) for x in r['cpu_affinity'])
                if float(r['solve_elapsed_s']) > M.ANYTIME_TIMED_S:
                    shares.append(float(r['cpu_share']))
        for mode in ('cold', 'warm-spt'):
            if S in ('Fifty', 'Eighty'):
                for r in M.anytime(ds, mode, 1800, mode, n=20):
                    cores.update(int(x) for x in r['cpu_affinity'])
                    if float(r['solve_elapsed_s']) > M.ANYTIME_TIMED_S:
                        shares.append(float(r['cpu_share']))
    del SOURCES[:]
    return cores, min(shares)


def core_ghz(cores):
    f = set()
    for c in sorted(cores):
        p = f'/sys/devices/system/cpu/cpu{c}/cpufreq/cpuinfo_max_freq'
        with open(p) as fh:
            f.add(int(fh.read().strip()))
    return f


def hardware(aff):
    cores, min_share = timed_cores_and_share()
    cores |= aff
    supp('SuNBoot', str(N_BOOT), 'make_supplement_tables.py',
         'bootstrap resamples (stats_tests.paired_mean_diff_ci, seed 0)')
    supp('SuMinShare', dec(np.floor(min_share * 100) / 100, 2),
         'or_solution/PPVC/anytime/*.jsonl',
         'minimum cpu_share of the CP-SAT runs longer than 10 s, rounded down')
    try:
        with open('/sys/devices/cpu_atom/cpus') as fh:
            a, b = (int(x) for x in fh.read().strip().split('-'))
        check(cores <= set(range(a, b + 1)),
              f'timed runs used cores {sorted(cores)} outside the efficiency '
              f'cores {a}-{b}; Section S11 states efficiency cores')
    except OSError as err:
        WARN.append(f'core type not verified: {err}')
    try:
        fr = core_ghz(cores)
        if len(fr) == 1:
            supp('SuTimedGHz', dec(fr.pop() / 1e6, 1),
                 '/sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_max_freq',
                 f'max frequency of cores {sorted(cores)} of the timed runs')
        else:
            supp_pend('SuTimedGHz', 'sysfs', f'timed runs on cores of '
                      f'different frequencies {sorted(fr)}')
    except OSError as err:
        supp_pend('SuTimedGHz', 'sysfs', str(err))
    try:
        with open('/proc/cpuinfo') as fh:
            name = re.search(r'model name\s*:\s*(.+)', fh.read()).group(1)
        name = name.replace('(R)', '').replace('(TM)', '').strip()
        name = re.sub(r'\s+', ' ', name)
        supp('SuCpu', name, '/proc/cpuinfo', 'model name')
    except (OSError, AttributeError) as err:
        supp_pend('SuCpu', '/proc/cpuinfo', str(err))
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30,
                             check=True).stdout.strip().splitlines()[0]
        gname, mem = [x.strip() for x in out.split(',')]
        supp('SuGpu', gname, 'nvidia-smi', 'GPU name')
        supp('SuGpuGB', dec(float(mem) / 1024.0, 0), 'nvidia-smi',
             'memory.total MiB / 1024')
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as err:
        # nvidia-smi unavailable (driver/library mismatch after a host update):
        # fall back to the values it reported for this machine, cached in
        # paper_inputs/gpu_info.json.
        cache = os.path.join(REPO, 'paper_inputs', 'gpu_info.json')
        if os.path.exists(cache):
            with open(cache) as fh:
                g = json.load(fh)
            supp('SuGpu', g['name'], 'paper_inputs/gpu_info.json (cached nvidia-smi)', 'GPU name')
            supp('SuGpuGB', dec(float(g['memory_total_mib']) / 1024.0, 0),
                 'paper_inputs/gpu_info.json (cached nvidia-smi)', 'memory.total MiB / 1024')
        else:
            supp_pend('SuGpu', 'nvidia-smi', str(err))
            supp_pend('SuGpuGB', 'nvidia-smi', str(err))
    import torch
    supp('SuTorch', '.'.join(torch.__version__.split('+')[0].split('.')[:2]),
         'torch.__version__', 'major.minor')
    try:
        import ortools
        supp('SuOrtools', '.'.join(ortools.__version__.split('.')[:2]),
             'ortools.__version__', 'major.minor')
    except ImportError as err:
        supp_pend('SuOrtools', 'ortools', str(err))


# --------------------------------------------------------------------------- #
# Output of the supplement macros, collisions, main
# --------------------------------------------------------------------------- #
def paper_macro_names():
    names = set()
    for p in paper_macro_paths(PAPER_MACRO_FILES):
        with open(p) as f:
            names |= set(re.findall(
                r'\\(?:newcommand|renewcommand|providecommand)\*?\{\\([A-Za-z]+)\}',
                f.read()))
    return names


def write_supp_macros(out_dir):
    clash = sorted(set(SUPP) & paper_macro_names())
    check(not clash, f'supplement macros clash with the paper: {clash}')
    lines = ['% supp_macros.tex: supplement-only numbers, generated by',
             '% make_supplement_tables.py; do not edit by hand.']
    for k, (v, s, c) in SUPP.items():
        lines.append(f'\\newcommand{{\\{k}}}{{{v}}} % source: {s}; {c}')
    with open(os.path.join(out_dir, 'supp_macros.tex'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return 'supp_macros.tex'


SUPP_TEX = os.environ.get('SUPP_TEX', os.path.join(PAPER_INPUTS, 'supplement.tex'))
BIBTEX = os.path.join(os.path.expanduser('~'), '.TinyTeX', 'bin',
                      'x86_64-linux', 'bibtex')


def make_bbl(out_dir):
    """Bibliography of the supplement: the keys cited in supplement.tex,
    formatted by bibtex with IEEEtran.bst from the paper's refs.bib, written
    to supp_refs.bbl. supplement.tex inputs this file, so its build needs no
    bibtex run and no bibliography path outside the supplement folder."""
    bib = os.path.join(M.OUT_DIR, 'refs.bib')
    rel_bib = os.path.relpath(bib, out_dir)[:-len('.bib')]
    if ' ' in rel_bib or not os.path.exists(SUPP_TEX):
        WARN.append(f'supp_refs.bbl not written (path {rel_bib})')
        return None
    with open(SUPP_TEX) as f:
        tex = re.sub(r'(?<!\\)%.*', '', f.read())
    keys = []
    for group in re.findall(r'\\cite\{([^}]*)\}', tex):
        for k in group.split(','):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    with open(bib) as f:
        have = set(re.findall(r'^@\w+\{([^,\s]+),', f.read(), re.M))
    missing = [k for k in keys if k not in have]
    check(not missing, f'keys cited in supplement.tex but absent from refs.bib: '
                       f'{missing}')
    aux = os.path.join(out_dir, 'supp_refs.aux')
    with open(aux, 'w') as f:
        f.write('\\relax\n' + ''.join(f'\\citation{{{k}}}\n' for k in keys)
                + '\\bibstyle{IEEEtran}\n' + f'\\bibdata{{{rel_bib}}}\n')
    exe = BIBTEX if os.path.exists(BIBTEX) else 'bibtex'
    r = subprocess.run([exe, 'supp_refs'], cwd=out_dir, capture_output=True,
                       text=True, timeout=120)
    blg = os.path.join(out_dir, 'supp_refs.blg')
    with open(blg) as f:
        log = f.read()
    check(r.returncode == 0 and 'Warning--' not in log
          and 'error message' not in log, f'bibtex failed: {log[-600:]}')
    for fn in (aux, blg):
        os.remove(fn)
    return 'supp_refs.bbl'


def reset():
    MAC.clear()
    CPU_GPU.clear()
    del SEED_HI[:]
    PENDING.clear()
    del WARN[:]
    SUPP.clear()
    del SOURCES[:]
    M.reset()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--strict', action='store_true',
                    help='fail when any cell is pending')
    ap.add_argument('--out_dir', default=OUT_DIR)
    args = ap.parse_args(argv)
    reset()
    with open(MAC_JSON) as f:
        MAC.update(json.load(f))
    os.makedirs(args.out_dir, exist_ok=True)
    out = []
    try:
        for S, ds in M.SIZES:
            out.append(s1_size(S, ds, args.out_dir))
        # Section S1 states that every seed is shorter than SPT at every size
        check(len(SEED_HI) == 15 and max(SEED_HI) < 0,
              f'a seed interval against SPT reaches zero: {SEED_HI}')
        out.append(s1_long(args.out_dir))
        out.append(s1_ind(args.out_dir))
        out.append(s1_mix(args.out_dir))
        out.append(s1_dose(args.out_dir))
        out.append(s2_stats(args.out_dir))
        out.append(s3_hgnn(args.out_dir))
        out.append(s4_bound(args.out_dir))
        out.append(s4_aids(args.out_dir))
        s4_curves()
        out.append(s5_byproducts(args.out_dir))
        out.append(s6_flowline(args.out_dir))
        out.append(s7_sensitivity(args.out_dir))
        out.append(s8_stability(args.out_dir))
        s9_gantt()
        out += list(s10_tables(args.out_dir))
        out.append(s11_hyper(args.out_dir))
        out.append(s11_testsets(args.out_dir))
        f, aff = s11_protocols(args.out_dir)
        out.append(f)
        metaheuristic_constants()
        hardware(aff)
        out.append(write_supp_macros(args.out_dir))
        bbl = make_bbl(args.out_dir)
        if bbl:
            out.append(bbl)
    except AssertionError as err:
        print(f'FAILED ASSERTION: {err}')
        return 1
    print(f'make_supplement_tables: wrote {len(out)} files to '
          f'{M.rel(args.out_dir)}')
    for f in out:
        print(f'  {f}')
    if PENDING:
        print('PENDING (\\pend):')
        for k, v in PENDING.items():
            print(f'  {k}: {v}')
    for w in WARN:
        print(f'WARNING: {w}')
    if args.strict and PENDING:
        print(f'--strict: {len(PENDING)} pending items')
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
