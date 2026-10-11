#!/usr/bin/env python3
"""make_paper_macros.py -- every result number of the TII manuscript,
computed from the result files, with its source and computation.

Writes to paper_tables/ (or $PAPER_OUT_DIR, or --out_dir):
  macros_results.tex          one \\newcommand per number, each followed by
                              "% source: <file(s)>; <computation>"
  macros_results.json         name -> value, raw value, source, computation
  macros_results_preview.md   one table per exhibit, the pending list, the
                              self-check results and the name collisions

Modes
  default   a missing or incomplete source gives \\pend{<Name>} and is listed
            on stdout; a failed assertion on a present source fails the run
            (exit 1, nothing written)
  --strict  a missing source also fails the run (exit 2, nothing written)

Conventions
  makespans in hours with 1 decimal; percentages with 1 decimal and no % sign;
  percent difference of X against Y = 100 * (mean(X) / mean(Y) - 1), sign
  kept (negative = X shorter), and <Name>Gain = -difference; counts as
  integers; p-values as $a\\times10^{-b}$ with two significant digits (a plain
  decimal from 0.01 up);
  rank-biserial r and contention shares with 2 decimals; seconds with 2
  decimals below 1 s and 1 decimal from 1 s. A 3-seed mean is the mean of the
  three per-seed means and its std the population std of those means. Values
  are rounded half-up on their shortest decimal representation.

Every source is only read. Nothing is written under test_results/,
or_solution/, trained_network/, train_log/ or data/, and no file inside a
no_evidence_backup/ folder is ever opened.

usage: python make_paper_macros.py [--strict] [--out_dir DIR]
"""
import argparse
import functools
import glob
import json
import math
import os
import re
import sys
import time
from collections import OrderedDict
from decimal import ROUND_HALF_UP, Decimal

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from stats_tests import (holm, paired_mean_diff_ci, rank_biserial,  # noqa: E402
                         wilcoxon_paired)

TR = os.path.join(REPO, 'test_results', 'PPVC')
AT = os.path.join(REPO, 'or_solution', 'PPVC', 'anytime')
DATA = os.path.join(REPO, 'data', 'PPVC')
TRAIN_LOG = os.path.join(REPO, 'train_log')
OUT_DIR = os.environ.get('PAPER_OUT_DIR', os.path.join(REPO, 'paper_tables'))
OUT_TEX, OUT_JSON, OUT_MD = ('macros_results.tex', 'macros_results.json',
                             'macros_results_preview.md')
EXISTING_MACRO_FILES = ('macros.tex', 'macros_pending.tex')

LOW_CPU_SHARE = 0.9       # a budget-limited solve below this share is invalid
ANYTIME_TIMED_S = 10.0    # anytime records with a longer solve are checked

SIZES = (('Ten', '10x25+ppvc-mixed'), ('Twenty', '20x25+ppvc-mixed'),
         ('Thirty', '30x25+ppvc-mixed'), ('Fifty', '50x25+ppvc-mixed'),
         ('Eighty', '80x25+ppvc-mixed'))
TIGHT = ('Tight', '10x15+ppvc-mixed')
SIZES_IND = (('Ten', '10x25+ppvc-mixed-ind'), ('Twenty', '20x25+ppvc-mixed-ind'),
             ('Fifty', '50x25+ppvc-mixed-ind'),
             ('Eighty', '80x25+ppvc-mixed-ind'))
SIZES_VAR = (('Ten', '10x25+ppvc-mixed-var'), ('Twenty', '20x25+ppvc-mixed-var'),
             ('Fifty', '50x25+ppvc-mixed-var'), ('Eighty', '80x25+ppvc-mixed-var'))
SIZES_LAG15 = (('Ten', '10x25+ppvc-mixed-lagx1.5'),
               ('Fifty', '50x25+ppvc-mixed-lagx1.5'))
N_INST = {'10x25+ppvc-mixed': 100, '20x25+ppvc-mixed': 50,
          '30x25+ppvc-mixed': 20, '50x25+ppvc-mixed': 50,
          '80x25+ppvc-mixed': 50, '10x15+ppvc-mixed': 100,
          '10x25+ppvc-mixed-ind': 100, '20x25+ppvc-mixed-ind': 50,
          '50x25+ppvc-mixed-ind': 50, '80x25+ppvc-mixed-ind': 50,
          '10x25+ppvc-mixed-var': 100, '20x25+ppvc-mixed-var': 50,
          '50x25+ppvc-mixed-var': 50, '80x25+ppvc-mixed-var': 50,
          '10x25+ppvc-mixed-lagx1.5': 100, '50x25+ppvc-mixed-lagx1.5': 50}

SEEDS = (301, 302, 303)
SEED_PAT = '30{1,2,3}'
HSTAR = 'greedy+10x25+ppvc-mixed+a1lagx05-2k-s{s}'
HSTAR_MODEL = '10x25+ppvc-mixed+a1lagx05-2k-s301'
HSTAR1 = 'greedy+' + HSTAR_MODEL
LAGFREE = 'greedy+10x25+ppvc-mixed+a0-2k-s{s}'
FULL = 'greedy+10x25+ppvc-mixed+a1-2k-s{s}'
FULLAIDS = 'greedy+10x25+ppvc-mixed+full2k-s{s}'
LFBOUND = 'greedy+10x25+ppvc-mixed+a1nolagbound-2k-s{s}'
TIGHTNET = 'greedy+10x15+ppvc-mixed+full2k-s{s}'
PLANHALF = 'greedy+10x25+ppvc-mixed+plan-a1lagx05-2k-s{s}'
PLAN_A0 = 'greedy+10x25+ppvc-mixed+plan-a0-2k-s301'
PLAN_A1 = 'greedy+10x25+ppvc-mixed+plan-a1-2k-s301'
PLAN_FULL = 'greedy+10x25+ppvc-mixed+plan-full2k-s301'
REPAIR_DANIEL = 'A0REPAIR+10x25+ppvc-mixed+a0-2k-s{s}'
REPAIR_PLAN = 'A0REPAIR+10x25+ppvc-mixed+plan-a0-2k-s301'
HGNN_A0 = 'greedy+hgnn-aware+hgnn-a0-s300'
HGNN_REPAIR = 'greedy+hgnn-blind+hgnn-a0-s300'
HGNN_A1 = 'greedy+hgnn-aware+hgnn-a1-s300'
HGNN_A1VS = 'greedy+hgnn-aware+hgnn-a1-vs1020-s300'
HGNN_HALF = 'greedy+hgnn-aware+hgnn-a1lagx05-vs1020-s300'
VAR_TRAINED = 'greedy+10x25+ppvc-mixed-var+full2k-s301'
# training-wait dose arms (one seed each): every training/validation wait x0.25 / x0.75
DOSE_Q = 'greedy+10x25+ppvc-mixed+a1lagx025-2k-s301'
DOSE_TQ = 'greedy+10x25+ppvc-mixed+a1lagx075-2k-s301'
FT_TIGHT = 'greedy+10x15+ppvc-mixed+fthalf-tight-s301'
FT_LAG = 'greedy+10x25+ppvc-mixed+fthalf-lag15-s301'
FT_LOGS = ('train_10x15+ppvc-mixed+fthalf-tight-s301.log',
           'train_10x25+ppvc-mixed+fthalf-lag15-s301.log')
# The 10x25 SPT result was written by test_trained_model.py under a model tag;
# the rule ignores the network, so the file holds the same rule rollout.
SPT_FALLBACK = {'10x25+ppvc-mixed': 'SPT+10x25+ppvc-mixed+smoke'}


# --------------------------------------------------------------------------- #
# Errors and checks
# --------------------------------------------------------------------------- #
class Missing(Exception):
    """A source is absent or not yet complete (default mode: \\pend)."""

    def __init__(self, source, why):
        super().__init__(f'{source}: {why}')
        self.source, self.why = source, why


def check(cond, msg):
    """Assertion on a present source; failure ends the run in both modes."""
    if not cond:
        raise AssertionError(msg)


def rel(path):
    return os.path.relpath(path, REPO)


def guard(path):
    check('no_evidence_backup' not in path.split(os.sep),
          f'{rel(path)}: files in no_evidence_backup/ are never read')
    return path


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def dec(x, nd):
    """x rounded half-up to nd decimals, as text; never '-0.0'."""
    x = float(x)
    check(math.isfinite(x), f'non-finite value {x}')
    q = Decimal(1).scaleb(-nd)
    d = Decimal(repr(x)).quantize(q, rounding=ROUND_HALF_UP)
    if d == 0:
        d = abs(d)
    return f'{d:f}'


def fmt_p(p):
    """p-value as $a\\times10^{-b}$ with two significant digits."""
    p = float(p)
    check(0 < p <= 1, f'p-value out of range: {p}')
    e = math.floor(math.log10(p))
    m = Decimal(repr(p / 10 ** e)).quantize(Decimal('0.1'),
                                             rounding=ROUND_HALF_UP)
    if m >= 10:
        m, e = Decimal('1.0'), e + 1
    if e >= -2:
        # 0.01 <= p <= 1: a plain decimal reads better than 8.7 x 10^-1
        return f'${format(m.scaleb(e), "f")}$'
    return f'${m}\\times10^{{{e}}}$'


def fmt(kind, x):
    if kind in ('h', 'pct'):
        return dec(x, 1)
    if kind in ('pct2', 'r', 'share'):
        return dec(x, 2)
    if kind == 's':
        return dec(x, 2) if abs(float(x)) < 1 else dec(x, 1)
    if kind == 'int':
        check(float(x).is_integer(), f'count is not an integer: {x}')
        return str(int(x))
    if kind == 'round_int':
        return dec(x, 0)
    if kind == 'median':
        return str(int(x)) if float(x).is_integer() else dec(x, 1)
    if kind == 'p':
        return fmt_p(x)
    raise ValueError(f'unknown format {kind}')


def pct(x, y):
    return 100.0 * (float(x) / float(y) - 1.0)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
REG = OrderedDict()
FAILS = []
NAME_RE = re.compile(r'^[A-Za-z]+$')


def define(name, exhibit, row, col, kind, fn, source, comp):
    """Compute one macro. fn() returns the raw number; a Missing source makes
    the macro pending, a failed check is recorded as a failure."""
    check(NAME_RE.match(name) is not None, f'macro name not letters only: {name}')
    check(name not in REG, f'macro defined twice: {name}')
    e = dict(name=name, exhibit=exhibit, row=row, col=col, kind=kind,
             source=source, computation=comp, raw=None, value=None,
             pending=False, reason=None, failed=None)
    try:
        raw = fn()
        e['raw'] = float(raw)
        e['value'] = fmt(kind, raw)
    except Missing as err:
        e['pending'], e['reason'] = True, str(err)
    except AssertionError as err:
        e['failed'] = str(err)
        FAILS.append((name, str(err)))
    REG[name] = e
    return e


def shown(name):
    """The one-decimal value an already defined 'h' macro prints, as a float,
    so that a derived gap equals the one a reader recomputes from the table."""
    e = REG[name]
    if e['pending']:
        raise Missing(name, e['reason'])
    check(e['kind'] in ('h', 'pct'), f'shown() on a {e["kind"]} macro: {name}')
    return float(fmt('h', e['raw']))


def raw_of(name):
    """Raw value of an already defined macro (for derived macros)."""
    e = REG[name]
    if e['pending']:
        raise Missing(name, f'depends on pending macro ({e["reason"]})')
    check(e['failed'] is None, f'depends on failed macro {name}')
    return e['raw']


# --------------------------------------------------------------------------- #
# Readers (read-only)
# --------------------------------------------------------------------------- #
def read_jsonl(path):
    """Records of a JSONL file. A line that does not parse is skipped only when
    it is the last one (a record being appended); elsewhere the run fails."""
    guard(path)
    if not os.path.exists(path):
        raise Missing(rel(path), 'file absent')
    with open(path) as f:
        lines = f.read().split('\n')
    while lines and not lines[-1].strip():
        lines.pop()
    recs = []
    for k, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            check(k == len(lines) - 1,
                  f'{rel(path)}: line {k + 1} of {len(lines)} does not parse')
    return recs


def read_json(path):
    """A JSON file written in one piece; unparseable means being written."""
    guard(path)
    if not os.path.exists(path):
        raise Missing(rel(path), 'file absent')
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise Missing(rel(path), 'does not parse (being written)')


def check_evidence(r, where):
    """Contention fields of one record of an instrumented timed source."""
    for k in ('n_low_share', 'n_no_evidence'):
        check(k in r, f'{where}: record without contention field {k}')
        check(int(r[k]) == 0, f'{where}: {k} = {r[k]}')
    cs = r.get('cpu_share_min')
    if cs is not None:
        check(float(cs) >= LOW_CPU_SHARE, f'{where}: cpu_share_min {cs}')


def check_contention_summary(c, where):
    check(c is not None, f'{where}: no contention summary')
    check(c.get('n_low_share') == 0, f'{where}: n_low_share {c.get("n_low_share")}')
    check(c.get('n_no_evidence') == 0,
          f'{where}: n_no_evidence {c.get("n_no_evidence")}')
    check(c.get('n_records_without_fields', 0) == 0,
          f'{where}: records without contention fields')
    cs = c.get('cpu_share_min')
    check(cs is None or cs >= LOW_CPU_SHARE, f'{where}: cpu_share_min {cs}')


def res_rel(tag, ds):
    return f'test_results/PPVC/{ds}/Result_{tag}_{ds}.npy'


@functools.lru_cache(maxsize=None)
def result(tag, ds):
    """Column 0 (makespan) of Result_<tag>_<ds>.npy, row i = instance i."""
    path = guard(os.path.join(REPO, res_rel(tag, ds)))
    if not os.path.exists(path):
        raise Missing(rel(path), 'file absent')
    a = np.load(path, allow_pickle=False)
    check(a.ndim == 2 and a.shape[1] == 2, f'{rel(path)}: shape {a.shape}')
    check(a.shape[0] == N_INST[ds],
          f'{rel(path)}: {a.shape[0]} rows, expected {N_INST[ds]}')
    col = a[:, 0].astype(float)
    check(np.all(np.isfinite(col)) and np.all(col > 0),
          f'{rel(path)}: non-positive or non-finite makespan')
    col.setflags(write=False)
    return col


def seeded(tpl):
    return '{s}' in tpl


def src(tpl, ds):
    return res_rel(tpl.format(s=SEED_PAT) if seeded(tpl) else tpl, ds)


def mean_of(tpl, ds):
    """3-seed mean of per-seed means, or the mean of one result file."""
    if seeded(tpl):
        return float(np.mean([result(tpl.format(s=s), ds).mean()
                              for s in SEEDS]))
    return float(result(tpl, ds).mean())


def std_of(tpl, ds):
    """Population std of the three per-seed means."""
    check(seeded(tpl), f'{tpl}: std needs three seeds')
    return float(np.std([result(tpl.format(s=s), ds).mean() for s in SEEDS]))


def inst_of(tpl, ds):
    """Per-instance values; per-instance mean over the seeds when seeded."""
    if seeded(tpl):
        return np.mean([result(tpl.format(s=s), ds) for s in SEEDS], axis=0)
    return np.asarray(result(tpl, ds))


def rule_tag(r, ds):
    tag = f'{r}+pdr'
    if os.path.exists(os.path.join(REPO, res_rel(tag, ds))):
        return tag
    if r == 'SPT' and ds in SPT_FALLBACK:
        return SPT_FALLBACK[ds]
    return tag


def rule_src(r, ds):
    return res_rel(rule_tag(r, ds), ds)


def rule_note(r, ds):
    if rule_tag(r, ds) == f'{r}+pdr':
        return ''
    return (f' [no {r}+pdr file for {ds}; this file is asserted equal per '
            f'instance to test_results/PPVC/{ds}/byproducts_hstar/'
            f'byproducts_{r}.json]')


@functools.lru_cache(maxsize=None)
def rule_col(r, ds):
    tag = rule_tag(r, ds)
    col = result(tag, ds)
    if tag != f'{r}+pdr':
        # a fallback file must equal an independent re-run of the same rule
        # (same tie-break seeds 50 + i) instance by instance
        bp = os.path.join(TR, ds, 'byproducts_hstar', f'byproducts_{r}.json')
        d = read_json(bp)
        check(d['dataset'] == ds and d['method'] == r and d['seed_test'] == 50
              and d['validated'] is True, f'{rel(bp)}: header mismatch')
        ms = np.array([x['makespan'] for x in d['per_instance']], dtype=float)
        check(ms.shape == col.shape and np.array_equal(ms, col),
              f'{res_rel(tag, ds)} differs from {rel(bp)}')
    return col


@functools.lru_cache(maxsize=None)
def anytime(ds, file_mode, t, rec_mode, n=None, model=None):
    """Records of one anytime CP-SAT file, ordered by instance id."""
    path = os.path.join(AT, f'{ds}_{file_mode}_{t}s_w4.jsonl')
    recs = read_jsonl(path)
    n = N_INST[ds] if n is None else n
    by = {}
    for r in recs:
        i = int(r['instance_id'])
        where = f'{rel(path)} instance {i}'
        check(r['dataset'] == ds and r['mode'] == rec_mode
              and float(r['time_limit_s']) == float(t), f'{where}: header')
        check(r['instance'] == f'instance_{i:03d}', f'{where}: name')
        check(i not in by, f'{where}: duplicate record')
        by[i] = r
    if len(by) < n:
        raise Missing(rel(path), f'{len(by)} of {n} instances')
    check(set(by) == set(range(n)), f'{rel(path)}: instance ids not 0..{n - 1}')
    for i in range(n):
        r, where = by[i], f'{rel(path)} instance {i}'
        check(r['objective'] is not None, f'{where}: no solution')
        if 'validator_feasible' in r:
            check(r['validator_feasible'] is True, f'{where}: validator')
        if model is not None:
            check(r.get('model_name') == model, f'{where}: model {r.get("model_name")}')
        if float(r['solve_elapsed_s']) > ANYTIME_TIMED_S:
            check('cpu_share' in r, f'{where}: no cpu_share')
            check(float(r['cpu_share']) >= LOW_CPU_SHARE,
                  f'{where}: cpu_share {r["cpu_share"]}')
    return tuple(by[i] for i in range(n))


def any_rel(ds, file_mode, t):
    return f'or_solution/PPVC/anytime/{ds}_{file_mode}_{t}s_w4.jsonl'


def cp_budget(S):
    return 300 if S in ('Ten', 'Twenty', 'Thirty') else 600


def cp_col(ds, S, mode, field='objective'):
    if mode == 'warm-policy':
        recs = anytime(ds, 'warm-policy-halflag-s301', 600, 'warm-policy',
                       model=HSTAR_MODEL)
    else:
        recs = anytime(ds, mode, cp_budget(S), mode)
    return np.array([float(r[field]) for r in recs])


def cp_src(ds, S, mode):
    if mode == 'warm-policy':
        return any_rel(ds, 'warm-policy-halflag-s301', 600)
    return any_rel(ds, mode, cp_budget(S))


def ma_rel(ds, budget=300):
    return f'test_results/PPVC/{ds}/Result_MA{budget}+memetic_{ds}.json'


@functools.lru_cache(maxsize=None)
def memetic(ds, budget=300):
    path = os.path.join(REPO, ma_rel(ds, budget))
    d = read_json(path)
    n, where = N_INST[ds], ma_rel(ds, budget)
    check(d['dataset'] == ds and float(d['budget_s']) == float(budget),
          f'{where}: header')
    check(d['n_instances'] == n and len(d['instances']) == n,
          f'{where}: {d["n_instances"]} instances, expected {n}')
    check_contention_summary(d.get('contention'), where)
    vals = []
    for i, r in enumerate(d['instances']):
        check(r['instance'] == f'instance_{i:03d}', f'{where}: order at {i}')
        check(r['validator_violations'] == 0, f'{where}: violations at {i}')
        check_evidence(r, f'{where} instance {i}')
        vals.append(float(r['best_makespan']))
    return np.array(vals)


def ga_rel(ds, budget):
    return f'test_results/PPVC/{ds}/Result_GA{budget}+{ds}_{ds}.json'


@functools.lru_cache(maxsize=None)
def ga(ds, budget):
    path = os.path.join(REPO, ga_rel(ds, budget))
    d = read_json(path)
    n, where = N_INST[ds], ga_rel(ds, budget)
    check(d['dataset'] == ds and float(d['budget_s']) == float(budget),
          f'{where}: header')
    check(d['n_instances'] == n and len(d['instances']) == n,
          f'{where}: {d["n_instances"]} instances, expected {n}')
    check_contention_summary(d.get('contention'), where)
    vals = []
    for i, r in enumerate(d['instances']):
        check(r['instance'] == f'instance_{i:03d}', f'{where}: order at {i}')
        check_evidence(r, f'{where} instance {i}')
        vals.append(float(r['makespan']))
    return np.array(vals)


TL_REL = 'test_results/PPVC/training_load.json'


@functools.lru_cache(maxsize=None)
def training_load():
    rows = read_json(os.path.join(REPO, TL_REL))
    out = {}
    for r in rows:
        key = (r['dataset'], float(r['lag_scale']))
        check(key not in out, f'{TL_REL}: duplicate row {key}')
        out[key] = r
    return out


def tl_value(ds, scale, field):
    rows = training_load()
    if (ds, scale) not in rows:
        raise Missing(TL_REL, f'no row for {ds} at lag scale {scale}')
    r = rows[(ds, scale)]
    check(int(r['n']) > 0, f'{TL_REL}: empty row {ds} {scale}')
    return float(r[field])


# ---- uncertain waits ---------------------------------------------------------
STOCH = {  # size: dataset, policy-arm dir, CP-SAT evidence dir, instances
    'Ten': ('10x25+ppvc-mixed', '10x25+ppvc-mixed',
            '10x25+ppvc-mixed/stoch_evidence', 100),
    # 50-module CP-SAT arm: rerun of 2026-10-11 whose nominal plan is CP-SAT
    # started from the SPT schedule with 600 s on 4 workers (the earlier run
    # repaired a cold 60-s plan); the policy arms keep their original files.
    'Fifty': ('50x25+ppvc-mixed', '50x25+ppvc-mixed/stoch20',
              '50x25+ppvc-mixed/stochastic_warm', 20)}
CVS = (('CvOne', 0.1), ('CvTwo', 0.2), ('CvThree', 0.3))
REPS = (0, 1, 2)
STOCH_METHODS = (('React', 'slab_reactive'), ('Repair', 'slab_repair'),
                 ('PerfInfo', 'slab_perfect_info'), ('Spt', 'spt_reactive'),
                 ('CpRepair', 'cpsat_repair'), ('CpRoll', 'cpsat_rolling'))


def stoch_methods(S):
    return [(m, f) for m, f in STOCH_METHODS if S == 'Ten' or m != 'CpRoll']


def stoch_rel(S, method):
    _ds, pdir, cdir, _n = STOCH[S]
    d = cdir if method.startswith('cpsat') else pdir
    return f'test_results/PPVC/{d}/stochastic_lags_{method}.jsonl'


@functools.lru_cache(maxsize=None)
def stoch_nominal(S):
    """The cached nominal CP-SAT plans of the evidence run (one per instance)."""
    ds, _p, cdir, n = STOCH[S]
    where = f'test_results/PPVC/{cdir}/stochastic_lags_cpsat_nominal.jsonl'
    recs = read_jsonl(os.path.join(REPO, where))
    names = [r['instance'] for r in recs]
    check(len(names) == len(set(names)), f'{where}: duplicate instances')
    if len(names) < n:
        raise Missing(where, f'{len(names)} of {n} nominal plans')
    check(set(names) == {f'instance_{i:03d}' for i in range(n)},
          f'{where}: instance set')
    for r in recs:
        t = r.get('timing')
        check(t is not None, f'{where}: plan without timing evidence')
        if t['hit_limit']:
            check(float(t['cpu_share']) >= LOW_CPU_SHARE,
                  f'{where}: {r["instance"]} cpu_share {t["cpu_share"]}')
    return True


@functools.lru_cache(maxsize=None)
def stoch(S, method):
    ds, _p, _c, n = STOCH[S]
    where = stoch_rel(S, method)
    recs = read_jsonl(os.path.join(REPO, where))
    keys = set()
    for r in recs:
        k = (int(r['instance_idx']), round(float(r['cv']), 3), int(r['rep']))
        check(r['dataset'] == ds and r['method'] == method, f'{where}: header')
        check(k not in keys, f'{where}: duplicate {k}')
        keys.add(k)
        check(r['instance'] == f'instance_{k[0]:03d}', f'{where}: name {k}')
        check(0 <= k[0] < n and k[1] in (0.1, 0.2, 0.3) and k[2] in REPS,
              f'{where}: unexpected key {k}')
        check(r['feasible'] is True and int(r['n_violations']) == 0,
              f'{where}: infeasible {k}')
        if method.startswith('slab'):
            check(r['model_name'] == HSTAR_MODEL,
                  f'{where}: model {r["model_name"]}')
        if method == 'spt_reactive':
            check(r['sig'].startswith('SPT|'), f'{where}: sig {r["sig"]}')
        if method.startswith('cpsat'):
            check_evidence(r, f'{where} {k}')
    if len(keys) < n * 9:
        raise Missing(where, f'{len(keys)} of {n * 9} runs')
    lagseeds = {r['sig'].split('|')[-1] for r in recs}
    check(len(lagseeds) == 1, f'{where}: several lag seeds {lagseeds}')
    if method.startswith('cpsat'):
        stoch_nominal(S)
    return tuple(recs)


@functools.lru_cache(maxsize=None)
def stoch_consistency(S):
    """Every present arm of one size covers the same runs with the same
    realized lags (same lag seed)."""
    ref = None
    for _m, f in stoch_methods(S):
        try:
            recs = stoch(S, f)
        except Missing:
            continue
        keys = {(r['instance_idx'], round(r['cv'], 3), r['rep']) for r in recs}
        seed = recs[0]['sig'].split('|')[-1]
        if ref is None:
            ref = (keys, seed, f)
        check(keys == ref[0], f'{S}: {f} and {ref[2]} cover different runs')
        check(seed == ref[1], f'{S}: {f} lag seed {seed} != {ref[1]}')
    return True


def stoch_mean(S, method, cv, field='realized_makespan'):
    recs = stoch(S, method)
    stoch_consistency(S)
    v = [float(r[field]) for r in recs if round(r['cv'], 3) == cv]
    check(len(v) == STOCH[S][3] * len(REPS), f'{S} {method} cv {cv}: {len(v)} runs')
    return float(np.mean(v))


# ---- disruptions ---------------------------------------------------------------
DIS_SIZES = (('Ten', '10x25+ppvc-mixed', 100), ('Fifty', '50x25+ppvc-mixed', 20))
DIS_TYPES = (('Lag', 'lag_perturb'), ('Brk', 'breakdown'),
             ('Urg', 'urgent_arrival'), ('Mat', 'material_shortage'),
             ('Ptd', 'pt_deviation'), ('Can', 'cancellation'))
DIS_METHODS = (('Slab', 'drl_reactive'), ('Spt', 'spt_reactive'),
               ('Ga', 'ga_reactive'), ('Rs', 'right_shift'),
               ('CpFive', 'cpsat_5'), ('CpThirty', 'cpsat_30'))
NERV_METHODS = ('Slab', 'Spt', 'Ga', 'Rs')


def dis_rel(ds, dtype):
    # breakdown: rerun of 2026-10-11 with the right-shift arm keeping every
    # executed operation fixed (the earlier run moved executed operations);
    # the other five types keep their original files.
    sub = 'reactive_hstar_hint' if dtype == 'breakdown' else 'reactive_hstar'
    return (f'test_results/PPVC/{ds}/{sub}/'
            f'reactive_{dtype}_{HSTAR_MODEL}.npy')


@functools.lru_cache(maxsize=None)
def reactive(ds, dtype, n):
    where = dis_rel(ds, dtype)
    path = guard(os.path.join(REPO, where))
    if not os.path.exists(path):
        raise Missing(where, 'file absent')
    d = np.load(path, allow_pickle=True).item()
    meta = d['meta']
    check(meta['model_name'] == HSTAR_MODEL and meta['dataset'] == ds
          and meta['disruption_type'] == dtype, f'{where}: meta')
    check(int(meta['n_instances']) == n,
          f'{where}: {meta["n_instances"]} instances, expected {n}')
    check(float(meta['ga_budget']) == 5.0, f'{where}: GA budget')
    check({5, 30} <= set(int(b) for b in meta['cpsat_budgets']),
          f'{where}: CP-SAT budgets')
    for _m, key in DIS_METHODS:   # only the arms used (cpsat_1 never read)
        rec = d['records'][key]
        check(len(rec['makespan']) == n, f'{where}: {key} length')
        check(bool(np.all(rec['feasible'])), f'{where}: {key} infeasible')
        check(bool(np.all(np.isfinite(rec['makespan']))), f'{where}: {key} NaN')
        check(int(np.sum(rec['n_low_share'])) == 0, f'{where}: {key} low share')
        check(int(np.sum(rec['n_no_evidence'])) == 0,
              f'{where}: {key} no evidence')
        cs = np.asarray(rec['cpu_share_min'], dtype=float)
        cs = cs[np.isfinite(cs)]
        check(cs.size == 0 or float(cs.min()) >= LOW_CPU_SHARE,
              f'{where}: {key} cpu_share_min {cs.min() if cs.size else None}')
    return d


def dis_value(ds, n, dtype, key, field):
    d = reactive(ds, dtype, n)
    v = np.asarray(d['records'][key][field], dtype=float)
    check(bool(np.all(np.isfinite(v))), f'{dis_rel(ds, dtype)}: {key} {field} NaN')
    return float(v.mean())


# ---- buffers ---------------------------------------------------------------------
BUF_NAMES = {None: 'Free', 16: 'Sixteen', 12: 'Twelve', 8: 'Eight', 6: 'Six',
             4: 'Four', 3: 'Three', 2: 'Two'}
BUF = {  # size: dataset, instances, B values, H* file, SPT file, CP-SAT files
    'Ten': dict(
        ds='10x25+ppvc-mixed', n=100, B=(None, 8, 6, 4, 3, 2),
        hstar=f'test_results/PPVC/10x25+ppvc-mixed/'
              f'buffers_masked_{HSTAR1}_shared.jsonl',
        spt='test_results/PPVC/10x25+ppvc-mixed/buffers_masked_SPT_shared.jsonl',
        cp=(('test_results/PPVC/10x25+ppvc-mixed/buffers_ev_a/'
             'buffers_cpsat_shared_t60.jsonl', (None, 8, 6), 60.0),
            ('test_results/PPVC/10x25+ppvc-mixed/buffers_ev_b/'
             'buffers_cpsat_shared_t60.jsonl', (None, 4, 3, 2), 60.0))),
    'Fifty': dict(
        ds='50x25+ppvc-mixed', n=20, B=(None, 16, 12, 8, 6, 4),
        hstar=f'test_results/PPVC/50x25+ppvc-mixed/buffers20/'
              f'buffers_masked_{HSTAR1}_shared.jsonl',
        spt='test_results/PPVC/50x25+ppvc-mixed/buffers20/'
            'buffers_masked_SPT_shared.jsonl',
        cp=(('test_results/PPVC/50x25+ppvc-mixed/buffers20/'
             'buffers_cpsat_shared_t120.jsonl', (None, 16, 12, 8, 6, 4), 120.0),))}


@functools.lru_cache(maxsize=None)
def buffer_records(path_rel, ds, n, expected_b):
    """(instance_id, B) -> record of a buffer JSONL. The file counts as
    present only when every instance 0..n-1 has every expected B."""
    recs = read_jsonl(guard(os.path.join(REPO, path_rel)))
    by = {}
    for r in recs:
        key = (int(r['instance_id']), r['B'])
        check(r['dataset'] == ds and r['buffer'] == 'shared',
              f'{path_rel}: dataset or buffer mode of {key}')
        check(r['instance'] == f'instance_{key[0]:03d}', f'{path_rel}: name {key}')
        check(key not in by, f'{path_rel}: duplicate {key}')
        by[key] = r
    want = {(i, b) for i in range(n) for b in expected_b}
    have = want & set(by)
    if len(have) < len(want):
        raise Missing(path_rel, f'{len(have)} of {len(want)} (instance, B) '
                                f'records')
    check(set(by) == want, f'{path_rel}: unexpected records '
                           f'{sorted(set(by) - want, key=str)[:5]}')
    return by


def buffer_column(path_rel, ds, n, expected_b, B, label=None, time_limit=None):
    by = buffer_records(path_rel, ds, n, tuple(expected_b))
    vals = []
    for i in range(n):
        r, where = by[(i, B)], f'{path_rel} instance {i} B={BUF_NAMES[B]}'
        if label is not None:
            check(r.get('label') == label, f'{where}: label {r.get("label")}')
        if time_limit is not None:
            check(float(r['time_limit_s']) == time_limit, f'{where}: limit')
            check_evidence(r, where)
        if r['objective'] is None:
            raise Missing(path_rel, f'B={BUF_NAMES[B]}: instance {i} unsolved')
        check(r['feasible'] is True and r['within_capacity'] is True,
              f'{where}: infeasible or over capacity')
        vals.append(float(r['objective']))
    return np.array(vals)


# ---- fine-tuning updates -------------------------------------------------------------
def ft_updates():
    vals = []
    for lg in FT_LOGS:
        path = os.path.join(TRAIN_LOG, lg)
        if not os.path.exists(path):
            raise Missing(rel(path), 'file absent')
        with open(path, errors='replace') as f:
            txt = f.read()
        check(f'[init_from] loaded ./trained_network/PPVC/{HSTAR_MODEL}.pth'
              in txt, f'{rel(path)}: not initialised from {HSTAR_MODEL}')
        eps = [int(x) for x in re.findall(r'Episode\s+(\d+)\s', txt)]
        check(len(eps) > 0, f'{rel(path)}: no episode lines')
        vals.append(max(eps))
    check(len(set(vals)) == 1, f'fine-tuning runs differ in updates: {vals}')
    return vals[0]


# --------------------------------------------------------------------------- #
# Exhibits
# --------------------------------------------------------------------------- #
H_MEAN = '3-seed mean of per-seed mean makespan (h)'
H_STD = 'population std of the 3 per-seed mean makespans (h)'
ONE_MEAN = 'mean makespan (h)'


def tpl_note(tpl):
    return H_MEAN if seeded(tpl) else ONE_MEAN


def table2():
    ex = 'Table II necessity'
    pens = []
    for bb, rep, dis in (('Daniel', REPAIR_DANIEL, LAGFREE),
                         ('Plan', REPAIR_PLAN, PLAN_A0),
                         ('Hgnn', HGNN_REPAIR, HGNN_A0)):
        for S, ds in SIZES:
            define(f'Repair{bb}{S}', ex, f'Repair{bb}', S, 'h',
                   lambda r=rep, d=ds: mean_of(r, d), src(rep, ds),
                   'lag-free-trained net, plan then right-shift repair; '
                   + tpl_note(rep))
            define(f'Dispatch{bb}{S}', ex, f'Dispatch{bb}', S, 'h',
                   lambda r=dis, d=ds: mean_of(r, d), src(dis, ds),
                   'same net dispatched in the SLAB transition; '
                   + tpl_note(dis))
            name = f'RepairPen{bb}{S}'
            define(name, ex, f'RepairPen{bb}', S, 'pct',
                   lambda r=rep, s=dis, d=ds: pct(mean_of(r, d), mean_of(s, d)),
                   f'{src(rep, ds)}, {src(dis, ds)}',
                   '100*(mean repair/mean dispatch-1)')
            pens.append(name)

    def lo_hi(f):
        return f([raw_of(p) for p in pens])
    define('RepairPenLo', ex, 'RepairPenLoHi', 'Lo', 'round_int',
           lambda: lo_hi(min), 'RepairPen{Daniel,Plan,Hgnn}{Ten..Eighty}',
           'min over 3 backbones x 5 sizes of the unrounded penalty, rounded')
    define('RepairPenHi', ex, 'RepairPenLoHi', 'Hi', 'round_int',
           lambda: lo_hi(max), 'RepairPen{Daniel,Plan,Hgnn}{Ten..Eighty}',
           'max over 3 backbones x 5 sizes of the unrounded penalty, rounded')


def table3():
    ex = 'Table III production batches'
    for S, ds in SIZES:
        n = N_INST[ds]
        sp, hs = rule_src('SPT', ds), src(HSTAR, ds)
        define(f'Spt{S}', ex, 'Spt', S, 'h',
               lambda d=ds: rule_col('SPT', d).mean(), sp, ONE_MEAN
               + ' of the SPT rule (tie-break seed 50 + i)'
               + rule_note('SPT', ds))
        define(f'Llf{S}', ex, 'Llf', S, 'h',
               lambda d=ds: rule_col('LLF', d).mean(), rule_src('LLF', ds),
               ONE_MEAN + ' of the LLF rule')
        define(f'Ma{S}', ex, 'Ma', S, 'h', lambda d=ds: memetic(d).mean(),
               ma_rel(ds), 'mean best_makespan of the memetic algorithm, '
               '300 s per instance')
        t = cp_budget(S)
        define(f'CpCold{S}', ex, 'CpCold', S, 'h',
               lambda d=ds, s=S: cp_col(d, s, 'cold').mean(),
               cp_src(ds, S, 'cold'), f'mean objective, cold CP-SAT {t} s, 4 workers')
        define(f'CpWarmSpt{S}', ex, 'CpWarmSpt', S, 'h',
               lambda d=ds, s=S: cp_col(d, s, 'warm-spt').mean(),
               cp_src(ds, S, 'warm-spt'),
               f'mean objective, CP-SAT warm-started from SPT {t} s, 4 workers')
        if S in ('Fifty', 'Eighty'):
            wp = cp_src(ds, S, 'warm-policy')
            define(f'CpWarmPol{S}', ex, 'CpWarmPol', S, 'h',
                   lambda d=ds, s=S: cp_col(d, s, 'warm-policy').mean(), wp,
                   'mean objective, CP-SAT warm-started from the H* s301 '
                   'schedule, 600 s, 4 workers')
            define(f'CpWarmPolHint{S}', ex, 'CpWarmPolHint', S, 'h',
                   lambda d=ds, s=S: cp_col(d, s, 'warm-policy',
                                            'hint_makespan').mean(), wp,
                   'mean hint_makespan (the H* s301 schedule given as hint)')
            define(f'CpWarmPolGain{S}', ex, 'CpWarmPolGain', S, 'pct',
                   lambda d=ds, s=S: -pct(
                       cp_col(d, s, 'warm-policy').mean(),
                       cp_col(d, s, 'warm-policy', 'hint_makespan').mean()),
                   wp, '-100*(mean objective/mean hint_makespan-1): % the '
                   'solver shortens the hint')
        define(f'Hstar{S}', ex, 'Hstar', S, 'h',
               lambda d=ds: mean_of(HSTAR, d), hs, H_MEAN)
        define(f'HstarStd{S}', ex, 'HstarStd', S, 'h',
               lambda d=ds: std_of(HSTAR, d), hs, H_STD)
        for rv, fn, rsrc, what in (
                ('Spt', lambda d: rule_col('SPT', d).mean(), sp, 'SPT'),
                ('Ma', lambda d: memetic(d).mean(), ma_rel(ds), 'memetic 300 s'),
                ('CpWarmSpt', lambda d, s=S: cp_col(d, s, 'warm-spt').mean(),
                 cp_src(ds, S, 'warm-spt'), f'CP-SAT warm-SPT {t} s')):
            define(f'HstarVs{rv}{S}', ex, f'HstarVs{rv}', S, 'pct',
                   lambda d=ds, f=fn: pct(mean_of(HSTAR, d), f(d)),
                   f'{hs}, {rsrc}', f'100*(H* 3-seed mean/{what} mean-1)')
            define(f'HstarVs{rv}Gain{S}', ex, f'HstarVs{rv}Gain', S, 'pct',
                   lambda d=ds, f=fn: -pct(mean_of(HSTAR, d), f(d)),
                   f'{hs}, {rsrc}', f'-100*(H* 3-seed mean/{what} mean-1)')
        define(f'HstarWinsSpt{S}', ex, 'HstarWinsSpt', S, 'int',
               lambda d=ds: int(np.sum(inst_of(HSTAR, d) < rule_col('SPT', d))),
               f'{hs}, {sp}', 'instances where the per-instance 3-seed mean '
               'of H* is shorter than SPT')
        define(f'nInst{S}', ex, 'nInst', S, 'int',
               lambda d=ds: len(result(HSTAR.format(s=301), d)), hs,
               f'instances in the test set (asserted {n})')
        if S in ('Fifty', 'Eighty'):
            define(f'ColdAboveSpt{S}', ex, 'ColdAboveSpt', S, 'pct',
                   lambda d=ds, s=S: 100.0 * float(np.mean(
                       cp_col(d, s, 'cold') > rule_col('SPT', d))),
                   f'{cp_src(ds, S, "cold")}, {sp}',
                   '% of instances where cold CP-SAT 600 s ends above SPT')
    ds10 = '10x25+ppvc-mixed'
    define('CpOptTen', ex, 'scalars', 'CpOptTen', 'int',
           lambda: sum(1 for r in anytime(ds10, 'cold', 300, 'cold')
                       if float(r['bound']) == float(r['objective'])),
           any_rel(ds10, 'cold', 300), 'count of records with bound == objective')
    for S, ds in (('Fifty', '50x25+ppvc-mixed'), ('Eighty', '80x25+ppvc-mixed')):
        define(f'CpLongWarm{S}', ex, 'scalars', f'CpLongWarm{S}', 'h',
               lambda d=ds: np.mean([r['objective'] for r in anytime(
                   d, 'warm-spt', 1800, 'warm-spt', n=20)]),
               any_rel(ds, 'warm-spt', 1800),
               'mean objective, CP-SAT warm-SPT 1800 s, first 20 instances')

    def long_n():
        ns = {len(anytime(d, 'warm-spt', 1800, 'warm-spt', n=20))
              for d in ('50x25+ppvc-mixed', '80x25+ppvc-mixed')}
        check(len(ns) == 1, f'1800 s files differ in size {ns}')
        return ns.pop()
    define('CpLongN', ex, 'scalars', 'CpLongN', 'int', long_n,
           f'{any_rel("50x25+ppvc-mixed", "warm-spt", 1800)}, '
           f'{any_rel("80x25+ppvc-mixed", "warm-spt", 1800)}',
           'instances per 1800 s file (ids 0..19 asserted)')
    for name, b in (('GaSixtyTen', 60), ('GaThreeHundredTen', 300)):
        define(name, ex, 'scalars', name, 'h',
               lambda b=b: ga(ds10, b).mean(), ga_rel(ds10, b),
               f'mean makespan of the GA, {b} s per instance')
    exi = 'Table III independent sets'
    for S, ds in SIZES_IND:
        sp, hs, lf = rule_src('SPT', ds), src(HSTAR, ds), src(LAGFREE, ds)
        define(f'SptInd{S}', exi, 'SptInd', S, 'h',
               lambda d=ds: rule_col('SPT', d).mean(), sp, ONE_MEAN + ' of SPT')
        define(f'HstarInd{S}', exi, 'HstarInd', S, 'h',
               lambda d=ds: mean_of(HSTAR, d), hs, H_MEAN)
        define(f'HstarIndVsSpt{S}', exi, 'HstarIndVsSpt', S, 'pct',
               lambda d=ds: pct(mean_of(HSTAR, d), rule_col('SPT', d).mean()),
               f'{hs}, {sp}', '100*(H* 3-seed mean/SPT mean-1)')
        define(f'HstarIndVsSptGain{S}', exi, 'HstarIndVsSptGain', S, 'pct',
               lambda d=ds: -pct(mean_of(HSTAR, d), rule_col('SPT', d).mean()),
               f'{hs}, {sp}', '-100*(H* 3-seed mean/SPT mean-1)')
        define(f'HstarIndWinsSpt{S}', exi, 'HstarIndWinsSpt', S, 'int',
               lambda d=ds: int(np.sum(inst_of(HSTAR, d) < rule_col('SPT', d))),
               f'{hs}, {sp}', 'instances where the per-instance 3-seed mean '
               'of H* is shorter than SPT')
        define(f'LagFreeInd{S}', exi, 'LagFreeInd', S, 'h',
               lambda d=ds: mean_of(LAGFREE, d), lf, H_MEAN)


CONT_CASES = (('TenLagZero', '10x25+ppvc-mixed', 0.0),
              ('TenLagQuarter', '10x25+ppvc-mixed', 0.25),
              ('TenLagHalf', '10x25+ppvc-mixed', 0.5),
              ('TenLagThreeQuarter', '10x25+ppvc-mixed', 0.75),
              ('TenLagFull', '10x25+ppvc-mixed', 1.0),
              ('Twenty', '20x25+ppvc-mixed', 1.0),
              ('Thirty', '30x25+ppvc-mixed', 1.0),
              ('Fifty', '50x25+ppvc-mixed', 1.0),
              ('Eighty', '80x25+ppvc-mixed', 1.0),
              ('Tight', '10x15+ppvc-mixed', 1.0))


def table4():
    # Flow-line restriction solved by CP-SAT (Section S6), as a further
    # solver start at production batches.
    def flow_cp(ds, n):
        rel = f'test_results/PPVC/{ds}/flowline/flowline_per_instance_{ds}.json'
        f = os.path.join(REPO, rel)
        if not os.path.exists(f):
            raise Missing(f'FlowCp{ds}', f'missing {rel}')
        with open(f) as fh:
            d = json.load(fh)
        pi = d['per_instance']
        check(len(pi) == n, f'{rel}: {len(pi)} instances, expected {n}')
        for r in pi:
            check(r['n_low_share'] == 0 and r['n_no_evidence'] == 0,
                  f'{rel}: {r["instance"]} contention evidence')
        return float(np.mean([float(r['cpsat_flowline']) for r in pi]))
    for S, ds, n in (('Fifty', '50x25+ppvc-mixed', 50), ('Eighty', '80x25+ppvc-mixed', 50)):
        define(f'FlowCpLine{S}', 'Table III production batches', 'scalars', f'FlowCpLine{S}', 'h',
               lambda ds=ds, n=n: flow_cp(ds, n),
               f'test_results/PPVC/{ds}/flowline/flowline_per_instance_{ds}.json',
               'mean cpsat_flowline over the orders (cold CP-SAT on the flow-line restriction, 300 s, 2 workers)')

    # Waits relative to processing time in the 10-module test orders (Intro).
    def lag_ratio_mean():
        from data_utils import text_to_matrix
        fs = sorted(glob.glob(os.path.join(DATA, '10x25+ppvc-mixed', 'instance_*.fjs')))
        check(len(fs) == N_INST['10x25+ppvc-mixed'], f'lag ratio: {len(fs)} instances')
        ratios = []
        for f in fs:
            with open(f) as fh:
                _jl, op_pt = text_to_matrix(fh.readlines())
            with open(f.replace('.fjs', '.meta.json')) as fh:
                lags = np.asarray(json.load(fh)['time_lag'], dtype=float)
            check(len(lags) == op_pt.shape[0], f'{f}: lag count')
            pt = np.array([row[row > 0].mean() for row in op_pt], dtype=float)
            ratios.append(lags.sum() / pt.sum())
        return float(np.mean(ratios))
    define('LagRatioMean', 'Intro', 'scalars', 'LagRatioMean', 'pct2', lag_ratio_mean,
           'data/PPVC/10x25+ppvc-mixed/instance_*.fjs and .meta.json',
           'mean over the 100 test orders of (sum of waits)/(sum over operations of the mean processing time over compatible stations)')

    ex = 'Table IV training orders'
    sizes = SIZES + (TIGHT,)
    for stem, tpl in (('TrainLagFree', LAGFREE), ('TrainHalf', HSTAR),
                      ('TrainFull', FULL), ('TrainFullAids', FULLAIDS),
                      ('TrainTight', TIGHTNET), ('PlanHalf', PLANHALF)):
        for S, ds in sizes:
            define(f'{stem}{S}', ex, stem, S, 'h',
                   lambda t=tpl, d=ds: mean_of(t, d), src(tpl, ds), H_MEAN)
            define(f'{stem}Std{S}', ex, f'{stem}Std', S, 'h',
                   lambda t=tpl, d=ds: std_of(t, d), src(tpl, ds), H_STD)
    for stem, tpl in (('PlanLagFree', PLAN_A0), ('PlanFull', PLAN_A1),
                      ('PlanFullAids', PLAN_FULL), ('TrainQuarter', DOSE_Q),
                      ('TrainThreeQuarter', DOSE_TQ)):
        for S, ds in sizes:
            define(f'{stem}{S}', ex, stem, S, 'h',
                   lambda t=tpl, d=ds: mean_of(t, d), src(tpl, ds),
                   ONE_MEAN + ', seed 301')
    for stem, tpl in (('HgnnLagFree', HGNN_A0), ('HgnnFull', HGNN_A1),
                      ('HgnnFullVs', HGNN_A1VS), ('HgnnHalf', HGNN_HALF)):
        for S, ds in SIZES:
            define(f'{stem}{S}', ex, stem, S, 'h',
                   lambda t=tpl, d=ds: mean_of(t, d), src(tpl, ds),
                   ONE_MEAN + ', seed 300')
    exc = 'Table IV contention shares'
    for prefix, field in (('Cont', 'queue_share'), ('Util', 'util_mean')):
        for suffix, ds, scale in CONT_CASES:
            define(f'{prefix}{suffix}', exc, prefix, suffix, 'share',
                   lambda d=ds, s=scale, f=field: tl_value(d, s, f), TL_REL,
                   f'{field} of SPT schedules, {ds}, lags x{scale:g}, '
                   f'first 20 instances')
    for S, ds in sizes:
        srcs = f'{src(LFBOUND, ds)}, {src(FULL, ds)}'

        def boot(d=ds):
            return paired_mean_diff_ci(inst_of(LFBOUND, d), inst_of(FULL, d),
                                       n_boot=2000, seed=0)
        comp = ('per-instance 3-seed means, lag-free bound minus lag-augmented '
                'bound (h); 95% percentile bootstrap, 2000 resamples, '
                'numpy default_rng(0) (stats_tests.paired_mean_diff_ci)')
        define(f'BoundDiff{S}', ex, 'BoundDiff', S, 'h',
               lambda b=boot: b()[0], srcs, 'mean paired difference, ' + comp)
        define(f'BoundCiLo{S}', ex, 'BoundCiLo', S, 'h',
               lambda b=boot: b()[1][0], srcs, 'CI low, ' + comp)
        define(f'BoundCiHi{S}', ex, 'BoundCiHi', S, 'h',
               lambda b=boot: b()[1][1], srcs, 'CI high, ' + comp)


def table5():
    ex = 'Table V uncertain waits'
    for S in ('Ten', 'Fifty'):
        for M, method in stoch_methods(S):
            for cvn, cv in CVS:
                define(f'Stoch{M}{S}{cvn}', ex, f'Stoch{M}{S}', cvn, 'h',
                       lambda s=S, m=method, c=cv: stoch_mean(s, m, c),
                       stoch_rel(S, method),
                       f'mean realized_makespan over instances x 3 '
                       f'replications at cv {cv} (h)')

    def gap_max():
        # computed from the one-decimal values that Table V prints, so a
        # reader who recomputes the gap from the table gets the same number
        gaps = [pct(fmt('h', stoch_mean(S, 'slab_reactive', cv)),
                    fmt('h', stoch_mean(S, 'slab_perfect_info', cv)))
                for S in ('Ten', 'Fifty') for _n, cv in CVS]
        return max(gaps)
    define('StochGapPiMax', ex, 'scalars', 'StochGapPiMax', 'pct2', gap_max,
           ', '.join(stoch_rel(S, m) for S in ('Ten', 'Fifty')
                     for m in ('slab_reactive', 'slab_perfect_info')),
           'max over cv and size of 100*(mean reactive/mean perfect '
           'information-1), from the one-decimal table values')

    def replan():
        recs = stoch('Ten', 'cpsat_rolling')
        stoch_consistency('Ten')
        return float(np.mean([r['replan_seconds'] for r in recs]))
    define('StochCpRollReplanS', ex, 'scalars', 'StochCpRollReplanS', 's',
           replan, stoch_rel('Ten', 'cpsat_rolling'),
           'mean replan_seconds (total re-planning seconds of one closed-loop '
           'run) over all cv and replications')


def table6():
    ex = 'Table VI disruptions'
    for S, ds, n in DIS_SIZES:
        for T, dtype in DIS_TYPES:
            for M, key in DIS_METHODS:
                define(f'Dis{T}{M}{S}', ex, f'Dis{T}{M}', S, 'h',
                       lambda d=ds, n=n, t=dtype, k=key:
                           dis_value(d, n, t, k, 'makespan'),
                       dis_rel(ds, dtype),
                       f'mean realized makespan of {key} over {n} instances (h)')
        allsrc = (f'test_results/PPVC/{ds}/reactive_hstar/reactive_'
                  '{lag_perturb,breakdown,urgent_arrival,material_shortage,'
                  f'pt_deviation,cancellation}}_{HSTAR_MODEL}.npy')
        for M, key in DIS_METHODS:
            if M in NERV_METHODS:
                define(f'DisNerv{M}{S}', ex, f'DisNerv{M}', S, 'h',
                       lambda d=ds, n=n, k=key: float(np.mean(
                           [dis_value(d, n, t, k, 'comb_nerv')
                            for _T, t in DIS_TYPES])), allsrc,
                       f'mean over the 6 types of the mean comb_nerv of {key} '
                       '(operations, 1 decimal)')
            define(f'DisTime{M}{S}', ex, f'DisTime{M}', S, 's',
                   lambda d=ds, n=n, k=key: float(np.mean(
                       [dis_value(d, n, t, k, 'seconds')
                        for _T, t in DIS_TYPES])), allsrc,
                   f'mean over the 6 types of the mean re-planning seconds '
                   f'of {key}')


def table7():
    ex = 'Table VII deployment limits'
    for S, cfg in BUF.items():
        ds, n, bl = cfg['ds'], cfg['n'], cfg['B']
        for B in bl:
            bn = BUF_NAMES[B]
            define(f'BufHstar{S}{bn}', ex, f'BufHstar{S}', bn, 'h',
                   lambda c=cfg, b=B: buffer_column(
                       c['hstar'], c['ds'], c['n'], c['B'], b,
                       label=HSTAR1).mean(), cfg['hstar'],
                   f'mean objective of buffer-masked H* s301, shared cap '
                   f'{bn}, {n} instances')
            define(f'BufSpt{S}{bn}', ex, f'BufSpt{S}', bn, 'h',
                   lambda c=cfg, b=B: buffer_column(
                       c['spt'], c['ds'], c['n'], c['B'], b,
                       label='SPT').mean(), cfg['spt'],
                   f'mean objective of buffer-masked SPT, shared cap {bn}, '
                   f'{n} instances')
            cp = [c for c in cfg['cp'] if B in c[1]][0]
            define(f'BufCp{S}{bn}', ex, f'BufCp{S}', bn, 'h',
                   lambda c=cfg, f=cp, b=B: buffer_column(
                       f[0], c['ds'], c['n'], f[1], b,
                       time_limit=f[2]).mean(), cp[0],
                   f'mean objective of CP-SAT {cp[2]:g} s with the shared cap '
                   f'{bn}, {n} instances')
        define(f'BufPeakFree{S}', ex, 'scalars', f'BufPeakFree{S}', 'median',
               lambda c=cfg: float(np.median([
                   buffer_records(c['hstar'], c['ds'], c['n'], c['B'])[
                       (i, None)]['occupancy']['peak'] for i in range(c['n'])])),
               cfg['hstar'], 'median over instances of the peak shared-buffer '
               'occupancy of unconstrained H* s301')
    exv = 'Table VII variable routes'
    for S, ds in SIZES_VAR:
        define(f'VarSpt{S}', exv, 'VarSpt', S, 'h',
               lambda d=ds: rule_col('SPT', d).mean(), rule_src('SPT', ds),
               ONE_MEAN + ' of SPT')
        define(f'VarHstarZs{S}', exv, 'VarHstarZs', S, 'h',
               lambda d=ds: mean_of(HSTAR1, d), src(HSTAR1, ds),
               ONE_MEAN + ', H* s301 zero-shot')
        define(f'VarTrained{S}', exv, 'VarTrained', S, 'h',
               lambda d=ds: mean_of(VAR_TRAINED, d), src(VAR_TRAINED, ds),
               ONE_MEAN + ', trained on variable routes, s301')
    exf = 'Table VII fine-tuning'
    tds = TIGHT[1]
    define('FtTightZs', exf, 'FtTight', 'Zs', 'h',
           lambda: mean_of(HSTAR1, tds), src(HSTAR1, tds),
           ONE_MEAN + ', H* s301 zero-shot on the 15-station factory')
    define('FtTightAfter', exf, 'FtTight', 'After', 'h',
           lambda: mean_of(FT_TIGHT, tds), src(FT_TIGHT, tds),
           ONE_MEAN + ', H* s301 fine-tuned to the 15-station factory')
    define('FtTightScratch', exf, 'FtTight', 'Scratch', 'h',
           lambda: mean_of(TIGHTNET, tds), src(TIGHTNET, tds),
           H_MEAN + ' (= TrainTightTight)')
    for S, ds in SIZES_LAG15:
        define(f'FtLagZs{S}', exf, 'FtLagZs', S, 'h',
               lambda d=ds: mean_of(HSTAR1, d), src(HSTAR1, ds),
               ONE_MEAN + ', H* s301 zero-shot, lags x1.5')
        define(f'FtLagAfter{S}', exf, 'FtLagAfter', S, 'h',
               lambda d=ds: mean_of(FT_LAG, d), src(FT_LAG, ds),
               ONE_MEAN + ', H* s301 fine-tuned to lags x1.5')
        define(f'FtLagSpt{S}', exf, 'FtLagSpt', S, 'h',
               lambda d=ds: rule_col('SPT', d).mean(), rule_src('SPT', ds),
               ONE_MEAN + ' of SPT, lags x1.5')
    define('FtUpdates', exf, 'scalars', 'FtUpdates', 'int', ft_updates,
           ', '.join(f'train_log/{lg}' for lg in FT_LOGS),
           'last PPO update of both fine-tuning runs (init from H* s301 '
           'asserted)')


RIVALS = (('Spt', 'SPT'), ('Ma', 'memetic 300 s'),
          ('CpWarm', 'CP-SAT warm-SPT'))


def rival_inst(R, S, ds):
    if R == 'Spt':
        return rule_col('SPT', ds)
    if R == 'Ma':
        return memetic(ds)
    return cp_col(ds, S, 'warm-spt')


def rival_src(R, S, ds):
    if R == 'Spt':
        return rule_src('SPT', ds)
    if R == 'Ma':
        return ma_rel(ds)
    return cp_src(ds, S, 'warm-spt')


@functools.lru_cache(maxsize=None)
def pval(R, S, ds):
    return wilcoxon_paired(inst_of(HSTAR, ds), rival_inst(R, S, ds),
                           zero_method='wilcox')


@functools.lru_cache(maxsize=None)
def holm_family(R):
    adj, _rej = holm([pval(R, S, ds) for S, ds in SIZES])
    return tuple(float(x) for x in adj)


def stats():
    ex = 'Statistics (supplement)'
    for R, what in RIVALS:
        for k, (S, ds) in enumerate(SIZES):
            srcs = f'{src(HSTAR, ds)}, {rival_src(R, S, ds)}'
            define(f'Pval{R}{S}', ex, f'Pval{R}', S, 'p',
                   lambda r=R, s=S, d=ds: pval(r, s, d), srcs,
                   f'two-sided Wilcoxon signed-rank, H* per-instance 3-seed '
                   f'mean vs {what}, zeros dropped (stats_tests.wilcoxon_paired)')
            define(f'Rbis{R}{S}', ex, f'Rbis{R}', S, 'r',
                   lambda r=R, s=S, d=ds: rank_biserial(
                       inst_of(HSTAR, d), rival_inst(r, s, d)), srcs,
                   f'matched-pairs rank-biserial r of H* minus {what} '
                   '(r < 0 favours H*; stats_tests.rank_biserial)')
            define(f'Pval{R}Holm{S}', ex, f'Pval{R}Holm', S, 'p',
                   lambda r=R, i=k: holm_family(r)[i],
                   f'Pval{R}{{Ten..Eighty}}',
                   'Holm-adjusted over the 5 sizes of this rival')


# --------------------------------------------------------------------------- #
# Global checks, self-checks, collisions
# --------------------------------------------------------------------------- #
def check_data_counts():
    for ds, n in N_INST.items():
        k = len(glob.glob(os.path.join(DATA, ds, 'instance_*.fjs')))
        check(k == n, f'data/PPVC/{ds}: {k} instances, expected {n}')


def self_checks():
    """Recompute five headline numbers by a separate path: direct file reads,
    no loader or cache of this script."""
    out = []

    def one(name, kind, fn, how):
        e = REG.get(name)
        if e is None or e['pending'] or e['failed']:
            out.append(dict(name=name, status='skipped (pending or failed)',
                            how=how))
            return
        try:
            v = float(fn())
        except (OSError, ValueError, KeyError, AssertionError) as err:
            out.append(dict(name=name, status=f'FAILED ({err})', how=how))
            FAILS.append((name, f'self-check could not read: {err}'))
            return
        ok = fmt(kind, v) == e['value'] and abs(v - e['raw']) < 1e-9
        out.append(dict(name=name, macro=e['value'], independent=fmt(kind, v),
                        status='ok' if ok else 'MISMATCH', how=how))
        if not ok:
            FAILS.append((name, f'self-check mismatch {e["value"]} vs '
                                f'{fmt(kind, v)}'))

    def hstar_fifty():
        fs = sorted(glob.glob(os.path.join(
            TR, '50x25+ppvc-mixed', 'Result_greedy+10x25+ppvc-mixed+'
            'a1lagx05-2k-s30[123]_50x25+ppvc-mixed.npy')))
        check(len(fs) == 3, f'{len(fs)} H* seed files at 50x25')
        return np.mean([np.load(f)[:, 0].mean() for f in fs])

    def spt_eighty():
        return np.load(os.path.join(
            TR, '80x25+ppvc-mixed', 'Result_SPT+pdr_80x25+ppvc-mixed.npy'))[:, 0].mean()

    def warm_spt_eighty():
        with open(os.path.join(AT, '80x25+ppvc-mixed_warm-spt_600s_w4.jsonl')) as f:
            v = [json.loads(x)['objective'] for x in f if x.strip()]
        check(len(v) == 50, f'{len(v)} warm-SPT records at 80x25')
        return np.mean(v)

    def pen_daniel_eighty():
        d = os.path.join(TR, '80x25+ppvc-mixed')
        rep = sorted(glob.glob(os.path.join(
            d, 'Result_A0REPAIR+10x25+ppvc-mixed+a0-2k-s30[123]_80x25+ppvc-mixed.npy')))
        dis = sorted(glob.glob(os.path.join(
            d, 'Result_greedy+10x25+ppvc-mixed+a0-2k-s30[123]_80x25+ppvc-mixed.npy')))
        check(len(rep) == 3 and len(dis) == 3, 'seed files at 80x25')
        r = np.mean([np.load(f)[:, 0].mean() for f in rep])
        g = np.mean([np.load(f)[:, 0].mean() for f in dis])
        return 100.0 * (r / g - 1.0)

    def react_fifty_cv3():
        p = os.path.join(TR, '50x25+ppvc-mixed', 'stoch20',
                         'stochastic_lags_slab_reactive.jsonl')
        with open(p) as f:
            rows = [json.loads(x) for x in f if x.strip()]
        v = [r['realized_makespan'] for r in rows if abs(r['cv'] - 0.3) < 1e-9]
        check(len(v) == 60, f'{len(v)} cv 0.3 runs at 50x25')
        return np.mean(v)

    one('HstarFifty', 'h', hstar_fifty, 'glob of the 3 seed files + np.load')
    one('SptEighty', 'h', spt_eighty, 'np.load of the SPT rule file')
    one('CpWarmSptEighty', 'h', warm_spt_eighty, 'plain JSONL parse')
    one('RepairPenDanielEighty', 'pct', pen_daniel_eighty,
        'glob of repair and dispatch seed files + np.load')
    one('StochReactFiftyCvThree', 'h', react_fifty_cv3, 'plain JSONL parse')
    return out


def collisions(out_dir):
    names = set()
    for fn in EXISTING_MACRO_FILES:
        p = os.path.join(out_dir, fn)
        if not os.path.exists(p):
            p = os.path.join(OUT_DIR, fn)
        if os.path.exists(p):
            with open(p) as f:
                names |= set(re.findall(
                    r'\\(?:newcommand|renewcommand|providecommand)\*?\{\\([A-Za-z]+)\}',
                    f.read()))
    return sorted(set(REG) & names)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def tex_value(e):
    return e['value'] if not e['pending'] else f'\\pend{{{e["name"]}}}'


def write_tex(path, mode, stamp):
    lines = ['% =====================================================================',
             '% macros_results.tex: every result number of the manuscript.',
             '% Generated by make_paper_macros.py from the result files; do not',
             '% edit by hand. Each line names its source file(s) and computation.',
             f'% Generated {stamp}, mode {mode}; '
             f'{sum(1 for e in REG.values() if not e["pending"])} values, '
             f'{sum(1 for e in REG.values() if e["pending"])} pending.',
             '% =====================================================================',
             '\\providecommand{\\pend}[1]{\\textbf{[#1]}}', '']
    last = None
    for e in REG.values():
        if e['exhibit'] != last:
            lines += ['', f'% ---- {e["exhibit"]} ----']
            last = e['exhibit']
        note = e['computation']
        if e['pending']:
            note += f'; PENDING: {e["reason"]}'
        lines.append(f'\\newcommand{{\\{e["name"]}}}{{{tex_value(e)}}} '
                     f'% source: {e["source"]}; {note}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def write_json(path):
    out = OrderedDict()
    for e in REG.values():
        out[e['name']] = dict(value=e['value'], raw=e['raw'],
                              pending=e['pending'], reason=e['reason'],
                              source=e['source'], computation=e['computation'],
                              exhibit=e['exhibit'])
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)


COL_ORDER = ([S for S, _ in SIZES] + ['Tight']
             + [suffix for suffix, _d, _s in CONT_CASES]
             + [BUF_NAMES[b] for b in (None, 16, 12, 8, 6, 4, 3, 2)]
             + [c for c, _ in CVS] + ['Zs', 'After', 'Scratch', 'Lo', 'Hi'])


def preview_tables():
    lines = []
    by_ex = OrderedDict()
    for e in REG.values():
        by_ex.setdefault(e['exhibit'], []).append(e)
    for ex, es in by_ex.items():
        lines += ['', f'## {ex}', '']
        grid = [e for e in es if e['row'] != 'scalars']
        scal = [e for e in es if e['row'] == 'scalars']
        if grid:
            rows, cols = OrderedDict(), OrderedDict()
            for e in grid:
                rows.setdefault(e['row'], {})[e['col']] = e
                cols.setdefault(e['col'], None)
            cols = sorted(cols, key=lambda c: (COL_ORDER.index(c)
                                               if c in COL_ORDER else 999))
            lines.append('| row | ' + ' | '.join(cols) + ' |')
            lines.append('|---' * (len(cols) + 1) + '|')
            for r, cells in rows.items():
                vals = [('PEND' if cells[c]['pending'] else
                         'FAIL' if cells[c]['failed'] else cells[c]['value'])
                        if c in cells else '' for c in cols]
                lines.append(f'| {r} | ' + ' | '.join(vals) + ' |')
        if scal:
            lines += ['', '| macro | value |', '|---|---|']
            for e in scal:
                v = 'PEND' if e['pending'] else 'FAIL' if e['failed'] else e['value']
                lines.append(f'| {e["name"]} | {v} |')
    return lines


def write_preview(path, mode, stamp, checks, coll):
    pend = [e for e in REG.values() if e['pending']]
    lines = ['# Result macros: preview', '',
             f'Generated {stamp} by make_paper_macros.py (mode {mode}). '
             f'{len(REG)} macros: {len(REG) - len(pend)} values, '
             f'{len(pend)} pending.', '',
             '## Self-checks (independent recomputation)', '',
             '| macro | macro value | independent value | status | path |',
             '|---|---|---|---|---|']
    for c in checks:
        lines.append(f'| {c["name"]} | {c.get("macro", "")} | '
                     f'{c.get("independent", "")} | {c["status"]} | {c["how"]} |')
    lines += ['', '## Name collisions with macros.tex / macros_pending.tex', '',
              ', '.join(coll) if coll else 'none']
    lines += ['', '## Pending macros', '']
    if pend:
        lines += ['| macro | reason |', '|---|---|']
        lines += [f'| {e["name"]} | {e["reason"]} |' for e in pend]
    else:
        lines.append('none')
    lines += preview_tables()
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def reset():
    REG.clear()
    del FAILS[:]
    for fn in (result, rule_col, anytime, memetic, ga, training_load,
               stoch_nominal, stoch, stoch_consistency, reactive,
               buffer_records, pval, holm_family):
        fn.cache_clear()


def extras():
    """Derived macros requested by the prose (frame draft 2026-10-10)."""
    ex = 'Derived (prose)'
    for S in ('Ten', 'Twenty', 'Thirty', 'Fifty', 'Eighty'):
        define(f'FullVsHalf{S}', ex, 'FullVsHalf', S, 'pct',
               lambda S=S: pct(raw_of(f'TrainFull{S}'), raw_of(f'TrainHalf{S}')),
               'TrainFull, TrainHalf', '100*(full waits / halved waits - 1), same network, 3-seed means')
        define(f'BoundGain{S}', ex, 'BoundGain', S, 'h',
               lambda S=S: -raw_of(f'BoundDiff{S}'), 'BoundDiff',
               'minus BoundDiff: hours by which the lag-free bound is shorter')
    for S, ds in (('Fifty', '50x25+ppvc-mixed'), ('Eighty', '80x25+ppvc-mixed')):
        define(f'LongSlab{S}', ex, 'LongSlab', S, 'h',
               lambda d=ds: float(np.mean(inst_of(HSTAR, d)[:20])), src(HSTAR, ds),
               'mean over the first 20 instances of the per-instance 3-seed mean (pairs with CpLongWarm)')
    stab = 'test_results/PPVC/10x25+ppvc-mixed/reactive_hstar/stability_sweep_lag_perturb.npy'

    def stab_payload():
        if not os.path.exists(stab):
            raise Missing('StabNervTen', f'missing {stab}')
        d = np.load(stab, allow_pickle=True).item()
        check(int(d['n']) == 30, f'stability sweep n={d["n"]}')
        for w, c in d['contention'].items():
            check(c['n_low_share'] == 0 and c['n_no_evidence'] == 0, f'stability contention w={w}')
        return d
    define('StabNervTen', ex, 'scalars', 'StabNervTen', 'h',
           lambda: float(np.mean(stab_payload()['frontier'][0.5]['nerv'])), stab,
           'mean combined nervousness, stability-aware residual CP-SAT w=0.5, lag perturbation, 30 inst')
    define('SlabNervTen', ex, 'scalars', 'SlabNervTen', 'h',
           lambda: float(np.mean(stab_payload()['drl']['nerv'])), stab,
           'mean combined nervousness, SLAB re-planning, same 30 disrupted instances')
    define('StabInflTen', ex, 'scalars', 'StabInflTen', 'pct',
           lambda: float(np.mean(stab_payload()['frontier'][0.5]['infl'])), stab,
           'mean makespan inflation %, stability-aware CP-SAT w=0.5')
    define('SlabInflTen', ex, 'scalars', 'SlabInflTen', 'pct',
           lambda: float(np.mean(stab_payload()['drl']['infl'])), stab,
           'mean makespan inflation %, SLAB re-planning, same instances')

    def stab_times():
        secs = []
        for t in ('breakdown', 'lag_perturb'):
            f = f'test_results/PPVC/10x25+ppvc-mixed/reactive_hstar/stability_sweep_{t}.npy'
            if not os.path.exists(f):
                raise Missing('StabTimeMaxTen', f'missing {f}')
            d = np.load(f, allow_pickle=True).item()
            for w, fr in d['frontier'].items():
                check(fr['optimal'] == d['n'], f'stability {t} w={w}: not all optimal')
                secs.extend(float(x) for x in fr['sec'])
        return secs
    define('StabTimeMaxTen', ex, 'scalars', 'StabTimeMaxTen', 'h',
           lambda: max(stab_times()),
           'reactive_hstar/stability_sweep_{breakdown,lag_perturb}.npy',
           'max residual CP-SAT solve seconds over both sweeps and all weights (all proven optimal)')
    define('StabTimeMeanTen', ex, 'scalars', 'StabTimeMeanTen', 'h',
           lambda: float(np.mean(stab_times())),
           'reactive_hstar/stability_sweep_{breakdown,lag_perturb}.npy',
           'mean residual CP-SAT solve seconds over both sweeps and all weights')

    def flow_gains():
        ds = '10x25+ppvc-mixed'
        f = f'test_results/PPVC/{ds}/flowline/flowline_per_instance_{ds}.json'
        if not os.path.exists(f):
            raise Missing('FlowGainSptTen', f'missing {f}')
        d = read_json(f)
        pi = d['per_instance']
        check(len(pi) == 100, 'flowline 10x25 n')
        check(all(r['cpsat_flowline_status'] == 'OPTIMAL'
                  and float(r['cpsat_flowline_bound']) == float(r['cpsat_flowline']) for r in pi),
              'flowline 10x25: every flow-line solve must be proven optimal (timing-independent)')
        spt_flow = np.array([r['spt_flowline'] for r in pi], float)
        spt_flex = np.array([r['spt_flexible'] for r in pi], float)
        cp_flow = np.array([r['cpsat_flowline'] for r in pi], float)
        cold = {r['instance_id'] if 'instance_id' in r else i: float(r['objective'])
                for i, r in enumerate(anytime(ds, 'cold', 300, 'cold'))}
        cp_flex = np.array([cold[i] for i in range(100)], float)
        return (float(np.mean(100 * (spt_flow - spt_flex) / spt_flow)),
                float(np.mean(100 * (cp_flow - cp_flex) / cp_flow)))
    define('FlowGainSptTen', ex, 'scalars', 'FlowGainSptTen', 'pct', lambda: flow_gains()[0],
           'flowline_per_instance_10x25+ppvc-mixed.json',
           'mean per-order 100*(flow line - flexible)/flow line, SPT rule, 10 modules')
    define('FlowGainCpTen', ex, 'scalars', 'FlowGainCpTen', 'pct', lambda: flow_gains()[1],
           'flowline_per_instance_10x25 + anytime cold 300 s',
           'mean per-order 100*(flow-line CP-SAT optimum - flexible CP-SAT 300 s)/flow line, 10 modules')

    TYPES = (('Lag', 'lag_perturb'), ('Brk', 'breakdown'), ('Urg', 'urgent_arrival'),
             ('Mat', 'material_shortage'), ('Ptd', 'pt_deviation'), ('Can', 'cancellation'))

    def resid(S, t):
        ds = {'Ten': '10x25+ppvc-mixed', 'Fifty': '50x25+ppvc-mixed'}[S]
        # hinted run of 2026-10-11 (right-shifted plan as AddHint); the cold run stays in reactive_hstar/
        f = f'test_results/PPVC/{ds}/reactive_hstar_hint/residual_cpsat_{t}.npy'
        if not os.path.exists(f):
            raise Missing(f'Res{S}', f'missing {f}')
        d = np.load(f, allow_pickle=True).item()
        check(d['meta']['model_name'] == HSTAR_MODEL, f'{f}: model {d["meta"]["model_name"]}')
        check(int(d['meta']['n_instances']) == (100 if S == 'Ten' else 20), f'{f}: n')
        check(int(d['meta']['cpsat_workers']) == 2, f'{f}: workers')
        for k, c in d['contention'].items():
            check(c.get('n_low_share', 0) == 0 and c.get('n_no_evidence', 0) == 0,
                  f'{f}: contention {k}')
        return d

    def res_mean(S, t, key):
        r = resid(S, t)['records'][key]
        check(all(bool(v) for v in r['feasible']), f'residual {S} {t} {key}: infeasible record')
        return float(np.mean(np.asarray(r['makespan'], dtype=float)))
    for S in ('Ten', 'Fifty'):
        for T, t in TYPES:
            for B, key in (('Five', 'residual_cpsat_5s'), ('Thirty', 'residual_cpsat_30s')):
                define(f'Dis{T}Res{B}{S}', ex, f'DisRes{B}{S}', T, 'h',
                       lambda S=S, t=t, k=key: res_mean(S, t, k),
                       f'reactive_hstar/residual_cpsat_{t}.npy',
                       f'mean makespan, residual CP-SAT {B} s re-plan keeping committed operations, 2 workers')
    def res_field_mean(S, key, field):
        return float(np.mean([np.mean(np.asarray(resid(S, t)['records'][key][field], dtype=float))
                              for _, t in TYPES]))
    for S in ('Ten', 'Fifty'):
        for B, key in (('Five', 'residual_cpsat_5s'), ('Thirty', 'residual_cpsat_30s')):
            define(f'DisNervRes{B}{S}', ex, f'DisNervRes{B}', S, 'h',
                   lambda S=S, k=key: res_field_mean(S, k, 'comb_nerv'),
                   'reactive_hstar/residual_cpsat_*.npy', 'mean over six types of mean comb_nerv (station reassignments + starts moved > 1 h)')
            define(f'DisTimeRes{B}{S}', ex, f'DisTimeRes{B}', S, 's',
                   lambda S=S, k=key: res_field_mean(S, k, 'seconds'),
                   'reactive_hstar/residual_cpsat_*.npy', 'mean over six types of mean re-plan seconds')
    define('ResGapMinFifty', ex, 'scalars', 'ResGapMinFifty', 'pct',
           lambda: min(pct(shown(f'Dis{T}ResThirtyFifty'), shown(f'Dis{T}SlabFifty')) for T, _ in TYPES),
           'Dis*ResThirtyFifty, Dis*SlabFifty', 'min over types of 100*(residual CP-SAT 30 s / SLAB re-planning - 1), 50 modules')
    define('ResGapMaxFifty', ex, 'scalars', 'ResGapMaxFifty', 'pct',
           lambda: max(pct(shown(f'Dis{T}ResThirtyFifty'), shown(f'Dis{T}SlabFifty')) for T, _ in TYPES),
           'Dis*ResThirtyFifty, Dis*SlabFifty', 'max over types of 100*(residual CP-SAT 30 s / SLAB re-planning - 1), 50 modules, from the one-decimal table values')
    define('ResFiveGapMaxFifty', ex, 'scalars', 'ResFiveGapMaxFifty', 'pct',
           lambda: max(pct(shown(f'Dis{T}ResFiveFifty'), shown(f'Dis{T}SlabFifty')) for T, _ in TYPES),
           'Dis*ResFiveFifty, Dis*SlabFifty', 'max over types of 100*(residual CP-SAT 5 s / SLAB re-planning - 1), 50 modules, from the one-decimal table values')
    define('ResFiveGapMinFifty', ex, 'scalars', 'ResFiveGapMinFifty', 'pct',
           lambda: min(pct(shown(f'Dis{T}ResFiveFifty'), shown(f'Dis{T}SlabFifty')) for T, _ in TYPES),
           'Dis*ResFiveFifty, Dis*SlabFifty', 'min over types of 100*(residual CP-SAT 5 s / SLAB re-planning - 1), 50 modules, from the one-decimal table values')
    define('ResOptMinTen', ex, 'scalars', 'ResOptMinTen', 'int',
           lambda: min(int(resid('Ten', t)['optimal_count'][5]) for _, t in TYPES),
           'reactive_hstar/residual_cpsat_*.npy (10x25)', 'min over types of orders re-planned to proven optimality within 5 s (of 100)')
    define('ResGainMaxTen', ex, 'scalars', 'ResGainMaxTen', 'h',
           lambda: max(shown(f'Dis{T}SlabTen') - shown(f'Dis{T}ResFiveTen') for T, _ in TYPES),
           'Dis*SlabTen, Dis*ResFiveTen', 'max over types of SLAB minus residual CP-SAT 5 s (h), 10 modules, from the one-decimal table values')
    define('ResGainMinTen', ex, 'scalars', 'ResGainMinTen', 'h',
           lambda: min(shown(f'Dis{T}SlabTen') - shown(f'Dis{T}ResFiveTen') for T, _ in TYPES),
           'Dis*SlabTen, Dis*ResFiveTen', 'min over types of SLAB minus residual CP-SAT 5 s (h), 10 modules, from the one-decimal table values')

    COST = os.environ.get('COST_TABLE', 'paper_inputs/cost_table_v3.json')
    QSTAT = 'train_log/quiet_cost.status'

    def cost_cpu(ds):
        if not os.path.exists(COST):
            raise Missing('LatSlab', f'missing {COST}')
        d = read_json(COST)
        return d['cpu'][HSTAR_MODEL][ds]

    def probe():
        if not os.path.exists(QSTAT):
            raise Missing('TrainGpuHours', f'missing {QSTAT}')
        txt = open(QSTAT).read()
        m = re.search(r'train 200 updates exit 0 wall_s ([0-9.]+)', txt)
        check(m is not None and 'done' in txt, 'quiet-window training probe not finished')
        return float(m.group(1)) / 200.0
    for S, ds in (('Ten', '10x25'), ('Twenty', '20x25'), ('Thirty', '30x25'), ('Fifty', '50x25'), ('Eighty', '80x25')):
        define(f'LatSlab{S}', ex, 'LatSlab', S, 'h', lambda ds=ds: cost_cpu(ds)['latency_s_per_instance_mean'], COST,
               'mean seconds per order, greedy SLAB (seed 301) on one efficiency core, one torch thread, quiet machine')
    define('LatSlabMemMb', ex, 'scalars', 'LatSlabMemMb', 'round_int',
           lambda: max(cost_cpu(ds)['peak_rss_MB'] for ds in ('10x25', '20x25', '30x25', '50x25', '80x25')), COST,
           'max peak resident memory (MB) of the CPU latency runs')
    define('TrainSecPerUpdate', ex, 'scalars', 'TrainSecPerUpdate', 'h', probe, QSTAT,
           'seconds per PPO update incl. validation, 200-update probe of the training recipe on the quiet GPU')
    define('TrainGpuHours', ex, 'scalars', 'TrainGpuHours', 'h', lambda: probe() * 2000 / 3600, QSTAT,
           '2000 updates x seconds per update / 3600')
    define('FtGpuMinutes', ex, 'scalars', 'FtGpuMinutes', 'round_int', lambda: probe() * 400 / 60, QSTAT,
           '400 updates x seconds per update / 60')

    def gpu_peak():
        f = 'train_log/quiet_cost_gpu_samples.csv'
        if not os.path.exists(f):
            raise Missing('TrainGpuMemGb', f'missing {f}')
        vals = [float(l.split(',')[0]) for l in open(f) if l.strip()]
        return max(vals) / 1024.0
    define('TrainGpuMemGb', ex, 'scalars', 'TrainGpuMemGb', 'h', gpu_peak, 'train_log/quiet_cost_gpu_samples.csv',
           'peak card memory (GiB) during the 200-update probe, whole card, nothing else on it')

    def cp_gap_min():
        gaps = []
        # the disruption study's whole-order CP-SAT arms are references, not re-planners: excluded
        for cvn in ('CvOne', 'CvTwo', 'CvThree'):
            gaps.append(pct(raw_of(f'StochCpRepairFifty{cvn}'), raw_of(f'StochReactFifty{cvn}')))
        return min(gaps)
    define('CpGapMinFifty', ex, 'scalars', 'CpGapMinFifty', 'round_int', cp_gap_min,
           'StochCpRepair/StochReact Fifty',
           'min over cv of 100*(CP-SAT plan + repair / SLAB re-planning - 1), uncertain waits, 50 modules')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--strict', action='store_true',
                    help='fail on any missing or incomplete source')
    ap.add_argument('--out_dir', default=OUT_DIR,
                    help='directory of the three output files')
    args = ap.parse_args(argv)
    mode = 'strict' if args.strict else 'default'
    reset()
    t0 = time.time()
    try:
        check_data_counts()
    except AssertionError as err:
        FAILS.append(('data', str(err)))
    for build in (table2, table3, table4, table5, table6, table7, stats, extras):
        build()
    checks = self_checks()
    coll = collisions(args.out_dir)
    pend = [e for e in REG.values() if e['pending']]

    print(f'make_paper_macros: {len(REG)} macros, {len(REG) - len(pend)} '
          f'values, {len(pend)} pending, {len(FAILS)} failed '
          f'({time.time() - t0:.1f} s, mode {mode})')
    if pend:
        print('PENDING (\\pend):')
        for e in pend:
            print(f'  {e["name"]}: {e["reason"]}')
    print('SELF-CHECKS:')
    for c in checks:
        print(f'  {c["name"]}: {c["status"]} '
              f'(macro {c.get("macro", "-")}, independent '
              f'{c.get("independent", "-")}; {c["how"]})')
    if coll:
        print('NAME COLLISIONS with ' + ', '.join(EXISTING_MACRO_FILES) + ': '
              + ', '.join(coll))
    if FAILS:
        print('FAILED ASSERTIONS (nothing written):')
        for name, msg in FAILS:
            print(f'  {name}: {msg}')
        return 1
    if args.strict and pend:
        print(f'--strict: {len(pend)} sources missing (nothing written)')
        return 2
    stamp = time.strftime('%Y-%m-%d %H:%M')
    os.makedirs(args.out_dir, exist_ok=True)
    write_tex(os.path.join(args.out_dir, OUT_TEX), mode, stamp)
    write_json(os.path.join(args.out_dir, OUT_JSON))
    write_preview(os.path.join(args.out_dir, OUT_MD), mode, stamp, checks, coll)
    print('wrote ' + ', '.join(rel(os.path.join(args.out_dir, f))
                               for f in (OUT_TEX, OUT_JSON, OUT_MD)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
