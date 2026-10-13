#!/usr/bin/env python3
"""analysis_stats_tables_7_9.py -- paired tests for the paper's Tables VII
(training orders), VIII (uncertain waits) and IX (disruptions), computed
from the existing result files with the conventions of Table S10 (s2_stats
of make_supplement_tables.py):

  * paired by order (Table VIII: by order and replication within one cv);
  * a policy with three seeds enters with its per-order seed mean;
  * two-sided Wilcoxon signed-rank test with zero differences dropped
    (stats_tests.wilcoxon_paired, zero_method='wilcox');
  * Holm's step-down correction within each family (stats_tests.holm);
  * matched-pairs rank-biserial r of SLAB minus rival, r < 0 = SLAB smaller
    (stats_tests.rank_biserial);
  * mean paired difference SLAB minus rival with a 95% percentile bootstrap
    interval from 2000 resamples, numpy default_rng(0)
    (stats_tests.paired_mean_diff_ci).

Families
  Table IX   one rival over the six disruption types at one size; makespan and
             the changed-operation count (comb_nerv) are separate families.
  Table VIII one rival over the three cv values at one size.
  Table VII  one rival over the five default-factory sizes; the 15-station set
             is tested on its own (a family of one, so its adjusted p equals
             its raw p).

Sources are read through the validated readers of make_paper_macros.py
(reactive(), stoch(), inst_of()), so every check those readers make (model
name, instance counts, feasibility, contention evidence) applies here too.
For Table IX the SLAB, SPTL, GA and right-shift columns come from the
reactive file named by make_paper_macros.dis_rel(), except the GA, which is
the fast-decoder rerun of make_paper_macros.gafast_rel() on the same states
(reactive_hstar_gafast/, SLAB records and state fingerprints asserted equal);
the SPT and LLF
re-planners from make_paper_macros.rules_rel() (reactive_hstar_rules/, whose
SLAB, SPTL and right-shift records are asserted identical to the Table IX
file); the residual CP-SAT columns from make_paper_macros.resid_rel()
(reactive_hstar_hint_fix/residual_cpsat_<type>.npy, re-solved on 2026-10-12
on exactly the Table IX states and asserted to match them). The residual
file holds its own SLAB re-plan (drl_reactive), which is the primary pairing
for the CP-SAT rivals; the pairing with the table's SLAB column is reported
as well, with the count of orders whose pre-disruption SLAB plan differs
between the two files (0 for the re-solved files).

Writes test_results/PPVC/stats_tables_7_9/stats.json and stats.md.
usage: python analysis_stats_tables_7_9.py
"""
import hashlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)
sys.dont_write_bytecode = True
import make_paper_macros as M  # noqa: E402
from stats_tests import (holm, paired_mean_diff_ci, rank_biserial,  # noqa: E402
                         wilcoxon_paired)

OUT = os.path.join(REPO, 'test_results', 'PPVC', 'stats_tables_7_9')
N_BOOT, BOOT_SEED = 2000, 0
SOURCES = {}


def note_src(rel):
    p = os.path.join(REPO, rel)
    if rel not in SOURCES:
        h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
        SOURCES[rel] = {'sha256': h, 'mtime': os.path.getmtime(p)}
    return rel


def test(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d, (lo, hi) = paired_mean_diff_ci(a, b, n_boot=N_BOOT, seed=BOOT_SEED)
    return {'n': int(a.size), 'mean_slab': float(a.mean()), 'mean_rival': float(b.mean()),
            'mean_diff': d, 'ci95': [lo, hi],
            'p': wilcoxon_paired(a, b, zero_method='wilcox'),
            'r': rank_biserial(a, b),
            'slab_smaller': int(np.sum(a < b)), 'equal': int(np.sum(a == b)),
            'slab_larger': int(np.sum(a > b))}


def add_holm(rows):
    adj, _ = holm([r['p'] for r in rows])
    for r, x in zip(rows, adj):
        r['p_holm'] = float(x)


# --------------------------------------------------------------------------- #
# Table IX
# --------------------------------------------------------------------------- #
def resid_file(ds, t):
    # the re-solved files of 2026-10-12 (see make_paper_macros.resid_rel)
    return M.resid_rel(ds, t)


def load_resid(ds, t, n):
    rel = note_src(resid_file(ds, t))
    d = np.load(os.path.join(REPO, rel), allow_pickle=True).item()
    meta = d['meta']
    M.check(meta['model_name'] == M.HSTAR_MODEL and meta['dataset'] == ds
            and meta['disruption_type'] == t and int(meta['n_instances']) == n
            and int(meta['cpsat_workers']) == 2, f'{rel}: meta')
    for k, c in d['contention'].items():
        M.check(c.get('n_low_share', 0) == 0 and c.get('n_no_evidence', 0) == 0,
                f'{rel}: contention {k}')
    for k in ('residual_cpsat_5s', 'residual_cpsat_30s', 'drl_reactive'):
        r = d['records'][k]
        M.check(len(r['makespan']) == n and bool(np.all(r['feasible'])),
                f'{rel}: {k}')
    M.resid_file(ds, t, n)   # match to the Table IX states (asserted)
    return d


def table9():
    rivals = (('LLF', 'rule_llf_reactive', 'rules'), ('SPT', 'rule_spt_reactive', 'rules'),
              ('SPTL', 'spt_reactive', 'reactive'), ('GA 5 s', 'ga_reactive', 'gafast'),
              ('Right-shift', 'right_shift', 'reactive'),
              ('Residual CP-SAT 5 s', 'residual_cpsat_5s', 'resid'),
              ('Residual CP-SAT 30 s', 'residual_cpsat_30s', 'resid'))
    out = {'makespan': [], 'changes': [], 'makespan_tableSLAB_vs_resid': [],
           'changes_tableSLAB_vs_resid': [], 'pairing_check': []}
    for S, ds, n in M.DIS_SIZES:
        files = {}
        for _T, t in M.DIS_TYPES:
            rea = M.reactive(ds, t, n)
            note_src(M.dis_rel(ds, t))
            res = load_resid(ds, t, n)
            rul = M.rules_file(ds, t, n)   # SLAB/SPTL/right-shift equal to rea (asserted)
            note_src(M.rules_rel(ds, t))
            gaf = M.gafast_file(ds, t, n)  # fast-decoder GA; SLAB and states equal (asserted)
            note_src(M.gafast_rel(ds, t))
            files[t] = (rea, res, rul, gaf)
            o_rea = np.asarray(rea['orig_makespans'], float)
            o_res = np.asarray(res['orig_makespans'], float)
            s_rea = np.asarray(rea['records']['drl_reactive']['makespan'], float)
            s_res = np.asarray(res['records']['drl_reactive']['makespan'], float)
            rs_rea = np.asarray(rea['records']['right_shift']['makespan'], float)
            rs_res = np.asarray(res['records']['residual_cpsat_5s']['rs_makespan'], float)
            out['pairing_check'].append({
                'size': S, 'type': t, 'n': n,
                'reactive_file': M.dis_rel(ds, t), 'residual_file': resid_file(ds, t),
                'orders_pre_disruption_SLAB_plan_differs': int(np.sum(o_rea != o_res)),
                'orders_SLAB_replan_differs': int(np.sum(s_rea != s_res)),
                'mean_SLAB_reactive_file': float(s_rea.mean()),
                'mean_SLAB_residual_file': float(s_res.mean()),
                'orders_right_shift_differs': int(np.sum(rs_rea != rs_res)),
                'mean_right_shift_reactive_file': float(rs_rea.mean()),
                'mean_right_shift_hint_residual_file': float(rs_res.mean())})
        for label, key, where in rivals:
            for field, fam_key, fam_key2 in (('makespan', 'makespan', 'makespan_tableSLAB_vs_resid'),
                                             ('comb_nerv', 'changes', 'changes_tableSLAB_vs_resid')):
                fam, fam2 = [], []
                for T, t in M.DIS_TYPES:
                    rea, res, rul, gaf = files[t]
                    if where == 'gafast':
                        a = gaf['records']['drl_reactive'][field]
                        b = gaf['records'][key][field]
                        src = M.gafast_rel(ds, t)
                    elif where == 'reactive':
                        a = rea['records']['drl_reactive'][field]
                        b = rea['records'][key][field]
                        src = M.dis_rel(ds, t)
                    elif where == 'rules':
                        # same states and the same SLAB re-plans as the reactive file
                        a = rul['records']['drl_reactive'][field]
                        b = rul['records'][key][field]
                        src = M.rules_rel(ds, t)
                    else:
                        a = res['records']['drl_reactive'][field]
                        b = res['records'][key][field]
                        src = resid_file(ds, t)
                        r2 = test(rea['records']['drl_reactive'][field], b)
                        r2.update(size=S, type=t, rival=label, metric=field,
                                  slab_source=M.dis_rel(ds, t), rival_source=src,
                                  pairing='table SLAB column (reactive file) vs residual file')
                        fam2.append(r2)
                    r = test(a, b)
                    r.update(size=S, type=t, rival=label, metric=field,
                             slab_source=src, rival_source=src,
                             pairing='same file, same disrupted state')
                    fam.append(r)
                add_holm(fam)
                out[fam_key].extend(fam)
                if fam2:
                    add_holm(fam2)
                    out[fam_key2].extend(fam2)
    return out


# --------------------------------------------------------------------------- #
# Table VIII
# --------------------------------------------------------------------------- #
def table8():
    rivals = (('SLAB, repair', 'slab_repair'), ('SPT, re-plan', 'spt_reactive'),
              ('CP-SAT, repair', 'cpsat_repair'), ('CP-SAT, rolling', 'cpsat_rolling'),
              ('SLAB, waits known (reference)', 'slab_perfect_info'))
    out = []
    for S in ('Ten', 'Fifty'):
        M.stoch_consistency(S)
        base = {(int(r['instance_idx']), round(float(r['cv']), 3), int(r['rep'])):
                float(r['realized_makespan']) for r in M.stoch(S, 'slab_reactive')}
        note_src(M.stoch_rel(S, 'slab_reactive'))
        for label, meth in rivals:
            if meth == 'cpsat_rolling' and S != 'Ten':
                continue
            riv = {(int(r['instance_idx']), round(float(r['cv']), 3), int(r['rep'])):
                   float(r['realized_makespan']) for r in M.stoch(S, meth)}
            note_src(M.stoch_rel(S, meth))
            M.check(set(riv) == set(base), f'{S} {meth}: run keys differ')
            fam = []
            for _C, cv in M.CVS:
                keys = sorted(k for k in base if k[1] == cv)
                a = [base[k] for k in keys]
                b = [riv[k] for k in keys]
                r = test(a, b)
                r.update(size=S, cv=cv, rival=label,
                         slab_source=M.stoch_rel(S, 'slab_reactive'),
                         rival_source=M.stoch_rel(S, meth),
                         pairing='(order, replication) within one cv')
                fam.append(r)
            add_holm(fam)
            out.extend(fam)
    return out


# --------------------------------------------------------------------------- #
# Table VII
# --------------------------------------------------------------------------- #
def table7():
    rivals = (('Without waits', M.LAGFREE), ('Full waits', M.FULL),
              ('Full waits, aids', M.FULLAIDS),
              ('Full waits, aids, 15 stations', M.TIGHTNET))
    out = []
    for label, tpl in rivals:
        fam = []
        for S, ds in M.SIZES:
            a = M.inst_of(M.HSTAR, ds)
            b = M.inst_of(tpl, ds)
            r = test(a, b)
            r.update(size=S, dataset=ds, rival=label,
                     slab_source=M.src(M.HSTAR, ds), rival_source=M.src(tpl, ds))
            for s in M.SEEDS:
                note_src(M.res_rel(M.HSTAR.format(s=s), ds))
                note_src(M.res_rel(tpl.format(s=s), ds))
            fam.append(r)
        add_holm(fam)
        out.extend(fam)
        ds = M.TIGHT[1]
        r = test(M.inst_of(M.HSTAR, ds), M.inst_of(tpl, ds))
        r.update(size='Tight (15 stations, 10 modules)', dataset=ds, rival=label,
                 slab_source=M.src(M.HSTAR, ds), rival_source=M.src(tpl, ds))
        for s in M.SEEDS:
            note_src(M.res_rel(M.HSTAR.format(s=s), ds))
            note_src(M.res_rel(tpl.format(s=s), ds))
        r['p_holm'] = r['p']   # family of one
        out.append(r)
    return out


# --------------------------------------------------------------------------- #
def fp(p):
    return f'{p:.2e}' if p < 1e-3 else f'{p:.4f}'


def row(r, first):
    return (f"| {first} | {r['n']} | {r['mean_slab']:.2f} | {r['mean_rival']:.2f} | "
            f"{r['mean_diff']:+.2f} | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | "
            f"{fp(r['p'])} | {fp(r['p_holm'])} | {r['r']:+.2f} | "
            f"{r['slab_smaller']}/{r['equal']}/{r['slab_larger']} |")


HEAD = ['| Rival | Size / type | n | SLAB mean | Rival mean | Diff SLAB-rival | 95% CI | p | Holm p | r | SLAB smaller/equal/larger |',
        '|---|---|---:|---:|---:|---:|---|---:|---:|---:|:-:|']


def main():
    t9, t8, t7 = table9(), table8(), table7()
    os.makedirs(OUT, exist_ok=True)
    res = {'script': 'analysis_stats_tables_7_9.py',
           'conventions': {'wilcoxon': 'two-sided, zero differences dropped',
                           'holm': 'within each family (see module docstring)',
                           'r': 'matched-pairs rank-biserial of SLAB minus rival; r < 0 = SLAB smaller',
                           'bootstrap': f'{N_BOOT} resamples, numpy default_rng({BOOT_SEED}), percentile 95%',
                           'seed_mean': 'per-order mean of seeds 301, 302, 303 where a policy has three'},
           'table_IX_disruptions': t9, 'table_VIII_uncertain_waits': t8,
           'table_VII_training_orders': t7, 'sources': SOURCES}
    with open(os.path.join(OUT, 'stats.json'), 'w') as f:
        json.dump(res, f, indent=1)
    L = ['# Paired tests for Tables VII, VIII and IX', '',
         'Conventions of Table S10: two-sided Wilcoxon signed-rank (zero differences dropped), '
         'Holm within each family, matched-pairs rank-biserial r (r < 0: SLAB smaller), mean paired '
         'difference SLAB minus rival with a 95% percentile bootstrap CI (2000 resamples, seed 0). '
         'Sources and their SHA-256 are in stats.json.', '',
         '## Table IX, disruptions: makespan (h)', '',
         'Family = one rival over the six types at one size. SPTL, GA and right-shift are paired with '
         'the SLAB re-plan of the same reactive file (the table\'s SLAB column). Residual CP-SAT is paired '
         'with the SLAB re-plan stored in the residual file itself, which starts from the same disrupted state.', ''] + HEAD
    for r in t9['makespan']:
        L.append(row(r, f"{r['rival']} | {r['size']} / {r['type']}"))
    L += ['', '## Table IX, disruptions: changed operations (station reassignments + starts moved > 1 h)', ''] + HEAD
    for r in t9['changes']:
        L.append(row(r, f"{r['rival']} | {r['size']} / {r['type']}"))
    L += ['', '## Table IX, alternative pairing: the table\'s SLAB column (reactive file) vs residual CP-SAT', '',
          'The two files carry pre-disruption SLAB plans that differ on some orders (see the pairing check), '
          'so these pairs do not always share the disrupted state.', ''] + HEAD
    for r in t9['makespan_tableSLAB_vs_resid'] + t9['changes_tableSLAB_vs_resid']:
        L.append(row(r, f"{r['rival']} ({r['metric']}) | {r['size']} / {r['type']}"))
    L += ['', '### Pairing check between the reactive and the residual files', '',
          '| Size | Type | n | orders with a different pre-disruption SLAB plan | orders with a different SLAB re-plan | SLAB mean, reactive file | SLAB mean, residual file | orders with a different right-shift | right-shift mean, reactive file | right-shift (hint) mean, residual file |',
          '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for c in t9['pairing_check']:
        L.append(f"| {c['size']} | {c['type']} | {c['n']} | {c['orders_pre_disruption_SLAB_plan_differs']} | "
                 f"{c['orders_SLAB_replan_differs']} | {c['mean_SLAB_reactive_file']:.2f} | "
                 f"{c['mean_SLAB_residual_file']:.2f} | {c['orders_right_shift_differs']} | "
                 f"{c['mean_right_shift_reactive_file']:.2f} | {c['mean_right_shift_hint_residual_file']:.2f} |")
    L += ['', '## Table VIII, uncertain waits: realized makespan (h)', '',
          'SLAB = SLAB re-plan. Pairs = (order, replication) within one cv. '
          'Family = one rival over the three cv values at one size.', ''] + HEAD
    for r in t8:
        L.append(row(r, f"{r['rival']} | {r['size']} / cv {r['cv']}"))
    L += ['', '## Table VII, training orders: mean makespan (h)', '',
          'SLAB = DANIEL trained with halved waits, per-order mean of seeds 301-303; each rival is the '
          'per-order mean of its three seeds. Family = one rival over the five default-factory sizes; the '
          '15-station set is a family of one.', ''] + HEAD
    for r in t7:
        L.append(row(r, f"{r['rival']} | {r['size']}"))
    with open(os.path.join(OUT, 'stats.md'), 'w') as f:
        f.write('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    os.chdir(REPO)
    main()
