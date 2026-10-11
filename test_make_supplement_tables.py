"""Tests for make_supplement_tables.py (short; a few seconds on one core).

Run:  python -B -m pytest -q -p no:cacheprovider test_make_supplement_tables.py
"""
import os
import re
import sys

import numpy as np
import pytest

sys.dont_write_bytecode = True
import make_supplement_tables as mst  # noqa: E402


def test_cell_text_helpers():
    assert mst.stext('-4.7') == '$-$4.7' and mst.stext('4.7') == '4.7'
    assert mst.ci_text(-6.04, -3.36) == '[$-$6.0, $-$3.4]'
    assert mst.wtl([1, 2, 3, 4], [2, 2, 2, 2]) == '1/1/2'
    assert mst.sci(3e-4) == r'$3\times10^{-4}$'
    assert mst.rng_text(2, 2) == '2' and mst.rng_text(4, 6) == '4--6'


def test_bold_best_marks_ties_and_skips_pending():
    a = mst.Cell('211.3', 211.31, key='211.3')
    b = mst.Cell('211.3', 211.29, key='211.3')
    c = mst.Cell('216.0', 216.0, key='216.0')
    p = mst.Cell(r'\pend{X}', None, True)
    mst.bold_best([a, b, c, p])
    assert a.text == b.text == r'\textbf{211.3}' and c.text == '216.0'
    u = mst.Cell('16.6', 16.6, key='16.6')
    v = mst.Cell('16.1', 16.1, key='16.1')
    mst.bold_best([u, v], maximize=True)
    assert u.text.startswith(r'\textbf') and not v.text.startswith(r'\textbf')


def test_at_time_reads_the_incumbent():
    rec = {'instance': 'instance_000', 'objective': 190.0,
           'trajectory': [[0.1, 250.0, 180.0], [30.0, 200.0, 180.0],
                          [200.0, 190.0, 185.0]]}
    assert mst.at_time(rec, 60) == 200.0 and mst.at_time(rec, 300) == 190.0
    with pytest.raises(AssertionError):
        mst.at_time(rec, 0.05)                 # no schedule yet
    bad = dict(rec, objective=189.0)
    with pytest.raises(AssertionError):
        mst.at_time(bad, 300)                  # trajectory must end on it


def test_mac_asserts_the_paper_value():
    mst.reset()
    mst.MAC.update({'XTen': {'value': '211.3', 'raw': 211.31, 'pending': False,
                             'reason': None},
                    'XPend': {'value': None, 'raw': None, 'pending': True,
                              'reason': 'file absent'}})
    c = mst.mac('XTen', lambda: 211.31, 'h')
    assert c.text == r'\XTen' and c.key == '211.3'
    with pytest.raises(AssertionError):
        mst.mac('XTen', lambda: 211.0, 'h')    # value drifted from the paper
    p = mst.mac('XPend', lambda: 1.0, 'h')     # source present, macro pending
    assert p.pending and mst.WARN and 'XPend' in mst.PENDING
    mst.reset()


def test_settle_update_finds_the_last_exit_from_the_band():
    upd = np.arange(10, 101, 10)
    # final level = mean of the last 3 points = 100; band 1 % = [99, 101]
    curve = [120, 100, 103, 100.5, 99.5, 100, 100, 99, 101, 100]
    assert mst.settle_update(upd, curve, tail=3, band=1.0) == 40
    assert mst.settle_update(upd, [100] * 10, tail=3, band=1.0) == 10


def test_full_run_writes_every_table(tmp_path):
    rc = mst.main(['--out_dir', str(tmp_path)])
    assert rc == 0
    files = {f.name for f in tmp_path.iterdir()}
    want = {f's1_size{n}.tex' for n in (10, 20, 30, 50, 80)} | {
        's1_long.tex', 's1_ind.tex', 's1_mix.tex', 's2_stats.tex', 's3_hgnn.tex',
        's4_bound.tex', 's4_aids.tex', 's5_byproducts.tex', 's6_flowline.tex',
        's7_sensitivity.tex', 's8_stability.tex', 's10_routes.tex',
        's10_stations.tex', 's11_hyper.tex', 's11_protocols.tex',
        'supp_macros.tex'}
    assert want <= files
    for name in want - {'supp_macros.tex'}:
        txt = (tmp_path / name).read_text()
        assert txt.count(r'\begin{tabular}') == 1
        assert 'nan' not in txt.lower().replace('nanosecond', '')
        assert '/home/' not in txt                   # no machine-local path
        body = [l for l in txt.splitlines() if not l.startswith('%')]
        for line in body:
            if line.endswith(r'\\') and '&' in line:
                n = len(re.findall(r'(?<!\\)&', line)) + 1
                assert n in (2, 3, 4, 5, 6, 7, 9), (name, line)
    macros = (tmp_path / 'supp_macros.tex').read_text()
    names = re.findall(r'\\newcommand\{\\(Su[A-Za-z]+)\}', macros)
    assert len(names) == len(set(names)) > 40
    # the S9 chart is one network under two executions: its numbers are the
    # two makespans and the penalties, and the macros of the former chart of
    # two different policies are gone
    for k in ('SuGanttDispatch', 'SuGanttRepair', 'SuGanttPenalty',
              'SuGanttPenaltyMedian', 'SuGanttN', 'SuCurveSettleAug',
              'SuCurveSettleFree', 'SuCurveSpreadAug', 'SuCurveSpreadFree',
              'SuMixEighty', 'SuSingleEighty'):
        assert k in names, k
    for k in ('SuGanttAware', 'SuGanttPlan', 'SuGanttRepaired',
              'SuGanttMedianGap', 'SuGanttMeanAware', 'SuGanttMeanRepaired',
              'SuGanttUpdates', 'SuGanttSeed'):
        assert k not in names, k
    m = dict(re.findall(r'\\newcommand\{\\(Su[A-Za-z]+)\}\{([^}]*)\}', macros))
    assert int(m['SuGanttDispatch']) < int(m['SuGanttRepair'])
    assert float(m['SuMixEighty']) >= float(m['SuSingleEighty'])
    # the seed rows of the per-size tables are never bold
    t = (tmp_path / 's1_size10.tex').read_text()
    assert not re.search(r'^Seed \d+ & \\textbf', t, re.M)
