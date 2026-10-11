"""Tests for make_paper_macros.py (short; about ten seconds on one core).

Run:  python -B -m pytest -q -p no:cacheprovider test_make_paper_macros.py
"""
import json
import os
import re
import sys

import pytest

sys.dont_write_bytecode = True
import make_paper_macros as mpm  # noqa: E402

DEF_RE = re.compile(r'^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\} % source: (.+)$')


def write_jsonl(path, recs):
    path.write_text(''.join(json.dumps(r) + '\n' for r in recs))


def test_format_rules():
    assert mpm.dec(211.25, 1) == '211.3'          # half-up on the decimal value
    assert mpm.dec(315.95, 1) == '316.0'          # binary 315.9499... still ties up
    assert mpm.dec(-0.04, 1) == '0.0'             # never '-0.0'
    assert mpm.dec(-2.25, 1) == '-2.3'
    assert mpm.fmt('round_int', 41.6) == '42'
    assert mpm.fmt('s', 0.472) == '0.47' and mpm.fmt('s', 11.337) == '11.3'
    assert mpm.fmt('median', 9.0) == '9' and mpm.fmt('median', 17.5) == '17.5'
    assert mpm.fmt_p(1.8e-10) == r'$1.8\times10^{-10}$'
    assert mpm.fmt_p(0.0496) == r'$5.0\times10^{-2}$'
    assert mpm.fmt_p(0.996) == '$1.0$'
    assert mpm.pct(90.0, 100.0) == pytest.approx(-10.0)
    with pytest.raises(AssertionError):
        mpm.fmt('int', 2.5)


def test_jsonl_partial_last_line(tmp_path):
    p = tmp_path / 'a.jsonl'
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": 3')
    assert [r['a'] for r in mpm.read_jsonl(str(p))] == [1, 2]
    p.write_text('{"a": 1}\n{"a"\n{"a": 3}\n')
    with pytest.raises(AssertionError):
        mpm.read_jsonl(str(p))
    with pytest.raises(mpm.Missing):
        mpm.read_jsonl(str(tmp_path / 'absent.jsonl'))


def _buf(i, B, obj, **kw):
    r = dict(dataset='X', instance=f'instance_{i:03d}', instance_id=i,
             buffer='shared', B=B, objective=obj, feasible=True,
             within_capacity=True, occupancy={'peak': 5}, time_limit_s=60.0,
             n_low_share=0, n_no_evidence=0, cpu_share_min=0.99)
    r.update(kw)
    return r


def test_buffer_file_present_only_when_complete(tmp_path):
    p = tmp_path / 'b.jsonl'
    recs = [_buf(i, B, 100.0 + i) for i in range(3) for B in (None, 8)]
    mpm.reset()
    write_jsonl(p, recs[:-1])                     # (instance 2, B 8) not yet
    with pytest.raises(mpm.Missing):
        mpm.buffer_column(str(p), 'X', 3, (None, 8), None, time_limit=60.0)
    mpm.reset()
    write_jsonl(p, recs)
    v = mpm.buffer_column(str(p), 'X', 3, (None, 8), 8, time_limit=60.0)
    assert list(v) == [100.0, 101.0, 102.0]
    mpm.reset()
    recs[1]['n_low_share'] = 1                    # contended solve
    write_jsonl(p, recs)
    with pytest.raises(AssertionError):
        mpm.buffer_column(str(p), 'X', 3, (None, 8), 8, time_limit=60.0)


def test_memetic_evidence_and_partial_json(tmp_path, monkeypatch):
    ds = '10x25+ppvc-mixed'
    monkeypatch.setattr(mpm, 'REPO', str(tmp_path))
    monkeypatch.setitem(mpm.N_INST, ds, 2)
    path = tmp_path / mpm.ma_rel(ds)
    path.parent.mkdir(parents=True)
    rec = dict(validator_violations=0, n_low_share=0, n_no_evidence=0,
               cpu_share_min=0.99)
    doc = dict(dataset=ds, budget_s=300.0, n_instances=2,
               contention=dict(n_low_share=0, n_no_evidence=0,
                               n_records_without_fields=0, cpu_share_min=0.99),
               instances=[dict(rec, instance='instance_000', best_makespan=200.0),
                          dict(rec, instance='instance_001', best_makespan=210.0)])
    mpm.reset()
    path.write_text(json.dumps(doc))
    assert mpm.memetic(ds).mean() == pytest.approx(205.0)
    mpm.reset()
    path.write_text(json.dumps(doc)[:-20])        # being written
    with pytest.raises(mpm.Missing):
        mpm.memetic(ds)
    mpm.reset()
    doc['instances'][1]['n_low_share'] = 1
    doc['contention']['n_low_share'] = 1
    path.write_text(json.dumps(doc))
    with pytest.raises(AssertionError):
        mpm.memetic(ds)


def test_full_run_default(tmp_path):
    assert mpm.main(['--out_dir', str(tmp_path)]) == 0
    data = json.loads((tmp_path / mpm.OUT_JSON).read_text())
    defs = [ln for ln in (tmp_path / mpm.OUT_TEX).read_text().splitlines()
            if ln.startswith('\\newcommand')]
    assert len(defs) == len(data) == len(mpm.REG)
    for ln in defs:
        m = DEF_RE.match(ln)
        assert m, ln
        name, val = m.group(1), m.group(2)
        assert val.count('{') == val.count('}')
        if data[name]['pending']:
            assert val == f'\\pend{{{name}}}'
        else:
            assert val == data[name]['value']
    # values checked by hand against the result files and the campaign log
    for name, val in (('RepairPenDanielTen', '44.2'), ('RepairPenLo', '42'),
                      ('RepairPenHi', '133'), ('SptTen', '216.0'),
                      ('HstarTen', '211.3'), ('HstarEighty', '570.3'),
                      ('CpOptTen', '83'), ('nInstThirty', '20'),
                      ('CpLongN', '20'), ('FtUpdates', '400')):
        assert data[name]['value'] == val, name
    assert (tmp_path / mpm.OUT_MD).exists()


def test_strict_mode_writes_nothing_when_pending(tmp_path):
    rc = mpm.main(['--strict', '--out_dir', str(tmp_path)])
    pending = any(e['pending'] for e in mpm.REG.values())
    assert rc == (2 if pending else 0)
    assert (tmp_path / mpm.OUT_TEX).exists() == (not pending)
