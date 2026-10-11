"""Tests for stats_tests.py.

Run:  python -m pytest -q test_stats_tests.py
  or: python test_stats_tests.py
"""
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.stats import wilcoxon

import stats_tests as st

REPO = os.path.dirname(os.path.abspath(__file__))
DATASET = "10x25+ppvc-mixed"
BASELINE = "SPT+10x25+ppvc-mixed+smoke"
METHODS = ["greedy+10x25+ppvc-mixed+full",
           "greedy+10x25+ppvc-mixed+a1-bare",
           "greedy+10x25+ppvc-mixed+a2-lagfeat"]

# Signed-rank textbook example (Wikipedia, "Wilcoxon signed-rank test",
# worked example): 10 pairs (x2, x1), one zero difference. Dropping the zero
# (Wilcoxon's original treatment), the ranks of |d| are
# 5,5,7,9,10,12,15,17,20 -> 1.5,1.5,3,4,5,6,7,8,9, so R+ = 27, R- = 18,
# the signed-rank sum W = 9, and r = (27 - 18) / 45 = 0.2.
X2 = np.array([125, 115, 130, 140, 140, 115, 140, 125, 140, 135], dtype=float)
X1 = np.array([110, 122, 125, 120, 140, 124, 123, 137, 135, 145], dtype=float)


# --------------------------------------------------------------------------- #
# Holm
# --------------------------------------------------------------------------- #
def test_holm_hand_example():
    # m = 4. Sorted: .005*4=.02, .01*3=.03, .03*2=.06, .04*1=.04 -> monotone .06
    p = [0.01, 0.04, 0.03, 0.005]
    adj, rej = st.holm(p, alpha=0.05)
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02])
    assert rej.tolist() == [True, False, False, True]


def test_holm_matches_statsmodels():
    multitest = pytest.importorskip("statsmodels.stats.multitest")
    rng = np.random.default_rng(1)
    cases = [
        [0.01, 0.04, 0.03, 0.005],
        [0.2, 0.2, 0.001, 0.049, 0.9],
        [0.5],
        [1e-8, 0.02, 0.025, 0.03, 0.8, 0.012],
    ] + [list(rng.uniform(0, 0.2, size=k)) for k in (3, 7, 12, 30)]
    for p in cases:
        rej_ref, adj_ref, _, _ = multitest.multipletests(p, alpha=0.05,
                                                         method="holm")
        adj, rej = st.holm(p, alpha=0.05)
        np.testing.assert_allclose(adj, adj_ref, rtol=0, atol=1e-12)
        assert rej.tolist() == rej_ref.tolist()


def test_holm_nan_left_out_of_family():
    adj, rej = st.holm([0.01, np.nan, 0.04])
    np.testing.assert_allclose(adj[[0, 2]], [0.02, 0.04])
    assert np.isnan(adj[1]) and not rej[1]


# --------------------------------------------------------------------------- #
# Rank-biserial and Wilcoxon
# --------------------------------------------------------------------------- #
def test_rank_biserial_textbook_example():
    # default (wilcox) drops the zero: R+ = 27, R- = 18 -> 9/45 = 0.2
    assert st.rank_biserial(X2, X1) == pytest.approx(0.2)
    # zsplit keeps the zero: ranks 1..10, R+ = 32 + 0.5, R- = 22 + 0.5 -> 10/55
    assert st.rank_biserial(X2, X1, zero_method="zsplit") == pytest.approx(10.0 / 55.0)
    # pratt ranks the zero but drops it: R+ = 32, R- = 22 -> 10/54
    assert st.rank_biserial(X2, X1, zero_method="pratt") == pytest.approx(10.0 / 54.0)


def test_rank_sums_agree_with_scipy():
    d = X2 - X1
    for zm in ("wilcox", "pratt", "zsplit"):
        r_plus, _ = st._signed_rank_sums(d, zm)
        ref = wilcoxon(d, zero_method=zm, alternative="greater").statistic
        assert r_plus == pytest.approx(ref)


def test_rank_biserial_sign_convention():
    a = np.array([3.0, 5.0, 7.0, 9.0])
    b = a - np.array([1.0, 2.0, 3.0, 4.0])
    assert st.rank_biserial(a, b) == pytest.approx(1.0)
    assert st.rank_biserial(b, a) == pytest.approx(-1.0)
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=50), rng.normal(size=50)
    assert st.rank_biserial(x, y) == pytest.approx(-st.rank_biserial(y, x))


def test_wilcoxon_paired():
    # Exact p-value of the textbook example with the zero dropped.
    assert st.wilcoxon_paired(X2, X1) == pytest.approx(0.6328125)
    assert st.wilcoxon_paired(X2, X1, zero_method="zsplit") == pytest.approx(
        wilcoxon(X2 - X1, zero_method="zsplit").pvalue)
    assert st.wilcoxon_paired(X1, X1) == 1.0
    rng = np.random.default_rng(4)
    a = rng.normal(size=80)
    assert st.wilcoxon_paired(a + 1.0, a + rng.normal(scale=0.3, size=80)) < 1e-6


# --------------------------------------------------------------------------- #
# Bootstrap CI, dz, gap
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_contains_mean_and_narrows():
    rng = np.random.default_rng(5)
    widths = {}
    for n in (20, 200, 2000):
        b = rng.normal(200.0, 10.0, size=n)
        a = b + rng.normal(-2.0, 4.0, size=n)
        md, (lo, hi) = st.paired_mean_diff_ci(a, b, n_boot=4000, seed=0)
        assert md == pytest.approx(np.mean(a - b))
        assert lo < md < hi
        widths[n] = hi - lo
    # Width scales like 1/sqrt(n): each tenfold increase shrinks it about 3.2x.
    assert widths[200] < widths[20] / 2
    assert widths[2000] < widths[200] / 2


def test_bootstrap_is_seeded():
    rng = np.random.default_rng(6)
    a, b = rng.normal(size=40), rng.normal(size=40)
    assert st.paired_mean_diff_ci(a, b, seed=7) == st.paired_mean_diff_ci(a, b, seed=7)


def test_cohen_dz():
    b = np.zeros(5)
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # mean 3, sd sqrt(2.5)
    assert st.cohen_dz(a, b) == pytest.approx(3.0 / np.sqrt(2.5))
    assert st.cohen_dz(b, a) == pytest.approx(-3.0 / np.sqrt(2.5))
    assert np.isnan(st.cohen_dz(a, a))


def test_gap_vs_reference():
    gaps, mean_gap = st.gap_vs_reference([110.0, 200.0, 300.0], [100.0, 200.0, 250.0])
    np.testing.assert_allclose(gaps, [10.0, 0.0, 20.0])
    assert mean_gap == pytest.approx(10.0)


def test_unpaired_lengths_rejected():
    with pytest.raises(ValueError):
        st.paired_mean_diff_ci([1.0, 2.0], [1.0])


# --------------------------------------------------------------------------- #
# CLI end to end on the existing 10x25 result files
# --------------------------------------------------------------------------- #
def _require_result_files():
    for tag in [BASELINE] + METHODS:
        if not os.path.isfile(st.result_path(st.DEFAULT_RESULTS_DIR, DATASET, tag)):
            pytest.skip(f"result file for {tag} not present")


def _run_cli(*extra):
    cmd = [sys.executable, os.path.join(REPO, "stats_tests.py"),
           "--dataset", DATASET, "--baseline", BASELINE,
           "--methods", *METHODS, *extra]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_cli_end_to_end():
    _require_result_files()
    out = _run_cli("--reference", "cpsat")
    print(out)
    assert "| Method (A) |" in out and "Holm p" in out and "gap A vs CP-SAT %" in out
    body = [ln for ln in out.splitlines() if ln.startswith("| `")]
    assert len(body) == 3
    for tag, mean_a in zip(METHODS, ["210.89", "211.07", "210.74"]):
        row = next(ln for ln in body if f"`{tag}`" in ln)
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert cells[2] == "100"            # n paired instances
        assert cells[3] == mean_a           # published per-method mean
        assert cells[4] == "215.98"         # SPT mean


def test_cli_pooled_seeds():
    _require_result_files()
    out = _run_cli("--pool-seeds")
    row = next(ln for ln in out.splitlines() if f"`{METHODS[0]}`" in ln)
    cells = [c.strip() for c in row.strip("|").split("|")]
    tags = [METHODS[0], METHODS[0] + "-s301", METHODS[0] + "-s302"]
    vals = [np.load(st.result_path(st.DEFAULT_RESULTS_DIR, DATASET, t))[:, 0]
            for t in tags]
    assert cells[1] == "3"
    assert cells[3] == f"{np.mean(vals):.2f}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-s"]))
