#!/usr/bin/env python3
"""stats_tests.py -- paired statistics for comparing scheduling methods
across a shared set of problem instances.

Every comparison is paired by instance index: method A and method B were
evaluated on the same instances, so the unit of analysis is the per-instance
difference d_i = a_i - b_i (A minus B). For makespan, d_i < 0 means A produced
the shorter schedule on instance i.

Library functions
-----------------
paired_mean_diff_ci(a, b, n_boot=10000, seed=0)
    Mean of d with a 95% percentile-bootstrap CI (pairs resampled jointly).
wilcoxon_paired(a, b, zero_method="wilcox")
    Two-sided Wilcoxon signed-rank p-value (scipy). Zero differences are
    dropped before ranking ("wilcox", scipy's default, the rule used in the
    submitted manuscript); "zsplit" and "pratt" are accepted alternatives.
rank_biserial(a, b, zero_method="wilcox")
    Matched-pairs rank-biserial correlation r = (R+ - R-) / (R+ + R-), where
    R+ and R- are the signed-rank sums of d = a - b. Range [-1, 1];
    r < 0 means A tends to be smaller than B (for makespan: A is better).
holm(pvalues, alpha=0.05)
    Holm step-down adjusted p-values and reject flags.
cohen_dz(a, b)
    Standardized mean of paired differences, mean(d) / sd(d, ddof=1).
gap_vs_reference(makespans, reference_makespans)
    Per-instance gap 100 * (m - ref) / ref and its mean, in percent.

Command line
------------
    python stats_tests.py --dataset 10x25+ppvc-mixed \
        --baseline SPT+10x25+ppvc-mixed+smoke \
        --methods greedy+10x25+ppvc-mixed+full greedy+10x25+ppvc-mixed+a1-bare \
        [--pool-seeds] [--reference cpsat]

A method tag is the part of a result file name between "Result_" and
"_<dataset>.npy". Result files are arrays of shape [N, 2] with columns
(makespan, seconds); row i is instance i. With --pool-seeds, every file whose
tag equals "<tag>-s<digits>" is loaded as well and the per-instance mean over
all seeds is used. Holm correction is applied over the family of
method-versus-baseline tests in one invocation.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np
from scipy.stats import rankdata, wilcoxon

REPO = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(REPO, "test_results", "PPVC")
DEFAULT_OR_DIR = os.path.join(REPO, "or_solution", "PPVC")
METRIC_COLUMNS = {"makespan": 0, "time": 1}


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def _paired_diff(a, b):
    """Return d = a - b as a float array after checking the pairing."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"paired samples differ in length: {a.size} vs {b.size}")
    if a.size == 0:
        raise ValueError("paired samples are empty")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("paired samples contain NaN or inf")
    return a - b


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def paired_mean_diff_ci(a, b, n_boot=10000, seed=0, level=0.95):
    """Mean paired difference mean(a - b) and its percentile-bootstrap CI.

    Instances are resampled with replacement n_boot times; each resample
    keeps the pairs intact. The CI is the (1-level)/2 and (1+level)/2
    percentiles of the resampled means.

    Returns (mean_diff, (ci_low, ci_high)).
    """
    d = _paired_diff(a, b)
    n = d.size
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=float)
    # Chunk the index matrix so large n stays within modest memory.
    chunk = max(1, min(int(n_boot), 2_000_000 // max(n, 1)))
    done = 0
    while done < n_boot:
        k = min(chunk, int(n_boot) - done)
        idx = rng.integers(0, n, size=(k, n))
        boot[done:done + k] = d[idx].mean(axis=1)
        done += k
    tail = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot, [tail, 1.0 - tail])
    return float(d.mean()), (float(lo), float(hi))


def wilcoxon_paired(a, b, zero_method="wilcox"):
    """Two-sided Wilcoxon signed-rank p-value for paired samples a and b.

    Zero differences are dropped by default ("wilcox", scipy's default and
    the rule used in the submitted manuscript). "pratt" and "zsplit" are
    accepted as alternatives. scipy's method="auto" chooses the exact null
    distribution for small samples without ties or zeros and the normal
    approximation otherwise. If every difference is zero there is no
    evidence of a difference and 1.0 is returned.
    """
    d = _paired_diff(a, b)
    if np.all(d == 0):
        return 1.0
    res = wilcoxon(d, zero_method=zero_method, alternative="two-sided",
                   method="auto")
    return float(res.pvalue)


def _signed_rank_sums(d, zero_method="wilcox"):
    """Return (R+, R-) for differences d under the given zero handling."""
    d = np.asarray(d, dtype=float)
    if zero_method == "wilcox":
        d = d[d != 0]
        if d.size == 0:
            return 0.0, 0.0
        ranks = rankdata(np.abs(d))
        return float(ranks[d > 0].sum()), float(ranks[d < 0].sum())
    ranks = rankdata(np.abs(d))
    r_plus = float(ranks[d > 0].sum())
    r_minus = float(ranks[d < 0].sum())
    if zero_method == "zsplit":
        r_zero = float(ranks[d == 0].sum())
        return r_plus + r_zero / 2.0, r_minus + r_zero / 2.0
    if zero_method == "pratt":
        # Zeros take part in ranking but are dropped from both sums.
        return r_plus, r_minus
    raise ValueError(f"unknown zero_method: {zero_method!r}")


def rank_biserial(a, b, zero_method="wilcox"):
    """Matched-pairs rank-biserial correlation of d = a - b.

    r = (R+ - R-) / (R+ + R-), with R+ and R- the Wilcoxon signed-rank sums
    of the positive and negative differences (Kerby's simple difference
    formula). Sign convention: r = +1 when every a_i > b_i, r = -1 when every
    a_i < b_i. For makespan with A = method and B = baseline, r < 0 favours
    the method. Zero handling matches wilcoxon_paired ("wilcox" default).
    Returns 0.0 when every difference is zero.
    """
    d = _paired_diff(a, b)
    r_plus, r_minus = _signed_rank_sums(d, zero_method)
    total = r_plus + r_minus
    if total == 0:
        return 0.0
    return float((r_plus - r_minus) / total)


def holm(pvalues, alpha=0.05):
    """Holm step-down correction over one family of tests.

    The i-th smallest of m p-values is multiplied by (m - i + 1), capped at
    1, and made monotone non-decreasing in the sorted order. A hypothesis is
    rejected when its adjusted p-value is <= alpha. NaN p-values are left out
    of the family (m counts finite values only) and returned as NaN with
    reject False.

    Returns (adjusted_pvalues, reject) as numpy arrays in the input order.
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    adj = np.full(p.shape, np.nan)
    reject = np.zeros(p.shape, dtype=bool)
    ok = np.flatnonzero(np.isfinite(p))
    m = ok.size
    if m == 0:
        return adj, reject
    order = ok[np.argsort(p[ok], kind="mergesort")]
    scaled = (m - np.arange(m)) * p[order]
    stepped = np.minimum(1.0, np.maximum.accumulate(scaled))
    adj[order] = stepped
    reject[order] = stepped <= alpha
    return adj, reject


def cohen_dz(a, b):
    """Cohen's d_z for paired samples: mean(a - b) / sd(a - b), ddof=1.

    Returns NaN for fewer than two pairs or when the differences have zero
    spread and zero mean, and +/-inf when they have zero spread but a
    nonzero mean.
    """
    d = _paired_diff(a, b)
    if d.size < 2:
        return float("nan")
    sd = d.std(ddof=1)
    mu = d.mean()
    if sd == 0:
        return float("nan") if mu == 0 else math.copysign(math.inf, mu)
    return float(mu / sd)


def gap_vs_reference(makespans, reference_makespans):
    """Per-instance percentage gap to a reference and the mean gap.

    gap_i = 100 * (m_i - ref_i) / ref_i. Returns (per_instance_gaps, mean_gap).
    """
    m = np.asarray(makespans, dtype=float).ravel()
    ref = np.asarray(reference_makespans, dtype=float).ravel()
    if m.shape != ref.shape:
        raise ValueError(f"lengths differ: {m.size} vs {ref.size}")
    if np.any(ref <= 0):
        raise ValueError("reference makespans must be positive")
    gaps = 100.0 * (m - ref) / ref
    return gaps, float(gaps.mean())


def compare(a, b, n_boot=10000, seed=0):
    """All paired quantities for A versus B, as a dict."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    diff, ci = paired_mean_diff_ci(a, b, n_boot=n_boot, seed=seed)
    return {
        "n": int(a.size),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": diff,
        "ci": ci,
        "p_wilcoxon": wilcoxon_paired(a, b),
        "rank_biserial": rank_biserial(a, b),
        "cohen_dz": cohen_dz(a, b),
    }


# --------------------------------------------------------------------------- #
# Result-file loading
# --------------------------------------------------------------------------- #
def result_path(results_dir, dataset, tag):
    return os.path.join(results_dir, dataset, f"Result_{tag}_{dataset}.npy")


def seed_files(results_dir, dataset, tag):
    """The file for `tag` plus every file tagged "<tag>-s<digits>"."""
    base = result_path(results_dir, dataset, tag)
    files = [base] if os.path.isfile(base) else []
    pattern = re.compile(re.escape(f"Result_{tag}-s") + r"\d+" +
                         re.escape(f"_{dataset}.npy") + "$")
    folder = os.path.join(results_dir, dataset)
    for f in sorted(glob.glob(os.path.join(folder, f"Result_{tag}-s*_{dataset}.npy"))):
        if pattern.match(os.path.basename(f)):
            files.append(f)
    return files


def load_method(results_dir, dataset, tag, metric="makespan", pool_seeds=False):
    """Load one method's per-instance values.

    Returns (values, files_used). With pool_seeds, values is the per-instance
    mean over all seed files, truncated to the shortest file.
    """
    col = METRIC_COLUMNS[metric]
    files = (seed_files(results_dir, dataset, tag) if pool_seeds
             else [result_path(results_dir, dataset, tag)])
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise FileNotFoundError(
            f"no result file for tag {tag!r}: "
            f"{result_path(results_dir, dataset, tag)}")
    arrays = []
    for f in files:
        arr = np.load(f)
        if arr.ndim != 2 or arr.shape[1] <= col:
            raise ValueError(f"unexpected result shape {arr.shape} in {f}")
        arrays.append(arr[:, col].astype(float))
    n = min(x.size for x in arrays)
    return np.mean([x[:n] for x in arrays], axis=0), files


def load_cpsat_reference(or_dir, dataset):
    """instance_id -> lag-aware CP-SAT makespan from <dataset>.jsonl."""
    path = os.path.join(or_dir, dataset + ".jsonl")
    ref = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid, mk = rec.get("instance_id"), rec.get("makespan_lag")
            if iid is not None and mk is not None:
                ref[int(iid)] = float(mk)
    return ref


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt_p(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def _fmt(x, nd=2):
    if x is None or not np.isfinite(x):
        return "n/a" if x is None or np.isnan(x) else ("inf" if x > 0 else "-inf")
    return f"{x:.{nd}f}"


def run_comparisons(dataset, baseline, methods, results_dir=DEFAULT_RESULTS_DIR,
                    or_dir=DEFAULT_OR_DIR, reference=None, pool_seeds=False,
                    metric="makespan", n_boot=10000, seed=0, alpha=0.05):
    """Compute every method-versus-baseline row and the Holm correction."""
    base_vals, base_files = load_method(results_dir, dataset, baseline, metric,
                                        pool_seeds)
    ref_map = load_cpsat_reference(or_dir, dataset) if reference == "cpsat" else None
    rows = []
    for tag in methods:
        vals, files = load_method(results_dir, dataset, tag, metric, pool_seeds)
        n = min(vals.size, base_vals.size)
        a, b = vals[:n], base_vals[:n]
        row = compare(a, b, n_boot=n_boot, seed=seed)
        row.update(tag=tag, files=files, n_seeds=len(files))
        if ref_map is not None:
            ids = [i for i in range(n) if i in ref_map]
            ref = np.array([ref_map[i] for i in ids])
            row["n_ref"] = len(ids)
            row["gap_a"] = gap_vs_reference(a[ids], ref)[1] if ids else float("nan")
            row["gap_b"] = gap_vs_reference(b[ids], ref)[1] if ids else float("nan")
        rows.append(row)
    adj, rej = holm([r["p_wilcoxon"] for r in rows], alpha=alpha)
    for r, pa, rj in zip(rows, adj, rej):
        r["p_holm"], r["reject"] = float(pa), bool(rj)
    return rows, base_files


def format_table(rows, baseline, base_files, reference=None, metric="makespan",
                 alpha=0.05):
    lines = [
        f"Metric: {metric}. A = method, B = baseline `{baseline}` "
        f"({len(base_files)} file(s)). Diff = A - B; negative favours A. "
        f"Wilcoxon: two-sided, zero differences dropped (wilcox). "
        f"Holm over {len(rows)} tests at alpha = {alpha}. "
        f"r = matched-pairs rank-biserial (negative favours A).",
        "",
    ]
    head = ["Method (A)", "seeds", "n", "mean A", "mean B",
            "mean diff [95% CI]", "Wilcoxon p", "Holm p", "r", "dz"]
    if reference == "cpsat":
        head += ["gap A vs CP-SAT %", "gap B vs CP-SAT %"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * (len(head) - 1)) + "|")
    for r in rows:
        cells = [
            f"`{r['tag']}`",
            str(r["n_seeds"]),
            str(r["n"]),
            _fmt(r["mean_a"]),
            _fmt(r["mean_b"]),
            f"{_fmt(r['mean_diff'])} [{_fmt(r['ci'][0])}, {_fmt(r['ci'][1])}]",
            _fmt_p(r["p_wilcoxon"]),
            _fmt_p(r["p_holm"]) + ("*" if r["reject"] else ""),
            _fmt(r["rank_biserial"], 3),
            _fmt(r["cohen_dz"], 3),
        ]
        if reference == "cpsat":
            cells += [f"{_fmt(r['gap_a'])} (n={r['n_ref']})", _fmt(r["gap_b"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"* Holm-adjusted p <= {alpha}.")
    if reference == "cpsat":
        lines.append("CP-SAT reference: lag-aware makespan within its time limit "
                     "(best found, not certified optimal where status is FEASIBLE).")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Paired method-versus-baseline statistics with Holm correction.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--baseline", required=True,
                    help="baseline tag, e.g. SPT+10x25+ppvc-mixed+smoke")
    ap.add_argument("--methods", required=True, nargs="+",
                    help="method tags (file name between 'Result_' and '_<dataset>.npy')")
    ap.add_argument("--reference", choices=["cpsat"], default=None,
                    help="add mean gap columns versus the CP-SAT reference")
    ap.add_argument("--pool-seeds", action="store_true",
                    help="average each tag with its '<tag>-s<digits>' seed files per instance")
    ap.add_argument("--metric", choices=sorted(METRIC_COLUMNS), default="makespan")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--or-dir", default=DEFAULT_OR_DIR)
    args = ap.parse_args(argv)

    rows, base_files = run_comparisons(
        args.dataset, args.baseline, args.methods, results_dir=args.results_dir,
        or_dir=args.or_dir, reference=args.reference, pool_seeds=args.pool_seeds,
        metric=args.metric, n_boot=args.n_boot, seed=args.seed, alpha=args.alpha)
    print(format_table(rows, args.baseline, base_files, reference=args.reference,
                       metric=args.metric, alpha=args.alpha))
    return 0


if __name__ == "__main__":
    sys.exit(main())
