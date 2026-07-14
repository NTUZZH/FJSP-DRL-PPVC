#!/usr/bin/env python3
"""analyze_results.py -- aggregate raw PPVC FJSP-DRL result files into the paper's
final markdown tables with paired significance tests.

PURE READ-ONLY. Never modifies any input file. The only file written is
`IEEE Paper/06_results_tables.md` (regenerated idempotently on every run).

Inputs (all relative to the repo root this script lives in):
  - Per-method results:
      test_results/PPVC/<dataset>/Result_<METHOD>+<model_name>_<dataset>.npy
      shape [N,2] = (makespan, seconds), rows positional 0..N-1 == instance_id.
  - CP-SAT reference:
      or_solution/PPVC/<dataset>.jsonl  (one JSON object per line; read defensively).
  - Dataset meta:
      data/PPVC/<dataset>/dataset_meta.json (optional).

Outputs:
  - IEEE Paper/06_results_tables.md (one section per discovered dataset).
  - The same tables printed to the console.

Run:  python analyze_results.py
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
from collections import OrderedDict

import numpy as np

try:
    from scipy.stats import wilcoxon
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy is expected to be present
    _HAVE_SCIPY = False


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO = os.path.dirname(os.path.abspath(__file__))
TEST_RESULTS_DIR = os.path.join(REPO, "test_results", "PPVC")
OR_SOLUTION_DIR = os.path.join(REPO, "or_solution", "PPVC")
DATA_DIR = os.path.join(REPO, "data", "PPVC")
OUT_MD = os.path.join(REPO, "IEEE Paper", "06_results_tables.md")

# Canonical PDR method names (checkpoint-independent priority dispatch rules).
PDR_METHODS = {"FIFO", "MOR", "SPT", "MWKR", "LWKR", "EDD", "RANDOM"}
# Methods that come from a trained model checkpoint (variant-dependent).
MODEL_METHODS = {"greedy", "sampling", "A0REPAIR", "A0BLIND"}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _mtime_str(path: str) -> str:
    try:
        ts = os.path.getmtime(path)
        return _dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except OSError:
        return "??"


def _fmt(x, nd=1):
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "n/a"
    return f"{x:.{nd}f}"


def _star(p):
    """Significance star for a p-value."""
    if p is None or np.isnan(p):
        return ""
    if p < 0.01:
        return "**"   # p<0.01
    if p < 0.05:
        return "*"    # p<0.05
    return ""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
class MethodResult:
    """One npy results file parsed into method/model/variant + arrays."""

    def __init__(self, path, method, model_name, variant, makespan, seconds):
        self.path = path
        self.method = method
        self.model_name = model_name        # e.g. 10x25+ppvc-mixed+full
        self.variant = variant              # last '+'-token, e.g. full / smoke / a1-bare
        self.makespan = makespan            # np.ndarray [N]
        self.seconds = seconds              # np.ndarray [N]
        self.n = makespan.shape[0]
        self.mtime = _mtime_str(path)

    @property
    def is_pdr(self):
        return self.method in PDR_METHODS

    @property
    def is_smoke(self):
        return "smoke" in (self.variant or "") or "smoke" in (self.model_name or "")

    # A short label used in tables: "greedy (full)", "SPT", ...
    @property
    def label(self):
        if self.is_pdr:
            return self.method
        return f"{self.method} ({self.variant})"

    @property
    def key(self):
        """Unique key for de-duplication / lookups."""
        return (self.method, self.variant)


def parse_npy_files(dataset):
    """Discover and load all Result_*.npy files for a dataset.

    Returns a list of MethodResult. Smoke-tagged files are loaded too; the
    selection policy (which to keep per method) is applied by the caller.
    """
    ddir = os.path.join(TEST_RESULTS_DIR, dataset)
    suffix = "_" + dataset + ".npy"
    out = []
    for path in sorted(glob.glob(os.path.join(ddir, "Result_*.npy"))):
        base = os.path.basename(path)
        if not base.startswith("Result_") or not base.endswith(suffix):
            continue
        core = base[len("Result_"):-len(suffix)]   # <METHOD>+<model_name>
        if "+" not in core:
            # No model_name component (e.g. a pure PDR with no tag). Treat
            # whole token as method, empty model.
            method, model_name = core, ""
        else:
            parts = core.split("+")
            method = parts[0]
            model_name = "+".join(parts[1:])
        variant = model_name.split("+")[-1] if model_name else ""
        try:
            arr = np.load(path)
        except Exception as exc:  # pragma: no cover
            print(f"[warn] could not load {path}: {exc}")
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            print(f"[warn] unexpected shape {arr.shape} in {path}; skipping")
            continue
        out.append(
            MethodResult(path, method, model_name, variant,
                         arr[:, 0].astype(float), arr[:, 1].astype(float))
        )
    return out


def select_methods(results):
    """Apply the smoke/variant selection policy.

    Policy (per task spec):
      - Model methods (greedy/sampling/A0*): keep every NON-smoke variant as a
        distinct row (full, a1-bare, a2-lagfeat, ...). Drop smoke variants if a
        non-smoke variant of the SAME method exists; if a model method ONLY has
        a smoke file, keep it but flag provenance.
      - PDR methods: checkpoint-independent. Keep ONE file per PDR method,
        preferring a non-smoke file, else any available (note provenance).

    Returns (selected_list, notes) where notes maps result.path -> provenance str.
    """
    notes = {}

    # --- model methods: group by method, keep non-smoke variants individually.
    model_by_method = OrderedDict()
    for r in results:
        if r.method in MODEL_METHODS:
            model_by_method.setdefault(r.method, []).append(r)

    selected = []
    for method, group in model_by_method.items():
        non_smoke = [r for r in group if not r.is_smoke]
        if non_smoke:
            # keep each distinct non-smoke variant
            seen = set()
            for r in non_smoke:
                if r.variant in seen:
                    continue
                seen.add(r.variant)
                selected.append(r)
        else:
            # only smoke available -> keep, flag
            r = group[0]
            notes[r.path] = "SMOKE-ONLY (no full checkpoint available)"
            selected.append(r)

    # --- PDR methods: one per method, prefer non-smoke.
    pdr_by_method = OrderedDict()
    for r in results:
        if r.is_pdr:
            pdr_by_method.setdefault(r.method, []).append(r)
    for method, group in pdr_by_method.items():
        non_smoke = [r for r in group if not r.is_smoke]
        chosen = non_smoke[0] if non_smoke else group[0]
        if chosen.is_smoke:
            notes[chosen.path] = (
                "PDR provenance: only a smoke-tagged file was available; "
                "PDRs are checkpoint-independent so values are valid."
            )
        selected.append(chosen)

    # --- any other (unknown) methods: keep as-is, prefer non-smoke per method.
    known = MODEL_METHODS | PDR_METHODS
    other_by_method = OrderedDict()
    for r in results:
        if r.method not in known:
            other_by_method.setdefault(r.method, []).append(r)
    for method, group in other_by_method.items():
        non_smoke = [r for r in group if not r.is_smoke]
        chosen = non_smoke[0] if non_smoke else group[0]
        if chosen.is_smoke:
            notes[chosen.path] = "SMOKE-tagged file used (no non-smoke variant)."
        selected.append(chosen)

    return selected, notes


def load_cpsat(dataset):
    """Load CP-SAT reference jsonl defensively.

    Returns dict keyed by instance_id -> record dict, plus list of warnings.
    Tolerates a truncated final line and partial coverage.
    """
    path = os.path.join(OR_SOLUTION_DIR, dataset + ".jsonl")
    records = {}
    warns = []
    if not os.path.isfile(path):
        return None, records, ["CP-SAT jsonl not found: %s" % path]
    with open(path, "r") as fh:
        for ln, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                warns.append(f"skipped unparseable line {ln + 1} (likely truncated)")
                continue
            iid = obj.get("instance_id")
            if iid is None:
                # fall back to parsing trailing digits of 'instance' field
                inst = obj.get("instance", "")
                digits = "".join(ch for ch in inst if ch.isdigit())
                iid = int(digits) if digits else None
            if iid is None:
                warns.append(f"line {ln + 1} has no instance_id; skipped")
                continue
            records[int(iid)] = obj
    return path, records, warns


def load_meta(dataset):
    path = os.path.join(DATA_DIR, dataset, "dataset_meta.json")
    if os.path.isfile(path):
        try:
            with open(path) as fh:
                return json.load(fh), path
        except Exception:
            return None, path
    return None, path


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def cpsat_makespan_map(cp_records):
    """instance_id -> lag-aware makespan (the reference objective)."""
    out = {}
    for iid, rec in cp_records.items():
        m = rec.get("makespan_lag")
        if m is not None:
            out[int(iid)] = float(m)
    return out


def per_instance_gap(makespan_arr, ref_map):
    """Per-instance gap% vs CP-SAT ref on the covered subset.

    gap_i = 100 * (method_i - ref_i) / ref_i, for instance ids present in ref_map
    and within the method array length. Returns (mean_gap, coverage_count, ids).
    """
    gaps = []
    ids = []
    n = makespan_arr.shape[0]
    for iid, ref in sorted(ref_map.items()):
        if iid < 0 or iid >= n:
            continue
        if ref <= 0:
            continue
        gaps.append(100.0 * (makespan_arr[iid] - ref) / ref)
        ids.append(iid)
    if not gaps:
        return None, 0, []
    return float(np.mean(gaps)), len(gaps), ids


def win_tie_loss(a, b, ids=None):
    """Counts where a < b (win), a == b (tie), a > b (loss) over given ids
    (default: all common positions). Lower makespan is better, so a 'win' for
    `a` means a has the smaller makespan."""
    if ids is None:
        n = min(a.shape[0], b.shape[0])
        ids = range(n)
    w = t = l = 0
    for i in ids:
        if a[i] < b[i]:
            w += 1
        elif a[i] == b[i]:
            t += 1
        else:
            l += 1
    return w, t, l


def paired_wilcoxon(a, b, ids):
    """Wilcoxon signed-rank test on paired (a,b) over instance ids.

    Returns (stat, p, n_pairs, direction) where direction is the sign of
    median(a-b): '<' means a tends to be smaller (better makespan).
    Handles the all-zero-difference degenerate case gracefully.
    """
    ids = list(ids)
    da = np.array([a[i] for i in ids], dtype=float)
    db = np.array([b[i] for i in ids], dtype=float)
    diff = da - db
    n_pairs = len(diff)
    if n_pairs == 0:
        return None, None, 0, ""
    nz = np.count_nonzero(diff)
    if nz == 0:
        return None, None, n_pairs, "= (all differences zero)"
    if not _HAVE_SCIPY:
        return None, None, n_pairs, ""
    try:
        # zero_method='wilcox' drops zero-differences (classic Wilcoxon).
        stat, p = wilcoxon(da, db, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        return None, None, n_pairs, ""
    med = float(np.median(diff))
    if med < 0:
        direction = "a<b (a better)"
    elif med > 0:
        direction = "a>b (b better)"
    else:
        direction = "a≈b"
    return float(stat), float(p), n_pairs, direction


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def build_dataset_section(dataset):
    lines = []
    raw = parse_npy_files(dataset)
    if not raw:
        lines.append(f"## Dataset: `{dataset}`\n")
        lines.append("_No Result_*.npy files found; skipping._\n")
        return "\n".join(lines), {}

    selected, prov_notes = select_methods(raw)
    cp_path, cp_records, cp_warns = load_cpsat(dataset)
    meta, meta_path = load_meta(dataset)
    ref_map = cpsat_makespan_map(cp_records) if cp_records else {}

    # ----- compute per-method stats -----
    rows = []
    for r in selected:
        mean_mk = float(np.mean(r.makespan))
        std_mk = float(np.std(r.makespan, ddof=1)) if r.n > 1 else 0.0
        mean_t = float(np.mean(r.seconds))
        gap, cov, _ = per_instance_gap(r.makespan, ref_map) if ref_map else (None, 0, [])
        rows.append({
            "res": r,
            "mean_mk": mean_mk,
            "std_mk": std_mk,
            "mean_t": mean_t,
            "gap": gap,
            "cov": cov,
        })

    # sort rows by mean makespan ascending (best first)
    rows.sort(key=lambda d: d["mean_mk"])

    # identify the best PDR (lowest mean makespan among PDRs) for w/t/l baseline
    pdr_rows = [d for d in rows if d["res"].is_pdr]
    best_pdr = min(pdr_rows, key=lambda d: d["mean_mk"]) if pdr_rows else None

    # ----- header -----
    lines.append(f"## Dataset: `{dataset}`\n")
    if meta:
        purpose = meta.get("purpose", "")
        ninst = meta.get("n_instances", "")
        seed0 = meta.get("seed0", "")
        lines.append(
            f"*Meta:* purpose = {purpose!r}; declared n_instances = {ninst}; "
            f"seed0 = {seed0}.\n"
        )
    ns = sorted({d["res"].n for d in rows})
    lines.append(f"*Instances per method (npy rows):* {ns}.\n")

    # ----- main table -----
    lines.append("### Main results\n")
    base_lbl = best_pdr["res"].label if best_pdr else "best PDR"
    lines.append(
        f"Gap% is computed per-instance against the CP-SAT lag-aware objective "
        f"(`makespan_lag`) on the covered subset, then averaged. "
        f"Win/Tie/Loss is vs the best PDR (**{base_lbl}**) on common instances "
        f"(lower makespan = win).\n"
    )
    header = (
        "| Method | Mean ± Std makespan | Mean time/inst (s) | "
        "Gap% vs CP-SAT (cov) | W/T/L vs best PDR |"
    )
    sep = "|---|---|---|---|---|"
    lines.append(header)
    lines.append(sep)
    for d in rows:
        r = d["res"]
        wtl = ""
        if best_pdr is not None and r is not best_pdr["res"]:
            n_common = min(r.n, best_pdr["res"].n)
            w, t, l = win_tie_loss(r.makespan, best_pdr["res"].makespan,
                                   range(n_common))
            wtl = f"{w}/{t}/{l}"
        elif best_pdr is not None and r is best_pdr["res"]:
            wtl = "— (baseline)"
        gap_str = (
            f"{_fmt(d['gap'], 1)}% (n={d['cov']})" if d["gap"] is not None
            else "n/a"
        )
        smoke_flag = " ⚠smoke" if (r.is_smoke and r.path in prov_notes) else ""
        lines.append(
            f"| {r.label}{smoke_flag} | {_fmt(d['mean_mk'],1)} ± {_fmt(d['std_mk'],1)} "
            f"| {_fmt(d['mean_t'],3)} | {gap_str} | {wtl} |"
        )
    lines.append("")

    # ----- Wilcoxon block -----
    lines.append("### Pairwise Wilcoxon signed-rank tests\n")
    lines.append(
        "Paired per instance on common ids. `**` = p<0.01, `*` = p<0.05. "
        "Direction is the sign of median(A−B) (lower makespan is better).\n"
    )

    # lookups by method/variant
    by_method = OrderedDict()
    for r in selected:
        by_method.setdefault(r.method, []).append(r)

    def get_one(method, variant=None):
        cands = by_method.get(method, [])
        if not cands:
            return None
        if variant is None:
            return cands[0]
        for r in cands:
            if r.variant == variant:
                return r
        return None

    # the reference model-greedy: prefer 'full', else 'a*', else any greedy
    greedy_all = by_method.get("greedy", [])
    greedy_ref = None
    for pref in ("full",):
        greedy_ref = next((r for r in greedy_all if r.variant == pref), None)
        if greedy_ref:
            break
    if greedy_ref is None and greedy_all:
        # prefer any non-smoke
        non_smoke = [r for r in greedy_all if not r.is_smoke]
        greedy_ref = non_smoke[0] if non_smoke else greedy_all[0]

    comparisons = []  # (label, A_res, B_res)

    # model-greedy vs best PDR
    if greedy_ref is not None and best_pdr is not None:
        comparisons.append(
            (f"{greedy_ref.label} vs {best_pdr['res'].label} (best PDR)",
             greedy_ref, best_pdr["res"]))

    # model-greedy vs each OTHER greedy variant (ablation: full vs a1-bare vs a2-lagfeat)
    if greedy_ref is not None:
        for r in greedy_all:
            if r is greedy_ref:
                continue
            if r.is_smoke and r.path in prov_notes:
                # still compare, but it's a smoke variant — label it
                pass
            comparisons.append(
                (f"{greedy_ref.label} vs {r.label} (ablation)", greedy_ref, r))

    # sampling vs greedy
    samp = get_one("sampling")
    if samp is not None and greedy_ref is not None:
        comparisons.append((f"sampling vs {greedy_ref.label}", samp, greedy_ref))

    # A0REPAIR vs model-greedy
    a0r = get_one("A0REPAIR")
    if a0r is not None and greedy_ref is not None:
        comparisons.append((f"A0REPAIR vs {greedy_ref.label}", a0r, greedy_ref))

    if not comparisons:
        lines.append("_No paired comparisons available (insufficient methods)._\n")
    else:
        lines.append("| Comparison (A vs B) | n pairs | Wilcoxon stat | p-value | sig | direction |")
        lines.append("|---|---|---|---|---|---|")
        for lbl, A, B in comparisons:
            n_common = min(A.n, B.n)
            ids = range(n_common)
            stat, p, npairs, direction = paired_wilcoxon(A.makespan, B.makespan, ids)
            if p is None and npairs == 0:
                lines.append(f"| {lbl} | 0 | — | — | | (no common instances) |")
                continue
            if p is None:
                # degenerate (all-zero diffs or scipy missing)
                lines.append(
                    f"| {lbl} | {npairs} | — | — | | {direction or 'n/a'} |")
                continue
            pstr = f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"
            lines.append(
                f"| {lbl} | {npairs} | {_fmt(stat,1)} | {pstr} | {_star(p)} | {direction} |"
            )
    lines.append("")

    # ----- CP-SAT block -----
    lines.append("### CP-SAT reference (lag-aware)\n")
    if not cp_records:
        lines.append("_No CP-SAT reference available for this dataset._\n")
    else:
        n_total = max((d["res"].n for d in rows), default=0)
        cov = len(cp_records)
        statuses = [str(rec.get("status_lag", rec.get("status", "?")))
                    for rec in cp_records.values()]
        n_opt = sum(1 for s in statuses if s.upper() == "OPTIMAL")
        n_feas = sum(1 for s in statuses if s.upper() == "FEASIBLE")
        lag_vals = [float(rec["makespan_lag"]) for rec in cp_records.values()
                    if rec.get("makespan_lag") is not None]
        free_vals = []
        infl_pairs = []
        for rec in cp_records.values():
            ml = rec.get("makespan_lag")
            mf = rec.get("makespan_free")
            if mf is not None:
                free_vals.append(float(mf))
            if ml is not None and mf is not None and float(mf) > 0:
                infl_pairs.append(100.0 * (float(ml) - float(mf)) / float(mf))
        mean_lag = float(np.mean(lag_vals)) if lag_vals else None
        mean_free = float(np.mean(free_vals)) if free_vals else None
        mean_infl = float(np.mean(infl_pairs)) if infl_pairs else None

        lines.append(
            f"- Coverage: {cov} / {n_total} instances "
            f"({100.0 * cov / n_total:.1f}%)." if n_total else
            f"- Coverage: {cov} instances."
        )
        lines.append(f"- Status: {n_opt} OPTIMAL, {n_feas} FEASIBLE "
                     f"(others: {cov - n_opt - n_feas}).")
        lines.append(f"- Mean lag-aware makespan: {_fmt(mean_lag,2)}; "
                     f"mean lag-free makespan: {_fmt(mean_free,2)}.")
        lines.append(f"- Mean per-instance lag inflation: {_fmt(mean_infl,2)}%.")
        lines.append("")

    # ----- provenance footnotes -----
    lines.append("### Provenance\n")
    lines.append("Every row is fed by exactly one npy file (path, mtime, instance count):\n")
    for d in rows:
        r = d["res"]
        rel = os.path.relpath(r.path, REPO)
        note = prov_notes.get(r.path, "")
        note = f" — {note}" if note else ""
        lines.append(f"- **{r.label}**: `{rel}` (mtime {r.mtime}, N={r.n}){note}")
    if cp_path:
        lines.append(f"- **CP-SAT ref**: `{os.path.relpath(cp_path, REPO)}` "
                     f"(mtime {_mtime_str(cp_path)}, coverage={len(cp_records)})")
    if cp_warns:
        for w in cp_warns:
            lines.append(f"  - CP-SAT note: {w}")
    if meta:
        lines.append(f"- **dataset_meta**: `{os.path.relpath(meta_path, REPO)}`")
    lines.append("")

    return "\n".join(lines), prov_notes


def discover_datasets():
    if not os.path.isdir(TEST_RESULTS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(TEST_RESULTS_DIR)):
        full = os.path.join(TEST_RESULTS_DIR, name)
        if os.path.isdir(full):
            out.append(name)
    return out


def main():
    datasets = discover_datasets()
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = []
    header.append("# PPVC FJSP-DRL — Results Tables\n")
    header.append(
        f"_Auto-generated by `analyze_results.py` on {now}. "
        f"Read-only aggregation of `test_results/PPVC/` and `or_solution/PPVC/`. "
        f"Do not edit by hand — re-run the script._\n"
    )
    if not _HAVE_SCIPY:
        header.append("> **Warning:** scipy not available — Wilcoxon tests skipped.\n")
    if not datasets:
        header.append("_No datasets found under `test_results/PPVC/`._\n")

    sections = []
    for ds in datasets:
        sec, _ = build_dataset_section(ds)
        sections.append(sec)

    full_md = "\n".join(header) + "\n" + "\n---\n\n".join(sections) + "\n"

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as fh:
        fh.write(full_md)

    print(full_md)
    print(f"\n[written] {OUT_MD}")


if __name__ == "__main__":
    main()
