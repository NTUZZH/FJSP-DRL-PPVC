#!/usr/bin/env python
"""
make_figures.py - Publication-quality figure pipeline for the IEEE TII PPVC paper.

Idempotent: re-running regenerates every figure and auto-includes any new data
(extra ablation arms, refreshed CP-SAT references) that has landed since.

Outputs (vector PDF + PNG preview) into IEEE Paper/draft/figures/:
    fig_gap_box        - per-method gap% distribution vs CP-SAT lag-aware reference
    fig_lag_inflation  - CP-SAT lag-free vs lag-aware makespan scatter
    fig_training_curves- validation makespan vs PPO update, per ablation arm

Run:  nice -n 19 python make_figures.py

Missing inputs are skipped gracefully with a log line; the script never crashes
on absent data so it can be re-run safely while the experiment chain produces
more results.
"""

import glob
import json
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(REPO, "test_results", "PPVC", "10x25+ppvc-mixed")
JSONL = os.path.join(REPO, "or_solution", "PPVC", "10x25+ppvc-mixed.jsonl")
TRAIN_LOG_DIR = os.path.join(REPO, "train_log")
FIG_DIR = os.path.join(REPO, "IEEE Paper", "draft", "figures")

DATASET = "10x25+ppvc-mixed"

os.makedirs(FIG_DIR, exist_ok=True)


def log(msg):
    print("[make_figures] " + msg, flush=True)


# ----------------------------------------------------------------------------
# IEEE TII style (pure matplotlib, no seaborn)
# ----------------------------------------------------------------------------
COL_W = 3.5          # single-column width (in)
DPI_PNG = 300

# Colorblind-safe palette (Wong / Okabe-Ito)
C_OURS = "#0072B2"   # blue   - ours / highlight
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_VERM = "#D55E00"
C_PURPLE = "#CC79A7"
C_SKY = "#56B4E9"
C_YELLOW = "#F0E442"
C_GREY = "#7F7F7F"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.0,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,   # embed TrueType (editable text in PDF)
    "ps.fonttype": 42,
})


def save(fig, stem):
    """Write a vector PDF and a PNG preview; report size."""
    pdf = os.path.join(FIG_DIR, stem + ".pdf")
    png = os.path.join(FIG_DIR, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=DPI_PNG)
    plt.close(fig)
    kb = os.path.getsize(pdf) / 1024.0
    flag = "OK" if kb > 2 else "WARN(<2KB)"
    log(f"wrote {stem}.pdf ({kb:.1f} KB) [{flag}] + .png preview")
    return kb


# ----------------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------------
def load_cpsat():
    """Return dict per instance from the CP-SAT jsonl, or None if absent.

    Rows are in jsonl order == sorted instance id order == npy row order.
    """
    if not os.path.exists(JSONL):
        return None
    rows = []
    with open(JSONL) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return None
    rows.sort(key=lambda r: r.get("instance_id", r.get("instance", 0)))
    return rows


def load_method_makespans(method_filename_token):
    """Load col-0 (makespan) array for a Result_<token>_<DATASET>.npy file.

    Returns the 1-D makespan array, or None if missing.
    """
    pat = os.path.join(RESULTS_DIR, f"Result_{method_filename_token}_{DATASET}.npy")
    matches = sorted(glob.glob(pat))
    if not matches:
        return None
    arr = np.load(matches[0])
    if arr.ndim != 2 or arr.shape[1] < 1:
        return None
    return arr[:, 0]


# Method registry for figure 1.
#  key        -> (filename token glob, display label, is_ours)
# PDR baselines carry the '+smoke' tag in filename but ARE valid (PDRs are
# checkpoint-independent). Our method is the greedy decode of the full model.
FIG1_METHODS = [
    ("greedy+10x25+ppvc-mixed+full",  "DANIEL-PPVC (ours)", True),
    ("SPT+10x25+ppvc-mixed*",         "SPT",                False),
    ("MWKR+10x25+ppvc-mixed*",        "MWKR",               False),
    ("FIFO+10x25+ppvc-mixed*",        "FIFO",               False),
    ("MOR+10x25+ppvc-mixed*",         "MOR",                False),
]


# ----------------------------------------------------------------------------
# Figure 1 - per-method gap% boxplot vs CP-SAT lag-aware reference
# ----------------------------------------------------------------------------
def fig_gap_box():
    stem = "fig_gap_box"
    cpsat = load_cpsat()
    if cpsat is None:
        log(f"SKIP {stem}: CP-SAT jsonl not found at {JSONL}")
        return None
    cp_lag = np.array([r["makespan_lag"] for r in cpsat], dtype=float)

    series = []  # (label, gap_array, is_ours)
    for token, label, is_ours in FIG1_METHODS:
        mk = load_method_makespans(token)
        if mk is None:
            log(f"  {stem}: method '{label}' missing (token {token}) - skipped")
            continue
        n = min(len(mk), len(cp_lag))
        if n == 0:
            continue
        gap = (mk[:n] - cp_lag[:n]) / cp_lag[:n] * 100.0
        gap = gap[np.isfinite(gap)]
        series.append((label, gap, is_ours))

    if not series:
        log(f"SKIP {stem}: no method result files found")
        return None

    # order by median ascending; ours stays visually highlighted wherever it lands
    series.sort(key=lambda s: np.median(s[1]))
    labels = [s[0] for s in series]
    data = [s[1] for s in series]
    ours_idx = [i for i, s in enumerate(series) if s[2]]

    fig, ax = plt.subplots(figsize=(COL_W, 2.45))
    positions = np.arange(1, len(data) + 1)

    bp = ax.boxplot(
        data, positions=positions, widths=0.55, orientation="vertical",
        patch_artist=True, showfliers=False,
        medianprops=dict(color="black", linewidth=1.0),
        whiskerprops=dict(linewidth=0.8, color="#444444"),
        capprops=dict(linewidth=0.8, color="#444444"),
        boxprops=dict(linewidth=0.8),
    )
    for i, box in enumerate(bp["boxes"]):
        if i in ours_idx:
            box.set_facecolor(C_OURS)
            box.set_alpha(0.85)
            box.set_edgecolor("black")
        else:
            box.set_facecolor("#D9D9D9")
            box.set_edgecolor("#555555")

    # strip / jitter overlay for the actual per-instance distribution
    rng = np.random.default_rng(0)
    for i, g in enumerate(data):
        x = positions[i] + rng.uniform(-0.16, 0.16, size=len(g))
        is_ours = i in ours_idx
        ax.scatter(
            x, g, s=4, alpha=0.45,
            color=(C_OURS if is_ours else "#555555"),
            edgecolors="none", zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Optimality gap (%)")
    ax.axhline(0.0, color="#999999", linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
    ax.set_ylim(bottom=min(-1, float(min(g.min() for g in data)) - 1))
    ax.grid(axis="y", linestyle="-", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    save(fig, stem)

    # report ranges
    rng_lines = []
    for label, g, _ in series:
        rng_lines.append(
            f"    {label:20s} n={len(g):3d}  median={np.median(g):5.2f}%  "
            f"mean={g.mean():5.2f}%  min={g.min():5.2f}%  max={g.max():5.2f}%"
        )
    log(f"  {stem} data ranges (ordered by median):\n" + "\n".join(rng_lines))
    return {"series": [(l, g) for l, g, _ in series]}


# ----------------------------------------------------------------------------
# Figure 2 - CP-SAT lag-free vs lag-aware makespan scatter (lag inflation)
# ----------------------------------------------------------------------------
def fig_lag_inflation():
    stem = "fig_lag_inflation"
    cpsat = load_cpsat()
    if cpsat is None:
        log(f"SKIP {stem}: CP-SAT jsonl not found at {JSONL}")
        return None

    free, lag, status = [], [], []
    for r in cpsat:
        f = r.get("makespan_free")
        l = r.get("makespan_lag")
        if f is None or l is None:
            continue
        free.append(float(f))
        lag.append(float(l))
        status.append(str(r.get("status_lag", r.get("status", "")) or "").upper())
    if not free:
        log(f"SKIP {stem}: no makespan_free/makespan_lag fields present")
        return None
    free = np.array(free)
    lag = np.array(lag)
    status = np.array(status)

    opt = status == "OPTIMAL"
    feas = ~opt
    # mean per-instance inflation
    infl = (lag - free) / free * 100.0
    mean_infl = float(infl.mean())

    # Square canvas: with set_aspect("equal") the axes is forced square, so a
    # square figure keeps the saved PDF ~column-width and avoids the upscaling
    # (and font enlargement) that a wide-but-short canvas would incur when
    # included at \columnwidth. Keeps tick/label sizes matching the other figs.
    fig, ax = plt.subplots(figsize=(COL_W, COL_W))

    lo = min(free.min(), lag.min())
    hi = max(free.max(), lag.max())
    pad = 0.04 * (hi - lo)
    line = np.array([lo - pad, hi + pad])
    ax.plot(line, line, color="#888888", linestyle=(0, (4, 3)),
            linewidth=0.9, zorder=1, label="$y=x$ (no lag)")

    if opt.any():
        ax.scatter(free[opt], lag[opt], s=18, marker="o",
                   facecolor=C_OURS, edgecolor="black", linewidth=0.4,
                   alpha=0.85, zorder=3, label="OPTIMAL")
    if feas.any():
        ax.scatter(free[feas], lag[feas], s=18, marker="o",
                   facecolor="none", edgecolor=C_VERM, linewidth=0.8,
                   alpha=0.9, zorder=3, label="FEASIBLE")

    ax.set_xlabel("CP-SAT lag-free makespan (h)")
    ax.set_ylabel("CP-SAT lag-aware makespan (h)")
    ax.set_xlim(line[0], line[1])
    ax.set_ylim(line[0], line[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="-", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # All points sit in the upper-left; the lower-right triangle (below y=x) is
    # empty. Stack the annotation (top) and legend (bottom) there, both clear of
    # the diagonal, the point cloud, and each other.
    ax.annotate(
        f"mean inflation\n+{mean_infl:.1f}%",
        xy=(0.97, 0.40), xycoords="axes fraction",
        ha="right", va="bottom", fontsize=7,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#bbbbbb", linewidth=0.6, alpha=0.9),
    )
    ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.04), frameon=False,
              handletextpad=0.4, borderaxespad=0.2, labelspacing=0.4)

    fig.tight_layout()
    save(fig, stem)
    log(f"  {stem}: n={len(free)}  OPTIMAL={int(opt.sum())} FEASIBLE={int(feas.sum())}  "
        f"free[{free.min():.0f},{free.max():.0f}] lag[{lag.min():.0f},{lag.max():.0f}]  "
        f"mean inflation={mean_infl:.1f}%")
    return {"mean_inflation": mean_infl, "n": len(free)}


# ----------------------------------------------------------------------------
# Figure 3 - validation makespan vs PPO update, per ablation arm (10x25 only)
# ----------------------------------------------------------------------------
# Map a log's model name / suffix to a canonical arm label + colour + order.
#   full       -> A3 (full method)
#   a1-bare    -> A1 (bare backbone)
#   a2-lagfeat -> A2 (+ lag features)
#   a0-lagblind-> A0 (lag-blind)
ARM_DEFS = [
    ("a0-lagblind", "A0 (lag-blind)",   C_VERM,   0),
    ("a1-bare",     "A1 (bare)",        C_ORANGE, 1),
    ("a2-lagfeat",  "A2 (+lag feat.)",  C_GREEN,  2),
    ("full",        "A3 (full, ours)",  C_OURS,   3),
]
VALI_EVERY = 10  # validation runs every 10 PPO updates


def classify_arm(model_name, suffix):
    """Return (label, color, order) for a known arm, else None."""
    text = f"{model_name or ''} {suffix or ''}".lower()
    # check most-specific tokens first so 'a1-bare' doesn't fall through to 'full'
    for token, label, color, order in ARM_DEFS:
        if token in text:
            return label, color, order
    # bare suffix 'full' may also be written as '+full'
    if re.search(r"(^|[+_ ])full([+_ ]|$)", text):
        return "A3 (full, ours)", C_OURS, 3
    return None


def parse_training_log(path):
    """Parse one training log. Return dict or None if not a usable 10x25 run.

    Keys: label, color, order, vali (list of floats), data_name.
    """
    try:
        txt = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return None

    m_model = re.search(r"model name\s*:\s*(\S+)", txt)
    m_vali = re.search(r"vali data\s*:\s*(\S+)", txt)
    m_suffix = re.search(r"--model_suffix\s+(\S+)", txt)
    model_name = m_model.group(1) if m_model else None
    suffix = m_suffix.group(1) if m_suffix else None

    # Determine which size this run targets (plot only 10x25 on this figure).
    blob = f"{model_name or ''} {m_vali.group(1) if m_vali else ''} {suffix or ''}"
    if "10x15" in blob or "+tight" in blob.lower():
        return None  # M-tight run -> excluded from the 10x25 training figure
    if "10x25" not in blob:
        # If we cannot positively confirm 10x25, be conservative and skip.
        return None

    arm = classify_arm(model_name, suffix)
    if arm is None:
        return None
    label, color, order = arm

    vali = [float(v) for v in re.findall(r"validation quality is:\s*([0-9.]+)", txt)]
    if not vali:
        return None
    return {
        "label": label, "color": color, "order": order,
        "vali": vali, "model_name": model_name, "path": path,
    }


def collect_training_runs():
    """Scan all candidate logs; keep the longest run per arm label."""
    candidates = []
    candidates += glob.glob(os.path.join(TRAIN_LOG_DIR, "run_*.log"))
    candidates += glob.glob(os.path.join(TRAIN_LOG_DIR, "PPVC", "*.log"))
    by_label = {}
    for path in sorted(candidates):
        parsed = parse_training_log(path)
        if parsed is None:
            continue
        label = parsed["label"]
        # keep the run with the most validation points (the real / latest run)
        if label not in by_label or len(parsed["vali"]) > len(by_label[label]["vali"]):
            by_label[label] = parsed
    return list(by_label.values())


def fig_training_curves():
    stem = "fig_training_curves"
    runs = collect_training_runs()
    if not runs:
        log(f"SKIP {stem}: no usable 10x25 training logs found")
        return None

    runs.sort(key=lambda r: r["order"])

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    rng_lines = []
    for r in runs:
        y = np.array(r["vali"])
        x = (np.arange(1, len(y) + 1)) * VALI_EVERY  # update index
        ax.plot(x, y, color=r["color"], linewidth=1.0, label=r["label"],
                marker=None, zorder=3)
        rng_lines.append(
            f"    {r['label']:18s} pts={len(y):3d}  start={y[0]:.2f}  "
            f"end={y[-1]:.2f}  best={y.min():.2f}  ({os.path.basename(r['path'])})"
        )

    ax.set_xlabel("PPO update")
    ax.set_ylabel("Validation makespan (h)")
    ax.grid(True, linestyle="-", alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper right", frameon=False, ncol=1,
              handletextpad=0.5, labelspacing=0.3, borderaxespad=0.3)

    fig.tight_layout()
    save(fig, stem)
    log(f"  {stem} arms ({len(runs)}):\n" + "\n".join(rng_lines))
    return {"arms": [(r["label"], r["vali"]) for r in runs]}


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def main():
    log(f"output dir: {FIG_DIR}")
    results = {}
    for fn in (fig_gap_box, fig_lag_inflation, fig_training_curves):
        try:
            results[fn.__name__] = fn()
        except Exception as exc:  # never crash the whole pipeline on one figure
            import traceback
            log(f"ERROR in {fn.__name__}: {exc}")
            traceback.print_exc()
            results[fn.__name__] = None
    produced = [k for k, v in results.items() if v is not None]
    skipped = [k for k, v in results.items() if v is None]
    log(f"done. produced: {produced or 'none'}; skipped: {skipped or 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
