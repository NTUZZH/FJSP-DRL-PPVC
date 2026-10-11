"""Anytime CP-SAT figure: best makespan found so far versus solve time.

Panels: production-batch instances with 50 and 80 modules. Each curve is the
median across instances of the best objective found so far, expressed as the
percent above that instance's SPT dispatching-rule makespan; the band is the
interquartile range. Cold start and warm start from the SPT schedule are
compared, and the SPT rule itself is the horizontal line at 0 percent.

Inputs (read-only):
  or_solution/PPVC/anytime/<dataset>_<mode>_600s_w4.jsonl
  test_results/PPVC/<dataset>/Result_SPT+pdr_<dataset>.npy
Outputs:
  IEEE Paper/draft/figures_src/fig_anytime.pdf and fig_anytime.png
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.abspath(__file__))
ANYTIME_DIR = os.path.join(ROOT, "or_solution", "PPVC", "anytime")
SPT_DIR = os.path.join(ROOT, "test_results", "PPVC")
OUT_DIR = os.path.join(ROOT, "IEEE Paper", "draft", "figures_src")

DATASETS = [("50x25+ppvc-mixed", "(a)"), ("80x25+ppvc-mixed", "(b)")]
MODES = ["cold", "warm-spt"]
MIN_CPU_SHARE = 0.9
T_MIN, T_MAX = 0.5, 600.0
TABLE_TIMES = [10.0, 60.0, 300.0, 600.0]
Y_LO, Y_HI = -5.0, 40.0          # y window in percent above SPT

# House palette: pastel fills with medium-depth lines of the same hue.
STYLE = {
    "cold":     dict(line="#c25b6a", fill="#eda9b0", ls="-",
                     label="Cold start"),
    "warm-spt": dict(line="#4f81ad", fill="#a8c6e3", ls=(0, (5, 1.6)),
                     label="Warm start from SPT"),
}
INK = "#1A1A1A"


def load_records(ds, mode):
    path = os.path.join(ANYTIME_DIR, f"{ds}_{mode}_600s_w4.jsonl")
    with open(path) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    kept = [r for r in recs if r.get("cpu_share", 0.0) >= MIN_CPU_SHARE]
    return recs, kept


def best_so_far(traj, grid):
    """Step function: best objective found at or before each grid time."""
    pts = sorted((float(t), float(o)) for t, o, *_ in traj if o is not None)
    out = np.full(grid.shape, np.inf)
    if not pts:
        return out
    ts = np.array([p[0] for p in pts])
    best = np.minimum.accumulate(np.array([p[1] for p in pts]))
    idx = np.searchsorted(ts, grid, side="right") - 1
    ok = idx >= 0
    out[ok] = best[idx[ok]]
    return out


def percent_matrix(recs, spt, grid):
    rows = []
    for r in recs:
        b = best_so_far(r["trajectory"], grid)
        ref = spt[int(r["instance_id"])]
        rows.append(100.0 * (b / ref - 1.0))
        # The step function at the time limit must equal the reported objective.
        last = best_so_far(r["trajectory"], np.array([T_MAX * 2]))[0]
        assert abs(last - float(r["objective"])) < 1e-6, (r["instance_id"], last)
    return np.vstack(rows)


def quantiles(mat):
    """Median and quartiles across instances; inf marks 'no solution yet'.

    A quantile is reported only when it is finite, i.e. when enough instances
    already hold a solution for that quantile to be defined.
    """
    q = np.quantile(mat, [0.25, 0.5, 0.75], axis=0, method="lower")
    q[~np.isfinite(q)] = np.nan
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired", action="store_true",
                    help="restrict both modes to instance ids present in both")
    args = ap.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman",
                       "TeX Gyre Termes"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black",
        "axes.edgecolor": INK, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })

    grid = np.unique(np.concatenate([np.geomspace(T_MIN, T_MAX, 600),
                                     TABLE_TIMES]))
    t_idx = {t: int(np.searchsorted(grid, t)) for t in TABLE_TIMES}

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.17, top=0.97,
                        wspace=0.08)
    report = []
    for ax, (ds, tag) in zip(axes, DATASETS):
        spt = np.load(os.path.join(SPT_DIR, ds, f"Result_SPT+pdr_{ds}.npy"))[:, 0]
        data = {}
        for mode in MODES:
            raw, kept = load_records(ds, mode)
            data[mode] = (raw, kept)
        if args.paired:
            common = set.intersection(*[{r["instance_id"] for r in data[m][1]}
                                        for m in MODES])
            data = {m: (raw, [r for r in kept if r["instance_id"] in common])
                    for m, (raw, kept) in data.items()}

        ax.axhline(0.0, color=INK, lw=0.8, zorder=2)
        for mode in MODES:
            raw, kept = data[mode]
            mat = percent_matrix(kept, spt, grid)
            q25, q50, q75 = quantiles(mat)
            st = STYLE[mode]
            ax.fill_between(grid, np.clip(q25, Y_LO, Y_HI + 50),
                            np.clip(q75, Y_LO, Y_HI + 50), color=st["fill"],
                            alpha=0.55, lw=0, zorder=1)
            ax.plot(grid, q50, color=st["line"], ls=st["ls"], lw=1.4,
                    zorder=4)
            first_ok = grid[np.isfinite(q50)][0]
            above_clip = q50 > Y_HI
            clip_until = grid[above_clip][-1] if above_clip.any() else None
            row = dict(dataset=ds, mode=mode, n_raw=len(raw), n_used=len(kept),
                       ids=sorted(r["instance_id"] for r in kept),
                       median_first_t=first_ok,
                       median_start=q50[np.isfinite(q50)][0],
                       q75_start=q75[np.isfinite(q75)][0] if np.isfinite(q75).any() else None,
                       clip_until=clip_until)
            for t in TABLE_TIMES:
                row[f"med@{int(t)}"] = q50[t_idx[t]]
            final = mat[:, t_idx[T_MAX]]
            row["share_above_600"] = float(np.mean(final > 0))
            row["n_above_600"] = int(np.sum(final > 0))
            row["share_below_600"] = float(np.mean(final < 0))
            row["n_below_600"] = int(np.sum(final < 0))
            row["min_600"] = float(final.min())
            row["max_600"] = float(final.max())
            report.append(row)
            if clip_until is not None:
                # Clip mark: an upward wedge where the median leaves the window.
                x_enter = grid[np.searchsorted(grid, clip_until) + 1]
                ax.plot([x_enter], [Y_HI], marker="^", ms=4.5,
                        color=st["line"], clip_on=False, zorder=4)

        ax.set_xscale("log")
        ax.set_xlim(T_MIN, T_MAX)
        ax.set_xticks([1, 10, 60, 600])
        ax.set_xticklabels(["1", "10", "60", "600"])
        ax.set_ylim(Y_LO, Y_HI)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.set_xlabel("Solve time (s, log scale)")
        ax.text(0.015, 0.975, tag, transform=ax.transAxes, ha="left",
                va="top", fontsize=8.5, fontweight="bold")
    axes[0].set_ylabel("Makespan above SPT rule (%)")

    handles = [
        (Line2D([], [], color=STYLE["cold"]["line"], ls=STYLE["cold"]["ls"],
                lw=1.4), Patch(color=STYLE["cold"]["fill"], alpha=0.55, lw=0)),
        (Line2D([], [], color=STYLE["warm-spt"]["line"],
                ls=STYLE["warm-spt"]["ls"], lw=1.4),
         Patch(color=STYLE["warm-spt"]["fill"], alpha=0.55, lw=0)),
        Line2D([], [], color=INK, lw=0.8),
    ]
    from matplotlib.legend_handler import HandlerTuple
    leg = axes[1].legend(
        [(h[1], h[0]) if isinstance(h, tuple) else h for h in handles],
        [STYLE["cold"]["label"], STYLE["warm-spt"]["label"], "SPT rule"],
        handler_map={tuple: HandlerTuple(ndivide=1)},
        loc="upper left", bbox_to_anchor=(0.075, 0.985), frameon=True, fancybox=False, framealpha=1.0,
        edgecolor=INK, borderpad=0.5, handlelength=2.4)
    leg.get_frame().set_linewidth(0.5)

    os.makedirs(OUT_DIR, exist_ok=True)
    pdf = os.path.join(OUT_DIR, "fig_anytime.pdf")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(OUT_DIR, "fig_anytime.png"), dpi=200,
                bbox_inches="tight", pad_inches=0.02)

    hdr = ("dataset", "mode", "n_used/n", "med@10", "med@60", "med@300",
           "med@600", "above SPT @600", "below SPT @600", "range @600")
    print(" | ".join(hdr))
    for r in report:
        print(" | ".join([
            r["dataset"], r["mode"], f"{r['n_used']}/{r['n_raw']}",
            *[f"{r[f'med@{int(t)}']:+.2f}" for t in TABLE_TIMES],
            f"{r['n_above_600']}/{r['n_used']} ({100*r['share_above_600']:.0f}%)",
            f"{r['n_below_600']}/{r['n_used']} ({100*r['share_below_600']:.0f}%)",
            f"[{r['min_600']:+.2f}, {r['max_600']:+.2f}]"]))
    for r in report:
        print(f"{r['dataset']} {r['mode']}: median defined from "
              f"{r['median_first_t']:.2f} s at {r['median_start']:+.1f}%; "
              f"Q3 at start {r['q75_start']}; above {Y_HI:.0f}% until "
              f"{r['clip_until']}; ids {r['ids'][0]}..{r['ids'][-1]}")
    print("saved", os.path.relpath(pdf, ROOT))


if __name__ == "__main__":
    main()
