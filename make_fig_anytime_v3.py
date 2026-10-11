"""Anytime CP-SAT figure (single column): mean best makespan found so far
versus solver time, for production batches of 50 and 80 modules.

Curves (mean over the 50 test instances of the best makespan found so far):
  CP-SAT from scratch, CP-SAT started from the SPT rule's schedule, and CP-SAT
  started from the SLAB policy's schedule (trained seed 301).
Horizontal reference lines: the SLAB policy (three-seed mean) and the SPT rule,
read from the manuscript macro file, never typed in.

Each panel uses a broken y axis: the lower window holds the warm-started
curves and both reference lines; the upper strip holds the early part of the
from-scratch curve, which starts several hundred hours higher.

Inputs (read-only):
  or_solution/PPVC/anytime/<ds>_{cold,warm-spt,warm-policy-halflag-s301}_600s_w4.jsonl
  test_results/PPVC/<ds>/Result_SPT+pdr_<ds>.npy
  paper_tables/macros_results.json (or $PAPER_OUT_DIR/macros_results.json)
Outputs:
  IEEE Paper/draft/figures_src/fig_anytime_v3.pdf (vector, newtx Times fonts)
  IEEE Paper/draft/figures_src/fig_anytime_v3.png (300 dpi render of the PDF)
"""
import json
import os
import subprocess

import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.abspath(__file__))
ANYTIME_DIR = os.path.join(ROOT, "or_solution", "PPVC", "anytime")
RES_DIR = os.path.join(ROOT, "test_results", "PPVC")
MACROS = os.path.join(os.environ.get("PAPER_OUT_DIR", os.path.join(ROOT, "paper_tables")),
                      "macros_results.json")
OUT_DIR = os.path.join(ROOT, "IEEE Paper", "draft", "figures_src")
OUT_NAME = "fig_anytime_v3"
TEX_BIN = os.path.expanduser("~/.TinyTeX/bin/x86_64-linux")

N_INST = 50
MIN_CPU_SHARE = 0.9
T_MIN, T_MAX = 0.5, 600.0
MODES = ["cold", "warm-spt", "warm-policy-halflag-s301"]

# Per panel: dataset, tag, macro suffix, lower window, upper window, ticks.
PANELS = [
    dict(ds="50x25+ppvc-mixed", tag="(a)", key="Fifty",
         lo=(400.0, 482.0), lo_ticks=[400, 420, 440, 460, 480],
         hi=(490.0, 830.0), hi_ticks=[600, 800]),
    dict(ds="80x25+ppvc-mixed", tag="(b)", key="Eighty",
         lo=(554.0, 632.0), lo_ticks=[560, 580, 600, 620],
         hi=(690.0, 1215.0), hi_ticks=[800, 1000, 1200]),
]

# House palette: one hue per starting schedule, medium-depth line variants,
# spaced in luminance so the series also separate in greyscale.
GREY = "#ababab"   # no starting schedule (context curve)
ROSE = "#cc6d7b"   # SPT rule and CP-SAT started from it
TEAL = "#1f6e6e"   # SLAB policy and CP-SAT started from it
DASH = (0, (4.0, 1.8))
STYLE = {
    "cold": dict(color=GREY, lw=1.1, ls="-", label="CP-SAT from scratch"),
    "warm-spt": dict(color=ROSE, lw=1.1, ls="-", label="CP-SAT from the SPT schedule"),
    "warm-policy-halflag-s301": dict(color=TEAL, lw=1.5, ls="-",
                                     label="CP-SAT from the SLAB schedule"),
    "ref-slab": dict(color=TEAL, lw=0.9, ls=DASH, label="SLAB policy, three seeds"),
    "ref-spt": dict(color=ROSE, lw=0.9, ls=DASH, label="SPT rule"),
}


def luminance(hex_color):
    h = hex_color.lstrip("#")
    lin = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255
        lin.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def load(ds, mode):
    path = os.path.join(ANYTIME_DIR, f"{ds}_{mode}_600s_w4.jsonl")
    with open(path) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    recs.sort(key=lambda r: int(r["instance_id"]))
    ids = [int(r["instance_id"]) for r in recs]
    assert ids == list(range(N_INST)), (path, ids[:3], len(ids))
    shares = [float(r["cpu_share"]) for r in recs]
    assert min(shares) >= MIN_CPU_SHARE, (path, min(shares))
    return recs


def mean_best_so_far(recs):
    """Exact mean step function of the per-instance best-so-far makespan.

    Improvements logged a few milliseconds after the 600 s limit (solver
    shutdown) are clamped to 600 s, so the final value equals the reported
    objective. A warm start's hint is logged at t = 0. The mean is defined
    from the time at which every instance holds a solution.
    """
    trajs = []
    for r in recs:
        pts = sorted((min(float(t), T_MAX), float(o)) for t, o, *_ in r["trajectory"]
                     if o is not None)
        ts = np.array([p[0] for p in pts])
        best = np.minimum.accumulate(np.array([p[1] for p in pts]))
        assert abs(best[-1] - float(r["objective"])) < 1e-9, r["instance_id"]
        trajs.append((ts, best))
    t_all = max(ts[0] for ts, _ in trajs)            # every instance solved
    knots = np.unique(np.concatenate(
        [[max(t_all, T_MIN), T_MAX]] + [ts[(ts > max(t_all, T_MIN)) & (ts <= T_MAX)]
                                        for ts, _ in trajs]))
    vals = np.zeros_like(knots)
    for ts, best in trajs:
        idx = np.searchsorted(ts, knots, side="right") - 1
        assert (idx >= 0).all()
        vals += best[idx]
    return t_all, knots, vals / len(trajs)


def value_at(knots, vals, t):
    i = np.searchsorted(knots, t, side="right") - 1
    return float(vals[i]) if i >= 0 else float("nan")


def main():
    macros = json.load(open(MACROS))
    raw = lambda k: float(macros[k]["raw"])

    plt.rcParams.update({
        "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
        "pgf.preamble": (r"\makeatletter\@namedef{ver@underscore.sty}{}\makeatother"
                         r"\usepackage[T1]{fontenc}\usepackage{newtxtext}\usepackage{newtxmath}"),
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    lums = {name: round(luminance(c), 3) for name, c in
            [("grey", GREY), ("rose", ROSE), ("teal", TEAL)]}

    W, H = 3.5, 2.78
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.34, 0.66], hspace=0.07, wspace=0.30,
                          left=0.135, right=0.962, top=0.726, bottom=0.135)
    report = []
    for col, p in enumerate(PANELS):
        ds, key = p["ds"], p["key"]
        ax_hi = fig.add_subplot(gs[0, col])
        ax_lo = fig.add_subplot(gs[1, col], sharex=ax_hi)

        # ---- reference lines from the macro file, cross-checked at source
        h_star = raw(f"Hstar{key}")
        spt_ref = raw(f"Spt{key}")
        spt = np.load(os.path.join(RES_DIR, ds, f"Result_SPT+pdr_{ds}.npy"))[:, 0]
        assert spt.shape == (N_INST,) and abs(spt.mean() - spt_ref) < 1e-6
        seeds = []
        for s in (301, 302, 303):
            f = os.path.join(RES_DIR, ds, f"Result_greedy+10x25+ppvc-mixed+a1lagx05-2k-s{s}_{ds}.npy")
            arr = np.load(f)
            seeds.append(arr[:, 0].mean() if arr.ndim == 2 else arr.mean())
        assert abs(np.mean(seeds) - h_star) < 1e-6, (np.mean(seeds), h_star)

        for ax in (ax_hi, ax_lo):
            for ref, val in (("ref-spt", spt_ref), ("ref-slab", h_star)):
                st = STYLE[ref]
                ax.axhline(val, color=st["color"], lw=st["lw"], ls=st["ls"], zorder=2)

        # ---- CP-SAT curves
        row = dict(dataset=ds, Hstar=h_star, SPT=spt_ref, seeds=[round(x, 2) for x in seeds])
        for mode in MODES:
            recs = load(ds, mode)
            if mode == "warm-spt":
                hint = np.array([float(r["hint_makespan"]) for r in recs])
                assert np.array_equal(hint, spt), "warm start is not the SPT schedule"
            t_all, knots, vals = mean_best_so_far(recs)
            st = STYLE[mode]
            for ax in (ax_hi, ax_lo):
                ax.plot(knots, vals, drawstyle="steps-post", color=st["color"], lw=st["lw"],
                        ls=st["ls"], zorder=4 if mode != "cold" else 3)
            end = value_at(knots, vals, T_MAX)
            macro_end = {"cold": f"CpCold{key}", "warm-spt": f"CpWarmSpt{key}",
                         "warm-policy-halflag-s301": f"CpWarmPol{key}"}[mode]
            assert abs(end - raw(macro_end)) < 0.006, (mode, end, raw(macro_end))
            if mode == "warm-policy-halflag-s301":
                start = np.mean([float(r["hint_makespan"]) for r in recs])
                assert abs(start - raw(f"CpWarmPolHint{key}")) < 1e-6
                assert abs(vals[0] - start) < 1e-9
            row[mode] = dict(first_all_solved_s=round(t_all, 2), start=round(vals[0], 2),
                             **{f"at{int(t)}s": round(value_at(knots, vals, t), 2)
                                for t in (1, 10, 60, 300, 600)})
            if mode == "cold":
                below_hi = knots[vals < p["hi"][0]]
                below_lo = knots[vals < p["lo"][1]]
                row[mode]["leaves_upper_strip_s"] = round(float(below_hi[0]), 1) if len(below_hi) else None
                row[mode]["enters_lower_window_s"] = round(float(below_lo[0]), 1) if len(below_lo) else None
        report.append(row)

        # ---- axes, ticks, break marks
        ax_hi.set_ylim(*p["hi"])
        ax_lo.set_ylim(*p["lo"])
        ax_hi.set_yticks(p["hi_ticks"])
        ax_lo.set_yticks(p["lo_ticks"])
        ax_lo.set_xscale("log")
        ax_lo.set_xlim(T_MIN, T_MAX)
        ax_lo.set_xticks([1, 10, 60, 600])
        ax_lo.set_xticklabels(["1", "10", "60", "600"])
        ax_hi.spines["bottom"].set_visible(False)
        ax_hi.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        mk = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=5, linestyle="none",
                  color="black", mec="black", mew=0.6, clip_on=False, zorder=6)
        ax_hi.plot([0], [0], transform=ax_hi.transAxes, **mk)
        ax_lo.plot([0], [1], transform=ax_lo.transAxes, **mk)
        ax_hi.annotate(p["tag"], xy=(0, 1), xycoords="axes fraction", xytext=(-14, 5),
                       textcoords="offset points", ha="right", va="bottom", fontsize=8,
                       fontweight="bold")

    fig.text(0.5 * (0.135 + 0.962), 0.012, "Solver time (s, log scale)",
             ha="center", va="bottom", fontsize=8)
    fig.text(0.012, 0.5 * (0.135 + 0.726), "Mean makespan (h)", rotation=90,
             ha="left", va="center", fontsize=8)

    # Two columns, filled column-wise: CP-SAT curves left, reference lines right.
    cols = ["cold", "warm-spt", "warm-policy-halflag-s301", "ref-spt", "ref-slab", None]
    handles = [Line2D([], [], color=STYLE[k]["color"], lw=STYLE[k]["lw"], ls=STYLE[k]["ls"])
               if k else Line2D([], [], lw=0) for k in cols]
    labels = [STYLE[k]["label"] if k else "" for k in cols]
    leg = fig.legend(handles, labels, ncol=2, loc="upper center",
                     bbox_to_anchor=(0.5 * (0.0 + 1.0), 0.995), frameon=True, fancybox=False,
                     framealpha=1.0, edgecolor="black", borderpad=0.45, handlelength=2.3,
                     handletextpad=0.5, columnspacing=1.1, labelspacing=0.28)
    leg.get_frame().set_linewidth(0.5)

    os.makedirs(OUT_DIR, exist_ok=True)
    pdf = os.path.join(OUT_DIR, OUT_NAME + ".pdf")
    for path in (pdf, os.path.join(OUT_DIR, OUT_NAME + ".png")):
        assert OUT_NAME in os.path.basename(path)
    os.environ["PATH"] = TEX_BIN + ":" + os.environ["PATH"]
    fig.savefig(pdf, metadata={"Creator": "Matplotlib", "Producer": "Matplotlib pgf backend"})
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", pdf,
                    os.path.join(OUT_DIR, OUT_NAME)], check=True)

    print("luminance (greyscale separation):", lums)
    print(json.dumps(report, indent=1))
    print("saved", os.path.relpath(pdf, ROOT))


if __name__ == "__main__":
    main()
