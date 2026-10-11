"""Validation curves of the bound comparison (Supplementary Material, S4).

The DANIEL network trained at the full waits without the input aids, under
the lag-augmented bound (models 10x25+ppvc-mixed+a1-2k-s30{1,2,3}) and under
the lag-free bound (10x25+ppvc-mixed+a1nolagbound-2k-s30{1,2,3}). The two
arms differ only in --bound_lag_free (asserted on the saved configurations).

Every 10 PPO updates, train.py greedy-decodes the 100 validation orders of 10
and of 20 modules (full waits) and appends to valicurve_<model>.json the entry
[updates done, selection score, seconds, {vali_sizes, mean_makespan, norm}],
where the selection score is the mean over the two sizes of the mean
makespan divided by a fixed SPT reference (216 h and 265 h). The figure plots
the mean validation makespan itself, one panel per size: a thin line per
seed and a thick line for the mean of the three seeds of each bound.

Inputs (read-only):
  train_log/PPVC/valicurve_<model>.json   searched first; further directories
      may be named with --log_dir. A curve read from such a directory is
      accepted only if that directory's run copy holds a checkpoint
      byte-identical to trained_network/PPVC/<model>.pth and a configuration
      identical to train_log/PPVC/config_<model>.json, so the curve belongs to
      the checkpoint that the paper evaluates.
  train_log/PPVC/config_<model>.json, trained_network/PPVC/<model>.pth
Outputs:
  IEEE Paper/draft/figures_src/fig_bound_curves.pdf   (vector, newtx Times)
  IEEE Paper/draft/figures_src/fig_bound_curves.png   (300 dpi render)
  IEEE Paper/draft/figures_src/fig_bound_curves_data.json
      every plotted curve with the sha256 of its log file and checkpoint;
      make_supplement_tables.py reads this file for the S4 text numbers.

usage:
  python make_fig_bound_curves.py [--log_dir DIR ...]
  python make_fig_bound_curves.py --plot_only        # re-plot from the JSON
Run on the CPU with one thread, e.g.
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      taskset -c 7 python make_fig_bound_curves.py
"""
import argparse
import hashlib
import json
import os
import subprocess

import numpy as np
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_REL = os.path.join("train_log", "PPVC")
CKPT_REL = os.path.join("trained_network", "PPVC")
OUT_DIR = os.path.join(ROOT, "IEEE Paper", "draft", "figures_src")
OUT_NAME = "fig_bound_curves"
TEX_BIN = os.path.expanduser("~/.TinyTeX/bin/x86_64-linux")

SEEDS = (301, 302, 303)
ARMS = (  # key, model template, label, bound_lag_free
    ("lag_augmented", "10x25+ppvc-mixed+a1-2k-s{s}", "Lag-augmented bound", False),
    ("lag_free", "10x25+ppvc-mixed+a1nolagbound-2k-s{s}", "Lag-free bound", True),
)
SIZES = (10, 20)
NORM = (216.0, 265.0)
N_VALI = 200          # validations per run: every 10 updates up to 2000
STEP = 10

# House palette (as in make_fig_anytime_v3.py): medium-depth line colours,
# spaced in luminance so the two bounds separate in greyscale; the mean line
# of the lag-free bound is also dashed.
TEAL = "#1f6e6e"
ROSE = "#cc6d7b"
DASH = (0, (4.0, 1.8))
STYLE = {"lag_augmented": dict(color=TEAL, ls="-"),
         "lag_free": dict(color=ROSE, ls=DASH)}
LW_SEED, LW_MEAN, ALPHA_SEED = 0.5, 1.5, 0.55


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def check_config(model, s, lag_free):
    c = read_json(os.path.join(ROOT, LOG_REL, f"config_{model}.json"))
    want = dict(framework="DANIEL", n_j=10, n_m=25, max_updates=2000,
                validate_timestep=STEP, vali_size=100, seed_train=s,
                lag_scale=1.0, use_lag_features=False, use_type_embedding=False,
                ppvc_lagblind=False, bound_lag_free=lag_free, ppvc_size_mix="",
                vali_sizes_resolved=list(SIZES))
    for k, v in want.items():
        assert c.get(k) == v, f"config_{model}.json: {k} = {c.get(k)!r}, expected {v!r}"
    return c


def find_curve(model, log_dirs):
    """Path of valicurve_<model>.json: train_log/PPVC first, then each extra
    directory, whose run copy must hold the same checkpoint and config."""
    name = f"valicurve_{model}.json"
    p = os.path.join(ROOT, LOG_REL, name)
    if os.path.exists(p):
        return p
    ck = os.path.join(ROOT, CKPT_REL, f"{model}.pth")
    cfg = os.path.join(ROOT, LOG_REL, f"config_{model}.json")
    for d in log_dirs:
        q = os.path.join(d, name)
        if not os.path.exists(q):
            continue
        run_root = os.path.dirname(os.path.dirname(os.path.abspath(d)))
        ck2 = os.path.join(run_root, CKPT_REL, f"{model}.pth")
        cfg2 = os.path.join(os.path.abspath(d), f"config_{model}.json")
        assert os.path.exists(ck2) and sha256(ck2) == sha256(ck), \
            f"{q}: its run copy holds no checkpoint identical to {CKPT_REL}/{model}.pth"
        assert os.path.exists(cfg2) and read_json(cfg2) == read_json(cfg), \
            f"{q}: its run configuration differs from {LOG_REL}/config_{model}.json"
        return q
    raise SystemExit(f"no {name} in {LOG_REL} or in {log_dirs}")


def read_curve(path, model):
    d = read_json(path)
    assert len(d) == N_VALI, f"{model}: {len(d)} validations"
    upd = [int(e[0]) for e in d]
    assert upd == list(range(STEP, STEP * N_VALI + 1, STEP)), f"{model}: updates"
    m = np.array([e[3]["mean_makespan"] for e in d], dtype=float)
    for e in d:
        assert e[3]["vali_sizes"] == list(SIZES) and e[3]["norm"] == list(NORM), \
            f"{model}: validation sizes or references"
    q = np.array([float(e[1]) for e in d])
    score = np.mean(m / np.array(NORM), axis=1)
    assert np.max(np.abs(q - score)) < 1e-9, f"{model}: score is not the normalized mean"
    return upd, m, q


def collect(log_dirs):
    out = {"description": "validation curves of the network trained at the full "
                          "waits under two bounds; makespans are means over the "
                          "100 validation orders of each size (full waits)",
           "updates": list(range(STEP, STEP * N_VALI + 1, STEP)),
           "vali_sizes": list(SIZES), "selection_reference_h": list(NORM),
           "arms": {}}
    cfgs = {}
    for key, tpl, label, lag_free in ARMS:
        runs = []
        for s in SEEDS:
            model = tpl.format(s=s)
            cfgs[(key, s)] = check_config(model, s, lag_free)
            path = find_curve(model, log_dirs)
            upd, m, q = read_curve(path, model)
            runs.append(dict(
                seed=s, model=model,
                log_file=f"{LOG_REL}/valicurve_{model}.json",
                log_sha256=sha256(path),
                checkpoint=f"{CKPT_REL}/{model}.pth",
                checkpoint_sha256=sha256(os.path.join(ROOT, CKPT_REL, f"{model}.pth")),
                makespan_10=[float(x) for x in m[:, 0]],
                makespan_20=[float(x) for x in m[:, 1]],
                score=[float(x) for x in q],
                kept_update=int(upd[int(np.argmin(q))])))
        out["arms"][key] = dict(label=label, bound_lag_free=lag_free, runs=runs)
    # the two arms differ in the bound only (and in the seed and the name)
    for s in SEEDS:
        a, b = cfgs[("lag_augmented", s)], cfgs[("lag_free", s)]
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        assert diff == ["bound_lag_free", "model_suffix"], f"seed {s}: configs differ in {diff}"
    return out


def plot(data):
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
        "pgf.preamble": (r"\makeatletter\@namedef{ver@underscore.sty}{}\makeatother"
                         r"\usepackage[T1]{fontenc}\usepackage{newtxtext}\usepackage{newtxmath}"),
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
        "text.color": "black", "axes.labelcolor": "black",
        "xtick.color": "black", "ytick.color": "black", "axes.edgecolor": "black",
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.grid": True, "grid.color": "#d9d9d9", "grid.linewidth": 0.4,
        "lines.solid_capstyle": "butt", "lines.dash_capstyle": "butt",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    upd = np.array(data["updates"])
    W, H = 6.5, 2.95
    fig = plt.figure(figsize=(W, H))
    left, right, bottom, top = 0.085, 0.968, 0.155, 0.755
    gs = fig.add_gridspec(1, 2, wspace=0.24, left=left, right=right,
                          bottom=bottom, top=top)
    ylims = {10: (210.0, 234.0), 20: (250.0, 314.0)}
    ysteps = {10: 5, 20: 10}
    for col, (size, tag) in enumerate(zip(SIZES, ("(a)", "(b)"))):
        ax = fig.add_subplot(gs[0, col])
        ax.set_axisbelow(True)
        for key, _tpl, _label, _lf in ARMS:
            st = STYLE[key]
            runs = data["arms"][key]["runs"]
            m = np.array([r[f"makespan_{size}"] for r in runs])
            lo, hi = ylims[size]
            assert m.min() >= lo and m.max() <= hi, f"{key} {size}: {m.min()}-{m.max()} outside axis"
            for r in m:
                ax.plot(upd, r, color=st["color"], lw=LW_SEED, ls="-",
                        alpha=ALPHA_SEED, zorder=2)
            ax.plot(upd, m.mean(axis=0), color=st["color"], lw=LW_MEAN, ls=st["ls"],
                    zorder=3)
        ax.set_xlim(0, 2000)
        ax.set_ylim(*ylims[size])
        ax.xaxis.set_major_locator(MultipleLocator(500))
        ax.yaxis.set_major_locator(MultipleLocator(ysteps[size]))
        ax.set_xlabel("PPO update")
        if col == 0:
            ax.set_ylabel("Mean validation makespan (h)")
        ax.annotate(tag, xy=(0, 1), xycoords="axes fraction", xytext=(-4, 6),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=9, fontweight="bold")
    handles, labels = [], []
    for key, _tpl, label, _lf in ARMS:
        st = STYLE[key]
        handles += [Line2D([], [], color=st["color"], lw=LW_MEAN, ls=st["ls"]),
                    Line2D([], [], color=st["color"], lw=LW_SEED, alpha=ALPHA_SEED)]
        labels += [f"{label}, mean of three seeds", f"{label}, single seed"]
    leg = fig.legend(handles, labels, ncol=2, loc="upper center",
                     bbox_to_anchor=(0.5 * (left + right), 0.995), frameon=True,
                     fancybox=False, framealpha=1.0, edgecolor="black",
                     borderpad=0.45, handlelength=2.6, handletextpad=0.5,
                     columnspacing=1.6, labelspacing=0.3)
    leg.get_frame().set_linewidth(0.5)
    pdf = os.path.join(OUT_DIR, OUT_NAME + ".pdf")
    os.environ["PATH"] = TEX_BIN + ":" + os.environ["PATH"]
    fig.savefig(pdf, metadata={"Creator": "Matplotlib", "Producer": "Matplotlib pgf backend"})
    subprocess.run(["pdftoppm", "-png", "-r", "300", "-singlefile", pdf,
                    os.path.join(OUT_DIR, OUT_NAME)], check=True)
    return pdf


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log_dir", action="append", default=[],
                    help="further directory holding valicurve_<model>.json "
                         "(the train_log/PPVC of a run copy of this repository)")
    ap.add_argument("--plot_only", action="store_true",
                    help="re-plot from fig_bound_curves_data.json")
    args = ap.parse_args()
    data_p = os.path.join(OUT_DIR, OUT_NAME + "_data.json")
    if args.plot_only:
        data = read_json(data_p)
    else:
        data = collect(args.log_dir)
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(data_p, "w") as f:
            json.dump(data, f, indent=1)
    pdf = plot(data)
    for key, arm in data["arms"].items():
        for r in arm["runs"]:
            print(f"{key:14s} seed {r['seed']}: kept update {r['kept_update']:5d}, "
                  f"first/last makespan 10: {r['makespan_10'][0]:.1f}/{r['makespan_10'][-1]:.1f}, "
                  f"20: {r['makespan_20'][0]:.1f}/{r['makespan_20'][-1]:.1f}")
    print("saved", os.path.relpath(pdf, ROOT), "and", os.path.relpath(data_p, ROOT))


if __name__ == "__main__":
    main()
