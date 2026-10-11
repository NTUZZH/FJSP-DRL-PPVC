"""
test_cpsat_anytime.py
---------------------
Tests and cost measurement for eval_cpsat_anytime.py.

Stages (select with --stage, default 'small'):
  model   the anytime model is identical to ortools_solver.fjsp_solver's model
          (proto equality with hints removed), on 10x25 instance 000.
  small   10x25 instances 000-001, 20 s, 4 workers, cold and warm-spt.
          Checks: warm trajectory point at t = 0 equals an independent SPT
          rollout makespan; the first CP-SAT solution under the hint is no
          worse than the hint; final objective <= hinted makespan; validator
          reports 0 violations; objective >= bound; a second run on the same
          file skips the done instances; a run with another configuration on
          the same file is refused.
  medium  50x25 instances 000-001, 60 s, 4 workers, cold and warm-spt.
  large   80x25 instance 000, 60 s, 4 workers, warm-spt.
  policy  warm-policy (greedy schedule of --model_name as the hint) on 10x25
          instances 000-001 at 20 s, then 50x25 instance 000 at 60 s.
          Checks: the t = 0 point equals an independent greedy rollout; the
          first CP-SAT solution equals the hinted policy makespan; final
          objective <= hint; validator 0 violations; resume skips done ids;
          a file is refused under another model, worker count or CP-SAT seed.

Every stage prints a measurement table (build seconds, peak RSS, improving
solutions, best objective at 10/30/60 s, final bound) and exits nonzero on any
failed check.

Run pinned, e.g.
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 20-23 \
      python test_cpsat_anytime.py --stage small [--out_dir DIR] [--fallback_root DIR]

--fallback_root names a directory holding <dataset>/instance_XXX.fjs copies,
used only when the repository copy of an instance is missing.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
import contextlib

_p = argparse.ArgumentParser()
_p.add_argument("--stage", default="small",
                choices=("model", "small", "medium", "large", "policy"))
_p.add_argument("--model_name", default="10x25+ppvc-mixed+full")
_p.add_argument("--out_dir", default=None)
_p.add_argument("--fallback_root", default=None)
_A = _p.parse_args()
sys.argv = [sys.argv[0]]

import numpy as np

import eval_cpsat_anytime as ea

REPO = os.path.dirname(os.path.abspath(__file__))
FAILS = []


def check(flag, label):
    print(f"[{'PASS' if flag else 'FAIL'}] {label}", flush=True)
    if not flag:
        FAILS.append(label)
    return flag


def data_dir_for(dataset, ids):
    repo_dir = os.path.join(REPO, "data", "PPVC", dataset)
    need = [f"instance_{i:03d}{ext}" for i in ids for ext in (".fjs", ".meta.json")]
    if all(os.path.exists(os.path.join(repo_dir, n)) for n in need):
        return repo_dir
    if _A.fallback_root:
        alt = os.path.join(_A.fallback_root, dataset)
        if all(os.path.exists(os.path.join(alt, n)) for n in need):
            print(f"  (repository copy missing, using fallback copy of {dataset})")
            return alt
    raise SystemExit(f"instances {ids} of {dataset} not found")


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:10]


def best_at(traj, t, limit=None, status=None):
    """Best objective found by time t; '-' past a non-optimal run's limit."""
    if limit is not None and t > limit and status != "OPTIMAL":
        return "-"
    vals = [p[1] for p in traj if p[0] <= t]
    return min(vals) if vals else None


def run_cfg(dataset, data_dir, mode, limit, workers, ids, out, extra=()):
    argv = ["--dataset", dataset, "--mode", mode, "--time_limit", str(limit),
            "--num_workers", str(workers), "--start", str(ids[0]),
            "--end", str(ids[-1] + 1), "--out", out, "--data_dir", data_dir,
            "--model_name", _A.model_name, *extra]
    return ea.run(ea.parse_args(argv))


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def table(rows):
    hdr = ("dataset", "id", "mode", "lim", "build_s", "peakRSS_MB", "base_MB",
           "n_impr", "obj@0", "obj@10", "obj@30", "obj@60", "final", "bound",
           "status", "solver_s", "load1", "viol")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        tr = r["trajectory"]
        t0 = tr[0][1] if tr and tr[0][0] == 0.0 else None
        cells = (r["dataset"].split("+")[0], r["instance_id"], r["mode"],
                 f"{r['time_limit_s']:g}", f"{r['model_build_s']:.2f}",
                 f"{r['peak_rss_mb']:.0f}", f"{r['rss_baseline_mb']:.0f}",
                 r["n_improving_solutions"], t0,
                 *(best_at(tr, t, r["time_limit_s"], r["status"]) for t in (10, 30, 60)),
                 r["objective"], r["bound"], r["status"],
                 f"{r['solver_wall_s']:.1f}", r["loadavg_start"][0],
                 r["validator_n_violations"])
        print("| " + " | ".join(str(c) for c in cells) + " |")


def common_checks(recs):
    for r in recs:
        tag = f"{r['dataset']} {r['instance']} {r['mode']}"
        check(r["status"] in ("OPTIMAL", "FEASIBLE"), f"{tag}: solution found ({r['status']})")
        check(r["validator_n_violations"] == 0, f"{tag}: validator 0 violations")
        check(r["objective"] == r["validator_makespan"], f"{tag}: objective == validator makespan")
        check(r["bound"] <= r["objective"] + 1e-9, f"{tag}: bound <= objective")
        objs = [p[1] for p in r["trajectory"]]
        check(all(a >= b for a, b in zip(objs, objs[1:])), f"{tag}: trajectory non-increasing")
        check(objs[-1] == r["objective"], f"{tag}: last trajectory point == final objective")
        if r["mode"] == "warm-spt":
            check(r["trajectory"][0][0] == 0.0 and r["trajectory"][0][1] == r["spt_makespan"],
                  f"{tag}: t=0 point is the SPT makespan {r['spt_makespan']}")
        if r["mode"] == "warm-policy":
            check(r["trajectory"][0][0] == 0.0
                  and r["trajectory"][0][1] == r["policy_makespan"] == r["hint_makespan"],
                  f"{tag}: t=0 point is the hinted policy makespan {r['policy_makespan']}")
            check(r["hint_first_solution"] is not None
                  and r["hint_first_solution"][1] == r["policy_makespan"],
                  f"{tag}: first CP-SAT solution {r['hint_first_solution']} equals the "
                  f"hinted policy makespan {r['policy_makespan']}")
        if r["mode"] in ("warm-spt", "warm-policy"):
            check(r["hint_taken"], f"{tag}: first CP-SAT solution {r['hint_first_solution']} "
                                   f"<= hint {r['hint_makespan']}")
            check(r["objective"] <= r["hint_makespan"], f"{tag}: final objective <= hinted")


def stage_model():
    """Proto of build_lag_model == proto fjsp_solver builds (hints removed)."""
    import ortools_solver
    from ortools.sat.python import cp_model
    from ppvc_instance_generator import load_instance
    ddir = data_dir_for("10x25+ppvc-mixed", [0])
    jl, pt, meta = load_instance(os.path.join(ddir, "instance_000"))
    jobs, nm = ortools_solver.matrix_to_the_format_for_solving(jl, pt)
    captured = {}

    class _Stop(Exception):
        pass

    orig = cp_model.CpSolver.Solve

    def fake_solve(self, model, cb=None):
        snapshot = type(model.Proto())()
        snapshot.CopyFrom(model.Proto())
        captured["proto"] = snapshot
        raise _Stop()

    cp_model.CpSolver.Solve = fake_solve
    try:
        lag_free_mine = ea.build_lag_model(jobs, nm, None).model.Proto()
        for tl in (meta["time_lag"], None):
            try:
                ortools_solver.fjsp_solver(jobs, nm, 1, time_lag=tl, return_schedule=True)
            except _Stop:
                pass
            ref = captured["proto"]
            ref_s = type(ref)()
            ref_s.CopyFrom(ref)
            ref_s.ClearField("solution_hint")
            mine = ea.build_lag_model(jobs, nm, tl).model.Proto()
            label = "lag-aware" if tl is not None else "lag-free"
            check(ref_s.SerializeToString(deterministic=True)
                  == mine.SerializeToString(deterministic=True),
                  f"model proto identical to ortools_solver.fjsp_solver ({label}, "
                  f"{len(mine.variables)} vars, {len(mine.constraints)} constraints)")
            if tl is not None:
                check(ref_s.SerializeToString(deterministic=True)
                      != lag_free_mine.SerializeToString(deterministic=True),
                      "comparison is sensitive: lag-free model differs from lag-aware model")
    finally:
        cp_model.CpSolver.Solve = orig


def stage_small(out_dir):
    ds, ids = "10x25+ppvc-mixed", [0, 1]
    ddir = data_dir_for(ds, ids)
    recs = []
    for mode in ("cold", "warm-spt"):
        out = os.path.join(out_dir, f"small_{mode}.jsonl")
        if os.path.exists(out):
            os.remove(out)
        run_cfg(ds, ddir, mode, 20, 4, ids, out)
        got = read_jsonl(out)
        check(sorted(r["instance_id"] for r in got) == ids, f"{mode}: one record per instance")
        recs += got

        # resume: a second run appends nothing
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_cfg(ds, ddir, mode, 20, 4, ids, out)
        check(len(read_jsonl(out)) == len(ids) and "todo=0" in buf.getvalue(),
              f"{mode}: resume skips the {len(ids)} done instances")
        # a different configuration on the same file is refused
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                run_cfg(ds, ddir, mode, 21, 4, ids, out)
            refused = False
        except SystemExit:
            refused = True
        check(refused and len(read_jsonl(out)) == len(ids),
              f"{mode}: a run with another time limit on the same file is refused")

    # t = 0 point equals an independent SPT rollout
    from ppvc_instance_generator import load_instance
    for r in recs:
        if r["mode"] != "warm-spt":
            continue
        jl, pt, meta = load_instance(os.path.join(ddir, r["instance"]))
        mk, _, asg, ct = ea.spt_schedule(jl, pt, meta, 50)
        check(r["trajectory"][0][1] == mk,
              f"{r['instance']}: warm t=0 objective {r['trajectory'][0][1]} == "
              f"independent SPT rollout {mk}")
    common_checks(recs)
    return recs


def refused(fn):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            fn()
        return False
    except SystemExit:
        return True


def stage_policy(out_dir):
    from ppvc_instance_generator import load_instance
    recs = []
    for ds, ids, limit in (("10x25+ppvc-mixed", [0, 1], 20), ("50x25+ppvc-mixed", [0], 60)):
        ddir = data_dir_for(ds, ids)
        out = os.path.join(out_dir, f"{ds.split('+')[0]}_warm-policy.jsonl")
        if os.path.exists(out):
            os.remove(out)
        run_cfg(ds, ddir, "warm-policy", limit, 4, ids, out)
        got = read_jsonl(out)
        check(sorted(r["instance_id"] for r in got) == ids, f"{ds}: one record per instance")
        check(all(r["model_name"] == _A.model_name for r in got),
              f"{ds}: records carry model_name {_A.model_name}")
        if ds.startswith("10x25"):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                run_cfg(ds, ddir, "warm-policy", limit, 4, ids, out)
            check(len(read_jsonl(out)) == len(ids) and "todo=0" in buf.getvalue(),
                  "warm-policy: resume skips the done instances")
            n = len(ids)
            check(refused(lambda: ea.run(ea.parse_args(
                ["--dataset", ds, "--mode", "warm-policy", "--time_limit", str(limit),
                 "--num_workers", "4", "--start", "0", "--end", str(n), "--out", out,
                 "--data_dir", ddir, "--model_name", _A.model_name + "-s301"])))
                  and len(read_jsonl(out)) == n,
                  "warm-policy: a run with another model on the same file is refused")
            check(refused(lambda: run_cfg(ds, ddir, "warm-policy", limit, 2, ids, out))
                  and len(read_jsonl(out)) == n,
                  "warm-policy: a run with another worker count is refused")
            check(refused(lambda: run_cfg(ds, ddir, "warm-policy", limit, 4, ids, out,
                                          ("--random_seed", "1")))
                  and len(read_jsonl(out)) == n,
                  "warm-policy: a run with another CP-SAT random_seed is refused")
        for r in got:
            jl, pt, meta = load_instance(os.path.join(ddir, r["instance"]))
            mk, _, _, _ = ea.policy_schedule(jl, pt, meta, _A.model_name, ddir)
            check(r["trajectory"][0][1] == mk,
                  f"{ds} {r['instance']}: t=0 objective {r['trajectory'][0][1]} == "
                  f"independent greedy rollout {mk}")
        recs += got
    common_checks(recs)
    return recs


def stage_sized(out_dir, ds, ids, modes, limit):
    ddir = data_dir_for(ds, ids)
    for i in ids:
        print(f"  {ds} instance_{i:03d}.fjs md5 {md5(os.path.join(ddir, f'instance_{i:03d}.fjs'))}"
              f" from {'repository' if ddir.startswith(REPO) else 'fallback'} copy")
    recs = []
    for mode in modes:
        out = os.path.join(out_dir, f"{ds.split('+')[0]}_{mode}.jsonl")
        if os.path.exists(out):
            os.remove(out)
        run_cfg(ds, ddir, mode, limit, 4, ids, out)
        recs += read_jsonl(out)
    common_checks(recs)
    return recs


def main():
    out_dir = _A.out_dir or tempfile.mkdtemp(prefix="cpsat_anytime_test_")
    os.makedirs(out_dir, exist_ok=True)
    print(f"stage={_A.stage} out_dir={out_dir} loadavg={os.getloadavg()}", flush=True)
    recs = []
    if _A.stage == "model":
        stage_model()
    elif _A.stage == "small":
        stage_model()
        recs = stage_small(out_dir)
    elif _A.stage == "medium":
        recs = stage_sized(out_dir, "50x25+ppvc-mixed", [0, 1], ("cold", "warm-spt"), 60)
    elif _A.stage == "large":
        recs = stage_sized(out_dir, "80x25+ppvc-mixed", [0], ("warm-spt",), 60)
    elif _A.stage == "policy":
        recs = stage_policy(out_dir)
    if recs:
        print()
        table(recs)
        extra = {r["dataset"].split("+")[0] + ":" + r["mode"] + ":" + r["instance"]: {
            k: r.get(k) for k in ("spt_makespan", "spt_rollout_s", "policy_makespan",
                                  "policy_rollout_s", "hint_first_solution",
                                  "cpu_share", "n_model_vars",
                                  "n_model_constraints", "rss_after_build_mb",
                                  "instance_wall_s", "solver_wall_s")} for r in recs}
        print(json.dumps(extra, indent=1))
    print()
    if FAILS:
        print(f"{len(FAILS)} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
