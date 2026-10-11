#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_cpsat_anytime.py
---------------------
Anytime CP-SAT evaluation on a PPVC instance set with post-operation time-lags.

For each instance the script
  1. builds the lag-aware CP-SAT model once (the same formulation as
     ortools_solver.fjsp_solver: the successor starts at or after the
     predecessor's completion plus its lag, and the machine is free during the
     lag);
  2. in mode 'warm-spt', runs the lag-aware SPT dispatching rollout
     (eval_ppvc.rollout_heuristic), validates it, hints the complete SPT
     schedule (start times, machine assignment, derived variables) and records
     the SPT makespan as the trajectory point at t = 0; mode 'warm-policy'
     does the same with the greedy schedule of a trained policy
     (eval_ppvc.load_train_config + eval_ppvc.rollout_greedy on the CPU with
     one torch thread);
  3. solves to the time limit with a solution callback that records
     (solver wall time, objective, best bound) for every improving solution,
     and a bound callback that records every improvement of the best bound;
  4. validates the final schedule with schedule_validator.validate_schedule;
  5. appends one JSON line per instance to the output file (resume-safe: ids
     already in the file are skipped, and a file written under a different
     dataset, mode, limit, worker count, CP-SAT seed, policy model or SPT
     seed rule is refused).

Every record carries the machine's load average at the start of the instance
and the solver's own wall time, so that runs on a contended machine can be
identified afterwards.

Usage
  python eval_cpsat_anytime.py --dataset 50x25+ppvc-mixed \
      --mode {cold,warm-spt,warm-policy} --time_limit 1800 --num_workers 4 \
      --start 0 --end 50 [--model_name 10x25+ppvc-mixed+full] \
      [--out or_solution/PPVC/anytime/<file>.jsonl] [--data_dir <dir>]

Run it pinned, with one thread per numerical runtime, for example
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 20-23 python ...
"""
import argparse
import collections
import gc
import json
import os
import resource
import sys
import threading
import time

# params.py parses sys.argv at import time (through ortools_solver and
# eval_ppvc), so our own flags are parsed first and argv is scrubbed before
# those imports.
_p = argparse.ArgumentParser(description="Anytime lag-aware CP-SAT evaluation")
_p.add_argument("--dataset", required=True,
                help="instance set name under data/PPVC/, e.g. 50x25+ppvc-mixed")
_p.add_argument("--mode", required=True, choices=("cold", "warm-spt", "warm-policy"))
_p.add_argument("--time_limit", type=float, required=True,
                help="CP-SAT max_time_in_seconds per instance")
_p.add_argument("--num_workers", type=int, default=4,
                help="CP-SAT num_search_workers (must be >= 1)")
_p.add_argument("--start", type=int, default=0, help="first instance id")
_p.add_argument("--end", type=int, default=None,
                help="one past the last instance id (default: all instances)")
_p.add_argument("--out", default=None,
                help="output .jsonl (default: or_solution/PPVC/anytime/"
                     "<dataset>_<mode>_<limit>s_w<workers>.jsonl)")
_p.add_argument("--data_dir", default=None,
                help="instance directory (default: data/PPVC/<dataset>)")
_p.add_argument("--model_name", default="10x25+ppvc-mixed+full",
                help="warm-policy: checkpoint stem under trained_network/PPVC/ "
                     "with its snapshot train_log/PPVC/config_<name>.json")
_p.add_argument("--seed_test", type=int, default=50,
                help="numpy seed for SPT tie-breaking")
_p.add_argument("--spt_seed_rule", default="fixed", choices=("fixed", "per-instance"),
                help="'fixed': every instance uses seed_test (as eval_cpsat_warmstart); "
                     "'per-instance': seed_test + instance id (as eval_ppvc's PDR loop)")
_p.add_argument("--random_seed", type=int, default=0,
                help="CP-SAT random_seed parameter")
_p.add_argument("--rss_sample_s", type=float, default=0.2,
                help="RSS sampling period during the solve, seconds")
_p.add_argument("--quiet", action="store_true")


def parse_args(argv=None):
    args = _p.parse_args(argv)
    if args.num_workers < 1:
        _p.error("--num_workers must be >= 1 (0 would use every core)")
    if args.time_limit <= 0:
        _p.error("--time_limit must be positive")
    return args


if __name__ == "__main__":
    _ARGS = parse_args()
    sys.argv = [sys.argv[0]]
else:
    _ARGS = None
    _saved_argv = sys.argv
    sys.argv = [sys.argv[0]]

import numpy as np
from ortools.sat.python import cp_model

from ortools_solver import matrix_to_the_format_for_solving
from ppvc_instance_generator import load_instance
from schedule_validator import validate_schedule

if __name__ != "__main__":
    sys.argv = _saved_argv

REPO = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Memory helpers
# ---------------------------------------------------------------------------
def current_rss_mb():
    """Resident set size of this process in MB (Linux /proc, else ru_maxrss)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class RssSampler:
    """Background thread that tracks the peak RSS over a code region."""

    def __init__(self, period_s=0.2):
        self.period_s = period_s
        self.peak_mb = current_rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.period_s):
            self.peak_mb = max(self.peak_mb, current_rss_mb())

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, current_rss_mb())
        return False


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
LagModel = collections.namedtuple(
    "LagModel", "model starts durations ends presences local makespan horizon")


def build_lag_model(jobs, num_machines, time_lag):
    """Build the lag-aware FJSP CP-SAT model.

    The variables, constraints and objective are created in the same order and
    with the same names as ortools_solver.fjsp_solver, so the two protos are
    identical apart from solution hints (test_cpsat_anytime checks this).
    Returns handles to every variable so a complete hint can be attached.
    """
    num_ops = sum(len(job) for job in jobs)
    lag = [0] * num_ops if time_lag is None else [int(x) for x in time_lag]
    assert len(lag) == num_ops, f"time_lag length {len(lag)} != {num_ops} ops"

    model = cp_model.CpModel()
    horizon = 0
    for job in jobs:
        for task in job:
            horizon += max(alt[0] for alt in task)
    horizon += sum(lag)

    intervals_per_resources = collections.defaultdict(list)
    starts, durations, ends, presences, local = {}, {}, {}, {}, {}
    job_ends = []

    op_idx = 0
    for job_id, job in enumerate(jobs):
        previous_end = None
        previous_lag = 0
        for task_id, task in enumerate(job):
            min_duration = min(alt[0] for alt in task)
            max_duration = max(alt[0] for alt in task)
            suffix = "_j%i_t%i" % (job_id, task_id)
            start = model.NewIntVar(0, horizon, "start" + suffix)
            duration = model.NewIntVar(min_duration, max_duration, "duration" + suffix)
            end = model.NewIntVar(0, horizon, "end" + suffix)
            interval = model.NewIntervalVar(start, duration, end, "interval" + suffix)
            starts[(job_id, task_id)] = start
            durations[(job_id, task_id)] = duration
            ends[(job_id, task_id)] = end

            if previous_end is not None:
                model.Add(start >= previous_end + previous_lag)
            previous_end = end
            previous_lag = lag[op_idx]
            op_idx += 1

            if len(task) > 1:
                l_presences = []
                for alt_id, (l_duration, mch) in enumerate(task):
                    alt_suffix = "_j%i_t%i_a%i" % (job_id, task_id, alt_id)
                    l_presence = model.NewBoolVar("presence" + alt_suffix)
                    l_start = model.NewIntVar(0, horizon, "start" + alt_suffix)
                    l_end = model.NewIntVar(0, horizon, "end" + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_start, l_duration, l_end, l_presence, "interval" + alt_suffix)
                    l_presences.append(l_presence)
                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)
                    intervals_per_resources[mch].append(l_interval)
                    presences[(job_id, task_id, alt_id)] = l_presence
                    local[(job_id, task_id, alt_id)] = (l_start, l_end)
                model.AddExactlyOne(l_presences)
            else:
                intervals_per_resources[task[0][1]].append(interval)
                presences[(job_id, task_id, 0)] = model.NewConstant(1)
        job_ends.append(previous_end)

    for machine_id in range(num_machines):
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)
    return LagModel(model, starts, durations, ends, presences, local, makespan, horizon)


def add_full_hint(lm, jobs, assigned_mch, op_ct):
    """Hint a complete feasible schedule (every decision and derived variable).

    Returns the hinted makespan.
    """
    op_idx = 0
    job_end_values = []
    for job_id, job in enumerate(jobs):
        end_value = 0
        for task_id, task in enumerate(job):
            mch = int(assigned_mch[op_idx])
            chosen = [a for a, (_, m) in enumerate(task) if m == mch]
            assert len(chosen) == 1, f"op {op_idx}: machine {mch} not in its alternatives"
            chosen = chosen[0]
            dur = int(task[chosen][0])
            end_value = int(round(float(op_ct[op_idx])))
            st = end_value - dur
            assert st >= 0, f"op {op_idx}: negative hinted start {st}"
            lm.model.AddHint(lm.starts[(job_id, task_id)], st)
            lm.model.AddHint(lm.durations[(job_id, task_id)], dur)
            lm.model.AddHint(lm.ends[(job_id, task_id)], end_value)
            if len(task) > 1:
                for alt_id, (l_dur, _) in enumerate(task):
                    present = alt_id == chosen
                    lm.model.AddHint(lm.presences[(job_id, task_id, alt_id)], int(present))
                    l_start, l_end = lm.local[(job_id, task_id, alt_id)]
                    ls = st if present else 0
                    lm.model.AddHint(l_start, ls)
                    lm.model.AddHint(l_end, ls + int(l_dur))
            op_idx += 1
        job_end_values.append(end_value)
    hinted = max(job_end_values)
    assert hinted <= lm.horizon, "hinted makespan exceeds the model horizon"
    lm.model.AddHint(lm.makespan, hinted)
    return hinted


def extract_schedule(solver, lm, jobs):
    """Per-op (assigned machine, completion time), flat in job-by-job order."""
    asg, ct = [], []
    for job_id, job in enumerate(jobs):
        for task_id, task in enumerate(job):
            st = solver.Value(lm.starts[(job_id, task_id)])
            machine, dur = -1, 0
            for alt_id, (d, m) in enumerate(task):
                if solver.Value(lm.presences[(job_id, task_id, alt_id)]):
                    machine, dur = m, d
            asg.append(machine)
            ct.append(st + dur)
    return np.array(asg, dtype=int), np.array(ct, dtype=int)


class TrajectoryRecorder(cp_model.CpSolverSolutionCallback):
    """Records (solver wall time, objective, best bound) per improving solution."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.points = []
        self._best = float("inf")

    def on_solution_callback(self):
        obj = self.ObjectiveValue()
        if obj < self._best:
            self._best = obj
            self.points.append([round(self.WallTime(), 4), float(obj),
                                float(self.BestObjectiveBound())])


# ---------------------------------------------------------------------------
# One instance
# ---------------------------------------------------------------------------
def _eval_ppvc(data_dir=None):
    """Import eval_ppvc with a clean argv and set its route variant."""
    saved = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already set once in this process
        import eval_ppvc
    finally:
        sys.argv = saved
    if data_dir is not None and hasattr(eval_ppvc, "detect_route_variant"):
        eval_ppvc.ROUTE_VARIANT = eval_ppvc.detect_route_variant(data_dir, "auto")
    return eval_ppvc


def spt_schedule(jl, pt, meta, seed, data_dir=None):
    """Lag-aware SPT rollout through eval_ppvc.rollout_heuristic."""
    ev = _eval_ppvc(data_dir)
    mk, secs, asg, op_ct = ev.rollout_heuristic("SPT", jl, pt, meta, seed)
    return float(mk), float(secs), asg, op_ct


_POLICY_CACHE = {}


def load_policy(model_name):
    """Build the policy on the CPU exactly as eval_ppvc.main does; cached.

    Returns (ppo, use_lag_features, use_type_embedding, framework).
    """
    if model_name in _POLICY_CACHE:
        return _POLICY_CACHE[model_name]
    ev = _eval_ppvc()
    import torch
    from params import configs
    cwd = os.getcwd()
    os.chdir(REPO)  # eval_ppvc reads its snapshot by a repository-relative path
    try:
        ev.load_train_config(model_name)
    finally:
        os.chdir(cwd)
    configs.device = "cpu"
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    ckpt = os.path.join(REPO, "trained_network", "PPVC", f"{model_name}.pth")
    if not os.path.exists(ckpt):
        raise SystemExit(f"[anytime] checkpoint not found: {ckpt}")
    from model.PPO import PPO_initialize
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")))
    ppo.policy.eval()
    out = (ppo, bool(getattr(configs, "use_lag_features", False)),
           bool(getattr(configs, "use_type_embedding", False)),
           getattr(configs, "framework", "DANIEL"))
    _POLICY_CACHE[model_name] = out
    return out


def policy_schedule(jl, pt, meta, model_name, data_dir=None):
    """Greedy rollout of the trained policy through eval_ppvc.rollout_greedy."""
    ev = _eval_ppvc(data_dir)
    ppo, lagf, tyemb, _ = load_policy(model_name)
    mk, secs, asg, op_ct = ev.rollout_greedy(ppo, jl, pt, meta, lagf, tyemb)
    return float(mk), float(secs), asg, op_ct


def solve_instance(inst_id, data_dir, mode, time_limit, num_workers,
                   seed_test=50, random_seed=0, rss_sample_s=0.2,
                   model_name=None, spt_seed_rule="fixed"):
    stem = os.path.join(data_dir, f"instance_{inst_id:03d}")
    loadavg_start = [round(x, 2) for x in os.getloadavg()]
    rss_baseline = current_rss_mb()
    wall0 = time.perf_counter()

    jl, pt, meta = load_instance(stem)
    time_lag = meta["time_lag"]
    jobs, num_machines = matrix_to_the_format_for_solving(jl, pt)

    rec = {
        "instance_id": int(inst_id),
        "instance": f"instance_{inst_id:03d}",
        "dataset": os.path.basename(os.path.normpath(data_dir)),
        "mode": mode,
        "time_limit_s": float(time_limit),
        "num_workers": int(num_workers),
        "random_seed": int(random_seed),
        "model_name": model_name if mode == "warm-policy" else None,
        "spt_seed_rule": spt_seed_rule if mode == "warm-spt" else None,
        "n_jobs": int(len(jl)),
        "n_ops": int(pt.shape[0]),
        "n_machines": int(num_machines),
        "loadavg_start": loadavg_start,
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    trajectory = []
    warm = None
    if mode == "warm-spt":
        seed = seed_test + (inst_id if spt_seed_rule == "per-instance" else 0)
        mk, secs, asg, ct = spt_schedule(jl, pt, meta, seed, data_dir)
        prefix, warm = "spt", (mk, asg, ct)
        rec["spt_seed"] = int(seed)
    elif mode == "warm-policy":
        mk, secs, asg, ct = policy_schedule(jl, pt, meta, model_name, data_dir)
        prefix, warm = "policy", (mk, asg, ct)
        rec["policy_framework"] = load_policy(model_name)[3]
    if warm is not None:
        v = validate_schedule(jl, pt, time_lag, asg, ct)
        assert v["feasible"], f"{prefix} schedule infeasible on {stem}: {v['violations'][:3]}"
        assert abs(v["makespan"] - mk) < 1e-6, f"{prefix} makespan != validator makespan"
        rec[f"{prefix}_makespan"] = mk
        rec[f"{prefix}_rollout_s"] = round(secs, 4)
        rec[f"{prefix}_validator_violations"] = len(v["violations"])
        trajectory.append([0.0, mk, None])

    t_build = time.perf_counter()
    lm = build_lag_model(jobs, num_machines, time_lag)
    if warm is not None:
        hinted = add_full_hint(lm, jobs, warm[1], warm[2])
        assert abs(hinted - warm[0]) < 1e-6, "hinted makespan != warm-start makespan"
        rec["hint_makespan"] = float(hinted)
    build_s = time.perf_counter() - t_build
    rss_after_build = current_rss_mb()
    rec["horizon"] = int(lm.horizon)
    rec["n_model_vars"] = len(lm.model.Proto().variables)
    rec["n_model_constraints"] = len(lm.model.Proto().constraints)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    solver.parameters.num_search_workers = int(num_workers)
    solver.parameters.random_seed = int(random_seed)
    recorder = TrajectoryRecorder()
    bound_traj = []
    t_solve = [None]

    def on_bound(bound):
        bound_traj.append([round(time.perf_counter() - t_solve[0], 4), float(bound)])

    solver.best_bound_callback = on_bound

    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    with RssSampler(rss_sample_s) as sampler:
        t_solve[0] = time.perf_counter()
        status = solver.Solve(lm.model, recorder)
        solve_elapsed = time.perf_counter() - t_solve[0]
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    solve_cpu_s = (ru1.ru_utime - ru0.ru_utime) + (ru1.ru_stime - ru0.ru_stime)

    status_name = solver.StatusName(status)
    has_sol = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    trajectory.extend(recorder.points)

    rec.update(
        status=status_name,
        objective=float(solver.ObjectiveValue()) if has_sol else None,
        bound=float(solver.BestObjectiveBound()),
        solver_wall_s=round(float(solver.WallTime()), 4),
        solve_elapsed_s=round(solve_elapsed, 4),
        solve_cpu_s=round(solve_cpu_s, 2),
        # CPU seconds per worker-second of the solve; well below 1 on a
        # contended machine
        cpu_share=round(solve_cpu_s / max(solve_elapsed * num_workers, 1e-9), 3),
        loadavg_end=[round(x, 2) for x in os.getloadavg()],
        model_build_s=round(build_s, 4),
        n_improving_solutions=len(recorder.points),
        rss_baseline_mb=round(rss_baseline, 1),
        rss_after_build_mb=round(rss_after_build, 1),
        peak_rss_mb=round(sampler.peak_mb, 1),
        process_maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
        trajectory=trajectory,
        bound_trajectory=bound_traj,
    )
    if warm is not None:
        first = recorder.points[0] if recorder.points else None
        rec["hint_first_solution"] = first
        rec["hint_taken"] = bool(first is not None and first[1] <= rec["hint_makespan"])

    if has_sol:
        asg, ct = extract_schedule(solver, lm, jobs)
        v = validate_schedule(jl, pt, time_lag, asg, ct)
        rec.update(validator_feasible=bool(v["feasible"]),
                   validator_n_violations=len(v["violations"]),
                   validator_makespan=float(v["makespan"]),
                   validator_lower_bound=float(v["lower_bound"]),
                   validator_violations=v["violations"][:20])
        if not v["feasible"]:
            print(f"!!! VALIDATOR VIOLATIONS on {rec['instance']}: {len(v['violations'])}",
                  flush=True)
    else:
        rec.update(validator_feasible=None, validator_n_violations=None)
    rec["instance_wall_s"] = round(time.perf_counter() - wall0, 4)
    return rec


# ---------------------------------------------------------------------------
# Resume-safe output
# ---------------------------------------------------------------------------
CONFIG_KEYS = ("dataset", "mode", "time_limit_s", "num_workers", "random_seed",
               "model_name", "spt_seed_rule")


def load_done(path, config):
    """Ids already in the file. Refuses a file written under another config."""
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line of an interrupted run
            for k in CONFIG_KEYS:
                if r.get(k) != config[k]:
                    raise SystemExit(f"[anytime] {path} holds a record with {k}="
                                     f"{r.get(k)!r}, this run has {config[k]!r}; "
                                     f"use another --out")
            done.add(int(r["instance_id"]))
    return done


def release_memory():
    """Return freed heap pages to the OS between instances (glibc only)."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def append_record(path, rec):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run(args):
    data_dir = args.data_dir or os.path.join(REPO, "data", "PPVC", args.dataset)
    if not os.path.isdir(data_dir):
        raise SystemExit(f"[anytime] instance directory not found: {data_dir}")
    n_avail = len([f for f in os.listdir(data_dir)
                   if f.startswith("instance_") and f.endswith(".fjs")])
    end = n_avail if args.end is None else min(args.end, n_avail)
    out = args.out or os.path.join(
        REPO, "or_solution", "PPVC", "anytime",
        f"{args.dataset}_{args.mode}_{args.time_limit:g}s_w{args.num_workers}.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    config = {"dataset": os.path.basename(os.path.normpath(data_dir)), "mode": args.mode,
              "time_limit_s": float(args.time_limit), "num_workers": int(args.num_workers),
              "random_seed": int(args.random_seed),
              "model_name": args.model_name if args.mode == "warm-policy" else None,
              "spt_seed_rule": args.spt_seed_rule if args.mode == "warm-spt" else None}
    if config["dataset"] != args.dataset:
        raise SystemExit(f"[anytime] --data_dir basename {config['dataset']!r} "
                         f"!= --dataset {args.dataset!r}")
    done = load_done(out, config)
    todo = [i for i in range(args.start, end) if i not in done]
    print(f"[anytime] {args.dataset} mode={args.mode} limit={args.time_limit:g}s "
          f"workers={args.num_workers} "
          + (f"model={args.model_name} " if args.mode == "warm-policy" else "")
          + f"ids [{args.start},{end}) "
          f"done={len(done & set(range(args.start, end)))} todo={len(todo)} -> {out}",
          flush=True)
    for i in todo:
        rec = solve_instance(i, data_dir, args.mode, args.time_limit, args.num_workers,
                             seed_test=args.seed_test, random_seed=args.random_seed,
                             rss_sample_s=args.rss_sample_s, model_name=args.model_name,
                             spt_seed_rule=args.spt_seed_rule)
        append_record(out, rec)
        release_memory()
        if not args.quiet:
            print(f"  {rec['instance']}: {rec['status']} obj={rec['objective']} "
                  f"bound={rec['bound']} sols={rec['n_improving_solutions']} "
                  f"build={rec['model_build_s']:.2f}s solve={rec['solver_wall_s']:.1f}s "
                  f"peakRSS={rec['peak_rss_mb']:.0f}MB load={rec['loadavg_start'][0]} "
                  f"viol={rec['validator_n_violations']}", flush=True)
    return out


if __name__ == "__main__":
    run(_ARGS)
