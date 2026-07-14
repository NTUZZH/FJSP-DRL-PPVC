"""
eval_ga.py
==========
Genetic-Algorithm (GA) metaheuristic baseline for FJSP with post-operation
time-lags (FJSP-TL), for the PPVC paper. Supplies a search-based peer
baseline alongside the priority-rule and CP-SAT baselines.

WHY THIS DESIGN
  A standard two-vector FJSP GA (Zhang et al.; Kacem et al.) adapted to the
  lag semantics WITHOUT reinventing the lag logic:

    chromosome = (operation-sequence vector, machine-assignment vector)
      * op-sequence (OS): the classic job-based / operation-precedence encoding
        -- a sequence in which job id j appears job_length[j] times; the k-th
        occurrence of j denotes the k-th operation of job j. Any permutation of
        this multiset is precedence-FEASIBLE by construction (a job's ops always
        appear in order), so crossover/mutation never break job precedence.
      * machine-assignment (MA): one compatible machine per GLOBAL op index.

  DECODE -> per-machine op ORDER -> right_shift_repair longest-path decoder.
    Reading the OS left-to-right yields, for each machine, the ORDER its ops are
    dispatched. That fixed per-machine order + the MA is exactly the input the
    repo's lag-aware longest-path decoder consumes. We REUSE that decoder
    (right_shift_repair.py:52-247) verbatim for fitness: it builds the
    precedence DAG with job-arcs weighted ct+lag and machine-arcs weighted ct
    (machine FREE during a lag), then takes the longest path -> lag-feasible
    completion times + makespan. Fitness = makespan (minimise). We do NOT
    re-implement the lag arithmetic anywhere -- the decoder is the single source
    of truth, identical to what A0/reactive use.

  The decoder is `right_shift_repair(job_length, op_pt, time_lag, assigned_mch,
  op_sequence_per_mch=seqs)` -- passing op_sequence_per_mch (NOT op_ct_lagblind)
  so the machine ORDER comes from the GA chromosome, not a lag-blind schedule.
  The DAG is acyclic by construction: within a job, op k precedes op k+1 in the
  OS, so a machine arc can never point backward against a job arc.

OPERATORS (time-budgeted, seeded)
  * tournament selection (size 3)
  * POX (Precedence-preserving Order Crossover) on the OS  +  uniform crossover
    on the MA
  * mutation: swap two positions in the OS  +  reassign a random op to a random
    compatible machine
  * elitism: the best individual always survives unchanged
  Population ~100; each instance runs until a wall-clock budget (--budget_s)
  elapses -- the controlled-compute comparison against CP-SAT / DRL.

VALIDATION
  Every reported schedule is independently re-checked with
  schedule_validator.validate_schedule against the TRUE lags (must be 0
  violations); the validator makespan is asserted equal to the decoder makespan.

CLI
  python eval_ga.py --data_path data/PPVC/10x25+ppvc-mixed \
      [--budget_s 60] [--max_instances N] [--seed 0] [--pop 100]

OUTPUT
  test_results/PPVC/<dataset>/Result_GA{budget}+<dataset>_<dataset>.npy
    shape [N,2] = (makespan, seconds); row order == instance id, identical to
    the other Result_*.npy so analyze_results.py auto-discovers it as method
    "GA<budget>" (e.g. GA60 / GA300) and computes gap + Wilcoxon vs the DRL
    methods.

The GA core (encoding, operators, fitness via the decoder, validation) is pure
numpy. The ONLY place the repo env is touched is initial-population seeding: a
few individuals are built from the real lag-aware PDR rollouts (SPT/MWKR/...)
so the GA starts on par with the strongest priority rule and spends its budget
BEATING it. That seeding imports the env (which imports params at load time), so
-- exactly like eval_ppvc.py -- we parse our own CLI FIRST and scrub argv before
any params-importing module loads. Pass --no_seed_pdr for a fully torch-free run
(random-only init); the GA then needs more budget to reach PDR quality.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np


def parse_cli():
    """Parse our flags FIRST, then scrub argv before any params-importing import.

    params.py calls parser.parse_args() at IMPORT time against the process argv
    (project-wide config singleton). The env module (imported for PDR seeding)
    pulls in params, so we strip argv to just the program name here, exactly as
    eval_ppvc.py / eval_a0_repair.py do.
    """
    ap = argparse.ArgumentParser(
        description='Genetic-Algorithm metaheuristic baseline for FJSP-TL')
    ap.add_argument('--data_path', type=str,
                    default='data/PPVC/10x25+ppvc-mixed',
                    help='directory with instance_*.fjs + .meta.json')
    ap.add_argument('--budget_s', type=float, default=60.0,
                    help='per-instance wall-clock budget in seconds')
    ap.add_argument('--max_instances', type=int, default=None,
                    help='cap the number of instances (verification convenience)')
    ap.add_argument('--seed', type=int, default=0, help='RNG seed (reproducible)')
    ap.add_argument('--pop', type=int, default=100, help='population size')
    ap.add_argument('--no_seed_pdr', action='store_true',
                    help='skip env-based PDR seeding (fully torch-free, '
                         'random-only initial population)')
    args = ap.parse_args()
    sys.argv = [sys.argv[0]]   # clean argv for params.py's import-time parse
    return args


_ARGS = parse_cli()

from right_shift_repair import right_shift_repair
from schedule_validator import validate_schedule


# --------------------------------------------------------------------------- #
# Instance loading
# --------------------------------------------------------------------------- #
def load_instance(stem):
    """Read (job_length, op_pt, meta) for one instance.

    Reimplemented locally (instead of importing ppvc_instance_generator) so the
    GA pulls in ZERO torch/params dependencies: that loader's import chain
    (data_utils -> common_utils) drags in torch. We only need the .fjs matrix +
    the time_lag side-car, both pure-text.
    """
    job_length, op_pt = _read_fjs(stem + '.fjs')
    with open(stem + '.meta.json') as f:
        js = json.load(f)
    meta = {k: (np.array(v) if isinstance(v, list) and k != 'op_name' else v)
            for k, v in js.items()}
    return job_length, op_pt, meta


def _read_fjs(path):
    """Parse a standard .fjs file -> (job_length [J], op_pt [N, M]).

    Format (first line: n_jobs n_machines [flex]; then one line per job:
      n_ops  (n_compat  (mch dur)*)*   with 1-based machine ids).
    Mirrors data_utils.text_to_matrix but without the torch-importing module.
    """
    with open(path) as f:
        lines = [ln for ln in f.readlines() if ln.strip()]
    header = lines[0].split()
    n_jobs = int(header[0])
    n_mch = int(header[1])
    job_rows = lines[1:1 + n_jobs]

    job_length = []
    parsed = []  # list of per-op dict {mch0: dur}
    for row in job_rows:
        tok = row.split()
        p = 0
        n_ops = int(tok[p]); p += 1
        job_length.append(n_ops)
        for _ in range(n_ops):
            n_compat = int(tok[p]); p += 1
            durs = {}
            for _ in range(n_compat):
                m = int(tok[p]) - 1; p += 1   # .fjs machine ids are 1-based
                d = int(tok[p]); p += 1
                durs[m] = d
            parsed.append(durs)
    n_ops_total = len(parsed)
    op_pt = np.zeros((n_ops_total, n_mch), dtype=int)
    for i, durs in enumerate(parsed):
        for m, d in durs.items():
            op_pt[i, m] = d
    return np.array(job_length, dtype=int), op_pt


def list_instances(data_path, cap=None):
    fjs = sorted(glob.glob(os.path.join(data_path, 'instance_*.fjs')))
    if not fjs:
        sys.exit(f'[eval_ga] no instance_*.fjs found under {data_path}')
    stems = [p[:-len('.fjs')] for p in fjs]
    if cap is not None:
        stems = stems[:cap]
    return stems


# --------------------------------------------------------------------------- #
# Decode + fitness
# --------------------------------------------------------------------------- #
def _job_starts(job_length):
    """Global op index of the FIRST op of each job."""
    starts = np.zeros(len(job_length), dtype=int)
    acc = 0
    for j, L in enumerate(job_length):
        starts[j] = acc
        acc += int(L)
    return starts


def decode(os_vec, ma_vec, job_length, job_starts):
    """OS + MA  ->  per-machine op ORDER (list length M of global-op lists).

    Walking the OS left to right: the next time job j is seen, the next op of
    job j (global index job_starts[j] + count_seen[j]) is appended to the
    sequence of its assigned machine ma_vec[op]. Because ops of a job are
    consumed in order, machine sequences never contradict job precedence ->
    the decoder's DAG is acyclic.
    """
    n_mch = _N_MCH_CACHE[0]
    seqs = [[] for _ in range(n_mch)]
    seen = np.zeros(len(job_length), dtype=int)
    for j in os_vec:
        op = job_starts[j] + seen[j]
        seqs[ma_vec[op]].append(op)
        seen[j] += 1
    return seqs


# n_mch is fixed within one instance; cache it so decode() needn't re-derive it
_N_MCH_CACHE = [0]


def fitness(os_vec, ma_vec, job_length, job_starts, op_pt, time_lag):
    """Lag-feasible makespan via the repo's longest-path decoder (REUSED).

    right_shift_repair builds the precedence DAG (job-arcs weight ct+lag,
    machine-arcs weight ct -- machine free during the lag) from the fixed
    per-machine order and returns lag-honouring completion times + makespan.
    """
    seqs = decode(os_vec, ma_vec, job_length, job_starts)
    _, makespan = right_shift_repair(
        job_length, op_pt, time_lag, ma_vec, op_sequence_per_mch=seqs)
    return makespan


# --------------------------------------------------------------------------- #
# GA operators
# --------------------------------------------------------------------------- #
def _compat_machines(op_pt):
    """List, per op, of compatible machine ids (op_pt > 0)."""
    return [np.where(op_pt[i] > 0)[0] for i in range(op_pt.shape[0])]


def random_individual(rng, base_os, compat):
    """A random (OS permutation, compatible MA)."""
    os_vec = base_os.copy()
    rng.shuffle(os_vec)
    ma_vec = np.array([compat[i][rng.integers(len(compat[i]))]
                       for i in range(len(compat))], dtype=int)
    return os_vec, ma_vec


def _seqs_from_starts(assigned_mch, starts, n_mch):
    """Per-machine op ORDER implied by start times (ties broken by op index)."""
    seqs = []
    for m in range(n_mch):
        ops = [i for i in range(len(assigned_mch)) if assigned_mch[i] == m]
        ops.sort(key=lambda i: (starts[i], i))
        seqs.append(ops)
    return seqs


def _os_from_seqs(seqs, op_to_job, starts):
    """Recover an OS multiset consistent with the per-machine order + starts.

    The OS is just the job-ids of all ops sorted by start time (ties by op
    index): decoding it reproduces the same per-machine order, so the seed
    round-trips through decode() to the same schedule.
    """
    n_ops = len(op_to_job)
    order = sorted(range(n_ops), key=lambda i: (starts[i], i))
    return np.array([op_to_job[i] for i in order], dtype=int)


def pdr_seeds_from_env(jl, op_pt, time_lag, rules, seed):
    """Build GA seed individuals from the REAL lag-aware PDR env rollouts.

    For each rule we roll out the env exactly as eval_ppvc.rollout_heuristic
    does (batch of 1, lag-aware), read back assigned_mch + true_op_ct, derive
    start times and per-machine order, and recover an OS that decodes to the
    same schedule. Each seed therefore matches its PDR baseline makespan.

    Imported lazily so a --no_seed_pdr run never touches torch/params/the env.
    Returns a list of (os_vec, ma_vec); empty on any failure (GA still runs).
    """
    try:
        import numpy as _np
        from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
        from common_utils import heuristic_select_action
    except Exception as exc:   # pragma: no cover - keep the GA running torch-free
        print(f'  [seed] env import failed ({exc}); random-only init')
        return []

    M = op_pt.shape[1]
    n_ops = op_pt.shape[0]
    op_to_job = np.empty(n_ops, dtype=int)
    idx = 0
    for j, L in enumerate(jl):
        op_to_job[idx:idx + int(L)] = j
        idx += int(L)

    seeds = []
    for r in rules:
        _np.random.seed(seed)
        env = FJSPEnvForSameOpNums(n_j=jl.shape[0], n_m=M)
        env.set_initial_data([jl], [op_pt], time_lag_list=[time_lag])
        assigned = np.full(n_ops, -1, dtype=int)
        while not env.done().all():
            a = heuristic_select_action(r, env)
            cj, cm = a // M, a % M
            assigned[env.candidate[0, cj]] = cm
            env.step(np.array([a]))
        if (assigned < 0).any():
            continue
        oct_ = env.true_op_ct[0]
        starts = oct_ - np.array([op_pt[i, assigned[i]] for i in range(n_ops)])
        os_vec = _os_from_seqs(_seqs_from_starts(assigned, starts, M),
                               op_to_job, starts)
        seeds.append((os_vec, assigned.astype(int)))
    return seeds


def tournament(rng, pop_fit, k=3):
    """Return the index of the tournament winner (min makespan)."""
    cand = rng.integers(0, len(pop_fit), size=k)
    return cand[np.argmin(pop_fit[cand])]


def pox_crossover(rng, p1_os, p2_os, n_jobs):
    """Precedence-preserving Order Crossover on the OS (job-based encoding).

    Partition jobs into set1 / set2. Child copies p1's positions for set1 jobs;
    the remaining positions are filled, in order, by p2's ops belonging to
    set2 jobs. The result is still a valid OS multiset (each job appears the
    right number of times) and stays precedence-feasible.
    """
    perm = rng.permutation(n_jobs)
    cut = rng.integers(1, n_jobs) if n_jobs > 1 else 1
    set1 = set(perm[:cut].tolist())

    child = np.full_like(p1_os, -1)
    mask1 = np.array([j in set1 for j in p1_os])
    child[mask1] = p1_os[mask1]
    fill = [j for j in p2_os if j not in set1]
    child[~mask1] = fill
    return child


def uniform_crossover_ma(rng, p1_ma, p2_ma):
    """Uniform crossover on the machine-assignment vector.

    Both parents' MA are individually compatible, so any gene-wise mix stays
    compatible.
    """
    take_p1 = rng.random(len(p1_ma)) < 0.5
    child = np.where(take_p1, p1_ma, p2_ma)
    return child


def mutate(rng, os_vec, ma_vec, compat, p_os=0.1, p_ma=0.1):
    """Mutation: a small perturbation on top of crossover.

    Crossover (POX + uniform) is the workhorse; mutation only injects diversity,
    so it stays low-rate (one OS swap, one machine reassignment) to avoid
    destroying good schedules.
    """
    if rng.random() < p_os and len(os_vec) > 1:
        a, b = rng.integers(0, len(os_vec), size=2)
        os_vec[a], os_vec[b] = os_vec[b], os_vec[a]
    if rng.random() < p_ma:
        i = rng.integers(0, len(ma_vec))
        choices = compat[i]
        ma_vec[i] = choices[rng.integers(len(choices))]
    return os_vec, ma_vec


# --------------------------------------------------------------------------- #
# GA driver (one instance, time-budgeted)
# --------------------------------------------------------------------------- #
def run_ga(job_length, op_pt, time_lag, rng, budget_s, pop_size=100,
           elite=2, tour_k=3, pdr_seeds=None):
    """Time-budgeted GA. Returns (best_makespan, best_assigned_mch, best_op_ct,
    n_generations).

    pdr_seeds : optional list of (os_vec, ma_vec) individuals built from the
                real PDR env rollouts (pdr_seeds_from_env). Each matches its PDR
                baseline makespan, giving the GA a strong starting front.

    best_op_ct is recomputed once at the end from the winning chromosome so the
    schedule can be independently validated against the TRUE lags.
    """
    n_jobs = len(job_length)
    job_starts = _job_starts(job_length)
    _N_MCH_CACHE[0] = op_pt.shape[1]
    compat = _compat_machines(op_pt)

    # base OS multiset: job j repeated job_length[j] times
    base_os = np.concatenate(
        [np.full(int(L), j, dtype=int) for j, L in enumerate(job_length)])

    # --- initial population: PDR seeds + random fill ---
    # The PDR seeds (one per lag-aware priority rule, each reproducing its
    # baseline makespan through the decoder) put the GA's starting front on par
    # with the strongest PDR, so the budget is spent BEATING the PDRs; the
    # random bulk supplies the recombination diversity needed to push below them.
    seeds = []
    if pdr_seeds:
        seeds = [(o.copy(), m.copy()) for (o, m) in pdr_seeds][:pop_size]
    pop = list(seeds)
    pop += [random_individual(rng, base_os, compat)
            for _ in range(pop_size - len(pop))]
    pop_fit = np.array(
        [fitness(o, m, job_length, job_starts, op_pt, time_lag)
         for (o, m) in pop])

    best_idx = int(np.argmin(pop_fit))
    best = (pop[best_idx][0].copy(), pop[best_idx][1].copy())
    best_fit = float(pop_fit[best_idx])

    def _fit(ind):
        return fitness(ind[0], ind[1], job_length, job_starts, op_pt, time_lag)

    # Stagnation-driven diversity injection. A plain GA on this encoding
    # converges fast then plateaus; when the incumbent has not improved for
    # `stagnate_limit` generations we replace the worst `immigrant_frac` of the
    # population with FRESH random individuals (random-immigrants scheme),
    # re-opening the search so the rest of the budget keeps finding better
    # schedules instead of churning a converged pool.
    stagnate_limit = 25
    immigrant_frac = 0.4
    stagnation = 0

    t0 = time.time()
    gen = 0
    while time.time() - t0 < budget_s:
        gen += 1
        # elitism: carry the best `elite` individuals forward unchanged
        order = np.argsort(pop_fit)
        new_pop = [(pop[i][0].copy(), pop[i][1].copy())
                   for i in order[:elite]]

        while len(new_pop) < pop_size:
            i1 = tournament(rng, pop_fit, tour_k)
            i2 = tournament(rng, pop_fit, tour_k)
            (o1, m1), (o2, m2) = pop[i1], pop[i2]
            c_os = pox_crossover(rng, o1, o2, n_jobs)
            c_ma = uniform_crossover_ma(rng, m1, m2)
            c_os, c_ma = mutate(rng, c_os, c_ma, compat)
            new_pop.append((c_os, c_ma))

        pop = new_pop
        pop_fit = np.array([_fit(ind) for ind in pop])

        cur_idx = int(np.argmin(pop_fit))
        if pop_fit[cur_idx] < best_fit - 1e-9:
            best_fit = float(pop_fit[cur_idx])
            best = (pop[cur_idx][0].copy(), pop[cur_idx][1].copy())
            stagnation = 0
        else:
            stagnation += 1

        # local-search polish of the incumbent (memetic step): try to move each
        # op of the critical (makespan-determining) machine to a faster
        # compatible machine; cheap and only run on stagnation to save budget.
        if stagnation == stagnate_limit:
            improved = _ma_local_search(best, job_length, job_starts, op_pt,
                                        time_lag, best_fit)
            if improved is not None:
                best, best_fit = improved
                # the polished incumbent re-enters the pool as the new elite
                pop[int(np.argmax(pop_fit))] = (best[0].copy(), best[1].copy())
                pop_fit[int(np.argmax(pop_fit))] = best_fit

            # random-immigrants: refresh the worst fraction to escape the plateau
            n_imm = int(pop_size * immigrant_frac)
            worst = np.argsort(pop_fit)[-n_imm:]
            for w in worst:
                pop[w] = random_individual(rng, base_os, compat)
                pop_fit[w] = _fit(pop[w])
            stagnation = 0

    # recompute completion times for the winner (for validation)
    best_os, best_ma = best
    seqs = decode(best_os, best_ma, job_length, job_starts)
    best_ct, best_ms = right_shift_repair(
        job_length, op_pt, time_lag, best_ma, op_sequence_per_mch=seqs)
    return float(best_ms), best_ma, best_ct, gen


def _ma_local_search(ind, job_length, job_starts, op_pt, time_lag, cur_fit):
    """First-improvement machine-reassignment local search on one individual.

    For each op (random order), try every other compatible machine; keep the
    first reassignment that lowers the decoded makespan. Returns (new_ind,
    new_fit) if any improvement was found, else None. Pure exploitation of the
    repo decoder -- no lag logic here.
    """
    os_vec, ma_vec = ind[0].copy(), ind[1].copy()
    n_ops = op_pt.shape[0]
    improved = False
    f = cur_fit
    for i in range(n_ops):
        choices = np.where(op_pt[i] > 0)[0]
        if len(choices) < 2:
            continue
        orig = ma_vec[i]
        for m in choices:
            if m == orig:
                continue
            ma_vec[i] = m
            seqs = decode(os_vec, ma_vec, job_length, job_starts)
            _, ms = right_shift_repair(job_length, op_pt, time_lag, ma_vec,
                                       op_sequence_per_mch=seqs)
            if ms < f - 1e-9:
                f = ms
                orig = m
                improved = True
            else:
                ma_vec[i] = orig
    if improved:
        return (os_vec, ma_vec), float(f)
    return None


# --------------------------------------------------------------------------- #
# CP-SAT reference (optional; mirrors eval_ppvc.load_or_reference)
# --------------------------------------------------------------------------- #
def load_or_reference(data_path):
    dataset = os.path.basename(os.path.normpath(data_path))
    path = f'./or_solution/PPVC/{dataset}.jsonl'
    ref = {}
    if not os.path.exists(path):
        return ref, path
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get('instance') or rec.get('stem') or rec.get('name')
            ms = rec.get('makespan_lag',
                         rec.get('makespan',
                                 rec.get('objective', rec.get('obj'))))
            if key is None or ms is None:
                continue
            ref[os.path.basename(str(key)).replace('.fjs', '')] = float(ms)
    return ref, path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
# PDR rules used to seed the initial population (each rolled out on the real
# lag-aware env so its seed matches the corresponding PDR baseline makespan).
SEED_RULES = ('SPT', 'MWKR', 'MOR', 'FIFO')


def main():
    args = _ARGS

    stems = list_instances(args.data_path, cap=args.max_instances)
    n = len(stems)
    dataset = os.path.basename(os.path.normpath(args.data_path))
    budget_tag = f'GA{int(round(args.budget_s))}'

    results = np.zeros((n, 2))   # [makespan, seconds]
    feas_count = 0

    ref, ref_path = load_or_reference(args.data_path)
    bases = [os.path.basename(s) for s in stems]

    print('=' * 70)
    print(f'eval_ga  dataset={dataset}  instances={n}  '
          f'budget={args.budget_s:.0f}s  pop={args.pop}  seed={args.seed}')
    print(f'  fitness decoder: right_shift_repair longest-path '
          f'(job-arc=ct+lag, mch-arc=ct)')
    print(f'  init: {"random-only (--no_seed_pdr)" if args.no_seed_pdr else "PDR-seeded (" + ",".join(SEED_RULES) + ") + random"}')
    print('=' * 70)

    for i, stem in enumerate(stems):
        jl, pt, meta = load_instance(stem)
        time_lag = np.asarray(meta['time_lag'])
        base = os.path.basename(stem)

        # per-instance derived seed so each instance is independently
        # reproducible yet the whole run is determined by --seed
        rng = np.random.default_rng(args.seed * 100003 + i)

        # PDR seeds from the real lag-aware env (skipped with --no_seed_pdr)
        pdr_seeds = None
        if not args.no_seed_pdr:
            pdr_seeds = pdr_seeds_from_env(
                jl, pt, time_lag, SEED_RULES, seed=args.seed * 100003 + i)

        t1 = time.time()
        ms, amch, oct_, gen = run_ga(
            jl, pt, time_lag, rng, args.budget_s, pop_size=args.pop,
            pdr_seeds=pdr_seeds)
        sec = time.time() - t1

        # independent feasibility check against the TRUE lags
        res = validate_schedule(jl, pt, time_lag, amch, oct_)
        assert res['feasible'], (
            f'GA produced an INFEASIBLE schedule on {base}:\n  '
            + '\n  '.join(res['violations'][:10]))
        # cross-check: validator makespan must equal the decoder makespan
        assert abs(res['makespan'] - ms) < 1e-6, (
            f'{base}: decoder makespan {ms} != validator {res["makespan"]}')
        feas_count += 1

        results[i] = [ms, sec]

        ref_str = ''
        if base in ref and ref[base] > 0:
            gap = 100.0 * (ms - ref[base]) / ref[base]
            ref_str = f'  cp-sat={ref[base]:.0f}h  gap={gap:+.1f}%'
        print(f'  [{i + 1:3d}/{n}] {base}  makespan={ms:7.1f}h  '
              f'gens={gen:5d}  {sec:5.1f}s{ref_str}')

    # save npy (analyze_results discovers method = budget_tag, e.g. GA60)
    save_dir = f'./test_results/PPVC/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir,
                       f'Result_{budget_tag}+{dataset}_{dataset}.npy')
    np.save(out, results)

    # summary
    mean_ms = results[:, 0].mean()
    std_ms = results[:, 0].std()
    mean_t = results[:, 1].mean()
    gap_str = 'n/a'
    if ref:
        gaps = [100.0 * (results[k, 0] - ref[b]) / ref[b]
                for k, b in enumerate(bases) if b in ref and ref[b] > 0]
        if gaps:
            gap_str = f'{np.mean(gaps):.2f}% (cov {len(gaps)}/{n})'

    print('\n' + '=' * 70)
    print(f'GA SUMMARY  ({dataset}, {n} instances, budget {args.budget_s:.0f}s)')
    print('=' * 70)
    print(f'  mean makespan      : {mean_ms:.1f}  (std {std_ms:.1f})')
    print(f'  mean time (s)      : {mean_t:.2f}')
    print(f'  feasible           : {feas_count}/{n}')
    print(f'  mean gap% vs CP-SAT: {gap_str}')
    print('=' * 70)
    print(f'  saved {out}')


if __name__ == '__main__':
    main()
