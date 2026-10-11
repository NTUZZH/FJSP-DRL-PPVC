"""
eval_memetic.py
===============
Memetic-algorithm (MA) baseline for the flexible job shop with minimum
post-operation time-lags (FJSP-TL): an adaptation of

    A. Caumond, P. Lacomme, N. Tchernev, "A memetic algorithm for the
    job-shop with time-lags", Computers & Operations Research 35(7),
    2331-2356, 2008. DOI 10.1016/j.cor.2006.11.007.

to routing flexibility. Problem semantics: after operation i of a job
completes at C_i, the next operation of the job may start at C_i + lag_i; the
machine is free during the lag (paper, Section 1.1: time lags "do not consume
any capacity of the machine"); makespan objective; no maximum lags.

WHAT IS TAKEN FROM THE PAPER (where the paper is specific)
----------------------------------------------------------
* Chromosome: a Bierwirth job-repetition sequence (operation sequence, OS).
  Job j appears job_length[j] times and its k-th occurrence denotes its k-th
  operation, so every OS is precedence-feasible.
* Evaluation (Algorithms 1-2, "Evaluate_with_TL" / "Evaluate_A_Sequence"): a
  longest-path labelling on the disjunctive graph whose conjunctive (job) arcs
  carry processing time + minimum lag and whose disjunctive (machine) arcs are
  oriented by the chromosome. Here that evaluation is the repository's shared
  decoder right_shift_repair(..., op_sequence_per_mch=...), which builds the
  same graph (job arc weight C + lag, machine arc weight C) and returns the
  semi-active completion times. No lag arithmetic is re-implemented. Without
  maximum lags there are no negative arcs and no positive cycles, so the
  paper's cycle detection and its hierarchical fitness omega_1 f_1 + omega_2
  f_2 reduce to f = makespan (f_1, the count of violated maximum lags, is
  always zero).
* Initial population: nc chromosomes, n1 = 14 of them built from priority
  dispatching rules, the rest random permutations. The rule schedules come
  from the lag-aware environment, so they are feasible and the paper's
  reparation step (Algorithm 4) is never needed.
* Crossover: GOX, Bierwirth's generalized order crossover, on the OS.
* Local search (Algorithm 5): walk backward along the critical path from the
  last vertex through the parent pointers; where two consecutive critical
  operations are joined by a machine arc (the end of one job block and the
  start of the next), swap them on that machine and re-evaluate; at most nm
  iterations; a move is kept only if it does not worsen the fitness.
* Population management: an array of nc chromosomes sorted by fitness;
  approximate clone detection by mark(C) = sum_i t_i^2 mod K with K = 49999
  and t_i the operation start times; rank-based parent selection
  P(rank i) = 2i / (nc (nc + 1)); local search applied to a child with
  probability pm; restart after np iterations without improvement of the best
  chromosome, replacing pr percent of the population by random chromosomes
  and never replacing the best.
* Parameters (Section 4.5, job shop with time lags): nc = 50, np = 20,000,
  pm = 5 %, mni = 1,000,000 iterations, nm = 10.

ADDITIONS REQUIRED BY THE FLEXIBLE SHOP (not in the paper)
----------------------------------------------------------
* A machine-assignment vector (MA), one compatible machine per operation,
  decoded together with the OS: reading the OS left to right appends each
  operation to the sequence of its assigned machine.
* MA crossover: uniform crossover (each gene from either parent), which keeps
  every gene compatible. This is the standard two-vector FJSP choice.
* A machine-reassignment move inside the local search: at every critical
  operation visited by the backward walk, each other compatible machine is
  tried in addition to the paper's swap move.
* Rule-seeded individuals take their machine choice from the rule schedule.

DESIGN CHOICES WHERE THE PAPER LEAVES FREEDOM OR IS SILENT
-------------------------------------------------------------------------
1. Steady-state loop. One iteration (reported as one "generation") selects two
   parents, builds one child by GOX + uniform MA crossover, applies the local
   search with probability pm, and inserts the child unless it is a clone.
   The child replaces a chromosome drawn uniformly from the worse half of the
   sorted array, the replacement rule of the Lacomme-Prins memetic algorithms
   this paper belongs to; the best chromosome can therefore never be lost.
2. Rank selection. Rank i = nc is the best chromosome, so the best has the
   largest probability 2 nc / (nc (nc + 1)). The two parents are distinct.
3. GOX details (Bierwirth, 1995): the donor is the first selected parent; the
   donor substring length is drawn uniformly between one third and one half
   of the chromosome length; the substring genes, identified as (job,
   occurrence) pairs, are deleted from the receiver and the substring is
   implanted where its first gene occurred in the receiver.
4. Local search structure (Algorithm 5). One local-search iteration is one
   backward walk over the current critical path, stopped at the first strictly
   improving move, which is kept; the next iteration walks the new critical
   path. The search ends after nm iterations or after a walk with no improving
   move. "Does not worsen" is implemented as strict improvement, so the walk
   cannot cycle on plateau moves.
5. Critical path. It is derived from the shared decoder's schedule: from the
   operation with the largest completion time, the parent of an operation v
   with start S_v > 0 is its machine predecessor q if C_q = S_v, otherwise its
   job predecessor p with C_p + lag_p = S_v (the lag-weighted job arc). When
   both arcs are tight the machine arc is preferred, because only machine arcs
   carry swap moves. In the flexible shop two consecutive operations of one
   job may also be consecutive on one machine; that arc is labelled a job arc
   and never swapped, since the swap would reverse job precedence. Ties for
   the last vertex go to the lowest operation id.
6. Swap move on the OS. Swapping two occurrences in the OS can relabel a
   job's operations when that job reappears between them. The move is
   therefore applied to the machine sequences (u, v -> v, u), and the new OS
   is the topological order of the new graph that stays closest to the old
   OS (Kahn's algorithm keyed by the old position, with v taking u's place).
   It decodes to exactly the swapped sequences. Reversing a critical machine
   arc never creates a cycle because all arc weights are positive.
7. Restart percentage pr is not given in the extraction; pr = 50 % (the worst
   half is replaced by random chromosomes, the best half kept).
8. Stopping rule. The paper stops after mni iterations; here the run stops
   at the first of a per-instance wall-clock budget (--budget_s) and the
   iteration cap (--max_gens, default mni = 1,000,000).
   np = 20,000 is kept at the paper's value. At our instance sizes and
   budgets a restart may never trigger (see the JSON field "restarts").
9. Rule seeds. Eight rules of common_utils.heuristic_select_action (FIFO,
   MOR, SPT, MWKR and the lag-aware SPTL, LLF, MWKRL, LRPTL) are rolled out
   once with the tie-break seed of eval_ppvc.py (seed_test + instance index,
   default 50 + i), so these seeds reproduce the published rule baselines,
   and once more with a tie-break seed derived from --seed. The first 14
   distinct schedules (by clone mark) enter the population. The OS of a seed
   lists the operations by start time, so it decodes to the rule's machine
   sequences.
10. Budget accounting. As in eval_ga.py, --budget_s bounds the search after
   the rule seeding. The seconds column of the .npy file reports the total
   wall-clock of the instance (seeding + search); the JSON file reports both
   parts. --budget_includes_seeding makes the budget cover both.
11. The rule environments run with the CPU device (the scheduler never reads
   a network), which also keeps this baseline off the GPU.

VALIDATION
  Every final schedule is re-checked with schedule_validator.validate_schedule
  against the true lags (0 violations required), and the validator makespan
  must equal the decoder makespan.

CLI
  python eval_memetic.py --data_path data/PPVC/10x25+ppvc-mixed \
      [--budget_s 60] [--max_gens N] [--max_instances N] [--seed 0]

OUTPUT
  <save_dir>/Result_MA<budget>+memetic_<dataset>.npy, shape [N, 2] =
  (makespan, seconds), row order = instance id; and a JSON file of the same
  stem with per-instance best makespan, generations, local-search calls and
  improvements, restarts, time-to-best and the rule-seed makespans. A run
  truncated by --max_instances writes to <default dir>/partial/ unless
  --save_dir is given, so partial files are never swept into the analysis.
  The search runs under a wall-clock budget, so its result depends on the CPU
  it received. Every instance record of the JSON file therefore carries the
  contention evidence of its search (eval_stochastic_lags.solve_timing and
  contention_fields): solve_wall_s, solve_cpu_s, workers (1, the search is
  single-threaded), cpu_share = CPU s / (wall s x workers), status ('budget',
  or 'max_gens' when the iteration cap ended the search), time_limit_s (null
  without a budget), hit_limit, and cpu_share_min, n_timed_solves,
  n_hit_limit, n_low_share, n_no_evidence, affinity, loadavg_start. With
  --budget_includes_seeding the clock covers seeding and search. The JSON
  key "contention" holds the run-level summary (contention_summary); a
  budget-limited search with cpu_share below 0.9 ran on shared cores and is
  not a valid measurement.
"""
import argparse
import glob
import heapq
import json
import os
import sys
import time

import numpy as np

from right_shift_repair import right_shift_repair
from schedule_validator import validate_schedule
from eval_stochastic_lags import (SolveClock, contention_fields,
                                  contention_summary, loadavg_now,
                                  solve_timing, LOW_CPU_SHARE)

# --------------------------------------------------------------------------- #
# Paper parameters (Section 4.5) and fixed choices
# --------------------------------------------------------------------------- #
NC = 50              # population size
N1 = 14              # rule-seeded chromosomes in the initial population
NP_RESTART = 20000   # unproductive iterations before a restart
PM = 0.05            # probability that a child undergoes local search
MNI = 1000000        # maximum number of iterations
NM = 10              # maximum local-search iterations per call
PR = 0.5             # restart fraction (not given in the extraction)
MARK_K = 49999       # clone-detection modulus

SEED_RULES = ('FIFO', 'MOR', 'SPT', 'MWKR', 'SPTL', 'LLF', 'MWKRL', 'LRPTL')
TOL = 1e-6


# --------------------------------------------------------------------------- #
# Instance loading (pure text, no torch)
# --------------------------------------------------------------------------- #
def read_fjs(path):
    """Parse a .fjs file into (job_length [J], op_pt [N, M]); 1-based machines."""
    with open(path) as f:
        lines = [ln for ln in f.readlines() if ln.strip()]
    n_jobs, n_mch = int(lines[0].split()[0]), int(lines[0].split()[1])
    job_length, parsed = [], []
    for row in lines[1:1 + n_jobs]:
        tok = row.split()
        p = 0
        n_ops = int(tok[p]); p += 1
        job_length.append(n_ops)
        for _ in range(n_ops):
            n_compat = int(tok[p]); p += 1
            durs = {}
            for _ in range(n_compat):
                m = int(tok[p]) - 1
                durs[m] = int(tok[p + 1])
                p += 2
            parsed.append(durs)
    op_pt = np.zeros((len(parsed), n_mch), dtype=int)
    for i, durs in enumerate(parsed):
        for m, d in durs.items():
            op_pt[i, m] = d
    return np.array(job_length, dtype=int), op_pt


def load_instance(stem):
    """Return (job_length, op_pt, meta) for one instance stem."""
    job_length, op_pt = read_fjs(stem + '.fjs')
    with open(stem + '.meta.json') as f:
        js = json.load(f)
    meta = {k: (np.array(v) if isinstance(v, list) and k != 'op_name' else v)
            for k, v in js.items()}
    return job_length, op_pt, meta


def list_instances(data_path, cap=None):
    """Sorted instance stems, identical ordering to eval_ppvc.list_instances."""
    fjs = sorted(glob.glob(os.path.join(data_path, 'instance_*.fjs')))
    if not fjs:
        sys.exit(f'[eval_memetic] no instance_*.fjs found under {data_path}')
    stems = [p[:-len('.fjs')] for p in fjs]
    return stems[:cap] if cap is not None else stems


def route_variant(data_path):
    meta_path = os.path.join(data_path, 'dataset_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f).get('route_variant', 'fixed')
    return 'fixed'


# --------------------------------------------------------------------------- #
# Problem wrapper: decoding and evaluation through the shared decoder
# --------------------------------------------------------------------------- #
class Problem:
    """Static data of one instance plus the chromosome decoder."""

    def __init__(self, job_length, op_pt, time_lag):
        self.job_length = np.asarray(job_length, dtype=int)
        self.op_pt = np.asarray(op_pt)
        self.time_lag = np.asarray(time_lag, dtype=float)
        self.n_jobs = len(self.job_length)
        self.n_ops, self.n_mch = self.op_pt.shape
        self.job_start = np.concatenate(
            [[0], np.cumsum(self.job_length)[:-1]]).astype(int)
        self.op_job = np.repeat(np.arange(self.n_jobs), self.job_length)
        self.job_pred = np.arange(self.n_ops) - 1
        self.job_pred[self.job_start] = -1
        self.compat = [np.where(self.op_pt[i] > 0)[0] for i in range(self.n_ops)]
        self.base_os = self.op_job.copy()
        self.n_evals = 0

    # ---- chromosome -> machine sequences -------------------------------- #
    def decode(self, os_vec, ma_vec):
        """OS + MA -> (per-machine sequences, operations in OS order)."""
        seqs = [[] for _ in range(self.n_mch)]
        nxt = self.job_start.tolist()
        ma = ma_vec.tolist()
        order = []
        for j in os_vec.tolist():
            op = nxt[j]
            nxt[j] = op + 1
            order.append(op)
            seqs[ma[op]].append(op)
        return seqs, np.array(order, dtype=int)

    def evaluate_seqs(self, seqs, ma_vec):
        """Shared lag-aware longest-path decoder -> (completion times, makespan)."""
        self.n_evals += 1
        ct, ms = right_shift_repair(self.job_length, self.op_pt, self.time_lag,
                                    ma_vec, op_sequence_per_mch=seqs)
        return ct, float(ms)

    def evaluate(self, os_vec, ma_vec):
        seqs, order = self.decode(os_vec, ma_vec)
        ct, ms = self.evaluate_seqs(seqs, ma_vec)
        return ms, ct, seqs, order

    def starts(self, ct, ma_vec):
        return ct - self.op_pt[np.arange(self.n_ops), ma_vec]

    def mark(self, ct, ma_vec):
        """Clone mark: sum of squared start times modulo K (paper)."""
        s = np.rint(self.starts(ct, ma_vec)).astype(np.int64)
        return int(np.sum(s * s) % MARK_K)

    # ---- critical path from the decoder's schedule --------------------- #
    def critical_path(self, ct, seqs, ma_vec):
        """Backward critical path from the last vertex.

        Returns (path, kinds): path[0] is the operation with the largest
        completion time; kinds[k] is 'mch' or 'job', the type of the arc
        path[k+1] -> path[k].
        """
        mch_prev = np.full(self.n_ops, -1, dtype=int)
        for seq in seqs:
            for a, b in zip(seq[:-1], seq[1:]):
                mch_prev[b] = a
        st = self.starts(ct, ma_vec)
        v = int(np.argmax(ct))
        path, kinds = [v], []
        while st[v] > TOL:
            q = mch_prev[v]
            p = self.job_pred[v]
            if q >= 0 and q != p and abs(ct[q] - st[v]) <= TOL:
                kinds.append('mch')
                v = int(q)
            elif p >= 0 and abs(ct[p] + self.time_lag[p] - st[v]) <= TOL:
                kinds.append('job')
                v = int(p)
            else:
                raise RuntimeError(
                    f'critical path: no tight arc into op {v} (start {st[v]})')
            path.append(v)
        return path, kinds

    # ---- machine sequences -> OS closest to a reference order ---------- #
    def os_from_seqs(self, seqs, key):
        """Topological order of the job + machine arc graph, keyed by `key`.

        key[op] is a sortable priority (the old OS position); the returned OS
        decodes to exactly `seqs`.
        """
        n = self.n_ops
        indeg = np.zeros(n, dtype=int)
        succ = [[] for _ in range(n)]
        for op in range(n):
            if self.job_pred[op] >= 0:
                succ[self.job_pred[op]].append(op)
                indeg[op] += 1
        for seq in seqs:
            for a, b in zip(seq[:-1], seq[1:]):
                succ[a].append(b)
                indeg[b] += 1
        heap = [(key[op], op) for op in range(n) if indeg[op] == 0]
        heapq.heapify(heap)
        out = []
        while heap:
            _, u = heapq.heappop(heap)
            out.append(u)
            for w in succ[u]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    heapq.heappush(heap, (key[w], w))
        if len(out) != n:
            raise RuntimeError('os_from_seqs: cycle in the swapped graph')
        return self.op_job[np.array(out, dtype=int)]


# --------------------------------------------------------------------------- #
# Individuals
# --------------------------------------------------------------------------- #
class Individual:
    __slots__ = ('os', 'ma', 'ms', 'ct', 'seqs', 'order', 'mark')

    def __init__(self, prob, os_vec, ma_vec):
        self.os = np.asarray(os_vec, dtype=int)
        self.ma = np.asarray(ma_vec, dtype=int)
        self.ms, self.ct, self.seqs, self.order = prob.evaluate(self.os, self.ma)
        self.mark = prob.mark(self.ct, self.ma)

    def copy(self):
        new = Individual.__new__(Individual)
        new.os, new.ma = self.os.copy(), self.ma.copy()
        new.ms, new.ct = self.ms, self.ct.copy()
        new.seqs = [list(s) for s in self.seqs]
        new.order = self.order.copy()
        new.mark = self.mark
        return new


def random_individual(prob, rng):
    os_vec = prob.base_os.copy()
    rng.shuffle(os_vec)
    ma_vec = np.array([c[rng.integers(len(c))] for c in prob.compat], dtype=int)
    return Individual(prob, os_vec, ma_vec)


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #
def gox(rng, donor, receiver, n_jobs):
    """Generalized order crossover (Bierwirth 1995) on job-repetition OS."""
    n = len(donor)

    def labels(os_vec):
        cnt = np.zeros(n_jobs, dtype=int)
        lab = np.empty(n, dtype=np.int64)
        for p, j in enumerate(os_vec):
            lab[p] = j * n + cnt[j]
            cnt[j] += 1
        return lab

    ld, lr = labels(donor), labels(receiver)
    lo, hi = max(1, n // 3), max(1, n // 2)
    length = int(rng.integers(lo, hi + 1))
    start = int(rng.integers(0, n - length + 1))
    sub = ld[start:start + length]
    sub_set = set(sub.tolist())
    first = sub[0]
    rest, insert_at = [], None
    for g in lr:
        if g == first:
            insert_at = len(rest)
        if g not in sub_set:
            rest.append(g)
    child_lab = rest[:insert_at] + sub.tolist() + rest[insert_at:]
    return np.array([g // n for g in child_lab], dtype=int)


def uniform_ma(rng, ma1, ma2):
    return np.where(rng.random(len(ma1)) < 0.5, ma1, ma2)


def local_search(prob, ind, nm=NM, deadline=float('inf')):
    """Critical-path local search (Algorithm 5) with machine reassignment.

    Returns (individual, n_improving_moves). The returned individual is
    never worse than the input.
    """
    cur = ind.copy()
    n_impr = 0
    for _ in range(nm):
        path, kinds = prob.critical_path(cur.ct, cur.seqs, cur.ma)
        pos_of = np.empty(prob.n_ops, dtype=float)
        pos_of[cur.order] = np.arange(prob.n_ops, dtype=float)
        found = None
        for k, v in enumerate(path):
            if time.time() > deadline:
                break
            # (a) paper move: swap the machine arc u -> v on the critical path
            if k < len(kinds) and kinds[k] == 'mch':
                u = path[k + 1]
                m = cur.ma[v]
                seqs = [list(s) for s in cur.seqs]
                iu = seqs[m].index(u)
                seqs[m][iu], seqs[m][iu + 1] = v, u
                key = pos_of.copy()
                key[v] = pos_of[u] - 0.5
                new_os = prob.os_from_seqs(seqs, key)
                cand = Individual(prob, new_os, cur.ma)
                if cand.ms < cur.ms - TOL:
                    found = cand
                    break
            # (b) flexible-shop move: reassign v to another compatible machine
            for m2 in prob.compat[v]:
                if m2 == cur.ma[v]:
                    continue
                ma2 = cur.ma.copy()
                ma2[v] = m2
                cand = Individual(prob, cur.os, ma2)
                if cand.ms < cur.ms - TOL:
                    found = cand
                    break
            if found is not None:
                break
        if found is None:
            break
        cur = found
        n_impr += 1
        if time.time() > deadline:
            break
    return cur, n_impr


# --------------------------------------------------------------------------- #
# Rule seeding through the lag-aware environment
# --------------------------------------------------------------------------- #
def _import_env(variant):
    """Import the environment with the CPU device, whatever the process argv."""
    if 'params' not in sys.modules:
        saved = sys.argv
        sys.argv = [saved[0], '--device', 'cpu']
        try:
            import params  # noqa: F401
        finally:
            sys.argv = saved
    from params import configs
    configs.device = 'cpu'
    import torch
    from fjsp_env_same_op_nums import EnvState, FJSPEnvForSameOpNums
    EnvState.device = torch.device('cpu')
    from common_utils import heuristic_select_action
    if variant == 'variable':
        from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums

        def make(n_j, n_m):
            return FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m,
                                           use_lag_features=False,
                                           isolate_padding=True)
    else:
        def make(n_j, n_m):
            return FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
    return make, heuristic_select_action


def rule_schedule(prob, rule, tie_seed, variant='fixed'):
    """Roll out one dispatching rule; return (assigned_mch, true_op_ct)."""
    make, select = _import_env(variant)
    np.random.seed(tie_seed)
    M = prob.n_mch
    env = make(prob.n_jobs, M)
    env.set_initial_data([prob.job_length], [prob.op_pt],
                         time_lag_list=[prob.time_lag])
    assigned = np.full(prob.n_ops, -1, dtype=int)
    while not env.done().all():
        a = select(rule, env)
        assigned[env.candidate[0, a // M]] = a % M
        env.step(np.array([a]))
    if (assigned < 0).any():
        raise RuntimeError(f'{rule}: some operation never scheduled')
    return assigned, env.true_op_ct[0].copy(), float(env.current_makespan[0])


def seed_individual(prob, assigned, op_ct):
    """Chromosome whose decoding keeps the rule's machine sequences."""
    st = op_ct - prob.op_pt[np.arange(prob.n_ops), assigned]
    order = sorted(range(prob.n_ops), key=lambda i: (st[i], i))
    return Individual(prob, prob.op_job[np.array(order)], assigned)


def build_seeds(prob, pdr_tie_seed, alt_tie_seed, variant='fixed', n1=N1,
                rules=SEED_RULES):
    """Up to n1 distinct rule-seeded individuals and the per-rule makespans."""
    seeds, marks, info = [], set(), []
    for rnd, tie in enumerate((pdr_tie_seed, alt_tie_seed)):
        for r in rules:
            if len(seeds) >= n1:
                break
            try:
                am, oct_, env_ms = rule_schedule(prob, r, tie, variant)
            except Exception as exc:  # a rule missing from common_utils
                info.append({'rule': r, 'round': rnd, 'error': str(exc)})
                continue
            ind = seed_individual(prob, am, oct_)
            info.append({'rule': r, 'round': rnd, 'tie_seed': int(tie),
                         'env_makespan': env_ms, 'decoded_makespan': ind.ms})
            if ind.mark not in marks:
                marks.add(ind.mark)
                seeds.append(ind)
    return seeds, info


# --------------------------------------------------------------------------- #
# Memetic algorithm (one instance)
# --------------------------------------------------------------------------- #
def run_memetic(prob, rng, seeds=(), budget_s=None, max_gens=MNI, nc=NC,
                pm=PM, nm=NM, np_restart=NP_RESTART, pr=PR, t_origin=None):
    """Steady-state memetic algorithm. Returns (best Individual, stats dict).

    budget_s bounds the wall-clock from the call; max_gens bounds the number
    of iterations. t_origin (time.time() at instance start) is used only to
    report time-to-best including the seeding time.
    """
    t0 = time.time()
    t_origin = t0 if t_origin is None else t_origin
    deadline = t0 + budget_s if budget_s is not None else float('inf')

    pop, marks = [], []
    for s in seeds:
        if len(pop) < nc and s.mark not in marks:
            pop.append(s.copy())
            marks.append(s.mark)
    tries = 0
    while len(pop) < nc:
        ind = random_individual(prob, rng)
        tries += 1
        if ind.mark in marks and tries < 50 * nc:
            continue
        pop.append(ind)
        marks.append(ind.mark)
    pop.sort(key=lambda x: x.ms)

    ranks = np.arange(nc, 0, -1, dtype=float)       # best has rank nc
    sel_p = 2.0 * ranks / (nc * (nc + 1))
    half = nc // 2

    best = pop[0].copy()
    stats = dict(generations=0, ls_calls=0, ls_improvements=0, restarts=0,
                 clones_rejected=0, iter_to_best=0,
                 time_to_best_s=time.time() - t_origin,
                 initial_best=best.ms, trace=[(0, round(time.time() - t_origin, 3),
                                                best.ms)])
    unproductive = 0
    gen = 0
    while gen < max_gens and time.time() < deadline:
        gen += 1
        i1 = int(rng.choice(nc, p=sel_p))
        i2 = i1
        while i2 == i1:
            i2 = int(rng.choice(nc, p=sel_p))
        p1, p2 = pop[i1], pop[i2]
        c_os = gox(rng, p1.os, p2.os, prob.n_jobs)
        c_ma = uniform_ma(rng, p1.ma, p2.ma)
        child = Individual(prob, c_os, c_ma)
        if rng.random() < pm:
            stats['ls_calls'] += 1
            child, n_impr = local_search(prob, child, nm=nm, deadline=deadline)
            stats['ls_improvements'] += n_impr

        cur_marks = set(x.mark for x in pop)
        if child.mark in cur_marks:
            stats['clones_rejected'] += 1
        else:
            j = int(rng.integers(half, nc))
            pop[j] = child
            pop.sort(key=lambda x: x.ms)

        if pop[0].ms < best.ms - TOL:
            best = pop[0].copy()
            unproductive = 0
            stats['iter_to_best'] = gen
            stats['time_to_best_s'] = time.time() - t_origin
            stats['trace'].append((gen, round(time.time() - t_origin, 3),
                                   best.ms))
        else:
            unproductive += 1

        if unproductive >= np_restart:
            n_keep = max(1, int(round(nc * (1.0 - pr))))
            kept = pop[:n_keep]
            km = set(x.mark for x in kept)
            fresh, tries = [], 0
            while len(kept) + len(fresh) < nc:
                ind = random_individual(prob, rng)
                tries += 1
                if ind.mark in km and tries < 50 * nc:
                    continue
                km.add(ind.mark)
                fresh.append(ind)
            pop = sorted(kept + fresh, key=lambda x: x.ms)
            stats['restarts'] += 1
            unproductive = 0

    stats['generations'] = gen
    stats['search_s'] = time.time() - t0
    return best, stats


# --------------------------------------------------------------------------- #
# Contention evidence of the budgeted search
# --------------------------------------------------------------------------- #
def _search_timing(args, budget, search_budget, st, clk, cpu_i, wall_i):
    """solve_timing of one instance's search (one worker thread). The clock
    covers the search, or seeding plus search with --budget_includes_seeding
    (the budget then bounds both). status is 'max_gens' when the iteration cap
    ended the search, 'budget' otherwise; time_limit_s is None without a
    budget (hit_limit is then False)."""
    if budget is not None and args.budget_includes_seeding:
        wall_s = time.perf_counter() - wall_i
        cpu_s = time.process_time() - cpu_i
        limit = budget
    else:
        wall_s, cpu_s = clk.wall_s, clk.cpu_s
        limit = search_budget if search_budget is not None else float('inf')
    status = 'max_gens' if st['generations'] >= args.max_gens else 'budget'
    t = solve_timing(wall_s, cpu_s, 1, limit, status)
    if budget is None:
        t['time_limit_s'] = None
    return t


def contention_line(method, s):
    """One summary line of a contention_summary dict."""
    cs = 'n/a' if s['cpu_share_min'] is None else f'{s["cpu_share_min"]:.3f}'
    return (f'contention {method}: records {s["n_records"]}, timed runs '
            f'{s["n_timed_solves"]}, budget-limited {s["n_hit_limit"]}, '
            f'min cpu_share {cs}, cpu_share < {LOW_CPU_SHARE} '
            f'{s["n_low_share"]}, no evidence {s["n_no_evidence"]}')


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_cli(argv=None):
    ap = argparse.ArgumentParser(
        description='Memetic algorithm (Caumond et al. 2008) for FJSP-TL')
    ap.add_argument('--data_path', type=str,
                    default='data/PPVC/10x25+ppvc-mixed')
    ap.add_argument('--budget_s', type=float, default=60.0,
                    help='per-instance wall-clock budget of the search (s)')
    ap.add_argument('--max_gens', type=int, default=MNI,
                    help='iteration cap (paper mni); used alone for '
                         'deterministic runs with --budget_s 0')
    ap.add_argument('--budget_includes_seeding', action='store_true',
                    help='count the rule seeding against --budget_s')
    ap.add_argument('--max_instances', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--seed_test', type=int, default=50,
                    help='rule tie-break seed base, as in eval_ppvc.py')
    ap.add_argument('--np_restart', type=int, default=NP_RESTART)
    ap.add_argument('--save_dir', type=str, default=None)
    ap.add_argument('--tag', type=str, default=None,
                    help='method tag in the file name (default MA<budget>)')
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_cli(argv)
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    all_stems = list_instances(args.data_path)
    stems = all_stems[:args.max_instances] if args.max_instances else all_stems
    n = len(stems)
    dataset = os.path.basename(os.path.normpath(args.data_path))
    variant = route_variant(args.data_path)
    budget = args.budget_s if args.budget_s and args.budget_s > 0 else None
    tag = args.tag or (f'MA{int(round(args.budget_s))}' if budget
                       else f'MAg{args.max_gens}')
    save_dir = args.save_dir or os.path.join('test_results', 'PPVC', dataset)
    if args.save_dir is None and n < len(all_stems):
        save_dir = os.path.join(save_dir, 'partial')

    print('=' * 72)
    print(f'eval_memetic  dataset={dataset} ({variant} routes)  instances={n}  '
          f'budget={budget}s  max_gens={args.max_gens}  seed={args.seed}')
    print(f'  nc={NC} n1={N1} pm={PM} nm={NM} np={args.np_restart} pr={PR}; '
          f'fitness = right_shift_repair longest path')
    print('=' * 72)

    results = np.zeros((n, 2))
    records = []
    for i, stem in enumerate(stems):
        base = os.path.basename(stem)
        jl, pt, meta = load_instance(stem)
        prob = Problem(jl, pt, np.asarray(meta['time_lag']))
        la_start = loadavg_now()
        cpu_i, wall_i = time.process_time(), time.perf_counter()
        t_start = time.time()
        seeds, seed_info = build_seeds(
            prob, pdr_tie_seed=args.seed_test + i,
            alt_tie_seed=args.seed * 100003 + 7919 + i, variant=variant)
        seed_s = time.time() - t_start
        search_budget = budget
        if budget is not None and args.budget_includes_seeding:
            search_budget = max(0.0, budget - seed_s)
        rng = np.random.default_rng(args.seed * 100003 + i)
        with SolveClock() as clk:
            best, st = run_memetic(prob, rng, seeds, budget_s=search_budget,
                                   max_gens=args.max_gens,
                                   np_restart=args.np_restart,
                                   t_origin=t_start)
        total_s = time.time() - t_start
        timing = _search_timing(args, budget, search_budget, st, clk,
                                cpu_i, wall_i)

        seqs, _ = prob.decode(best.os, best.ma)
        ct, ms = prob.evaluate_seqs(seqs, best.ma)
        res = validate_schedule(prob.job_length, prob.op_pt, prob.time_lag,
                                best.ma, ct)
        assert res['feasible'], (f'{base}: infeasible schedule\n  '
                                 + '\n  '.join(res['violations'][:10]))
        assert abs(res['makespan'] - best.ms) < TOL and abs(ms - best.ms) < TOL
        results[i] = [best.ms, total_s]

        seed_ms = [d['decoded_makespan'] for d in seed_info
                   if 'decoded_makespan' in d]
        best_seed = min(seed_ms) if seed_ms else None
        spt = [d['decoded_makespan'] for d in seed_info
               if d.get('rule') == 'SPT' and d.get('round') == 0
               and 'decoded_makespan' in d]
        records.append(dict(
            instance=base, best_makespan=best.ms,
            best_seed_makespan=best_seed,
            spt_makespan=spt[0] if spt else None,
            generations=st['generations'], ls_calls=st['ls_calls'],
            ls_improvements=st['ls_improvements'], restarts=st['restarts'],
            clones_rejected=st['clones_rejected'],
            iter_to_best=st['iter_to_best'],
            time_to_best_s=round(st['time_to_best_s'], 3),
            seed_s=round(seed_s, 3), search_s=round(st['search_s'], 3),
            total_s=round(total_s, 3),
            s_per_generation=(st['search_s'] / st['generations']
                              if st['generations'] else None),
            n_seeds=len(seeds), validator_violations=len(res['violations']),
            seeds=seed_info, improvement_trace=st['trace'],
            **timing, **contention_fields([timing], la_start)))
        print(f'  [{i + 1:3d}/{n}] {base}  MA={best.ms:7.1f}  '
              f'best_seed={best_seed}  gens={st["generations"]:6d}  '
              f'ls={st["ls_calls"]}/{st["ls_improvements"]}  '
              f'ttb={st["time_to_best_s"]:.1f}s  seed={seed_s:.1f}s  '
              f'total={total_s:.1f}s  violations={len(res["violations"])}  '
              f'cpu_share={timing["cpu_share"]:.2f}')

    contention = contention_summary(records)
    os.makedirs(save_dir, exist_ok=True)
    stem_out = os.path.join(save_dir, f'Result_{tag}+memetic_{dataset}')
    np.save(stem_out + '.npy', results)
    with open(stem_out + '.json', 'w') as f:
        json.dump(dict(
            method='memetic (Caumond, Lacomme, Tchernev 2008), flexible-shop '
                   'adaptation', dataset=dataset, route_variant=variant,
            n_instances=n, budget_s=budget,
            budget_includes_seeding=bool(args.budget_includes_seeding),
            max_gens=args.max_gens, seed=args.seed, seed_test=args.seed_test,
            params=dict(nc=NC, n1=N1, np=args.np_restart, pm=PM, mni=MNI,
                        nm=NM, pr=PR, mark_K=MARK_K, rules=list(SEED_RULES)),
            instances=records, contention=contention), f, indent=1)
    print('-' * 72)
    print(f'  mean makespan {results[:, 0].mean():.2f}  '
          f'mean seconds {results[:, 1].mean():.1f}  feasible {n}/{n}')
    print('  ' + contention_line('memetic', contention))
    print(f'  saved {stem_out}.npy and .json')
    return results, records


if __name__ == '__main__':
    main()
