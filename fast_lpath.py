"""
fast_lpath.py
=============
Fast exact longest-path evaluation of flexible-job-shop schedules with minimum
post-operation time-lags, and the chromosome kernels of the memetic algorithm
(eval_memetic.py --fast_eval).

Schedule semantics (identical to right_shift_repair.right_shift_repair with
op_sequence_per_mch): every operation starts at

    max(completion of its station predecessor,
        completion of its job predecessor + that predecessor's lag,
        release floor of its station if it is the station's first operation)

and completes at start + processing time on its assigned station. This is the
semi-active schedule, i.e. the longest path in the disjunctive graph whose job
arcs carry p + lag and whose machine arcs carry p. The arithmetic is the same
float64 arithmetic as right_shift_repair (ready = ct[pred] + lag[pred] or
ct[pred]; start = max of the readies and 0.0 or the floor; ct = start + pt),
and a maximum of float64 values does not depend on the order in which it is
taken, so the completion times are bit-identical to right_shift_repair
(test_fast_lpath.py checks this on random and real chromosomes).

Kernels
-------
longest_path(job_length, op_pt, time_lag, assigned_mch, op_sequence_per_mch,
             mch_ready_time=None)
    Drop-in replacement of right_shift_repair(..., op_sequence_per_mch=...):
    same validation, a ValueError on a cyclic arc set, returns (ct, makespan).
decode_eval(os, ma, job_start, job_end, op_pt_f, time_lag, mark_k)
    Bierwirth job-repetition chromosome (OS) + machine assignment (MA) ->
    completion times, makespan, operations in OS order, machine predecessor
    of every operation, and the clone mark sum(rint(start)^2) mod K. The OS
    order is a topological order of the job and machine arcs, so a single
    left-to-right pass gives the longest path.
critical_path(ct, ma, op_pt_f, mch_prev, job_pred, time_lag, tol)
    The backward critical path of eval_memetic.Problem.critical_path.
os_after_swap(job_pred, op_job, mch_prev, u, v, pos_of)
    eval_memetic.Problem.os_from_seqs after swapping the machine arc u -> v:
    the topological order of the swapped graph with the smallest key first,
    keyed by the old OS position and with v taking u's place.
gox(donor, receiver, job_start, start, length)
    Bierwirth's generalized order crossover of eval_memetic.gox for a given
    donor substring (start, length).
residual_makespan(...), pox_fill(p1_os, p2_os, in_set1)
    The residual decoder (ResidualProblem.makespan_os) and the deterministic
    part of eval_ga.pox_crossover, for the reactive GA of eval_reactive.py
    --fast_eval.

numba (0.68.0 in the fjsp environment) compiles every kernel. Without numba,
or with FAST_LPATH_NO_NUMBA=1, the same Python source runs uncompiled
(identical results, slower); each compiled kernel also exposes the
uncompiled function as .py_func.
"""
import os

import numpy as np

try:
    if os.environ.get('FAST_LPATH_NO_NUMBA', '0') == '1':
        raise ImportError('numba disabled by FAST_LPATH_NO_NUMBA')
    import numba
    from numba import njit
    HAVE_NUMBA = True
    NUMBA_VERSION = numba.__version__
except ImportError:  # pure-Python fallback with the same source
    HAVE_NUMBA = False
    NUMBA_VERSION = None

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            fn = args[0]
            fn.py_func = fn
            return fn

        def deco(fn):
            fn.py_func = fn
            return fn
        return deco


# --------------------------------------------------------------------------- #
# General evaluator: per-station sequences -> longest path (with validation)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _lpath_kernel(job_pred, pt, lag, assigned, flat_ops, flat_mch, floor_m):
    """Validate the station sequences and compute the longest path.

    Returns (ct, code, bad): code 0 = ok, 1 = op index out of range,
    2 = op on a station it is not assigned to, 3 = op listed twice,
    4 = op missing, 5 = cycle; bad is the offending op (or -1).
    """
    n = pt.shape[0]
    ct = np.zeros(n, dtype=np.float64)
    mch_prev = np.full(n, -1, dtype=np.int64)
    mch_next = np.full(n, -1, dtype=np.int64)
    floor = np.zeros(n, dtype=np.float64)
    seen = np.zeros(n, dtype=np.bool_)
    last = -1
    last_m = -1
    for k in range(flat_ops.shape[0]):
        i = flat_ops[k]
        m = flat_mch[k]
        if i < 0 or i >= n:
            return ct, 1, i
        if assigned[i] != m:
            return ct, 2, i
        if seen[i]:
            return ct, 3, i
        seen[i] = True
        if m == last_m:
            mch_prev[i] = last
            mch_next[last] = i
        else:
            # first op of station m: its release floor (0.0 when none)
            fm = floor_m[m]
            if fm > 0.0:
                floor[i] = fm
        last = i
        last_m = m
    for i in range(n):
        if not seen[i]:
            return ct, 4, i
    # Kahn topological order; every op has at most one job successor and
    # one machine successor.
    indeg = np.zeros(n, dtype=np.int64)
    job_next = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        if job_pred[i] >= 0:
            indeg[i] += 1
            job_next[job_pred[i]] = i
        if mch_prev[i] >= 0:
            indeg[i] += 1
    stack = np.empty(n, dtype=np.int64)
    top = 0
    for i in range(n):
        if indeg[i] == 0:
            stack[top] = i
            top += 1
    done = 0
    while top > 0:
        top -= 1
        u = stack[top]
        done += 1
        start = floor[u]
        p = job_pred[u]
        if p >= 0:
            ready = ct[p] + lag[p]
            if ready > start:
                start = ready
        q = mch_prev[u]
        if q >= 0:
            ready = ct[q]
            if ready > start:
                start = ready
        ct[u] = start + pt[u]
        w = job_next[u]
        if w >= 0:
            indeg[w] -= 1
            if indeg[w] == 0:
                stack[top] = w
                top += 1
        w = mch_next[u]
        if w >= 0:
            indeg[w] -= 1
            if indeg[w] == 0:
                stack[top] = w
                top += 1
    if done != n:
        for i in range(n):
            if indeg[i] > 0:
                return ct, 5, i
    return ct, 0, -1


def job_pred_of(job_length):
    """Global job predecessor of every operation (-1 for a job's first op)."""
    job_length = np.asarray(job_length, dtype=np.int64)
    n = int(job_length.sum())
    pred = np.arange(n, dtype=np.int64) - 1
    starts = np.concatenate([[0], np.cumsum(job_length)[:-1]]).astype(np.int64)
    pred[starts[job_length > 0]] = -1
    return pred


def longest_path(job_length, op_pt, time_lag, assigned_mch,
                 op_sequence_per_mch, mch_ready_time=None, job_pred=None):
    """Drop-in replacement of right_shift_repair(..., op_sequence_per_mch=...).

    Returns (completion times [N] float64, makespan float). Raises ValueError
    on malformed sequences and on a cyclic job + machine arc set.
    """
    op_pt = np.asarray(op_pt, dtype=float)
    time_lag = np.asarray(time_lag, dtype=np.float64)
    assigned = np.asarray(assigned_mch, dtype=np.int64)
    n_ops, n_mch = op_pt.shape
    if job_pred is None:
        job_pred = job_pred_of(job_length)
    if mch_ready_time is not None:
        mch_ready_time = np.asarray(mch_ready_time, dtype=float)
        if mch_ready_time.shape != (n_mch,):
            raise ValueError(
                f"mch_ready_time must have shape ({n_mch},), got "
                f"{mch_ready_time.shape}.")
        floor_m = mch_ready_time.astype(np.float64)
    else:
        floor_m = np.zeros(n_mch, dtype=np.float64)
    if len(op_sequence_per_mch) != n_mch:
        raise ValueError(
            f"op_sequence_per_mch has {len(op_sequence_per_mch)} machines, "
            f"expected {n_mch}.")
    lens = [len(s) for s in op_sequence_per_mch]
    flat_ops = (np.concatenate([np.asarray(s, dtype=np.int64)
                                for s in op_sequence_per_mch])
                if sum(lens) else np.zeros(0, dtype=np.int64))
    flat_mch = np.repeat(np.arange(n_mch, dtype=np.int64), lens)
    pt = op_pt[np.arange(n_ops), assigned] if n_ops else np.zeros(0)
    pt = np.ascontiguousarray(pt, dtype=np.float64)
    ct, code, bad = _lpath_kernel(job_pred, pt, time_lag, assigned,
                                  flat_ops, flat_mch, floor_m)
    if code != 0:
        msg = {1: f"op index {bad} out of range in the machine sequences.",
               2: f"op {bad} is in a machine sequence other than its "
                  f"assigned machine {assigned[bad] if 0 <= bad < n_ops else '?'}.",
               3: f"op {bad} appears more than once across machine sequences.",
               4: f"ops missing from machine sequences (first: {bad}).",
               5: "precedence+machine arc graph contains a CYCLE (first "
                  f"blocked op: {bad})."}[int(code)]
        raise ValueError('fast_lpath.longest_path: ' + msg)
    ms = float(ct.max()) if n_ops > 0 else 0.0
    return ct, ms


# --------------------------------------------------------------------------- #
# Memetic-algorithm kernels
# --------------------------------------------------------------------------- #
@njit(cache=True)
def decode_eval(os_vec, ma_vec, job_start, job_end, op_pt_f, lag, mark_k):
    """OS + MA -> (ct, makespan, order, mch_prev, mark, code).

    code 0 = ok; 1 = the OS is not a job-repetition sequence of the instance
    (too many or too few occurrences of a job).
    """
    n = op_pt_f.shape[0]
    n_m = op_pt_f.shape[1]
    ct = np.zeros(n, dtype=np.float64)
    order = np.empty(n, dtype=np.int64)
    mch_prev = np.full(n, -1, dtype=np.int64)
    last_on = np.full(n_m, -1, dtype=np.int64)
    nxt = job_start.copy()
    if os_vec.shape[0] != n:
        return ct, 0.0, order, mch_prev, 0, 1
    for p in range(n):
        j = os_vec[p]
        op = nxt[j]
        if op >= job_end[j]:
            return ct, 0.0, order, mch_prev, 0, 1
        nxt[j] = op + 1
        order[p] = op
        m = ma_vec[op]
        q = last_on[m]
        mch_prev[op] = q
        last_on[m] = op
        start = 0.0
        if op > job_start[j]:
            ready = ct[op - 1] + lag[op - 1]
            if ready > start:
                start = ready
        if q >= 0:
            ready = ct[q]
            if ready > start:
                start = ready
        ct[op] = start + op_pt_f[op, m]
    ms = ct[0]
    acc = 0
    for i in range(n):
        if ct[i] > ms:
            ms = ct[i]
        s = np.int64(np.rint(ct[i] - op_pt_f[i, ma_vec[i]]))
        acc += s * s
    return ct, ms, order, mch_prev, acc % mark_k, 0


@njit(cache=True)
def critical_path(ct, ma_vec, op_pt_f, mch_prev, job_pred, lag, tol):
    """Backward critical path (see eval_memetic.Problem.critical_path).

    Returns (path, kinds, code): kinds[k] = 1 for a machine arc path[k+1] ->
    path[k], 0 for a job arc; code 0 = ok, otherwise the op with no tight
    incoming arc is path[-1].
    """
    n = ct.shape[0]
    v = 0
    for i in range(1, n):
        if ct[i] > ct[v]:
            v = i
    path = np.empty(n, dtype=np.int64)
    kinds = np.empty(n, dtype=np.int64)
    length = 0
    path[0] = v
    length = 1
    nk = 0
    st_v = ct[v] - op_pt_f[v, ma_vec[v]]
    while st_v > tol:
        q = mch_prev[v]
        p = job_pred[v]
        if q >= 0 and q != p and abs(ct[q] - st_v) <= tol:
            kinds[nk] = 1
            v = q
        elif p >= 0 and abs(ct[p] + lag[p] - st_v) <= tol:
            kinds[nk] = 0
            v = p
        else:
            return path[:length], kinds[:nk], 1
        nk += 1
        path[length] = v
        length += 1
        st_v = ct[v] - op_pt_f[v, ma_vec[v]]
    return path[:length], kinds[:nk], 0


@njit(cache=True)
def os_after_swap(job_pred, op_job, mch_prev, u, v, pos_of):
    """OS of the graph after swapping the machine arc u -> v.

    The order is Kahn's algorithm that always takes the ready op with the
    smallest key, key = pos_of except key[v] = pos_of[u] - 0.5, exactly as
    eval_memetic.Problem.os_from_seqs with its heap of (key, op).
    Returns (os, code): code 0 = ok, 1 = v is not u's machine successor,
    2 = cycle.
    """
    n = job_pred.shape[0]
    out_os = np.empty(n, dtype=np.int64)
    if mch_prev[v] != u:
        return out_os, 1
    key = pos_of.copy()
    key[v] = pos_of[u] - 0.5
    # machine successor array of the swapped sequences: a -> v -> u -> b
    mprev = mch_prev.copy()
    b = -1
    for i in range(n):
        if mch_prev[i] == v:
            b = i
            break
    a = mch_prev[u]
    mprev[v] = a
    mprev[u] = v
    if b >= 0:
        mprev[b] = u
    mnext = np.full(n, -1, dtype=np.int64)
    jnext = np.full(n, -1, dtype=np.int64)
    indeg = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if mprev[i] >= 0:
            mnext[mprev[i]] = i
            indeg[i] += 1
        if job_pred[i] >= 0:
            jnext[job_pred[i]] = i
            indeg[i] += 1
    # binary min-heap of (key, op)
    hk = np.empty(n, dtype=np.float64)
    hv = np.empty(n, dtype=np.int64)
    size = 0
    for i in range(n):
        if indeg[i] == 0:
            # push
            c = size
            size += 1
            hk[c] = key[i]
            hv[c] = i
            while c > 0:
                par = (c - 1) // 2
                if hk[c] < hk[par] or (hk[c] == hk[par] and hv[c] < hv[par]):
                    tk = hk[c]; hk[c] = hk[par]; hk[par] = tk
                    tv = hv[c]; hv[c] = hv[par]; hv[par] = tv
                    c = par
                else:
                    break
    cnt = 0
    while size > 0:
        x = hv[0]
        # pop root
        size -= 1
        hk[0] = hk[size]
        hv[0] = hv[size]
        c = 0
        while True:
            l = 2 * c + 1
            r = l + 1
            s = c
            if l < size and (hk[l] < hk[s] or (hk[l] == hk[s] and hv[l] < hv[s])):
                s = l
            if r < size and (hk[r] < hk[s] or (hk[r] == hk[s] and hv[r] < hv[s])):
                s = r
            if s == c:
                break
            tk = hk[c]; hk[c] = hk[s]; hk[s] = tk
            tv = hv[c]; hv[c] = hv[s]; hv[s] = tv
            c = s
        out_os[cnt] = op_job[x]
        cnt += 1
        for w in (jnext[x], mnext[x]):
            if w >= 0:
                indeg[w] -= 1
                if indeg[w] == 0:
                    c = size
                    size += 1
                    hk[c] = key[w]
                    hv[c] = w
                    while c > 0:
                        par = (c - 1) // 2
                        if hk[c] < hk[par] or (hk[c] == hk[par]
                                               and hv[c] < hv[par]):
                            tk = hk[c]; hk[c] = hk[par]; hk[par] = tk
                            tv = hv[c]; hv[c] = hv[par]; hv[par] = tv
                            c = par
                        else:
                            break
    if cnt != n:
        return out_os, 2
    return out_os, 0


@njit(cache=True)
def gox(donor, receiver, job_start, start, length):
    """Generalized order crossover (eval_memetic.gox) for a drawn substring.

    Genes are (job, occurrence) pairs, identified here by the global op id
    job_start[job] + occurrence. The donor substring [start, start+length)
    is removed from the receiver and implanted where its first gene occurred
    in the receiver.
    """
    n = donor.shape[0]
    n_jobs = job_start.shape[0]
    cnt = np.zeros(n_jobs, dtype=np.int64)
    in_sub = np.zeros(n, dtype=np.bool_)
    first = -1
    for p in range(start + length):
        j = donor[p]
        op = job_start[j] + cnt[j]
        cnt[j] += 1
        if p >= start:
            in_sub[op] = True
            if p == start:
                first = op
    cnt[:] = 0
    child = np.empty(n, dtype=np.int64)
    r = 0
    insert_at = -1
    rest = np.empty(n - length, dtype=np.int64)
    for p in range(n):
        j = receiver[p]
        op = job_start[j] + cnt[j]
        cnt[j] += 1
        if op == first:
            insert_at = r
        if not in_sub[op]:
            rest[r] = j
            r += 1
    k = 0
    for p in range(insert_at):
        child[k] = rest[p]
        k += 1
    for p in range(start, start + length):
        child[k] = donor[p]
        k += 1
    for p in range(insert_at, n - length):
        child[k] = rest[p]
        k += 1
    return child


def warmup():
    """Compile (or load from the numba cache) every kernel on a tiny input
    with the production dtypes, so no compilation falls inside a budget."""
    job_length = np.array([2, 2], dtype=np.int64)
    op_pt = np.array([[1, 2], [2, 0], [0, 3], [1, 1]], dtype=np.int64)
    lag = np.array([0.5, 0.0, 1.0, 0.0])
    ma = np.array([0, 0, 1, 1], dtype=np.int64)
    jp = job_pred_of(job_length)
    longest_path(job_length, op_pt, lag, ma, [[0, 1], [2, 3]], job_pred=jp)
    js = np.array([0, 2], dtype=np.int64)
    je = np.array([2, 4], dtype=np.int64)
    os_vec = np.array([0, 1, 0, 1], dtype=np.int64)
    opf = op_pt.astype(np.float64)
    ct, ms, order, mp, mark, code = decode_eval(os_vec, ma, js, je, opf,
                                                lag, 49999)
    critical_path(ct, ma, opf, mp, jp, lag, 1e-6)
    pos = np.arange(4, dtype=np.float64)
    os_after_swap(jp, np.array([0, 0, 1, 1], dtype=np.int64), mp, 0, 1, pos)
    gox(os_vec, os_vec[::-1].copy(), js, 1, 2)
    return HAVE_NUMBA


# --------------------------------------------------------------------------- #
# Residual GA of the disruption study (eval_reactive.py --fast_eval)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def residual_makespan(os_vec, ma_vec, job_next0, job_end, ct0, tail0, ready0,
                      pt, lag, op_release, mch_release, t_d):
    """eval_reactive.ResidualProblem.makespan_os on arrays.

    The OS lists the jobs of the uncommitted ops; the k-th occurrence of job j
    is its k-th uncommitted op (job_next0[j] + k). Each op starts at
    max(t_d, job ready, machine tail, op release, machine release), in the
    order of comparisons of ResidualProblem.place, and the makespan is the
    largest completion time over all ops (committed ones keep ct0).
    Returns -1.0 when the OS lists a job more often than it has ops left.
    """
    ct = ct0.copy()
    tail = tail0.copy()
    ready = ready0.copy()
    nxt = job_next0.copy()
    for p in range(os_vec.shape[0]):
        j = os_vec[p]
        op = nxt[j]
        if op >= job_end[j]:
            return -1.0
        nxt[j] = op + 1
        m = ma_vec[op]
        s = t_d
        if ready[j] > s:
            s = ready[j]
        if tail[m] > s:
            s = tail[m]
        if op_release[op] > s:
            s = op_release[op]
        if mch_release[m] > s:
            s = mch_release[m]
        c = s + pt[op, m]
        ct[op] = c
        tail[m] = c
        ready[j] = c + lag[op]
    ms = ct[0]
    for i in range(1, ct.shape[0]):
        if ct[i] > ms:
            ms = ct[i]
    return ms


@njit(cache=True)
def pox_fill(p1_os, p2_os, in_set1):
    """eval_ga.pox_crossover after its two random draws: the child keeps p1's
    genes of the set-1 jobs in place and fills the other positions, in order,
    with p2's genes of the other jobs."""
    n = p1_os.shape[0]
    child = np.empty(n, dtype=p1_os.dtype)
    k = 0
    for pos in range(n):
        j = p1_os[pos]
        if in_set1[j]:
            child[pos] = j
        else:
            while in_set1[p2_os[k]]:
                k += 1
            child[pos] = p2_os[k]
            k += 1
    return child


def warmup_residual():
    """Compile (or load) the residual kernels with the production dtypes."""
    os_vec = np.array([0, 1, 0], dtype=np.int64)
    ma = np.array([0, 1, 0, 1], dtype=np.int64)
    residual_makespan(os_vec, ma, np.array([1, 3], dtype=np.int64),
                      np.array([2, 4], dtype=np.int64),
                      np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0]),
                      np.array([1.5, 0.0]), np.ones((4, 2)), np.zeros(4),
                      np.zeros(4), np.zeros(2), 1.0)
    pox_fill(os_vec, os_vec[::-1].copy(), np.array([True, False]))
    return HAVE_NUMBA
