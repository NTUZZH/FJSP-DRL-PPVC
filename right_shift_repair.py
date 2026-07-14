"""
right_shift_repair.py
---------------------
Right-shift repair for the ablation arm A0 ("lag-blind + right-shift repair")
of the PPVC paper. Models current industry practice: a scheduler that IGNORES
post-operation time-lags produces machine sequences; the realized schedule must
then be repaired by shifting operations RIGHT until all constraints hold.

Lag semantics (mirrors schedule_validator.validate_schedule):
  - After op i of a job completes at ct[i], the job's next op may start only at
    ct[i] + lag[i].  Machines are FREE during a lag (the lag occupies the
    module, not the station).
  - The machine SEQUENCES (the ORDER of ops on each machine) coming from the
    lag-blind schedule are KEPT.  Only start times shift right; there is no
    resequencing — that is the whole point of the baseline.

Operation numbering: ops are numbered job-by-job in precedence order, exactly
as in schedule_validator (job 0's ops first, then job 1's, ...). op_pt[i, m]==0
means machine m is incompatible with op i.

No repo imports — pure numpy.
"""

import numpy as np


def _job_op_ranges(job_length):
    """Return list of (start_idx, end_idx) global-op ranges, one per job."""
    ranges = []
    idx = 0
    for length in job_length:
        ranges.append((idx, idx + int(length)))
        idx += int(length)
    return ranges


def _derive_mch_sequences_from_starts(assigned_mch, lagblind_starts, n_mch):
    """
    Build, per machine, the op order implied by the lag-blind schedule, by
    sorting the ops on each machine by their lag-blind start time. Ties (equal
    start times) are broken by op index for determinism — they cannot truly
    overlap in a feasible lag-blind schedule, so the tie-break is cosmetic.
    """
    seqs = []
    for m in range(n_mch):
        ops_on_m = [i for i in range(len(assigned_mch)) if assigned_mch[i] == m]
        ops_on_m.sort(key=lambda i: (lagblind_starts[i], i))
        seqs.append(ops_on_m)
    return seqs


def right_shift_repair(job_length, op_pt, time_lag, assigned_mch,
                       op_sequence_per_mch=None, op_ct_lagblind=None,
                       mch_ready_time=None):
    """
    Repair a lag-blind schedule by shifting ops right until job-precedence
    (with lags) and machine no-overlap both hold, while KEEPING the machine
    sequences fixed.

    Parameters
    ----------
    job_length          : array-like, shape (J,)  — number of ops per job
    op_pt               : array-like, shape (N, M) — processing times; 0 = incompatible
    time_lag            : array-like, shape (N,)  — post-op lag after each op
    assigned_mch        : array-like, shape (N,)  — machine chosen for each op
    op_sequence_per_mch : optional, list length M; each entry an ordered list of
                          global op indices giving the fixed order on that machine.
    op_ct_lagblind      : optional, shape (N,) — lag-blind completion times, used
                          to DERIVE the per-machine op order (sort by start =
                          ct - pt) when op_sequence_per_mch is not supplied.
    mch_ready_time      : optional, shape (M,) — earliest time each machine is
                          available. Used for the MACHINE-BREAKDOWN disruption: a
                          machine down until t_release seeds a release floor on
                          the START of its FIRST sequenced op (which then
                          propagates right through the machine arcs). None (the
                          default) means every machine is ready at t=0, i.e. the
                          behavior is BIT-IDENTICAL to the original operator —
                          no machine-arc weights or job-arc weights change.

    Exactly ONE of {op_sequence_per_mch, op_ct_lagblind} must be given — the
    machine ORDER must come from the lag-blind schedule, not be re-optimised.

    Returns
    -------
    (repaired_op_ct [N], repaired_makespan float)

    Raises
    ------
    ValueError if neither / both optional args are given, on bad sequences, or
    if the precedence+machine arc graph contains a cycle (impossible when both
    arc sets come from one feasible lag-blind schedule, but guarded anyway).
    """
    job_length = np.asarray(job_length, dtype=int)
    op_pt = np.asarray(op_pt, dtype=float)
    time_lag = np.asarray(time_lag, dtype=float)
    assigned_mch = np.asarray(assigned_mch, dtype=int)

    n_ops = op_pt.shape[0]
    n_mch = op_pt.shape[1]

    if mch_ready_time is not None:
        mch_ready_time = np.asarray(mch_ready_time, dtype=float)
        if mch_ready_time.shape != (n_mch,):
            raise ValueError(
                f"mch_ready_time must have shape ({n_mch},), got "
                f"{mch_ready_time.shape}.")

    # ------------------------------------------------------------------
    # Require exactly one of the two ways of specifying the machine order
    # ------------------------------------------------------------------
    have_seq = op_sequence_per_mch is not None
    have_ct = op_ct_lagblind is not None
    if have_seq == have_ct:
        raise ValueError(
            "right_shift_repair: provide EXACTLY one of "
            "{op_sequence_per_mch, op_ct_lagblind} (machine order must come "
            "from the lag-blind schedule)."
        )

    # Per-op chosen processing time (on its assigned machine)
    pt = np.array([op_pt[i, assigned_mch[i]] for i in range(n_ops)], dtype=float)

    # ------------------------------------------------------------------
    # Obtain the fixed per-machine op sequences
    # ------------------------------------------------------------------
    if have_seq:
        mch_sequences = [list(seq) for seq in op_sequence_per_mch]
        if len(mch_sequences) != n_mch:
            raise ValueError(
                f"op_sequence_per_mch has {len(mch_sequences)} machines, "
                f"expected {n_mch}."
            )
        # validate: every op appears exactly once, on its assigned machine
        seen = np.zeros(n_ops, dtype=bool)
        for m, seq in enumerate(mch_sequences):
            for i in seq:
                if not (0 <= i < n_ops):
                    raise ValueError(f"op index {i} out of range in machine {m} sequence.")
                if assigned_mch[i] != m:
                    raise ValueError(
                        f"op {i} is in machine {m}'s sequence but assigned_mch[{i}]"
                        f"={assigned_mch[i]}."
                    )
                if seen[i]:
                    raise ValueError(f"op {i} appears more than once across machine sequences.")
                seen[i] = True
        if not seen.all():
            missing = np.where(~seen)[0].tolist()
            raise ValueError(f"ops missing from machine sequences: {missing}")
    else:
        op_ct_lagblind = np.asarray(op_ct_lagblind, dtype=float)
        lagblind_starts = op_ct_lagblind - pt
        mch_sequences = _derive_mch_sequences_from_starts(
            assigned_mch, lagblind_starts, n_mch)

    # ------------------------------------------------------------------
    # Build the arc set (precedence DAG):
    #   job-arcs    : op (k-1) -> op k, weight ct[k-1] + lag[k-1]
    #                 (successor may not start before predecessor's lag elapses)
    #   machine-arcs: prev-on-mch -> next-on-mch, weight ct[prev]
    #                 (machine no-overlap; next may not start before prev ends)
    # We store, per op, its list of predecessors with a tag telling which kind
    # of arc so the right Kahn-relaxation can be applied:
    #   ('job', pred)  -> child_start >= ct[pred] + lag[pred]
    #   ('mch', pred)  -> child_start >= ct[pred]
    # ------------------------------------------------------------------
    preds = [[] for _ in range(n_ops)]       # list of (kind, pred_op)
    indeg = np.zeros(n_ops, dtype=int)
    succs = [[] for _ in range(n_ops)]       # for Kahn: list of successor ops

    def add_arc(u, v, kind):
        preds[v].append((kind, u))
        succs[u].append(v)
        indeg[v] += 1

    # job-arcs (precedence within each job, in global numbering order)
    for (s, e) in _job_op_ranges(job_length):
        for k in range(s + 1, e):
            add_arc(k - 1, k, 'job')

    # machine-arcs (consecutive ops in each machine's fixed sequence)
    for seq in mch_sequences:
        for a, b in zip(seq[:-1], seq[1:]):
            add_arc(a, b, 'mch')

    # ------------------------------------------------------------------
    # Kahn topological sort over the union graph (detects cycles)
    # ------------------------------------------------------------------
    indeg_work = indeg.copy()
    # use a simple list as queue; order among ready nodes does not affect the
    # longest-path result, only the iteration order
    queue = [i for i in range(n_ops) if indeg_work[i] == 0]
    topo_order = []
    while queue:
        u = queue.pop()
        topo_order.append(u)
        for v in succs[u]:
            indeg_work[v] -= 1
            if indeg_work[v] == 0:
                queue.append(v)

    if len(topo_order) != n_ops:
        # some node never reached in-degree 0 -> a cycle exists
        in_cycle = [i for i in range(n_ops) if indeg_work[i] > 0]
        raise ValueError(
            "right_shift_repair: precedence+machine arc graph contains a CYCLE "
            f"(ops still blocked: {in_cycle[:20]}). The lag-blind machine "
            "sequences are inconsistent with job precedence; cannot repair by "
            "right-shifting alone."
        )

    # ------------------------------------------------------------------
    # Machine-breakdown release floor (only when mch_ready_time is given):
    # the FIRST op in each machine's fixed sequence may not start before that
    # machine is back up. Seeding just the first op is sufficient — the floor
    # propagates right through the machine arcs to every later op on the
    # machine in the longest-path pass below. With mch_ready_time=None this
    # dict is empty and nothing changes (default path bit-identical).
    # ------------------------------------------------------------------
    start_floor = {}
    if mch_ready_time is not None:
        for m, seq in enumerate(mch_sequences):
            if seq:
                first_op = seq[0]
                if mch_ready_time[m] > 0.0:
                    start_floor[first_op] = float(mch_ready_time[m])

    # ------------------------------------------------------------------
    # Longest-path / right-shift propagation in topological order.
    # For each op, start = max over predecessors of the required ready time;
    # ct = start + pt. Because we process in topo order, every predecessor's
    # ct is already final when we reach an op.
    # ------------------------------------------------------------------
    repaired_ct = np.zeros(n_ops, dtype=float)
    for u in topo_order:
        start = start_floor.get(u, 0.0)
        for kind, p in preds[u]:
            if kind == 'job':
                ready = repaired_ct[p] + time_lag[p]
            else:  # 'mch'
                ready = repaired_ct[p]
            if ready > start:
                start = ready
        repaired_ct[u] = start + pt[u]

    repaired_makespan = float(repaired_ct.max()) if n_ops > 0 else 0.0
    return repaired_ct, repaired_makespan
