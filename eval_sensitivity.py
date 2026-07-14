"""
eval_sensitivity.py
===================
Duration-sensitivity robustness experiment for the PPVC time-lag-aware DRL paper.

WHY
  The benchmark processing-time DURATIONS are literature-based estimates ([E]),
  so the paper's conclusions must not rest on their precise values.
  This script DEMONSTRATES, rather than asserts, that the
  central relative ranking -- the trained A3 greedy policy beats the best
  priority rule (SPT) -- is INVARIANT to multiplicative perturbations of the
  processing times. The time-lags are NOT perturbed: they are [N] industry
  ranges that are separately justified in the paper.

WHAT IT DOES
  For each perturbation magnitude delta in {0, 0.1, 0.2, 0.3}, and for each of
  the 100 M-class test instances (data/PPVC/10x25+ppvc-mixed):
    * Draw, with a per-(instance, delta) seeded RNG, an i.i.d. factor in
      [1-delta, 1+delta] for EVERY compatible (op, machine) processing time,
      multiply, and round to a >=1 integer (delta=0 leaves pt unchanged).
    * Re-run the trained A3 greedy policy AND the SPT priority rule on the
      perturbed instance, REUSING eval_ppvc.rollout_greedy / rollout_heuristic
      (which themselves drive the real FJSP env + the real DANIEL policy +
      common_utils.heuristic_select_action). Nothing is reimplemented here.
    * Re-validate the produced A3 schedule with schedule_validator.validate_schedule
      against the (unperturbed) TRUE lags + the perturbed pt; assert 0 violations.
    * Record per-instance makespans for A3 and SPT.

  KEY OUTPUT per delta: mean A3 makespan, mean SPT makespan, the A3-beats-SPT
  win/tie/loss count over the 100 instances, and the paired Wilcoxon p-value
  (A3 vs SPT). Headline claim to verify: A3 < SPT (A3 wins the majority of
  pairs, p < 0.05) at EVERY delta. delta=0 reproduces the paper baseline
  (A3 ~210.9 / SPT ~216.0).

REUSE (do NOT reimplement the env or the policy)
  * eval_ppvc.rollout_greedy   -> eval_ppvc.py:164-197  (DRL greedy rollout)
  * eval_ppvc.rollout_heuristic -> eval_ppvc.py:139-161 (PDR rollout; SPT path
    routes to common_utils.heuristic_select_action, common_utils.py:133-142)
  * eval_ppvc._new_env          -> eval_ppvc.py:127-136  (lag-aware env builder)
  * eval_ppvc.load_train_config -> eval_ppvc.py:95-110   (arch flags -> configs)
  * eval_ppvc.list_instances    -> eval_ppvc.py:113-118

CLI
  python eval_sensitivity.py [--data_path data/PPVC/10x25+ppvc-mixed]
      [--deltas 0,0.1,0.2,0.3] [--seed 0] [--max_instances N]
      [--model_name 10x25+ppvc-mixed+full] [--seed_test 50]

OUTPUT
  * a clean summary table to stdout
  * test_results/PPVC/<dataset>/sensitivity_summary.md (table + provenance)
  * a copy-paste \newcommand macro block for the paper appended to that file
"""
import argparse
import os
import sys


def parse_cli():
    """Parse our CLI, then scrub argv BEFORE params.py's import-time parse_args.

    Mirrors eval_ppvc.parse_cli (eval_ppvc.py:44-67): params.configs is built by
    a parser.parse_args() that runs at IMPORT time against process argv, so our
    own flags must be parsed and removed first.
    """
    ap = argparse.ArgumentParser(
        description='Duration-sensitivity robustness sweep for PPVC A3 vs SPT')
    ap.add_argument('--data_path', type=str,
                    default='data/PPVC/10x25+ppvc-mixed',
                    help='directory with instance_*.fjs + .meta.json')
    ap.add_argument('--deltas', type=str, default='0,0.1,0.2,0.3',
                    help='comma-separated perturbation magnitudes; each pt is '
                         'scaled by U[1-delta, 1+delta]')
    ap.add_argument('--seed', type=int, default=0,
                    help='master seed for the per-instance perturbation RNG')
    ap.add_argument('--max_instances', type=int, default=None,
                    help='cap the number of instances (smoke testing)')
    ap.add_argument('--model_name', type=str, default='10x25+ppvc-mixed+full',
                    help='A3 checkpoint stem under trained_network/PPVC/')
    ap.add_argument('--seed_test', type=int, default=50,
                    help='global eval seed (matches eval_ppvc default; SPT '
                         'tie-breaks reseed to seed_test + instance index)')
    args = ap.parse_args()
    args.deltas = [float(x) for x in args.deltas.split(',') if x.strip() != '']
    sys.argv = [sys.argv[0]]  # clean argv for params.py's import-time parse
    return args


_ARGS = parse_cli()

# Importing eval_ppvc runs its module-level parse_cli() against the (now
# scrubbed) argv -> harmless defaults; it does NOT import params at that point.
# We then reuse its rollout/env/config helpers verbatim.
import numpy as np

from params import configs
import eval_ppvc


def perturb_pt(op_pt, delta, rng):
    """Return a perturbed copy of op_pt for one instance.

    Every COMPATIBLE (op, machine) processing time (op_pt > 0) is multiplied by
    an i.i.d. factor drawn uniformly from [1-delta, 1+delta], then rounded to a
    >=1 integer. Incompatible entries (0) stay 0 so the routing/flexibility
    structure is preserved. delta == 0 is an exact no-op (returns op_pt rounded
    to its original integer values).

    Parameters
    ----------
    op_pt : (N, M) float array, 0 = incompatible
    delta : float perturbation magnitude
    rng   : numpy Generator (seeded per (instance, delta) for reproducibility)
    """
    op_pt = np.asarray(op_pt, dtype=float)
    out = op_pt.copy()
    mask = op_pt > 0
    if delta == 0.0:
        # exact baseline: keep original integer durations
        out[mask] = np.rint(op_pt[mask])
    else:
        factors = rng.uniform(1.0 - delta, 1.0 + delta, size=int(mask.sum()))
        scaled = op_pt[mask] * factors
        out[mask] = np.maximum(1.0, np.rint(scaled))
    return out


def instance_seed(master_seed, delta, instance_idx):
    """Deterministic per-(instance, delta) seed for the perturbation RNG.

    Reproducible from --seed alone; distinct deltas / instances get independent
    factor draws. delta is folded in as round(delta*1000) so {0,0.1,0.2,0.3} map
    to {0,100,200,300}.
    """
    return (int(master_seed) * 1_000_003
            + int(round(delta * 1000)) * 10_007
            + int(instance_idx))


def run():
    args = _ARGS

    # ---- arch flags -> configs (reuse eval_ppvc.load_train_config) ----------
    snap = eval_ppvc.load_train_config(args.model_name)
    use_lag_features = bool(getattr(configs, 'use_lag_features', False))
    use_type_embedding = bool(getattr(configs, 'use_type_embedding', False))

    # ---- torch / device setup (mirror eval_ppvc.main lines 309-318) ---------
    os.environ['CUDA_VISIBLE_DEVICES'] = configs.device_id
    import torch
    from common_utils import setup_seed
    device = torch.device(configs.device)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device('cuda' if device.type == 'cuda' else 'cpu')
    setup_seed(args.seed_test)

    # ---- load the A3 checkpoint into a matching DANIEL ----------------------
    from model.PPO import PPO_initialize
    ckpt_path = f'./trained_network/PPVC/{args.model_name}.pth'
    if not os.path.exists(ckpt_path):
        sys.exit(f'[eval_sensitivity] checkpoint not found: {ckpt_path}')
    ppo = PPO_initialize()
    ppo.policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    ppo.policy.eval()

    # ---- instances ----------------------------------------------------------
    from ppvc_instance_generator import load_instance
    from schedule_validator import validate_schedule
    stems = eval_ppvc.list_instances(args.data_path)
    if args.max_instances is not None:
        stems = stems[:args.max_instances]
    n = len(stems)
    dataset = os.path.basename(os.path.normpath(args.data_path))

    print('=' * 78)
    print(f'eval_sensitivity  model={args.model_name}  dataset={dataset}  '
          f'instances={n}')
    print(f'  use_lag_features={use_lag_features}  '
          f'use_type_embedding={use_type_embedding}  '
          f'seed={args.seed}  seed_test={args.seed_test}')
    print(f'  deltas={args.deltas}  (pt scaled by U[1-d,1+d], rounded >=1; '
          f'lags UNCHANGED)')
    print('=' * 78)

    # pre-load every instance once (lags + base pt reused across deltas)
    loaded = [load_instance(stem) for stem in stems]
    bases = [os.path.basename(s) for s in stems]

    # per-delta results
    rows = []          # (delta, a3_mean, spt_mean, win, tie, loss, pval, holds)
    macro_records = [] # (delta, a3_mean, spt_mean, pval) for the macro block
    total_infeas = 0

    for delta in args.deltas:
        a3 = np.zeros(n)
        spt = np.zeros(n)
        for i, (jl, pt_base, meta) in enumerate(loaded):
            rng = np.random.default_rng(
                instance_seed(args.seed, delta, i))
            pt = perturb_pt(pt_base, delta, rng)

            # --- A3 greedy (reuse eval_ppvc.rollout_greedy, eval_ppvc.py:164) -
            ms_a3, _, amch, oct_ = eval_ppvc.rollout_greedy(
                ppo, jl, pt, meta, use_lag_features, use_type_embedding)
            # validate A3 schedule against TRUE lags + PERTURBED pt
            res = validate_schedule(jl, pt, meta['time_lag'], amch, oct_)
            if not res['feasible']:
                total_infeas += 1
                print(f'  [INFEASIBLE] delta={delta} {bases[i]}: '
                      + '; '.join(res['violations'][:5]))
            assert res['feasible'], (
                f'A3 produced an INFEASIBLE schedule (delta={delta}) on '
                f'{bases[i]}:\n  ' + '\n  '.join(res['violations'][:10]))
            assert abs(res['makespan'] - ms_a3) < 1e-6, (
                f'A3 {bases[i]}: env makespan {ms_a3} != validator '
                f'{res["makespan"]}')
            a3[i] = ms_a3

            # --- SPT priority rule (reuse eval_ppvc.rollout_heuristic, :139) --
            # per-instance reseed mirrors eval_ppvc.main:367 (seed_test + i) so
            # the PDR tie-breaks match the paper's main-table run at delta=0.
            ms_spt, _, _, _ = eval_ppvc.rollout_heuristic(
                'SPT', jl, pt, meta, seed=args.seed_test + i)
            spt[i] = ms_spt

        # --- paired stats over the n instances ------------------------------
        diff = a3 - spt
        win = int((diff < 0).sum())   # A3 strictly beats SPT
        tie = int((diff == 0).sum())
        loss = int((diff > 0).sum())
        pval = _wilcoxon_p(a3, spt)
        a3_mean = float(a3.mean())
        spt_mean = float(spt.mean())
        # headline: A3 wins the majority AND the difference is significant
        holds = (win > loss) and (pval < 0.05)
        rows.append((delta, a3_mean, float(a3.std()), spt_mean,
                     float(spt.std()), win, tie, loss, pval, holds))
        macro_records.append((delta, a3_mean, spt_mean, pval))

        print(f'  delta={delta:<4}  A3={a3_mean:6.1f}  SPT={spt_mean:6.1f}  '
              f'W/T/L={win}/{tie}/{loss}  p={_fmt_p(pval)}  '
              f'A3<SPT? {"YES" if holds else "NO"}')

    print(f'\n  total A3 infeasibilities across all deltas: {total_infeas}')
    assert total_infeas == 0, (
        f'{total_infeas} infeasible A3 schedule(s) detected -- aborting')

    _write_outputs(args, dataset, n, ckpt_path, use_lag_features,
                   use_type_embedding, rows, macro_records, snap)
    _print_table(dataset, n, rows)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _wilcoxon_p(a, b):
    """Two-sided paired Wilcoxon signed-rank p-value for a vs b.

    Uses scipy if available (the fjsp conda env has it); if every pair is
    identical (all-zero differences, e.g. a degenerate single-instance smoke)
    scipy raises -> return 1.0 (no evidence of a difference).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    from scipy.stats import wilcoxon
    if np.all(a - b == 0):
        return 1.0
    try:
        # zero_method='wilcox' drops zero-differences (scipy default);
        # 'pratt' would keep them. We keep the scipy default to match how the
        # paper's main greedy-vs-SPT Wilcoxon p-value (\WilcoxonP) was computed.
        stat, p = wilcoxon(a, b, alternative='two-sided')
        return float(p)
    except ValueError:
        # raised when all differences are zero after dropping ties
        return 1.0


def _fmt_p(p):
    """Compact p-value string for console / table."""
    if p < 1e-4:
        return f'{p:.1e}'
    return f'{p:.4f}'


# ---------------------------------------------------------------------------
# Output: markdown summary + paper macro block
# ---------------------------------------------------------------------------

_DELTA_WORD = {0.0: 'Zero', 0.1: 'Ten', 0.2: 'Twenty', 0.3: 'Thirty',
               0.4: 'Forty', 0.5: 'Fifty'}


def _delta_word(delta):
    """Map a delta to the CamelCase word used in the \\Sens... macro names."""
    if delta in _DELTA_WORD:
        return _DELTA_WORD[delta]
    # generic fallback: 0.15 -> 'Pct15'
    return 'Pct' + str(int(round(delta * 100)))


def _macro_p(p):
    """LaTeX-math-friendly p-value (matches macros.tex \\WilcoxonP style)."""
    if p < 1e-4:
        mant, exp = f'{p:.1e}'.split('e')
        return f'{mant}\\times 10^{{{int(exp)}}}'
    return f'{p:.4f}'


def _write_outputs(args, dataset, n, ckpt_path, use_lag_features,
                   use_type_embedding, rows, macro_records, snap):
    save_dir = f'./test_results/PPVC/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, 'sensitivity_summary.md')

    L = []
    L.append('# Duration-sensitivity robustness summary\n')
    L.append(f'- dataset: `{dataset}`  ({n} instances'
             + ('' if args.max_instances is None
                else f', capped from --max_instances={args.max_instances}')
             + ')')
    L.append(f'- A3 checkpoint: `{ckpt_path}`')
    L.append(f'- architecture (from snapshot): '
             f'use_lag_features={use_lag_features}, '
             f'use_type_embedding={use_type_embedding}')
    L.append(f'- perturbation: every compatible processing time scaled by '
             f'i.i.d. U[1-delta, 1+delta], rounded to >=1 integer hours; '
             f'time-lags UNCHANGED')
    L.append(f'- master seed: {args.seed}  (per-instance RNG = '
             f'f(seed, delta, instance))  |  seed_test: {args.seed_test}')
    L.append(f'- stat: two-sided paired Wilcoxon signed-rank (A3 vs SPT)')
    L.append(f'- headline claim verified at each delta: '
             f'A3 wins the majority of pairs AND p < 0.05')
    L.append('')
    L.append('| delta | A3 mean (std) | SPT mean (std) | A3 W/T/L vs SPT | '
             'Wilcoxon p | A3 < SPT? |')
    L.append('|---|---|---|---|---|---|')
    for (delta, a3m, a3s, sptm, spts, win, tie, loss, pval, holds) in rows:
        L.append(f'| {delta:g} | {a3m:.1f} ({a3s:.1f}) | '
                 f'{sptm:.1f} ({spts:.1f}) | {win}/{tie}/{loss} | '
                 f'{_fmt_p(pval)} | {"YES" if holds else "NO"} |')
    L.append('')

    all_hold = all(r[-1] for r in rows)
    L.append(f'**Conclusion:** the A3 < SPT ranking '
             + ('HOLDS at every tested delta '
                if all_hold else 'does NOT hold at some delta ')
             + '(see the A3 < SPT column). The central relative ranking is '
             + ('robust to' if all_hold else 'sensitive to')
             + ' multiplicative duration perturbations of up to '
             + f'+/-{max(r[0] for r in rows):g}.')
    L.append('')

    # ---- paper macro block --------------------------------------------------
    L.append('---\n')
    L.append('## Copy-paste \\newcommand macro block (for macros.tex)\n')
    L.append('```latex')
    L.append('% --- duration-sensitivity robustness (eval_sensitivity.py) ---')
    for (delta, a3m, sptm, pval) in macro_records:
        w = _delta_word(delta)
        L.append(f'\\newcommand{{\\SensDelta{w}A3}}{{{a3m:.1f}}}'
                 f'   % A3 greedy mean makespan, delta={delta:g} (h)')
        L.append(f'\\newcommand{{\\SensDelta{w}Spt}}{{{sptm:.1f}}}'
                 f'  % SPT mean makespan, delta={delta:g} (h)')
        L.append(f'\\newcommand{{\\SensDelta{w}P}}{{{_macro_p(pval)}}}'
                 f'  % Wilcoxon p, A3 vs SPT, delta={delta:g}')
    # convenience: the max delta at which the ranking still holds
    held = [r[0] for r in rows if r[-1]]
    if held:
        L.append(f'\\newcommand{{\\SensMaxDelta}}{{{max(held):g}}}'
                 f'  % largest delta at which A3 < SPT still holds')
    L.append('```')
    L.append('')

    with open(out_path, 'w') as f:
        f.write('\n'.join(L) + '\n')
    print(f'\nwrote {out_path}')


def _print_table(dataset, n, rows):
    print('\n' + '=' * 78)
    print(f'SENSITIVITY SUMMARY  ({dataset}, {n} instances)')
    print('=' * 78)
    print(f'{"delta":>6}{"A3 mean":>10}{"SPT mean":>10}'
          f'{"A3 W/T/L":>12}{"p-value":>12}{"A3<SPT?":>10}')
    print('-' * 78)
    for (delta, a3m, a3s, sptm, spts, win, tie, loss, pval, holds) in rows:
        wtl = f'{win}/{tie}/{loss}'
        print(f'{delta:>6g}{a3m:>10.1f}{sptm:>10.1f}{wtl:>12}'
              f'{_fmt_p(pval):>12}{("YES" if holds else "NO"):>10}')
    print('=' * 78)
    if all(r[-1] for r in rows):
        print('RESULT: A3 < SPT holds at EVERY delta -> ranking is robust.')
    else:
        bad = [f'{r[0]:g}' for r in rows if not r[-1]]
        print(f'RESULT: A3 < SPT FAILS at delta(s): {", ".join(bad)}')
    print('=' * 78)


if __name__ == '__main__':
    run()
