"""
Collect the dispatching-rule campaign (run_pdr_scale.sh) into one markdown
table: mean makespan (std) in hours per rule and test-set size.

On 10x25 the classic rules come from the stored
Result_<rule>+10x25+ppvc-mixed+smoke files (the published numbers); every
other cell comes from Result_<rule>+pdr_<set>.npy.

usage: python make_pdr_scale_table.py
"""
import os

import numpy as np

RULES = ('FIFO', 'MOR', 'SPT', 'MWKR', 'SPTL', 'LLF', 'MWKRL', 'LRPTL')
SETS = ('10x25', '20x25', '30x25', '50x25', '80x25')
OUT = os.path.join('paper_tables', 'pdr_scale.md')


def result_path(rule, size):
    name = f'{size}+ppvc-mixed'
    tag = ('10x25+ppvc-mixed+smoke'
           if size == '10x25' and rule in RULES[:4] else 'pdr')
    return os.path.join('test_results', 'PPVC', name,
                        f'Result_{rule}+{tag}_{name}.npy')


def main():
    stats = {}
    for size in SETS:
        for rule in RULES:
            path = result_path(rule, size)
            if os.path.exists(path):
                arr = np.load(path)
                stats[rule, size] = (arr[:, 0].mean(), arr[:, 0].std(),
                                     arr[:, 1].mean(), len(arr))

    lines = ['# Dispatching rules across batch sizes (PPVC, lag-aware env)', '',
             'Mean makespan in hours over the test set, population std in '
             'parentheses. Every schedule was re-validated against the true '
             'lags (eval_ppvc.py). Bold marks the lowest mean in each column. '
             'Rules: FIFO, MOR, SPT and MWKR are the classic rules; SPTL = '
             'shortest processing time plus post-op lag; LLF = longest post-op '
             'lag first, ties by SPT; MWKRL = most remaining work including '
             'the lags on the remaining route (mean processing time); LRPTL = '
             'longest remaining route including lags (min processing time).',
             '']
    n_row = ['instances'] + [str(next((stats[r, s][3] for r in RULES
                                       if (r, s) in stats), '-'))
                             for s in SETS]
    header = '| rule | ' + ' | '.join(f'{s}' for s in SETS) + ' |'
    lines += [header, '|---' * (len(SETS) + 1) + '|',
              '| ' + ' | '.join(n_row) + ' |']
    best = {}
    for s in SETS:
        cand = [(stats[r, s][0], r) for r in RULES if (r, s) in stats]
        if cand:
            best[s] = min(cand)[1]
    for r in RULES:
        cells = []
        for s in SETS:
            if (r, s) not in stats:
                cells.append('pending')
                continue
            m, sd = stats[r, s][:2]
            txt = f'{m:.1f} ({sd:.1f})'
            cells.append(f'**{txt}**' if best.get(s) == r else txt)
        lines.append(f'| {r} | ' + ' | '.join(cells) + ' |')

    lines += ['', '## Best rule per size', '']
    for s in SETS:
        if s not in best:
            lines.append(f'- {s}: pending')
            continue
        b = best[s]
        ranked = sorted((stats[r, s][0], r) for r in RULES if (r, s) in stats)
        runner = ranked[1] if len(ranked) > 1 else None
        spt = stats.get(('SPT', s))
        txt = f'- {s}: {b} ({stats[b, s][0]:.1f} h)'
        if runner:
            txt += f'; next {runner[1]} ({runner[0]:.1f} h)'
        if spt and b != 'SPT':
            txt += (f'; {100 * (spt[0] - stats[b, s][0]) / spt[0]:.1f}% '
                    f'below SPT ({spt[0]:.1f} h)')
        lines.append(txt)

    lines += ['', 'Paired LLF minus SPT per instance (negative = LLF shorter), '
              'two-sided Wilcoxon signed-rank test:', '']
    from scipy.stats import wilcoxon
    for s in SETS:
        if ('LLF', s) not in stats or ('SPT', s) not in stats:
            continue
        diff = (np.load(result_path('LLF', s))[:, 0]
                - np.load(result_path('SPT', s))[:, 0])
        p = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        lines.append(f'- {s}: mean {diff.mean():+.2f} h; LLF shorter on '
                     f'{(diff < 0).sum()}, tied on {(diff == 0).sum()}, '
                     f'longer on {(diff > 0).sum()} instances; p = {p:.3f}')

    lines += ['', '## Mean rollout time per instance (s)', '',
              'Indicative only: during the campaign other single-thread jobs '
              'ran on the same four cores, so these times are contended. The 10x25 '
              'classic-rule times come from the earlier stored run.', '',
              header, '|---' * (len(SETS) + 1) + '|']
    for r in RULES:
        lines.append(f'| {r} | ' + ' | '.join(
            f'{stats[r, s][2]:.2f}' if (r, s) in stats else 'pending'
            for s in SETS) + ' |')
    lines += ['', 'Source files: test_results/PPVC/<size>+ppvc-mixed/'
              'Result_<rule>+pdr_<size>+ppvc-mixed.npy; the 10x25 classic '
              'rules use Result_<rule>+10x25+ppvc-mixed+smoke_10x25+ppvc-mixed'
              '.npy. The 50x25 and 80x25 sets hold 50 instances each '
              '(seed0 100000 and 110000).', '']

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
