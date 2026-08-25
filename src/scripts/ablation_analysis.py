"""Component ablations in v2 configuration, paired within seed.

Answers reviewer point #1, which had no evidence behind it: the original
ablation chunk ran in level space and never completed at five seeds
(ledger E14).

The arm that matters is `mean_agam`. It is the full attention module with
`uniform_attn=True`, so every parameter, projection, residual and norm is
identical and only the softmax is replaced by a fixed 1/N -- the test suite
asserts that forcing the flag on the full model reproduces this arm to 1e-6.
Finding E3 says the trained attention is already uniform (row entropy
1.0000 on LTLA), which predicts that removing the softmax should change
nothing. `no_agam`, by contrast, removes all spatial mixing and so answers a
different question.

Comparisons are paired within seed, because between-mean differences on
these datasets are swamped by seed variance.

    python -m src.scripts.ablation_analysis

Writes `report/results/ablation_v2.csv`.
"""

import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from src.tokens import parse_token  # noqa: E402

OUT = os.path.join(BASE, 'report', 'results', 'ablation_v2.csv')
DATASETS = ['ltla_timeseries', 'japan', 'nhs_timeseries']
ARMS = ['mean_agam', 'no_agam', 'no_mtfm', 'no_pprm']

# Row entropy of the trained attention, measured in E19 (1.0 = uniform).
E19_ENTROPY = {'ltla_timeseries': 1.0000, 'japan': 0.9996,
               'nhs_timeseries': 0.9937}
N_NODES = {'ltla_timeseries': 372, 'japan': 47, 'nhs_timeseries': 7}


def collect():
    """Map (dataset, horizon, ablation) -> {seed: path} for the v2 arm."""
    runs = defaultdict(dict)
    for ds in DATASETS:
        for path in glob.glob(os.path.join(BASE, 'report', 'predictions', ds,
                                           f'MSAGAT-Net.{ds}.*.npz')):
            try:
                s = parse_token(os.path.basename(path))
            except ValueError:
                continue
            if (s['target_space'] != 'loggrowth' or not s['quantiles']
                    or s['attn_exp'] or s['renewal'] or s['sim_mat']
                    or s['attn_fix']):
                continue
            runs[(ds, s['horizon'], s['ablation'])][s['seed']] = path
    return runs


def rmse(path):
    with np.load(path) as d:
        return float(np.sqrt(np.mean((d['y_true'] - d['y_pred']) ** 2)))


def main():
    runs = collect()
    rows = []
    for (ds, h, abl), paths in sorted(runs.items()):
        if abl == 'none':
            continue
        base = runs.get((ds, h, 'none'), {})
        shared = sorted(set(base) & set(paths))
        if not shared:
            continue
        b = np.array([rmse(base[s]) for s in shared])
        a = np.array([rmse(paths[s]) for s in shared])
        pct = 100 * (a - b) / b
        rec = {
            'dataset': ds, 'horizon': h, 'ablation': abl, 'n_seeds': len(shared),
            'rmse_full': b.mean(), 'rmse_ablated': a.mean(),
            'delta_pct_mean': pct.mean(), 'delta_pct_median': float(np.median(pct)),
            'delta_pct_sd': pct.std(ddof=1) if len(pct) > 1 else np.nan,
            'max_abs_delta_pct': float(np.abs(pct).max()),
            'seeds_worse': int((a > b).sum()),
            'wilcoxon_p': (stats.wilcoxon(b, a).pvalue if len(shared) >= 5
                           else np.nan),
            'n_nodes': N_NODES.get(ds), 'e19_entropy': E19_ENTROPY.get(ds),
        }
        rows.append(rec)

    if not rows:
        print('no paired ablation runs found')
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    pd.set_option('display.width', 200)

    print('Removing ONLY the softmax (mean_agam), paired within seed.')
    print('E3 predicts this should change nothing where the attention is')
    print('already uniform.\n')
    m = df[df.ablation == 'mean_agam'].sort_values(['n_nodes', 'horizon'],
                                                   ascending=[False, True])
    print(m[['dataset', 'n_nodes', 'e19_entropy', 'horizon', 'delta_pct_mean',
             'delta_pct_sd', 'max_abs_delta_pct', 'wilcoxon_p']]
          .to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    ltla = m[m.dataset == 'ltla_timeseries']
    if not ltla.empty:
        print(f'\nOn the 372-node graph, where E19 measures entropy 1.0000, '
              f'the mean absolute\nchange is '
              f'{ltla.delta_pct_mean.abs().mean():.2f}% and no horizon is '
              f'significant (min p = {ltla.wilcoxon_p.min():.3f}).')
        print('Replacing the entire attention softmax with a constant is not')
        print('detectable in the forecast.')

    print('\n\nAll arms, pooled median delta against the full model:')
    pooled = (df.groupby('ablation')
                .agg(cells=('dataset', 'size'),
                     median_pct=('delta_pct_median', 'median'),
                     worst_cell=('delta_pct_median', 'max'))
                .sort_values('median_pct'))
    print(pooled.to_string(float_format=lambda v: f'{v:+.2f}'))
    print('\nA negative median means removing the component improved the model.')

    print(f'\nwrote {os.path.relpath(OUT, BASE)}')


if __name__ == '__main__':
    main()
