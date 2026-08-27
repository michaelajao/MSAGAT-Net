"""Adjacency-threshold sensitivity, paired within seed.

Answers reviewer point #3. The 150 km Haversine threshold was never
justified or tuned, and until now the sweep had never been run: no artefact
on disk carried `sim_mat != 'default'` (ledger E14).

It matters more than a hyperparameter usually would. E19 shows the learned
graph bias underflows to zero and the content term is negligible, leaving
the static adjacency as the only attention logit with any spread -- so if
anything in the spatial pathway depends on graph structure, the threshold
*is* the spatial model. E20 then shows the attention softmax can be deleted
on LTLA with no measurable effect, which predicts the threshold should not
matter either. This tests that.

The shipped `{ltla,nhs}-adj.txt` are the 150 km matrices
(`build_adjacency.py` reproduces NHS exactly and LTLA at Jaccard 0.9993), so
the default arm is the 150 km arm and 100/200/250 km are compared against it,
paired within seed.

    python -m src.scripts.sensitivity_analysis

Writes `report/results/sensitivity_v2.csv`.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

INDEX = os.path.join(BASE, 'report', 'results', 'runs_index.csv')
OUT = os.path.join(BASE, 'report', 'results', 'sensitivity_v2.csv')

ADJ = {'ltla_timeseries': 'ltla-adj', 'nhs_timeseries': 'nhs-adj'}
THRESHOLDS = [100, 200, 250]
DEFAULT_KM = 150


def graph_density(name):
    path = os.path.join(BASE, 'data', f'{name}.txt')
    if not os.path.exists(path):
        return np.nan
    a = np.loadtxt(path, delimiter=',')
    return float(a.mean())


def load():
    df = pd.read_csv(INDEX)
    df = df[(df.arm == 'current') & (df.family == 'MSAGAT-Net')
            & (df.target_space == 'loggrowth') & (df.quantiles == True)   # noqa: E712
            & df.attn_exp.isna() & (df.renewal == False)                  # noqa: E712
            & (df.ablation == 'none') & df.rmse_npz.notna()]
    if 'attn_fix' in df.columns:
        df = df[df.attn_fix != True]                                      # noqa: E712
    return df


def main():
    df = load()
    rows = []
    for ds, adj in ADJ.items():
        base = df[(df.dataset == ds) & (df.sim_mat == 'default')]
        for thr in THRESHOLDS:
            arm = df[(df.dataset == ds) & (df.sim_mat == f'{adj}-{thr}')]
            for h in sorted(arm.horizon.unique()):
                b = base[base.horizon == h].set_index('seed').rmse_npz
                a = arm[arm.horizon == h].set_index('seed').rmse_npz
                shared = sorted(set(b.index) & set(a.index))
                if not shared:
                    continue
                bv = b.loc[shared].to_numpy()
                av = a.loc[shared].to_numpy()
                pct = 100 * (av - bv) / bv
                rows.append({
                    'dataset': ds, 'horizon': h, 'threshold_km': thr,
                    'n_seeds': len(shared),
                    'density': graph_density(f'{adj}-{thr}'),
                    'density_default': graph_density(adj),
                    'rmse_150km': bv.mean(), 'rmse_thr': av.mean(),
                    'delta_pct_mean': pct.mean(),
                    'delta_pct_median': float(np.median(pct)),
                    'delta_pct_sd': pct.std(ddof=1) if len(pct) > 1 else np.nan,
                    'wilcoxon_p': (stats.wilcoxon(bv, av).pvalue
                                   if len(shared) >= 5 else np.nan),
                })

    if not rows:
        print('no paired sensitivity runs found')
        return

    out = pd.DataFrame(rows).sort_values(['dataset', 'horizon', 'threshold_km'])
    out.to_csv(OUT, index=False)
    pd.set_option('display.width', 200)

    print('Graph density by threshold (the 150 km default is the comparison arm):')
    dens = out.groupby(['dataset', 'threshold_km']).density.first().unstack()
    dens['150 (default)'] = out.groupby('dataset').density_default.first()
    print(dens.to_string(float_format=lambda v: f'{v:.3f}'))

    print('\nRMSE change against the 150 km default, paired within seed:\n')
    print(out[['dataset', 'horizon', 'threshold_km', 'rmse_150km', 'rmse_thr',
               'delta_pct_mean', 'delta_pct_sd', 'wilcoxon_p']]
          .to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    print('\nPooled by threshold:')
    print(out.groupby(['dataset', 'threshold_km'])
             .agg(delta_mean=('delta_pct_mean', 'mean'),
                  delta_median=('delta_pct_median', 'median'),
                  worst=('delta_pct_mean', 'max'),
                  min_p=('wilcoxon_p', 'min'))
             .to_string(float_format=lambda v: f'{v:.3f}'))

    sig = out[out.wilcoxon_p < 0.05]
    print(f'\ncells where the threshold makes a significant difference: '
          f'{len(sig)} of {len(out)}')
    if len(sig):
        print(sig[['dataset', 'horizon', 'threshold_km', 'delta_pct_mean',
                   'wilcoxon_p']].to_string(index=False,
                                            float_format=lambda v: f'{v:.3f}'))
    print(f'\nlargest absolute mean change anywhere: '
          f'{out.delta_pct_mean.abs().max():.2f}%')
    print(f'wrote {os.path.relpath(OUT, BASE)}')


if __name__ == '__main__':
    main()
