"""Does the spatial prior earn its place, and at which horizons?

Finding E11 -- that structural priors help at long horizons and hurt at
short ones -- is described as the most transferable result either research
track produced, but its numbers have only ever existed inside local
variables in `paper_figures.py` and were rendered into a PNG without being
persisted. Prose in the paper quotes them; no artefact holds them.

This computes the attention side of E11 from the prediction archives:
the revived attention configuration (`nodecay,regpre`, which restores a
non-uniform learned graph bias -- see E19) against the frozen v2 model that
is byte-identical apart from that recipe, per dataset and horizon, over all
available seeds, on the test split.

A positive delta means the revived attention made the forecast worse.

    python -m src.scripts.horizon_threshold

Writes `report/results/horizon_threshold.csv`.

The renewal side of E11 -- the learned residual weight gamma rising from
about zero at h=3 to 0.64-0.75 at h=14 -- belongs to the generation-interval
paper and is computed by `src/scripts/renewal_paper.py`; the two papers each
present their own evidence and cite the other.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

INDEX = os.path.join(BASE, 'report', 'results', 'runs_index.csv')
OUT = os.path.join(BASE, 'report', 'results', 'horizon_threshold.csv')

REVIVED = 'nodecay-regpre'


def main():
    df = pd.read_csv(INDEX)
    df = df[(df.arm == 'current') & (df.family == 'MSAGAT-Net')
            & (df.ablation == 'none') & (df.sim_mat == 'default')
            & (df.renewal == False) & df.rmse_npz.notna()]      # noqa: E712

    # `attnfix` is a separate attention experiment that carries no attn_exp
    # token, so it would otherwise be swept into the frozen arm and duplicate
    # seeds in the pairing.
    df = df[~df.run_token.str.contains('.attnfix', regex=False)]

    frozen = df[(df.target_space == 'loggrowth') & (df.quantiles == True)  # noqa: E712
                & df.attn_exp.isna()]
    revived = df[df.attn_exp == REVIVED]

    rows = []
    for (ds, h), g in revived.groupby(['dataset', 'horizon']):
        f = frozen[(frozen.dataset == ds) & (frozen.horizon == h)]
        if f.empty:
            continue
        # Pair on seed so the comparison is within-seed, not between means.
        merged = (g[['seed', 'rmse_npz']]
                  .merge(f[['seed', 'rmse_npz']], on='seed',
                         suffixes=('_revived', '_frozen')))
        if merged.empty:
            continue
        d = merged.rmse_npz_revived - merged.rmse_npz_frozen
        pct = 100 * d / merged.rmse_npz_frozen
        rec = {
            'dataset': ds, 'horizon': h, 'n_seeds': len(merged),
            'rmse_frozen': merged.rmse_npz_frozen.mean(),
            'rmse_revived': merged.rmse_npz_revived.mean(),
            # The mean is outlier-driven on the noisier cells, so the median
            # is reported alongside it and used for the pooled summary.
            'delta_pct_mean': pct.mean(), 'delta_pct_median': pct.median(),
            'delta_pct_sd': pct.std(ddof=1),
            'seeds_improved': int((d < 0).sum()),
        }
        if len(merged) >= 3:
            t, p = stats.wilcoxon(merged.rmse_npz_revived,
                                  merged.rmse_npz_frozen)
            rec['wilcoxon_p'] = p
        else:
            rec['wilcoxon_p'] = np.nan
        rows.append(rec)

    if not rows:
        print('no paired revived/frozen runs found')
        return

    out = pd.DataFrame(rows).sort_values(['dataset', 'horizon'])
    out.to_csv(OUT, index=False)

    pd.set_option('display.width', 200)
    print('Revived attention against the frozen model, paired by seed, on test.')
    print('delta > 0 means restoring a non-uniform attention made it worse.\n')
    print(out.to_string(index=False, float_format=lambda v: f'{v:.3f}'))

    daily = out[out.dataset.isin(['ltla_timeseries', 'nhs_timeseries',
                                  'australia-covid'])]
    if not daily.empty:
        print('\nDaily series only, pooled by horizon (the E11 claim):')
        by_h = (daily.groupby('horizon')
                     .agg(cells=('dataset', 'size'),
                          delta_median=('delta_pct_median', 'median'),
                          delta_mean=('delta_pct_mean', 'mean'),
                          improved=('seeds_improved', 'sum'),
                          seeds=('n_seeds', 'sum')))
        by_h['improved_frac'] = by_h.improved / by_h.seeds
        print(by_h.to_string(float_format=lambda v: f'{v:.2f}'))
        print('\nWeekly ILI series:')
        weekly = out[~out.dataset.isin(daily.dataset.unique())]
        if not weekly.empty:
            print(weekly.groupby('horizon')
                        .agg(cells=('dataset', 'size'),
                             delta_pct=('delta_pct_mean', 'mean'))
                        .to_string(float_format=lambda v: f'{v:.2f}'))

    print(f'\nwrote {os.path.relpath(OUT, BASE)}')


if __name__ == '__main__':
    main()
