"""Grade MSAGAT-Net under the *old* pooled protocol, symmetrically.

Finding E1 is that a preliminary version of this comparison scored the
baselines on lead times h..2h-1 pooled while MSAGAT-Net was scored at lead h
only. The correction adopted everywhere else in this project fixes that by
pulling the baselines *down* to lead-h scoring and retraining them.

This script closes the same loop from the other side: it pushes MSAGAT-Net
*up* to the pooled protocol. Its stored lead-h prediction is replicated
across leads h..2h-1 and scored against pooled targets -- which is exactly
how LSTNet and CNNRNN-Res were graded, since both emitted a single
prediction and expanded it across every step they were scored on
(`.unsqueeze(1).expand(-1, self.h, -1)`).

Neither direction is the paper's headline; the corrected lead-h protocol is,
because it matches upstream Cola-GNN and the wider literature. This is the
appendix table that shows the conclusion does not depend on which direction
the correction is applied in.

    python -m src.scripts.pooled_symmetric

Writes `report/results/pooled_symmetric.csv`.

Two honest limitations, stated here and to be stated in the paper:

  * The pooled targets need h-1 observations beyond each scored index, so
    the last h-1 test samples are dropped. The comparison is therefore on a
    slightly shorter window than the lead-h one; the count is reported.
  * Replicating one prediction across h leads is what the two affected
    baselines did. It is not what a multi-step model would do, so this
    reproduces the old protocol's *handicap*, not a well-specified
    multi-step task.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

OUT = os.path.join(BASE, 'report', 'results', 'pooled_symmetric.csv')
PUBLISHED = os.path.join(BASE, 'doc', 'archive', 'paper_results_final.csv')

TRAIN_FRAC, VAL_FRAC = 0.6, 0.2
SEEDS = [42, 30, 45, 123, 1000]

# The cells the AIIM manuscript headlined.
CELLS = [('ltla_timeseries', 3), ('ltla_timeseries', 7), ('ltla_timeseries', 14),
         ('nhs_timeseries', 3), ('nhs_timeseries', 7), ('nhs_timeseries', 14)]


def series(dataset):
    return np.clip(np.loadtxt(os.path.join(BASE, 'data', f'{dataset}.txt'),
                              delimiter=','), 0, None)


def msagat_paths(dataset, horizon, suffix):
    out = {}
    for seed in SEEDS:
        pat = os.path.join(BASE, 'report', 'predictions', dataset,
                           f'MSAGAT-Net.{dataset}.w-20.h-{horizon}.none.'
                           f'seed-{seed}.{suffix}.npz')
        hits = [p for p in glob.glob(pat) if '.oldckpt' not in p]
        if hits:
            out[seed] = hits[0]
    return out


def score_cell(dataset, horizon, suffix):
    """RMSE at lead h, and under the pooled protocol, for the same runs."""
    raw = series(dataset)
    n = len(raw)
    val_end = int((TRAIN_FRAC + VAL_FRAC) * n)
    idx = np.arange(val_end, n)

    # Pooled scoring needs raw[i + j] for j = 0..h-1, so drop the tail.
    keep = idx + horizon - 1 < n
    idx_p = idx[keep]

    rows = []
    for seed, path in msagat_paths(dataset, horizon, suffix).items():
        with np.load(path, allow_pickle=True) as d:
            y_true, y_pred = np.asarray(d['y_true']), np.asarray(d['y_pred'])

        if not np.allclose(y_true, raw[idx], rtol=1e-4):
            raise AssertionError(
                f'{dataset} h={horizon} seed={seed}: stored targets do not '
                f'match raw[test]; refusing to score a misaligned pair.')

        lead_h = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        # The old protocol: one prediction, replicated across leads
        # h .. 2h-1, scored against raw[i], raw[i+1], ... raw[i+h-1].
        pooled_true = np.stack([raw[idx_p + j] for j in range(horizon)], axis=1)
        pooled_pred = np.repeat(y_pred[keep][:, None, :], horizon, axis=1)
        pooled = float(np.sqrt(np.mean((pooled_true - pooled_pred) ** 2)))

        rows.append({'dataset': dataset, 'horizon': horizon, 'seed': seed,
                     'rmse_lead_h': lead_h, 'rmse_pooled': pooled,
                     'n_lead_h': len(idx), 'n_pooled': len(idx_p)})
    return rows


# How the submitted manuscript named the models.
PUBLISHED_OURS = 'msagat'


def published_table():
    """The numbers as submitted, one row per (model, dataset, horizon).

    The file carries duplicate rows and literal 'TIMEOUT' entries from runs
    that never finished; the manuscript used the best completed run per
    cell, so that is what is reproduced here.
    """
    if not os.path.exists(PUBLISHED):
        return None
    df = pd.read_csv(PUBLISHED)
    df['rmse'] = pd.to_numeric(df.rmse, errors='coerce')
    df = df.dropna(subset=['rmse'])
    return (df.sort_values('rmse')
              .groupby(['model', 'dataset', 'horizon'], as_index=False)
              .first())


def main():
    all_rows = []
    for ds, h in CELLS:
        for label, suffix in [('v1', 'with_adj'),
                              ('v2', 'with_adj.loggrowth.quant')]:
            for r in score_cell(ds, h, suffix):
                r['arm'] = label
                all_rows.append(r)

    if not all_rows:
        print('no MSAGAT-Net predictions found for the headline cells')
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT, index=False)

    agg = (df.groupby(['dataset', 'horizon', 'arm'])
             .agg(lead_h=('rmse_lead_h', 'mean'),
                  lead_h_sd=('rmse_lead_h', 'std'),
                  pooled=('rmse_pooled', 'mean'),
                  pooled_sd=('rmse_pooled', 'std'),
                  seeds=('seed', 'nunique'),
                  n_lead_h=('n_lead_h', 'first'),
                  n_pooled=('n_pooled', 'first'))
             .reset_index())
    agg['pooled_penalty_pct'] = 100 * (agg.pooled - agg.lead_h) / agg.lead_h

    pd.set_option('display.width', 200)
    print('MSAGAT-Net scored both ways, mean over seeds (RMSE)\n')
    print(agg.to_string(index=False, float_format=lambda v: f'{v:.2f}'))

    pub = published_table()
    if pub is None:
        print('\npublished table not found; skipping the comparison')
        return

    print('\n\nAgainst the baselines exactly as submitted (both sides pooled):')
    print('v1 is the level-space model the manuscript describes, so it is the')
    print('like-for-like arm; v2 adds log-growth targets and quantile heads.\n')
    print(f"{'dataset':16s} {'h':>3s} {'arm':>3s} {'submitted':>10s} "
          f"{'ours pooled':>12s} {'best baseline':>14s} {'name':>11s} "
          f"{'as submitted':>13s} {'symmetric':>11s}")
    print('-' * 102)
    for _, r in agg.sort_values(['dataset', 'horizon', 'arm']).iterrows():
        cell = pub[(pub.dataset == r.dataset) & (pub.horizon == r.horizon)]
        b = cell[cell.model != PUBLISHED_OURS]
        mine = cell[cell.model == PUBLISHED_OURS]
        if b.empty:
            continue
        best = b.loc[b.rmse.idxmin()]
        submitted = float(mine.rmse.iloc[0]) if len(mine) else float('nan')
        claimed = 100 * (best.rmse - submitted) / best.rmse
        symmetric = 100 * (best.rmse - r.pooled) / best.rmse
        print(f'{r.dataset:16s} {int(r.horizon):3d} {r.arm:>3s} '
              f'{submitted:10.2f} {r.pooled:12.2f} {best.rmse:14.2f} '
              f'{best.model:>11s} {claimed:+12.1f}% {symmetric:+10.1f}%')

    print('\n"as submitted" compares a lead-h MSAGAT-Net against pooled')
    print('baselines -- the asymmetry finding E1 identifies. "symmetric"')
    print('scores both sides pooled. The submitted manuscript claimed a')
    print('23.5% RMSE reduction on LTLA and 22.2% on NHS.')
    print(f'\nwrote {os.path.relpath(OUT, BASE)}')


if __name__ == '__main__':
    main()
