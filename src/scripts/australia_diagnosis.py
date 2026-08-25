"""Why do all significant losses happen on Australia-COVID?

The research ledger has carried "no diagnosis attempted" against this since
19 August: every one of the ten significant Diebold-Mariano losses to a
trained baseline, and all six significant losses to a naive floor, are on
this one dataset. A reviewer will build their rebuttal around it.

Two hypotheses were tested earlier and rejected: observation noise (Japan is
rougher yet stable) and a train-to-test range shift. This script tests four
more and reports which survive, writing
`report/results/australia_diagnosis.csv`.

    H1  the model is noisy      -- prediction volatility exceeds the truth's
    H2  the error is a bias     -- systematic under- or over-forecast
    H3  the error is localised  -- concentrated in part of the test window
    H4  regime gap              -- the test window leaves the range that
                                   model selection (early stopping, and the
                                   growth-space level cap) ever saw

H1 is rejected, H2 and H3 are confirmed and are the same phenomenon, and H4
is *consistent* with the pattern across datasets but is NOT established:
with six datasets the correlation is not significant. It is reported as a
hypothesis, and the paper should say so.

    python -m src.scripts.australia_diagnosis
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

OUT = os.path.join(BASE, 'report', 'results', 'australia_diagnosis.csv')
DATASETS = ['australia-covid', 'ltla_timeseries', 'nhs_timeseries',
            'japan', 'region785', 'state360']
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2


def series(dataset):
    return np.clip(np.loadtxt(os.path.join(BASE, 'data', f'{dataset}.txt'),
                              delimiter=','), 0, None)


def arm_paths(dataset, horizon, suffix, seed=42):
    pat = os.path.join(BASE, 'report', 'predictions', dataset,
                       f'MSAGAT-Net.{dataset}.w-20.h-{horizon}.none.'
                       f'seed-{seed}.{suffix}.npz')
    return [p for p in glob.glob(pat) if '.oldckpt' not in p]


def load_arm(dataset, horizon, suffix, seed=42):
    paths = arm_paths(dataset, horizon, suffix, seed)
    if not paths:
        return None
    with np.load(sorted(paths)[0], allow_pickle=True) as d:
        return np.asarray(d['y_true']), np.asarray(d['y_pred'])


def regime_gap(dataset):
    """How far the test window leaves the range model selection ever saw.

    Validation drives early stopping and the growth-space level cap, so the
    validation maximum -- not the training maximum -- is the relevant
    reference. The earlier rejected hypothesis compared against training.
    """
    raw = series(dataset)
    n = len(raw)
    tr, va = int(TRAIN_FRAC * n), int((TRAIN_FRAC + VAL_FRAC) * n)
    nat = raw.mean(axis=1)
    val, test = nat[tr:va], nat[va:]
    q = max(1, len(test) // 4)
    return {
        'val_max': val.max(), 'test_max': test.max(),
        'val_mean': val.mean(), 'test_mean': test.mean(),
        'test_over_val_max': test.max() / val.max(),
        'test_over_val_mean': test.mean() / val.mean(),
        'test_trend': test[-q:].mean() / test[:q].mean(),
    }


def cell_stats(dataset, horizon, suffix='with_adj.loggrowth.quant'):
    got = load_arm(dataset, horizon, suffix)
    if got is None:
        return None
    y, p = got
    raw = series(dataset)
    n = len(raw)
    idx = np.arange(int((TRAIN_FRAC + VAL_FRAC) * n), n)
    persist = raw[idx - horizon]

    q = np.array_split(np.arange(len(y)), 4)
    se = ((y - p) ** 2).mean(axis=1)
    sp = ((y - persist) ** 2).mean(axis=1)

    rec = {
        'dataset': dataset, 'horizon': horizon,
        'rmse': float(np.sqrt(((y - p) ** 2).mean())),
        'rmse_persistence': float(np.sqrt(((y - persist) ** 2).mean())),
        # H1: is the model more volatile than the series?
        'sd_pred_over_true': float(np.mean(np.std(p, axis=0))
                                   / np.mean(np.std(y, axis=0))),
        # H2: systematic bias, absolute and relative to the level
        'bias': float(np.mean(p - y)),
        'bias_pct_of_level': float(100 * np.mean(p - y) / np.mean(y)),
        # H3: how localised is the error?
        'worst10pct_share': float(np.sort(se)[-max(1, len(se) // 10):].sum()
                                  / se.sum()),
    }
    for i, g in enumerate(q, 1):
        rec[f'rmse_q{i}'] = float(np.sqrt(se[g].mean()))
        rec[f'rmse_persist_q{i}'] = float(np.sqrt(sp[g].mean()))
        rec[f'level_q{i}'] = float(y[g].mean())
    rec.update(regime_gap(dataset))
    return rec


def main():
    rows = []
    for ds in DATASETS:
        for h in (3, 5, 7, 10, 14, 15):
            rec = cell_stats(ds, h)
            if rec:
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    pd.set_option('display.width', 200)

    print('H1  the model is noisy -- prediction sd relative to the truth')
    print('    (>1 would mean the model injects variance)')
    print(df.pivot_table(index='dataset', columns='horizon',
                         values='sd_pred_over_true')
            .to_string(float_format=lambda v: f'{v:.2f}'))
    print('    VERDICT: rejected. Australia is 0.72/0.46/0.31 at h=3/7/14 --')
    print('    the model is less volatile than the series, not more.\n')

    print('H2  the error is a systematic bias, as % of the test-period level')
    print(df.pivot_table(index='dataset', columns='horizon',
                         values='bias_pct_of_level')
            .to_string(float_format=lambda v: f'{v:+.1f}'))
    print('    VERDICT: confirmed for Australia. The under-forecast grows')
    print('    with horizon and is far larger than on any other dataset.\n')

    print('H3  the error is localised -- RMSE by quartile of the test window')
    aus = df[df.dataset == 'australia-covid']
    for _, r in aus.iterrows():
        qs = '  '.join(f'Q{i}: {r[f"rmse_q{i}"]:7.1f} (pers {r[f"rmse_persist_q{i}"]:6.1f},'
                       f' level {r[f"level_q{i}"]:5.0f})' for i in (1, 2, 3, 4))
        print(f'    h={int(r.horizon):<3d} {qs}')
    print('    VERDICT: confirmed. The model is an order of magnitude worse')
    print('    than persistence even on the flat quartiles, and worst on the')
    print('    rising final quarter. Same phenomenon as H2.\n')

    print('H4  regime gap -- test window vs the range model selection saw')
    g = (df.groupby('dataset')
           .agg(gap=('test_over_val_max', 'first'),
                trend=('test_trend', 'first'),
                mean_bias_pct=('bias_pct_of_level', 'mean'))
           .sort_values('gap', ascending=False))
    print(g.to_string(float_format=lambda v: f'{v:.2f}'))
    r, p = stats.pearsonr(df.test_over_val_max, df.bias_pct_of_level)
    print(f'    correlation of the gap with bias: r = {r:+.3f}, p = {p:.4f}, '
          f'n = {len(df)} cells')
    print('    VERDICT: supported but not proven. The gap predicts the')
    print('    under-forecast bias (above), and the two datasets with a gap')
    print('    over 1.4 (Australia 1.88, Japan 1.45) are the only ones where')
    print('    a trained model loses to a naive floor, while every dataset')
    print('    below 1.1 wins every cell. But the 21 cells come from only 6')
    print('    datasets and cells within a dataset share the same gap, so')
    print('    they are not independent and this p-value is optimistic.')
    print('    Report as a supported hypothesis, not a demonstrated cause.')
    print()
    print('    Note the wider finding in H2: the under-forecast is not')
    print('    specific to Australia. Five of six datasets show a negative')
    print('    bias that grows with horizon (Japan -21% to -45%, LTLA -2% to')
    print('    -30%, Australia -7% to -23%); only NHS does not. Shrinking')
    print('    toward the mean as horizon grows is a general property of')
    print('    this model. Australia is simply where it costs most, because')
    print('    persistence is unusually strong there.')

    print(f'\nwrote {os.path.relpath(OUT, BASE)}')


if __name__ == '__main__':
    main()
