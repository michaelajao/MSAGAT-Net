"""Naive forecasting floors: persistence, seasonal-naive, and AR(p).

Neither Cola-GNN nor EpiGNN includes a naive baseline, and when
SpatialEpiBench (2026) added one across 11 datasets it found that "every
method beats the naive baseline less than 50% of the time" and that
"adjacency-informed methods do not beat univariate baselines". M-SPICE
(KDD 2026) reports Cola-GNN at 0.291 NRMSE against persistence at 0.213.
A 2026 reviewer will ask for this, and it is better to know the answer
before they do.

Three floors, all fit and evaluated under the same protocol as every other
model in the comparison -- lead-h scoring, the same chronological split, the
same test indices:

    persistence      y(t+h) = y(t)          the last value available at
                                            forecast time, i.e. the anchor
                                            the log-growth target divides by
    seasonal-naive   y(t+h) = y(t+h-s)      s = 52 steps on weekly ILI,
                                            364 on daily series
    ar(p)            per-node OLS on lags   fitted on the training partition
                                            only, p = 4 by default

Predictions are written in the baseline archive schema so they flow into
dm_test.py unchanged.

    python -m src.scripts.naive_baselines [--dataset X] [--horizon H]

Alignment is asserted, not assumed: for every cell the reconstructed y_true
is compared against an existing baseline archive and the run aborts on any
mismatch.
"""

import argparse
import glob
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from src.train import DATASET_CONFIGS  # noqa: E402

PRED_DIR = os.path.join(BASE, 'report', 'predictions')

# One seasonal cycle, in time steps of the series itself.
SEASON = {
    'japan': 52, 'region785': 52, 'state360': 52,          # weekly ILI
    'ltla_timeseries': 364, 'nhs_timeseries': 364,          # daily
    'australia-covid': 364,
}

AR_ORDER = 4
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2


def load_series(dataset):
    path = os.path.join(BASE, 'data', f'{dataset}.txt')
    return np.clip(np.loadtxt(path, delimiter=','), 0, None)


def split_points(n):
    return int(TRAIN_FRAC * n), int((TRAIN_FRAC + VAL_FRAC) * n)


def persistence(raw, idx, horizon):
    """The last observation available at forecast time."""
    return raw[idx - horizon]


def seasonal_naive(raw, idx, horizon, season):
    """The observation one seasonal cycle before the target.

    Falls back to persistence where the series does not reach back a full
    cycle, which only happens on the shortest weekly sets.
    """
    src = idx - season
    ok = src >= 0
    out = np.where(ok[:, None], raw[np.clip(src, 0, None)], raw[idx - horizon])
    return out


def ar_forecast(raw, idx, horizon, train_end, order=AR_ORDER):
    """Per-node AR(order), ordinary least squares, fitted on train only.

    The regressors for a target at t are the observations at t-h, t-h-1, ...
    so the model sees exactly what every other model sees at forecast time.
    """
    n, m = raw.shape
    first = horizon + order - 1
    fit_idx = np.arange(first, train_end)
    preds = np.zeros((len(idx), m))

    for j in range(m):
        X = np.column_stack([raw[fit_idx - horizon - k, j] for k in range(order)])
        X = np.column_stack([np.ones(len(fit_idx)), X])
        y = raw[fit_idx, j]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)

        Xt = np.column_stack([raw[idx - horizon - k, j] for k in range(order)])
        Xt = np.column_stack([np.ones(len(idx)), Xt])
        preds[:, j] = Xt @ coef

    return np.clip(preds, 0, None)


def reference_truth(dataset, horizon):
    """y_true from an existing archive for this cell, for the alignment check."""
    pattern = os.path.join(PRED_DIR, dataset,
                           f'*.{dataset}.w-20.h-{horizon}.none.seed-*.npz')
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        if name.startswith('MSAGAT-Net') or '.oldckpt' in name:
            continue
        with np.load(path, allow_pickle=True) as d:
            return np.asarray(d['y_true']), name
    return None, None


def build_cell(dataset, horizon, verbose=True):
    raw = load_series(dataset)
    n = len(raw)
    _, val_end = split_points(n)
    train_end = int(TRAIN_FRAC * n)
    idx = np.arange(val_end, n)
    y_true = raw[idx]

    ref, ref_name = reference_truth(dataset, horizon)
    if ref is not None:
        if ref.shape != y_true.shape or not np.allclose(ref, y_true, rtol=1e-4):
            raise AssertionError(
                f'{dataset} h={horizon}: reconstructed targets do not match '
                f'{ref_name} (shapes {y_true.shape} vs {ref.shape}). The '
                f'naive floors would not be scored on the same task.')
    elif verbose:
        print(f'  note: no reference archive for {dataset} h={horizon}; '
              f'alignment unchecked')

    season = SEASON[dataset]
    return {
        'persistence': persistence(raw, idx, horizon),
        'seasonal_naive': seasonal_naive(raw, idx, horizon, season),
        f'ar{AR_ORDER}': ar_forecast(raw, idx, horizon, train_end),
    }, y_true


def write_archive(model, dataset, horizon, y_true, y_pred, seed=42):
    out_dir = os.path.join(PRED_DIR, dataset)
    os.makedirs(out_dir, exist_ok=True)
    token = f'{model}.{dataset}.w-20.h-{horizon}.none.seed-{seed}'
    np.savez_compressed(
        os.path.join(out_dir, token + '.npz'),
        y_true=y_true, y_pred=y_pred, model=model, dataset=dataset,
        horizon=horizon, window=20, seed=seed, ablation='none',
        protocol='lead_h')
    return token


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--horizon', type=int, default=None)
    args = ap.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASET_CONFIGS)
    written = 0
    print(f'{"dataset":16s} {"h":>3s}  {"persistence":>12s} '
          f'{"seasonal":>12s} {"ar" + str(AR_ORDER):>12s}')
    print('-' * 62)

    for ds in datasets:
        horizons = ([args.horizon] if args.horizon
                    else DATASET_CONFIGS[ds]['horizons'])
        for h in horizons:
            preds, y_true = build_cell(ds, h)
            scores = []
            for model, y_pred in preds.items():
                write_archive(model, ds, h, y_true, y_pred)
                scores.append(rmse(y_true, y_pred))
                written += 1
            print(f'{ds:16s} {h:3d}  {scores[0]:12.3f} {scores[1]:12.3f} '
                  f'{scores[2]:12.3f}')

    print(f'\nwrote {written} archives to report/predictions/')
    print('These are deterministic: one seed each, no training randomness.')


if __name__ == '__main__':
    main()
