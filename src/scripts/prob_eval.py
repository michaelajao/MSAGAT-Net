"""Probabilistic evaluation: WIS and empirical coverage from persisted quantiles.

Implements the weighted interval score exactly as defined by Bracher et al.
(PLoS Comp Biol 2021) and used by the CDC FluSight / COVID-19 Forecast Hub:
23 quantiles = median + K=11 central intervals, weights w_k = alpha_k / 2,
w_0 = 1/2, with the dispersion / overprediction / underprediction decomposition.

Usage (from the repository root):
    python -m src.scripts.prob_eval
    python -m src.scripts.prob_eval --dataset nhs_timeseries
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from ..csvmerge import merge_rows

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED_DIR = os.path.join(BASE_DIR, 'report', 'predictions')
OUT_CSV = os.path.join(BASE_DIR, 'report', 'results', 'prob_metrics.csv')

TOKEN_RE = re.compile(
    r'^(?P<model>.+?)\.(?P<ds>.+)\.w-20\.h-(?P<h>\d+)\.(?P<abl>[a-z_]+)'
    r'\.seed-(?P<s>\d+)\.(?P<rest>.*)\.npz$')


def wis_components(y, q, levels):
    """WIS with decomposition.

    Args:
        y: [n, nodes] observations
        q: [n, nodes, Q] quantile forecasts (levels ascending)
        levels: [Q] quantile levels; must contain 0.5 and symmetric pairs.
    Returns:
        dict with wis, dispersion, overprediction, underprediction,
        and coverage at the 50% and 90% central intervals.
    """
    levels = np.asarray(levels)
    med_idx = int(np.argmin(np.abs(levels - 0.5)))
    m = q[..., med_idx]

    alphas, pairs = [], []
    for i, lv in enumerate(levels):
        if lv < 0.5 - 1e-9:
            j = int(np.argmin(np.abs(levels - (1.0 - lv))))
            alphas.append(2.0 * lv)
            pairs.append((i, j))
    K = len(alphas)

    total_disp = np.zeros_like(y, dtype=float)
    total_over = np.zeros_like(y, dtype=float)
    total_under = np.zeros_like(y, dtype=float)
    for alpha, (i, j) in zip(alphas, pairs):
        l, u = q[..., i], q[..., j]
        w = alpha / 2.0
        total_disp += w * (u - l)
        total_under += w * (2.0 / alpha) * np.maximum(l - y, 0.0)
        total_over += w * (2.0 / alpha) * np.maximum(y - u, 0.0)

    denom = K + 0.5
    abs_med = 0.5 * np.abs(y - m)
    wis = (abs_med + total_disp + total_over + total_under) / denom

    def coverage(central):
        a = (1.0 - central) / 2.0
        i = int(np.argmin(np.abs(levels - a)))
        j = int(np.argmin(np.abs(levels - (1.0 - a))))
        return float(((y >= q[..., i]) & (y <= q[..., j])).mean())

    return {
        'wis': float(wis.mean()),
        'wis_dispersion': float((total_disp / denom).mean()),
        'wis_overprediction': float((total_over / denom).mean()),
        'wis_underprediction': float((total_under / denom).mean()),
        'cov50': coverage(0.50),
        'cov90': coverage(0.90),
        'cov95': coverage(0.95),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--split', choices=['test', 'val'], default='test')
    ap.add_argument('--replace-all', dest='replace_all', action='store_true',
                    help='rewrite the whole CSV instead of merging; use only '
                         'for a deliberate full regeneration')
    args = ap.parse_args()

    rows = []
    pattern = os.path.join(PRED_DIR, args.dataset or '*', '*.npz')
    for path in sorted(glob.glob(pattern)):
        m = TOKEN_RE.match(os.path.basename(path))
        if m is None:
            continue
        d = np.load(path)
        qkey = 'y_pred_q' if args.split == 'test' else 'y_pred_q_val'
        ykey = 'y_true' if args.split == 'test' else 'y_true_val'
        if qkey not in d:
            continue
        res = wis_components(d[ykey], d[qkey], d['quantile_levels'])
        res.update(model=str(m.group('model')), dataset=m.group('ds'),
                   horizon=int(m.group('h')), seed=int(m.group('s')),
                   ablation=m.group('abl'), variant=m.group('rest'),
                   split=args.split, n_test=d[ykey].shape[0])
        rows.append(res)

    if not rows:
        print('No quantile predictions found.')
        return

    df = merge_rows(OUT_CSV, rows,
                    keys=['model', 'dataset', 'horizon', 'seed', 'variant',
                          'split'],
                    replace_all=args.replace_all)
    agg = (df.groupby(['dataset', 'horizon', 'variant'])
             .agg(wis=('wis', 'mean'), cov50=('cov50', 'mean'),
                  cov90=('cov90', 'mean'), n_seeds=('seed', 'nunique'))
             .reset_index())
    print(agg.to_string(index=False, float_format=lambda v: f'{v:.4g}'))
    print(f'\nwrote {OUT_CSV}')


if __name__ == '__main__':
    main()
