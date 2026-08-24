"""Attention-coupled adaptive conformal calibration (ACCC).

Calibrates the raw quantile forecasts of MSAGAT-Net v2 so that every central
interval attains its nominal coverage per region, by combining:

1.  CQR-style nonconformity scores per central interval
    (Romano et al. 2019): E = max(l - y, y - u).
2.  Cross-region calibration sharing: region i's score pool is the union of
    all regions' scores, weighted by a transfer kernel
    W~ = lam * I + (1 - lam) * row-normalised weights. The kernel is either
    the forecaster's own learned attention matrix (ours), binary adjacency
    (Jiang et al. 2024 style), uniform pooling, or identity (no sharing).
    Validity under weighted pooling follows Barber et al. (2023).
3.  Online coverage control: an ACI update (Gibbs & Candes 2021) per region
    and interval adapts the working miscoverage level on the test stream;
    scores observed during testing enter the pool with a rolling window.

Outputs before/after WIS and coverage per method for the ablation table.

Usage (from the repository root):
    python -m src.scripts.conformal --dataset nhs_timeseries --horizon 3
    python -m src.scripts.conformal                # everything available
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from ..csvmerge import merge_rows
from .prob_eval import wis_components

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED_DIR = os.path.join(BASE_DIR, 'report', 'predictions')
ATTN_DIR = os.path.join(BASE_DIR, 'report', 'attention')
OUT_CSV = os.path.join(BASE_DIR, 'report', 'results', 'conformal_metrics.csv')

TOKEN_RE = re.compile(
    r'^MSAGAT-Net\.(?P<ds>.+)\.w-20\.h-(?P<h>\d+)\.none'
    r'\.seed-(?P<s>\d+)\.with_adj\.loggrowth\.quant\.npz$')

TOP_K = 16          # neighbours contributing to a region's pool (large graphs)
WINDOW = 200        # rolling window of most recent scores per region
GAMMA = 0.01        # ACI learning rate
LAM = 0.5           # self-weight in the transfer kernel


def build_kernel(W, method, adj=None):
    """Row-stochastic calibration-transfer kernel [N, N]."""
    n = W.shape[0] if W is not None else adj.shape[0]
    if method == 'identity':
        return np.eye(n)
    if method == 'uniform':
        K = np.ones((n, n)) / n
    elif method == 'adjacency':
        A = (adj > 0).astype(float)
        np.fill_diagonal(A, 1.0)
        K = A / A.sum(1, keepdims=True)
    elif method == 'attention':
        K = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    else:
        raise ValueError(method)
    K = LAM * np.eye(n) + (1.0 - LAM) * K
    # sparsify to top-k per row for tractability on large graphs
    if n > TOP_K:
        idx = np.argsort(-K, axis=1)[:, TOP_K:]
        for i in range(n):
            K[i, idx[i]] = 0.0
        K = K / K.sum(1, keepdims=True)
    return K


def weighted_quantile(scores, weights, q):
    """Smallest score s with cumulative normalised weight >= q."""
    order = np.argsort(scores)
    s, w = scores[order], weights[order]
    cw = np.cumsum(w)
    total = cw[-1] + 1e-12
    k = np.searchsorted(cw / total, min(q, 1.0))
    return s[min(k, len(s) - 1)]


def calibrate_run(npz_path, W, kernel_method, adj):
    d = np.load(npz_path)
    levels = d['quantile_levels']
    y_val, q_val = d['y_true_val'], d['y_pred_q_val']
    y_test, q_test = d['y_true'], d['y_pred_q']
    n_test, n_reg, n_q = q_test.shape

    lower_ids = [i for i, lv in enumerate(levels) if lv < 0.5 - 1e-9]
    pair_ids = [(i, int(np.argmin(np.abs(levels - (1 - levels[i])))))
                for i in lower_ids]
    n_int = len(pair_ids)
    alphas = np.array([2 * levels[i] for i in lower_ids])

    K = build_kernel(W, kernel_method, adj)

    # score pools: list per (region, interval) of recent nonconformity scores
    pools = [[list() for _ in range(n_int)] for _ in range(n_reg)]
    for t in range(y_val.shape[0]):
        for k, (i, j) in enumerate(pair_ids):
            e = np.maximum(q_val[t, :, i] - y_val[t],
                           y_val[t] - q_val[t, :, j])
            for r in range(n_reg):
                pools[r][k].append(e[r])

    alpha_eff = np.tile(alphas, (n_reg, 1))          # working miscoverage
    q_adj = q_test.copy()

    neigh = [np.nonzero(K[r] > 0)[0] for r in range(n_reg)]
    for t in range(n_test):
        for k, (i, j) in enumerate(pair_ids):
            for r in range(n_reg):
                ns = neigh[r]
                sc = np.concatenate([np.asarray(pools[nb][k][-WINDOW:])
                                     for nb in ns])
                wt = np.concatenate([np.full(len(pools[nb][k][-WINDOW:]),
                                             K[r, nb]) for nb in ns])
                a = float(np.clip(alpha_eff[r, k], 0.005, 0.995))
                eps = weighted_quantile(sc, wt, 1.0 - a)
                q_adj[t, r, i] = q_test[t, r, i] - eps
                q_adj[t, r, j] = q_test[t, r, j] + eps
        # enforce monotonicity across the full quantile vector
        q_adj[t] = np.sort(q_adj[t], axis=-1)
        # observe y_t: ACI update + append scores
        for k, (i, j) in enumerate(pair_ids):
            covered = ((y_test[t] >= q_adj[t, :, i]) &
                       (y_test[t] <= q_adj[t, :, j]))
            err = (~covered).astype(float)
            alpha_eff[:, k] = alpha_eff[:, k] + GAMMA * (alphas[k] - err)
            e = np.maximum(q_test[t, :, i] - y_test[t],
                           y_test[t] - q_test[t, :, j])
            for r in range(n_reg):
                pools[r][k].append(e[r])

    return y_test, q_test, q_adj, levels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default=None)
    ap.add_argument('--horizon', type=int, default=None)
    ap.add_argument('--methods', nargs='+',
                    default=['identity', 'adjacency', 'uniform', 'attention'])
    ap.add_argument('--replace-all', dest='replace_all', action='store_true',
                    help='rewrite the whole CSV instead of merging; use only '
                         'for a deliberate full regeneration')
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(PRED_DIR, '*', '*.loggrowth.quant.npz'))):
        m = TOKEN_RE.match(os.path.basename(path))
        if m is None:
            continue
        ds, h, seed = m.group('ds'), int(m.group('h')), int(m.group('s'))
        if args.dataset and ds != args.dataset:
            continue
        if args.horizon and h != args.horizon:
            continue

        token = os.path.basename(path)[:-4]
        attn_path = os.path.join(ATTN_DIR, token + '.npy')
        W = np.load(attn_path) if os.path.exists(attn_path) else None

        from ..train import DATASET_CONFIGS
        adj = np.loadtxt(os.path.join(BASE_DIR, 'data',
                                      DATASET_CONFIGS[ds]['sim_mat'] + '.txt'),
                         delimiter=',')

        d = np.load(path)
        raw = wis_components(d['y_true'], d['y_pred_q'], d['quantile_levels'])
        rows.append({'dataset': ds, 'horizon': h, 'seed': seed,
                     'method': 'raw', **raw})
        print(f"{ds} h={h} seed={seed} raw: wis={raw['wis']:.4g} "
              f"cov50={raw['cov50']:.3f} cov90={raw['cov90']:.3f}")

        for method in args.methods:
            if method == 'attention' and W is None:
                print(f'  {method}: no attention matrix at {attn_path}, skipped')
                continue
            y, q0, q1, levels = calibrate_run(path, W, method, adj)
            res = wis_components(y, q1, levels)
            rows.append({'dataset': ds, 'horizon': h, 'seed': seed,
                         'method': method, **res})
            print(f"  {method:>9s}: wis={res['wis']:.4g} "
                  f"cov50={res['cov50']:.3f} cov90={res['cov90']:.3f}")

    if rows:
        # Merge, never replace: a filtered run must not delete the rows it
        # did not recompute (see src/csvmerge.py).
        merge_rows(OUT_CSV, rows,
                   keys=['dataset', 'horizon', 'seed', 'method'],
                   replace_all=args.replace_all)


if __name__ == '__main__':
    main()
