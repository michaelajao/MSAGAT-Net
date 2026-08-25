"""Diebold-Mariano significance tests from persisted per-timestep predictions.

For each (dataset, horizon) and each baseline, tests whether MSAGAT-Net's
squared-error loss differs from the baseline's, on the exact same test
timesteps:

    d_t = mean_over_nodes(e_MSAGAT,t^2) - mean_over_nodes(e_base,t^2)

The DM statistic uses a Newey-West (Bartlett kernel) long-run variance with
truncation lag h-1 (h-step-ahead errors are MA(h-1)), the
Harvey-Leybourne-Newbold small-sample correction, and Student-t critical
values with n-1 degrees of freedom. Within each (dataset, horizon) family the
p-values across baselines are Holm-Bonferroni adjusted.

Primary comparison: median-RMSE seed for each model (no ensemble advantage).
Supplementary: per-seed agreement counts.

Each baseline exists in two arms: the published level-space configuration and
the log-growth arm from the target-space generality experiment (`_lg` tag).
With `--arms best` (the default) every baseline is represented by whichever arm
scores better, so MSAGAT-Net is tested against the strongest available version
of each competitor. That is deliberately conservative -- selecting the
baseline's arm on test RMSE favours the baseline, never us.

Usage (from the repository root):
    python -m src.scripts.dm_test
    python -m src.scripts.dm_test --variant v1 --arms level
    python -m src.scripts.dm_test --selftest
"""

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRED_DIR = os.path.join(BASE_DIR, 'report', 'predictions')
OUT_CSV = os.path.join(BASE_DIR, 'report', 'results', 'dm_tests.csv')

BASELINES = ['cola_gnn', 'CNNRNN_Res', 'lstnet', 'dcrnn', 'epignn']

# Naive forecasting floors (src/scripts/naive_baselines.py). They are
# deterministic -- one seed, no training randomness -- so the median-seed
# selection below trivially returns that seed. Neither Cola-GNN nor EpiGNN
# reports a naive baseline; SpatialEpiBench (2026) shows that omission
# flatters the whole family, so they are tested here on the same footing.
FLOORS = ['persistence', 'seasonal_naive', 'ar4']

# MSAGAT-Net arms, distinguished by the log-token suffix the trainer writes.
# v2 (log-growth targets + 23 quantile heads) is the paper's model; v1 is the
# level-space target ablation.
VARIANTS = {'v1': 'with_adj', 'v2': 'with_adj.loggrowth.quant'}

# Baselines carry a '_lg' tag when trained on log-growth targets; the floors
# have no arms. Longest-first so 'seasonal_naive' cannot be shadowed.
BASE_RE = re.compile(
    r'^(?P<m>(?:'
    + '|'.join(sorted(BASELINES + FLOORS, key=len, reverse=True))
    + r')(?:_lg)?)\.(?P<ds>[^.]+)'
    r'\.w-20\.h-(?P<h>\d+)\.none\.seed-(?P<s>\d+)\.npz$')


def msagat_re(variant):
    return re.compile(
        r'^MSAGAT-Net\.(?P<ds>[^.]+)\.w-20\.h-(?P<h>\d+)\.none\.seed-(?P<s>\d+)\.'
        + re.escape(VARIANTS[variant]) + r'\.npz$')


def collect(variant):
    """-> {(dataset, horizon): {model: {seed: path}}}"""
    ours_re = msagat_re(variant)
    runs = {}
    for path in glob.glob(os.path.join(PRED_DIR, '*', '*.npz')):
        fname = os.path.basename(path)
        m = ours_re.match(fname)
        model = 'MSAGAT-Net' if m else None
        if m is None:
            m = BASE_RE.match(fname)
            model = m.group('m') if m else None
        if m is None:
            continue
        key = (m.group('ds'), int(m.group('h')))
        runs.setdefault(key, {}).setdefault(model, {})[int(m.group('s'))] = path
    return runs


def choose_arm(models, base, mode):
    """Pick which arm of a baseline family to test against.

    'level' uses the published level-space configuration; 'best' uses whichever
    of {level, log-growth} has the lower median-seed RMSE, so the comparison is
    against the strongest available version of that competitor.
    Returns (tag, {seed: path}) or (None, None) when the family is absent.
    """
    arms = [(t, models.get(t, {})) for t in (base, base + '_lg')]
    arms = [(t, sp) for t, sp in arms if sp]
    if not arms:
        return None, None
    if mode == 'level':
        return (base, models[base]) if models.get(base) else (None, None)
    return min(arms, key=lambda ts: rmse(ts[1][median_seed(ts[1])]))


def rmse(path):
    d = np.load(path)
    return float(np.sqrt(np.mean((d['y_true'] - d['y_pred']) ** 2)))


def median_seed(seed_paths):
    scored = sorted((rmse(p), s) for s, p in seed_paths.items())
    return scored[len(scored) // 2][1]


def dm_stat(e2_a, e2_b, horizon):
    """DM test on per-timestep mean-squared-error differential.

    Returns (dm, p) under H0: equal predictive accuracy, two-sided,
    HLN-corrected, t(n-1).
    """
    d = e2_a.mean(axis=1) - e2_b.mean(axis=1)
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    lag = max(horizon - 1, 0)
    gamma0 = np.mean(dc * dc)
    lrv = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * w * np.mean(dc[k:] * dc[:-k])
    lrv = max(lrv, 1e-300)
    dm = dbar / np.sqrt(lrv / n)
    h = horizon
    hln = np.sqrt(max(n + 1 - 2 * h + h * (h - 1) / n, 1e-12) / n)
    dm *= hln
    p = 2.0 * stats.t.sf(abs(dm), df=n - 1)
    return dm, p


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (preserving input order)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def load_pair(path_a, path_b):
    a, b = np.load(path_a), np.load(path_b)
    n = min(a['y_true'].shape[0], b['y_true'].shape[0])
    ya, pa = a['y_true'][-n:], a['y_pred'][-n:]
    yb, pb = b['y_true'][-n:], b['y_pred'][-n:]
    scale = max(np.abs(ya).max(), 1.0)
    aligned = bool(np.allclose(ya, yb, rtol=1e-3, atol=1e-3 * scale))
    return aligned, (ya - pa) ** 2, (yb - pb) ** 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--variant', choices=sorted(VARIANTS), default='v2',
                    help='which MSAGAT-Net arm to test (default v2, the '
                         'log-growth + quantile model the paper reports)')
    ap.add_argument('--arms', choices=['best', 'level'], default='best',
                    help="baseline arm: 'best' of level/log-growth per "
                         "baseline (conservative), or the published 'level'")
    ap.add_argument('--include-floors', dest='include_floors',
                    action='store_true',
                    help='also test against the naive floors (persistence, '
                         'seasonal-naive, AR(4)) as a SEPARATE Holm family, '
                         'so the trained-baseline comparison is unchanged')
    args = ap.parse_args()

    runs = collect(args.variant)

    if args.selftest:
        # Two seeds of the same architecture are NOT guaranteed equal in
        # expected loss, so individual seed-vs-seed significances are
        # legitimate. The calibration check is the *rate*: under approximate
        # exchangeability it should sit near alpha, far below the ~100%
        # detection rate against deliberately degraded predictions.
        seed_sig, seed_n, degr_sig, degr_n = 0, 0, 0, 0
        for (ds, h), models in sorted(runs.items()):
            seeds = models.get('MSAGAT-Net', {})
            if len(seeds) < 2:
                continue
            s = sorted(seeds)
            aligned, e2a, e2b = load_pair(seeds[s[0]], seeds[s[1]])
            dm, p = dm_stat(e2a, e2b, h)
            seed_n += 1
            seed_sig += int(p < 0.05)
            noise = np.load(seeds[s[0]])
            rng = np.random.default_rng(0)
            bad = (noise['y_true'] - (noise['y_pred']
                   + rng.normal(0, np.abs(noise['y_true']).mean(),
                                noise['y_pred'].shape))) ** 2
            dm2, p2 = dm_stat(e2a, bad, h)
            degr_n += 1
            degr_sig += int(p2 < 0.05)
            print(f'{ds} h={h}: seed-vs-seed p={p:.3f} | '
                  f'vs-degraded p={p2:.2e}')
        seed_rate = seed_sig / max(seed_n, 1)
        degr_rate = degr_sig / max(degr_n, 1)
        ok = seed_rate <= 0.25 and degr_rate >= 0.95
        print(f'seed-vs-seed significant: {seed_sig}/{seed_n} '
              f'({seed_rate:.0%}, want near alpha) | degraded detected: '
              f'{degr_sig}/{degr_n} ({degr_rate:.0%}, want ~100%)')
        print('SELFTEST', 'PASS' if ok else 'FAIL')
        return

    rows = []
    for (ds, h), models in sorted(runs.items()):
        ours = models.get('MSAGAT-Net', {})
        if not ours:
            continue
        our_seed = median_seed(ours)
        # Two Holm families, deliberately kept apart. The trained baselines
        # answer "is this better than competing methods?"; the naive floors
        # answer "is it better than no method at all?". Pooling them would
        # enlarge the correction on the primary comparison for an unrelated
        # question, and would silently move the published W/L/T counts.
        families = {'baselines': list(BASELINES)}
        if args.include_floors:
            families['floors'] = list(FLOORS)

        for family_name, members in families.items():
            family = []
            for base in members:
                tag, theirs = choose_arm(models, base, args.arms)
                if not theirs:
                    continue
                their_seed = median_seed(theirs)
                aligned, e2_ours, e2_base = load_pair(ours[our_seed],
                                                      theirs[their_seed])
                if not aligned:
                    print(f'WARN {ds} h={h} {tag}: y_true misaligned, skipped')
                    continue
                dm, p = dm_stat(e2_ours, e2_base, h)
                agree = 0
                for s, path in ours.items():
                    al2, ea, eb = load_pair(path, theirs[their_seed])
                    if al2:
                        dm_i, p_i = dm_stat(ea, eb, h)
                        agree += int(p_i < args.alpha and dm_i < 0)
                family.append({
                    'dataset': ds, 'horizon': h, 'baseline': base,
                    'family': family_name,
                    'baseline_arm': ('loggrowth' if tag.endswith('_lg')
                                     else 'level'),
                    'msagat_seed': our_seed, 'baseline_seed': their_seed,
                    'msagat_rmse': rmse(ours[our_seed]),
                    'baseline_rmse': rmse(theirs[their_seed]),
                    'dm': dm, 'p_raw': p, 'n_test': len(e2_ours),
                    'seeds_agree': f'{agree}/{len(ours)}',
                })
            if family:
                adj = holm([r['p_raw'] for r in family])
                for r, pa in zip(family, adj):
                    r['p_holm'] = pa
                    r['significant'] = bool(pa < args.alpha)
                    r['direction'] = ('MSAGAT better' if r['dm'] < 0
                                      else 'baseline better')
                rows.extend(family)

    if not rows:
        print('No comparable prediction pairs found yet.')
        return

    df = pd.DataFrame(rows)
    out_csv = OUT_CSV.replace('.csv', f'_{args.variant}_{args.arms}.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False,
                       float_format=lambda v: f'{v:.4g}'))
    wins = df[(df['dm'] < 0) & df['significant']]
    losses = df[(df['dm'] > 0) & df['significant']]
    print(f'\n{len(df)} comparisons: {len(wins)} significant wins, '
          f'{len(losses)} significant losses, '
          f'{len(df) - len(wins) - len(losses)} ties (Holm, alpha={args.alpha})')
    print(f'wrote {out_csv}')


if __name__ == '__main__':
    main()
