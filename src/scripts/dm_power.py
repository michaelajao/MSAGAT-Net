"""Minimum detectable effect for each Diebold-Mariano comparison.

Most of the DM grid is ties: 75 of 104 against trained baselines, and Japan
and US-States are ties in every single cell. A tie means the test did not
reject equal predictive accuracy -- it does not mean the models are equally
good, and reporting it as "no difference found" invites exactly the wrong
reading. With 70 test points a 20% RMSE gap can sit comfortably inside the
noise, which is what happens on Japan.

This computes, for every comparison already in the grid, the smallest RMSE
reduction the test could have detected at 80% power. Ties can then be
reported as "correctly underpowered, MDE = x%" rather than as evidence of
equivalence.

Method. The DM statistic is dbar / sqrt(LRV/n), scaled by the
Harvey-Leybourne-Newbold factor, where LRV is the Newey-West long-run
variance at truncation lag h-1 and d_t is the per-timestep mean squared-error
differential. Under a fixed alternative the same variance applies, so

    |dbar| detectable  =  (t_{alpha/2,n-1} + t_{beta,n-1}) * sqrt(LRV/n) / HLN

and since d = MSE_ours - MSE_base, a detectable MSE gap of D corresponds to a
relative RMSE reduction of 1 - sqrt(1 - D / MSE_base).

The variance is taken from the observed loss differential of the actual pair,
so the MDE is conditional on that realised series -- it is a diagnostic of
what this test on this data could resolve, not a design calculation for a
future experiment.

alpha is Holm-adjusted for the family the comparison belongs to. The most
conservative member of a family of m faces alpha/m, so that is what is used;
this makes the reported MDE an upper bound on what the family could resolve.

    python -m src.scripts.dm_power [--variant v2] [--arms best] [--power 0.8]

Writes `report/results/dm_power_{variant}_{arms}.csv`.

Nothing in `dm_test.py` is modified or re-implemented: `dm_stat` is imported
and its variance recomputed by the same code path.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from src.scripts.dm_test import (BASELINES, FLOORS, VARIANTS,  # noqa: E402
                                 choose_arm, collect, load_pair, median_seed)

RESULTS = os.path.join(BASE, 'report', 'results')


def long_run_variance(d, horizon):
    """Newey-West long-run variance of the loss differential.

    Identical to the computation inside dm_test.dm_stat; kept here only
    because that function returns the statistic rather than its parts.
    """
    n = len(d)
    dc = d - d.mean()
    lag = max(horizon - 1, 0)
    lrv = np.mean(dc * dc)
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        lrv += 2.0 * w * np.mean(dc[k:] * dc[:-k])
    return max(lrv, 1e-300)


def mde(e2_ours, e2_base, horizon, alpha, power):
    """Smallest detectable MSE gap, and the RMSE reduction it implies."""
    d = e2_ours.mean(axis=1) - e2_base.mean(axis=1)
    n = len(d)
    lrv = long_run_variance(d, horizon)
    hln = np.sqrt(max(n + 1 - 2 * horizon + horizon * (horizon - 1) / n,
                      1e-12) / n)

    se = np.sqrt(lrv / n) / hln
    crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    beta = stats.t.ppf(power, df=n - 1)
    d_min = (crit + beta) * se

    mse_base = float(e2_base.mean())
    ratio = 1 - d_min / mse_base
    rmse_pct = 100 * (1 - np.sqrt(ratio)) if ratio > 0 else np.nan
    return {
        'n_test': n, 'mse_base': mse_base, 'lrv': lrv,
        'mde_mse': d_min, 'mde_rmse_pct': rmse_pct,
        'observed_rmse_pct': 100 * (1 - np.sqrt(max(
            float(e2_ours.mean()) / mse_base, 0.0))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', choices=sorted(VARIANTS), default='v2')
    ap.add_argument('--arms', choices=['best', 'level'], default='best')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--power', type=float, default=0.8)
    ap.add_argument('--include-floors', dest='include_floors',
                    action='store_true')
    args = ap.parse_args()

    runs = collect(args.variant)
    rows = []

    for (ds, h), models in sorted(runs.items()):
        ours = models.get('MSAGAT-Net', {})
        if not ours:
            continue
        our_seed = median_seed(ours)
        families = {'baselines': list(BASELINES)}
        if args.include_floors:
            families['floors'] = list(FLOORS)

        for fam, members in families.items():
            present = [b for b in members if choose_arm(models, b, args.arms)[1]]
            m = max(len(present), 1)
            for base in present:
                tag, theirs = choose_arm(models, base, args.arms)
                their_seed = median_seed(theirs)
                aligned, e2_ours, e2_base = load_pair(ours[our_seed],
                                                      theirs[their_seed])
                if not aligned:
                    continue
                # Holm's most conservative step within the family.
                rec = mde(e2_ours, e2_base, h, args.alpha / m, args.power)
                rec.update({'dataset': ds, 'horizon': h, 'baseline': base,
                            'family': fam, 'family_size': m,
                            'alpha_holm': args.alpha / m})
                rows.append(rec)

    if not rows:
        print('no comparable pairs found')
        return

    df = pd.DataFrame(rows)
    out = os.path.join(RESULTS, f'dm_power_{args.variant}_{args.arms}.csv')
    df.to_csv(out, index=False)

    pd.set_option('display.width', 200)
    piv = (df[df.family == 'baselines']
           .pivot_table(index=['dataset', 'horizon'],
                        values=['n_test', 'mde_rmse_pct'], aggfunc='median'))
    piv = piv[['n_test', 'mde_rmse_pct']].sort_values('mde_rmse_pct',
                                                      ascending=False)
    print(f'Smallest RMSE reduction detectable at {args.power:.0%} power, '
          f'median over the baselines in each cell\n')
    print(piv.to_string(float_format=lambda v: f'{v:.1f}'))

    print('\nRead this against the observed gaps. A cell whose MDE exceeds')
    print('the gap the model actually achieves is underpowered by')
    print('construction, and its tie carries no information about which')
    print('model is better.')

    obs = df[df.family == 'baselines'].copy()
    obs['underpowered'] = obs.mde_rmse_pct > obs.observed_rmse_pct.abs()
    by_ds = obs.groupby('dataset').agg(
        comparisons=('baseline', 'size'),
        median_mde=('mde_rmse_pct', 'median'),
        underpowered=('underpowered', 'sum'))
    print('\n' + by_ds.to_string(float_format=lambda v: f'{v:.1f}'))

    q = obs.mde_rmse_pct.dropna()
    print(f'\nAcross {len(q)} comparisons: median MDE {q.median():.1f}%, '
          f'best {q.min():.1f}%, worst {q.max():.1f}%.')
    print('Effect sizes this literature reports, against that threshold:')
    for name, eff in [('EpiGNN 2022 headline', 5.6),
                      ('HeatGNN vs Cola-GNN, flu', 4.1),
                      ('this manuscript, LTLA claim', 23.5),
                      ('this manuscript, NHS claim', 22.2)]:
        n = int((q < eff).sum())
        print(f'  {name:28s} {eff:5.1f}%  resolvable in {n:2d}/{len(q)} '
              f'({n / len(q):.0%})')

    print("""
Caveats, all of which belong in the paper alongside the number:

  * The MDE is conditional on the realised loss-differential variance of
    each pair, so it describes what this test could resolve on this data.
    It is a diagnostic, not a sample-size calculation for a new study.
  * alpha is Holm-adjusted to alpha/m for the most conservative member of
    each family, which is deliberately pessimistic; the first-rejected
    comparison faces the full alpha.
  * 80% power is a convention, not a property of the data.
  * The published effect sizes above were obtained under a 50/20/30 split,
    which gives a larger test window than the 60/20/20 protocol used here
    (for Japan, 104 points against 70). That improves the standard error by
    roughly a fifth -- not enough to move a 5% effect above these
    thresholds, but the calculation should be repeated on their split
    before making a claim about their specific results.""")
    print(f'\nwrote {os.path.relpath(out, BASE)}')


if __name__ == '__main__':
    main()
