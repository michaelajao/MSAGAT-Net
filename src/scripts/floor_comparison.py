"""Score every trained model against the naive floors, per cell.

Writes `report/results/floor_comparison.csv`, one row per (dataset, horizon):
the median-seed RMSE of MSAGAT-Net's arms, of the best trained baseline, and
of each naive floor, plus which floor is hardest to beat.

This is the table SpatialEpiBench (2026) asks for and that no paper in the
Cola-GNN lineage reports. Its finding there was that "every method beats the
naive baseline less than 50% of the time"; this script answers the same
question on this benchmark family.

    python -m src.scripts.floor_comparison [--seasonal-diagnostic]

`--seasonal-diagnostic` additionally reports RMSE by seasonal lag, which is
what distinguishes a genuine annual cycle from an artefact of one particular
lag choice.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from src.scripts.naive_baselines import SEASON, load_series, split_points  # noqa: E402

INDEX = os.path.join(BASE, 'report', 'results', 'runs_index.csv')
OUT = os.path.join(BASE, 'report', 'results', 'floor_comparison.csv')
FLOORS = ['persistence', 'seasonal_naive', 'ar4']


def load_index():
    df = pd.read_csv(INDEX)
    return df[(df.arm == 'current') & df.rmse_npz.notna()]


def build(df):
    msa = df[(df.family == 'MSAGAT-Net') & (df.ablation == 'none')
             & (df.sim_mat == 'default')]
    base = df[(~df.family.str.startswith('MSAGAT')) & (~df.family.isin(FLOORS))]
    floor = df[df.family.isin(FLOORS)]

    rows = []
    for (ds, h), g in msa.groupby(['dataset', 'horizon']):
        # The paper's model: log-growth targets with quantile heads, no
        # attention-revival token, direct decoder.
        v2 = g[(g.target_space == 'loggrowth') & (g.quantiles == True)  # noqa: E712
               & g.attn_exp.isna() & (g.renewal == False)]              # noqa: E712
        v1 = g[(g.target_space == 'level') & g.attn_exp.isna()]
        arms = g.groupby(['target_space', 'quantiles', 'attn_exp', 'renewal'],
                         dropna=False).rmse_npz.median()

        b = base[(base.dataset == ds) & (base.horizon == h)]
        per_base = b.groupby('family').rmse_npz.median()
        f = floor[(floor.dataset == ds) & (floor.horizon == h)] \
            .set_index('family').rmse_npz

        rec = {
            'dataset': ds, 'horizon': h,
            'msagat_v1': v1.rmse_npz.median() if len(v1) else np.nan,
            'msagat_v2': v2.rmse_npz.median() if len(v2) else np.nan,
            'msagat_best_arm': arms.min() if len(arms) else np.nan,
            'best_baseline': per_base.min() if len(per_base) else np.nan,
            'best_baseline_name': per_base.idxmin() if len(per_base) else '',
        }
        for name in FLOORS:
            rec[name] = f.get(name, np.nan)
        rec['best_floor'] = f.min() if len(f) else np.nan
        rec['best_floor_name'] = f.idxmin() if len(f) else ''
        rec['best_trained'] = np.nanmin([rec['msagat_best_arm'],
                                         rec['best_baseline']])
        rec['v2_beats_floor'] = bool(rec['msagat_v2'] < rec['best_floor'])
        rec['any_trained_beats_floor'] = bool(rec['best_trained'] < rec['best_floor'])
        rec['floor_margin_pct'] = round(
            100 * (rec['best_trained'] - rec['best_floor']) / rec['best_trained'], 1)
        rows.append(rec)

    return pd.DataFrame(rows).sort_values(['dataset', 'horizon'])


def seasonal_diagnostic(dataset, lags=(13, 26, 39, 52, 65, 78, 104)):
    """RMSE of a seasonal-naive forecast at several lags.

    A genuine annual cycle shows a sharp minimum at one seasonal period and
    its multiples; an artefact of the chosen lag does not.
    """
    raw = load_series(dataset)
    _, val_end = split_points(len(raw))
    idx = np.arange(val_end, len(raw))
    y = raw[idx]
    out = {}
    for lag in lags:
        src = idx - lag
        if src.min() < 0:
            continue
        out[lag] = float(np.sqrt(np.mean((y - raw[src]) ** 2)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seasonal-diagnostic', action='store_true')
    args = ap.parse_args()

    df = build(load_index())
    df.to_csv(OUT, index=False)

    pd.set_option('display.width', 200)
    show = ['dataset', 'horizon', 'msagat_v2', 'msagat_best_arm',
            'best_baseline', 'best_floor', 'best_floor_name',
            'any_trained_beats_floor']
    print(df[show].to_string(index=False, float_format=lambda v: f'{v:.2f}'))

    n = len(df)
    print(f'\nMSAGAT-Net v2 beats the best floor in '
          f'{int(df.v2_beats_floor.sum())}/{n} cells')
    print(f'The best of ALL trained models beats it in '
          f'{int(df.any_trained_beats_floor.sum())}/{n} cells')
    lost = df[~df.any_trained_beats_floor]
    if len(lost):
        print('\nCells where no trained model beats a naive floor:')
        for _, r in lost.iterrows():
            print(f"  {r.dataset:16s} h={r.horizon:<3d} {r.best_floor_name:15s}"
                  f" {r.best_floor:9.1f} vs best trained {r.best_trained:9.1f}"
                  f"  ({r.floor_margin_pct:+.0f}%)")

    if args.seasonal_diagnostic:
        print('\nSeasonal-naive RMSE by lag (is one period genuinely special?)')
        for ds in sorted(df.dataset.unique()):
            res = seasonal_diagnostic(ds)
            if not res:
                continue
            best = min(res, key=res.get)
            cells = '  '.join(f'{lag}:{v:.0f}' for lag, v in res.items())
            print(f'  {ds:16s} season={SEASON[ds]:>3}  best lag {best:>3}  {cells}')

    print(f'\nwrote {os.path.relpath(OUT, BASE)}')


if __name__ == '__main__':
    main()
