"""Build threshold-parameterized spatial adjacency matrices from centroids.

The centroid files in ``data/geo/`` encode the recovered provenance of the
shipped adjacency matrices:

- ``ltla_centroids.csv`` — ONS Local Authority Districts (December 2021, UK,
  BUC) minus City of London and Isles of Scilly (which UKHSA reporting merges
  into Hackney and Cornwall), sorted by area name. This row order matches the
  columns of ``data/ltla_timeseries.txt``; at 150 km it reproduces the shipped
  ``data/ltla-adj.txt`` with Jaccard 0.9993 (15 borderline pairs at 141-156 km
  flip with centroid-definition differences).
- ``nhs_centroids.csv`` — ONS NHS England Regions (April 2021) centroids in
  alphabetical order; at 150 km this reproduces ``data/nhs-adj.txt`` exactly.

Usage (from the repository root):
    python -m src.scripts.build_adjacency --thresholds 100 150 200 250
    python -m src.scripts.build_adjacency --validate
"""

import argparse
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
GEO_DIR = os.path.join(DATA_DIR, 'geo')

TARGETS = {
    'ltla': ('ltla_centroids.csv', 'ltla-adj'),
    'nhs': ('nhs_centroids.csv', 'nhs-adj'),
}


def haversine_km(lat, lon):
    phi, lam = np.radians(lat), np.radians(lon)
    a = (np.sin((phi[:, None] - phi[None, :]) / 2) ** 2
         + np.cos(phi[:, None]) * np.cos(phi[None, :])
         * np.sin((lam[:, None] - lam[None, :]) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def build(key, threshold):
    csv_name, _ = TARGETS[key]
    df = pd.read_csv(os.path.join(GEO_DIR, csv_name))
    D = haversine_km(df['lat'].values, df['lon'].values)
    return (D <= threshold).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--thresholds', nargs='+', type=int,
                    default=[100, 150, 200, 250])
    ap.add_argument('--validate', action='store_true',
                    help='compare the 150 km rebuild against the shipped matrices')
    args = ap.parse_args()

    if args.validate:
        for key, (_, adj_name) in TARGETS.items():
            shipped = np.loadtxt(os.path.join(DATA_DIR, f'{adj_name}.txt'),
                                 delimiter=',')
            A = build(key, 150)
            inter = np.logical_and(A == 1, shipped == 1).sum()
            union = np.logical_or(A == 1, shipped == 1).sum()
            print(f'{key}: n={A.shape[0]} jaccard={inter / union:.4f} '
                  f'exact={np.array_equal(A.astype(float), shipped)} '
                  f'density={A.mean():.3f} (shipped {shipped.mean():.3f})')
        return

    for key, (_, adj_name) in TARGETS.items():
        for thr in args.thresholds:
            A = build(key, thr)
            out = os.path.join(DATA_DIR, f'{adj_name}-{thr}.txt')
            np.savetxt(out, A, fmt='%d', delimiter=',')
            print(f'{out}: density={A.mean():.3f} '
                  f'isolated={int((A.sum(1) == 1).sum())}')


if __name__ == '__main__':
    main()
