#!/usr/bin/env python3
"""
Aggregate Multi-Seed Experiment Results

Reads individual seed results and computes mean ± std for statistical reporting.

Usage:
    python -m src.scripts.aggregate_results
    python -m src.scripts.aggregate_results --datasets japan region785
"""

import os
import sys
import glob
import argparse
import pandas as pd
import numpy as np
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS_DIR = os.path.join(BASE_DIR, 'report', 'results')

# Metrics to aggregate
METRICS = ['mae', 'rmse', 'pcc', 'R2']


def find_seed_results(dataset: str, horizon: int, ablation: str) -> pd.DataFrame:
    """Find all seed results for a configuration from all_results.csv."""
    dataset_dir = os.path.join(METRICS_DIR, dataset)
    csv_path = os.path.join(dataset_dir, 'all_results.csv')
    
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    
    df = pd.read_csv(csv_path)
    
    # Filter by horizon and ablation
    mask = (df['horizon'] == horizon) & (df['ablation'] == ablation)
    return df[mask]


def load_all_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Return filtered dataframe (already loaded)."""
    return df


def aggregate_metrics(df: pd.DataFrame) -> dict:
    """Compute mean and std for each metric."""
    result = {
        'n_seeds': len(df),
        'seeds': list(df['seed'].values) if 'seed' in df.columns else []
    }
    
    for metric in METRICS:
        # Handle case-insensitive column names
        col = None
        for c in df.columns:
            if c.lower() == metric.lower():
                col = c
                break
        
        if col is not None:
            values = df[col].values
            result[f'{metric}_mean'] = np.mean(values)
            result[f'{metric}_std'] = np.std(values)
            result[f'{metric}_min'] = np.min(values)
            result[f'{metric}_max'] = np.max(values)
    
    return result


def generate_aggregate_table(dataset: str, horizons: List[int], 
                              ablations: List[str]) -> pd.DataFrame:
    """Generate aggregated results table."""
    records = []
    
    for horizon in horizons:
        for ablation in ablations:
            df = find_seed_results(dataset, horizon, ablation)
            
            if df.empty:
                continue
            
            # Check if we have multiple seeds
            if 'seed' not in df.columns or len(df) < 2:
                # Single seed - no aggregation needed
                continue
            
            agg = aggregate_metrics(df)
            agg['dataset'] = dataset
            agg['horizon'] = horizon
            agg['ablation'] = ablation
            records.append(agg)
    
    return pd.DataFrame(records)


def format_mean_std(mean: float, std: float, precision: int = 2) -> str:
    """Format as mean ± std."""
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def print_latex_table(df: pd.DataFrame, dataset: str):
    """Print results in LaTeX table format."""
    print(f"\n% LaTeX Table for {dataset}")
    print("\\begin{table}[h]")
    print("\\centering")
    print(f"\\caption{{Results for {dataset} (mean $\\pm$ std over multiple seeds)}}")
    print("\\begin{tabular}{l|c|cccc}")
    print("\\toprule")
    print("Horizon & Ablation & RMSE & MAE & PCC & R² \\\\")
    print("\\midrule")
    
    for horizon in sorted(df['horizon'].unique()):
        h_df = df[df['horizon'] == horizon]
        for _, row in h_df.iterrows():
            ablation = row['ablation']
            rmse = format_mean_std(row['rmse_mean'], row['rmse_std'])
            mae = format_mean_std(row['mae_mean'], row['mae_std'])
            pcc = format_mean_std(row['pcc_mean'], row['pcc_std'], 4)
            r2 = format_mean_std(row['R2_mean'], row['R2_std'], 4)
            print(f"{horizon} & {ablation} & {rmse} & {mae} & {pcc} & {r2} \\\\")
        print("\\midrule")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def print_markdown_table(df: pd.DataFrame, dataset: str):
    """Print results in Markdown table format."""
    print(f"\n## {dataset} Results (mean ± std)")
    print()
    print("| Horizon | Ablation | RMSE | MAE | PCC | R² | Seeds |")
    print("|---------|----------|------|-----|-----|----|----|")
    
    for horizon in sorted(df['horizon'].unique()):
        h_df = df[df['horizon'] == horizon]
        for _, row in h_df.iterrows():
            ablation = row['ablation']
            rmse = format_mean_std(row['rmse_mean'], row['rmse_std'])
            mae = format_mean_std(row['mae_mean'], row['mae_std'])
            pcc = format_mean_std(row['pcc_mean'], row['pcc_std'], 4)
            r2 = format_mean_std(row['R2_mean'], row['R2_std'], 4)
            n_seeds = int(row['n_seeds'])
            print(f"| {horizon} | {ablation} | {rmse} | {mae} | {pcc} | {r2} | {n_seeds} |")


def main():
    parser = argparse.ArgumentParser(description='Aggregate multi-seed results')
    parser.add_argument('--datasets', nargs='+', default=['japan', 'region785', 'state360'],
                        help='Datasets to aggregate')
    parser.add_argument('--horizons', nargs='+', type=int, default=[3, 5, 10, 15],
                        help='Horizons to include')
    parser.add_argument('--ablations', nargs='+', 
                        default=['none', 'no_agam', 'no_mtfm', 'no_pprm'],
                        help='Ablations to include')
    parser.add_argument('--format', choices=['markdown', 'latex', 'both'], default='markdown',
                        help='Output format')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path')
    args = parser.parse_args()
    
    all_results = []
    
    for dataset in args.datasets:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset}")
        print('='*60)
        
        df = generate_aggregate_table(dataset, args.horizons, args.ablations)
        
        if df.empty:
            print(f"No multi-seed results found for {dataset}")
            continue
        
        all_results.append(df)
        
        if args.format in ['markdown', 'both']:
            print_markdown_table(df, dataset)
        
        if args.format in ['latex', 'both']:
            print_latex_table(df, dataset)
    
    # Save combined results
    if all_results and args.output:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.output, index=False)
        print(f"\nSaved aggregated results to: {args.output}")
    elif all_results:
        combined = pd.concat(all_results, ignore_index=True)
        output_path = os.path.join(METRICS_DIR, 'aggregated_multiseed_results.csv')
        combined.to_csv(output_path, index=False)
        print(f"\nSaved aggregated results to: {output_path}")


if __name__ == '__main__':
    main()
