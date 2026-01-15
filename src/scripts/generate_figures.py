#!/usr/bin/env python3
"""
Publication Figure Generator for MSAGAT-Net

Generates publication-ready figures from experiment results.

Usage:
    python -m src.scripts.generate_figures
    python -m src.scripts.generate_figures --output-dir report/figures/paper
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS_DIR = os.path.join(BASE_DIR, 'report', 'results')
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, 'report', 'figures', 'paper')
DEFAULT_WINDOW = 20

# Consolidated file names
ALL_RESULTS_CSV = "all_results.csv"
ALL_ABLATION_SUMMARY_CSV = "all_ablation_summary.csv"

# Dataset configurations: (name, horizons)
DATASET_CONFIGS = [
    ('japan', [3, 5, 10, 15]),
    ('region785', [3, 5, 10, 15]),
    ('nhs_timeseries', [3, 7, 14]),
    ('ltla_timeseries', [3, 7, 14]),
]

ABLATIONS = ['none', 'no_agam', 'no_mtfm', 'no_pprm']
METRICS = ['rmse', 'mae', 'pcc', 'r2']

# Display names
METRIC_NAMES = {'rmse': 'RMSE', 'mae': 'MAE', 'pcc': 'PCC', 'r2': 'R²'}
COMP_NAMES = {'agam': 'AGAM', 'mtfm': 'MTFM', 'pprm': 'PPRM'}
DATASET_NAMES = {
    'japan': 'Japan',
    'region785': 'US Region',
    'nhs_timeseries': 'NHS',
    'ltla_timeseries': 'LTLA'
}


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcdefaults()
    
    colors = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161"]
    sns.set_palette(colors)
    sns.set_style('ticks', {'axes.grid': False})
    
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.figsize'] = (7, 5)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'


def normalize(name: str) -> str:
    """Normalize column name for matching."""
    return name.strip().lower().replace('-', '').replace('_', '')


def get_metric(df: pd.DataFrame, key: str) -> float:
    """Extract metric value from DataFrame."""
    search = normalize(key)
    for col in df.columns:
        if normalize(col) == search:
            return df[col].iloc[0]
    return float('nan')


def load_metrics(dataset: str, horizon: int, ablation: str) -> Optional[pd.DataFrame]:
    """Load metrics from consolidated or individual CSV file."""
    consolidated_path = os.path.join(METRICS_DIR, ALL_RESULTS_CSV)
    if os.path.exists(consolidated_path):
        df = pd.read_csv(consolidated_path)
        mask = (
            (df['dataset'] == dataset) & 
            (df['window'] == DEFAULT_WINDOW) & 
            (df['horizon'] == horizon) & 
            (df['ablation'] == ablation)
        )
        if mask.any():
            return df[mask].iloc[[0]]
    
    # Fallback to individual files
    patterns = [f"final_metrics_*{dataset}*w-{DEFAULT_WINDOW}*h-{horizon}*{ablation}.csv"]
    for pat in patterns:
        matches = glob.glob(os.path.join(METRICS_DIR, pat))
        if matches:
            return pd.read_csv(matches[0])
    return None


def load_summary(dataset: str, horizon: int) -> Optional[pd.DataFrame]:
    """Load ablation summary from consolidated CSV file."""
    consolidated_path = os.path.join(METRICS_DIR, ALL_ABLATION_SUMMARY_CSV)
    if os.path.exists(consolidated_path):
        df = pd.read_csv(consolidated_path)
        mask = (
            (df['dataset'] == dataset) & 
            (df['window'] == DEFAULT_WINDOW) & 
            (df['horizon'] == horizon)
        )
        if mask.any():
            return df[mask].set_index('ablation')
    return None


def save_fig(fig, name: str, out_dir: str):
    """Save figure as PNG."""
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"{name}.png")
    fig.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"  Saved: {name}.png")
    plt.close(fig)


def fig_performance_vs_horizon(out_dir: str):
    """Create line plots showing metrics across prediction horizons."""
    setup_style()
    
    markers = ['o', 's', 'D', '^']
    linestyles = ['-', '--', '-.', ':']
    colors = ['#0173B2', '#DE8F05', '#029E73', '#D55E00']
    
    for metric in ['rmse', 'pcc']:
        fig, ax = plt.subplots(figsize=(7, 5))
        
        for idx, (ds, horizons) in enumerate(DATASET_CONFIGS):
            vals, hs = [], []
            for h in horizons:
                df = load_metrics(ds, h, 'none')
                if df is not None:
                    val = get_metric(df, metric)
                    if not np.isnan(val):
                        vals.append(val)
                        hs.append(h)
            
            if vals:
                ax.plot(hs, vals, 
                       marker=markers[idx], 
                       linestyle=linestyles[idx],
                       color=colors[idx],
                       linewidth=2, 
                       markersize=8,
                       label=DATASET_NAMES.get(ds, ds))
        
        ax.set_xlabel('Prediction Horizon (days)')
        ax.set_ylabel(METRIC_NAMES.get(metric, metric))
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        save_fig(fig, f'performance_vs_horizon_{metric}', out_dir)


def fig_ablation_comparison(out_dir: str):
    """Create bar charts comparing ablation variants."""
    setup_style()
    
    for ds, horizons in DATASET_CONFIGS:
        for horizon in horizons:
            summary = load_summary(ds, horizon)
            if summary is None or len(summary) < 2:
                continue
            
            # Get change columns
            change_cols = [c for c in summary.columns if 'CHANGE' in c]
            if not change_cols:
                continue
            
            ablation_names = ['no_agam', 'no_mtfm', 'no_pprm']
            ablation_labels = [COMP_NAMES.get(a.replace('no_', ''), a) for a in ablation_names]
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            x = np.arange(len(change_cols))
            width = 0.25
            
            for i, abl in enumerate(ablation_names):
                if abl in summary.index:
                    vals = [summary.loc[abl, c] if c in summary.columns else 0 for c in change_cols]
                    ax.bar(x + i * width, vals, width, label=ablation_labels[i])
            
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_xlabel('Metric')
            ax.set_ylabel('% Change from Full Model')
            ax.set_title(f'{DATASET_NAMES.get(ds, ds)} (h={horizon})')
            ax.set_xticks(x + width)
            ax.set_xticklabels([c.replace('_CHANGE', '') for c in change_cols])
            ax.legend()
            
            save_fig(fig, f'ablation_{ds}_h{horizon}', out_dir)


def fig_dataset_comparison(out_dir: str):
    """Create grouped bar chart comparing datasets."""
    setup_style()
    
    datasets = [ds for ds, _ in DATASET_CONFIGS]
    
    for metric in ['rmse', 'mae']:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_horizons = sorted(set(h for _, horizons in DATASET_CONFIGS for h in horizons))
        x = np.arange(len(datasets))
        width = 0.2
        
        for i, horizon in enumerate(all_horizons[:4]):  # Limit to 4 horizons
            vals = []
            for ds, horizons in DATASET_CONFIGS:
                if horizon in horizons:
                    df = load_metrics(ds, horizon, 'none')
                    if df is not None:
                        vals.append(get_metric(df, metric))
                    else:
                        vals.append(np.nan)
                else:
                    vals.append(np.nan)
            
            ax.bar(x + i * width, vals, width, label=f'h={horizon}')
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel(METRIC_NAMES.get(metric, metric))
        ax.set_xticks(x + 1.5 * width)
        ax.set_xticklabels([DATASET_NAMES.get(ds, ds) for ds in datasets])
        ax.legend(title='Horizon')
        
        save_fig(fig, f'dataset_comparison_{metric}', out_dir)


def generate_results_table(out_dir: str):
    """Generate LaTeX table of results."""
    rows = []
    
    for ds, horizons in DATASET_CONFIGS:
        for horizon in horizons:
            df = load_metrics(ds, horizon, 'none')
            if df is not None:
                row = {
                    'Dataset': DATASET_NAMES.get(ds, ds),
                    'Horizon': horizon,
                    'MAE': f"{get_metric(df, 'mae'):.4f}",
                    'RMSE': f"{get_metric(df, 'rmse'):.4f}",
                    'PCC': f"{get_metric(df, 'pcc'):.4f}",
                    'R²': f"{get_metric(df, 'r2'):.4f}",
                }
                rows.append(row)
    
    if rows:
        table_df = pd.DataFrame(rows)
        latex_path = os.path.join(out_dir, 'results_table.tex')
        table_df.to_latex(latex_path, index=False, escape=False)
        print(f"  Saved: results_table.tex")
        
        csv_path = os.path.join(out_dir, 'results_table.csv')
        table_df.to_csv(csv_path, index=False)
        print(f"  Saved: results_table.csv")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate publication figures')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUT_DIR,
                        help='Output directory for figures')
    parser.add_argument('--figures', nargs='+', 
                        default=['horizon', 'ablation', 'comparison', 'table'],
                        help='Which figures to generate')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    out_dir = args.output_dir
    
    print(f"Generating figures to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    
    if 'horizon' in args.figures:
        print("\nGenerating performance vs horizon plots...")
        fig_performance_vs_horizon(out_dir)
    
    if 'ablation' in args.figures:
        print("\nGenerating ablation comparison plots...")
        fig_ablation_comparison(out_dir)
    
    if 'comparison' in args.figures:
        print("\nGenerating dataset comparison plots...")
        fig_dataset_comparison(out_dir)
    
    if 'table' in args.figures:
        print("\nGenerating results table...")
        generate_results_table(out_dir)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
