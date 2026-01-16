#!/usr/bin/env python3
"""
Generate publication-ready figures for MSAGAT-Net research paper.
Outputs PNG files formatted for academic publication.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
METRICS_DIR = os.path.join(BASE_DIR, 'report', 'results')
OUT_DIR = os.path.join(BASE_DIR, 'report', 'figures', 'paper')
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

# Ablation variants
ABLATIONS = ['none', 'no_agam', 'no_mtfm', 'no_pprm']

# Metrics
METRICS = ['rmse', 'mae', 'pcc', 'r2']
METRIC_NAMES = {
    'rmse': 'RMSE',
    'mae': 'MAE',
    'pcc': 'PCC',
    'r2': 'R┬▓'
}

# Component names
COMP_NAMES = {
    'agam': 'AGAM',
    'mtfm': 'MTFM',
    'pprm': 'PPRM'
}

# Dataset display names
DATASET_NAMES = {
    'japan': 'Japan',
    'region785': 'US Region',
    'nhs_timeseries': 'NHS',
    'ltla_timeseries': 'LTLA'
}

# =============================================================================
# UTILITIES
# =============================================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize(name: str) -> str:
    return name.strip().lower().replace('-', '').replace('_', '')


def get_metric(df: pd.DataFrame, key: str) -> float:
    search = normalize(key)
    for col in df.columns:
        if normalize(col) == search:
            return df[col].iloc[0]
    return float('nan')


def load_metrics(dataset: str, horizon: int, ablation: str) -> pd.DataFrame:
    """Load metrics from consolidated or individual CSV file."""
    # Try dataset-specific directory first
    dataset_dir = os.path.join(METRICS_DIR, dataset)
    consolidated_path = os.path.join(dataset_dir, ALL_RESULTS_CSV)
    
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
    
    # Fallback to root METRICS_DIR
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
    patterns = [
        f"final_metrics_*{dataset}*w-{DEFAULT_WINDOW}*h-{horizon}*{ablation}.csv",
    ]
    for pat in patterns:
        matches = glob.glob(os.path.join(METRICS_DIR, pat))
        if matches:
            return pd.read_csv(matches[0])
    return None


def load_summary(dataset: str, horizon: int) -> pd.DataFrame:
    """Load ablation summary from consolidated or individual CSV file."""
    # Try dataset-specific directory first
    dataset_dir = os.path.join(METRICS_DIR, dataset)
    consolidated_path = os.path.join(dataset_dir, ALL_ABLATION_SUMMARY_CSV)
    
    if os.path.exists(consolidated_path):
        df = pd.read_csv(consolidated_path)
        mask = (
            (df['dataset'] == dataset) & 
            (df['window'] == DEFAULT_WINDOW) & 
            (df['horizon'] == horizon)
        )
        if mask.any():
            return df[mask].set_index('ablation')
    
    # Fallback to root METRICS_DIR
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
    
    # Fallback to individual files
    path = os.path.join(METRICS_DIR, f"ablation_summary_{dataset}.w-{DEFAULT_WINDOW}.h-{horizon}.csv")
    return pd.read_csv(path, index_col=0) if os.path.exists(path) else None


# =============================================================================
# PUBLICATION STYLE SETUP
# =============================================================================
def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcdefaults()
    
    # Colorblind-friendly palette
    colors = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161"]
    sns.set_palette(colors)
    sns.set_style('ticks', {'axes.grid': False})
    
    # Figure settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.figsize'] = (7, 5)
    
    # Typography
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 11
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['legend.frameon'] = False
    
    # Line styling
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6
    
    # Axes
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'


def save_fig(fig, name: str):
    """Save figure as PNG only."""
    ensure_dir(OUT_DIR)
    filepath = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(filepath, bbox_inches='tight', dpi=300, facecolor='white')
    print(f"  Saved: {name}.png")
    plt.close(fig)


# =============================================================================
# FIGURE 1: PERFORMANCE ACROSS HORIZONS
# =============================================================================
def fig_performance_vs_horizon():
    """Create line plots showing RMSE and PCC across prediction horizons."""
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
                       markeredgecolor='black',
                       markeredgewidth=0.5,
                       label=DATASET_NAMES.get(ds, ds.upper()))
        
        ax.set_xlabel('Prediction Horizon (days)')
        ax.set_ylabel(METRIC_NAMES[metric])
        ax.legend(loc='best')
        
        # Set x-ticks to all unique horizons
        all_horizons = sorted(set(h for _, horizons in DATASET_CONFIGS for h in horizons))
        ax.set_xticks(all_horizons)
        
        save_fig(fig, f'fig1_{metric}_vs_horizon')


# =============================================================================
# FIGURE 2: ABLATION STUDY RESULTS
# =============================================================================
def fig_ablation_study():
    """Create bar charts showing ablation study results for each dataset/horizon."""
    setup_style()
    
    ablation_colors = {
        'none': '#0173B2',
        'no_agam': '#DE8F05',
        'no_mtfm': '#029E73',
        'no_pprm': '#D55E00'
    }
    
    ablation_labels = {
        'none': 'Full Model',
        'no_agam': 'w/o AGAM',
        'no_mtfm': 'w/o MTFM',
        'no_pprm': 'w/o PPRM'
    }
    
    for ds, horizons in DATASET_CONFIGS:
        for h in horizons:
            vals, labels, colors = [], [], []
            
            for ab in ABLATIONS:
                df = load_metrics(ds, h, ab)
                if df is not None:
                    rmse = get_metric(df, 'rmse')
                    if not np.isnan(rmse):
                        vals.append(rmse)
                        labels.append(ablation_labels.get(ab, ab))
                        colors.append(ablation_colors.get(ab, '#333333'))
            
            if len(vals) < 2:
                continue
            
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.8)
            
            # Add value labels on bars
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            
            # Add baseline reference
            if vals:
                baseline = vals[0]  # Full model
                ax.axhline(y=baseline, color='navy', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.set_ylabel('RMSE')
            ax.set_title(f'{DATASET_NAMES.get(ds, ds)} - Horizon {h}')
            ax.set_ylim(0, max(vals) * 1.15)
            
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            save_fig(fig, f'fig2_ablation_{ds}_h{h}')


# =============================================================================
# FIGURE 3: COMPONENT IMPORTANCE HEATMAP
# =============================================================================
def fig_component_importance_heatmap():
    """Create heatmap showing component importance across datasets and horizons."""
    setup_style()
    
    data = []
    for ds, horizons in DATASET_CONFIGS:
        for h in horizons:
            sum_df = load_summary(ds, h)
            if sum_df is None:
                continue
            
            for ab in ['no_agam', 'no_mtfm', 'no_pprm']:
                if 'RMSE_CHANGE' in sum_df.columns and ab in sum_df.index:
                    try:
                        value = sum_df.loc[ab, 'RMSE_CHANGE']
                        if not pd.isna(value):
                            comp = ab.replace('no_', '')
                            data.append({
                                'Dataset': DATASET_NAMES.get(ds, ds),
                                'Horizon': h,
                                'Component': COMP_NAMES.get(comp, comp.upper()),
                                'RMSE Change (%)': abs(value)
                            })
                    except (KeyError, TypeError):
                        continue
    
    if not data:
        print("  No data for component importance heatmap")
        return
    
    df = pd.DataFrame(data)
    
    # Create heatmap with Dataset-Horizon as columns
    df['Dataset_Horizon'] = df['Dataset'] + ' H' + df['Horizon'].astype(str)
    pivot = df.pivot_table(
        values='RMSE Change (%)',
        index='Component',
        columns='Dataset_Horizon',
        aggfunc='first'  # Use first value, not mean
    ).fillna(0)
    
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd',
               linewidths=0.5, ax=ax, cbar_kws={'label': '% RMSE Increase'})
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_fig(fig, 'fig3_component_importance_heatmap')


# =============================================================================
# FIGURE 4: COMPONENT CONTRIBUTION BY DATASET
# =============================================================================
def fig_component_contribution():
    """Create grouped bar chart showing component contribution per dataset."""
    setup_style()
    
    for metric in ['rmse', 'pcc']:
        data = []
        for ds, horizons in DATASET_CONFIGS:
            for h in horizons:
                sum_df = load_summary(ds, h)
                if sum_df is None:
                    continue
                
                col = f'{metric.upper()}_CHANGE'
                for ab in ['no_agam', 'no_mtfm', 'no_pprm']:
                    if col in sum_df.columns and ab in sum_df.index:
                        try:
                            value = sum_df.loc[ab, col]
                            if not pd.isna(value):
                                comp = ab.replace('no_', '')
                                data.append({
                                    'Dataset': DATASET_NAMES.get(ds, ds),
                                    'Horizon': h,
                                    'Component': COMP_NAMES.get(comp, comp.upper()),
                                    'Impact (%)': abs(value)
                                })
                        except (KeyError, TypeError):
                            continue
        
        if not data:
            continue
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Dataset', y='Impact (%)', hue='Component', data=df,
                   ax=ax, palette='viridis', edgecolor='black', linewidth=0.8,
                   errorbar=None)
        
        ax.set_xlabel('')
        ax.set_ylabel(f'% {METRIC_NAMES[metric]} Change')
        ax.legend(title='Component', loc='upper center')
        plt.tight_layout()
        save_fig(fig, f'fig5_component_contribution_{metric}')


# =============================================================================
# FIGURE 6: COMPONENT IMPACT BAR CHART
# =============================================================================
def fig_component_impact():
    """Create bar chart showing impact of removing each component (% RMSE increase)."""
    setup_style()
    
    # Aggregate impact across all horizons for each dataset
    for ds, horizons in DATASET_CONFIGS:
        data = []
        
        for h in horizons:
            # Get baseline (full model) RMSE
            full_df = load_metrics(ds, h, 'none')
            if full_df is None:
                continue
            baseline_rmse = get_metric(full_df, 'rmse')
            if np.isnan(baseline_rmse) or baseline_rmse == 0:
                continue
            
            # Calculate % increase for each ablation
            for ab in ['no_agam', 'no_mtfm', 'no_pprm']:
                ab_df = load_metrics(ds, h, ab)
                if ab_df is not None:
                    ab_rmse = get_metric(ab_df, 'rmse')
                    if not np.isnan(ab_rmse):
                        pct_increase = ((ab_rmse - baseline_rmse) / baseline_rmse) * 100
                        comp = ab.replace('no_', '')
                        data.append({
                            'Horizon': f'H{h}',
                            'Component': COMP_NAMES.get(comp, comp.upper()),
                            'RMSE Increase (%)': pct_increase
                        })
        
        if not data:
            continue
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        
        comp_colors = {'AGAM': '#0173B2', 'MTFM': '#DE8F05', 'PPRM': '#029E73'}
        
        x = np.arange(len(df['Horizon'].unique()))
        width = 0.25
        horizons_list = df['Horizon'].unique()
        
        for i, comp in enumerate(['AGAM', 'MTFM', 'PPRM']):
            vals = []
            for h in horizons_list:
                subset = df[(df['Horizon'] == h) & (df['Component'] == comp)]
                if len(subset) > 0:
                    vals.append(subset['RMSE Increase (%)'].values[0])
                else:
                    vals.append(0)
            
            bars = ax.bar(x + i * width, vals, width, 
                         label=comp, color=comp_colors[comp],
                         edgecolor='black', linewidth=0.8)
            
            # Add value labels
            for bar, val in zip(bars, vals):
                if val != 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Prediction Horizon')
        ax.set_ylabel('RMSE Increase (%)')
        ax.set_title(f'{DATASET_NAMES.get(ds, ds)} - Component Impact')
        ax.set_xticks(x + width)
        ax.set_xticklabels(horizons_list)
        ax.legend(title='Removed Component')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Set y-axis to show both positive and negative
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(min(ymin, -5), max(ymax, 10))
        
        plt.tight_layout()
        save_fig(fig, f'fig6_component_impact_{ds}')


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Generate all publication figures."""
    print("=" * 60)
    print("MSAGAT-Net Publication Figure Generator")
    print("=" * 60)
    print(f"Output directory: {OUT_DIR}")
    ensure_dir(OUT_DIR)
    
    print("\n1. Generating performance vs horizon plots...")
    fig_performance_vs_horizon()
    
    print("\n2. Generating ablation study bar charts...")
    fig_ablation_study()
    
    print("\n3. Generating component importance heatmap...")
    fig_component_importance_heatmap()
    
    print("\n4. Generating component contribution charts...")
    fig_component_contribution()
    
    print("\n5. Generating component impact charts...")
    fig_component_impact()
    
    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
