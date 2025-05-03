#!/usr/bin/env python3
"""
Full pipeline script that:
 1) Invokes your existing `train_ablation.py` to train MSTAGAT-Net and ablations
    across multiple datasets, horizons, and on CPU/GPU.
 2) Loads the resulting CSV metrics (from report/results) and generates:
    - A performance summary table
    - Cross-horizon RMSE/PCC plots
    - Component importance bar charts
    - Ablation-impact heatmap grids
    - Performance-comparison bar-chart grids
    - A final overview figure
"""

import os
import subprocess
import logging
import argparse
import glob
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# =============================================================================
#  CONFIGURATION
# =============================================================================

# Where `train.py` writes its final CSV metrics
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(BASE_DIR, 'report', 'results')

# We still point at your checkpoint folder, but do not create it here
SAVE_DIR    = 'save_all'

# Folders for our visual outputs
FIG_DIR     = 'paper_figures'
OUT_DIR     = os.path.join('report', 'paper_figures')

# Updated training script path
TRAIN_SCRIPT    = os.path.join('src', 'train.py')
DATASET_CONFIGS = [
    ('japan',           'japan-adj',      [3, 5, 10, 15]),
    ('region785',       'region-adj',     [3, 5, 10, 15]),
    ('state360',        'state-adj-49',   [3, 5, 10, 15]),
    ('australia-covid', 'australia-adj',  [3, 7, 14]),
    ('spain-covid',     'spain-adj',      [3, 7, 14]),
    ('nhs_timeseries',  'nhs-adj',        [3, 7, 14]),
    ('ltla_timeseries', 'ltla-adj',       [3, 7, 14]),
]
ABLATIONS = ['none', 'no_agam', 'no_mtfm', 'no_pprm']
DEVICES   = [
    {'flags': ['--cuda', '--gpu', '0'], 'name': 'gpu'},
    # {'flags': [],                      'name': 'cpu'}
]

# For paper figures
DATASETS_FOR_PLOTS = ['japan', 'region785', 'nhs_timeseries', 'ltla_timeseries']
HORIZONS_FOR_PLOTS = [3, 5, 10, 15]

METRICS_NAMES = {
    'RMSE': 'Root Mean Square Error',
    'MAE':  'Mean Absolute Error',
    'PCC':  'Pearson Correlation Coefficient',
    'R2':   'R-squared'
}

ABL_NAMES = {
    'none':    'Full Model',
    'no_agam': 'No LR‑AGAM',
    'no_mtfm': 'No MTFM',
    'no_pprm': 'No PPRM'
}

COMP_FULL = {
    'agam': 'Low‑Rank Adaptive Graph\nAttention Module',
    'mtfm': 'Multi‑scale Temporal\nFusion Module',
    'pprm': 'Progressive Multi‑step\nPrediction Refinement Module'
}

COMP_COLORS = {
    'Low‑Rank Adaptive Graph\nAttention Module': '#1f77b4',  # Blue
    'Multi‑scale Temporal\nFusion Module': '#ff7f0e',  # Orange
    'Progressive Multi‑step\nPrediction Refinement Module': '#2ca02c'  # Green
}

# =============================================================================
#  LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger('ablation_pipeline')

# =============================================================================
#  METRICS LOADING HELPERS
# =============================================================================

def get_metric_value(df, col_name):
    """Helper function to safely get a metric value with case-insensitive lookup"""
    if df is None:
        return np.nan
        
    # Create a lowercase mapping to handle case-insensitive matching
    col_map = {col.lower(): col for col in df.columns}
    
    # Try lowercase version of the requested column
    if col_name.lower() in col_map:
        return df[col_map[col_name.lower()]][0]
    
    # Check for common variations
    variants = [
        col_name,              # Original
        col_name.upper(),      # Uppercase 
        col_name.lower(),      # Lowercase
        col_name.title(),      # Title case
        col_name.replace('_', ' ')  # Replace underscores with spaces
    ]
    
    for variant in variants:
        if variant in df.columns:
            return df[variant][0]
    
    # If not found, log this for debugging
    logger.debug(f"Column {col_name} not found in DataFrame with columns: {list(df.columns)}")
    return np.nan

def load_metrics(dataset: str, window: int, horizon: int, ablation: str = 'none') -> pd.DataFrame:
    # Use glob pattern to find the right file regardless of naming convention
    pattern = f"final_metrics_*{dataset}*w-{window}*h-{horizon}*{ablation}.csv"
    paths = glob.glob(os.path.join(METRICS_DIR, pattern))
    
    # Also try with periods instead of dashes in case filenames use different format
    if not paths:
        pattern = f"final_metrics_*{dataset}*w-{window}*h-{horizon}*{ablation}.csv"
        paths = glob.glob(os.path.join(METRICS_DIR, pattern))
    
    if paths:
        path = paths[0]  # Take the first matching file
        logger.info(f"Found metrics file: {os.path.basename(path)}")
        df = pd.read_csv(path)
        if len(df) > 1:
            logger.info(f"Metrics file has {len(df)} rows, using first row (index 0)")
        return df
    else:
        # Fall back to the specific names we expect for better error messages
        filename1 = f"final_metrics_{dataset}.w-{window}.h-{horizon}.{ablation}.csv"
        filename2 = f"final_metrics_MSTAGAT-Net.{dataset}.w-{window}.h-{horizon}.{ablation}.csv"
        logger.warning(f"Missing metrics file: tried patterns like {filename1} and {filename2}")
        return None

def load_summary(dataset: str, window: int, horizon: int) -> pd.DataFrame:
    # Try both naming patterns: with and without "MSAGAT-Net" prefix
    filename1 = f"ablation_summary_{dataset}.w-{window}.h-{horizon}.csv"
    filename2 = f"ablation_summary_MSAGAT-Net.{dataset}.w-{window}.h-{horizon}.csv"
    
    path1 = os.path.join(METRICS_DIR, filename1)
    path2 = os.path.join(METRICS_DIR, filename2)
    
    if os.path.exists(path1):
        return pd.read_csv(path1, index_col=0)
    elif os.path.exists(path2):
        return pd.read_csv(path2, index_col=0)
    else:
        logger.warning(f"Missing summary file: {path1}")
        # Try to compute the summary on the fly
        summary = compute_ablation_summary(dataset, window, horizon)
        if summary is not None:
            logger.info(f"Generated and saved summary: {path1}")
            return summary
        return None

def compute_ablation_summary(dataset: str, window: int, horizon: int) -> pd.DataFrame:
    """Compute summary of ablation results by comparing metrics across variants."""
    metrics = {}
    
    # Load metrics for each ablation variant
    for ablation in ABLATIONS:
        df = load_metrics(dataset, window, horizon, ablation)
        if df is not None:
            metrics[ablation] = df
    
    if not metrics or 'none' not in metrics:
        logger.warning(f"Insufficient data to compute ablation summary for {dataset}, w={window}, h={horizon}")
        return None
    
    # Check for duplicated data (identical metrics files)
    identical_files = []
    for ablation in ABLATIONS:
        if ablation == 'none' or ablation not in metrics:
            continue
            
        # Compare key metrics to baseline
        all_identical = True
        for col in ['mae', 'rmse', 'pcc', 'r2']:
            abl_val = get_metric_value(metrics[ablation], col)
            base_val = get_metric_value(metrics['none'], col)
            if abs(abl_val - base_val) > 1e-10:  # Allow for floating point precision
                all_identical = False
                break
                
        if all_identical:
            identical_files.append(ablation)
            logger.warning(f"⚠️ The metrics for '{ablation}' are identical to 'none' for {dataset}, h={horizon}!")
            logger.warning(f"This suggests the ablation might not be working correctly.")
            
    if identical_files:
        logger.warning(f"Identical metrics were found for components: {', '.join(identical_files)}")
        # Provide more guidance about the likely issue
        logger.warning(f"This is likely due to a mismatch between CLI argument names and code implementation:")
        logger.warning(f"  - Check if 'no_mtfm' in CLI corresponds to 'no_dmtm' in ablation.py")
        logger.warning(f"  - Check if 'no_pprm' in CLI corresponds to 'no_ppm' in ablation.py") 
        logger.warning(f"Recommendation: Verify parameter names in ablation.py match those in train.py and run_ablation_and_visualize.py")
            
    # Organize performance metrics
    metric_cols = ['mae', 'rmse', 'pcc', 'r2']
    baseline_metrics = {col: get_metric_value(metrics['none'], col) for col in metric_cols}
    
    # Build summary DataFrame
    summary_data = {}
    for ablation, df in metrics.items():
        row_data = {}
        
        # Add raw metrics
        for col in metric_cols:
            row_data[col.upper()] = get_metric_value(df, col)
        
        # Compute percentage changes for non-baseline models
        if ablation != 'none':
            for col in metric_cols:
                value = get_metric_value(df, col)
                baseline = baseline_metrics[col]
                if not np.isnan(value) and not np.isnan(baseline) and baseline != 0:
                    pct_change = 100 * (value - baseline) / baseline
                    row_data[f"{col.upper()}_change"] = pct_change
                    
                    # Log very small changes that might be numerical errors
                    if abs(pct_change) < 1e-10:
                        logger.debug(f"Very small change detected for {dataset}, {ablation}, {col}: {pct_change}")
                    logger.info(f"{dataset}, h={horizon}, {ablation}, {col}: {value:.6f} vs baseline {baseline:.6f} = {pct_change:.2f}%")
        
        summary_data[ablation] = row_data
    
    summary_df = pd.DataFrame(summary_data).T
    
    # Save summary to CSV
    # Use the same format as metrics files - check which format was used for metrics
    first_ablation = list(metrics.keys())[0]
    first_file = metrics[first_ablation].iloc[0]['model'] if 'model' in metrics[first_ablation].columns else "MSTAGAT-Net"
    
    if first_file.startswith("MSAGAT-Net"):
        # Use format with model name
        filename = f"ablation_summary_MSAGAT-Net.{dataset}.w-{window}.h-{horizon}.csv"
    else:
        # Use standard format
        filename = f"ablation_summary_{dataset}.w-{window}.h-{horizon}.csv"
    
    path = os.path.join(METRICS_DIR, filename)
    summary_df.to_csv(path)
    logger.info(f"Saved ablation summary to {path}")
    
    return summary_df

# =============================================================================
#  VISUALIZATION FUNCTIONS
# =============================================================================

def generate_performance_table(datasets, window, horizons, output_dir):
    rows = []
    for ds in datasets:
        for h in horizons:
            df = load_metrics(ds, window, h)
            if df is None: continue
            
            rows.append({
                'Dataset': ds,
                'Horizon': h,
                'RMSE':    get_metric_value(df, 'rmse'),
                'MAE':     get_metric_value(df, 'mae'),
                'PCC':     get_metric_value(df, 'pcc'),
                'R2':      get_metric_value(df, 'r2'),
            })
    if not rows:
        logger.warning("No performance data available.")
        return
    perf_df = pd.DataFrame(rows)
    perf_df.to_csv(os.path.join(output_dir, f"performance_table_w{window}.csv"),
                   index=False, float_format='%.4f')
    logger.info("Saved performance table")

def generate_cross_horizon_performance(datasets, window, horizons, output_dir):
    
    for metric in ['rmse', 'pcc']:
        plt.figure()
        for ds in datasets:
            vals = [get_metric_value(load_metrics(ds, window, h), metric) for h in horizons]
            plt.plot(horizons, vals, 'o-', label=ds)
        plt.title(f"{METRICS_NAMES[metric.upper()]} vs Horizon")
        plt.xlabel("Horizon (days)"); plt.ylabel(METRICS_NAMES[metric.upper()])
        plt.xticks(horizons); plt.grid(alpha=0.3, linestyle='--'); plt.legend()
        plt.savefig(os.path.join(output_dir, f"{metric}_vs_horizon_w{window}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved cross-horizon {metric} plot")

def generate_component_importance_comparison(datasets, window, horizons, output_dir):
    data = []
    for ds in datasets:
        for h in horizons:
            summary = load_summary(ds, window, h)
            if summary is None: continue
            for ab in ['no_agam','no_mtfm','no_pprm']:
                if ab in summary.index and 'RMSE_change' in summary.columns:
                    comp = ab.replace('no_','')
                    data.append({
                        'Horizon':   h,
                        'Component': COMP_FULL[comp],
                        'Importance': abs(summary.loc[ab, 'RMSE_change'])
                    })
    if not data:
        logger.warning("No component-importance data.")
        return
    df = pd.DataFrame(data)
    plt.figure()
    sns.barplot(x='Horizon', y='Importance', hue='Component',
                data=df, palette=list(COMP_COLORS.values()))
    plt.title("Component Importance Across Horizons")
    plt.xlabel("Horizon (days)"); plt.ylabel("% RMSE Change")
    plt.legend(title="Component")
    plt.savefig(os.path.join(output_dir, f"component_importance_w{window}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved component-importance chart")

def generate_ablation_impact_grid(dataset, window, horizons, output_dir):
    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 5), sharey=True)
    for i, h in enumerate(horizons):
        summary = load_summary(dataset, window, h)
        if summary is None:
            axes[i].text(0.5, 0.5, f"No data h={h}", ha='center')
            axes[i].axis('off')
            continue
            
        # Log the summary data to help with debugging
        logger.info(f"Summary data for {dataset}, h={h}:\n{summary}")
        
        heat = summary.filter(regex='change$').drop('none_change', errors='ignore')
        heat.index = [ABL_NAMES[idx] for idx in heat.index]
        
        # Check if all changes are zero for any components and log this
        zero_components = []
        for idx, row in heat.iterrows():
            if (abs(row) < 1e-10).all():
                zero_components.append(idx)
        
        if zero_components:
            logger.warning(f"⚠️ The following components show no impact (all zeros) for {dataset}, h={h}: {', '.join(zero_components)}")
            logger.warning(f"This suggests the ablation is not working correctly - metrics are identical to baseline.")
            
            # Add a special marker for zero impact components 
            # We'll create a mask to highlight zero-impact cells
            mask = np.zeros_like(heat.T.values, dtype=bool)
            for comp_idx, comp_name in enumerate(heat.index):
                if comp_name in zero_components:
                    mask[:, heat.index.get_loc(comp_name)] = True
            
            # Create the heatmap with the mask for highlighting
            ax = axes[i]
            sns.heatmap(heat.T, annot=True, fmt='.1f', cmap='RdYlGn_r',
                      ax=ax, cbar=(i == len(horizons)-1))
            
            # Add warning text for components with zero impact
            for comp_name in zero_components:
                comp_col = heat.index.get_loc(comp_name)
                for metric_idx in range(len(heat.columns)):
                    ax.text(comp_col + 0.5, metric_idx + 0.5, 'ZERO IMPACT!\nNeeds fixing',
                           ha='center', va='center', color='red', fontweight='bold', fontsize=8)
        else:
            sns.heatmap(heat.T, annot=True, fmt='.1f', cmap='RdYlGn_r',
                      ax=axes[i], cbar=(i == len(horizons)-1))
            
        axes[i].set_title(f"{h}-day")
    plt.suptitle(f"Ablation Impact ({dataset})")
    plt.savefig(os.path.join(output_dir, f"ablation_impact_{dataset}_w{window}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved ablation-impact grid")

def generate_performance_comparison_grid(dataset, window, horizons, output_dir):
    
    metrics = ['rmse','mae','pcc','r2']
    abs_    = ['none','no_agam','no_mtfm','no_pprm']
    colmap  = {'none':'#2ca02c','no_agam':'#d62728','no_mtfm':'#ff7f0e','no_pprm':'#1f77b4'}
    fig, axes = plt.subplots(len(metrics), len(horizons),
                             figsize=(4*len(horizons),3*len(metrics)),
                             constrained_layout=True)
    for i, metr in enumerate(metrics):
        for j, h in enumerate(horizons):
            ax = axes[i,j]
            vs, ls, cs = [], [], []
            for ab in abs_:
                df = load_metrics(dataset, window, h, ab)
                if df is not None:
                    value = get_metric_value(df, metr)
                    if not np.isnan(value):
                        vs.append(value)
                        ls.append(ABL_NAMES[ab])
                        cs.append(colmap[ab])
            ax.bar(ls, vs, color=cs)
            if j==0: ax.set_ylabel(METRICS_NAMES[metr.upper()])
            if i==0: ax.set_title(f"{h}-day")
            ax.grid(alpha=0.3, linestyle='--')
            for xi, v in enumerate(vs):
                ax.text(xi, v, f"{v:.3f}", ha='center', va='bottom', fontsize=8)
    plt.suptitle(f"Performance Comparison ({dataset})")
    plt.savefig(os.path.join(output_dir, f"performance_comp_{dataset}_w{window}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved performance-comparison grid")

def generate_single_horizon_metrics(dataset, window, horizon, output_dir):
    """Generate individual performance comparison charts for a specific horizon."""
    
    metrics = ['rmse', 'mae', 'pcc', 'r2']
    abs_    = ['none', 'no_agam', 'no_mtfm', 'no_pprm']
    colmap  = {'none': '#2ca02c', 'no_agam': '#d62728', 'no_mtfm': '#ff7f0e', 'no_pprm': '#1f77b4'}
    
    for metr in metrics:
        vs, ls, cs = [], [], []
        for ab in abs_:
            df = load_metrics(dataset, window, horizon, ab)
            if df is not None:
                value = get_metric_value(df, metr)
                if not np.isnan(value):
                    vs.append(value)
                    ls.append(ABL_NAMES[ab])
                    cs.append(colmap[ab])
        
        if not vs:
            logger.warning(f"No data for {dataset}, h={horizon}, metric={metr}")
            continue
            
        plt.figure(figsize=(10, 6))
        bars = plt.bar(ls, vs, color=cs)
        plt.ylabel(METRICS_NAMES[metr.upper()])
        plt.title(f"{METRICS_NAMES[metr.upper()]} Comparison - {dataset} ({horizon}-day Horizon)")
        plt.grid(alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f"{height:.3f}", ha='center', va='bottom')
        
        plt.savefig(os.path.join(output_dir, f"{metr}_comparison_{dataset}_h{horizon}_w{window}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {metr} comparison for {dataset} h={horizon}")

def generate_component_contribution_charts(datasets, window, horizon, output_dir):
    """Generate bar charts showing relative contribution of components for specific horizons."""
    metrics = ['rmse', 'pcc']
    data = {}
    
    for metric in metrics:
        metric_data = []
        for ds in datasets:
            summary = load_summary(ds, window, horizon)
            if summary is None:
                continue
                
            for ab in ['no_agam', 'no_mtfm', 'no_pprm']:
                if ab in summary.index and f"{metric.upper()}_change" in summary.columns:
                    comp = ab.replace('no_', '')
                    metric_data.append({
                        'Dataset': ds,
                        'Component': COMP_FULL[comp],
                        'Contribution': abs(summary.loc[ab, f"{metric.upper()}_change"])
                    })
        
        if not metric_data:
            logger.warning(f"No component contribution data for h={horizon}, metric={metric}")
            continue
            
        data[metric] = pd.DataFrame(metric_data)
    
    # Create visualizations for each metric
    for metric, df in data.items():
        if df.empty:
            continue
            
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Component', y='Contribution', hue='Dataset', data=df)
        plt.title(f"Component Contribution to {METRICS_NAMES[metric.upper()]} - {horizon}-day Forecast")
        plt.ylabel(f"% {metric.upper()} Change")
        plt.xticks(rotation=0)
        plt.legend(title="Dataset")
        
        plt.savefig(os.path.join(output_dir, f"component_contribution_{metric}_{horizon}day_w{window}.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved component contribution chart for {metric}, h={horizon}")

def generate_relative_contribution_chart(datasets, window, horizon, output_dir):
    """Generate visualization of relative contribution of each component at specific horizon."""
    all_data = []
    for ds in datasets:
        summary = load_summary(ds, window, horizon)
        if summary is None:
            continue
            
        for ab in ['no_agam', 'no_mtfm', 'no_pprm']:
            if ab in summary.index and 'RMSE_change' in summary.columns:
                comp = ab.replace('no_', '')
                all_data.append({
                    'Dataset': ds,
                    'Component': COMP_FULL[comp],
                    'Color': COMP_COLORS[COMP_FULL[comp]],
                    'Contribution': abs(summary.loc[ab, 'RMSE_change'])
                })
    
    if not all_data:
        logger.warning(f"No relative contribution data for h={horizon}")
        return
        
    df = pd.DataFrame(all_data)
    
    plt.figure(figsize=(14, 8))
    
    # Create a grouped bar chart
    ax = plt.subplot(1, 2, 1)
    sns.barplot(x='Dataset', y='Contribution', hue='Component', data=df, 
                palette=list(COMP_COLORS.values()))
    plt.title(f"Component Contributions - {horizon}-day Forecast")
    plt.ylabel("% RMSE Degradation")
    plt.legend(title="Component")
    
    # Create a pie chart of average contributions
    ax2 = plt.subplot(1, 2, 2)
    avg_contribution = df.groupby('Component')['Contribution'].mean().reset_index()
    colors = [row['Color'] for _, row in avg_contribution.iterrows()]
    
    # Calculate the average contribution percentage for each component
    total = avg_contribution['Contribution'].sum()
    percentages = [100 * val / total for val in avg_contribution['Contribution']]
    
    # Create labels with component name and percentage
    labels = [f"{comp}\n({pct:.1f}%)" for comp, pct in zip(avg_contribution['Component'], percentages)]
    
    ax2.pie(avg_contribution['Contribution'], labels=labels, colors=colors,
           autopct='', startangle=90, wedgeprops={'edgecolor': 'w'})
    ax2.set_title(f"Average Relative Contribution - {horizon}-day")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"relative_contribution_{horizon}day_w{window}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved relative contribution chart for h={horizon}")

# Function enhance_existing_figures removed as requested

def generate_ablation_report(dataset, window, horizon, output_dir):
    """Generate a detailed text report of the ablation study results."""
    summary = load_summary(dataset, window, horizon)
    if summary is None:
        logger.warning(f"No summary data for {dataset}, w={window}, h={horizon}")
        return None
    
    # Check for zero impact components
    heat = summary.filter(regex='change$').drop('none_change', errors='ignore')
    zero_components = []
    for idx, row in heat.iterrows():
        if (abs(row) < 1e-10).all():
            zero_components.append(ABL_NAMES[idx])
    
    # Save reports to the report/results directory instead of figures directory
    report_dir = os.path.join(BASE_DIR, 'report', 'results')
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"ablation_report_{dataset}.w-{window}.h-{horizon}.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"MSAGAT-Net Ablation Study Report\n")
        f.write(f"==============================\n\n")
        
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Window Size: {window}\n")
        f.write(f"Forecast Horizon: {horizon}\n\n")
        
        f.write(f"Performance Metrics by Model Variant\n")
        f.write(f"-----------------------------------\n\n")
        metric_cols = [col for col in summary.columns if not col.endswith('_change')]
        f.write(f"{summary[metric_cols].to_string()}\n\n")
        
        change_cols = [col for col in summary.columns if col.endswith('_change')]
        if change_cols:
            f.write(f"Percentage Change from Full Model\n")
            f.write(f"--------------------------------\n\n")
            f.write(f"{summary[change_cols].to_string()}\n\n")
        
        f.write(f"Component Importance Analysis\n")
        f.write(f"----------------------------\n\n")
        
        # Map ablation types to component descriptions
        component_desc = {
            'no_agam': "Low‑Rank Adaptive Graph Attention Module (LR‑AGAM): Captures spatial dependencies between regions with linear complexity",
            'no_mtfm': "Multi‑scale Temporal Fusion Module (MTFM): Processes time-series patterns at different temporal resolutions",
            'no_pprm': "Progressive Multi‑step Prediction Refinement Module (PPRM): Enables region-aware multi-step forecasting with refinement"
        }
        
        # Calculate importance ranking based on RMSE degradation
        if 'RMSE_change' in summary.columns:
            importance = {}
            for ablation in summary.index:
                if ablation != 'none' and ablation in component_desc and not pd.isna(summary.loc[ablation, 'RMSE_change']):
                    importance[ablation] = abs(summary.loc[ablation, 'RMSE_change'])
            
            # Sort by importance
            sorted_components = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            # Write importance ranking
            for i, (ablation, imp) in enumerate(sorted_components):
                f.write(f"{i+1}. {component_desc[ablation]}\n")
                f.write(f"   Impact when removed: {imp:.2f}% RMSE degradation\n\n")
        
        f.write(f"Conclusion\n")
        f.write(f"----------\n")
        
        # Check if we have ablation problems
        if zero_components:
            f.write(f"⚠️ ABLATION STUDY INCOMPLETE: Some components ({', '.join(zero_components)}) show no impact\n")
            f.write(f"when ablated, which suggests those ablations are not being properly implemented.\n")
            f.write(f"The ablation parameter names in train.py may not match those in ablation.py.\n")
            f.write(f"Please fix the ablation implementation and re-run the evaluation.\n\n")
        
        # Automatically generate conclusion
        if 'RMSE_change' in summary.columns and importance:
            # Get the most important non-zero component if possible
            valid_components = [(abl, imp) for abl, imp in sorted_components if abl not in zero_components]
            
            if valid_components:
                most_important = valid_components[0][0].replace('no_', '')
                f.write(f"The {COMP_FULL[most_important]} has the most significant impact on model performance.\n")
                f.write(f"Removing this component causes a {importance[valid_components[0][0]]:.2f}% degradation in RMSE.\n\n")
                
                if any(imp < 0 for abl, imp in valid_components):
                    better_ablations = [(abl, imp) for abl, imp in valid_components if imp < 0]
                    if better_ablations:
                        better_abl, better_imp = sorted(better_ablations, key=lambda x: x[1])[0]
                        f.write(f"Interestingly, removing the {better_abl.replace('no_', '')} component slightly improves performance by {abs(better_imp):.2f}%.\n")
                        f.write(f"This suggests potential optimization opportunities or redundancy in this component.\n\n")
            else:
                f.write(f"⚠️ Unable to draw meaningful conclusions due to issues with the ablation implementation.\n\n")
    
    logger.info(f"Generated ablation report: {report_path}")
    return report_path

def generate_overview_figure(output_dir):
    texts = ["Architecture", "Performance", "Component Importance", "Ablation Impact"]
    fig = plt.figure(figsize=(12,10))
    gs  = gridspec.GridSpec(2,2)
    for idx, txt in enumerate(texts):
        ax = fig.add_subplot(gs[idx//2, idx%2])
        ax.text(0.5,0.5, txt, ha='center', va='center', fontsize=14)
        ax.axis('off')
    plt.suptitle("Study Overview", y=0.95, fontsize=16)
    out = os.path.join(output_dir, "overview.png")
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("Saved overview figure")

# =============================================================================
#  MAIN
# =============================================================================

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run ablation studies and generate visualizations')
    parser.add_argument('--only_visualize', action='store_true',
                        help='Skip training and only generate visualizations')
    args = parser.parse_args()

    # 1) TRAIN (directories created by src/train.py)
    if not args.only_visualize:
        for dataset, sim_mat, horizons in DATASET_CONFIGS:
            for h in horizons:
                for ab, dev in product(ABLATIONS, DEVICES):
                    cmd = [
                        'python', TRAIN_SCRIPT,
                        '--dataset',  dataset,
                        '--sim_mat',  sim_mat,
                        '--window',   '20',
                        '--horizon',  str(h),
                        '--ablation', ab,
                        '--save_dir', SAVE_DIR
                    ] + dev['flags']
                    logger.info(f"Running training: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
    else:
        logger.info("Skipping training, generating visualizations only")

    # Make sure output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # 2) VISUALIZE
    generate_performance_table(
        DATASETS_FOR_PLOTS, 20, HORIZONS_FOR_PLOTS, OUT_DIR
    )
    generate_cross_horizon_performance(
        DATASETS_FOR_PLOTS, 20, HORIZONS_FOR_PLOTS, OUT_DIR
    )
    generate_component_importance_comparison(
        DATASETS_FOR_PLOTS, 20, HORIZONS_FOR_PLOTS, OUT_DIR
    )
    
    # Process each dataset and horizon to create visualizations and reports
    for ds in DATASETS_FOR_PLOTS:
        generate_ablation_impact_grid(
            ds, 20, HORIZONS_FOR_PLOTS, OUT_DIR
        )
        generate_performance_comparison_grid(
            ds, 20, HORIZONS_FOR_PLOTS, OUT_DIR
        )
        
        # Generate individual horizon metric charts
        for h in HORIZONS_FOR_PLOTS:
            generate_single_horizon_metrics(ds, 20, h, OUT_DIR)
            generate_ablation_report(ds, 20, h, OUT_DIR)
    
    # Generate component contribution charts for 5-day and 10-day horizons
    for h in [5, 10]:
        # Only generate if this horizon is in our plots
        if h in HORIZONS_FOR_PLOTS:
            generate_component_contribution_charts(DATASETS_FOR_PLOTS, 20, h, OUT_DIR)
            generate_relative_contribution_chart(DATASETS_FOR_PLOTS, 20, h, OUT_DIR)
    
    # Also generate reports for other datasets if metrics are available
    for dataset, _, horizons in DATASET_CONFIGS:
        if dataset not in DATASETS_FOR_PLOTS:
            for h in horizons:
                generate_ablation_report(dataset, 20, h, OUT_DIR)
    # enhance_existing_figures function removed
    generate_overview_figure(OUT_DIR)

    logger.info("Pipeline complete!")