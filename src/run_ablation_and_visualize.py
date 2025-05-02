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
    - Enhanced versions of saved figures
    - A final overview figure
"""

import os
import subprocess
import logging
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
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(BASE_DIR, 'report', 'results')

# We still point at your checkpoint folder, but do not create it here
SAVE_DIR    = 'save_all'

# Folders for our visual outputs
FIG_DIR     = 'figures'
OUT_DIR     = 'paper_figures'

# Updated training script path
TRAIN_SCRIPT    = os.path.join('src', 'train.py')
DATASET_CONFIGS = [
    ('japan',           'japan-adj',      [3, 5, 10, 15]),
    ('region785',       'region-adj',     [3, 5, 10, 15]),
    ('state360',        'state-adj-50',   [3, 5, 10, 15]),
    ('australia-covid', 'australia-adj',  [3, 7, 14]),
    ('spain-covid',     'spain-adj',      [3, 7, 14]),
    ('nhs_timeseries',  'nhs-adj',        [3, 7, 14]),
    ('ltla_timeseries', 'ltla-adj',       [3, 7, 14]),
]
ABLATIONS = ['none', 'no_agam', 'no_mtfm', 'no_pprm']
DEVICES   = [
    {'flags': ['--cuda', '--gpu', '0'], 'name': 'gpu'},
    {'flags': [],                      'name': 'cpu'}
]

# For paper figures
DATASETS_FOR_PLOTS = ['japan', 'region785', 'state360']
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
    'no_mtfm': 'No DMTFM',
    'no_pprm': 'No PPRM'
}

COMP_FULL = {
    'agam': 'Low‑Rank Adaptive Graph\nAttention Module',
    'mtfm': 'Dilated Multi‑scale Temporal\nFusion Module',
    'pprm': 'Progressive Multi‑step\nPrediction Refinement Module'
}

COMP_COLORS = {
    'Low‑Rank Adaptive Graph\nAttention Module': '#1f77b4',  # Blue
    'Dilated Multi‑scale Temporal\nFusion Module': '#ff7f0e',  # Orange
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

def load_metrics(dataset: str, window: int, horizon: int, ablation: str = 'none') -> pd.DataFrame:
    filename = f"final_metrics_{dataset}.w-{window}.h-{horizon}.{ablation}.csv"
    path     = os.path.join(METRICS_DIR, filename)
    if not os.path.exists(path):
        logger.warning(f"Missing metrics file: {path}")
        return None
    return pd.read_csv(path)

def load_summary(dataset: str, window: int, horizon: int) -> pd.DataFrame:
    filename = f"ablation_summary_{dataset}.w-{window}.h-{horizon}.csv"
    path     = os.path.join(METRICS_DIR, filename)
    if not os.path.exists(path):
        logger.warning(f"Missing summary file: {path}")
        return None
    return pd.read_csv(path, index_col=0)

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
                'RMSE':    df['rmse'][0],
                'MAE':     df['mae'][0],
                'PCC':     df['pcc'][0],
                'R2':      df['r2'][0],
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
            vals = [ (load_metrics(ds, window, h)[metric][0]
                      if load_metrics(ds, window, h) is not None else np.nan)
                    for h in horizons ]
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
        heat = summary.filter(regex='change$').drop('none_change', errors='ignore')
        heat.index = [ABL_NAMES[idx] for idx in heat.index]
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
                    vs.append(df[metr][0]); ls.append(ABL_NAMES[ab]); cs.append(colmap[ab])
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

def enhance_existing_figures(source_dir, output_dir):
    count = 0
    for fname in os.listdir(source_dir):
        if not fname.endswith('.png'):
            continue
        img = plt.imread(os.path.join(source_dir, fname))
        fig = plt.figure(figsize=(10,6), dpi=300)
        plt.imshow(img); plt.axis('off')
        plt.title(fname.replace('_',' ').replace('.png','').title(), pad=20)
        dst = os.path.join(output_dir, f"enh_{fname}")
        plt.savefig(dst, bbox_inches='tight'); plt.close()
        logger.info(f"Enhanced figure: {dst}")
        count += 1
    return count

def generate_ablation_report(dataset, window, horizon, output_dir):
    """Generate a detailed text report of the ablation study results."""
    summary = load_summary(dataset, window, horizon)
    if summary is None:
        logger.warning(f"No summary data for {dataset}, w={window}, h={horizon}")
        return None
        
    report_path = os.path.join(output_dir, f"ablation_report_{dataset}.w-{window}.h-{horizon}.txt")
    
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
            'no_mtfm': "Dilated Multi‑scale Temporal Fusion Module (DMTFM): Processes time-series patterns at different temporal resolutions",
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
        # Automatically generate conclusion
        if 'RMSE_change' in summary.columns and importance:
            most_important = sorted_components[0][0].replace('no_', '')
            least_important = sorted_components[-1][0].replace('no_', '')
            f.write(f"The {COMP_FULL[most_important]} has the most significant impact on model performance.\n")
            f.write(f"Removing this component causes a {importance[sorted_components[0][0]]:.2f}% degradation in RMSE.\n\n")
            
            if any(imp < 0 for _, imp in sorted_components):
                better_ablations = [(abl, imp) for abl, imp in importance.items() if imp < 0]
                if better_ablations:
                    better_abl, better_imp = sorted(better_ablations, key=lambda x: x[1])[0]
                    f.write(f"Interestingly, removing the {better_abl.replace('no_', '')} component slightly improves performance by {abs(better_imp):.2f}%.\n")
                    f.write(f"This suggests potential optimization opportunities or redundancy in this component.\n\n")
    
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
    # 1) TRAIN (directories created by src/train.py)
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
    for ds in DATASETS_FOR_PLOTS:
        generate_ablation_impact_grid(
            ds, 20, HORIZONS_FOR_PLOTS, OUT_DIR
        )
        generate_performance_comparison_grid(
            ds, 20, HORIZONS_FOR_PLOTS, OUT_DIR
        )
    enhance_existing_figures(FIG_DIR, OUT_DIR)
    generate_overview_figure(OUT_DIR)

    logger.info("Pipeline complete!")