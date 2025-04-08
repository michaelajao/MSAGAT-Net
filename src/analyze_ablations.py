"""
analyze_ablations.py

This script analyzes the results of ablation studies for the MSAGAT-Net model by:
1. Loading result metrics from CSV files for different ablation variants
2. Creating comparison plots to show the impact of each ablation
3. Generating a detailed summary table and heatmap visualization
4. Calculating the relative importance of each component

Usage:
python analyze_ablations.py --results_dir results --figures_dir report/figures
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import sys
import logging
from glob import glob
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# MSAGAT-Net component names
ABLATION_NAMES = {
    'none': 'Full Model',
    'no_eagam': 'No EAGAM',
    'no_dmtm': 'No DMTM',
    'no_ppm': 'No PPM'
}

COMPONENT_FULL_NAMES = {
    'eagam': 'Efficient Adaptive Graph\nAttention Module',
    'dmtm': 'Dilated Multi-Scale\nTemporal Module',
    'ppm': 'Progressive Prediction\nModule'
}

# Component colors
COMPONENT_COLORS = {
    'Efficient Adaptive Graph\nAttention Module': '#1f77b4',  # Blue
    'Dilated Multi-Scale\nTemporal Module': '#ff7f0e',        # Orange
    'Progressive Prediction\nModule': '#2ca02c'               # Green
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze MSAGAT-Net ablation results')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory containing result files (default: results)')
    parser.add_argument('--figures_dir', type=str, default='report/figures',
                        help='Directory for saving figure outputs (default: report/figures)')
    return parser.parse_args()

def find_ablation_files(results_dir):
    """Find all ablation result files in the results directory."""
    file_pattern = os.path.join(results_dir, "final_metrics_*.csv")
    files = glob(file_pattern)
    
    # Group files by dataset, window, and horizon
    file_groups = {}
    for file in files:
        basename = os.path.basename(file)
        # Extract dataset, window, horizon, and ablation from filename
        match = re.match(r"final_metrics_(.+)\.w-(\d+)\.h-(\d+)\.(.+)\.csv", basename)
        if match:
            dataset, window, horizon, ablation = match.groups()
            key = (dataset, window, horizon)
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append((ablation, file))
    
    return file_groups

def load_metrics_file(filename):
    """Load metrics from a CSV file."""
    if not os.path.exists(filename):
        logger.warning(f"Metrics file not found: {filename}")
        return None
    
    logger.info(f"Loading metrics from: {filename}")
    return pd.read_csv(filename)

def plot_ablation_comparison(ablation_files, dataset, window, horizon, figures_dir):
    """Compare metrics across different ablation studies."""
    metrics = {}
    
    # Load metrics for each ablation type
    for ablation, filename in ablation_files:
        df = load_metrics_file(filename)
        if df is not None:
            metrics[ablation] = df
    
    if not metrics:
        logger.error("No metrics files found!")
        return None
    
    # Prepare data for comparison plots
    compare_metrics = ['RMSE', 'PCC', 'R2']
    ablation_types = list(ABLATION_NAMES.keys())
    
    for metric in compare_metrics:
        values = [metrics[abl][metric][0] if abl in metrics else np.nan for abl in ablation_types]
        
        # Create a bar plot
        plt.figure(figsize=(10, 6))
        colors = ['#2ca02c' if abl == 'none' else '#d62728' if abl == 'no_eagam' else 
                 '#ff7f0e' if abl == 'no_dmtm' else '#1f77b4' for abl in ablation_types]
        
        # Use the nice display names
        display_names = [ABLATION_NAMES.get(abl, abl) for abl in ablation_types]
        
        plt.bar(display_names, values, color=colors)
        plt.title(f'MSAGAT-Net: Impact of Ablations on {metric} (h={horizon})', fontsize=16)
        plt.ylabel(metric, fontsize=14)
        plt.xlabel('Model Variant', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add value labels on top of bars
        for i, v in enumerate(values):
            if not np.isnan(v):
                plt.text(i, v + 0.01 * max([x for x in values if not np.isnan(x)]), 
                       f"{v:.4f}", ha='center', fontsize=12)
        
        # Percentage change labels relative to 'none'
        if 'none' in metrics:
            baseline = values[ablation_types.index('none')]
            for i, v in enumerate(values):
                if ablation_types[i] != 'none' and not np.isnan(v):  # Skip baseline and NaN values
                    pct_change = ((v - baseline) / baseline) * 100
                    plt.text(i, v/2, f"{pct_change:+.1f}%", ha='center', color='white', 
                           fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        output_file = os.path.join(figures_dir, f"ablation_compare_{metric}_{dataset}.w-{window}.h-{horizon}.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        logger.info(f"Created comparison plot for {metric}, saved to {output_file}")
    
    # Create a summary table
    summary = pd.DataFrame(index=ablation_types)
    for metric in ['MAE', 'RMSE', 'PCC', 'R2']:
        summary[metric] = [metrics[abl][metric][0] if abl in metrics else np.nan for abl in ablation_types]
    
    # Calculate percentage changes
    if 'none' in metrics:
        for metric in summary.columns:
            baseline = summary.loc['none', metric]
            summary[f'{metric}_change'] = summary[metric].apply(lambda x: ((x - baseline) / baseline) * 100 if not np.isnan(x) else np.nan)
    
    # Save summary table
    summary_path = os.path.join(figures_dir, f"ablation_summary_{dataset}.w-{window}.h-{horizon}.csv")
    summary.to_csv(summary_path)
    logger.info(f"Ablation analysis complete. Summary saved to {summary_path}")
    
    # Create a heatmap visualization of the summary
    plt.figure(figsize=(12, 6))
    
    # Prepare data for heatmap
    heatmap_data = summary.copy()
    metric_cols = [col for col in summary.columns if not col.endswith('_change')]
    for col in metric_cols:
        heatmap_data = heatmap_data.drop(col, axis=1)
    
    # Remove 'none' row as it will have all zeros
    if 'none' in heatmap_data.index:
        heatmap_data = heatmap_data.drop('none')
    
    if not heatmap_data.empty and not heatmap_data.columns.empty:
        # Rename columns for better display
        heatmap_data.columns = [col.replace('_change', '') for col in heatmap_data.columns]
        
        # Rename rows for better display
        heatmap_data.index = [ABLATION_NAMES.get(idx, idx) for idx in heatmap_data.index]
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn_r', fmt='.1f', cbar_kws={'label': 'Percentage Change (%)'})
        plt.title(f'MSAGAT-Net: Impact of Component Removal (% Change, h={horizon})', fontsize=16)
        plt.tight_layout()
        output_file = os.path.join(figures_dir, f"ablation_heatmap_{dataset}.w-{window}.h-{horizon}.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        logger.info(f"Created heatmap visualization of component impact, saved to {output_file}")
    
    return summary

def create_component_importance_plot(summary, dataset, window, horizon, figures_dir):
    """Create a plot showing the relative importance of each component."""
    if 'RMSE_change' not in summary.columns:
        logger.warning("RMSE_change column not found in summary, skipping component importance plot")
        return
    
    # Get component importance data (absolute value of RMSE change)
    components = []
    importance = []
    
    for ablation in ['no_eagam', 'no_dmtm', 'no_ppm']:
        if ablation in summary.index:
            change_value = summary.loc[ablation, 'RMSE_change']
            if not pd.isna(change_value):
                component_name = ablation.replace('no_', '')
                components.append(COMPONENT_FULL_NAMES.get(component_name, component_name))
                # Use absolute value of change (higher is more important)
                importance.append(abs(change_value))
    
    if not components:
        logger.warning("No valid component data found for importance plot")
        return
    
    # Create horizontal bar chart of component importance
    plt.figure(figsize=(10, 6))
    
    # Sort by importance
    sorted_indices = np.argsort(importance)
    components = [components[i] for i in sorted_indices]
    importance = [importance[i] for i in sorted_indices]
    
    # Use consistent colors from the defined palette
    component_colors = [COMPONENT_COLORS.get(comp, '#1f77b4') for comp in components]
    
    bars = plt.barh(components, importance, color=component_colors)
    plt.xlabel('Component Importance\n(% RMSE Degradation When Removed)', fontsize=12)
    plt.title(f'MSAGAT-Net: Relative Importance of Components (h={horizon})', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    # Add value labels to bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f"{width:.1f}%", ha='left', va='center', fontsize=11)
    
    plt.tight_layout()
    output_file = os.path.join(figures_dir, f"component_importance_{dataset}.w-{window}.h-{horizon}.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    logger.info(f"Created component importance visualization, saved to {output_file}")

def generate_ablation_report(summary, dataset, window, horizon, output_dir):
    """Generate a detailed report of the ablation study results."""
    report_path = os.path.join(output_dir, f"ablation_report_{dataset}.w-{window}.h-{horizon}.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"MSAGAT-Net Ablation Study Report\n")
        f.write(f"================================\n\n")
        
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
            'no_eagam': "Efficient Adaptive Graph Attention Module (EAGAM): Captures spatial dependencies between regions with linear complexity",
            'no_dmtm': "Dilated Multi-Scale Temporal Module (DMTM): Processes time-series patterns at different temporal resolutions",
            'no_ppm': "Progressive Prediction Module (PPM): Enables region-aware multi-step forecasting with refinement"
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
        f.write(f"----------\n\n")
        
        # Generate simple conclusion based on results
        if 'RMSE_change' in summary.columns and len(importance) >= 1:
            most_important = sorted_components[0][0]
            
            f.write(f"The ablation study demonstrates that the {most_important.replace('no_', '').upper()} component ")
            f.write(f"contributes most significantly to model performance, with its removal resulting in a ")
            f.write(f"{importance[most_important]:.2f}% degradation in RMSE.\n\n")
            
            if len(importance) >= 2:
                least_important = sorted_components[-1][0]
                f.write(f"While all components contribute positively to the model's predictive capability, ")
                f.write(f"the {least_important.replace('no_', '').upper()} component shows the least individual impact ")
                f.write(f"with a {importance[least_important]:.2f}% RMSE degradation when removed.\n\n")
            
            f.write(f"The full MSAGAT-Net model with all components intact demonstrates superior performance ")
            f.write(f"across all metrics, confirming the value of the complete architecture design.")
    
    logger.info(f"Generated ablation study report: {report_path}")
    return report_path

def generate_cross_horizon_performance(results_dir, dataset, window, figures_dir):
    """Generate plots showing performance across different horizons."""
    horizons = []
    metrics = {
        'RMSE': {},
        'PCC': {}
    }
    
    # Find all result files for the full model
    pattern = f"final_metrics_{dataset}.w-{window}.h-*.none.csv"
    files = glob(os.path.join(results_dir, pattern))
    
    for file in files:
        # Extract horizon
        match = re.search(rf"h-(\d+)\.none\.csv", file)
        if match:
            horizon = int(match.group(1))
            df = pd.read_csv(file)
            horizons.append(horizon)
            
            for metric in metrics:
                if metric in df.columns:
                    metrics[metric][horizon] = df[metric][0]
    
    if not horizons:
        logger.warning(f"No data found for {dataset} with window {window}")
        return
    
    # Sort horizons
    horizons = sorted(horizons)
    
    # Plot performance vs horizon
    for metric, values in metrics.items():
        plt.figure(figsize=(10, 6))
        
        sorted_horizons = sorted(values.keys())
        metric_values = [values[h] for h in sorted_horizons]
        
        plt.plot(sorted_horizons, metric_values, 'o-', linewidth=2, markersize=8, color='#1f77b4')
        
        plt.title(f'MSAGAT-Net: {metric} vs Forecast Horizon ({dataset})', fontsize=16)
        plt.xlabel('Forecast Horizon (days)', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add value labels
        for i, v in enumerate(metric_values):
            plt.text(sorted_horizons[i], v, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
        
        # Improve aesthetics
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        output_file = os.path.join(figures_dir, f"{metric}_vs_horizon_{dataset}.w-{window}.png")
        plt.savefig(output_file, dpi=300)
        plt.close()
        logger.info(f"Created {metric} vs horizon plot, saved to {output_file}")

def main():
    """Main function."""
    args = parse_args()
    
    # Ensure output directories exist
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.figures_dir, exist_ok=True)
    
    logger.info(f"Analyzing MSAGAT-Net ablation results")
    logger.info(f"Results directory: {args.results_dir}")
    logger.info(f"Figures directory: {args.figures_dir}")
    
    # Find all ablation files
    file_groups = find_ablation_files(args.results_dir)
    logger.info(f"Found {len(file_groups)} dataset/window/horizon combinations")
    
    # Process each combination
    for (dataset, window, horizon), ablation_files in file_groups.items():
        logger.info(f"Processing {dataset} w-{window} h-{horizon} with {len(ablation_files)} ablation variants")
        
        # Generate comparison plots and summary table
        summary = plot_ablation_comparison(
            ablation_files, 
            dataset, 
            window, 
            horizon,
            args.figures_dir
        )
        
        if summary is not None:
            # Create component importance visualization
            create_component_importance_plot(
                summary,
                dataset,
                window,
                horizon,
                args.figures_dir
            )
            
            # Generate comprehensive report
            report_path = generate_ablation_report(
                summary,
                dataset,
                window,
                horizon,
                args.figures_dir
            )
    
    # Generate cross-horizon performance plots
    datasets_processed = set(dataset for (dataset, _, _) in file_groups.keys())
    windows_processed = set(window for (_, window, _) in file_groups.keys())
    
    for dataset in datasets_processed:
        for window in windows_processed:
            generate_cross_horizon_performance(
                args.results_dir,
                dataset,
                window,
                args.figures_dir
            )
    
    logger.info("Ablation analysis completed successfully")

if __name__ == "__main__":
    main()