#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Publication-Quality Visualizations for MSAGAT-Net Results

This script generates high-quality visualizations from training results for use in research papers.
It processes multiple datasets and horizons, creating:
1. Performance comparison across datasets and horizons
2. Component importance visualizations
3. Attention pattern analysis
4. Temporal dynamics visualizations

Usage:
python generate_paper_visualizations.py --output_dir paper_figures
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import json
from datetime import datetime
from collections import defaultdict
import re

# Add the src directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(current_dir, "paper_viz_log.txt")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define configuration
DATASETS = [
    ('japan', 'Japan-COVID'),
    ('state360', 'US-States'),
    ('region785', 'US-Regions'),
]

HORIZONS = [3, 5, 10, 15]

METRICS = {
    'RMSE': 'Root Mean Square Error',
    'MAE': 'Mean Absolute Error',
    'PCC': 'Pearson Correlation Coefficient',
    'R2': 'R-squared'
}

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

# Update COMPONENT_COLORS to use the full names as keys
COMPONENT_COLORS = {
    'Efficient Adaptive Graph\nAttention Module': '#1f77b4',  # Blue
    'Dilated Multi-Scale\nTemporal Module': '#ff7f0e',        # Orange
    'Progressive Prediction\nModule': '#2ca02c'               # Green
}

# Paper-quality plot settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'figure.figsize': (10, 6),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'axes.titlesize': 14,
    'axes.labelsize': 12
})

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate publication-quality visualizations')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing result files')
    parser.add_argument('--figures_dir', type=str, default='figures',
                       help='Directory for existing figures')
    parser.add_argument('--output_dir', type=str, default='paper_figures',
                       help='Directory for saving publication-quality figures')
    parser.add_argument('--window', type=int, default=20,
                       help='Window size to analyze')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=[d[0] for d in DATASETS],
                       help='Datasets to analyze')
    parser.add_argument('--horizons', type=int, nargs='+', default=HORIZONS,
                       help='Horizons to analyze')
    return parser.parse_args()

def load_metrics(results_dir, dataset, window, horizon, ablation='none'):
    """Load metrics from a CSV file."""
    # First try the detailed format
    filename = os.path.join(results_dir, f"final_metrics_{dataset}.w-{window}.h-{horizon}.{ablation}.csv")
    
    if not os.path.exists(filename):
        # Try the consolidated CSV format from the repository
        consolidated_file = os.path.join(results_dir, "metrics_MSAGATNet.csv")
        if os.path.exists(consolidated_file):
            # Load the CSV and filter for the specific configuration
            df = pd.read_csv(consolidated_file)
            filtered_df = df[(df['dataset'] == dataset) & 
                           (df['window'] == window) & 
                           (df['horizon'] == horizon)]
            
            if not filtered_df.empty:
                # Create a DataFrame in the expected format
                metrics_df = pd.DataFrame({
                    'RMSE': [filtered_df['rmse'].values[0]],
                    'MAE': [filtered_df['mae'].values[0]],
                    'PCC': [filtered_df['pcc'].values[0]],
                    'R2': [filtered_df['r2'].values[0]]
                })
                return metrics_df
        
        logger.warning(f"Metrics file not found: {filename}")
        return None
    
    logger.info(f"Loading metrics from: {filename}")
    return pd.read_csv(filename)

def load_summary(results_dir, dataset, window, horizon):
    """Load summary metrics from a CSV file."""
    filename = os.path.join(results_dir, f"ablation_summary_{dataset}.w-{window}.h-{horizon}.csv")
    if not os.path.exists(filename):
        logger.warning(f"Summary file not found: {filename}")
        return None
    
    logger.info(f"Loading summary from: {filename}")
    return pd.read_csv(filename, index_col=0)

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def generate_performance_table(results_dir, datasets, window, horizons, output_dir):
    """Generate performance table across datasets and horizons."""
    # Initialize metrics dataframe
    metrics_data = []
    
    # Collect metrics for each dataset and horizon
    for dataset_code, dataset_name in [(d, d) if isinstance(d, str) else d for d in datasets]:
        for horizon in horizons:
            metrics = load_metrics(results_dir, dataset_code, window, horizon)
            if metrics is not None:
                row = {
                    'Dataset': dataset_name,
                    'Horizon': horizon,
                    'RMSE': metrics['RMSE'][0],
                    'MAE': metrics['MAE'][0],
                    'PCC': metrics['PCC'][0],
                    'R2': metrics['R2'][0]
                }
                metrics_data.append(row)
    
    if not metrics_data:
        logger.warning("No metrics data found to generate performance table")
        return
    
    # Create DataFrame and save to CSV
    performance_df = pd.DataFrame(metrics_data)
    output_file = os.path.join(output_dir, f"performance_table_w{window}.csv")
    performance_df.to_csv(output_file, index=False, float_format='%.4f')
    logger.info(f"Performance table saved to: {output_file}")
    
    # Create a styled HTML version for easy viewing
    styled_df = performance_df.style\
        .background_gradient(subset=['RMSE', 'MAE'], cmap='Blues_r')\
        .background_gradient(subset=['PCC', 'R2'], cmap='Greens')\
        .format({'RMSE': '{:.2f}', 'MAE': '{:.2f}', 'PCC': '{:.4f}', 'R2': '{:.4f}'})
    
    html_file = os.path.join(output_dir, f"performance_table_w{window}.html")
    with open(html_file, 'w') as f:
        f.write(styled_df.to_html())
    
    return performance_df

def generate_cross_horizon_performance(results_dir, datasets, window, horizons, output_dir):
    """Generate plots showing performance trends across forecast horizons."""
    for metric in ['RMSE', 'PCC']:
        plt.figure(figsize=(10, 6))
        
        for dataset_code, dataset_name in [(d, d) if isinstance(d, str) else d for d in datasets]:
            values = []
            for horizon in horizons:
                metrics = load_metrics(results_dir, dataset_code, window, horizon)
                if metrics is not None:
                    values.append(metrics[metric][0])
                else:
                    values.append(np.nan)
            
            if not all(np.isnan(values)):
                plt.plot(horizons, values, 'o-', linewidth=2, label=dataset_name, markersize=8)
        
        plt.title(f'{METRICS[metric]} vs Forecast Horizon', fontsize=16)
        plt.xlabel('Forecast Horizon (days)', fontsize=14)
        plt.ylabel(METRICS[metric], fontsize=14)
        plt.xticks(horizons)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12, loc='best')
        
        # Add minor improvements for publication quality
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Save figure
        output_file = os.path.join(output_dir, f"{metric}_vs_horizon_w{window}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Cross-horizon {metric} plot saved to: {output_file}")

def generate_component_importance_comparison(results_dir, datasets, window, horizons, output_dir):
    """Generate plots comparing component importance across horizons."""
    component_data = []
    
    for dataset_code, dataset_name in [(d, d) if isinstance(d, str) else d for d in datasets]:
        for horizon in horizons:
            summary = load_summary(results_dir, dataset_code, window, horizon)
            if summary is None:
                continue
            
            for ablation in ['no_eagam', 'no_dmtm', 'no_ppm']:
                if ablation in summary.index and 'RMSE_change' in summary.columns:
                    component_name = ablation.replace('no_', '')
                    rmse_change = summary.loc[ablation, 'RMSE_change']
                    component_data.append({
                        'Dataset': dataset_name,
                        'Horizon': horizon,
                        'Component': COMPONENT_FULL_NAMES.get(component_name, component_name),
                        'Importance': abs(rmse_change)
                    })
    
    if not component_data:
        logger.warning("No component importance data found")
        return
    
    # Create DataFrame
    df = pd.DataFrame(component_data)
    
    # Plot component importance across horizons
    plt.figure(figsize=(12, 8))
    
    # Use seaborn for improved aesthetics
    ax = sns.barplot(x='Horizon', y='Importance', hue='Component', data=df, palette=COMPONENT_COLORS)
    
    plt.title('MSAGAT-Net: Component Importance Across Forecast Horizons', fontsize=16)
    plt.xlabel('Forecast Horizon (days)', fontsize=14)
    plt.ylabel('Component Importance\n(% RMSE Degradation When Removed)', fontsize=14)
    plt.legend(title='Component', fontsize=12, title_fontsize=13)
    
    # Add value labels to bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)
    
    # Save figure
    output_file = os.path.join(output_dir, f"component_importance_comparison_w{window}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Component importance comparison saved to: {output_file}")
    
    # Also create a heatmap version showing component importance shifts
    plt.figure(figsize=(10, 6))
    
    # Pivot data for heatmap
    pivot_df = df.pivot_table(
        index='Component', 
        columns='Horizon', 
        values='Importance',
        aggfunc='mean'  # Average across datasets
    )
    
    # Create custom diverging colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Plot heatmap
    sns.heatmap(pivot_df, annot=True, fmt='.1f', cmap=cmap, 
                linewidths=.5, cbar_kws={'label': 'Component Importance (%)'})
    
    plt.title('MSAGAT-Net: Component Importance Shift Across Forecast Horizons', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, f"component_importance_heatmap_w{window}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Component importance heatmap saved to: {output_file}")

def generate_ablation_impact_grid(results_dir, dataset, window, horizons, output_dir):
    """Generate grid of ablation impact heatmaps across horizons."""
    if len(horizons) > 1:
        fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 5), sharey=True)
        
        for i, horizon in enumerate(horizons):
            summary = load_summary(results_dir, dataset, window, horizon)
            if summary is None:
                axes[i].text(0.5, 0.5, f"No data for h={horizon}", 
                            ha='center', va='center', fontsize=14)
                continue
                
            # Prepare data for heatmap
            heatmap_data = summary.copy()
            metric_cols = [col for col in summary.columns if not col.endswith('_change')]
            for col in metric_cols:
                heatmap_data = heatmap_data.drop(col, axis=1)
            
            # Remove 'none' row as it will have all zeros
            if 'none' in heatmap_data.index:
                heatmap_data = heatmap_data.drop('none')
            
            if not heatmap_data.empty:
                # Rename columns for better display
                heatmap_data.columns = [col.replace('_change', '') for col in heatmap_data.columns]
                
                # Rename rows for better display
                heatmap_data.index = [ABLATION_NAMES.get(idx, idx) for idx in heatmap_data.index]
                
                # Create heatmap
                sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn_r', fmt='.1f', ax=axes[i],
                            cbar=(i == len(horizons)-1), cbar_kws={'label': 'Percentage Change (%)'})
                
                axes[i].set_title(f'{horizon}-Day Forecast')
            else:
                axes[i].text(0.5, 0.5, f"No ablation data for h={horizon}", 
                            ha='center', va='center', fontsize=14)
        
        plt.suptitle(f'Impact of Component Removal Across Forecast Horizons ({dataset})', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for super title
        
        # Save figure
        output_file = os.path.join(output_dir, f"ablation_impact_grid_{dataset}_w{window}.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Ablation impact grid saved to: {output_file}")

def generate_performance_comparison_grid(results_dir, dataset, window, horizons, output_dir):
    """Generate grid of performance comparison bar charts across horizons and metrics."""
    metrics_to_plot = ['RMSE', 'MAE', 'PCC', 'R2']
    ablation_types = ['none', 'no_eagam', 'no_dmtm', 'no_ppm']
    
    # Create a figure with subplots for each metric and horizon
    fig, axes = plt.subplots(len(metrics_to_plot), len(horizons), 
                            figsize=(4*len(horizons), 3*len(metrics_to_plot)),
                            sharex='col', sharey='row')
    
    # Define colors and better labels
    ablation_colors = {
        'none': '#2ca02c',      # Green for full model
        'no_eagam': '#d62728',   # Red for EAGAM ablation
        'no_dmtm': '#ff7f0e',   # Orange for DMTM ablation
        'no_ppm': '#1f77b4'    # Blue for PPM ablation
    }
    
    # Loop through metrics and horizons to create each subplot
    for i, metric in enumerate(metrics_to_plot):
        for j, horizon in enumerate(horizons):
            ax = axes[i, j]
            
            # Collect values for this metric and horizon
            values = []
            labels = []
            colors = []
            
            for ablation in ablation_types:
                metrics = load_metrics(results_dir, dataset, window, horizon, ablation)
                if metrics is not None and metric in metrics:
                    values.append(metrics[metric][0])
                    labels.append(ABLATION_NAMES.get(ablation, ablation))
                    colors.append(ablation_colors.get(ablation, 'gray'))
            
            if values:
                # Plot bars
                bars = ax.bar(labels, values, color=colors)
                
                # Add metric name to the first column
                if j == 0:
                    ax.set_ylabel(METRICS.get(metric, metric), fontsize=12)
                
                # Add horizon value to the top row
                if i == 0:
                    ax.set_title(f'{horizon}-Day Forecast', fontsize=12)
                
                # Customize appearance
                ax.grid(True, linestyle='--', alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Performance Comparison Across Ablations and Horizons ({dataset})', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for super title
    
    # Save figure
    output_file = os.path.join(output_dir, f"performance_comparison_grid_{dataset}_w{window}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Performance comparison grid saved to: {output_file}")

def enhance_existing_figures(figures_dir, output_dir):
    """Enhance existing figures with publication-quality standards."""
    # Define patterns to identify figure types
    patterns = {
        'ablation_heatmap': r'ablation_heatmap_(.+)\.w-(\d+)\.h-(\d+)\.png',
        'component_importance': r'component_importance_(.+)\.w-(\d+)\.h-(\d+)\.png',
        'ablation_compare': r'ablation_compare_(\w+)_(.+)\.w-(\d+)\.h-(\d+)\.png'
    }
    
    enhanced_count = 0
    
    # Check if figures_dir exists
    if not os.path.exists(figures_dir):
        logger.warning(f"Figures directory not found: {figures_dir}")
        return enhanced_count
    
    # Process files in the figures directory
    for filename in os.listdir(figures_dir):
        if not filename.endswith('.png'):
            continue
            
        source_path = os.path.join(figures_dir, filename)
        
        # Determine figure type and extract metadata
        fig_type = None
        metadata = {}
        
        for pattern_name, pattern in patterns.items():
            match = re.match(pattern, filename)
            if match:
                fig_type = pattern_name
                if pattern_name == 'ablation_heatmap':
                    metadata = {
                        'dataset': match.group(1),
                        'window': match.group(2),
                        'horizon': match.group(3)
                    }
                elif pattern_name == 'component_importance':
                    metadata = {
                        'dataset': match.group(1),
                        'window': match.group(2),
                        'horizon': match.group(3)
                    }
                elif pattern_name == 'ablation_compare':
                    metadata = {
                        'metric': match.group(1),
                        'dataset': match.group(2),
                        'window': match.group(3),
                        'horizon': match.group(4)
                    }
                break
        
        if fig_type:
            # Create enhanced version
            try:
                # Read original image
                img = plt.imread(source_path)
                
                # Create new figure with publication-quality settings
                fig = plt.figure(figsize=(10, 6), dpi=300)
                plt.imshow(img)
                plt.axis('off')  # Turn off axes
                
                # Add better title based on figure type and metadata
                title = ''
                if fig_type == 'ablation_heatmap':
                    title = f"Impact of Component Removal on {metadata['dataset'].upper()} Dataset (h={metadata['horizon']})"
                elif fig_type == 'component_importance':
                    title = f"Component Importance Analysis for {metadata['dataset'].upper()} Dataset (h={metadata['horizon']})"
                elif fig_type == 'ablation_compare':
                    title = f"{metadata['metric']} Comparison Across Ablations ({metadata['dataset'].upper()}, h={metadata['horizon']})"
                
                plt.title(title, fontsize=14, pad=20)
                
                # Save enhanced figure
                dest_path = os.path.join(output_dir, f"enhanced_{filename}")
                plt.savefig(dest_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                enhanced_count += 1
                logger.info(f"Enhanced figure saved to: {dest_path}")
                
            except Exception as e:
                logger.error(f"Error enhancing {filename}: {e}")
    
    return enhanced_count

def generate_overview_figure(output_dir):
    """Generate a comprehensive overview figure for the paper."""
    # This creates a multi-panel figure showcasing key aspects of MSAGAT-Net
    
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # Panel 1: Component Architecture Diagram (placeholder text)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.text(0.5, 0.5, "MSAGAT-Net Architecture\n(Replace with diagram)", 
             ha='center', va='center', fontsize=14)
    ax1.axis('off')
    
    # Panel 2: Performance Across Horizons (placeholder)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.text(0.5, 0.5, "Performance Across Horizons\n(Generated automatically)", 
             ha='center', va='center', fontsize=14)
    ax2.axis('off')
    
    # Panel 3: Component Importance (placeholder)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.text(0.5, 0.5, "Component Importance\n(Generated automatically)", 
             ha='center', va='center', fontsize=14)
    ax3.axis('off')
    
    # Panel 4: Ablation Impact (placeholder)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.text(0.5, 0.5, "Ablation Impact\n(Generated automatically)", 
             ha='center', va='center', fontsize=14)
    ax4.axis('off')
    
    plt.suptitle('MSAGAT-Net: Model Architecture and Performance Overview', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for super title
    
    # Save figure
    output_file = os.path.join(output_dir, "msagat_net_overview.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Overview figure saved to: {output_file}")

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    create_directory(args.output_dir)
    
    # Process datasets
    datasets_to_process = []
    for dataset_code in args.datasets:
        # Check if the dataset code is in our predefined list
        dataset_entry = next((d for d in DATASETS if d[0] == dataset_code), None)
        if dataset_entry:
            datasets_to_process.append(dataset_entry)
        else:
            datasets_to_process.append(dataset_code)
    
    logger.info(f"Processing datasets: {[d[0] if isinstance(d, tuple) else d for d in datasets_to_process]}")
    logger.info(f"Processing horizons: {args.horizons}")
    
    # Generate only essential visualizations
    logger.info("Generating cross-horizon performance plots...")
    generate_cross_horizon_performance(
        args.results_dir, datasets_to_process, args.window, args.horizons, args.output_dir)
    
    logger.info("Generating component importance comparison...")
    generate_component_importance_comparison(
        args.results_dir, datasets_to_process, args.window, args.horizons, args.output_dir)
    
    # Generate README with figure descriptions
    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, 'w') as f:
        f.write("# MSAGAT-Net Publication Figures\n\n")
        f.write(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Figure Descriptions\n\n")
        
        f.write("### Performance Comparisons\n")
        f.write("- **RMSE_vs_horizon_w{}.png**: Comparison of RMSE across different forecast horizons\n".format(args.window))
        f.write("- **PCC_vs_horizon_w{}.png**: Comparison of correlation (PCC) across different forecast horizons\n\n".format(args.window))
        
        f.write("### Component Analysis\n")
        f.write("- **component_importance_comparison_w{}.png**: Bar chart comparing the importance of each component across horizons\n".format(args.window))
        f.write("- **component_importance_heatmap_w{}.png**: Heatmap showing how component importance shifts across horizons\n\n".format(args.window))
    
    logger.info(f"README with figure descriptions saved to: {readme_path}")
    logger.info("Visualizations completed successfully!")

if __name__ == "__main__":
    main()