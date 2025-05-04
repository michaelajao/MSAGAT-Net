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
BASE_DIR = os.path.dirname(SCRIPT_DIR)
METRICS_DIR = os.path.join(BASE_DIR, 'report', 'results')
OUT_DIR = os.path.join(BASE_DIR, 'report', 'paper_figures')
DEFAULT_WINDOW = 20

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
    'rmse': 'Root Mean Square Error',
    'mae': 'Mean Absolute Error',
    'pcc': 'Pearson Correlation Coefficient',
    'r2':  'R-squared'
}

# Component full names for legends
COMP_FULL = {
    'agam': 'Low-Rank Adaptive Graph Attention Module',
    'mtfm': 'Multi-scale Temporal Fusion Module',
    'pprm': 'Progressive Prediction Refinement Module'
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


def find_metrics_file(dataset: str, horizon: int, ablation: str) -> str:
    patterns = [
        f"final_metrics_*{dataset}*w-{DEFAULT_WINDOW}*h-{horizon}*{ablation}.csv",
        f"final_metrics_*{dataset}*horizon-{horizon}*{ablation}.csv",
    ]
    for pat in patterns:
        matches = glob.glob(os.path.join(METRICS_DIR, pat))
        if matches:
            return matches[0]
    return None


def load_metrics(dataset: str, horizon: int, ablation: str) -> pd.DataFrame:
    path = find_metrics_file(dataset, horizon, ablation)
    return pd.read_csv(path) if path and os.path.exists(path) else None


def load_summary(dataset: str, horizon: int) -> pd.DataFrame:
    path = os.path.join(METRICS_DIR, f"ablation_summary_{dataset}.w-{DEFAULT_WINDOW}.h-{horizon}.csv")
    return pd.read_csv(path, index_col=0) if os.path.exists(path) else None

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def setup_style():
    """Configure the matplotlib and seaborn styling for publication-quality visualizations."""
    # Reset to defaults first to avoid any unexpected styling
    plt.rcdefaults()
    
    # Set the color palette - using a scientific publication-friendly palette
    # Using a colorblind-friendly palette with improved contrast that works well in print
    colors = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161", 
              "#3690c0", "#41ab5d", "#807dba", "#e6550d"]
    sns.set_palette(colors)
    
    # Use a clean, minimal style suitable for academic publications
    sns.set_style('ticks', {'axes.grid': False})
    
    # Configure figure parameters for publication quality
    plt.rcParams['figure.dpi'] = 300      # Higher DPI for sharper display
    plt.rcParams['savefig.dpi'] = 600     # High DPI for publication-quality figures
    plt.rcParams['figure.figsize'] = (8, 6)  # Standard size for single-column figures
    
    # Typography settings for academic publications
    plt.rcParams['font.family'] = 'serif'  # Serif fonts are standard in academic publishing
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman']
    plt.rcParams['font.size'] = 10         # Standard size for academic papers
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    
    # Enhanced legend styling for publications
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams['legend.frameon'] = False  # Cleaner look without frames
    plt.rcParams['legend.markerscale'] = 1.2
    plt.rcParams['legend.handlelength'] = 1.5
    
    # Line and marker styling for better visibility in publications
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6
    plt.rcParams['lines.markeredgewidth'] = 1.0
    
    # Other settings for better layout - avoid constrained_layout to prevent conflicts with tight_layout
    plt.rcParams['figure.constrained_layout.use'] = False  # Disable to avoid conflict with tight_layout
    plt.rcParams['figure.autolayout'] = False  # Let us control the layout explicitly
    plt.rcParams['axes.axisbelow'] = True  # grid lines behind data
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    
    # Tick styling for academic publications
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['ytick.major.width'] = 1.0
    
    # Clean white background for publication
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Specific settings for publication-quality figures
    plt.rcParams['figure.titlesize'] = 12
    plt.rcParams['figure.titleweight'] = 'bold'
    plt.rcParams['axes.formatter.useoffset'] = False


def save_fig(fig, name: str, custom_dir=None, add_timestamp=False, formats=None, 
            quality=100, add_watermark=False, transparent_bg=False):
    """
    Save figure in multiple formats with enhanced options for publication-quality.
    
    Args:
        fig: The matplotlib figure to save
        name: Base name for the saved file
        custom_dir: Optional custom directory (default is OUT_DIR)
        add_timestamp: If True, adds a timestamp to prevent overwriting
        formats: List of formats to save (default: png, pdf, eps)
        quality: JPEG quality (0-100) if jpg/jpeg format is selected
        add_watermark: Whether to add a subtle watermark
        transparent_bg: Whether to use transparent background for formats that support it
    """
    save_dir = custom_dir if custom_dir else OUT_DIR
    ensure_dir(save_dir)
    
    # Add timestamp to filename if requested to prevent overwriting
    if add_timestamp:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{name}_{timestamp}"
    
    # Default formats for research papers
    if formats is None:
        formats = ('png', 'pdf', 'eps', 'svg')  # EPS is commonly used for LaTeX papers
    
    # Save in specified formats with publication-quality settings
    for ext in formats:
        filepath = os.path.join(save_dir, f"{name}.{ext}")
        
        # Format-specific options for publication quality
        kwargs = {
            'bbox_inches': 'tight',
            'transparent': transparent_bg,
            'facecolor': 'none' if transparent_bg else 'white',
            'dpi': 600,  # High DPI for publication quality
            'metadata': {'Creator': 'MSAGAT-Net'}
        }
        
        # Add format-specific options
        if ext.lower() in ('jpg', 'jpeg'):
            kwargs['quality'] = quality
        elif ext.lower() == 'svg':
            kwargs['format'] = 'svg'
        elif ext.lower() == 'eps':
            kwargs['format'] = 'eps'
            
        # Save the figure
        fig.savefig(filepath, **kwargs)
    
    # Print info about saved file
    print(f"  - Saved {name}.{', '.join(formats)}")


def performance_table(visualize=True):
    """
    Create a comprehensive performance table and optionally visualize it.
    
    Args:
        visualize: If True, creates a heatmap visualization of the performance table
    
    Returns:
        pd.DataFrame: Table with performance metrics
    """
    rows = []
    for ds, horizons in DATASET_CONFIGS:
        for h in horizons:
            df = load_metrics(ds, h, 'none')
            if df is None: continue
            rows.append({
                'Dataset': ds.upper(),
                'Horizon': h,
                **{METRIC_NAMES[m]: get_metric(df, m) for m in METRICS}
            })
            
    if not rows:
        print("No performance data available")
        return None
        
    # Create the table DataFrame
    table = pd.DataFrame(rows)
    
    # Save the CSV
    csv_path = os.path.join(OUT_DIR, 'performance_table.csv')
    table.to_csv(csv_path, index=False)
    print(f"Performance table saved to {csv_path}")
    
    # Create a styled version for rendering in notebooks or reports
    styled_csv_path = os.path.join(OUT_DIR, 'performance_table_styled.csv')
    
    # Round numeric columns to 4 decimal places for better readability
    styled_table = table.copy()
    for m in METRIC_NAMES.values():
        if m in styled_table.columns:
            styled_table[m] = styled_table[m].round(4)
    
    styled_table.to_csv(styled_csv_path, index=False)
    
    # Optional visualization
    if visualize:
        setup_style()
        
        # Create multiple visualizations of the table
        # 1. Heatmap for each metric to visualize patterns
        metrics_to_plot = [METRIC_NAMES[m] for m in METRICS]
        
        # Create a figure with a heatmap for each metric
        fig, axes = plt.subplots(len(metrics_to_plot), 1, 
                               figsize=(12, 5 * len(metrics_to_plot)),
                               sharex=True)
        
        if len(metrics_to_plot) == 1:
            axes = [axes]  # Convert to list if only one subplot
        
        for i, metric in enumerate(metrics_to_plot):
            if metric not in table.columns:
                continue
                
            # Create pivot table for the heatmap
            pivot = table.pivot_table(
                index='Dataset', 
                columns='Horizon',
                values=metric,
                aggfunc='mean'
            )
            
            # Define color map based on metric (RMSE/MAE: lower is better, PCC/R2: higher is better)
            if metric in ['Root Mean Square Error', 'Mean Absolute Error']:
                cmap = 'YlOrRd_r'  # Reversed colormap (green is good/low)
                title_suffix = "(lower is better)"
            else:
                cmap = 'YlGn'  # Standard colormap (green is good/high)
                title_suffix = "(higher is better)"
            
            # Create heatmap with publication styling
            sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, 
                       linewidths=.5, ax=axes[i], cbar_kws={'shrink': 0.7})
            
            # Simplified title for publication
            axes[i].set_title(f'{metric}', fontsize=11, pad=10)
            
        plt.tight_layout()
        save_fig(fig, 'performance_heatmaps')
        
        # 2. Create a radar chart for comparing datasets across metrics
        # Normalize each metric to 0-1 scale for fair comparison
        radar_data = table.copy()
        
        # Create a new figure for the radar chart - publication quality
        radar_fig = plt.figure(figsize=(7, 7))
        radar_ax = radar_fig.add_subplot(111, polar=True)
        
        # Get unique datasets
        datasets = radar_data['Dataset'].unique()
        
        # For each dataset, create a radar chart
        for ds in datasets:
            ds_data = radar_data[radar_data['Dataset'] == ds]
            
            # Average across horizons
            ds_avg = ds_data.groupby('Dataset').mean().reset_index()
            
            # Extract just the metrics
            metric_values = [ds_avg[METRIC_NAMES[m]].values[0] for m in METRICS]
            
            # Need to handle RMSE and MAE differently (lower is better)
            # Normalize so that best value = 1 for all metrics
            normalized_values = []
            for i, m in enumerate(METRICS):
                val = metric_values[i]
                if m in ['rmse', 'mae']:
                    # For these metrics, lower is better so invert
                    min_val = table[METRIC_NAMES[m]].min()
                    max_val = table[METRIC_NAMES[m]].max()
                    if max_val - min_val != 0:
                        normalized = 1 - ((val - min_val) / (max_val - min_val))
                    else:
                        normalized = 1
                else:
                    # For these metrics, higher is better
                    min_val = table[METRIC_NAMES[m]].min()
                    max_val = table[METRIC_NAMES[m]].max() 
                    if max_val - min_val != 0:
                        normalized = (val - min_val) / (max_val - min_val)
                    else:
                        normalized = 1
                
                normalized_values.append(normalized)
            
            # Close the loop for the radar chart
            values = normalized_values + [normalized_values[0]]
            
            # Define the angles for each metric
            angles = [n / float(len(METRICS)) * 2 * np.pi for n in range(len(METRICS))]
            angles += angles[:1]  # Close the loop
            
            # Plot the radar chart
            radar_ax.plot(angles, values, linewidth=2, linestyle='solid', label=ds, marker='o', markersize=8)
            radar_ax.fill(angles, values, alpha=0.1)
        
        # Set the labels and customize the chart
        labels = [METRIC_NAMES[m] for m in METRICS]
        radar_ax.set_xticks(angles[:-1])
        radar_ax.set_xticklabels(labels, fontsize=12)
        
        # Draw y-axis labels
        radar_ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        radar_ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10, alpha=0.75)
        
        # Add legend and title with cleaner style for publication
        radar_ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        radar_fig.suptitle('Comparative Performance Across Metrics', 
                         fontsize=12, fontweight='bold')
        
        # Save the radar chart
        save_fig(radar_fig, 'performance_radar')
        
        plt.close('all')
    
    return table


def cross_horizon_performance():
    """Generate enhanced line plots showing model performance across different prediction horizons."""
    setup_style()
    
    # Define markers and line styles for consistent visual identity with better distinction
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*']
    linestyles = ['-', '--', '-.', ':']
    
    # Custom colors for datasets to ensure consistency across different plots
    dataset_colors = {
        'JAPAN': '#1f77b4',
        'REGION785': '#ff7f0e',
        'NHS_TIMESERIES': '#2ca02c',
        'LTLA_TIMESERIES': '#d62728'
    }
    
    # Iterate through metrics to create separate plots with enhanced styling
    for m in ['rmse', 'pcc']:
        # Create a larger figure for better readability
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Track min/max values for annotations
        all_vals = []
        legend_items = []
        
        # For each dataset, plot performance across horizons
        for idx, (ds, horizons) in enumerate(DATASET_CONFIGS):
            vals, hs = [], []
            for h in horizons:
                try:
                    df = load_metrics(ds, h, 'none')
                    if df is None: continue
                    val = get_metric(df, m)
                    if not np.isnan(val): 
                        vals.append(val)
                        hs.append(h)
                except Exception as e:
                    print(f"Error processing {ds} with horizon {h}: {e}")
            
            if vals:
                # Use consistent marker and line style based on dataset index
                marker = markers[idx % len(markers)]
                linestyle = linestyles[idx % len(linestyles)]
                color = dataset_colors.get(ds.upper(), None)  # Use consistent colors
                
                # Plot with enhanced styling and bigger markers for visibility
                line = ax.plot(hs, vals, marker=marker, linestyle=linestyle, 
                         linewidth=2.5, markersize=10, label=ds.upper(),
                         color=color, markeredgecolor='black', markeredgewidth=1.0,
                         alpha=0.9)
                
                # Add data labels to the points with cleaner styling
                for x, y in zip(hs, vals):
                    ax.annotate(f'{y:.3f}', 
                               (x, y), 
                               textcoords="offset points",
                               xytext=(0, 10), 
                               ha='center',
                               fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
                               zorder=100)  # Ensure labels are on top
                
                all_vals.extend(vals)
                legend_items.append(line[0])
        
        # Set concise plot title for publication
        title = f"{METRIC_NAMES[m]}"
        ax.set_title(f"{title}", fontsize=11, pad=10)
        
        # Cleaner axis labels for publication
        ax.set_xlabel('Prediction Horizon', fontsize=10)
        ax.set_ylabel(METRIC_NAMES[m], fontsize=10)
        
        # Subtle grid for academic style
        ax.grid(False)
        
        # Show data range to give context
        if all_vals:
            y_min, y_max = min(all_vals), max(all_vals)
            y_range = y_max - y_min
            ax.set_ylim([max(0, y_min - 0.1 * y_range), y_max + 0.15 * y_range])
        
        # Add x-ticks if they're not already clearly shown
        all_horizons = sorted(list(set([h for _, horizons in DATASET_CONFIGS for h in horizons])))
        ax.set_xticks(all_horizons)
        
        # Enhanced legend with better positioning and style
        legend = ax.legend(loc='best', frameon=True, framealpha=0.95, 
                          edgecolor='gray', fancybox=True, shadow=True)
                          
        # No explanatory text for publication-ready figures
        
        # Save with consistent naming
        save_fig(fig, f"{m}_vs_horizon")
        plt.close(fig)


def component_importance_comparison():
    """
    Create enhanced bar charts showing the importance of each component 
    across different horizons based on RMSE change.
    """
    setup_style()
    data = []
    
    # Collect and prepare data
    for ds, horizons in DATASET_CONFIGS:
        for h in horizons:
            sum_df = load_summary(ds, h)
            if sum_df is None: continue
            for ab in ABLATIONS[1:]:
                col = 'RMSE_CHANGE'
                if col in sum_df.columns and ab in sum_df.index:
                    try:
                        # Safe extraction of value with error handling
                        value = sum_df.loc[ab, col]
                        if pd.isna(value):
                            continue
                            
                        comp = ab.replace('no_', '')
                        if comp in COMP_FULL:  # Make sure component name exists
                            data.append({
                                'Dataset': ds.upper(),
                                'Horizon': h,
                                'Component': COMP_FULL[comp],
                                'ComponentShort': comp.upper(),  # Short name for more compact displays
                                'Importance': abs(value)
                            })
                    except (KeyError, TypeError) as e:
                        print(f"Error processing {ab} for {ds}, horizon {h}: {e}")
                        continue
    
    if not data:
        print("No data available for component importance comparison")
        return
        
    df = pd.DataFrame(data)
    
    # Create a larger figure with improved styling
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), 
                                   gridspec_kw={'height_ratios': [1, 1]})
    
    # Use a gradient color palette for better visual distinction
    palette = sns.color_palette("viridis", len(df['Component'].unique()))
    
    # 1. By horizon visualization with improved styling
    horizon_plot = sns.barplot(x='Horizon', y='Importance', hue='Component', 
                data=df, ax=ax1, palette=palette, 
                edgecolor='black', linewidth=1, errorbar=None)
    
    # Add value labels only for significant bars (publication-style)
    for p in ax1.patches:
        height = p.get_height()
        if height > 1.0:  # Only label significant bars
            ax1.text(p.get_x() + p.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha="center", fontsize=8)
    
    # Enhanced styling for first subplot - more suitable for academic publication
    ax1.set_ylabel('% RMSE Increase', fontsize=11)
    ax1.set_title('Component Importance by Horizon', 
                 fontsize=12, pad=10)
    
    # Clean legend for publication
    legend1 = ax1.legend(title='Component', title_fontsize=10,
                        loc='upper right', frameon=False)
    
    # 2. By dataset visualization - group by dataset instead of horizon
    sns.barplot(x='Dataset', y='Importance', hue='ComponentShort', 
                data=df, ax=ax2, palette='Blues_d',
                edgecolor='black', linewidth=0.8, errorbar=None)
    
    # Add value labels only for significant bars
    for p in ax2.patches:
        height = p.get_height()
        if height > 1.0:  # Only label significant bars
            ax2.text(p.get_x() + p.get_width()/2., height + 0.1,
                    f'{height:.1f}%', ha="center", fontsize=8)
    
    # Enhanced styling for second subplot
    ax2.set_ylabel('% RMSE Change', fontsize=14, fontweight='bold')
    ax2.set_title('Component Importance Across Different Datasets', 
                 fontsize=16, fontweight='bold', pad=20)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Enhance legend for second plot
    legend2 = ax2.legend(title='Model Component', title_fontsize=14,
                       loc='upper right', frameon=True, fancybox=True, 
                       framealpha=0.9, shadow=True)
    
    # No explanatory text boxes for publication-ready figures
    
    # Add tight layout and save with both views
    plt.tight_layout()
    save_fig(fig, 'component_importance_comparison')
    
    # Create a heatmap variation for better visualization of patterns
    plt.figure(figsize=(12, 8))
    
    # Pivot data for the heatmap
    if 'Component' in df.columns and 'Horizon' in df.columns and 'Importance' in df.columns:
        heatmap_data = df.pivot_table(
            values='Importance', 
            index='Component', 
            columns='Horizon', 
            aggfunc='mean'
        ).fillna(0)
        
        # Create the heatmap with better visual appeal
        fig_heat, ax_heat = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                   linewidths=.5, ax=ax_heat, cbar_kws={'label': '% RMSE Change'})
        
        # Enhance the heatmap styling
        ax_heat.set_title('Component Importance Heatmap by Horizon', 
                         fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        save_fig(fig_heat, 'component_importance_heatmap')
    
    plt.close('all')


def single_horizon_metrics():
    """
    Create enhanced bar charts comparing RMSE across different ablation settings
    for each dataset and horizon combination.
    """
    setup_style()
    
    # Define a better color palette for the ablations
    ablation_colors = {
        'none': '#1f77b4',       # Blue for complete model
        'no_agam': '#ff7f0e',    # Orange for no AGAM
        'no_mtfm': '#2ca02c',    # Green for no MTFM
        'no_pprm': '#d62728'     # Red for no PPRM
    }
    
    # Better label mapping for ablations
    ablation_labels = {
        'none': 'Full Model',
        'no_agam': 'No Graph\nAttention',
        'no_mtfm': 'No Temporal\nFusion',
        'no_pprm': 'No Prediction\nRefinement'
    }
    
    # Process each dataset and horizon combination
    for ds, horizons in DATASET_CONFIGS:
        # Use a more descriptive dataset name
        ds_name = ds.upper()
        
        for h in horizons:
            # Collect data
            vals, labels, colors = [], [], []
            ablation_data = {}
            
            # Get values for each ablation
            for ab in ABLATIONS:
                df = load_metrics(ds, h, ab)
                if df is None: continue
                
                # Get both RMSE and PCC for comparison
                rmse_val = get_metric(df, 'rmse')
                pcc_val = get_metric(df, 'pcc')
                
                if not np.isnan(rmse_val): 
                    ablation_data[ab] = {
                        'rmse': rmse_val,
                        'pcc': pcc_val if not np.isnan(pcc_val) else 0
                    }
                    vals.append(rmse_val)
                    labels.append(ablation_labels.get(ab, ab))
                    colors.append(ablation_colors.get(ab, '#333333'))
            
            if not vals:
                continue
            
            # Create figure with two subplots side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
            
            # 1. RMSE Bar Chart (left side)
            bars = ax1.bar(labels, vals, color=colors, edgecolor='black', linewidth=1)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))
            
            # Styling for RMSE chart
            ax1.set_title(f'RMSE Comparison', fontsize=15, fontweight='bold')
            ax1.set_ylabel('RMSE (lower is better)', fontsize=12, fontweight='bold')
            ax1.set_ylim(0, max(vals) * 1.15)  # Add space for labels
            
            # Add baseline reference line for the full model
            if 'none' in ablation_data:
                baseline = ablation_data['none']['rmse']
                ax1.axhline(y=baseline, color='navy', linestyle='--', alpha=0.7)
                ax1.text(len(labels)-1, baseline, f'Baseline: {baseline:.3f}', 
                        color='navy', va='bottom', ha='right', fontsize=9)
            
            # Add RMSE percentage differences as annotations
            if 'none' in ablation_data:
                baseline = ablation_data['none']['rmse']
                for i, ab in enumerate([ab for ab in ABLATIONS if ab in ablation_data]):
                    if ab != 'none':
                        pct_diff = ((ablation_data[ab]['rmse'] - baseline) / baseline) * 100
                        ax1.text(i, ablation_data[ab]['rmse'] / 2, 
                               f"{'+' if pct_diff > 0 else ''}{pct_diff:.1f}%", 
                               color='white', ha='center', fontweight='bold', fontsize=10)
            
            # 2. Normalized Comparison Chart (right side) - Show relative changes from baseline
            # Create a normalized view to better visualize the relative impact
            metrics_to_plot = ['rmse', 'pcc']
            metrics_labels = {'rmse': 'RMSE Impact', 'pcc': 'PCC Impact'}
            metrics_colors = {'rmse': 'tab:red', 'pcc': 'tab:blue'}
            metrics_markers = {'rmse': 'o', 'pcc': 's'}
            
            # Calculate normalized values and prepare grouped bar chart
            normalized_data = []
            for ab in [a for a in ABLATIONS if a in ablation_data]:
                for m in metrics_to_plot:
                    if m in ablation_data[ab] and 'none' in ablation_data and m in ablation_data['none']:
                        baseline_val = ablation_data['none'][m]
                        if baseline_val != 0:
                            if m == 'rmse':
                                # For RMSE, higher is worse
                                change = ((ablation_data[ab][m] - baseline_val) / baseline_val) * 100
                            else:
                                # For PCC, lower is worse
                                change = ((baseline_val - ablation_data[ab][m]) / baseline_val) * 100
                                
                            normalized_data.append({
                                'Ablation': ablation_labels.get(ab, ab),
                                'Metric': metrics_labels[m],
                                'Change (%)': change,
                                'Color': metrics_colors[m]
                            })
            
            if normalized_data:
                norm_df = pd.DataFrame(normalized_data)
                
                # Create a grouped bar chart
                bar_width = 0.35
                
                # Filter out 'Full Model' for the plot
                plot_ablations = [abl for abl in norm_df['Ablation'].unique() if abl != 'Full Model']
                index = np.arange(len(plot_ablations))
                
                for i, metric in enumerate(['RMSE Impact', 'PCC Impact']):
                    metric_data = norm_df[norm_df['Metric'] == metric]
                    # Filter out 'Full Model'
                    data = metric_data[metric_data['Ablation'] != 'Full Model']
                    offset = i * bar_width - bar_width/2
                    
                    bars = ax2.bar(index + offset, data['Change (%)'], 
                                  bar_width, label=metric,
                                  color=data['Color'].iloc[0] if not data.empty else metrics_colors[metric.split()[0].lower()],
                                  alpha=0.7, edgecolor='black', linewidth=1)
                    
                    # Add data labels
                    for bar in bars:
                        height = bar.get_height()
                        y_pos = height + 1 if height > 0 else height - 5
                        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                                f'{height:.1f}%', ha='center', va='bottom')
                
                ax2.set_xticks(index)
                ax2.set_xticklabels(plot_ablations)
                ax2.set_ylabel('Performance Impact (%)', fontsize=12, fontweight='bold')
                ax2.set_title('Component Impact on Performance', fontsize=15, fontweight='bold')
                ax2.legend(title='Metric')
                
                # Add a zero reference line
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
            # Master title for the entire figure - more concise for publication
            fig.suptitle(f'{ds_name} (H={h})', fontsize=14, fontweight='bold', y=0.98)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for the suptitle
            save_fig(fig, f'rmse_comp_{ds}_h{h}')
            
            # Also create individual figures for each plot (for use in papers)
            # 1. RMSE comparison as a standalone
            fig_rmse = plt.figure(figsize=(5, 4))
            ax_rmse = fig_rmse.add_subplot(111)
            
            # Recreate the RMSE bar chart
            bars = ax_rmse.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.8)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax_rmse.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Styling for RMSE chart
            ax_rmse.set_title(f'RMSE Comparison: {ds_name} (H={h})', fontsize=11, pad=10)
            ax_rmse.set_ylabel('RMSE', fontsize=10)
            
            # Add baseline reference line for the full model
            if 'none' in ablation_data:
                baseline = ablation_data['none']['rmse']
                ax_rmse.axhline(y=baseline, color='navy', linestyle='--', alpha=0.5)
            
            fig_rmse.tight_layout()
            save_fig(fig_rmse, f'rmse_only_{ds}_h{h}')
            plt.close(fig_rmse)
            
            # 2. Component impact as a standalone
            if normalized_data:
                fig_impact = plt.figure(figsize=(5, 4))
                ax_impact = fig_impact.add_subplot(111)
                
                # Create a grouped bar chart
                bar_width = 0.35
                index = np.arange(len(plot_ablations))
                
                for i, metric in enumerate(['RMSE Impact', 'PCC Impact']):
                    metric_data = norm_df[norm_df['Metric'] == metric]
                    data = metric_data[metric_data['Ablation'] != 'Full Model']
                    offset = i * bar_width - bar_width/2
                    
                    bars = ax_impact.bar(index + offset, data['Change (%)'], 
                              bar_width, label=metric,
                              color=data['Color'].iloc[0] if not data.empty else metrics_colors[metric.split()[0].lower()],
                              alpha=0.8, edgecolor='black', linewidth=0.8)
                
                ax_impact.set_xticks(index)
                ax_impact.set_xticklabels(plot_ablations)
                ax_impact.set_ylabel('Performance Impact (%)', fontsize=10)
                ax_impact.set_title(f'Component Impact: {ds_name} (H={h})', fontsize=11, pad=10)
                ax_impact.legend(fontsize=8)
                ax_impact.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                
                fig_impact.tight_layout()
                save_fig(fig_impact, f'component_impact_{ds}_h{h}')
                plt.close(fig_impact)
            plt.close(fig)


def component_contribution_charts():
    """
    Create enhanced visualization showing component contributions across datasets,
    with multiple visualization types for better insights.
    """
    setup_style()
    metrics = ['rmse', 'pcc']
    
    # Component abbreviations for better labeling
    comp_abbrev = {
        'Low-Rank Adaptive Graph Attention Module': 'AGAM',
        'Multi-scale Temporal Fusion Module': 'MTFM',
        'Progressive Prediction Refinement Module': 'PPRM'
    }
    
    for m in metrics:
        data = []
        # Collect data
        for ds, horizons in DATASET_CONFIGS:
            for h in horizons:
                sum_df = load_summary(ds, h)
                if sum_df is None: continue
                for ab in ABLATIONS[1:]:
                    col = f'{m.upper()}_CHANGE'
                    if col in sum_df.columns and ab in sum_df.index:
                        comp = ab.replace('no_', '')
                        data.append({
                            'Dataset': ds.upper(), 
                            'Horizon': h,
                            'Component': COMP_FULL[comp], 
                            'ComponentShort': comp_abbrev.get(COMP_FULL[comp], comp),
                            'Contribution': abs(sum_df.loc[ab,col])
                        })
        
        if not data:
            print(f"No data available for {m} component contribution")
            continue
            
        df = pd.DataFrame(data)
        
        # Create a multi-view figure with different visualizations
        fig = plt.figure(figsize=(18, 15))
        
        # Set up a 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Bar plot by component - averaged across datasets
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Group by component and calculate mean and std
        comp_summary = df.groupby('ComponentShort')['Contribution'].agg(['mean', 'std']).reset_index()
        comp_summary = comp_summary.sort_values('mean', ascending=False)
        
        # Plot with error bars
        bars = ax1.bar(comp_summary['ComponentShort'], comp_summary['mean'], 
                      yerr=comp_summary['std'],
                      capsize=10, error_kw={'capthick': 2, 'ecolor': 'black'},
                      color=sns.color_palette('viridis', 3), 
                      edgecolor='black', linewidth=1.5)
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.2f}%', ha='center', va='bottom', fontsize=11,
                   fontweight='bold', bbox=dict(boxstyle='round', alpha=0.8, 
                                               facecolor='white', edgecolor='gray'))
        
        ax1.set_title(f'Average Component Contribution to {METRIC_NAMES[m]}', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('% Change in Performance', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, comp_summary['mean'].max() * 1.2)
        ax1.set_xticklabels(comp_summary['ComponentShort'], rotation=0, fontsize=11)
        
        # 2. Stacked bar chart by dataset
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Pivot data for stacked bars
        pivot_df = df.pivot_table(index='Dataset', columns='ComponentShort', 
                                values='Contribution', aggfunc='mean').fillna(0)
        pivot_df.plot(kind='bar', stacked=True, ax=ax2, colormap='viridis',
                     edgecolor='black', linewidth=0.8)
        
        # Enhance stacked bar chart
        ax2.set_title(f'Component Contribution by Dataset', 
                     fontsize=14, fontweight='bold')
        ax2.set_ylabel('% Change in Performance', fontsize=12, fontweight='bold')
        ax2.legend(title='Component', title_fontsize=12, 
                  frameon=True, fancybox=True, framealpha=0.9,
                  loc='upper right', bbox_to_anchor=(1, 0.95))
        
        # Add total values on top of stacked bars
        for i, dataset in enumerate(pivot_df.index):
            total = pivot_df.loc[dataset].sum()
            ax2.text(i, total + 0.5, f'Total: {total:.2f}%', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle='round', alpha=0.8, facecolor='white', edgecolor='gray'))
        
        # 3. Heatmap showing component contribution across datasets and horizons
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Prepare heatmap data
        if len(df['Horizon'].unique()) > 1:
            heatmap_data = df.pivot_table(
                index='ComponentShort', 
                columns=['Dataset', 'Horizon'], 
                values='Contribution',
                aggfunc='mean'
            ).fillna(0)
            
            # Create a more sophisticated heatmap
            sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                       linewidths=.5, ax=ax3, cbar_kws={'label': '% Change'})
            ax3.set_title(f'Component Contribution Heatmap', 
                         fontsize=14, fontweight='bold')
            
            # Rotate x-tick labels for better readability
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            # Alternative visualization if no horizon variation
            alternative_data = df.pivot_table(
                index='ComponentShort', 
                columns='Dataset', 
                values='Contribution',
                aggfunc='mean'
            ).fillna(0)
            
            sns.heatmap(alternative_data, annot=True, fmt='.2f', cmap='YlOrRd',
                       linewidths=.5, ax=ax3, cbar_kws={'label': '% Change'})
            ax3.set_title(f'Component Contribution by Dataset', 
                         fontsize=14, fontweight='bold')
        
        # 4. Radar/spider chart for multi-dimensional comparison
        ax4 = fig.add_subplot(gs[1, 1], polar=True)
        
        # Prepare radar chart data - use datasets as dimensions and components as categories
        if len(df['Dataset'].unique()) >= 3:  # Need at least 3 dimensions for a meaningful radar plot
            radar_data = df.pivot_table(
                index='ComponentShort', 
                columns='Dataset', 
                values='Contribution',
                aggfunc='mean'
            ).fillna(0)
            
            # Number of variables
            categories = list(radar_data.columns)
            N = len(categories)
            
            # Create angles for each category
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Draw the radar chart for each component
            for i, comp in enumerate(radar_data.index):
                values = radar_data.loc[comp].values.flatten().tolist()
                values += values[:1]  # Close the loop
                
                # Draw the plot
                ax4.plot(angles, values, linewidth=2, linestyle='solid', 
                        label=comp, marker='o', markersize=8)
                ax4.fill(angles, values, alpha=0.1)
            
            # Customize radar chart
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(categories, fontsize=10)
            ax4.set_yticklabels([])
            
            # Draw axis lines for each angle and label
            ax4.set_rlabel_position(0)
            ax4.grid(True)
            
            # Add a legend
            ax4.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        else:
            # Alternative: simple pie chart if radar chart not applicable
            comp_summary = df.groupby('ComponentShort')['Contribution'].mean()
            ax4.pie(comp_summary, labels=comp_summary.index, autopct='%1.1f%%',
                   shadow=True, startangle=90, explode=[0.05]*len(comp_summary))
            ax4.set_title('Overall Component Contribution Share', fontsize=14)
            ax4.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            
        # Concise title suitable for publication
        fig.suptitle(f'Component Contributions to {METRIC_NAMES[m]}', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Save the multi-view figure
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        save_fig(fig, f'component_contribution_{m}')
        
        # Also create a simpler version focused just on the bar chart for papers
        simple_fig, simple_ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='ComponentShort', y='Contribution', hue='Dataset', 
                   data=df, ax=simple_ax, palette='tab10',
                   edgecolor='black', linewidth=1)
        
        simple_ax.set_title(f'Component Contribution to {METRIC_NAMES[m]}', 
                          fontsize=16, fontweight='bold')
        simple_ax.set_ylabel('% Change in Performance', fontsize=14, fontweight='bold')
        simple_ax.set_xlabel('Component', fontsize=14, fontweight='bold')
        
        # Enhance legend
        simple_ax.legend(title='Dataset', title_fontsize=12, 
                       frameon=True, fancybox=True, framealpha=0.9)
        
        # Save the simpler version too
        plt.tight_layout()
        save_fig(simple_fig, f'component_contribution_{m}_simple')
        
        plt.close('all')


def interactive_comparison():
    """
    Creates an interactive visualization using Plotly to compare 
    model performance across datasets and horizons.
    
    This function creates an HTML file that can be opened in a browser
    for interactive exploration of the results.
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly is not installed. Run 'pip install plotly' to use this function.")
        return
        
    # Collect all performance data
    all_data = []
    for ds, horizons in DATASET_CONFIGS:
        for h in horizons:
            for ab in ABLATIONS:
                df = load_metrics(ds, h, ab)
                if df is None: continue
                
                # Get all metrics
                metrics_data = {m: get_metric(df, m) for m in METRICS}
                
                # Only add if we have valid data
                if not all(np.isnan(v) for v in metrics_data.values()):
                    all_data.append({
                        'Dataset': ds.upper(),
                        'Horizon': h,
                        'Ablation': ab,
                        'AblationName': ab.replace('no_', 'No ').replace('none', 'Full Model').title(),
                        **metrics_data
                    })
    
    if not all_data:
        print("No data available for interactive visualization")
        return
        
    # Convert to DataFrame
    interactive_df = pd.DataFrame(all_data)
    
    # Create subplot figure with tabs for different metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[METRIC_NAMES[m] for m in METRICS],
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]],
        horizontal_spacing=0.1,
        vertical_spacing=0.2
    )
    
    # Position mapping for subplots
    positions = {
        'rmse': (1, 1),
        'mae': (1, 2),
        'pcc': (2, 1),
        'r2': (2, 2)
    }
    
    # Color mapping for ablations
    ablation_colors = {
        'none': '#1f77b4',      # Blue for complete model
        'no_agam': '#ff7f0e',   # Orange for no AGAM
        'no_mtfm': '#2ca02c',   # Green for no MTFM
        'no_pprm': '#d62728'    # Red for no PPRM
    }
    
    # Populate each subplot
    for m, (row, col) in positions.items():
        # Create grouped bar chart
        for ds in sorted(interactive_df['Dataset'].unique()):
            ds_data = interactive_df[interactive_df['Dataset'] == ds]
            
            # Group by horizon and ablation
            for h in sorted(ds_data['Horizon'].unique()):
                h_data = ds_data[ds_data['Horizon'] == h]
                
                # Create labels
                labels = [f"{row['AblationName']} (H={row['Horizon']})" 
                        for _, row in h_data.iterrows()]
                
                # Add bars
                fig.add_trace(
                    go.Bar(
                        x=[f"{ds}-H{h}"] * len(h_data),
                        y=h_data[m],
                        name=f"{ds}-H{h}",
                        text=h_data['AblationName'],
                        hovertemplate="<b>%{text}</b><br>" +
                                     f"{METRIC_NAMES[m]}: %{{y:.4f}}<br>" +
                                     "Dataset: %{x}<br>" +
                                     "Horizon: {h}<extra></extra>",
                        marker_color=[ablation_colors.get(ab, '#333333') for ab in h_data['Ablation']],
                        showlegend=False
                    ),
                    row=row, col=col
                )
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Interactive Model Performance Comparison",
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        height=800,
        showlegend=False,
        template="plotly_white",
    )
    
    # Add axis titles to each subplot
    metric_pos = {m: pos for m, pos in positions.items()}
    for m, (row, col) in metric_pos.items():
        # Y-axis title depends on the metric
        ylabel = METRIC_NAMES[m]
        
        direction = ""
        if m in ['rmse', 'mae']:
            direction = "(lower is better)"
        elif m in ['pcc', 'r2']:
            direction = "(higher is better)"
            
        fig.update_yaxes(title_text=f"{ylabel} {direction}", row=row, col=col)
        
        # X-axis title only on bottom plots
        if row == 2:
            fig.update_xaxes(title_text="Dataset and Horizon", row=row, col=col)
            
        # Rotate x-axis labels for readability
        fig.update_xaxes(tickangle=45, row=row, col=col)
    
    # Make annotations clearer
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=16, color="#000000")
    
    # Save to HTML file
    html_path = os.path.join(OUT_DIR, 'interactive_comparison.html')
    fig.write_html(html_path, include_plotlyjs='cdn', full_html=True)
    print(f"  - Interactive visualization saved to: {html_path}")
    
    # Also create a summary figure
    summary_fig = px.box(
        interactive_df, 
        x="Dataset", 
        y="rmse", 
        color="AblationName",
        facet_col="Horizon",
        title="RMSE Distribution Across Datasets and Horizons",
        labels={"rmse": "RMSE (lower is better)"},
        height=600
    )
    
    summary_html_path = os.path.join(OUT_DIR, 'performance_summary.html')
    summary_fig.write_html(summary_html_path, include_plotlyjs='cdn', full_html=True)
    print(f"  - Performance summary saved to: {summary_html_path}")
    
    return interactive_df


def create_correlation_matrix():
    """
    Create a correlation matrix visualization to show relationships between metrics.
    
    This helps understand how different metrics relate to each other across datasets and horizons.
    """
    setup_style()
    
    # Collect all metrics data
    all_data = []
    for ds, horizons in DATASET_CONFIGS:
        for h in horizons:
            df = load_metrics(ds, h, 'none')
            if df is None: continue
            
            metrics_data = {m: get_metric(df, m) for m in METRICS}
            if not all(np.isnan(v) for v in metrics_data.values()):
                metrics_data['Dataset'] = ds.upper()
                metrics_data['Horizon'] = h
                all_data.append(metrics_data)
    
    if not all_data:
        print("No data available for correlation matrix")
        return
    
    # Create DataFrame and calculate correlation
    corr_df = pd.DataFrame(all_data)
    
    # Only include metric columns for correlation
    metric_cols = [METRIC_NAMES[m] for m in METRICS]
    available_metrics = [col for col in metric_cols if col in corr_df.columns]
    
    # Calculate correlation matrix
    corr_matrix = corr_df[available_metrics].corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Create correlation matrix visualization
    plt.figure(figsize=(10, 8))
    
    # Use diverging colormap with clear distinction around 0
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    # Create heatmap with improved styling
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5,
                cbar_kws={"shrink": .8, "label": "Correlation Coefficient"})
    
    plt.title("Correlation Matrix of Performance Metrics", 
              fontsize=16, fontweight='bold', pad=20)
    
    # Add explanatory annotation
    plt.figtext(0.5, 0.01, 
               "Positive correlations indicate metrics that tend to improve together.\n"
               "Negative correlations indicate opposing metrics (one improves when the other worsens).",
               ha="center", fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    # Save the correlation matrix
    save_fig(plt.gcf(), "metrics_correlation_matrix")
    
    # Create dataset-specific correlation matrices if we have enough data
    dataset_groups = corr_df.groupby('Dataset')
    for ds_name, ds_group in dataset_groups:
        if len(ds_group) >= 3:  # Need at least 3 points for a meaningful correlation
            ds_corr = ds_group[available_metrics].corr()
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(ds_corr, mask=np.triu(np.ones_like(ds_corr, dtype=bool)),
                       cmap=cmap, vmax=1, vmin=-1, center=0,
                       annot=True, fmt=".2f", square=True, linewidths=.5)
            
            plt.title(f"Correlation Matrix for {ds_name} Dataset",
                     fontsize=14, fontweight='bold')
            
            # Save dataset-specific correlation matrix
            save_fig(plt.gcf(), f"correlation_matrix_{ds_name.lower()}")
    
    plt.close('all')


def create_3d_visualization():
    """
    Creates a 3D visualization showing the relationship between RMSE, PCC, and prediction horizons.
    
    This provides an intuitive view of how prediction quality changes across horizons.
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("Could not import 3D plotting tools - skipping 3D visualization")
        return
    
    setup_style()
    
    # Collect data for 3D visualization
    data_3d = []
    for ds, horizons in DATASET_CONFIGS:
        for h in horizons:
            df = load_metrics(ds, h, 'none')
            if df is None: continue
            
            rmse = get_metric(df, 'rmse')
            pcc = get_metric(df, 'pcc')
            
            if not np.isnan(rmse) and not np.isnan(pcc):
                data_3d.append({
                    'Dataset': ds.upper(),
                    'Horizon': h,
                    'RMSE': rmse,
                    'PCC': pcc
                })
    
    if not data_3d:
        print("No data available for 3D visualization")
        return
    
    # Convert to DataFrame
    df_3d = pd.DataFrame(data_3d)
    
    # Create a figure for 3D visualization
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define a color map for different datasets
    datasets = df_3d['Dataset'].unique()
    colormap = plt.cm.tab10
    colors = [colormap(i) for i in range(len(datasets))]
    
    # Create a scatter plot for each dataset
    for i, ds in enumerate(datasets):
        ds_data = df_3d[df_3d['Dataset'] == ds]
        
        # Plot 3D scatter
        sc = ax.scatter(
            ds_data['Horizon'], 
            ds_data['RMSE'], 
            ds_data['PCC'],
            color=colors[i],
            s=100,
            alpha=0.8,
            edgecolors='black',
            label=ds
        )
        
        # Add connecting lines to show trends for each dataset
        ax.plot(
            ds_data['Horizon'],
            ds_data['RMSE'],
            ds_data['PCC'],
            color=colors[i],
            alpha=0.6,
            linestyle='--'
        )
        
        # Add text labels with horizon information
        for _, row in ds_data.iterrows():
            ax.text(
                row['Horizon'], 
                row['RMSE'], 
                row['PCC'],
                f"H={int(row['Horizon'])}",
                size=9,
                zorder=1,
                color='black',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
            )
    
    # Set labels and title
    ax.set_xlabel('Prediction Horizon', fontweight='bold', labelpad=10)
    ax.set_ylabel('RMSE (lower is better)', fontweight='bold', labelpad=10)
    ax.set_zlabel('PCC (higher is better)', fontweight='bold', labelpad=10)
    ax.set_title('3D Performance Visualization\nRMSE vs PCC vs Horizon', 
                fontsize=16, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add a legend
    ax.legend(title="Datasets", bbox_to_anchor=(1.05, 0.5), loc='center left')
    
    # Adjust the viewing angle for better visibility
    ax.view_init(elev=25, azim=45)
    
    # Save the visualization
    save_fig(fig, "3d_performance_visualization")
    
    # Also create a 2D version showing the RMSE-PCC relationship
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Create a scatter plot for each dataset
    for i, ds in enumerate(datasets):
        ds_data = df_3d[df_3d['Dataset'] == ds]
        
        # Use horizon as size parameter for the scatter points
        sizes = ds_data['Horizon'] * 20
        
        # Plot scatter with horizon as size
        ax2.scatter(
            ds_data['RMSE'], 
            ds_data['PCC'],
            s=sizes,
            color=colors[i],
            alpha=0.8,
            edgecolors='black',
            label=f"{ds}"
        )
        
        # Add text labels with horizon information
        for _, row in ds_data.iterrows():
            ax2.annotate(
                f"H={int(row['Horizon'])}",
                (row['RMSE'], row['PCC']),
                xytext=(5, 0),
                textcoords='offset points',
                size=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
            )
    
    # Set labels and title for 2D plot
    ax2.set_xlabel('RMSE (lower is better)', fontweight='bold')
    ax2.set_ylabel('PCC (higher is better)', fontweight='bold')
    ax2.set_title('Performance Relationship: RMSE vs PCC', 
                fontsize=16, fontweight='bold')
    
    # Add grid for better readability
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # Add a legend
    ax2.legend(title="Datasets", bbox_to_anchor=(1.05, 0.5), loc='center left')
    
    # Add explanatory text
    fig2.text(0.5, 0.01, 
             "Bubble size represents prediction horizon: larger bubbles = longer horizons\n"
             "Ideal performance is toward the bottom right (low RMSE, high PCC)",
             ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    # Save the 2D visualization
    save_fig(fig2, "rmse_pcc_relationship")
    
    plt.close('all')


# =============================================================================
# MAIN
# =============================================================================

def create_summary_dashboard():
    """
    Creates a summary dashboard that combines key visualizations into a single HTML file.
    
    This dashboard serves as a central location to view all key performance metrics
    and analyses in one place, with links to interactive visualizations.
    """
    try:
        import base64
    except ImportError:
        print("Base64 module not available, cannot create summary dashboard")
        return
    
    # Create an HTML file that shows all the key visualizations in a dashboard layout
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>MSAGAT-Net Performance Dashboard</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }
            .dashboard {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                grid-gap: 20px;
                margin-bottom: 30px;
            }
            .dashboard-item {
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                transition: transform 0.2s, box-shadow 0.2s;
            }
            .dashboard-item:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.15);
            }
            .full-width {
                grid-column: span 2;
            }
            h1, h2 {
                color: #2c3e50;
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
                font-size: 32px;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            h2 {
                margin-top: 0;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            img {
                max-width: 100%;
                height: auto;
                display: block;
                margin: 0 auto;
                border-radius: 5px;
            }
            .button-container {
                text-align: center;
                margin: 30px 0;
            }
            .button {
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 12px 25px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                text-decoration: none;
                font-size: 16px;
                margin: 0 10px;
                transition: background-color 0.2s;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .button:hover {
                background-color: #2980b9;
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }
            .footer {
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #7f8c8d;
                font-size: 14px;
            }
            .description {
                color: #666;
                font-style: italic;
                margin-top: 5px;
                margin-bottom: 15px;
            }
        </style>
    </head>
    <body>
        <div class="full-width">
            <h1>MSAGAT-Net Performance Dashboard</h1>
            <p class="description" style="text-align: center;">Comprehensive visualization of model performance and component analyses</p>
            
            <div class="button-container">
                <a class="button" href="interactive_comparison.html" target="_blank">Interactive Comparison</a>
                <a class="button" href="performance_summary.html" target="_blank">Performance Summary</a>
            </div>
        </div>
        
        <div class="dashboard">
            <div class="dashboard-item">
                <h2>Performance Across Horizons</h2>
                <p class="description">RMSE performance for each dataset at different prediction horizons</p>
                <img src="rmse_vs_horizon.png" alt="RMSE vs Horizon">
            </div>
            
            <div class="dashboard-item">
                <h2>Correlation Performance</h2>
                <p class="description">Pearson correlation (PCC) at different prediction horizons</p>
                <img src="pcc_vs_horizon.png" alt="PCC vs Horizon">
            </div>
            
            <div class="dashboard-item">
                <h2>Component Importance</h2>
                <p class="description">Relative importance of different model components</p>
                <img src="component_importance_heatmap.png" alt="Component Importance">
            </div>
            
            <div class="dashboard-item">
                <h2>Component Contribution</h2>
                <p class="description">How each component contributes to overall model performance</p>
                <img src="component_contribution_rmse_simple.png" alt="Component Contribution">
            </div>
            
            <div class="dashboard-item full-width">
                <h2>Performance Heatmaps</h2>
                <p class="description">Comprehensive view of model performance across datasets and horizons</p>
                <img src="performance_heatmaps.png" alt="Performance Heatmaps">
            </div>
        </div>
        
        <div class="footer">
            <p>MSAGAT-Net Performance Visualization Dashboard | Generated on DATE_PLACEHOLDER</p>
        </div>
    </body>
    </html>
    """
    
    # Add current date to the dashboard
    from datetime import datetime
    current_date = datetime.now().strftime("%B %d, %Y")
    html_content = html_content.replace("DATE_PLACEHOLDER", current_date)
    
    # Create the HTML file
    dashboard_path = os.path.join(OUT_DIR, 'dashboard.html')
    with open(dashboard_path, 'w') as f:
        f.write(html_content)
    
    print(f"  - Summary dashboard created at {dashboard_path}")
    return dashboard_path


def generate_publication_figures():
    """
    Creates publication-ready individual figures with specific sizes
    that match standard journal column widths.
    
    This function generates a set of carefully sized and formatted figures
    that are ready for direct inclusion in research papers.
    """
    setup_style()
    print("\n7. Creating publication-ready figures...")
    
    # Standard sizes for publication figures
    single_column = (3.5, 2.7)  # inches - standard single column width
    double_column = (7.0, 5.0)  # inches - standard double column width
    
    # 1. Create optimized cross-horizon performance figure (single column)
    for metric in ['rmse', 'pcc']:
        fig, ax = plt.subplots(figsize=single_column)
        
        # Define consistent markers and line styles
        markers = ['o', 's', 'D', '^']
        linestyles = ['-', '--', '-.', ':']
        
        # Custom colors for datasets
        dataset_colors = {
            'JAPAN': '#1f77b4',
            'REGION785': '#ff7f0e',
            'NHS_TIMESERIES': '#2ca02c',
            'LTLA_TIMESERIES': '#d62728'
        }
        
        # Plot data
        for idx, (ds, horizons) in enumerate(DATASET_CONFIGS):
            vals, hs = [], []
            for h in horizons:
                df = load_metrics(ds, h, 'none')
                if df is None: continue
                val = get_metric(df, metric)
                if not np.isnan(val): 
                    vals.append(val)
                    hs.append(h)
            
            if vals:
                color = dataset_colors.get(ds.upper(), None)
                ax.plot(hs, vals, marker=markers[idx % len(markers)], 
                       linestyle=linestyles[idx % len(linestyles)],
                       linewidth=1.2, markersize=5, label=ds.upper(),
                       color=color, markeredgecolor='black', markeredgewidth=0.5)
        
        # Remove annotations for cleaner look
        ax.set_xlabel('Prediction Horizon', fontsize=9)
        ax.set_ylabel(METRIC_NAMES[metric], fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc='best')
        ax.grid(False)
        
        plt.tight_layout()
        save_fig(fig, f"pub_{metric}_vs_horizon", formats=['pdf', 'eps'])
        plt.close(fig)
    
    # 2. Create optimized component impact figure (single column)
    data = []
    for ds, horizons in DATASET_CONFIGS:
        for h in horizons:
            sum_df = load_summary(ds, h)
            if sum_df is None: continue
            for ab in ABLATIONS[1:]:
                col = 'RMSE_CHANGE'
                if col in sum_df.columns and ab in sum_df.index:
                    try:
                        value = sum_df.loc[ab, col]
                        if pd.isna(value): continue
                        comp = ab.replace('no_', '')
                        if comp in COMP_FULL:
                            data.append({
                                'Dataset': ds.upper(),
                                'Horizon': h,
                                'Component': comp.upper(),
                                'Importance': abs(value)
                            })
                    except Exception:
                        continue
    
    if data:
        df = pd.DataFrame(data)
        
        # Create publication-ready component importance figure
        fig, ax = plt.subplots(figsize=single_column)
        
        # Create a cleaner grouped bar chart
        sns.barplot(x='Component', y='Importance', data=df, 
                   ax=ax, palette='Blues',
                   edgecolor='black', linewidth=0.5, errorbar=None)
        
        ax.set_ylabel('% RMSE Impact', fontsize=9)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
        ax.tick_params(axis='y', labelsize=7)
        
        plt.tight_layout()
        save_fig(fig, "pub_component_impact", formats=['pdf', 'eps'])
        plt.close(fig)
    
    # 3. Create optimized radar chart (single column)
    performance_data = []
    for ds, horizons in DATASET_CONFIGS:
        for h in horizons:
            row_data = {'Dataset': ds.upper(), 'Horizon': h}
            df = load_metrics(ds, h, 'none')
            if df is None: continue
            
            for m in METRICS:
                val = get_metric(df, m)
                if not np.isnan(val):
                    row_data[METRIC_NAMES[m]] = val
            
            if len(row_data) > 2:  # if we have metrics beyond Dataset and Horizon
                performance_data.append(row_data)
    
    if performance_data:
        radar_df = pd.DataFrame(performance_data)
        
        # Create radar chart for each dataset
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, polar=True)
        
        for ds in radar_df['Dataset'].unique():
            ds_data = radar_df[radar_df['Dataset'] == ds]
            
            # Average across horizons
            ds_avg = ds_data.groupby('Dataset').mean().reset_index()
            
            # Get normalized values for each metric
            metric_values = []
            for m in METRICS:
                key = METRIC_NAMES[m]
                if key in ds_avg.columns:
                    val = ds_avg[key].values[0]
                    
                    # Get min/max for normalization
                    min_val = radar_df[key].min()
                    max_val = radar_df[key].max()
                    
                    # Normalize based on metric type
                    if m in ['rmse', 'mae']:
                        norm_val = 1 - ((val - min_val) / (max_val - min_val)) if max_val != min_val else 0.5
                    else:
                        norm_val = ((val - min_val) / (max_val - min_val)) if max_val != min_val else 0.5
                        
                    metric_values.append(norm_val)
            
            # Close the loop
            values = metric_values + [metric_values[0]]
            
            # Define angles
            angles = [n / float(len(METRICS)) * 2 * np.pi for n in range(len(METRICS))]
            angles += angles[:1]
            
            # Plot
            ax.plot(angles, values, linewidth=1.2, label=ds)
            ax.fill(angles, values, alpha=0.1)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([METRIC_NAMES[m] for m in METRICS], fontsize=7)
        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(True, alpha=0.3)
        
        # Remove yticks for cleaner look
        ax.set_yticks([])
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=7, frameon=False)
        
        plt.tight_layout()
        save_fig(fig, "pub_performance_radar", formats=['pdf', 'eps'])
        plt.close(fig)
    
    print("  - Publication-ready figures created successfully")


def main():
    """Main function to generate all visualizations."""
    # Start with a welcome message
    print("="*80)
    print("MSAGAT-Net Visualization Generator")
    print("="*80)
    print(f"Generating visualizations in: {OUT_DIR}")
    
    # Ensure output directory exists
    ensure_dir(OUT_DIR)
    
    # Generate all visualizations
    print("\n1. Creating performance table...")
    performance_table(visualize=True)
    
    print("\n2. Generating cross-horizon performance charts...")
    cross_horizon_performance()
    
    print("\n3. Analyzing component importance...")
    component_importance_comparison()
    
    print("\n4. Creating single horizon metrics comparisons...")
    single_horizon_metrics()
    
    print("\n5. Generating component contribution charts...")
    component_contribution_charts()
    
    print("\n6. Creating interactive visualizations...")
    try:
        interactive_comparison()
    except Exception as e:
        print(f"Warning: Could not create interactive visualizations. Error: {e}")
        print("To enable interactive visualizations, run: pip install plotly")
    
    print("\n7. Creating summary dashboard...")
    try:
        create_summary_dashboard()
    except Exception as e:
        print(f"Warning: Could not create summary dashboard. Error: {e}")
    
    print("\n"+"="*80)
    print(f"Visualization generation complete. Results saved in: {OUT_DIR}")
    print("="*80)
    interactive_comparison()

if __name__ == '__main__':
    main()
