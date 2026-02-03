"""
Utility Module for MSAGAT-Net

Contains helper functions for:
    - Graph operations (Laplacian computation)
    - Error metrics (peak error)
    - Visualization (matrices, predictions, loss curves)
    - Metrics saving and logging
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error
from typing import Optional, Dict, List, Any


# =============================================================================
# GRAPH UTILITIES
# =============================================================================

def get_laplace_matrix(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute symmetric normalized Laplacian matrix.

    Args:
        adj: Adjacency matrices [batch, N, N]

    Returns:
        Normalized Laplacian [batch, N, N]
    """
    B, N, _ = adj.shape
    # Add self-loops
    I = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
    A_hat = adj + I
    # Degree matrix
    degree = A_hat.sum(dim=-1)
    # Inverse sqrt degree
    d_inv_sqrt = torch.pow(degree + 1e-5, -0.5)
    D_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    # Normalized Laplacian
    L = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return L


# =============================================================================
# ERROR METRICS
# =============================================================================

def peak_error(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """
    Compute MAE focusing on peak regions above a threshold.

    Args:
        y_true: True values [samples, nodes]
        y_pred: Predicted values [samples, nodes]
        threshold: Peak threshold per node

    Returns:
        Mean absolute error on peak regions
    """
    mask = y_true >= threshold
    if not mask.any():
        return 0.0
    true_peaks = y_true[mask]
    pred_peaks = y_pred[mask]
    return mean_absolute_error(true_peaks, pred_peaks)


# =============================================================================
# VISUALIZATION STYLE
# =============================================================================

def setup_visualization_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "grid.color": "gray",
        "grid.linewidth": 0.5,
        "axes.grid": True,
        "figure.figsize": (8, 6),
        "figure.constrained_layout.use": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,
    })


# =============================================================================
# MATRIX VISUALIZATION
# =============================================================================

def visualize_matrices(loader, model, save_path: str, device: torch.device, 
                       logger=None):
    """
    Plot adjacency, input correlation, and learned attention matrices.
    
    Args:
        loader: DataBasicLoader instance
        model: Trained model
        save_path: Path to save figure
        device: Torch device
        logger: Optional logger
    """
    model.eval()
    
    # Get adjacency matrix
    adj = getattr(loader, "adj", None)
    geo = adj.cpu().numpy() if adj is not None else np.eye(loader.m)
    
    # Compute input correlation
    raw = loader.rawdat
    corr = np.corrcoef(raw.T)
    
    # Get model attention
    batch = next(loader.get_batches(loader.test, min(32, len(loader.test[0])), shuffle=False))
    X, _, idx = batch
    X = X.to(device)
    idx = idx.to(device) if idx is not None else None
    
    with torch.no_grad():
        _, _ = model(X, idx)

    # Extract attention from model
    attn = _extract_attention(model, loader.m, device, logger)
    
    # Average over heads/batch if needed
    if isinstance(attn, np.ndarray):
        if attn.ndim == 4:
            attn = attn.mean(axis=(0, 1))
        elif attn.ndim == 3:
            attn = attn.mean(axis=0)

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    
    for ax, mat, title in zip(
        axes,
        [geo, corr, attn],
        ["Adjacency", "Input Correlation", "Learned Attention"],
    ):
        im = ax.imshow(mat, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("Node")
        ax.set_ylabel("Node")
        plt.colorbar(im, ax=ax)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    
    if logger:
        logger.info(f"Matrices saved to {save_path}")


def _extract_attention(model, num_nodes: int, device: torch.device, 
                       logger=None) -> np.ndarray:
    """Extract attention weights from various model architectures."""
    attn = None

    # MSTAGAT_Net with learned low-rank graph bias
    if hasattr(model, "spatial_module"):
        sm = model.spatial_module
        if hasattr(sm, "u") and hasattr(sm, "v"):
            u = sm.u.detach()
            v = sm.v.detach()
            graph_bias = torch.matmul(u, v)
            attn = graph_bias.mean(dim=0).cpu().numpy()
        elif hasattr(sm, "attn"):
            attn = sm.attn.detach().cpu().numpy()
        elif hasattr(sm, "attention"):
            attn = sm.attention_weights.detach().cpu().numpy()

    # Models with graph_attention attribute
    elif hasattr(model, "graph_attention"):
        ga = model.graph_attention
        if hasattr(ga, "u") and hasattr(ga, "v"):
            u = ga.u.detach()
            v = ga.v.detach()
            graph_bias = torch.matmul(u, v)
            attn = graph_bias.mean(dim=0).cpu().numpy()
        elif hasattr(ga, "attn"):
            attn_obj = ga.attn
            if isinstance(attn_obj, list):
                attn_list = [a.detach().cpu().numpy() for a in attn_obj if hasattr(a, 'detach')]
                if len(attn_list) == 1:
                    attn = attn_list[0]
                elif len(attn_list) > 1:
                    attn = np.mean(np.stack(attn_list), axis=0)
            elif hasattr(attn_obj, 'detach'):
                attn = attn_obj.detach().cpu().numpy()

    # Fallback
    if attn is None:
        attn = np.zeros((num_nodes, num_nodes))
        if logger:
            logger.warning("No attention weights found; using zeros")

    return attn


# =============================================================================
# PREDICTION VISUALIZATION
# =============================================================================

def _compute_region_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute metrics for a single region."""
    from scipy.stats import pearsonr
    
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Handle constant arrays for PCC
    if np.std(y_true) < 1e-8 or np.std(y_pred) < 1e-8:
        pcc = 0.0
    else:
        pcc, _ = pearsonr(y_true, y_pred)
    
    return {'rmse': rmse, 'mae': mae, 'pcc': pcc}


def visualize_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: str,
                          regions: int = 6, logger=None, dataset_name: str = None,
                          region_names: List[str] = None):
    """
    Create publication-quality prediction plots with time series comparison.
    
    Args:
        y_true: Ground truth [timesteps, nodes]
        y_pred: Predictions [timesteps, nodes]
        save_path: Path to save figure
        regions: Number of regions to plot (default 6 for 2x3 grid)
        logger: Optional logger
        dataset_name: Optional dataset name for title
        region_names: Optional list of region names
    """
    # Setup publication style
    plt.rcdefaults()
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
    })
    
    n_nodes = y_true.shape[1]
    n_regions = min(regions, n_nodes)
    timesteps = np.arange(y_true.shape[0])
    
    # Select regions with highest variance (most interesting to visualize)
    variances = np.var(y_true, axis=0)
    selected_indices = np.argsort(variances)[-n_regions:][::-1]
    
    # Determine grid layout
    if n_regions <= 3:
        nrows, ncols = 1, n_regions
        figsize = (4.5 * n_regions, 3.5)
    elif n_regions <= 6:
        nrows, ncols = 2, 3
        figsize = (14, 7)
    else:
        nrows, ncols = 3, 3
        figsize = (14, 10)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    # Colors for publication
    color_true = '#1f77b4'  # Blue
    color_pred = '#d62728'  # Red
    
    for plot_idx, region_idx in enumerate(selected_indices):
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        true_vals = y_true[:, region_idx]
        pred_vals = y_pred[:, region_idx]
        
        # Compute metrics for this region
        metrics = _compute_region_metrics(true_vals, pred_vals)
        
        # Plot ground truth
        ax.plot(timesteps, true_vals, 
                color=color_true, 
                linewidth=1.5, 
                label='Ground Truth',
                zorder=2)
        
        # Plot prediction
        ax.plot(timesteps, pred_vals, 
                color=color_pred, 
                linewidth=1.5, 
                linestyle='--',
                label='Prediction',
                alpha=0.9,
                zorder=3)
        
        # Fill between to show error
        ax.fill_between(timesteps, true_vals, pred_vals,
                        color='gray', alpha=0.15, zorder=1)
        
        # Region name
        if region_names and region_idx < len(region_names):
            region_label = region_names[region_idx]
        else:
            region_label = f'Region {region_idx + 1}'
        
        # Title with metrics
        ax.set_title(f'{region_label}\nRMSE={metrics["rmse"]:.3f}, PCC={metrics["pcc"]:.3f}',
                     fontsize=10, fontweight='normal')
        
        # Axis labels
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        
        # Light grid
        ax.grid(True, linestyle=':', alpha=0.4, zorder=0)
        
        # Legend only on first subplot
        if plot_idx == 0:
            ax.legend(loc='upper right', frameon=True, framealpha=0.9,
                      edgecolor='gray', fancybox=False)
    
    # Hide unused subplots
    for idx in range(len(selected_indices), len(axes)):
        axes[idx].set_visible(False)
    
    # Overall title
    if dataset_name:
        fig.suptitle(f'{dataset_name} - Prediction Results', fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    
    if logger:
        logger.info(f"Predictions saved to {save_path}")


def visualize_predictions_summary(y_true: np.ndarray, y_pred: np.ndarray, save_path: str,
                                   logger=None, dataset_name: str = None):
    """
    Create a single summary plot showing aggregated predictions across all regions.
    Useful for showing overall model performance.
    
    Args:
        y_true: Ground truth [timesteps, nodes]
        y_pred: Predictions [timesteps, nodes]
        save_path: Path to save figure
        logger: Optional logger
        dataset_name: Optional dataset name for title
    """
    # Setup publication style
    plt.rcdefaults()
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    timesteps = np.arange(y_true.shape[0])
    
    # Compute aggregated values (mean across regions)
    true_mean = np.mean(y_true, axis=1)
    pred_mean = np.mean(y_pred, axis=1)
    true_std = np.std(y_true, axis=1)
    pred_std = np.std(y_pred, axis=1)
    
    # Compute overall metrics
    overall_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    overall_mae = np.mean(np.abs(y_true - y_pred))
    
    # Per-timestep RMSE across regions
    from scipy.stats import pearsonr
    flat_true = y_true.flatten()
    flat_pred = y_pred.flatten()
    if np.std(flat_true) > 1e-8 and np.std(flat_pred) > 1e-8:
        overall_pcc, _ = pearsonr(flat_true, flat_pred)
    else:
        overall_pcc = 0.0
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Colors
    color_true = '#1f77b4'
    color_pred = '#d62728'
    
    # Left plot: Mean prediction with confidence bands
    ax1 = axes[0]
    ax1.plot(timesteps, true_mean, color=color_true, linewidth=2, label='Ground Truth (mean)')
    ax1.fill_between(timesteps, true_mean - true_std, true_mean + true_std,
                     color=color_true, alpha=0.2, label='±1 std')
    ax1.plot(timesteps, pred_mean, color=color_pred, linewidth=2, linestyle='--', 
             label='Prediction (mean)')
    ax1.fill_between(timesteps, pred_mean - pred_std, pred_mean + pred_std,
                     color=color_pred, alpha=0.15)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value (averaged across regions)')
    ax1.set_title('Aggregated Predictions')
    ax1.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle=':', alpha=0.4)
    
    # Right plot: Error distribution over time
    ax2 = axes[1]
    errors = np.abs(y_true - y_pred)
    error_mean = np.mean(errors, axis=1)
    error_std = np.std(errors, axis=1)
    
    ax2.plot(timesteps, error_mean, color='#2ca02c', linewidth=2, label='Mean Abs. Error')
    ax2.fill_between(timesteps, 
                     np.maximum(0, error_mean - error_std), 
                     error_mean + error_std,
                     color='#2ca02c', alpha=0.2, label='±1 std')
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Prediction Error Over Time')
    ax2.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle=':', alpha=0.4)
    
    # Add metrics text box
    metrics_text = f'Overall Metrics:\nRMSE = {overall_rmse:.4f}\nMAE = {overall_mae:.4f}\nPCC = {overall_pcc:.4f}'
    ax2.text(0.02, 0.98, metrics_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Overall title
    if dataset_name:
        fig.suptitle(f'{dataset_name} - Summary', fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    
    if logger:
        logger.info(f"Summary predictions saved to {save_path}")


# =============================================================================
# LOSS CURVES
# =============================================================================

def plot_loss_curves(train_losses: List[float], val_losses: List[float], 
                     save_path: str, args=None, logger=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save plot
        args: Command line arguments (optional)
        logger: Logger object (optional)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Mark best epoch
    best_val_epoch = val_losses.index(min(val_losses)) + 1
    best_val_loss = min(val_losses)
    ax.scatter(best_val_epoch, best_val_loss, color='green', s=100, zorder=5,
               label=f'Best Val Loss: {best_val_loss:.6f}')
               
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    
    # Add title with model info if args provided
    if args:
        title = f'Training Progress\nDataset: {args.dataset}, Window: {args.window}, Horizon: {args.horizon}'
        ax.set_title(title, fontsize=14, pad=10)
        
        textstr = f'Learning Rate: {args.lr}\nBatch Size: {args.batch}\nBest Epoch: {best_val_epoch}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    else:
        ax.set_title('Training Progress', fontsize=14, pad=10)
            
    ax.legend(loc='upper right', frameon=True, framealpha=0.8)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    if logger:
        logger.info(f"Loss curves saved to {save_path}")


# =============================================================================
# METRICS SAVING
# =============================================================================

ALL_RESULTS_CSV = "all_results.csv"
ALL_RESULTS_TXT = "all_results.txt"


def save_metrics(metrics: Dict, save_path: str, dataset: str = None,
                 window: int = None, horizon: int = None, logger=None,
                 model_name: str = None, ablation: str = None, seed: int = None,
                 use_adj_prior: bool = False):
    """
    Append metrics to consolidated CSV and TXT files in dataset-specific folder.
    
    Args:
        metrics: Dictionary of metric values
        save_path: Path for individual file (used to extract directory)
        dataset: Dataset name
        window: Window size
        horizon: Prediction horizon
        logger: Logger object
        model_name: Model name
        ablation: Ablation variant
        seed: Random seed used for experiment
        use_adj_prior: Whether adjacency prior was used
    """
    # Use the dataset-specific directory from save_path
    results_dir = os.path.dirname(save_path)
    os.makedirs(results_dir, exist_ok=True)
    
    # Build record
    info = {
        "model": model_name or "MSTAGAT",
        "dataset": dataset or "",
        "window": window or 0,
        "horizon": horizon or 0,
        "ablation": ablation or "none",
        "seed": seed or 42,
        "use_adj": use_adj_prior,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    
    data = {
        **info,
        **{k: v for k, v in metrics.items() if not isinstance(v, np.ndarray)},
    }
    
    # Append to consolidated CSV
    csv_path = os.path.join(results_dir, ALL_RESULTS_CSV)
    df = pd.DataFrame([data])
    
    if os.path.exists(csv_path):
        df_old = pd.read_csv(csv_path)
        # Add seed column if it doesn't exist (backward compatibility)
        if 'seed' not in df_old.columns:
            df_old['seed'] = 42  # Default seed for old entries
        # Add use_adj column if it doesn't exist (backward compatibility)
        if 'use_adj' not in df_old.columns:
            df_old['use_adj'] = False  # Default for old entries
        # Remove duplicate entries (now including seed and use_adj)
        mask = ~(
            (df_old['dataset'] == dataset) & 
            (df_old['window'] == window) & 
            (df_old['horizon'] == horizon) & 
            (df_old['ablation'] == (ablation or "none")) &
            (df_old['seed'] == (seed or 42)) &
            (df_old['use_adj'] == use_adj_prior)
        )
        df_old = df_old[mask]
        df = pd.concat([df_old, df], ignore_index=True)
    
    # Sort and save - only include seed in sort if column exists
    sort_cols = ['dataset', 'window', 'horizon', 'ablation', 'seed', 'use_adj']
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    
    # Append to text log with header
    txt_path = os.path.join(results_dir, ALL_RESULTS_TXT)
    
    # Add header if file doesn't exist or is empty
    if not os.path.exists(txt_path) or os.path.getsize(txt_path) == 0:
        header = (
            "# MSTAGAT-Net Experiment Results\n"
            "# Format: [Timestamp] Model | Dataset, Window, Horizon, Ablation, Seed | Metrics\n"
            "# " + "="*80 + "\n"
        )
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(header)
    
    txt_line = (
        f"[{info['timestamp']}] {info['model']} | "
        f"Dataset: {info['dataset']}, Window: {info['window']}, "
        f"Horizon: {info['horizon']}, Ablation: {info['ablation']}, Seed: {info['seed']} | "
        f"MAE: {metrics.get('mae', 0):.4f}, RMSE: {metrics.get('rmse', 0):.4f}, "
        f"PCC: {metrics.get('pcc', 0):.4f}, R2: {metrics.get('R2', 0):.4f}\n"
    )
    with open(txt_path, 'a', encoding='utf-8') as f:
        f.write(txt_line)
    
    if logger:
        logger.info(f"Metrics appended to {csv_path}")
        logger.info(f"Log appended to {txt_path}")
