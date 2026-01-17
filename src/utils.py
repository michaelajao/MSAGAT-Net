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

def visualize_predictions(y_true: np.ndarray, y_pred: np.ndarray, save_path: str,
                          regions: int = 5, logger=None):
    """
    Plot ground truth vs. predictions for selected regions.
    
    Args:
        y_true: Ground truth [timesteps, nodes]
        y_pred: Predictions [timesteps, nodes]
        save_path: Path to save figure
        regions: Number of regions to plot
        logger: Optional logger
    """
    n_regions = min(regions, y_true.shape[1])
    timesteps = np.arange(y_true.shape[0])
    
    fig, axes = plt.subplots(n_regions, 1, figsize=(12, 3 * n_regions), sharex=True)
    if n_regions == 1:
        axes = [axes]
        
    for i in range(n_regions):
        axes[i].plot(timesteps, y_true[:, i], label="Ground Truth", alpha=0.7)
        axes[i].plot(timesteps, y_pred[:, i], label="Prediction", alpha=0.7)
        axes[i].set_title(f"Region {i}")
        axes[i].set_ylabel("Value")
        axes[i].grid(True, linestyle='--', alpha=0.5)
        axes[i].legend()
        
    axes[-1].set_xlabel("Time Step")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    
    if logger:
        logger.info(f"Predictions saved to {save_path}")


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
                 model_name: str = None, ablation: str = None, seed: int = None):
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
        # Remove duplicate entries (now including seed)
        mask = ~(
            (df_old['dataset'] == dataset) & 
            (df_old['window'] == window) & 
            (df_old['horizon'] == horizon) & 
            (df_old['ablation'] == (ablation or "none")) &
            (df_old['seed'] == (seed or 42))
        )
        df_old = df_old[mask]
        df = pd.concat([df_old, df], ignore_index=True)
    
    # Sort and save - only include seed in sort if column exists
    sort_cols = ['dataset', 'window', 'horizon', 'ablation', 'seed']
    sort_cols = [c for c in sort_cols if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    
    # Append to text log
    txt_path = os.path.join(results_dir, ALL_RESULTS_TXT)
    txt_line = (
        f"[{info['timestamp']}] {info['model']} | "
        f"Dataset: {info['dataset']}, Window: {info['window']}, "
        f"Horizon: {info['horizon']}, Ablation: {info['ablation']} | "
        f"MAE: {metrics.get('mae', 0):.4f}, RMSE: {metrics.get('rmse', 0):.4f}, "
        f"PCC: {metrics.get('pcc', 0):.4f}, R2: {metrics.get('R2', 0):.4f}\n"
    )
    with open(txt_path, 'a', encoding='utf-8') as f:
        f.write(txt_line)
    
    if logger:
        logger.info(f"Metrics appended to {csv_path}")
        logger.info(f"Log appended to {txt_path}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'get_laplace_matrix',
    'peak_error',
    'setup_visualization_style',
    'visualize_matrices',
    'visualize_predictions',
    'plot_loss_curves',
    'save_metrics',
    'ALL_RESULTS_CSV',
    'ALL_RESULTS_TXT',
]
