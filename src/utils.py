import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import time


# -------------------------
# Graph Utilities
# -------------------------
def get_laplace_matrix(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute symmetric normalized Laplacian matrix for a batch of adjacency matrices.

    Args:
        adj (torch.Tensor): Adjacency matrices [batch, N, N]

    Returns:
        torch.Tensor: Normalized Laplacian [batch, N, N]
    """
    B, N, _ = adj.shape
    # Add self-loops
    I = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
    A_hat = adj + I
    # Degree matrix
    degree = A_hat.sum(dim=-1)  # [B, N]
    # inv sqrt degree
    d_inv_sqrt = torch.pow(degree + 1e-5, -0.5)
    D_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    # Normalized Laplacian
    L = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return L


# -------------------------
# Error Metrics
# -------------------------
def peak_error(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
    """
    Compute MAE focusing on peak regions above a threshold.

    Args:
        y_true (np.ndarray): True values [samples, nodes]
        y_pred (np.ndarray): Predicted values [samples, nodes]
        threshold (float): Peak threshold

    Returns:
        float: Mean absolute error on peak regions
    """
    mask = y_true >= threshold
    if not mask.any():
        return 0.0
    true_peaks = y_true[mask]
    pred_peaks = y_pred[mask]
    return mean_absolute_error(true_peaks, pred_peaks)


# -------------------------
# Visualization Style
# -------------------------
def setup_visualization_style():
    """
    Configure matplotlib for consistent, publication-quality figures.
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            # corrected grid parameters
            "grid.color": "gray",
            "grid.linewidth": 0.5,
            "axes.grid": True,
            "figure.figsize": (8, 6),
            "figure.constrained_layout.use": True,
            "grid.alpha": 0.3,
            "grid.linestyle": ":",
            "axes.axisbelow": True,
            "lines.linewidth": 2.0,
        }
    )


# -------------------------
# Matrix Visualization
# -------------------------
def visualize_matrices(
    loader, model, save_path: str, device: torch.device, logger=None
):
    """
    Plot adjacency, input correlation, and learned attention matrices side-by-side.
    """
    model.eval()
    # adjacency
    adj = getattr(loader, "adj", None)
    geo = adj.cpu().numpy() if adj is not None else np.eye(loader.m)
    # input correlation
    raw = loader.rawdat  # [T, N]
    corr = np.corrcoef(raw.T)
    # attention
    batch = next(
        loader.get_batches(loader.test, min(32, len(loader.test)), shuffle=False)
    )
    X, _, idx = batch
    X, idx = X.to(device), idx.to(device) if idx is not None else None
    with torch.no_grad():
        _, _ = model(X, idx)

    # Extract attention from different model architectures
    attn = None

    # Try different ways to access attention based on model type
    # MSTAGAT_Net with learned low-rank graph bias (U @ V)
    if hasattr(model, "spatial_module") and hasattr(model.spatial_module, "u") and hasattr(model.spatial_module, "v"):
        # Compute learned adjacency from low-rank factors
        u = model.spatial_module.u.detach()  # [heads, N, r]
        v = model.spatial_module.v.detach()  # [heads, r, N]
        graph_bias = torch.matmul(u, v)  # [heads, N, N]
        # Average over attention heads
        attn = graph_bias.mean(dim=0).cpu().numpy()  # [N, N]
    # Original MSTAGAT_Net
    elif hasattr(model, "spatial_module") and hasattr(model.spatial_module, "attn"):
        attn = model.spatial_module.attn.detach().cpu().numpy()

    # LocationAwareMSAGAT_Net
    elif hasattr(model, "spatial_module") and hasattr(
        model.spatial_module, "attention"
    ):
        attn = model.spatial_module.attention_weights.detach().cpu().numpy()
    elif hasattr(model, "spatial_module") and hasattr(model.spatial_module, "attn"):
        attn = model.spatial_module.attn.detach().cpu().numpy()

    # DynaGraphNet
    elif hasattr(model, "attention_weights"):
        attn = model.attention_weights.detach().cpu().numpy()
    elif hasattr(model, "graph_generator") and hasattr(model, "attention_layers"):
        # Use the dynamically generated graph as attention
        try:
            test_input = torch.randn(1, loader.m, model.hidden_dim).to(device)
            with torch.no_grad():
                attn = model.graph_generator(test_input).detach().cpu().numpy()[0]
        except:
            if logger:
                logger.warning("Could not generate attention from DynaGraphNet")

    # AFGNet
    elif hasattr(model, "attention") and hasattr(model.attention, "attention"):
        attn = model.attention.attention.detach().cpu().numpy()
    elif hasattr(model, "graph_inference"):
        # Generate a graph structure for visualization
        try:
            test_input = torch.randn(1, loader.m, model.feature_dim).to(device)
            with torch.no_grad():
                attn = model.graph_inference(test_input).detach().cpu().numpy()[0]
        except:
            if logger:
                logger.warning("Could not generate attention from AFGNet")

    # Any model with graph_attention
    elif hasattr(model, "graph_attention") and hasattr(model.graph_attention, "attn"):
        attn_obj = model.graph_attention.attn
        # If attn is a list of tensors, process each
        if isinstance(attn_obj, list):
            # Convert each tensor to numpy and stack/average as needed
            attn_list = [a.detach().cpu().numpy() for a in attn_obj if hasattr(a, 'detach')]
            if len(attn_list) == 1:
                attn = attn_list[0]
            elif len(attn_list) > 1:
                # Average over the list (e.g., heads or layers)
                attn = np.mean(np.stack(attn_list), axis=0)
            else:
                attn = None
        elif hasattr(attn_obj, 'detach'):
            attn = attn_obj.detach().cpu().numpy()
        else:
            attn = None

    # Fallback to zeros if no attention found
    if attn is None:
        attn = np.zeros_like(corr)
        if logger:
            logger.warning("No attention weights found; using zeros")

    # average over heads/batch if needed
    if isinstance(attn, np.ndarray) and attn.ndim == 4:
        attn = attn.mean(axis=(0, 1))
    elif isinstance(attn, np.ndarray) and attn.ndim == 3:
        attn = attn.mean(axis=0)

    # plotting
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

    # constrained_layout handles spacing; directly save figure
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    if logger:
        logger.info(f"Matrices saved to {save_path}")


# -------------------------
# Prediction Visualization
# -------------------------
def visualize_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    regions: int = 5,
    logger=None,
):
    """
    Plot ground truth vs. predictions for selected regions over time.
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


# -------------------------
# Loss Curve
# -------------------------
# def plot_loss_curves(train_losses, val_losses, save_path: str, args=None, logger=None):
#     """
#     Plot and save training vs. validation loss curves.
#     """
#     fig, ax = plt.subplots(figsize=(10, 6))
#     epochs = np.arange(1, len(train_losses) + 1)
#     ax.plot(epochs, train_losses, label="Train")
#     ax.plot(epochs, val_losses, label="Val")
#     best = np.argmin(val_losses) + 1
#     ax.scatter(best, val_losses[best - 1], marker="o", label=f"Best Epoch ({best})")
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Loss")
#     ax.legend()
#     if args:
#         ax.set_title(f"Loss: {args.dataset}, w={args.window}, h={args.horizon}")
#     plt.tight_layout()
#     plt.savefig(save_path, bbox_inches="tight")
#     plt.close(fig)
#     if logger:
#         logger.info(f"Loss curve saved to {save_path}")

def plot_loss_curves(train_losses, val_losses, save_path: str, args=None, logger=None):
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
    
    best_val_epoch = val_losses.index(min(val_losses)) + 1
    best_val_loss = min(val_losses)
    ax.scatter(best_val_epoch, best_val_loss, color='green', s=100, zorder=5,
               label=f'Best Val Loss: {best_val_loss:.6f}')
               
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    
    # Add title with model information if args provided
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
        logger.info(f"loss curves to {save_path}")

# -------------------------
# Metrics Saving
# -------------------------
# Consolidated results file names
ALL_RESULTS_CSV = "all_results.csv"
ALL_RESULTS_TXT = "all_results.txt"


def save_metrics(
    metrics: dict,
    save_path: str,
    dataset: str = None,
    window: int = None,
    horizon: int = None,
    logger=None,
    model_name: str = None,
    ablation: str = None,
):
    """
    Append metrics to consolidated CSV and TXT files.
    
    Args:
        metrics: Dictionary of metric values
        save_path: Path for individual file (used to extract directory)
        dataset: Dataset name
        window: Window size
        horizon: Prediction horizon
        logger: Logger object
        model_name: Model name
        ablation: Ablation variant (none, no_agam, no_mtfm, no_pprm)
    """
    results_dir = os.path.dirname(save_path)
    os.makedirs(results_dir, exist_ok=True)
    
    # Build the record
    info = {
        "model": model_name or "MSTAGAT",
        "dataset": dataset or "",
        "window": window or 0,
        "horizon": horizon or 0,
        "ablation": ablation or "none",
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
        # Remove duplicate entries (same dataset, window, horizon, ablation)
        mask = ~(
            (df_old['dataset'] == dataset) & 
            (df_old['window'] == window) & 
            (df_old['horizon'] == horizon) & 
            (df_old['ablation'] == (ablation or "none"))
        )
        df_old = df_old[mask]
        df = pd.concat([df_old, df], ignore_index=True)
    
    # Sort for consistent ordering
    df = df.sort_values(['dataset', 'window', 'horizon', 'ablation']).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    
    # Append to consolidated TXT (human-readable log)
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
