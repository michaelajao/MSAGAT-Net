import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import pearsonr

def getLaplaceMat(batch_size, num_nodes, adj):
    """Get normalized Laplacian matrix."""
    # Add self-loops
    adj = adj + torch.eye(num_nodes, device=adj.device)[None, :, :]
    
    # Calculate degree matrix
    degree = torch.sum(adj, dim=-1)  # [B, N]
    
    # Normalize adjacency matrix
    degree_inv_sqrt = torch.pow(degree + 1e-5, -0.5)
    degree_inv_sqrt = degree_inv_sqrt.unsqueeze(-1) * torch.eye(num_nodes, device=adj.device)[None, :, :]
    
    # Compute normalized Laplacian
    laplace = torch.matmul(torch.matmul(degree_inv_sqrt, adj), degree_inv_sqrt)
    
    return laplace

def peak_error(y_true_states, y_pred_states, threshold): 
    """Calculate mean absolute error in peak regions."""
    # Mask low values using threshold
    y_true_states[y_true_states < threshold] = 0
    mask_idx = np.argwhere(y_true_states <= threshold)
    for idx in mask_idx:
        y_pred_states[idx[0]][idx[1]] = 0
    
    # Calculate MAE only in peak regions
    peak_mae_raw = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    peak_mae = np.mean(peak_mae_raw)
    return peak_mae

# === Visualization Functions ===

def setup_visualization_style():
    """Setup matplotlib visualization style with more modest parameters."""
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 12,
        'figure.figsize': (8, 6),
        'figure.constrained_layout.use': True,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
        'axes.axisbelow': True,
        'lines.linewidth': 2.0,
    })

def visualize_matrices(data_loader, model, save_path, device, logger=None):
    """
    Visualize adjacency, input correlation, and learned attention matrices.
    
    Args:
        data_loader: DataLoader object
        model: Trained model
        save_path: Path to save visualization
        device: Device to run model on
        logger: Logger object (optional)
    """
    model.eval()

    # 1. Adjacency/Geolocation matrix
    geo_mat = data_loader.adj.cpu().numpy() if hasattr(data_loader, 'adj') else np.eye(data_loader.m)

    # 2. Input correlation matrix from raw data
    raw_data = data_loader.rawdat
    input_corr = np.corrcoef(raw_data.T)

    # 3. Forward pass to update model's attention
    batch = next(data_loader.get_batches(data_loader.test, min(32, len(data_loader.test)), shuffle=False))
    X, Y, index = batch
    X = X.to(device)
    if index is not None:
        index = index.to(device)
    with torch.no_grad():
        _ = model(X, index)

    # 4. Retrieve attention weights
    attn_mat = None
    if hasattr(model, 'graph_attention') and hasattr(model.graph_attention, 'attn'):
        attn_tensor = model.graph_attention.attn
        if len(attn_tensor.shape) == 4:  # (B, heads, N, N)
            attn_mat = attn_tensor.mean(dim=(0, 1)).detach().cpu().numpy()
        else:
            attn_mat = attn_tensor.detach().cpu().numpy()
    else:
        attn_mat = np.zeros_like(input_corr)
        if logger:
            logger.warning("Model does not have 'graph_attention.attn'; using zero matrix.")

    # 5. Plot matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    im0 = axes[0].imshow(geo_mat, cmap='viridis')
    axes[0].set_title("(a) Adjacency Matrix", fontsize=14)
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_xlabel("Region Index")
    axes[0].set_ylabel("Region Index")
    
    im1 = axes[1].imshow(input_corr, cmap='viridis')
    axes[1].set_title("(b) Input Correlation Matrix", fontsize=14)
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_xlabel("Region Index")
    axes[1].set_ylabel("Region Index")
    
    im2 = axes[2].imshow(attn_mat, cmap='viridis')
    axes[2].set_title("(c) Learned Attention Matrix", fontsize=14)
    plt.colorbar(im2, ax=axes[2])
    axes[2].set_xlabel("Region Index")
    axes[2].set_ylabel("Region Index")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    if logger:
        logger.info(f"Saved matrix visualization to {save_path}")

def visualize_predictions(y_true, y_pred, save_path, regions=5, logger=None):
    """
    Visualize predictions vs. ground truth for selected regions.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        save_path: Path to save visualization
        regions: Number of regions to visualize
        logger: Logger object (optional)
    """
    n_regions = min(regions, y_true.shape[1])
    fig, axes = plt.subplots(n_regions, 1, figsize=(12, 3 * n_regions), sharex=True)
    if n_regions == 1:
        axes = [axes]
    time_steps = range(y_true.shape[0])
    for i in range(n_regions):
        axes[i].plot(time_steps, y_true[:, i], 'b-', label='Ground Truth', alpha=0.7)
        axes[i].plot(time_steps, y_pred[:, i], 'r-', label='Prediction', alpha=0.7)
        axes[i].set_title(f'Region {i+1}', fontsize=12)
        axes[i].set_ylabel('Value', fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        axes[i].legend(loc='upper right')
    axes[-1].set_xlabel('Time Steps', fontsize=10)
    plt.suptitle('Predictions vs. Ground Truth', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    if logger:
        logger.info(f"Saved prediction visualization to {save_path}")

def plot_loss_curves(train_losses, val_losses, save_path, args=None, logger=None):
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
        logger.info(f"Saved loss curves to {save_path}")

def save_metrics(metrics, save_path, dataset=None, window=None, horizon=None, logger=None):
    """
    Save metrics to a CSV file or create a new one if it doesn't exist.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save CSV
        dataset: Dataset name (optional)
        window: Window size (optional)
        horizon: Prediction horizon (optional)
        logger: Logger object (optional)
    """
    import time
    
    # Add model and configuration info to metrics if provided
    metrics_with_info = {
        'model': 'MSAGATNet',
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),
        **metrics
    }
    
    if dataset:
        metrics_with_info['dataset'] = dataset
    if window:
        metrics_with_info['window'] = window
    if horizon:
        metrics_with_info['horizon'] = horizon
    
    metrics_df = pd.DataFrame([metrics_with_info])
    
    # Check if file exists to append or create new
    if os.path.exists(save_path):
        existing_df = pd.read_csv(save_path)
        updated_df = pd.concat([existing_df, metrics_df], ignore_index=True)
        updated_df.to_csv(save_path, index=False)
        if logger:
            logger.info(f"Appended metrics to file: {save_path}")
    else:
        metrics_df.to_csv(save_path, index=False)
        if logger:
            logger.info(f"Created new metrics file: {save_path}")