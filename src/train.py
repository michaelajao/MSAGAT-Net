#!/usr/bin/env python3
"""
Training Script for MSAGATNet Model
- Comprehensive logging, visualization, evaluation, and result savings
- Includes publication-quality figures
- Multi-step predictions (horizon steps) with configurable parameters
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import logging
import random
import shutil
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from math import sqrt
from scipy.stats import pearsonr
import pandas as pd
import argparse
import datetime
from utils import *  # Ensure peak_error is defined in utils.py

# Add parent directory to path (if needed)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the MSAGATNet model
from model import MSAGATNet

# ----------------- Argument Parsing -----------------
def parse_arguments():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description='Training script for MSAGATNet model')
    parser.add_argument('--model', type=str, default='MSAGATNet',
                        help='Model name (only MSAGATNet supported)')
    parser.add_argument('--dataset', type=str, default='region785')
    parser.add_argument('--sim_mat', type=str, default='region-adj')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--train', type=float, default=0.5)
    parser.add_argument('--val', type=float, default=0.2)
    parser.add_argument('--test', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--mylog', action='store_false', default=True)
    parser.add_argument('--extra', type=str, default='')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--pcc', type=str, default='')
    parser.add_argument('--result', type=int, default=0)
    parser.add_argument('--record', type=str, default='')

    # Model hyperparameters
    parser.add_argument('--hidden_dim', type=int, default=32, help='Dimension of hidden representations')
    parser.add_argument('--attention_reg_weight', type=float, default=1e-4, help='Weight for attention regularization')
    parser.add_argument('--attn_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_scales', type=int, default=6, help='Number of temporal scales in MTFM')
    parser.add_argument('--kernel_size', type=int, default=9, help='Size of temporal convolution kernel')
    parser.add_argument('--temp_conv_out_channels', type=int, default=64, help='Output channels for temporal convolution')
    parser.add_argument('--low_rank_dim', type=int, default=8, help='Dimension for low-rank decompositions')
    parser.add_argument('--gru_layers', type=int, default=2, help='Number of GRU layers in the predictor')

    # Learning rate scheduler parameters
    parser.add_argument('--lr_patience', type=int, default=20, help='Patience for learning rate scheduler')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='Factor for learning rate reduction')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')

    # Start date for forecast visualization (if needed)
    parser.add_argument('--start_date', type=str, default='2020-01-01', help='Start date for forecast visualization')

    return parser.parse_args()

# ----------------- Logging & Visualization Setup -----------------
def setup_logging_and_visualization():
    """Setup logging configuration and matplotlib visualization style."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'mathtext.fontset': 'cm',
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 12,
        'legend.title_fontsize': 14,
        'font.size': 14,
        'figure.figsize': (8, 6),
        'figure.constrained_layout.use': True,
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': ':',
        'axes.axisbelow': True,
        'lines.linewidth': 2.0,
        'lines.markersize': 6,
        'errorbar.capsize': 3,
        'axes.prop_cycle': plt.cycler('color', [
            '#0173B2', '#DE8F05', '#029E73', '#D55E00',
            '#CC78BC', '#CA9161', '#FBAFE4', '#949494',
            '#ECE133', '#56B4E9'
        ]),
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.edgecolor': 'k',
        'legend.facecolor': 'white',
        'legend.shadow': False,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'xtick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.major.size': 6,
        'ytick.minor.size': 3,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2
    })
    
    return logger

# ----------------- Reproducibility & Device Setup -----------------
def get_available_device(requested_gpu=None):
    """
    Select the optimal device for training:
    - Returns CPU device if CUDA is not available
    - Returns requested GPU if valid
    - Otherwise finds the GPU with most free memory
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    num_gpus = torch.cuda.device_count()
    if requested_gpu is not None and 0 <= requested_gpu < num_gpus:
        return torch.device(f'cuda:{requested_gpu}')
    
    # If requested GPU is invalid, find GPU with most free memory
    if num_gpus > 0:
        free_memory = []
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            free_memory.append(torch.cuda.memory_reserved(i))
        
        best_gpu = free_memory.index(min(free_memory))
        return torch.device(f'cuda:{best_gpu}')
    
    return torch.device('cpu')

def setup_environment(args, logger):
    """Setup reproducible environment and select device."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    args.cuda = args.cuda and torch.cuda.is_available()
    device = get_available_device(args.gpu if args.cuda else None)
    if torch.cuda.is_available() and args.cuda:
        gpu_id = int(str(device).split(':')[1])
        torch.cuda.set_device(gpu_id)
        logger.info(f"Selected GPU {gpu_id} for training")

    logger.info(f'Using device: {device}')
    
    return device

# ----------------- Evaluation & Training Functions -----------------
def evaluate(data_loader, data, model, device, args, tag='val', show=0):
    """
    Evaluate model on given data.
    
    Args:
        data_loader: DataLoader object
        data: Data to evaluate on
        model: Model to evaluate
        device: Device to run evaluation on
        args: Command line arguments
        tag: Data tag (val/test)
        show: Whether to print debug information
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.
    n_samples = 0.
    x_value_mx, y_true_mx, y_pred_mx = [], [], []
    with torch.no_grad():
        for inputs in data_loader.get_batches(data, args.batch, False):
            X, Y, index = inputs
            X, Y = X.to(device), Y.to(device)
            if index is not None:
                index = index.to(device)
            output, attn_reg_loss = model(X, index)
            y_expanded = Y.unsqueeze(1).expand(-1, args.horizon, -1)
            loss_train = nn.MSELoss()(output, y_expanded) + attn_reg_loss
            total_loss += loss_train.item()
            n_samples += (output.size(0) * data_loader.m)
            x_value_mx.append(X.cpu())
            y_true_mx.append(Y.cpu())
            y_pred_mx.append(output.cpu())
    x_value_mx = torch.cat(x_value_mx)
    y_true_mx = torch.cat(y_true_mx)
    y_pred_mx = torch.cat(y_pred_mx)
    y_pred_mx = y_pred_mx[:, -1, :]  # final timestep

    # Denormalize
    x_value_states = x_value_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min

    rmse_states = np.mean(np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values')))
    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae)
    if not args.pcc:
        pcc_tmp = [pearsonr(y_true_states[:, k], y_pred_states[:, k])[0] for k in range(data_loader.m)]
        pcc_states = np.mean(np.array(pcc_tmp))
    else:
        pcc_states = 1
    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))

    y_true_flat = y_true_states.reshape(-1)
    y_pred_flat = y_pred_states.reshape(-1)
    rmse = sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    if show == 1:
        print('X values:', x_value_states)
        print('Ground truth shape:', y_true_flat.shape)
        print('Prediction shape:', y_pred_flat.shape)

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mape = np.mean(np.abs((y_pred_flat - y_true_flat) / (y_true_flat + 1e-5))) / 1e7
    pcc = pearsonr(y_true_flat, y_pred_flat)[0] if not args.pcc else 1
    r2 = r2_score(y_true_flat, y_pred_flat)
    var = explained_variance_score(y_true_flat, y_pred_flat)
    peak_mae = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)

    return {
        'loss': total_loss / n_samples,
        'mae': mae,
        'std_mae': std_mae,
        'rmse': rmse,
        'rmse_states': rmse_states,
        'pcc': pcc,
        'pcc_states': pcc_states,
        'mape': mape,
        'r2': r2,
        'r2_states': r2_states,
        'var': var,
        'var_states': var_states,
        'peak_mae': peak_mae
    }

def train_epoch(data_loader, data, model, optimizer, device, args):
    """
    Train model for one epoch.
    
    Args:
        data_loader: DataLoader object
        data: Data to train on
        model: Model to train
        optimizer: Optimizer to use
        device: Device to run training on
        args: Command line arguments
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.
    n_samples = 0.
    for inputs in data_loader.get_batches(data, args.batch, True):
        X, Y, index = inputs
        X, Y = X.to(device), Y.to(device)
        if index is not None:
            index = index.to(device)
        optimizer.zero_grad()
        output, attn_reg_loss = model(X, index)
        y_expanded = Y.unsqueeze(1).expand(-1, args.horizon, -1)
        loss = nn.MSELoss()(output, y_expanded) + attn_reg_loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        n_samples += (output.size(0) * data_loader.m)
    return total_loss / n_samples

# ----------------- Visualization Functions -----------------
def visualize_matrices(data_loader, model, save_path, device, logger):
    """
    Visualize adjacency (or geolocation), input correlation, and learned attention matrices.
    
    Args:
        data_loader: DataLoader object
        model: Trained model
        save_path: Path to save visualization
        device: Device to run model on
        logger: Logger object
    """
    model.eval()

    # 1. Adjacency/Geolocation matrix (or identity if not present)
    geo_mat = data_loader.adj.cpu().numpy() if hasattr(data_loader, 'adj') else np.eye(data_loader.m)

    # 2. Input correlation matrix from raw data
    raw_data = data_loader.rawdat
    input_corr = np.corrcoef(raw_data.T)

    # 3. Forward pass on a small batch to update model's attention
    batch = next(data_loader.get_batches(data_loader.test, args.batch, shuffle=False))
    X, _, index = batch
    X = X.to(device)
    if index is not None:
        index = index.to(device)
    with torch.no_grad():
        _ = model(X, index)

    # 4. Retrieve attention weights - for MSAGATNet, find them in the graph_attention module
    attn_mat = None
    if hasattr(model, 'graph_attention') and hasattr(model.graph_attention, 'attn'):
        attn_tensor = model.graph_attention.attn
        if len(attn_tensor.shape) == 4:  # (B, heads, N, N)
            attn_mat = attn_tensor.mean(dim=(0, 1)).detach().cpu().numpy()
        else:
            attn_mat = attn_tensor.detach().cpu().numpy()
    else:
        attn_mat = np.zeros_like(input_corr)
        logger.warning("Model does not have 'graph_attention.attn'; using zero matrix for visualization.")

    # 5. Clustering for more interpretable visualization
    clustered = False
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import squareform
        
        # Ensure matrix is square
        if attn_mat.shape[0] != attn_mat.shape[1]:
            min_dim = min(attn_mat.shape[0], attn_mat.shape[1])
            attn_mat = attn_mat[:min_dim, :min_dim]
            
        # Convert attention to distance
        distance = 1 - attn_mat
        np.fill_diagonal(distance, 0)
        distance = (distance + distance.T) / 2
        
        # Handle numerical issues
        if np.isnan(distance).any() or np.isinf(distance).any():
            distance = np.nan_to_num(distance)
            
        # Perform clustering
        condensed_dist = squareform(distance)
        Z = linkage(condensed_dist, method='ward')
        d = dendrogram(Z, no_plot=True)
        idx = d['leaves']
        
        # Reorder matrices based on clustering
        geo_mat = geo_mat[idx, :][:, idx]
        input_corr = input_corr[idx, :][:, idx]
        attn_mat = attn_mat[idx, :][:, idx]
        clustered = True
    except Exception as e:
        logger.warning(f"Clustering failed: {str(e)}")

    # 6. Plot matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    im0 = axes[0].imshow(geo_mat, cmap='viridis')
    axes[0].set_title("(a) Adjacency/Geolocation Matrix", fontsize=14)
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
    logger.info(f"Saved matrix visualization to {save_path}")

def visualize_predictions(y_true, y_pred, save_path, regions=5, logger=None):
    """
    Visualize predictions vs. ground truth for selected regions.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        save_path: Path to save visualization
        regions: Number of regions to visualize
        logger: Logger object
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

def plot_loss_curves(train_losses, val_losses, save_path, args, logger=None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save plot
        args: Command line arguments
        logger: Logger object
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
    title = f'Training Progress\nModel: MSAGATNet, Dataset: {args.dataset}, Window: {args.window}, Horizon: {args.horizon}'
    ax.set_title(title, fontsize=14, pad=10)
    
    textstr = f'Learning Rate: {args.lr}\nBatch Size: {args.batch}\nBest Epoch: {best_val_epoch}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    ax.legend(loc='upper right', frameon=True, framealpha=0.8)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved loss curves to {save_path}")

def save_metrics_to_csv(metrics, save_path, logger=None):
    """
    Save metrics to CSV file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save CSV
        logger: Logger object
    """
    metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()})
    metrics_df.to_csv(save_path, index=False)
    if logger:
        logger.info(f"Saved metrics to {save_path}")
    
def save_consolidated_metrics(metrics, dataset, window, horizon, save_path, logger=None):
    """
    Save metrics to a metrics CSV file or create a new one if it doesn't exist.
    Includes configuration details in the saved data.
    
    Args:
        metrics: Dictionary of metrics
        dataset: Dataset name
        window: Window size
        horizon: Prediction horizon
        save_path: Path to save CSV
        logger: Logger object
    """
    # Add model and configuration info to metrics
    metrics_with_info = {
        'model': 'MSAGATNet',
        'dataset': dataset,
        'window': window,
        'horizon': horizon,
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),
        **metrics
    }
    
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

# ----------------- Main Training Script -----------------
if __name__ == '__main__':
    args = parse_arguments()
    logger = setup_logging_and_visualization()
    device = setup_environment(args, logger)
    
    # Setup directories
    figures_dir = os.path.join(parent_dir, "report", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup TensorBoard logging
    log_token = f'MSAGATNet.{args.dataset}.w-{args.window}.h-{args.horizon}'
    if args.mylog:
        tensorboard_log_dir = os.path.join('tensorboard', log_token)
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        writer = SummaryWriter(tensorboard_log_dir)
        logger.info('TensorBoard logging to %s', tensorboard_log_dir)
    else:
        writer = None
    
    # Load data
    from data import DataBasicLoader
    data_loader = DataBasicLoader(args)
    
    # Initialize MSAGATNet model
    model = MSAGATNet(args, data_loader)
    model = model.to(device)
    logger.info('Model: %s', model)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor,
        patience=args.lr_patience, verbose=True
    )
    
    # Count trainable parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('#params:', pytorch_total_params)
    
    # Define metrics filename
    metrics_file = os.path.join(results_dir, 'metrics_MSAGATNet.csv')
    
    # Main training loop
    train_losses, val_losses = [], []
    bad_counter, best_epoch, best_val = 0, 0, 1e+20
    
    try:
        print('Begin training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train_loss = train_epoch(data_loader, data_loader.train, model, optimizer, device, args)
            eval_metrics = evaluate(data_loader, data_loader.val, model, device, args)
            val_loss = eval_metrics['loss']
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print('Epoch {:3d}| time: {:5.2f}s | train_loss: {:5.8f} | val_loss: {:5.8f}'.format(
                epoch, (time.time() - epoch_start_time), train_loss, val_loss))
            scheduler.step(val_loss)
            
            # TensorBoard logging
            if writer:
                writer.add_scalars('data/loss', {'train': train_loss, 'val': val_loss}, epoch)
                writer.add_scalar('data/mae', eval_metrics['mae'], epoch)
                writer.add_scalar('data/rmse', eval_metrics['rmse'], epoch)
                writer.add_scalar('data/pcc', eval_metrics['pcc'], epoch)
                writer.add_scalar('data/learning_rate', optimizer.param_groups[0]['lr'], epoch)

            # Save best model
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                bad_counter = 0
                model_path = os.path.join(args.save_dir, f'{log_token}.pt')
                torch.save(model.state_dict(), model_path)
                best_model_path = os.path.join(args.save_dir, 'best_model_MSAGATNet.pt')
                shutil.copy(model_path, best_model_path)
                print('Best validation epoch:', epoch, time.ctime())
                
                # Evaluate on test set
                test_metrics = evaluate(data_loader, data_loader.test, model, device, args, tag='test')
                print('TEST MAE: {:.4f} (±{:.4f}), RMSE: {:.4f}, RMSEs: {:.4f}'.format(
                    test_metrics['mae'], test_metrics['std_mae'], test_metrics['rmse'], test_metrics['rmse_states']))
                print('PCC: {:.4f}, PCCs: {:.4f}, MAPE: {:.4f}'.format(
                    test_metrics['pcc'], test_metrics['pcc_states'], test_metrics['mape']))
                print('R2: {:.4f}, R2s: {:.4f}, Var: {:.4f}, Vars: {:.4f}'.format(
                    test_metrics['r2'], test_metrics['r2_states'], test_metrics['var'], test_metrics['var_states']))
                print('Peak MAE: {:.4f}'.format(test_metrics['peak_mae']))
            else:
                bad_counter += 1

            # Early stopping
            if bad_counter == args.patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early at epoch', epoch)

    # Plot loss curves
    loss_fig_path = os.path.join(figures_dir, f"loss_curve_{log_token}.png")
    plot_loss_curves(train_losses, val_losses, loss_fig_path, args, logger)

    # Visualize learned matrices
    matrices_fig_path = os.path.join(figures_dir, f"matrices_{log_token}.png")
    visualize_matrices(data_loader, model, matrices_fig_path, device, logger)

    # Load best model and perform final evaluation
    model_path = os.path.join(args.save_dir, f'{log_token}.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_metrics = evaluate(data_loader, data_loader.test, model, device, args, tag='test', show=args.result)
    
    # Print final results
    print('\nFinal Evaluation:')
    print('TEST MAE: {:.4f} (±{:.4f}), RMSE: {:.4f}, RMSEs: {:.4f}'.format(
        test_metrics['mae'], test_metrics['std_mae'],
        test_metrics['rmse'], test_metrics['rmse_states']))
    print('PCC: {:.4f}, PCCs: {:.4f}, MAPE: {:.4f}'.format(
        test_metrics['pcc'], test_metrics['pcc_states'], test_metrics['mape']))
    print('R2: {:.4f}, R2s: {:.4f}, Var: {:.4f}, Vars: {:.4f}'.format(
        test_metrics['r2'], test_metrics['r2_states'], test_metrics['var'], test_metrics['var_states']))
    print('Peak MAE: {:.4f}'.format(test_metrics['peak_mae']))

    # Save metrics
    save_consolidated_metrics(
        metrics=test_metrics,
        dataset=args.dataset,
        window=args.window,
        horizon=args.horizon,
        save_path=metrics_file,
        logger=logger
    )

    # Record summary if requested
    if args.record:
        record_path = args.record
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
        with open(record_path, "a", encoding="utf-8") as f:
            f.write('Model: MSAGATNet, dataset: {}, window: {}, horizon: {}, seed: {}, MAE: {:.4f}, RMSE: {:.4f}, PCC: {:.4f}, R2: {:.4f}, lr: {}, dropout: {}, timestamp: {}\n'.format(
                args.dataset, args.window, args.horizon, args.seed,
                test_metrics['mae'], test_metrics['rmse'], test_metrics['pcc'],
                test_metrics['r2'], args.lr, args.dropout, time.strftime("%Y%m%d_%H%M%S")))
    
    # Close TensorBoard writer
    if writer:
        writer.close()

    # Final log messages
    logger.info(f"Training completed. Best epoch: {best_epoch}")
    logger.info(f"Model saved to: {model_path}")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Figures saved to: {figures_dir}")
    logger.info(f"Metrics saved to: {metrics_file}")
    logger.info(f"Training script completed successfully.")