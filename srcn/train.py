#!/usr/bin/env python3
"""
Training Script for MSAGATNet Model
- Streamlined training, evaluation, and visualization
- Multi-step predictions with configurable parameters
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
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from scipy.stats import pearsonr
import pandas as pd
from srcn.utils import *  # Keep this as requested

# Add parent directory to path (if needed)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the models
from model import MSAGATNet
from model import MSTAGAT_Net

# ----------------- Argument Parsing -----------------
def parse_arguments():
    """Parse and return command line arguments."""
    parser = argparse.ArgumentParser(description='Training script for MSAGATNet model')
    parser.add_argument('--dataset', type=str, default='region785')
    parser.add_argument('--sim_mat', type=str, default='region-adj')
    parser.add_argument('--window', type=int, default=20)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--train', type=float, default=0.5)
    parser.add_argument('--val', type=float, default=0.2)
    parser.add_argument('--test', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--save_dir', type=str, default='save')
    parser.add_argument('--mylog', action='store_true', help='Enable TensorBoard logging')
    parser.add_argument('--extra', type=str, default='')
    parser.add_argument('--label', type=str, default='')
    parser.add_argument('--pcc', type=str, default='')
    parser.add_argument('--result', type=int, default=0)
    parser.add_argument('--record', type=str, default='')    
    parser.add_argument('--model', type=str, default='msagatnet', choices=['msagatnet', 'mstagat'], help='Choose which model to use: msagatnet or mstagat')

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

    return parser.parse_args()

# ----------------- Setup & Environment -----------------
def setup_environment(args, logger): # Add logger as an argument
    """Setup reproducible environment and select device."""
    # Set up seed for reproducibility
    if args.cuda and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    
    figures_dir = os.path.join(parent_dir, "report", "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    results_dir = os.path.join(parent_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup visualization style
    setup_visualization_style()
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if args.cuda and torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}') # Now logger is defined
    
    return device, figures_dir, results_dir

# ----------------- Evaluation Function -----------------
def evaluate(data_loader, data, model, device, args, tag='val'):
    """
    Evaluate model on given data.
    
    Args:
        data_loader: DataLoader object
        data: Data to evaluate on
        model: Model to evaluate
        device: Device to run evaluation on
        args: Command line arguments
        tag: Data tag (val/test)
        
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
            loss = nn.MSELoss()(output, y_expanded) + attn_reg_loss
            
            total_loss += loss.item()
            n_samples += (output.size(0) * data_loader.m)
            
            x_value_mx.append(X.cpu())
            y_true_mx.append(Y.cpu())
            y_pred_mx.append(output.cpu())
            
    # Concatenate batch results
    x_value_mx = torch.cat(x_value_mx)
    y_true_mx = torch.cat(y_true_mx)
    y_pred_mx = torch.cat(y_pred_mx)
    y_pred_mx = y_pred_mx[:, -1, :]  # final timestep prediction
    
    # Denormalize for evaluation
    x_value_states = x_value_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    
    # Flatten for aggregate metrics
    y_true_flat = y_true_states.reshape(-1)
    y_pred_flat = y_pred_states.reshape(-1)
    
    # Calculate metrics
    rmse = sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    
    # Per-region metrics (averaged)
    rmse_states = np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values')).mean()
    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae)
    
    # Calculate PCC if not disabled
    if not args.pcc:
        pcc = pearsonr(y_true_flat, y_pred_flat)[0]
        pcc_states = np.mean([pearsonr(y_true_states[:, k], y_pred_states[:, k])[0] 
                            for k in range(data_loader.m)])
    else:
        pcc = pcc_states = 1.0
    
    # Calculate R2 score
    r2 = r2_score(y_true_flat, y_pred_flat)
    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    
    # Calculate peak error
    peak_mae = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)
    
    return {
        'loss': total_loss / n_samples,
        'mae': mae,
        'std_mae': std_mae,
        'rmse': rmse,
        'rmse_states': rmse_states,
        'pcc': pcc,
        'pcc_states': pcc_states, 
        'r2': r2,
        'r2_states': r2_states,
        'peak_mae': peak_mae,
        'y_true': y_true_states,
        'y_pred': y_pred_states
    }

# ----------------- Training Function -----------------
def train_epoch(data_loader, model, optimizer, device, args):
    """
    Train model for one epoch.
    
    Args:
        data_loader: DataLoader object
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
    
    for inputs in data_loader.get_batches(data_loader.train, args.batch, True):
        X, Y, index = inputs
        X, Y = X.to(device), Y.to(device)
        if index is not None:
            index = index.to(device)
            
        optimizer.zero_grad()
        output, attn_reg_loss = model(X, index)
        y_expanded = Y.unsqueeze(1).expand(-1, args.horizon, -1)
        loss = nn.MSELoss()(output, y_expanded) + attn_reg_loss
        
        # Ensure gradient clipping is applied during training
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        
        total_loss += loss.item()
        n_samples += (output.size(0) * data_loader.m)
        
    return total_loss / n_samples

# ----------------- Main Training Function -----------------
def main():
    # Parse command line arguments
    args = parse_arguments()

    # Setup logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # Setup environment - pass logger here
    device, figures_dir, results_dir = setup_environment(args, logger)

    # Setup TensorBoard logging if enabled
    writer = None
    log_token_base = f'{args.model.upper()}.{args.dataset}.w-{args.window}.h-{args.horizon}' # Base token
    if args.mylog:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_log_dir = os.path.join('tensorboard', log_token_base)
            os.makedirs(tensorboard_log_dir, exist_ok=True)
            writer = SummaryWriter(tensorboard_log_dir)
            logger.info(f'TensorBoard logging to {tensorboard_log_dir}')
        except ImportError:
            logger.warning("TensorBoard not available, skipping TensorBoard logging")

    # Load data
    from data import DataBasicLoader
    data_loader = DataBasicLoader(args)

    # Instantiate the correct model based on the command-line argument
    if args.model == 'msagatnet':
        model = MSAGATNet(args, data_loader)
        logger.info('Using model: MSAGATNet')
    elif args.model == 'mstagat':
        model = MSTAGAT_Net(args, data_loader)
        logger.info('Using model: MSTAGAT_Net')
    else:
        logger.error(f"Unknown model type: {args.model}. Defaulting to MSAGATNet.")
        model = MSAGATNet(args, data_loader) # Default or raise error

    model = model.to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Model initialized with {param_count} parameters')

    # Setup optimizer and scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor,
        patience=args.lr_patience, verbose=True
    )

    # Define output paths using the base log token
    model_save_path = os.path.join(args.save_dir, f'{log_token_base}.pt')
    loss_fig_path = os.path.join(figures_dir, f"loss_curve_{log_token_base}.png")
    matrices_fig_path = os.path.join(figures_dir, f"matrices_{log_token_base}.png")
    pred_fig_path = os.path.join(figures_dir, f"predictions_{log_token_base}.png")
    results_csv_path = os.path.join(results_dir, f"metrics_{log_token_base}.csv")

    # Training loop variables
    train_losses, val_losses = [], []
    bad_counter, best_epoch, best_val = 0, 0, float('inf')

    try:
        logger.info('Beginning training')
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()

            # Train for one epoch
            train_loss = train_epoch(data_loader, model, optimizer, device, args)
            train_losses.append(train_loss)

            # Evaluate on validation set
            val_metrics = evaluate(data_loader, data_loader.val, model, device, args)
            val_loss = val_metrics['loss']
            val_losses.append(val_loss)

            epoch_time = time.time() - epoch_start_time
            logger.info(f'Epoch {epoch:3d} | time: {epoch_time:5.2f}s | train_loss: {train_loss:5.8f} | val_loss: {val_loss:5.8f}')

            # TensorBoard logging if enabled
            if writer:
                writer.add_scalars('data/loss', {'train': train_loss, 'val': val_loss}, epoch)
                writer.add_scalar('data/mae', val_metrics['mae'], epoch)
                writer.add_scalar('data/rmse', val_metrics['rmse'], epoch)
                writer.add_scalar('data/pcc', val_metrics['pcc'], epoch)
                writer.add_scalar('data/learning_rate', optimizer.param_groups[0]['lr'], epoch)

            # Update learning rate scheduler
            scheduler.step(val_loss)

            # Save best model and evaluate on test set if improved
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                bad_counter = 0

                # Save the model
                torch.save(model.state_dict(), model_save_path)
                logger.info(f'Best validation epoch: {epoch}, saving model to {model_save_path}')

                # Evaluate on test set
                test_metrics = evaluate(data_loader, data_loader.test, model, device, args, tag='test')
                logger.info(f'Test MAE: {test_metrics["mae"]:.4f}, RMSE: {test_metrics["rmse"]:.4f}, PCC: {test_metrics["pcc"]:.4f}')
            else:
                bad_counter += 1

            # Early stopping
            if bad_counter == args.patience:
                logger.info(f'Early stopping at epoch {epoch}')
                break

    except KeyboardInterrupt:
        logger.info(f'Exiting from training early at epoch {epoch}')

    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, loss_fig_path, args, logger)

    # Load best model for final evaluation - with explicit device mapping for reliability
    # Before loading the saved model
    model_class_name = model.__class__.__name__
    logger.info(f"Loading best {model_class_name} from {model_save_path}")
    try:
        # First attempt with standard mapping
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        # After loading the model
        logger.info(f"Verified model type after loading: {model.__class__.__name__}")
        # Print first few parameter names to verify model structure
        param_names = [name for name, _ in model.named_parameters()][:5]
        logger.info(f"Sample model parameters: {param_names}")
    except RuntimeError as e:
        # If that fails, use a more explicit approach
        logger.warning(f"Error loading model with standard mapping: {e}")
        logger.info("Trying alternative loading approach...")

        # Explicitly map any CUDA device to the current device
        if torch.cuda.is_available():
            model_state = torch.load(model_save_path, map_location=lambda storage, loc: storage.cuda(device.index))
        else:
            model_state = torch.load(model_save_path, map_location='cpu')

        model.load_state_dict(model_state)
        logger.info("Model loaded successfully with alternative approach")
        # After loading the model (alternative path)
        logger.info(f"Verified model type after loading: {model.__class__.__name__}")
        # Print first few parameter names to verify model structure
        param_names = [name for name, _ in model.named_parameters()][:5]
        logger.info(f"Sample model parameters: {param_names}")

    # Final test evaluation
    test_metrics = evaluate(data_loader, data_loader.test, model, device, args, tag='test')

    # Visualize matrices
    visualize_matrices(data_loader, model, matrices_fig_path, device, logger)

    # Visualize predictions (first few regions)
    visualize_predictions(
        test_metrics['y_true'], 
        test_metrics['y_pred'], 
        pred_fig_path, 
        regions=min(5, data_loader.m), 
        logger=logger
    )

    # Remove predictions from saved metrics
    y_true = test_metrics.pop('y_true', None)
    y_pred = test_metrics.pop('y_pred', None)

    # Save metrics
    save_metrics(test_metrics, results_csv_path, args.dataset, args.window, args.horizon, 
                 logger, model_name=args.model.upper())

    # Print final results
    logger.info('\nFinal Evaluation Results:')
    logger.info(f'MAE: {test_metrics["mae"]:.4f} (±{test_metrics["std_mae"]:.4f}), RMSE: {test_metrics["rmse"]:.4f}, RMSEs: {test_metrics["rmse_states"]:.4f}')
    logger.info(f'PCC: {test_metrics["pcc"]:.4f}, PCCs: {test_metrics["pcc_states"]:.4f}')
    logger.info(f'R2: {test_metrics["r2"]:.4f}, R2s: {test_metrics["r2_states"]:.4f}')
    logger.info(f'Peak MAE: {test_metrics["peak_mae"]:.4f}')

    # Record summary if requested
    if args.record:
        os.makedirs(os.path.dirname(args.record), exist_ok=True)
        with open(args.record, "a", encoding="utf-8") as f:
            f.write(f'Model: MSAGATNet, dataset: {args.dataset}, window: {args.window}, horizon: {args.horizon}, '
                   f'seed: {args.seed}, MAE: {test_metrics["mae"]:.4f}, RMSE: {test_metrics["rmse"]:.4f}, '
                   f'PCC: {test_metrics["pcc"]:.4f}, R2: {test_metrics["r2"]:.4f}, lr: {args.lr}, '
                   f'dropout: {args.dropout}, timestamp: {time.strftime("%Y%m%d_%H%M%S")}\n')

    # Clean up
    if writer:
        writer.close()

    logger.info(f"Training completed. Best epoch: {best_epoch}")
    logger.info(f"Model saved to: {model_save_path}")
    logger.info(f"Results saved to: {results_csv_path}")

if __name__ == '__main__':
    main()
    
    
    
    
# #!/usr/bin/env python3
# """
# Streamlined Training Script for MSTAGAT_Net Model
# - Flat structure without conditional model selection
# - Well-arranged into logical sections
# """

# import os
# import sys
# import time
# import random
# import argparse
# import logging

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from math import sqrt
# from scipy.stats import pearsonr

# from .utils import (  # Changed to relative import
#     setup_visualization_style,
#     plot_loss_curves,
#     visualize_matrices,
#     visualize_predictions,
#     save_metrics,
#     peak_error
# )
# from .data import DataBasicLoader # Changed to relative import
# from .model import MSTAGAT_Net # Changed to relative import

# # -------------------------
# # Argument Parsing
# # -------------------------
# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Training script for MSTAGAT_Net')
#     # Data settings
#     parser.add_argument('--dataset', type=str, default='region785')
#     parser.add_argument('--sim_mat', type=str, default='region-adj')
#     parser.add_argument('--window', type=int, default=20)
#     parser.add_argument('--horizon', type=int, default=5)
#     parser.add_argument('--train_frac', type=float, default=0.5)
#     parser.add_argument('--val_frac', type=float, default=0.2)
#     parser.add_argument('--test_frac', type=float, default=0.3)
#     parser.add_argument('--extra', type=str, default=None, help='Path to external data directory') # Added extra argument
#     # Training parameters
#     parser.add_argument('--epochs', type=int, default=1500)
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--learning_rate', type=float, default=1e-3)
#     parser.add_argument('--weight_decay', type=float, default=5e-4)
#     parser.add_argument('--dropout', type=float, default=0.2)
#     parser.add_argument('--seed', type=int, default=42)
#     parser.add_argument('--gpu_id', type=int, default=0)
#     parser.add_argument('--use_cuda', action='store_true')
#     parser.add_argument('--patience', type=int, default=100)
#     parser.add_argument('--save_dir', type=str, default='save')
#     parser.add_argument('--enable_tb', action='store_true', help='Enable TensorBoard logging')
#     # Model hyperparameters
#     parser.add_argument('--hidden_dim', type=int, default=32)
#     parser.add_argument('--attention_heads', type=int, default=4)
#     parser.add_argument('--attention_reg_weight', type=float, default=1e-5)
#     parser.add_argument('--num_scales', type=int, default=4)
#     parser.add_argument('--kernel_size', type=int, default=3)
#     parser.add_argument('--bottleneck_dim', type=int, default=8)
#     return parser.parse_args()

# # -------------------------
# # Environment Setup
# # -------------------------
# def setup_environment(args):
#     # Calculate parent directory for constructing paths
#     current_dir = os.path.dirname(os.path.abspath(__file__))
#     parent_dir = os.path.dirname(current_dir)

#     random.seed(args.seed)
#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     if args.use_cuda and torch.cuda.is_available():
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
#         torch.cuda.manual_seed(args.seed)
#         torch.backends.cudnn.deterministic = True
#     os.makedirs(args.save_dir, exist_ok=True)
#     figures_dir = os.path.join(parent_dir, 'report', 'figures')
#     os.makedirs(figures_dir, exist_ok=True)
#     results_dir = os.path.join(parent_dir, 'results')
#     os.makedirs(results_dir, exist_ok=True)
#     setup_visualization_style()
#     device = torch.device(f'cuda:{args.gpu_id}' if args.use_cuda and torch.cuda.is_available() else 'cpu')
#     return device, figures_dir, results_dir

# # -------------------------
# # Evaluation Function
# # -------------------------
# def evaluate(loader, data_split, model, device, args):
#     model.eval()
#     total_loss, count = 0.0, 0
#     preds, trues = [], []
#     criterion = nn.MSELoss()
#     with torch.no_grad():
#         for X, Y, idx in loader.get_batches(data_split, args.batch_size, shuffle=False):
#             X, Y = X.to(device), Y.to(device)
#             idx = idx.to(device) if idx is not None else None
#             output, reg_loss = model(X, idx)
#             target = Y.unsqueeze(1).expand(-1, args.horizon, -1)
#             loss = criterion(output, target) + reg_loss
#             total_loss += loss.item() * X.size(0)
#             count += X.size(0)
#             preds.append(output[:, -1, :].cpu().numpy()) # Store last step prediction
#             trues.append(Y.cpu().numpy()) # Store ground truth

#     preds = np.vstack(preds)
#     trues = np.vstack(trues)

#     # Denormalize predictions and true values
#     preds_denorm = loader.unnormalize(preds)
#     trues_denorm = loader.unnormalize(trues)

#     # Flatten denormalized arrays for standard metrics
#     flat_pred_denorm = preds_denorm.flatten()
#     flat_true_denorm = trues_denorm.flatten()

#     # Calculate metrics on denormalized data
#     rmse = sqrt(mean_squared_error(flat_true_denorm, flat_pred_denorm))
#     mae = mean_absolute_error(flat_true_denorm, flat_pred_denorm)
#     pcc = pearsonr(flat_true_denorm, flat_pred_denorm)[0]
#     r2 = r2_score(flat_true_denorm, flat_pred_denorm)

#     # Peak error (uses unflattened, denormalized data implicitly via threshold)
#     # Assuming loader.peak_thold is on the original scale
#     peak_mae = peak_error(trues_denorm.copy(), preds_denorm.copy(), loader.peak_thold)

#     return {
#         'loss': total_loss / count, # Loss is still calculated on normalized scale
#         'rmse': rmse,
#         'mae': mae,
#         'pcc': pcc,
#         'r2': r2,
#         'peak_mae': peak_mae,
#         'y_true': trues_denorm, # Return denormalized true values
#         'y_pred': preds_denorm  # Return denormalized predictions
#     }

# # -------------------------
# # Training Function
# # -------------------------
# def train_epoch(loader, model, optimizer, device, args):
#     model.train()
#     total_loss, count = 0.0, 0
#     criterion = nn.MSELoss()
#     for X, Y, idx in loader.get_batches(loader.train, args.batch_size, shuffle=True):
#         X, Y = X.to(device), Y.to(device)
#         idx = idx.to(device) if idx is not None else None
#         optimizer.zero_grad()
#         output, reg_loss = model(X, idx)
#         target = Y.unsqueeze(1).expand(-1, args.horizon, -1)
#         loss = criterion(output, target) + reg_loss
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         total_loss += loss.item() * X.size(0)
#         count += X.size(0)
#     return total_loss / count

# # -------------------------
# # Script Execution
# # -------------------------
# args = parse_arguments()
# logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
# logger = logging.getLogger('train')
# device, figures_dir, results_dir = setup_environment(args)
# logger.info(f'Using device: {device}')

# # Data Loader
# loader = DataBasicLoader(args)

# # Model
# model = MSTAGAT_Net(args, loader).to(device)
# logger.info(f'Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

# # Optimizer & Scheduler
# optimizer = optim.Adam(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=args.learning_rate,
#     weight_decay=args.weight_decay
# )
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True
# )

# # TensorBoard (optional)
# writer = None
# if args.enable_tb:
#     try:
#         from torch.utils.tensorboard import SummaryWriter
#         tb_dir = os.path.join('tensorboard', 'MSTAGAT')
#         os.makedirs(tb_dir, exist_ok=True)
#         writer = SummaryWriter(tb_dir)
#         logger.info(f'TensorBoard enabled at {tb_dir}')
#     except ImportError:
#         logger.warning('TensorBoard not installed; skipping')

# # Training Loop
# train_losses, val_losses = [], []
# best_val, epochs_no_improve = float('inf'), 0
# logger.info('Starting training')
# for epoch in range(1, args.epochs + 1):
#     t0 = time.time()
#     train_loss = train_epoch(loader, model, optimizer, device, args)
#     val_metrics = evaluate(loader, loader.val, model, device, args)
#     train_losses.append(train_loss)
#     val_losses.append(val_metrics['loss'])
#     scheduler.step(val_metrics['loss'])
#     logger.info(f"Epoch {epoch}/{args.epochs}: train={train_loss:.4f}, val={val_metrics['loss']:.4f}, time={time.time()-t0:.1f}s")
#     if writer:
#         writer.add_scalars('Loss', {'Train': train_loss, 'Val': val_metrics['loss']}, epoch)
#     if val_metrics['loss'] < best_val:
#         best_val = val_metrics['loss']
#         epochs_no_improve = 0
#         best_path = os.path.join(args.save_dir, 'best_model.pt')
#         torch.save(model.state_dict(), best_path)
#         logger.info(f'Saved best model to {best_path}')
#     else:
#         epochs_no_improve += 1
#         if epochs_no_improve >= args.patience:
#             logger.info('Early stopping triggered')
#             break

# # Post-training Evaluation and Visualization
# plot_loss_curves(train_losses, val_losses, os.path.join(figures_dir, 'loss_curve.png'), args, logger)
# model.load_state_dict(torch.load(best_path, map_location=device))
# test_metrics = evaluate(loader, loader.test, model, device, args)
# logger.info(f"Test Results - MAE: {test_metrics['mae']:.4f}, RMSE: {test_metrics['rmse']:.4f}, PCC: {test_metrics['pcc']:.4f}, R2: {test_metrics['r2']:.4f}")
# visualize_matrices(loader, model, os.path.join(figures_dir, 'attention_matrices.png'), device, logger)
# visualize_predictions(test_metrics['y_true'], test_metrics['y_pred'], os.path.join(figures_dir, 'predictions.png'), regions=min(5, loader.m), logger=logger)
# save_metrics(test_metrics, os.path.join(results_dir, 'metrics.csv'), args.dataset, args.window, args.horizon, logger, model_name='MSTAGAT_NET')
# if writer:
#     writer.close()
# logger.info('Training complete')

