#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Script for MSTAGAT_Net Model
- Multi-step forecasting with attention regularization
- Comprehensive evaluation metrics
- TensorBoard logging and early stopping
"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from __future__ import print_function

import os, itertools, random, argparse, time, datetime
import numpy as np
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from math import sqrt
import csv
import stat
import shutil
import logging
import glob
import sys
import time
from tensorboardX import SummaryWriter
from scipy.stats import pearsonr

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s') # include timestamp

# Add current directory to path if needed
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import model
from model1 import MSTAGAT_Net

# Command line argument parsing
ap = argparse.ArgumentParser(description='Training script for MSTAGAT_Net model')
# Dataset parameters
ap.add_argument('--dataset', type=str, default='japan', help="Dataset name")
ap.add_argument('--sim_mat', type=str, default='japan-adj', help="Adjacency matrix filename")
ap.add_argument('--window', type=int, default=20, help='Window size for historical data')
ap.add_argument('--horizon', type=int, default=5, help='Prediction horizon')
ap.add_argument('--train', type=float, default=0.5, help="Training ratio (0, 1)")
ap.add_argument('--val', type=float, default=0.2, help="Validation ratio (0, 1)")
ap.add_argument('--test', type=float, default=0.3, help="Testing ratio (0, 1)")
ap.add_argument('--extra', type=str, default='', help='External data directory')

# Training parameters
ap.add_argument('--epochs', type=int, default=1500, help='Maximum number of training epochs')
ap.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
ap.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 regularization)')
ap.add_argument('--dropout', type=float, default=0.355, help='Dropout rate')
ap.add_argument('--batch', type=int, default=32, help="Batch size")
ap.add_argument('--patience', type=int, default=100, help='Early stopping patience')
ap.add_argument('--max_grad_norm', type=float, default=1.0, help='Maximum gradient norm for clipping')
ap.add_argument('--lr_factor', type=float, default=0.5, help='Factor for learning rate reduction')
ap.add_argument('--lr_patience', type=int, default=20, help='Patience for learning rate scheduler')

# Model-specific parameters
ap.add_argument('--hidden_dim', type=int, default=32, help='Dimension of hidden representations')
ap.add_argument('--attn_heads', type=int, default=4, help='Number of attention heads')
ap.add_argument('--attention_reg_weight', type=float, default=1e-5, help='Weight for attention regularization')
ap.add_argument('--num_scales', type=int, default=4, help='Number of temporal scales')
ap.add_argument('--kernel_size', type=int, default=3, help='Size of temporal convolution kernel')
ap.add_argument('--temp_conv_out_channels', type=int, default=16, help='Output channels for temporal convolution')
ap.add_argument('--low_rank_dim', type=int, default=8, help='Dimension for low-rank decompositions')

# System settings
ap.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
ap.add_argument('--gpu', type=int, default=0, help='GPU ID')
ap.add_argument('--cuda', action='store_true', default=False, help='Use CUDA')
ap.add_argument('--save_dir', type=str, default='save', help='Directory to save models')
ap.add_argument('--mylog', action='store_true', default=True, help='Enable TensorBoard logging')
ap.add_argument('--result', type=int, default=0, help='Show detailed results when set to 1')
ap.add_argument('--pcc', type=str, default='', help='Skip PCC calculation if not empty')
ap.add_argument('--record', type=str, default='', help='File to record results')
ap.add_argument('--eval', type=str, default='', help='Evaluation test file')

args = ap.parse_args()
print('--------Parameters--------')
print(args)
print('--------------------------')

# Set GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Create necessary directories
os.makedirs('result', exist_ok=True)
os.makedirs('save', exist_ok=True)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

# Set seeds for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# Check CUDA availability
args.cuda = args.cuda and torch.cuda.is_available()
args.cuda = args.gpu is not None and torch.cuda.is_available()

# Create device object for consistent usage
if args.cuda:
    try:
        torch.cuda.set_device(args.gpu)
        device = torch.device(f'cuda:{args.gpu}')
    except (AttributeError, AssertionError):
        device = torch.device(f'cuda:{args.gpu}')
else:
    device = torch.device('cpu')

logger.info('cuda %s, using device %s', args.cuda, device)

# Setup logging with TensorBoard
time_token = str(time.time()).split('.')[0]
log_token = f'MSTAGAT_Net.{args.dataset}.w-{args.window}.h-{args.horizon}'

if args.mylog:
    tensorboard_log_dir = f'tensorboard/{log_token}'
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    try:
        shutil.rmtree(tensorboard_log_dir)  # remove folder
    except PermissionError as e:
        err_file_path = str(e).split("\'", 2)[1]
        if os.path.exists(err_file_path):
            os.chmod(err_file_path, stat.S_IWUSR)
    logger.info('tensorboard logging to %s', tensorboard_log_dir)

# Load data
from data import DataBasicLoader
data_loader = DataBasicLoader(args)

# Map attention_reg_weight to parameter name used internally by the model
setattr(args, 'attention_regularization_weight', args.attention_reg_weight)
setattr(args, 'bottleneck_dim', args.low_rank_dim)
setattr(args, 'feature_channels', args.temp_conv_out_channels)
setattr(args, 'attention_heads', args.attn_heads)

# Initialize the model
model = MSTAGAT_Net(args, data_loader)
logger.info('model: %s', model)

# Move model to device
if args.cuda:
    model = model.to(device)

# Setup optimizer and scheduler
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=args.lr, weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=args.lr_factor,
    patience=args.lr_patience, verbose=True
)

# Count trainable parameters
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('#params:', pytorch_total_params)

# Peak error calculation for specific domain regions
def peak_error(y_true_states, y_pred_states, threshold):
    """Calculate mean absolute error in peak regions."""
    # Create copies to avoid modifying original data
    y_true_copy = y_true_states.copy()
    y_pred_copy = y_pred_states.copy()
    
    # Mask low values using threshold
    mask_idx = np.argwhere(y_true_copy < threshold)
    for idx in mask_idx:
        y_true_copy[idx[0], idx[1]] = 0
        y_pred_copy[idx[0], idx[1]] = 0
    
    # Calculate MAE only in peak regions where values > threshold
    peak_mae_raw = mean_absolute_error(y_true_copy, y_pred_copy, multioutput='raw_values')
    peak_mae = np.mean(peak_mae_raw)
    
    return peak_mae

# Evaluation function
def evaluate(data_loader, data, tag='val', show=0):
    model.eval()
    total = 0.0
    n_samples = 0.0
    total_loss = 0.0
    y_true, y_pred = [], []
    batch_size = args.batch
    y_pred_mx = []
    y_true_mx = []
    x_value_mx = []
    
    with torch.no_grad():
        for inputs in data_loader.get_batches(data, batch_size, False):
            X, Y = inputs[0], inputs[1]
            index = inputs[2]
            
            # Move data to device
            if args.cuda:
                X, Y = X.to(device), Y.to(device)
                if index is not None:
                    index = index.to(device)
            
            output, attn_reg_loss = model(X, index)
            y_expanded = Y.unsqueeze(1).expand(-1, args.horizon, -1)
            loss = F.mse_loss(output, y_expanded) + attn_reg_loss
            
            total_loss += loss.item()
            n_samples += (output.size(0) * data_loader.m)
            
            # MSTAGAT_Net returns multi-step predictions, take the last one
            output = output[:, -1, :]
            
            x_value_mx.append(X.cpu())
            y_true_mx.append(Y.cpu())
            y_pred_mx.append(output.cpu())
    
    # Concatenate batch results
    x_value_mx = torch.cat(x_value_mx)
    y_pred_mx = torch.cat(y_pred_mx)
    y_true_mx = torch.cat(y_true_mx)  # [n_samples, 47]
    
    # Denormalize data
    x_value_states = x_value_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    
    # Calculate metrics on state/regional level
    rmse_states = np.mean(np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values')))
    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae)  # Standard deviation of MAEs for all states/places
    
    # Calculate PCC (Pearson Correlation Coefficient)
    if not args.pcc:
        pcc_tmp = []
        for k in range(data_loader.m):
            pcc_tmp.append(pearsonr(y_true_states[:, k], y_pred_states[:, k])[0])
        pcc_states = np.mean(np.array(pcc_tmp))
    else:
        pcc_states = 1.0
    
    # Additional metrics
    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))
    
    # Convert to flattened arrays for aggregate metrics
    y_true = np.reshape(y_true_states, (-1))
    y_pred = np.reshape(y_pred_states, (-1))
    
    if show == 1:
        print('x value', x_value_states)
        print('ground true', y_true.shape)
        print('predict value', y_pred.shape)
    
    # Calculate aggregate metrics
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_pred - y_true) / (y_true + 0.00001)))
    mape /= 10000000  # scaling factor
    
    if not args.pcc:
        pcc = pearsonr(y_true, y_pred)[0]
    else:
        pcc = 1.0
    
    r2 = r2_score(y_true, y_pred, multioutput='uniform_average')
    var = explained_variance_score(y_true, y_pred, multioutput='uniform_average')
    peak_mae = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)
    
    # Store true/pred values for visualization
    global y_true_t
    global y_pred_t
    y_true_t = y_true_states
    y_pred_t = y_pred_states
    
    return (float(total_loss / n_samples), mae, std_mae, rmse, rmse_states, pcc, 
            pcc_states, mape, r2, r2_states, var, var_states, peak_mae)

# Training function
def train(data_loader, data):
    model.train()
    total_loss = 0.0
    n_samples = 0.0
    batch_size = args.batch
    
    for inputs in data_loader.get_batches(data, batch_size, True):
        X, Y = inputs[0], inputs[1]
        index = inputs[2]
        
        # Move data to device
        if args.cuda:
            X, Y = X.to(device), Y.to(device)
            if index is not None:
                index = index.to(device)
        
        optimizer.zero_grad()
        output, attn_reg_loss = model(X, index)
        
        # MSTAGAT_Net returns predictions for all horizons, expand Y to match
        y_expanded = Y.unsqueeze(1).expand(-1, args.horizon, -1)
        loss = F.mse_loss(output, y_expanded) + attn_reg_loss
        
        total_loss += loss.item()
        loss.backward()
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
        optimizer.step()
        
        n_samples += (output.size(0) * data_loader.m)
    
    return float(total_loss / n_samples)

# Visualization function for matrices
def visualize_matrices(model, data_loader, save_path):
    import matplotlib.pyplot as plt
    
    model.eval()
    
    # Adjacency/Geolocation matrix
    geo_mat = data_loader.adj.cpu().numpy() if hasattr(data_loader, 'adj') else np.eye(data_loader.m)
    
    # Input correlation matrix from raw data
    raw_data = data_loader.rawdat
    input_corr = np.corrcoef(raw_data.T)
    
    # Forward pass to update model's attention
    batch = next(data_loader.get_batches(data_loader.test, min(32, len(data_loader.test)), False))
    X, Y, index = batch
    if args.cuda:
        X = X.to(device)
        if index is not None:
            index = index.to(device)
    
    with torch.no_grad():
        _ = model(X, index)
    
    # Retrieve attention weights
    attn_mat = None
    if hasattr(model, 'spatial_module') and hasattr(model.spatial_module, 'attn'):
        attn_tensor = model.spatial_module.attn
        if len(attn_tensor.shape) == 4:  # (B, heads, N, N)
            attn_mat = attn_tensor.mean(dim=(0, 1)).detach().cpu().numpy()
        else:
            attn_mat = attn_tensor.detach().cpu().numpy()
    else:
        attn_mat = np.zeros_like(input_corr)
        logger.warning("Model does not have 'spatial_module.attn'; using zero matrix.")
    
    # Plot matrices
    plt.figure(figsize=(18, 6))
    
    plt.subplot(131)
    plt.imshow(geo_mat, cmap='viridis')
    plt.colorbar()
    plt.title("Adjacency Matrix")
    
    plt.subplot(132)
    plt.imshow(input_corr, cmap='viridis')
    plt.colorbar()
    plt.title("Input Correlation Matrix")
    
    plt.subplot(133)
    plt.imshow(attn_mat, cmap='viridis')
    plt.colorbar()
    plt.title("Learned Attention Matrix")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved matrix visualization to {save_path}")

# Main training loop
bad_counter = 0
best_epoch = 0
best_val = 1e+20

try:
    print('begin training')
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train_loss = train(data_loader, data_loader.train)
        
        val_loss, mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae = evaluate(data_loader, data_loader.val)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        print('Epoch {:3d}|time:{:5.2f}s|train_loss {:5.8f}|val_loss {:5.8f}'.format(
            epoch, (time.time() - epoch_start_time), train_loss, val_loss))
        
        # TensorBoard logging
        if args.mylog:
            writer.add_scalars('data/loss', {'train': train_loss, 'val': val_loss}, epoch)
            writer.add_scalars('data/mae', {'val': mae}, epoch)
            writer.add_scalars('data/rmse', {'val': rmse}, epoch)
            writer.add_scalars('data/rmse_states', {'val': rmse_states}, epoch)
            writer.add_scalars('data/pcc', {'val': pcc}, epoch)
            writer.add_scalars('data/pcc_states', {'val': pcc_states}, epoch)
            writer.add_scalars('data/R2', {'val': r2}, epoch)
            writer.add_scalars('data/R2_states', {'val': r2_states}, epoch)
            writer.add_scalars('data/var', {'val': var}, epoch)
            writer.add_scalars('data/var_states', {'val': var_states}, epoch)
            writer.add_scalars('data/peak_mae', {'val': peak_mae}, epoch)
            writer.add_scalar('data/learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save the model if validation loss improved
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            bad_counter = 0
            model_path = f'{args.save_dir}/{log_token}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, model_path)
            print('Best validation epoch:', epoch, time.ctime())
            
            # Test the model on test data
            test_loss, mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae = evaluate(data_loader, data_loader.test, tag='test')
            print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} MAPE {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format(
                mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae))
        else:
            bad_counter += 1
        
        # Early stopping
        if bad_counter == args.patience:
            break
            
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early, epoch', epoch)

# Create directories for figures
figures_dir = os.path.join(parent_dir, "report", "figures")
os.makedirs(figures_dir, exist_ok=True)

# Load the best saved model
model_path = f'{args.save_dir}/{log_token}.pt'
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Final test evaluation
test_loss, mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae = evaluate(data_loader, data_loader.test, tag='test', show=args.result)
print('Final evaluation')

# Visualize matrices
matrices_fig_path = os.path.join(figures_dir, f"matrices_{log_token}.png")
visualize_matrices(model, data_loader, matrices_fig_path)

# Record test results
if args.record != '':
    with open(args.record, "a", encoding="utf-8") as f:
        f.write('Model: MSTAGAT_Net, dataset: {}, window: {}, horizon: {}, seed: {}, MAE: {:5.4f}, RMSE: {:5.4f}, PCC: {:5.4f}, R2: {:5.4f}, lr: {}, num_scales: {}, kernel_size: {}, temp_conv_out_channels: {}, attention_reg_weight: {}, dropout: {}, timestamp: {}\n'.format(
            args.dataset, args.window, args.horizon, args.seed, mae, rmse, pcc, r2, args.lr, 
            args.num_scales, args.kernel_size, args.temp_conv_out_channels, 
            args.attention_reg_weight, args.dropout, time.strftime("%Y%m%d_%H%M%S")))

# Print final results
print('TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} MAPE {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}'.format(
    mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae))

# Test additional data if provided
if args.eval != '':
    testdata = np.loadtxt(open(f"data/{args.eval}.txt"), delimiter=',')
    testdata = (testdata - data_loader.min) / (data_loader.max - data_loader.min)
    testdata = torch.Tensor(testdata)
    testdata = testdata.unsqueeze(0)
    testdata = Variable(testdata)
    if args.cuda:
        testdata = testdata.to(device)
    
    model.eval()
    with torch.no_grad():
        # Need to modify this part to match MSTAGAT_Net interface
        output, attn_reg_loss = model(testdata, None)
        output = output[:, -1, :]  # Take the last time step prediction
        output = output.cpu().numpy() * (data_loader.max - data_loader.min) + data_loader.min
        output = output.squeeze(0)
    
    # Record additional test results
    output_list = output.tolist()
    with open(f"save/{args.eval}result.txt", "a") as f:
        f.write("\n" + "window" + str(args.window) + ", horizon" + str(args.horizon) + "\n")
        f.write(str(output_list))
        f.write('\n')

# Save comprehensive metrics to CSV
metrics = [mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae]
csv_filename = "result/metrics.csv"
header = ['model', 'dataset', 'window', 'horizon', 'seed', 'mae', 'std_mae', 'rmse', 'rmse_states', 
          'pcc', 'pcc_states', 'mape', 'r2', 'r2_states', 'var', 'var_states', 'peak_mae',
          'num_scales', 'kernel_size', 'temp_conv_out_channels', 'attention_reg_weight']

# Check if file exists or not to write header only once
try:
    with open(csv_filename, 'x', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
except FileExistsError:
    pass

# After evaluation, write a new row with the metrics and parameters
with open(csv_filename, mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['MSTAGAT_Net', args.dataset, args.window, args.horizon, args.seed, 
                    mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae,
                    args.num_scales, args.kernel_size, args.temp_conv_out_channels, args.attention_reg_weight])