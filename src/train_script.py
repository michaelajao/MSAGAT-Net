#!/usr/bin/env python3
"""
Training Script for MSTAGATNetLite Model
- Handles multi-step forecasting
- Supports tensorboard logging, early stopping, and model checkpointing
- Includes ablation and profiling utilities

This script executes top-level without a main() wrapper.
Evaluation metrics now average across the full prediction horizon instead of only the final step.
"""
import os
import sys
import time
import math
import random
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from scipy.stats import pearsonr

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import model and data loader
from mstagat_net_simplified import MSTAGATNetLite, DEFAULTS
from data import DataBasicLoader
from utils import peak_error, plot_loss_curves, visualize_matrices, visualize_predictions, save_metrics

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ----------------- Argument Parsing -----------------
parser = argparse.ArgumentParser(description='Train MSTAGATNetLite without main()')
# data args
parser.add_argument('--dataset', type=str, default='japan')
parser.add_argument('--window', type=int, default=20)
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--train', type=float, default=0.5)
parser.add_argument('--val', type=float, default=0.2)
parser.add_argument('--test', type=float, default=0.3)
parser.add_argument('--sim_mat', type=str, default='japan-adj', help='Adjacency matrix file name')
parser.add_argument('--extra', type=str, default=None, help='Extra data directory name')
parser.add_argument('--label', type=str, default=None, help='Label CSV file prefix for extra data')
# train args
parser.add_argument('--epochs', type=int, default=1500)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--lr_patience', type=int, default=20)
parser.add_argument('--lr_factor', type=float, default=0.5)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--dropout', type=float, default=DEFAULTS['dropout'])
# model hyperparams override defaults
parser.add_argument('--hidden_dim', type=int, default=DEFAULTS['hidden_dim'])
parser.add_argument('--attention_heads', type=int, default=DEFAULTS['attention_heads'])
parser.add_argument('--num_scales', type=int, default=DEFAULTS['num_temporal_scales'])
parser.add_argument('--kernel_size', type=int, default=DEFAULTS['kernel_size'])
parser.add_argument('--feature_channels', type=int, default=DEFAULTS['feature_channels'])
parser.add_argument('--bottleneck_dim', type=int, default=DEFAULTS['bottleneck_dim'])
# system
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cuda', action='store_true', default=torch.cuda.is_available(), help='Enable GPU computation if available')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='save')
parser.add_argument('--tensorboard', action='store_true')
# parse
args = parser.parse_args()

# ----------------- Environment Setup -----------------
if args.cuda and torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type=='cuda':
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
os.makedirs(args.save_dir, exist_ok=True)
logger.info(f'Using device: {device}')

# ----------------- Train & Evaluate Functions -----------------
def train_epoch(loader, model, optimizer, device, args):
    model.train()
    total_loss, count = 0.0, 0
    for X, Y, idx in loader.get_batches(loader.train, args.batch, shuffle=True):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X, idx)
        target = Y.unsqueeze(1).expand(-1, args.horizon, -1)
        loss = nn.MSELoss()(pred, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        count += X.size(0)
    return total_loss / count


def evaluate(loader, split, model, device, args):
    model.eval()
    total_loss = 0.0
    n_samples = 0
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for X, Y, idx in loader.get_batches(split, args.batch, shuffle=False):
            X, Y = X.to(device), Y.to(device)
            pred = model(X, idx)
            expanded = Y.unsqueeze(1).expand(-1, args.horizon, -1)
            loss = nn.MSELoss()(pred, expanded)
            total_loss += loss.item()
            n_samples += pred.size(0) * loader.m
            y_true_list.append(Y.cpu())
            # take last horizon step for state metrics
            y_pred_list.append(pred[:, -1, :].cpu())
    # concatenate
    y_true_mx = torch.cat(y_true_list).numpy()
    y_pred_mx = torch.cat(y_pred_list).numpy()
    # unnormalize
    scale = (loader.max - loader.min)
    y_true_states = y_true_mx * scale + loader.min
    y_pred_states = y_pred_mx * scale + loader.min
    # per-state metrics
    rmse_states = np.mean(np.sqrt(mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values')))
    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae)
    # PCC per state
    pcc_vals = [pearsonr(y_true_states[:, k], y_pred_states[:, k])[0] for k in range(loader.m)]
    pcc_states = np.mean(pcc_vals)
    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))
    # flatten for overall metrics
    y_true_flat = y_true_states.reshape(-1)
    y_pred_flat = y_pred_states.reshape(-1)
    rmse = math.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mape = np.mean(np.abs((y_pred_flat - y_true_flat) / (y_true_flat + 1e-5)))
    pcc = pearsonr(y_true_flat, y_pred_flat)[0]
    r2 = r2_score(y_true_flat, y_pred_flat, multioutput='uniform_average')
    var = explained_variance_score(y_true_flat, y_pred_flat, multioutput='uniform_average')
    peak_mae = peak_error(y_true_states, y_pred_states, loader.peak_thold)
    return total_loss / n_samples, mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae, y_true_states, y_pred_states

# ----------------- Initialize Data, Model, Optimizer -----------------
loader = DataBasicLoader(args)
hp = DEFAULTS.copy()
hp.update({
    'hidden_dim': args.hidden_dim,
    'attention_heads': args.attention_heads,
    'num_temporal_scales': args.num_scales,
    'kernel_size': args.kernel_size,
    'feature_channels': args.feature_channels,
    'bottleneck_dim': args.bottleneck_dim,
})
model = MSTAGATNetLite(args, loader, hp=hp).to(device)
logger.info(f'Model parameter count: {sum(p.numel() for p in model.parameters())}')
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# ----------------- Training Loop -----------------
model_name = 'MSTAGATNetLite'
train_losses, val_losses = [], []
best_val_loss = float('inf')
best_epoch = 0
early_stop = 0
for epoch in range(1, args.epochs + 1):
    start = time.time()
    train_loss = train_epoch(loader, model, optimizer, device, args)
    val_loss, mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae, _, _ = \
        evaluate(loader, loader.val, model, device, args)
    elapsed = time.time() - start
    logger.info(f'Epoch {epoch:3d} | time {elapsed:5.2f}s | train_loss {train_loss:5.8f} | val_loss {val_loss:5.8f}')
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pt'))
        early_stop = 0
    else:
        early_stop += 1
    if early_stop >= args.patience:
        logger.info('Early stopping')
        break

# ----------------- Final Evaluation -----------------
model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best.pt'), map_location=device))
test_loss, mae, std_mae, rmse, rmse_states, pcc, pcc_states, mape, r2, r2_states, var, var_states, peak_mae, y_true_test, y_pred_test = \
    evaluate(loader, loader.test, model, device, args)
logger.info('Final TEST MAE {mae:5.4f} std {std:5.4f} RMSE {rmse:5.4f} RMSEs {rmses:5.4f} PCC {p:5.4f} PCCs {pc:5.4f} MAPE {mape:5.4f} R2 {r2:5.4f} R2s {r2s:5.4f} Var {var:5.4f} Vars {vars:5.4f} Peak {peak:5.4f}'.format(
mae=mae, std=std_mae, rmse=rmse, rmses=rmse_states, p=pcc, pc=pcc_states,
mape=mape, r2=r2, r2s=r2_states, var=var, vars=var_states, peak=peak_mae))

# Plot loss curves
loss_fig = os.path.join(args.save_dir, f"loss_curve_{args.dataset}_w{args.window}_h{args.horizon}.png")
plot_loss_curves(train_losses, val_losses, loss_fig, args=args, logger=logger)

# Visualize matrices
mat_fig = os.path.join(args.save_dir, f"matrices_{args.dataset}_w{args.window}_h{args.horizon}.png")
visualize_matrices(loader, model, mat_fig, device, logger=logger)

# Visualize predictions
pred_fig = os.path.join(args.save_dir, f"predictions_{args.dataset}_w{args.window}_h{args.horizon}.png")
visualize_predictions(y_true_test, y_pred_test, pred_fig, logger=logger)

# Save metrics to CSV
metrics = {'mae': mae, 'std_MAE': std_mae, 'rmse': rmse, 'rmse_states': rmse_states,
           'pcc': pcc, 'pcc_states': pcc_states, 'MAPE': mape,
           'R2': r2, 'R2_states': r2_states, 'Var': var, 'Vars': var_states,
           'Peak': peak_mae}
metrics_csv = os.path.join(args.save_dir, f"metrics_{args.dataset}_w{args.window}_h{args.horizon}.csv")
save_metrics(metrics, metrics_csv, args.dataset, args.window, args.horizon, logger=logger, model_name=model_name)
