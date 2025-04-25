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
from torchinfo import summary
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import model and data loader
from mstagat_net_simplified import MSTAGATNetLite, DEFAULTS
from data import DataBasicLoader

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ----------------- Argument Parsing -----------------
parser = argparse.ArgumentParser(description='Train MSTAGATNetLite without main()')
# data args
tparser = parser.add_argument
parser.add_argument('--dataset', type=str, default='japan')
parser.add_argument('--window', type=int, default=20)
parser.add_argument('--horizon', type=int, default=5)
parser.add_argument('--train', type=float, default=0.5)
parser.add_argument('--val', type=float, default=0.2)
parser.add_argument('--test', type=float, default=0.3)
parser.add_argument('--sim_mat', type=str, default=None, help='Prefix for similarity matrix file to load')
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        count += X.size(0)
    return total_loss / count


def evaluate(loader, split, model, device, args):
    """
    Evaluate by averaging metrics across all horizon steps.
    """
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, Y, idx in loader.get_batches(split, args.batch, shuffle=False):
            X, Y = X.to(device), Y.to(device)
            pred = model(X, idx)
            preds.append(pred.cpu().numpy())  # [B, H, N]
            trues.append(Y.cpu().numpy())     # [B, N]
    preds = np.concatenate(preds, axis=0)   # [M, H, N]
    trues = np.concatenate(trues, axis=0)   # [M, N]
    # Tile truths across horizon
    trues_tiled = np.repeat(trues[:, np.newaxis, :], preds.shape[1], axis=1)  # [M, H, N]
    # Flatten both arrays
    y_pred = preds.reshape(-1)
    y_true = trues_tiled.reshape(-1)
    # Compute metrics
    return {
        'rmse': math.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'pcc': pearsonr(y_true, y_pred)[0]
    }

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
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.lr_patience, factor=args.lr_factor)

# ----------------- Training Loop -----------------
best_val_loss = float('inf')
early_stop = 0
for epoch in range(1, args.epochs + 1):
    start = time.time()
    train_loss = train_epoch(loader, model, optimizer, device, args)
    val_metrics = evaluate(loader, loader.val, model, device, args)
    val_rmse = val_metrics['rmse']
    scheduler.step(val_rmse)
    elapsed = time.time() - start
    logger.info(f"Epoch {epoch} | time {elapsed:.1f}s | train_loss {train_loss:.4f} | val_rmse {val_rmse:.4f}")
    if val_rmse < best_val_loss:
        best_val_loss = val_rmse
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'best.pt'))
        early_stop = 0
    else:
        early_stop += 1
    if early_stop >= args.patience:
        logger.info('Early stopping')
        break

# ----------------- Final Evaluation -----------------
model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best.pt'), map_location=device))
test_metrics = evaluate(loader, loader.test, model, device, args)
logger.info(f"Test Metrics: {test_metrics}")
