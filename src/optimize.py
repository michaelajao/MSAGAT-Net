#!/usr/bin/env python3
"""
Hyperparameter Optimization for MSAGAT-Net

Uses Optuna for Bayesian hyperparameter optimization with support for
pruning, parallel trials, and automatic result saving.

Usage:
    python -m src.optimize --dataset japan --sim-mat japan-adj --trials 50
    python -m src.optimize --dataset japan --pruner hyperband --gpu 0
"""

import os
import sys
import json
import time
import random
import tempfile
import argparse
import logging
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import MSTAGAT_Net
from src.data import DataBasicLoader

# =============================================================================
# DEFAULT HYPERPARAMETERS
# =============================================================================
DEFAULT_EPOCHS = 1500
DEFAULT_BATCH_SIZE = 32
DEFAULT_PATIENCE = 100

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Output directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "optim_results")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_torch_save(state_dict, target_path, retries=5, backoff=0.2):
    """Safely save PyTorch state dict with atomic replace and retries."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    last_err = None
    for attempt in range(1, retries + 1):
        tmp_file = None
        try:
            tmp_fd, tmp_file = tempfile.mkstemp(
                dir=os.path.dirname(target_path), 
                prefix='.tmp_save_', 
                suffix='.pt'
            )
            os.close(tmp_fd)
            torch.save(state_dict, tmp_file)
            os.replace(tmp_file, target_path)
            return True
        except Exception as e:
            last_err = e
            logger.warning(f"Attempt {attempt}/{retries} to save model failed: {e}")
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except:
                    pass
            time.sleep(backoff * attempt)
    
    logger.error(f"Failed to save model after {retries} attempts: {last_err}")
    return False


def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def train_epoch(data_loader, model, optimizer, args, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0.0
    
    for inputs in data_loader.get_batches(data_loader.train, args.batch, True):
        X, Y, index = inputs[0], inputs[1], inputs[2]
        X, Y = X.to(device), Y.to(device)
        index = index.to(device)
            
        optimizer.zero_grad()
        output, attn_reg_loss = model(X, index)
        y_expanded = Y.unsqueeze(1).expand(-1, args.horizon, -1)
        loss = nn.MSELoss()(output, y_expanded) + attn_reg_loss
        loss.backward()
        
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
        optimizer.step()
        total_loss += loss.item()
        n_samples += output.size(0) * data_loader.m
        
    return total_loss / n_samples


def evaluate(data_loader, model, args, device, dataset='val'):
    """Evaluate model on validation or test set."""
    model.eval()
    total_loss = 0.0
    n_samples = 0.0
    y_true_list, y_pred_list = [], []
    
    data = data_loader.val if dataset == 'val' else data_loader.test
    
    with torch.no_grad():
        for inputs in data_loader.get_batches(data, args.batch, False):
            X, Y, index = inputs[0], inputs[1], inputs[2]
            X, Y = X.to(device), Y.to(device)
            index = index.to(device)
                
            output, attn_reg_loss = model(X, index)
            y_expanded = Y.unsqueeze(1).expand(-1, args.horizon, -1)
            loss = nn.MSELoss()(output, y_expanded) + attn_reg_loss
            total_loss += loss.item()
            n_samples += output.size(0) * data_loader.m
            y_true_list.append(Y.cpu())
            y_pred_list.append(output.cpu())
    
    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)[:, -1, :]
    
    # Convert to original scale
    y_true_states = y_true.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    y_pred_states = y_pred.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    
    y_true_flat = y_true_states.flatten()
    y_pred_flat = y_pred_states.flatten()
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    
    if np.std(y_true_flat) > 0 and np.std(y_pred_flat) > 0:
        pcc = float(np.corrcoef(y_true_flat, y_pred_flat)[0, 1])
    else:
        pcc = 0.0

    return rmse, pcc


# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================

def objective(trial):
    """Optuna objective function for hyperparameter optimization."""
    
    class Args:
        def __init__(self):
            self.dataset = trial.study.user_attrs.get('dataset', 'japan')
            self.sim_mat = trial.study.user_attrs.get('sim_mat', 'japan-adj')
            self.window = trial.study.user_attrs.get('window', 20)
            self.horizon = trial.study.user_attrs.get('horizon', 5)
            self.train = 0.5
            self.val = 0.2
            self.test = 0.3
            self.epochs = trial.study.user_attrs.get('epochs', DEFAULT_EPOCHS)
            self.batch = DEFAULT_BATCH_SIZE
            self.seed = trial.study.user_attrs.get('seed', 42)
            self.patience = DEFAULT_PATIENCE
            self.save_dir = MODELS_DIR
            self.cuda = True
            self.gpu = trial.study.user_attrs.get('gpu', 0)
            self.max_grad_norm = 1.0
            self.lr_factor = 0.5
            self.extra = ''
            self.label = ''
            self.pcc = ''
    
    args = Args()
    
    # Hyperparameter search space
    args.hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
    args.attention_heads = trial.suggest_categorical("attention_heads", [2, 4, 8])
    args.bottleneck_dim = trial.suggest_categorical("bottleneck_dim", [4, 8, 16])
    args.num_scales = trial.suggest_int("num_scales", 2, 6)
    args.kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    args.feature_channels = trial.suggest_categorical("feature_channels", [8, 16, 32])
    args.dropout = trial.suggest_float("dropout", 0.1, 0.5)
    args.attention_regularization_weight = trial.suggest_float(
        "attention_regularization_weight", 1e-6, 1e-4, log=True
    )
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # Validate head dimension compatibility
    if args.hidden_dim % args.attention_heads != 0:
        raise optuna.TrialPruned()

    set_seeds(args.seed)
    
    device = torch.device(f'cuda:{args.gpu}' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    # Initialize data and model
    data_loader = DataBasicLoader(args)
    model = MSTAGAT_Net(args, data_loader).to(device)
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, 
        weight_decay=weight_decay
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=25)
    
    # Training loop
    best_model_path = os.path.join(MODELS_DIR, f"trial_{trial.number}_best.pt")
    best_val_rmse = float('inf')
    best_epoch = 0
    no_improve_count = 0
    min_epochs = 100
    
    trial_timeout = trial.study.user_attrs.get('trial_timeout_seconds', 0)
    t0 = time.perf_counter()
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(data_loader, model, optimizer, args, device)
        val_rmse, val_pcc = evaluate(data_loader, model, args, device, dataset='val')
        scheduler.step(val_rmse)
        
        # Timeout check
        if trial_timeout and (time.perf_counter() - t0) > trial_timeout:
            logger.info(f"Trial {trial.number}: Timeout at epoch {epoch}")
            raise optuna.TrialPruned()

        # Pruning check
        if epoch > min_epochs:
            trial.report(val_rmse, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Track best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            no_improve_count = 0
            safe_torch_save(model.state_dict(), best_model_path)
        else:
            no_improve_count += 1
        
        # Early stopping
        if no_improve_count >= args.patience * 2 and epoch > min_epochs:
            break
    
    # Evaluate on test set
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_rmse, test_pcc = evaluate(data_loader, model, args, device, dataset='test')
        trial.set_user_attr("test_rmse", test_rmse)
        trial.set_user_attr("test_pcc", test_pcc)
    
    trial.set_user_attr("best_rmse", best_val_rmse)
    trial.set_user_attr("best_epoch", best_epoch)
    
    return best_val_rmse


# =============================================================================
# MAIN
# =============================================================================

def setup_study(args):
    """Setup and return an Optuna study."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = args.study_name or f"msagat_opt_{timestamp}"
    storage_path = os.path.join(OUTPUT_DIR, f"{study_name}.pkl")
    
    # Select pruner
    if args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=25)
    elif args.pruner == 'hyperband':
        pruner = optuna.pruners.HyperbandPruner(min_resource=50, max_resource=args.epochs, reduction_factor=3)
    else:
        pruner = optuna.pruners.NopPruner()

    if args.continue_from and os.path.exists(args.continue_from):
        study = joblib.load(args.continue_from)
        logger.info(f"Loaded existing study from {args.continue_from}")
    else:
        study = optuna.create_study(study_name=study_name, direction="minimize", pruner=pruner)
    
    # Store study attributes
    study.set_user_attr("seed", args.seed)
    study.set_user_attr("dataset", args.dataset)
    study.set_user_attr("sim_mat", args.sim_mat)
    study.set_user_attr("window", args.window)
    study.set_user_attr("horizon", args.horizon)
    study.set_user_attr("gpu", args.gpu)
    study.set_user_attr("epochs", args.epochs)
    study.set_user_attr("trial_timeout_seconds", args.trial_timeout_seconds)
    
    return study, study_name, storage_path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MSAGAT Hyperparameter Optimization')
    parser.add_argument('--trials', type=int, default=100, help='Number of trials')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--study-name', type=str, default=None, help='Study name')
    parser.add_argument('--continue-from', type=str, default=None, help='Continue from previous study')
    parser.add_argument('--parallel', type=int, default=1, help='Parallel processes')
    parser.add_argument('--dataset', type=str, default='japan', help='Dataset name')
    parser.add_argument('--sim-mat', dest='sim_mat', type=str, default='japan-adj', help='Adjacency matrix')
    parser.add_argument('--window', type=int, default=20, help='Window size')
    parser.add_argument('--horizon', type=int, default=5, help='Forecast horizon')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Max epochs per trial')
    parser.add_argument('--pruner', type=str, default='median', choices=['median', 'hyperband', 'none'])
    parser.add_argument('--trial-timeout-seconds', type=int, default=0, help='Per-trial timeout')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    set_seeds(args.seed)
    
    study, study_name, storage_path = setup_study(args)
    
    logger.info(f"Study '{study_name}': Starting {args.trials} trials...")
    
    if args.parallel > 1:
        study.optimize(objective, n_trials=args.trials, n_jobs=args.parallel)
    else:
        study.optimize(objective, n_trials=args.trials)

    # Save study
    joblib.dump(study, storage_path)
    logger.info(f"Study saved to {storage_path}")
    
    # Report best trial
    best_trial = study.best_trial
    logger.info(f"\nBest trial {best_trial.number}:")
    logger.info(f"  RMSE: {best_trial.value:.4f}")
    logger.info(f"  Test RMSE: {best_trial.user_attrs.get('test_rmse', 'N/A')}")
    logger.info("  Hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save JSON summary
    summary = {
        "study_name": study_name,
        "dataset": args.dataset,
        "horizon": args.horizon,
        "trials_total": len(study.trials),
        "trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "best_trial_number": best_trial.number,
        "best_val_rmse": best_trial.value,
        "best_params": best_trial.params,
        "test_rmse": best_trial.user_attrs.get("test_rmse"),
        "test_pcc": best_trial.user_attrs.get("test_pcc"),
    }
    
    json_path = os.path.join(OUTPUT_DIR, f"{study_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {json_path}")


if __name__ == "__main__":
    main()
