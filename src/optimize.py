import os
import json
import time
import random
import tempfile
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import argparse
import logging
import joblib
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

# =============================================================================
# TUNABLE PARAMETERS (DEFAULTS)
# =============================================================================
DEFAULT_EPOCHS = 1500
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.0005
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_PATIENCE = 100
DEFAULT_HIDDEN_DIM = 16
DEFAULT_ATTENTION_HEADS = 16
DEFAULT_ATTENTION_REG_WEIGHT = 1e-05
DEFAULT_DROPOUT = 0.249
DEFAULT_NUM_SCALES = 5
DEFAULT_KERNEL_SIZE = 3
DEFAULT_TEMP_CONV_OUT_CHANNELS = 12
DEFAULT_LOW_RANK_DIM = 6
DEFAULT_GRU_LAYERS = 1

# Set up project imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import basic models and data loader
from model import MSTAGAT_Net
from data import DataBasicLoader
from utils import *

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create output directories
OUTPUT_DIR = os.path.join(parent_dir, "optim_results")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

for directory in [OUTPUT_DIR, MODELS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# =============================================================================
# Utility Functions
# =============================================================================
def safe_torch_save(state_dict, target_path, retries=5, backoff=0.2):
    """Safely save a PyTorch state dict with atomic replace and retries.

    - Ensures parent directory exists
    - Writes to a temporary file first, then atomically replaces target
    - Retries on common intermittent Windows file errors
    Returns True on success, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
    except Exception as e:
        logger.warning(f"Failed to ensure directory for {target_path}: {e}")
        return False

    last_err = None
    for attempt in range(1, retries + 1):
        tmp_file = None
        try:
            # Create temp file in same directory to allow atomic replace
            tmp_fd, tmp_file = tempfile.mkstemp(dir=os.path.dirname(target_path), prefix='.tmp_save_', suffix='.pt')
            os.close(tmp_fd)
            torch.save(state_dict, tmp_file)
            # On Windows, os.replace is atomic if same drive/dir
            os.replace(tmp_file, target_path)
            return True
        except Exception as e:
            last_err = e
            logger.warning(f"Attempt {attempt}/{retries} to save model failed: {e}")
            # Clean up temp file if created
            try:
                if tmp_file and os.path.exists(tmp_file):
                    os.remove(tmp_file)
            except Exception:
                pass
            time.sleep(backoff * attempt)
    logger.error(f"Failed to save model to {target_path} after {retries} attempts. Last error: {last_err}")
    return False
def set_seeds(seed):
    """Set all random seeds to ensure reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")

# =============================================================================
# Training and Evaluation Functions
# =============================================================================
def train_epoch(data_loader, model, optimizer, args, device, scheduler=None):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.
    n_samples = 0.
    
    for inputs in data_loader.get_batches(data_loader.train, args.batch, True):
        X, Y = inputs[0], inputs[1]
        index = inputs[2]
        
        # Move data to device
        X = X.to(device)
        Y = Y.to(device)
        index = index.to(device)
            
        optimizer.zero_grad()
        output, attn_reg_loss = model(X, index)
        y_expanded = Y.unsqueeze(1).expand(-1, args.horizon, -1)
        loss = nn.MSELoss()(output, y_expanded) + attn_reg_loss
        loss.backward()
        
        # Apply gradient clipping
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
        optimizer.step()
        total_loss += loss.item()
        n_samples += (output.size(0) * data_loader.m)
        
    return total_loss / n_samples

def evaluate(data_loader, model, args, device, dataset='val'):
    """Evaluate model on validation or test set"""
    model.eval()
    total_loss = 0.
    n_samples = 0.
    y_true_list = []
    y_pred_list = []
    
    # Select dataset
    if dataset == 'val':
        data = data_loader.val
    elif dataset == 'test':
        data = data_loader.test
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    with torch.no_grad():
        for inputs in data_loader.get_batches(data, args.batch, False):
            X, Y = inputs[0], inputs[1]
            index = inputs[2]
            
            # Move data to device
            X = X.to(device)
            Y = Y.to(device)
            index = index.to(device)
                
            output, attn_reg_loss = model(X, index)
            y_expanded = Y.unsqueeze(1).expand(-1, args.horizon, -1)
            loss = nn.MSELoss()(output, y_expanded) + attn_reg_loss
            total_loss += loss.item()
            n_samples += (output.size(0) * data_loader.m)
            y_true_list.append(Y.cpu())
            y_pred_list.append(output.cpu())
    
    y_true = torch.cat(y_true_list)
    y_pred = torch.cat(y_pred_list)[:, -1, :]  # use the last prediction in the horizon
    
    # Convert predictions to original scale
    y_true_states = y_true.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    y_pred_states = y_pred.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    
    # Calculate metrics
    y_true_flat = y_true_states.flatten()
    y_pred_flat = y_pred_states.flatten()
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    # Pearson without try/except: guard zero variance
    if np.std(y_true_flat) > 0 and np.std(y_pred_flat) > 0:
        # Use numpy corrcoef for stability
        pcc = float(np.corrcoef(y_true_flat, y_pred_flat)[0, 1])
    else:
        pcc = 0.0

    return rmse, pcc

# =============================================================================
# Objective Function for Optuna
# =============================================================================
def objective(trial):
    # Define Args class with default training parameters
    class Args:
        def __init__(self):
            self.dataset = trial.study.user_attrs.get('dataset', 'region785')
            self.sim_mat = trial.study.user_attrs.get('sim_mat', 'region-adj-49')
            self.window = trial.study.user_attrs.get('window', 20)
            self.horizon = trial.study.user_attrs.get('horizon', 5)
            self.model_type = trial.study.user_attrs.get('model_type', 'msagat')
            self.train = 0.5
            self.val = 0.2
            self.test = 0.3
            self.epochs = DEFAULT_EPOCHS
            self.batch = DEFAULT_BATCH_SIZE
            self.seed = trial.study.user_attrs.get('seed', 42)
            self.patience = DEFAULT_PATIENCE
            self.save_dir = MODELS_DIR
            self.mylog = True
            self.cuda = True
            self.gpu = trial.study.user_attrs.get('gpu', 0)
            self.max_grad_norm = 1.0
            self.lr_patience = 100
            self.lr_factor = 0.5
            
            # Initialize hyperparameter attributes (will be set by trial)
            self.hidden_dim = None
            self.attention_heads = None
            self.bottleneck_dim = None
            self.num_scales = None
            self.kernel_size = None
            self.feature_channels = None
            self.dropout = None
            self.attention_regularization_weight = None
            
            # Required by external modules
            self.extra = ''
            self.label = ''
            self.pcc = ''
            self.result = 0
            self.record = ''
    
    args = Args()
    
    # === HYPERPARAMETER SELECTION ===
    # Structural parameters
    args.hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
    args.attention_heads = trial.suggest_categorical("attention_heads", [2, 4, 8])
    
    # For Linformer, use larger bottleneck_dim to avoid dimension mismatch issues
    if args.model_type == 'linformer':
        args.bottleneck_dim = trial.suggest_categorical("bottleneck_dim", [8, 16, 32])
    else:
        args.bottleneck_dim = trial.suggest_categorical("bottleneck_dim", [4, 8, 16])
    
    args.num_scales = trial.suggest_int("num_scales", 2, 6)
    args.kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
    args.feature_channels = trial.suggest_categorical("feature_channels", [8, 16, 32])
    
    # Regularization parameters
    args.dropout = trial.suggest_float("dropout", 0.1, 0.5)
    args.attention_regularization_weight = trial.suggest_float("attention_regularization_weight", 1e-6, 1e-4, log=True)
    
    # Optimization parameters
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # Prune invalid head dimension combos early
    if args.hidden_dim % args.attention_heads != 0:
        raise optuna.TrialPruned()

    logger.info(f"Trial {trial.number}: Starting with seed {args.seed}")
    
    # Set seeds for reproducibility
    set_seeds(args.seed)
    
    # Device handling
    device = torch.device(f'cuda:{args.gpu}' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    # Initialize data loader and model
    data_loader = DataBasicLoader(args)
    if args.model_type == 'linformer':
        from model_true_linformer import MSTAGAT_Net_Linformer
        model = MSTAGAT_Net_Linformer(args, data_loader)
    else:
        model = MSTAGAT_Net(args, data_loader)
    
    # Log model parameters for debugging
    logger.info(f"Trial {trial.number} model parameters: " +
               f"hidden_dim={args.hidden_dim}, " +
               f"attention_heads={args.attention_heads}, " +
               f"num_scales={args.num_scales}, " +
               f"kernel_size={args.kernel_size}, " +
               f"dropout={args.dropout:.4f}")
    
    model = model.to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                         lr=lr, weight_decay=weight_decay)
    
    # Add learning rate scheduler with reduced patience to allow faster adaptation
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor, 
        patience=25  # Reduced patience
    )
    
    # Training loop
    best_model_path = os.path.join(MODELS_DIR, f"trial_{trial.number}_best.pt")
    best_val_rmse = float('inf')
    best_val_pcc = 0.0
    best_epoch = 0
    no_improve_count = 0
    min_epochs = 100  # Minimum epochs before pruning to allow for more complete training

    # Per-trial timeout (seconds). If exceeded, prune the trial immediately.
    trial_timeout_seconds = trial.study.user_attrs.get('trial_timeout_seconds', 0)
    t0 = time.perf_counter()
    
    for epoch in range(1, args.epochs + 1):
        # Training phase
        train_loss = train_epoch(data_loader, model, optimizer, args, device)
        
        # Validation phase
        val_rmse, val_pcc = evaluate(data_loader, model, args, device, dataset='val')
        
        # Update scheduler
        scheduler.step(val_rmse)
        
        # Check current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log detailed information about this trial
        if epoch == 1 or epoch % 50 == 0:
            logger.info(f"Trial {trial.number}, Epoch {epoch}: "
                      f"LR={current_lr:.6f}, "
                      f"Train Loss={train_loss:.4f}, Val RMSE={val_rmse:.2f}")
        
        # Check per-trial timeout regardless of epoch count
        if trial_timeout_seconds and (time.perf_counter() - t0) > trial_timeout_seconds:
            logger.info(f"Trial {trial.number}: Pruned due to timeout at epoch {epoch}")
            raise optuna.TrialPruned()

        # Report to Optuna for pruning after minimum epochs
        if epoch > min_epochs:
            trial.report(val_rmse, epoch)
            if trial.should_prune():
                logger.info(f"Trial {trial.number}: Pruned at epoch {epoch} with RMSE={val_rmse:.2f}")
                raise optuna.TrialPruned()
        
        # Track best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_pcc = val_pcc
            best_epoch = epoch
            no_improve_count = 0
            
            # Save best model safely
            if not safe_torch_save(model.state_dict(), best_model_path):
                logger.warning(f"Could not save best model to {best_model_path}, continuing without crash.")
        else:
            no_improve_count += 1
        
        # Simple early stopping with increased patience to avoid premature stopping
        if no_improve_count >= args.patience * 2 and epoch > min_epochs:
            logger.info(f'Trial {trial.number}: Early stopping at epoch {epoch}')
            break
        
        # Reduced logging frequency
        if epoch % 100 == 0 or epoch == 1:
            logger.info(f"Trial {trial.number}, Epoch {epoch}: "
                      f"RMSE={val_rmse:.2f}, PCC={val_pcc:.4f}")
    
    # Trial successfully completed
    logger.info(f"Trial {trial.number}: Completed with best RMSE={best_val_rmse:.2f} at epoch {best_epoch}")
    
    # Load best model and evaluate on test set
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_rmse, test_pcc = evaluate(data_loader, model, args, device, dataset='test')
        logger.info(f"Trial {trial.number} test metrics: RMSE={test_rmse:.2f}, PCC={test_pcc:.4f}")
        
        # Store minimal test metrics
        trial.set_user_attr("test_rmse", test_rmse)
        trial.set_user_attr("test_pcc", test_pcc)
    
    # Store minimal trial attributes
    trial.set_user_attr("best_rmse", best_val_rmse)
    trial.set_user_attr("best_epoch", best_epoch)
    
    # Return the RMSE as our optimization target
    return best_val_rmse

# =============================================================================
# Main Function
# =============================================================================
def setup_study(args):
    """Setup and return an Optuna study"""
    # Create unique study name if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.study_name is None:
        study_name = f"msagat_opt_{timestamp}"
    else:
        study_name = args.study_name
    
    # Storage for the study
    storage_path = os.path.join(OUTPUT_DIR, f"{study_name}.pkl")
    
    # Setup study based on whether continuing from existing or creating new
    # Select pruner
    if args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=25)
    elif args.pruner == 'hyperband':
        # Use epochs as the resource; reduction factor 3 is a common default
        pruner = optuna.pruners.HyperbandPruner(min_resource=50, max_resource=args.epochs, reduction_factor=3)
    else:
        pruner = optuna.pruners.NopPruner()

    if args.continue_from is not None and os.path.exists(args.continue_from):
        study = joblib.load(args.continue_from)
        logger.info(f"Loaded existing study from {args.continue_from}")
    else:
        # Create new study
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            pruner=pruner
        )
    
    # Add minimal study user attributes
    study.set_user_attr("seed", args.seed)
    study.set_user_attr("dataset", args.dataset)
    study.set_user_attr("sim_mat", args.sim_mat)
    study.set_user_attr("window", args.window)
    study.set_user_attr("horizon", args.horizon)
    study.set_user_attr("model_type", args.model)
    study.set_user_attr("gpu", args.gpu)
    study.set_user_attr("trial_timeout_seconds", args.trial_timeout_seconds)
    study.set_user_attr("pruner", args.pruner)
    study.set_user_attr("epochs", args.epochs)
    
    return study, study_name, storage_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MSAGAT Hyperparameter Optimization')
    parser.add_argument('--trials', type=int, default=100, 
                        help='Number of optimization trials')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--study-name', type=str, default=None, 
                        help='Study name (default: auto-generated)')
    parser.add_argument('--continue-from', type=str, default=None, 
                        help='Continue from a previous study (storage path)')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel optimization processes')
    parser.add_argument('--dataset', type=str, default='japan',
                        help='Dataset name')
    parser.add_argument('--sim-mat', '--sim_mat', dest='sim_mat', type=str, default='japan-adj',
                        help='Adjacency matrix name')
    parser.add_argument('--window', type=int, default=20,
                        help='Window size')
    parser.add_argument('--horizon', type=int, default=5,
                        help='Forecast horizon')
    parser.add_argument('--model', type=str, default='msagat', 
                        choices=['msagat', 'linformer'],
                        help='Model type to optimize (msagat or linformer)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Max training epochs per trial')
    # New CLI options
    parser.add_argument('--pruner', type=str, default='median', choices=['median', 'hyperband', 'none'],
                        help='Optuna pruner to use (default: median)')
    parser.add_argument('--trial-timeout-seconds', type=int, default=0,
                        help='Per-trial timeout in seconds (0 disables)')
    parser.add_argument('--post-eval-seeds', type=int, default=0,
                        help='After optimization, re-train best params with this many seeds and summarize test metrics')

    args = parser.parse_args()
    
    # Set global random seed
    set_seeds(args.seed)
    
    # Setup study
    study, study_name, storage_path = setup_study(args)
    
    logger.info(f"Study '{study_name}'")
    logger.info(f"Starting optimization with {args.trials} trials...")
    
    # Run optimization
    if args.parallel > 1:
        study.optimize(objective, n_trials=args.trials, n_jobs=args.parallel)
    else:
        study.optimize(objective, n_trials=args.trials)

    # Save the study
    joblib.dump(study, storage_path)
    logger.info(f"Study saved to {storage_path}")
    
    logger.info("\nStudy completed!")
    
    # Get best trial and log
    best_trial = study.best_trial
    logger.info("Best trial:")
    logger.info(f"  Trial Number: {best_trial.number}")
    logger.info(f"  RMSE: {best_trial.value:.4f}")
    logger.info(f"  Test RMSE: {best_trial.user_attrs.get('test_rmse', 'Not recorded')}")
    logger.info("  Hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save JSON summary for quick table generation
    summary = {
        "study_name": study_name,
        "dataset": args.dataset,
        "sim_mat": args.sim_mat,
        "window": args.window,
        "horizon": args.horizon,
        "model": args.model,
        "trials_total": len(study.trials),
        "trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "best_trial_number": best_trial.number,
        "best_val_rmse": best_trial.value,
        "best_epoch": best_trial.user_attrs.get("best_epoch"),
        "best_params": best_trial.params,
        "test_rmse": best_trial.user_attrs.get("test_rmse"),
        "test_pcc": best_trial.user_attrs.get("test_pcc"),
        "pruner": study.user_attrs.get("pruner"),
        "trial_timeout_seconds": study.user_attrs.get("trial_timeout_seconds"),
    }
    json_path = os.path.join(OUTPUT_DIR, f"{study_name}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {json_path}")

    # Optional: re-train best hyperparameters across multiple seeds and summarize
    if args.post_eval_seeds and args.post_eval_seeds > 0:
        seeds = [args.seed + i for i in range(args.post_eval_seeds)]
        test_rmses = []
        test_pccs = []

        for s in seeds:
            set_seeds(s)

            class EvalArgs:
                def __init__(self):
                    self.dataset = args.dataset
                    self.sim_mat = args.sim_mat
                    self.window = args.window
                    self.horizon = args.horizon
                    self.model_type = args.model
                    self.train = 0.5
                    self.val = 0.2
                    self.test = 0.3
                    self.epochs = DEFAULT_EPOCHS
                    self.batch = DEFAULT_BATCH_SIZE
                    self.seed = s
                    self.patience = DEFAULT_PATIENCE
                    self.save_dir = MODELS_DIR
                    self.mylog = True
                    self.cuda = True
                    self.gpu = args.gpu
                    self.max_grad_norm = 1.0
                    self.lr_patience = 100
                    self.lr_factor = 0.5

                    # hyperparameters
                    self.hidden_dim = best_trial.params["hidden_dim"]
                    self.attention_heads = best_trial.params["attention_heads"]
                    self.bottleneck_dim = best_trial.params["bottleneck_dim"]
                    self.num_scales = best_trial.params["num_scales"]
                    self.kernel_size = best_trial.params["kernel_size"]
                    self.feature_channels = best_trial.params["feature_channels"]
                    self.dropout = best_trial.params["dropout"]
                    self.attention_regularization_weight = best_trial.params["attention_regularization_weight"]

                    # Not optimized here; fixed reasonable defaults
                    self.extra = ''
                    self.label = ''
                    self.pcc = ''
                    self.result = 0
                    self.record = ''

            eargs = EvalArgs()
            device = torch.device(f'cuda:{eargs.gpu}' if eargs.cuda and torch.cuda.is_available() else 'cpu')
            data_loader = DataBasicLoader(eargs)
            if eargs.model_type == 'linformer':
                from model_true_linformer import MSTAGAT_Net_Linformer
                model = MSTAGAT_Net_Linformer(eargs, data_loader)
            else:
                model = MSTAGAT_Net(eargs, data_loader)
            model = model.to(device)

            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=best_trial.params["lr"],
                                   weight_decay=best_trial.params["weight_decay"])
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=eargs.lr_factor, patience=25)

            best_val = float('inf')
            best_path = os.path.join(MODELS_DIR, f"post_eval_seed_{s}_best.pt")
            no_improve = 0
            for epoch in range(1, eargs.epochs + 1):
                _ = train_epoch(data_loader, model, optimizer, eargs, device)
                val_rmse, _ = evaluate(data_loader, model, eargs, device, dataset='val')
                scheduler.step(val_rmse)
                if val_rmse < best_val:
                    best_val = val_rmse
                    no_improve = 0
                    if not safe_torch_save(model.state_dict(), best_path):
                        logger.warning(f"Post-eval: could not save model for seed {s} to {best_path}")
                else:
                    no_improve += 1
                if no_improve >= eargs.patience and epoch > 100:
                    break

            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location=device))
            else:
                logger.warning(f"Post-eval: best model file missing at {best_path}; using last epoch weights for evaluation.")
            test_rmse, test_pcc = evaluate(data_loader, model, eargs, device, dataset='test')
            test_rmses.append(test_rmse)
            test_pccs.append(test_pcc)

        post_eval = {
            "seeds": seeds,
            "test_rmse_mean": float(np.mean(test_rmses)) if test_rmses else None,
            "test_rmse_std": float(np.std(test_rmses)) if test_rmses else None,
            "test_pcc_mean": float(np.mean(test_pccs)) if test_pccs else None,
            "test_pcc_std": float(np.std(test_pccs)) if test_pccs else None,
        }

        # Update and resave JSON summary
        summary["post_eval"] = post_eval
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Post-eval summary updated in {json_path}")