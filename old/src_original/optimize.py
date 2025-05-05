import os
import random
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

# Import model and data loader
from model1 import MSAGATNet
from data import DataBasicLoader
from srcn.utils import *

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
        if index is not None:
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
            if index is not None:
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
    rmse = np.sqrt(mean_squared_error(y_true_states.flatten(), y_pred_states.flatten()))
    
    try:
        pcc, _ = pearsonr(y_true_states.flatten(), y_pred_states.flatten())
    except:
        pcc = 0.0
    
    return rmse, pcc

# =============================================================================
# Objective Function for Optuna
# =============================================================================
def objective(trial):
    # Define Args class with default training parameters
    class Args:
        dataset = trial.study.user_attrs.get('dataset', 'region785')
        sim_mat = trial.study.user_attrs.get('sim_mat', 'region-adj-49')
        window = trial.study.user_attrs.get('window', 20)
        horizon = trial.study.user_attrs.get('horizon', 5)
        train = 0.5
        val = 0.2
        test = 0.3
        epochs = DEFAULT_EPOCHS
        batch = DEFAULT_BATCH_SIZE
        seed = trial.study.user_attrs.get('seed', 42)
        patience = DEFAULT_PATIENCE
        save_dir = MODELS_DIR
        mylog = True
        cuda = True
        gpu = trial.study.user_attrs.get('gpu', 0)
        max_grad_norm = 1.0
        lr_patience = 100
        lr_factor = 0.5
        
        # Add all hyperparameters being optimized
        hidden_dim = None  # Will be set by trial.suggest
        attn_heads = None  # Will be set by trial.suggest
        attention_reg_weight = None  # Will be set by trial.suggest
        dropout = None  # Will be set by trial.suggest
        num_scales = None  # Will be set by trial.suggest
        kernel_size = None  # Will be set by trial.suggest
        temp_conv_out_channels = None  # Will be set by trial.suggest
        low_rank_dim = None  # Will be set by trial.suggest
        
        # Required by external modules
        extra = ''
        label = ''
        pcc = ''
        result = 0
        record = ''
    
    args = Args()
    logger.info(f"Trial {trial.number}: Starting with seed {args.seed}")
    
    # === HYPERPARAMETER SELECTION ===
    # Structural parameters for MSAGATNet
    args.hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
    args.attn_heads = trial.suggest_categorical("attn_heads", [2, 4, 8])
    args.low_rank_dim = trial.suggest_categorical("low_rank_dim", [4, 8, 16])
    args.num_scales = trial.suggest_int("num_scales", 2, 6)
    args.kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7, 9])
    args.temp_conv_out_channels = trial.suggest_categorical("temp_conv_out_channels", [8, 16, 32, 64])
    
    # Regularization parameters
    args.dropout = trial.suggest_float("dropout", 0.1, 0.5)
    args.attention_reg_weight = trial.suggest_float("attention_reg_weight", 1e-6, 1e-4, log=True)
    args.gru_layers = trial.suggest_int("gru_layers", 1, 3)
    
    # Optimization parameters
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    # Set seeds for reproducibility
    set_seeds(args.seed)
    
    # Device handling
    device = torch.device(f'cuda:{args.gpu}' if args.cuda and torch.cuda.is_available() else 'cpu')
    
    # Initialize data loader and model
    data_loader = DataBasicLoader(args)
    model = MSAGATNet(args, data_loader)
    
    # Log model parameters for debugging
    logger.info(f"Trial {trial.number} model parameters: " +
               f"hidden_dim={args.hidden_dim}, " +
               f"attn_heads={args.attn_heads}, " +
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
        patience=25, verbose=False  # Reduced patience, disable verbose output
    )
    
    # Training loop
    best_model_path = os.path.join(MODELS_DIR, f"trial_{trial.number}_best.pt")
    best_val_rmse = float('inf')
    best_val_pcc = 0.0
    best_epoch = 0
    no_improve_count = 0
    min_epochs = 100  # Minimum epochs before pruning to allow for more complete training
    
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
        
        # Report to Optuna for pruning, but only after minimum epochs
        if epoch > min_epochs:
            trial.report(val_rmse, epoch)
            # Only prune if significantly underperforming (2000+ RMSE after 100 epochs)
            if trial.should_prune() and val_rmse > 2000:
                logger.info(f"Trial {trial.number}: Pruned at epoch {epoch} with RMSE={val_rmse:.2f}")
                raise optuna.TrialPruned()
        
        # Track best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_pcc = val_pcc
            best_epoch = epoch
            no_improve_count = 0
            
            # Save best model
            torch.save(model.state_dict(), best_model_path)
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
        trial.set_user_attr("test_rmse", float(test_rmse))
        trial.set_user_attr("test_pcc", float(test_pcc))
    
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
    if args.continue_from is not None and os.path.exists(args.continue_from):
        try:
            study = joblib.load(args.continue_from)
            logger.info(f"Loaded existing study from {args.continue_from}")
        except Exception as e:
            logger.error(f"Error loading study: {str(e)}")
            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=25)
            )
    else:
        # Create new study
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=25)
        )
    
    # Add minimal study user attributes
    study.set_user_attr("seed", args.seed)
    study.set_user_attr("dataset", args.dataset)
    study.set_user_attr("sim_mat", args.sim_mat)
    study.set_user_attr("window", args.window)
    study.set_user_attr("horizon", args.horizon)
    study.set_user_attr("gpu", args.gpu)
    
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
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index to use')
    args = parser.parse_args()
    
    # Set global random seed
    set_seeds(args.seed)
    
    # Setup study
    study, study_name, storage_path = setup_study(args)
    
    logger.info(f"Study '{study_name}'")
    logger.info(f"Starting optimization with {args.trials} trials...")
    
    try:
        # Run optimization
        if args.parallel > 1:
            # Parallel optimization
            study.optimize(objective, n_trials=args.trials, n_jobs=args.parallel)
        else:
            # Single process
            study.optimize(objective, n_trials=args.trials)
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user!")
    finally:
        # Save the study
        joblib.dump(study, storage_path)
        logger.info(f"Study saved to {storage_path}")
    
    logger.info("\nStudy completed!")
    
    # Get best trial with error handling
    try:
        best_trial = study.best_trial
        
        # Log best trial information
        logger.info("Best trial:")
        logger.info(f"  Trial Number: {best_trial.number}")
        logger.info(f"  RMSE: {best_trial.value:.4f}")
        logger.info(f"  Test RMSE: {best_trial.user_attrs.get('test_rmse', 'Not recorded')}")
        logger.info("  Hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")
            
    except ValueError as e:
        logger.warning(f"Could not get best trial: {str(e)}")