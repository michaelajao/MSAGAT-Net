import os
import random
import math
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
import json
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
DEFAULT_MAX_PARAMS = 50000
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
from model import MSAGATNet
from data import DataBasicLoader
from utils import *

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create output directories
OUTPUT_DIR = os.path.join(parent_dir, "optim_results")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

for directory in [OUTPUT_DIR, MODELS_DIR, LOGS_DIR]:
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
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")

def count_parameters(model):
    """Count number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_efficiency_score(rmse, pcc, param_count, 
                              rmse_weight=0.65, 
                              pcc_weight=0.25, 
                              size_weight=0.1, 
                              size_threshold=100000):
    """Calculate an efficiency score balancing RMSE, PCC and model size."""
    normalized_pcc = 1.0 - pcc
    size_penalty = math.log10(max(param_count, 1000)) / math.log10(size_threshold)
    size_penalty = min(size_penalty, 1.0)
    
    score = (rmse * rmse_weight) + \
            (normalized_pcc * pcc_weight) + \
            (size_penalty * size_weight)
    
    return score

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
    
    metrics = {
        'loss': float(total_loss / n_samples),
        'rmse': float(rmse),
        'pcc': float(pcc)
    }
    
    return metrics

# =============================================================================
# Metric Tracking and Early Stopping
# =============================================================================
class MetricTracker:
    """Track multiple metrics during training with early stopping functionality"""
    def __init__(self, patience=DEFAULT_PATIENCE, verbose=False, delta=0, 
                 rmse_weight=0.7, pcc_weight=0.3, min_epoch=20,
                 trace_func=print, trial=None, prune_threshold_multiplier=3.0):
        self.metrics = {}
        self.best_values = {}
        self.best_epochs = {}
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.trace_func = trace_func
        self.trial = trial
        self.prune_threshold = float('inf')
        self.prune_threshold_multiplier = prune_threshold_multiplier
        self.rmse_weight = rmse_weight
        self.pcc_weight = pcc_weight
        self.min_epoch = min_epoch
    
    def update(self, metrics_dict, epoch):
        """Update all metrics and check for early stopping"""
        # First update all metrics
        for metric_name, value in metrics_dict.items():
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
                self.best_values[metric_name] = None
                self.best_epochs[metric_name] = None
            
            self.metrics[metric_name].append(value)
            
            # Update best value if improved
            is_better = False
            if self.best_values[metric_name] is None:
                is_better = True
            elif metric_name == 'pcc':  # Higher is better for PCC
                is_better = value > self.best_values[metric_name]
            else:  # Lower is better for other metrics
                is_better = value < self.best_values[metric_name]
                
            if is_better:
                self.best_values[metric_name] = value
                self.best_epochs[metric_name] = epoch
        
        # Get current RMSE and PCC
        rmse = metrics_dict.get('rmse', float('inf'))
        pcc = metrics_dict.get('pcc', 0.0)
        
        # Early stopping check based on weighted score
        score = (rmse * self.rmse_weight) - (pcc * self.pcc_weight)
        
        # Set initial best score
        if 'best_score' not in self.best_values:
            self.best_values['best_score'] = score
            self.best_epochs['best_score'] = epoch
            self.prune_threshold = rmse * self.prune_threshold_multiplier
            return False
            
        # Optuna pruning based on current RMSE
        if self.trial is not None and epoch > self.min_epoch:
            # Report to Optuna
            self.trial.report(score, epoch)
            
            # Check for pruning if current RMSE is much worse than best
            if rmse > self.prune_threshold:
                if self.verbose:
                    self.trace_func(f'Pruning trial: current RMSE ({rmse:.6f}) much worse than best')
                if self.trial.should_prune():
                    raise optuna.TrialPruned()
        
        # Check if score improved
        if score < self.best_values['best_score'] - self.delta:
            # Improvement in weighted score
            self.best_values['best_score'] = score
            self.best_epochs['best_score'] = epoch
            self.prune_threshold = self.best_values.get('rmse', float('inf')) * self.prune_threshold_multiplier
            self.counter = 0
            return False
        else:
            # No improvement in weighted score
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.min_epoch:
                self.early_stop = True
                return True
            return False
    
    def get_best(self, metric_name, lower_is_better=True):
        """Get best value for a metric"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return None
        if lower_is_better:
            return min(self.metrics[metric_name])
        else:
            return max(self.metrics[metric_name])
    
    def get_best_epoch(self, metric_name):
        """Get epoch where the best value for a metric occurred"""
        return self.best_epochs.get(metric_name, None)
    
    def get_latest(self, metric_name):
        """Get most recent value for a metric"""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return None
        return self.metrics[metric_name][-1]
    
    def get_history(self, metric_name=None):
        """Get full history for a metric or all metrics"""
        if metric_name:
            return self.metrics.get(metric_name, [])
        return self.metrics

# =============================================================================
# Objective Function for Optuna
# =============================================================================
def objective(trial):
    # Set up trial logger
    trial_log_file = os.path.join(LOGS_DIR, f"trial_{trial.number}.log")
    file_handler = logging.FileHandler(trial_log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    try:
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
            max_params = trial.study.user_attrs.get('max_params', DEFAULT_MAX_PARAMS)
            
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
        hidden_dim_options = [16, 24, 32, 48, 64]
        args.hidden_dim = trial.suggest_categorical("hidden_dim", hidden_dim_options)
        
        # Attention heads - use fixed options and validate compatibility after selection
        head_options = [2, 4, 8, 16]
        args.attn_heads = trial.suggest_categorical("attn_heads", head_options)
        
        # Validate that hidden_dim is divisible by attn_heads, otherwise adjust
        if args.hidden_dim % args.attn_heads != 0:
            # Find the largest valid head option that divides hidden_dim
            valid_heads = [h for h in head_options if args.hidden_dim % h == 0]
            if valid_heads:
                args.attn_heads = max(valid_heads)
                logger.info(f"Trial {trial.number}: Adjusted attn_heads to {args.attn_heads} to be compatible with hidden_dim={args.hidden_dim}")
            else:
                # Default fallback to 2 which often works with most dimensions
                args.attn_heads = 2
                logger.info(f"Trial {trial.number}: No compatible attn_heads found for hidden_dim={args.hidden_dim}, using default of 2")

        # Low-rank dimension
        low_rank_options = [4, 6, 8, 12, 16]
        args.low_rank_dim = trial.suggest_categorical("low_rank_dim", low_rank_options)

        # Temporal parameters
        args.num_scales = trial.suggest_int("num_scales", 2, 6)
        args.kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7])
        
        # Temporal convolution channels
        temp_channels_options = [8, 12, 16, 24, 32]
        args.temp_conv_out_channels = trial.suggest_categorical("temp_conv_out_channels", temp_channels_options)

        # Regularization parameters
        args.dropout = trial.suggest_float("dropout", 0.1, 0.4)
        args.attention_reg_weight = trial.suggest_categorical("attention_reg_weight", [0, 1e-6, 1e-5, 1e-4, 1e-3])
        
        # Optimization parameters
        lr = trial.suggest_categorical("lr", [5e-5, 1e-4, 2e-4, 5e-4])
        weight_decay = trial.suggest_categorical("weight_decay", [0, 1e-6, 1e-5, 1e-4, 5e-4])
        
        # Set seeds for reproducibility
        set_seeds(args.seed)
        
        # Device handling
        device = torch.device(f'cuda:{args.gpu}' if args.cuda and torch.cuda.is_available() else 'cpu')
        logger.info(f"Trial {trial.number}: Using device {device}")
        
        # Initialize data loader and model
        data_loader = DataBasicLoader(args)
        model = MSAGATNet(args, data_loader)
        model = model.to(device)
        
        # Calculate and store parameter count
        param_count = count_parameters(model)
        trial.set_user_attr("param_count", param_count)
        logger.info(f"Trial {trial.number}: Model has {param_count} parameters")
        
        # Check if model exceeds parameter budget
        if param_count > args.max_params:
            logger.info(f"Trial {trial.number}: Model size ({param_count}) exceeds budget ({args.max_params})")
            raise optuna.TrialPruned()
        
        # Setup optimizer
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=lr, weight_decay=weight_decay)
        
        # Add learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=args.lr_factor, 
            patience=50, verbose=True
        )
        
        # Initialize unified metrics tracker with early stopping
        metrics_tracker = MetricTracker(
            patience=args.patience, 
            verbose=True, 
            trial=trial, 
            rmse_weight=0.8, 
            pcc_weight=0.2, 
            min_epoch=25,
            trace_func=logger.info
        )
        
        # Save trial hyperparameters
        trial_params = {
            "hidden_dim": args.hidden_dim,
            "attn_heads": args.attn_heads,
            "low_rank_dim": args.low_rank_dim,
            "num_scales": args.num_scales,
            "kernel_size": args.kernel_size,
            "temp_conv_out_channels": args.temp_conv_out_channels,
            "dropout": args.dropout,
            "attention_reg_weight": args.attention_reg_weight,
            "lr": lr,
            "weight_decay": weight_decay,
            "param_count": param_count
        }
        
        # Training loop
        best_model_path = os.path.join(MODELS_DIR, f"trial_{trial.number}_best.pt")
        for epoch in range(1, args.epochs + 1):
            # Training phase
            train_loss = train_epoch(data_loader, model, optimizer, args, device)
            
            # Validation phase
            val_metrics = evaluate(data_loader, model, args, device, dataset='val')
            val_loss = val_metrics['loss']
            rmse = val_metrics['rmse']
            pcc = val_metrics['pcc']
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Calculate efficiency score
            efficiency_score = calculate_efficiency_score(
                rmse, pcc, param_count, 
                rmse_weight=0.7,
                pcc_weight=0.2,
                size_weight=0.1
            )
            
            # Track all metrics and check for early stopping
            metrics_dict = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "rmse": rmse,
                "pcc": pcc,
                "efficiency_score": efficiency_score
            }
            
            # Check for pruning based on RMSE target
            if rmse > 1200 and epoch > 50:  # Prune very poor performers early
                logger.info(f"Trial {trial.number}: Pruning with poor RMSE={rmse:.4f} at epoch {epoch}")
                raise optuna.TrialPruned()
            
            # Update metrics and check for early stopping
            should_stop = metrics_tracker.update(metrics_dict, epoch)
            
            # Log progress periodically
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Trial {trial.number}, Epoch {epoch}: "
                          f"RMSE={rmse:.4f}, PCC={pcc:.4f}, "
                          f"Params={param_count}, Score={efficiency_score:.4f}, "
                          f"LR={optimizer.param_groups[0]['lr']:.6f}")
            
            # Save model if it's the best so far for RMSE
            if metrics_tracker.get_best_epoch('rmse') == epoch:
                torch.save(model.state_dict(), best_model_path)
            
            # Early stopping check
            if should_stop:
                logger.info(f'Trial {trial.number}: Early stopping at epoch {epoch}')
                break
        
        # Get final best metrics
        best_rmse = float(metrics_tracker.get_best('rmse'))
        best_rmse_epoch = metrics_tracker.get_best_epoch('rmse')
        best_pcc = float(metrics_tracker.get_best('pcc', lower_is_better=False))
        best_pcc_epoch = metrics_tracker.get_best_epoch('pcc')
        best_efficiency = float(metrics_tracker.get_best('efficiency_score'))
        
        # Bonus for models beating target RMSE of 779
        if best_rmse < 779:
            logger.info(f"Trial {trial.number}: BEAT TARGET RMSE with {best_rmse:.4f}!")
            # Apply a reward factor to the efficiency score
            best_efficiency *= 0.8  # 20% bonus (lower score is better)
            
        # Load best model and evaluate on test set
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            test_metrics = evaluate(data_loader, model, args, device, dataset='test')
            test_rmse = test_metrics['rmse']
            test_pcc = test_metrics['pcc']
            
            logger.info(f"Trial {trial.number} test metrics: RMSE={test_rmse:.4f}, PCC={test_pcc:.4f}")
            
            # Store test metrics
            trial.set_user_attr("test_rmse", float(test_rmse))
            trial.set_user_attr("test_pcc", float(test_pcc))
        
        # Store trial attributes for analysis
        trial.set_user_attr("best_rmse", best_rmse)
        trial.set_user_attr("best_rmse_epoch", best_rmse_epoch)
        trial.set_user_attr("best_pcc", best_pcc)
        trial.set_user_attr("best_pcc_epoch", best_pcc_epoch)
        trial.set_user_attr("best_efficiency", best_efficiency)
        trial.set_user_attr("epochs_trained", epoch)
        trial.set_user_attr("final_rmse", float(metrics_tracker.get_latest("rmse")))
        trial.set_user_attr("final_pcc", float(metrics_tracker.get_latest("pcc")))
        trial.set_user_attr("beats_target", best_rmse < 779)
        
        # Save compact trial history (only every 10th epoch to save space)
        compact_history = {metric: values[::10] for metric, values in metrics_tracker.get_history().items()}
        history = {
            "params": trial_params,
            "epochs": epoch,
            "metrics_history": compact_history,
            "best_rmse": best_rmse,
            "best_pcc": best_pcc,
            "best_efficiency": best_efficiency
        }
        
        json_filename = os.path.join(OUTPUT_DIR, f"trial_{trial.number}_history.json")
        with open(json_filename, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Log results summary
        logger.info(f"Trial {trial.number} completed: RMSE={best_rmse:.4f}, PCC={best_pcc:.4f}, "
                   f"Params={param_count}, Efficiency={best_efficiency:.4f}")
        
        # Remove handler at the end
        logger.removeHandler(file_handler)
        
        # Return the efficiency score as our optimization target
        return best_efficiency
    
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {str(e)}", exc_info=True)
        logger.removeHandler(file_handler)
        raise

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
            logger.info(f"Continuing with {len(study.trials)} existing trials")
        except Exception as e:
            logger.error(f"Error loading study: {str(e)}")
            logger.info("Creating new study instead")
            study = optuna.create_study(
                study_name=study_name,
                direction="minimize",
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=25)
            )
    else:
        if args.continue_from is not None:
            logger.warning(f"Study file {args.continue_from} not found. Creating new study.")
        
        # Create new study
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=25)
        )
    
    # Add study user attributes
    study.set_user_attr("max_params", args.max_params)
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
    parser.add_argument('--max-params', type=int, default=DEFAULT_MAX_PARAMS, 
                        help='Maximum parameter count threshold')
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
    logger.info(f"Dataset: {args.dataset}, Window: {args.window}, Horizon: {args.horizon}")
    logger.info(f"Starting optimization with {args.trials} trials...")
    logger.info(f"Parameter budget: {args.max_params} parameters")
    logger.info(f"Parallel optimization processes: {args.parallel}")
    
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
        logger.info(f"  Efficiency Score: {best_trial.value:.4f}")
        logger.info(f"  RMSE: {best_trial.user_attrs.get('best_rmse', 'Not recorded')}")
        logger.info(f"  PCC: {best_trial.user_attrs.get('best_pcc', 'Not recorded')}")
        logger.info(f"  Test RMSE: {best_trial.user_attrs.get('test_rmse', 'Not recorded')}")
        logger.info(f"  Test PCC: {best_trial.user_attrs.get('test_pcc', 'Not recorded')}")
        logger.info(f"  Parameter Count: {best_trial.user_attrs.get('param_count', 'Not recorded')}")
        logger.info("  Hyperparameters:")
        for key, value in best_trial.params.items():
            logger.info(f"    {key}: {value}")
            
        # Save the study results as dataframe
        csv_filename = os.path.join(OUTPUT_DIR, f"{study_name}_results.csv")
        results_df = study.trials_dataframe()
        results_df.to_csv(csv_filename, index=False)
        logger.info(f"Results saved to {csv_filename}")
        
    except ValueError as e:
        logger.warning(f"Could not get best trial: {str(e)}")
        logger.warning("All trials may have been pruned or failed. Try adjusting search parameters.")
        
        # Try to get any completed trials for analysis
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed_trials:
            logger.info(f"Found {len(completed_trials)} completed trials.")
            best_manual = min(completed_trials, key=lambda t: t.value)
            logger.info(f"Best completed trial: #{best_manual.number} with score {best_manual.value:.4f}")
            
            # Save partial results
            csv_filename = os.path.join(OUTPUT_DIR, f"{study_name}_partial_results.csv")
            results_df = study.trials_dataframe()
            results_df.to_csv(csv_filename, index=False)
            logger.info(f"Partial results saved to {csv_filename}")
        else:
            logger.warning("No trials completed successfully.")