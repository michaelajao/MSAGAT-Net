import os
import time
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import joblib
from math import sqrt
import json  # Added import for json

# Project modules
from data import DataBasicLoader
from model import MSTAGAT_Net
from utils import save_metrics, peak_error

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Determine project root and output directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPTIM_DIR = os.path.join(BASE_DIR, "optim_results")
MODELS_DIR = os.path.join(OPTIM_DIR, "models")
os.makedirs(OPTIM_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(data_loader, model, optimizer, device, args):
    """Train model for one epoch"""
    model.train()
    total_loss = 0.0
    n_samples = 0.0

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
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_samples += output.size(0) * data_loader.m

    return float(total_loss / n_samples)


def evaluate(data_loader, model, device, args, dataset="val"):
    """Evaluate model on validation or test set"""
    model.eval()
    total_loss = 0.0
    n_samples = 0.0
    y_pred_mx = []
    y_true_mx = []
    x_value_mx = []

    # Select dataset
    if dataset == "val":
        data = data_loader.val
    elif dataset == "test":
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
            n_samples += output.size(0) * data_loader.m

            x_value_mx.append(X.cpu())
            y_true_mx.append(Y.cpu())
            y_pred_mx.append(output.cpu())

    # Process for metrics calculation
    x_value_mx = torch.cat(x_value_mx)
    y_true_mx = torch.cat(y_true_mx)
    y_pred_mx = torch.cat(y_pred_mx)
    y_pred_mx = y_pred_mx[:, -1, :]  # final timestep prediction

    # Denormalize
    x_value_states = (
        x_value_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    )
    y_true_states = (
        y_true_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    )
    y_pred_states = (
        y_pred_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    )

    # Flatten for metrics
    y_true_flat = y_true_states.reshape(-1)
    y_pred_flat = y_pred_states.reshape(-1)

    # Calculate metrics
    rmse = sqrt(mean_squared_error(y_true_flat, y_pred_flat))

    try:
        pcc = pearsonr(y_true_flat, y_pred_flat)[0]
    except:
        pcc = 0.0

    return {
        "loss": total_loss / n_samples,
        "rmse": rmse,
        "pcc": pcc,
        "y_true": y_true_states,
        "y_pred": y_pred_states,
    }


def objective(trial):
    """Optuna objective function that trains and evaluates a model with given hyperparameters"""

    # Create a custom args object with suggested hyperparameters
    class Args:
        dataset = args.dataset
        sim_mat = args.sim_mat
        window = args.window
        horizon = args.horizon
        train = args.train
        val = args.val
        test = args.test
        epochs = args.epochs
        batch = args.batch
        patience = args.patience
        save_dir = MODELS_DIR
        mylog = False  # Disable TensorBoard for trials
        cuda = torch.cuda.is_available()
        gpu = args.gpu

        # Hyperparameters to optimize
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64])
        attention_heads = trial.suggest_categorical("attention_heads", [2, 4, 8, 16])
        attention_regularization_weight = trial.suggest_float(
            "attention_regularization_weight", 1e-5, 1e-2, log=True
        )
        num_scales = trial.suggest_int("num_scales", 3, 7)
        kernel_size = trial.suggest_categorical("kernel_size", [3, 5, 7, 9])
        feature_channels = trial.suggest_categorical(
            "feature_channels", [8, 12, 16, 32]
        )
        bottleneck_dim = trial.suggest_categorical("bottleneck_dim", [4, 6, 8, 12])

        # Required by model but not being optimized
        seed = args.seed
        extra = ""
        label = ""
        pcc = ""
        result = 0
        record = ""
        start_date = "2020-01-01"

    trial_args = Args()

    # Set individual seed for this trial
    trial_seed = args.seed + trial.number
    set_seed(trial_seed)

    # Setup device
    device = (
        torch.device(f"cuda:{args.gpu}")
        if args.cuda and torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Initialize data loader and model
    data_loader = DataBasicLoader(trial_args)
    model = MSTAGAT_Net(trial_args, data_loader).to(device)

    # Log model parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trial {trial.number} - Model parameters: {n_params}")
    logger.info(
        f"Trial {trial.number} - Hyperparameters: lr={trial_args.lr}, dropout={trial_args.dropout}, hidden_dim={trial_args.hidden_dim}"
    )

    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=trial_args.lr,
        weight_decay=trial_args.weight_decay,
    )
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", factor=0.5, patience=20, verbose=False
    # )

    # Training variables
    best_val_rmse = float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    bad_counter = 0

    # Save path for best model in this trial
    best_model_path = os.path.join(MODELS_DIR, f"trial_{trial.number}_best.pt")

    # Training loop
    for epoch in range(1, trial_args.epochs + 1):
        # Training
        train_loss = train_epoch(data_loader, model, optimizer, device, trial_args)

        # Validation
        val_metrics = evaluate(data_loader, model, device, trial_args, dataset="val")
        val_loss = val_metrics["loss"]
        val_rmse = val_metrics["rmse"]
        val_pcc = val_metrics["pcc"]

        # Update scheduler
        # scheduler.step(val_loss)

        # Log progress (less frequently for optimization)
        if epoch == 1 or epoch % 50 == 0 or (epoch < 10 and epoch % 2 == 0):
            logger.info(
                f"Trial {trial.number}, Epoch {epoch}: loss={val_loss:.6f}, RMSE={val_rmse:.4f}, PCC={val_pcc:.4f}"
            )

        # Report to Optuna for pruning
        trial.report(val_rmse, epoch)
        if trial.should_prune() and epoch > trial_args.patience // 2:
            raise optuna.exceptions.TrialPruned()

        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_loss = val_loss
            best_epoch = epoch
            bad_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            bad_counter += 1

        # Early stopping
        if bad_counter >= trial_args.patience:
            logger.info(f"Trial {trial.number}: Early stopping at epoch {epoch}")
            break

    # Load best model and evaluate on test set
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_metrics = evaluate(data_loader, model, device, trial_args, dataset="test")
        test_rmse = test_metrics["rmse"]
        test_pcc = test_metrics["pcc"]

        logger.info(
            f"Trial {trial.number} completed: Best val RMSE={best_val_rmse:.4f} at epoch {best_epoch}"
        )
        logger.info(
            f"Trial {trial.number} test results: RMSE={test_rmse:.4f}, PCC={test_pcc:.4f}"
        )

        # Store important metrics as trial attributes
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("test_rmse", test_rmse)
        trial.set_user_attr("test_pcc", test_pcc)
        trial.set_user_attr("param_count", n_params)
    else:
        logger.warning(
            f"Trial {trial.number}: No model saved, using validation metrics"
        )

    return best_val_rmse


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for MSTAGAT-Net"
    )
    parser.add_argument("--dataset", type=str, default="japan")
    parser.add_argument("--sim_mat", type=str, default="japan-adj")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--train", type=float, default=0.5)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=500)  # Reduced for optimization
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=50)  # Reduced for optimization
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cuda", action="store_true", default=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--study_name", type=str, default=None)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--n_jobs", type=int, default=1)
    global args
    args = parser.parse_args()

    # Set up seed for reproducibility
    set_seed(args.seed)

    # Build study name from parameters if not provided
    if args.study_name is None:
        args.study_name = f"MSTAGAT_Net_{args.dataset}_w{args.window}_h{args.horizon}_{int(time.time())}"

    # Create storage path
    storage_path = os.path.join(OPTIM_DIR, f"{args.study_name}.db")
    if args.storage is None:
        args.storage = f"sqlite:///{storage_path}"

    logger.info(f"Starting hyperparameter optimization study: {args.study_name}")
    logger.info(f"Storage: {args.storage}")
    logger.info(f"Number of trials: {args.trials}")

    # Create Optuna study
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        pruner=pruner,
        load_if_exists=True,
    )

    try:
        if args.n_jobs > 1:
            study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs)
        else:
            study.optimize(objective, n_trials=args.trials)
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user.")

    # Save study for later analysis
    joblib.dump(study, os.path.join(OPTIM_DIR, f"{args.study_name}.pkl"))

    # Print best hyperparameters
    logger.info("Optimization completed.")
    logger.info("Best trial:")
    best_trial = study.best_trial
    logger.info(f"  Trial number: {best_trial.number}")
    logger.info(f"  RMSE: {best_trial.value:.4f}")
    logger.info(
        f"  Test RMSE: {best_trial.user_attrs.get('test_rmse', 'Not recorded')}"
    )
    logger.info(f"  Test PCC: {best_trial.user_attrs.get('test_pcc', 'Not recorded')}")
    logger.info(
        f"  Best epoch: {best_trial.user_attrs.get('best_epoch', 'Not recorded')}"
    )
    logger.info(
        f"  Parameters count: {best_trial.user_attrs.get('param_count', 'Not recorded')}"
    )
    logger.info("  Hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save best parameters to a JSON file
    best_params_path = os.path.join(OPTIM_DIR, f"{args.study_name}_best_params.json")
    with open(best_params_path, 'w') as f:
        json.dump(best_trial.params, f, indent=4)
    logger.info(f"Best hyperparameters saved to: {best_params_path}")

    # Generate command for best configuration
    best_params = best_trial.params
    best_cmd = [
        "python src/train.py",
        f"--dataset {args.dataset}",
        f"--sim_mat {args.sim_mat}",
        f"--window {args.window}",
        f"--horizon {args.horizon}",
        f"--train {args.train}",
        f"--val {args.val}",
        f"--test {args.test}",
        f"--epochs 1500",
        f"--batch {args.batch}",
        f"--lr {best_params['lr']}",
        f"--weight_decay {best_params['weight_decay']}",
        f"--dropout {best_params['dropout']}",
        f"--patience 100",
        f"--hidden_dim {best_params['hidden_dim']}",
        f"--attention_heads {best_params['attention_heads']}",
        f"--attention_regularization_weight {best_params['attention_regularization_weight']}",
        f"--num_scales {best_params['num_scales']}",
        f"--kernel_size {best_params['kernel_size']}",
        f"--feature_channels {best_params['feature_channels']}",
        f"--bottleneck_dim {best_params['bottleneck_dim']}",
        f"--seed {args.seed}",
        f"--gpu {args.gpu}",
        "--save_dir save",
        "--mylog",
    ]

    logger.info("\nCommand to run best configuration:")
    logger.info(" \\\n  ".join(best_cmd))


if __name__ == "__main__":
    main()
