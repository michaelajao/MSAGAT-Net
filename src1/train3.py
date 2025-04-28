import os
import time
import random
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import shutil
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)
from scipy.stats import pearsonr
from math import sqrt
import pandas as pd

# project modules
from data import DataBasicLoader
from model import MSTAGAT_Net
from utils import (
    visualize_matrices,
    visualize_predictions,
    plot_loss_curves,
    save_metrics,
    peak_error,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Determine project root and output directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, "report", "figures")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="region785")
parser.add_argument("--sim_mat", type=str, default="region-adj")
parser.add_argument("--window", type=int, default=20)
parser.add_argument("--horizon", type=int, default=5)
parser.add_argument("--train", type=float, default=0.5)
parser.add_argument("--val", type=float, default=0.2)
parser.add_argument("--test", type=float, default=0.3)
parser.add_argument("--epochs", type=int, default=1500)
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--patience", type=int, default=100)
parser.add_argument("--save_dir", type=str, default="save")
parser.add_argument("--mylog", action="store_false", default=True)
parser.add_argument("--extra", type=str, default="")
parser.add_argument("--label", type=str, default="")
parser.add_argument("--pcc", type=str, default="")
parser.add_argument("--result", type=int, default=0)
parser.add_argument("--record", type=str, default="")
# New argument: starting date for forecast visualization (assume weekly frequency)
parser.add_argument(
    "--start_date",
    type=str,
    default="2020-01-01",
    help="Start date for forecast visualization",
)
# Add missing arguments based on the command provided
parser.add_argument("--hidden_dim", type=int, default=16)
parser.add_argument("--attention_heads", type=int, default=8)
parser.add_argument("--bottleneck_dim", type=int, default=6)
parser.add_argument("--num_scales", type=int, default=5)
parser.add_argument("--kernel_size", type=int, default=3)
parser.add_argument("--feature_channels", type=int, default=12)
parser.add_argument("--attention_regularization_weight", type=float, default=1e-3)
args = parser.parse_args()
print("--------Parameters--------")
print(args)
print("--------------------------")

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

args.cuda = args.cuda and torch.cuda.is_available()
args.cuda = args.gpu is not None
if args.cuda:
    torch.cuda.set_device(args.gpu)
logger.info("cuda %s", args.cuda)

# Define model name and logging token (removed ablation part)
model_name = "MSTAGAT-Net"
log_token = f"{model_name}.{args.dataset}.w-{args.window}.h-{args.horizon}"

if args.mylog:
    tensorboard_log_dir = os.path.join("tensorboard", log_token)
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    writer = SummaryWriter(tensorboard_log_dir)
    logger.info("tensorboard logging to %s", tensorboard_log_dir)

data_loader = DataBasicLoader(args)

# Instantiate the full MSTAGAT_Net model directly
logger.info("Using full MSTAGAT-Net model")
model = MSTAGAT_Net(args, data_loader)

logger.info(
    "model %s", model.__class__.__name__
)  # Log the actual model class being used
if args.cuda:
    model.cuda()
# Set device for torch operations
device = torch.device(f"cuda:{args.gpu}") if args.cuda else torch.device("cpu")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    weight_decay=args.weight_decay,
)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("#params:", pytorch_total_params)


def evaluate(data_loader, data, tag="val", show=0):
    model.eval()
    total_loss = 0.0
    n_samples = 0.0
    y_true, y_pred = [], []
    batch_size = args.batch
    y_pred_mx = []
    y_true_mx = []
    x_value_mx = []

    for inputs in data_loader.get_batches(data, batch_size, False):
        X, Y = inputs[0], inputs[1]
        index = inputs[2]
        output, attn_reg_loss = model(X, index)
        y_expanded = Y.unsqueeze(1).expand(-1, args.horizon, -1)
        loss_train = nn.MSELoss()(output, y_expanded) + attn_reg_loss
        total_loss += loss_train.item()
        n_samples += output.size(0) * data_loader.m

        x_value_mx.append(X.data.cpu())
        y_true_mx.append(Y.data.cpu())
        y_pred_mx.append(output.data.cpu())

    x_value_mx = torch.cat(x_value_mx)
    y_pred_mx = torch.cat(y_pred_mx)
    y_true_mx = torch.cat(y_true_mx)
    y_pred_mx = y_pred_mx[:, -1, :]

    x_value_states = (
        x_value_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    )
    y_true_states = (
        y_true_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    )
    y_pred_states = (
        y_pred_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    )

    rmse_states = np.mean(
        np.sqrt(
            mean_squared_error(y_true_states, y_pred_states, multioutput="raw_values")
        )
    )
    raw_mae = mean_absolute_error(
        y_true_states, y_pred_states, multioutput="raw_values"
    )
    std_mae = np.std(raw_mae)
    if not args.pcc:
        pcc_tmp = []
        for k in range(data_loader.m):
            pcc_tmp.append(pearsonr(y_true_states[:, k], y_pred_states[:, k])[0])
        pcc_states = np.mean(np.array(pcc_tmp))
    else:
        pcc_states = 1
    r2_states = np.mean(
        r2_score(y_true_states, y_pred_states, multioutput="raw_values")
    )
    var_states = np.mean(
        explained_variance_score(y_true_states, y_pred_states, multioutput="raw_values")
    )

    y_true_flat = np.reshape(y_true_states, (-1))
    y_pred_flat = np.reshape(y_pred_states, (-1))
    rmse = sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    if show == 1:
        print("x value", x_value_states)
        print("ground true", y_true_flat.shape)
        print("predict value", y_pred_flat.shape)

    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mape = np.mean(np.abs((y_pred_flat - y_true_flat) / (y_true_flat + 1e-5))) / 1e7
    if not args.pcc:
        pcc = pearsonr(y_true_flat, y_pred_flat)[0]
    else:
        pcc = 1
        pcc_states = 1
    r2 = r2_score(y_true_flat, y_pred_flat, multioutput="uniform_average")
    var = explained_variance_score(
        y_true_flat, y_pred_flat, multioutput="uniform_average"
    )
    peak_mae = peak_error(
        y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold
    )

    # Return predictions along with metrics
    return (
        float(total_loss / n_samples),
        mae,
        std_mae,
        rmse,
        rmse_states,
        pcc,
        pcc_states,
        mape,
        r2,
        r2_states,
        var,
        var_states,
        peak_mae,
        y_true_states,
        y_pred_states,
    )


def train_epoch(data_loader, data):
    model.train()
    total_loss = 0.0
    n_samples = 0.0
    batch_size = args.batch

    for inputs in data_loader.get_batches(data, batch_size, True):
        X, Y = inputs[0], inputs[1]
        index = inputs[2]
        optimizer.zero_grad()
        output, attn_reg_loss = model(X, index)
        y_expanded = Y.unsqueeze(1).expand(-1, args.horizon, -1)
        loss = nn.MSELoss()(output, y_expanded) + attn_reg_loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        n_samples += output.size(0) * data_loader.m
    return float(total_loss / n_samples)


# Training loop
print("Begin training")
os.makedirs(args.save_dir, exist_ok=True)
train_losses, val_losses = [], []
bad_counter, best_epoch, best_val = 0, 0, float("inf")
for epoch in range(1, args.epochs + 1):
    epoch_start_time = time.time()
    train_loss = train_epoch(data_loader, data_loader.train)
    # Capture all metrics from the evaluate call during the training loop for logging
    (
        val_loss,
        mae,
        std_mae,
        rmse,
        rmse_states,
        pcc,
        pcc_states,
        mape,
        r2,
        r2_states,
        var,
        var_states,
        peak_mae,
        _,
        _,
    ) = evaluate(data_loader, data_loader.val)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(
        "Epoch {:3d}|time:{:5.2f}s|train_loss {:5.8f}|val_loss {:5.8f}".format(
            epoch, (time.time() - epoch_start_time), train_loss, val_loss
        )
    )

    if args.mylog:
        writer.add_scalars("data/loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("data/mae", {"val": mae}, epoch)
        # Ensure rmse_states is captured and logged
        writer.add_scalars("data/rmse_states", {"val": rmse_states}, epoch)
        writer.add_scalars("data/pcc", {"val": pcc}, epoch)
        writer.add_scalars("data/pcc_states", {"val": pcc_states}, epoch)
        writer.add_scalars("data/R2", {"val": r2}, epoch)
        writer.add_scalars("data/R2_states", {"val": r2_states}, epoch)
        writer.add_scalars("data/var", {"val": var}, epoch)
        writer.add_scalars("data/var_states", {"val": var_states}, epoch)
        writer.add_scalars("data/peak_mae", {"val": peak_mae}, epoch)

    if val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch
        bad_counter = 0
        model_path = os.path.join(args.save_dir, "{}.pt".format(log_token))
        torch.save(model.state_dict(), model_path)
        # Backup the best model checkpoint
        best_model_path = os.path.join(args.save_dir, "best_model.pt")
        shutil.copy(model_path, best_model_path)
        print("Best validation epoch:", epoch, time.ctime())
        # Capture all metrics from the evaluate call when a new best model is found (already correct)
        (
            test_loss,
            test_mae,
            test_std_mae,
            test_rmse,
            test_rmse_states,
            test_pcc,
            test_pcc_states,
            test_mape,
            test_r2,
            test_r2_states,
            test_var,
            test_var_states,
            test_peak_mae,
            _,
            _,
        ) = evaluate(data_loader, data_loader.test)
        print(
            "TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} MAPE {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}".format(
                test_mae,
                test_std_mae,
                test_rmse,
                test_rmse_states,
                test_pcc,
                test_pcc_states,
                test_mape,
                test_r2,
                test_r2_states,
                test_var,
                test_var_states,
                test_peak_mae,
            )
        )
    else:
        bad_counter += 1

    if bad_counter == args.patience:
        break

# Plot and save the enhanced loss curves
loss_fig_path = os.path.join(FIGURES_DIR, f"loss_curve_{log_token}.png")
plot_loss_curves(train_losses, val_losses, loss_fig_path, args)
logger.info("Loss curve saved to %s", loss_fig_path)

# Visualize matrices: Geolocation, Input Correlation, and Learned Attention
matrices_fig_path = os.path.join(FIGURES_DIR, f"matrices_{log_token}.png")
visualize_matrices(data_loader, model, matrices_fig_path, device)
logger.info("Matrices comparison figure saved to %s", matrices_fig_path)

# Load the best model for final evaluation and print final metrics
model.load_state_dict(
    torch.load(os.path.join(args.save_dir, f"{log_token}.pt"), map_location="cpu")
)
# Capture metrics AND predictions from the final evaluation
(
    test_loss,
    mae,
    std_mae,
    rmse,
    rmse_states,
    pcc,
    pcc_states,
    mape,
    r2,
    r2_states,
    var,
    var_states,
    peak_mae,
    y_true_final,
    y_pred_final,
) = evaluate(data_loader, data_loader.test, tag="test")
print(
    "Final TEST MAE {:5.4f} std {:5.4f} RMSE {:5.4f} RMSEs {:5.4f} PCC {:5.4f} PCCs {:5.4f} MAPE {:5.4f} R2 {:5.4f} R2s {:5.4f} Var {:5.4f} Vars {:5.4f} Peak {:5.4f}".format(
        mae,
        std_mae,
        rmse,
        rmse_states,
        pcc,
        pcc_states,
        mape,
        r2,
        r2_states,
        var,
        var_states,
        peak_mae,
    )
)

# Save final metrics using utils.save_metrics (updated path)
results_csv = os.path.join(RESULTS_DIR, f"final_metrics_{log_token}.csv")
save_metrics(
    {
        "mae": mae,
        "std_MAE": std_mae,
        "rmse": rmse,
        "rmse_states": rmse_states,
        "pcc": pcc,
        "pcc_states": pcc_states,
        "MAPE": mape,
        "R2": r2,
        "R2_states": r2_states,
        "Var": var,
        "Vars": var_states,
        "Peak": peak_mae,
    },
    results_csv,
    args.dataset,
    args.window,
    args.horizon,
    logger,
    model_name,
)
print(f"Saved final metrics to {results_csv}")

# Visualize final predictions
predictions_fig_path = os.path.join(FIGURES_DIR, f"predictions_{log_token}.png")
# Corrected call: Pass save_path correctly, omit unused args/data_loader, use default regions=5
visualize_predictions(y_true_final, y_pred_final, predictions_fig_path, logger=logger)
logger.info("Predictions visualization saved to %s", predictions_fig_path)


if args.record != "":
    with open("result/result.txt", "a", encoding="utf-8") as f:
        # Updated result logging format (removed ablation)
        f.write(
            f"{args.dataset} {args.window} {args.horizon} {args.train} {args.val} {args.test} "
            f"{mae:.4f} {std_mae:.4f} {rmse:.4f} {rmse_states:.4f} "
            f"{pcc:.4f} {pcc_states:.4f} {mape:.4f} "
            f"{r2:.4f} {r2_states:.4f} {var:.4f} {var_states:.4f} "
            f"{peak_mae:.4f}\n"
        )
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation loss: {best_val:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Final metrics saved to {results_csv}\n")
        f.write(f"Model saved to {os.path.join(args.save_dir, 'best_model.pt')}\n")
        f.write(f"Tensorboard logs saved to {tensorboard_log_dir}\n")
        f.write(f"Figures saved to {FIGURES_DIR}\n")
        f.write(f"Results saved to {RESULTS_DIR}\n")
        f.write(f"Training completed at {time.ctime()}\n")
        f.write("========================================\n")
        f.write(f"Training completed at {time.ctime()}\n")
        f.write("========================================\n")
