"""
Training module for MSAGAT-Net.

Handles single experiments, batch training, ablation studies,
and includes the Trainer class with early stopping and checkpointing.

Usage:
    python -m src.train --single --dataset japan --horizon 5 --seed 42
    python -m src.train --experiment main --datasets japan australia-covid
    python -m src.train --experiment ablation --datasets japan
    python -m src.train --dry-run
"""

import os
import sys
import time
import random
import logging
import argparse
import atexit
import signal
from typing import Dict, List
from argparse import Namespace
from dataclasses import dataclass, field
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, explained_variance_score
)
from scipy.stats import pearsonr

from .utils import peak_error, plot_loss_curves, save_metrics
from .data import DataBasicLoader
from .models import MSAGATNet_Ablation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger(__name__)


# ── GPU cleanup ──────────────────────────────────────────────────────────────

def _cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

atexit.register(_cleanup_gpu)
signal.signal(signal.SIGINT, lambda s, f: (_cleanup_gpu(), sys.exit(1)))
signal.signal(signal.SIGTERM, lambda s, f: (_cleanup_gpu(), sys.exit(1)))


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    epochs: int = 1500
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 5e-4
    patience: int = 100
    max_grad_norm: float = 1.0
    save_dir: str = 'save'
    device: str = 'cpu'
    use_tensorboard: bool = True


@dataclass
class MetricsResult:
    loss: float
    mae: float
    mae_std: float
    rmse: float
    rmse_states: float
    pcc: float
    pcc_states: float
    mape: float
    r2: float
    r2_states: float
    var: float
    var_states: float
    peak_mae: float
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)

    def to_dict(self) -> Dict:
        return {
            'mae': self.mae, 'std_MAE': self.mae_std,
            'rmse': self.rmse, 'rmse_states': self.rmse_states,
            'pcc': self.pcc, 'pcc_states': self.pcc_states,
            'MAPE': self.mape, 'R2': self.r2, 'R2_states': self.r2_states,
            'Var': self.var, 'Vars': self.var_states, 'Peak': self.peak_mae,
        }


# ── Core training / evaluation ──────────────────────────────────────────────

def train_epoch(model, data_loader, optimizer, batch_size, horizon, device,
                max_grad_norm=1.0):
    model.train()
    total_loss, n_samples = 0.0, 0.0

    for inputs in data_loader.get_batches(data_loader.train, batch_size, shuffle=True):
        X, Y, index = inputs[0], inputs[1], inputs[2]
        optimizer.zero_grad()
        output, attn_reg_loss = model(X, index)
        y_expanded = Y.unsqueeze(1).expand(-1, horizon, -1)
        loss = nn.MSELoss()(output, y_expanded) + attn_reg_loss
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        n_samples += output.size(0) * data_loader.m

    return total_loss / n_samples


def evaluate(model, data_loader, batch_size, horizon, device,
             dataset='val', compute_pcc=True):
    model.eval()
    total_loss, n_samples = 0.0, 0.0
    y_true_list, y_pred_list, x_value_list = [], [], []

    data = data_loader.val if dataset == 'val' else data_loader.test

    with torch.no_grad():
        for inputs in data_loader.get_batches(data, batch_size, shuffle=False):
            X, Y, index = inputs[0], inputs[1], inputs[2]
            output, attn_reg_loss = model(X, index)
            y_expanded = Y.unsqueeze(1).expand(-1, horizon, -1)
            loss = nn.MSELoss()(output, y_expanded) + attn_reg_loss
            total_loss += loss.item()
            n_samples += output.size(0) * data_loader.m
            x_value_list.append(X.cpu())
            y_true_list.append(Y.cpu())
            y_pred_list.append(output.cpu())

    x_value_mx = torch.cat(x_value_list)
    y_pred_mx = torch.cat(y_pred_list)[:, -1, :]
    y_true_mx = torch.cat(y_true_list)

    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min

    rmse_states = np.mean(np.sqrt(
        mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values')))
    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae)

    pcc_states = 1.0
    if compute_pcc:
        pcc_tmp = []
        for k in range(data_loader.m):
            if np.std(y_true_states[:, k]) < 1e-10 or np.std(y_pred_states[:, k]) < 1e-10:
                pcc_tmp.append(0.0)
            else:
                corr, _ = pearsonr(y_true_states[:, k], y_pred_states[:, k])
                pcc_tmp.append(corr)
        pcc_states = np.mean(pcc_tmp)

    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))

    y_true_flat = y_true_states.flatten()
    y_pred_flat = y_pred_states.flatten()
    rmse = sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mape = np.mean(np.abs((y_pred_flat - y_true_flat) / (y_true_flat + 1e-5))) / 1e7

    pcc = 1.0
    if compute_pcc:
        if np.std(y_true_flat) < 1e-10 or np.std(y_pred_flat) < 1e-10:
            pcc = 0.0
        else:
            pcc, _ = pearsonr(y_true_flat, y_pred_flat)

    r2 = r2_score(y_true_flat, y_pred_flat)
    var = explained_variance_score(y_true_flat, y_pred_flat)
    peak_mae_val = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)

    return MetricsResult(
        loss=total_loss / n_samples, mae=mae, mae_std=std_mae,
        rmse=rmse, rmse_states=rmse_states, pcc=pcc, pcc_states=pcc_states,
        mape=mape, r2=r2, r2_states=r2_states, var=var, var_states=var_states,
        peak_mae=peak_mae_val, y_true=y_true_states, y_pred=y_pred_states,
    )


# ── Trainer class ────────────────────────────────────────────────────────────

class Trainer:
    """Training orchestrator with early stopping, checkpointing, and TensorBoard."""

    def __init__(self, model, data_loader, config, log_token='model'):
        self.model = model
        self.data_loader = data_loader
        self.log_token = log_token

        if isinstance(config, TrainingConfig):
            self.config = config
        else:
            self.config = TrainingConfig(
                epochs=getattr(config, 'epochs', 1500),
                batch_size=getattr(config, 'batch', 32),
                lr=getattr(config, 'lr', 1e-3),
                weight_decay=getattr(config, 'weight_decay', 5e-4),
                patience=getattr(config, 'patience', 100),
                max_grad_norm=getattr(config, 'max_grad_norm', 1.0),
                save_dir=getattr(config, 'save_dir', 'save'),
                use_tensorboard=getattr(config, 'mylog', True),
            )

        if hasattr(config, 'cuda') and config.cuda:
            self.device = torch.device(f'cuda:{config.gpu}')
        else:
            self.device = torch.device('cpu')

        self.horizon = config.horizon
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.lr, weight_decay=self.config.weight_decay,
        )

        self.writer = None
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join('tensorboard', log_token)
                os.makedirs(tb_dir, exist_ok=True)
                self.writer = SummaryWriter(tb_dir)
            except ImportError:
                pass

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val = float('inf')
        self.best_epoch = 0
        self.bad_counter = 0

    def train(self) -> MetricsResult:
        os.makedirs(self.config.save_dir, exist_ok=True)
        print(f'Begin training  |  Parameters: '
              f'{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = train_epoch(
                self.model, self.data_loader, self.optimizer,
                self.config.batch_size, self.horizon, self.device,
                max_grad_norm=self.config.max_grad_norm)
            val_metrics = evaluate(
                self.model, self.data_loader, self.config.batch_size,
                self.horizon, self.device, dataset='val')

            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics.loss)

            print(f'Epoch {epoch:3d} | {time.time()-t0:5.2f}s | '
                  f'train: {train_loss:.8f} | val: {val_metrics.loss:.8f}')

            if self.writer:
                self.writer.add_scalars('loss', {'train': train_loss, 'val': val_metrics.loss}, epoch)

            if val_metrics.loss < self.best_val:
                self.best_val = val_metrics.loss
                self.best_epoch = epoch
                self.bad_counter = 0
                self._save_checkpoint()
                test_metrics = evaluate(
                    self.model, self.data_loader, self.config.batch_size,
                    self.horizon, self.device, dataset='test')
                print(f'  TEST  MAE {test_metrics.mae:.4f}  RMSE {test_metrics.rmse:.4f}  '
                      f'PCC {test_metrics.pcc:.4f}  R2 {test_metrics.r2:.4f}')
            else:
                self.bad_counter += 1

            if self.bad_counter >= self.config.patience:
                print(f'Early stopping at epoch {epoch}')
                break

        self._load_best_checkpoint()
        final = evaluate(self.model, self.data_loader, self.config.batch_size,
                         self.horizon, self.device, dataset='test')
        print(f'\nFinal  MAE {final.mae:.4f}  RMSE {final.rmse:.4f}  '
              f'PCC {final.pcc:.4f}  R2 {final.r2:.4f}')

        if self.writer:
            self.writer.close()
        return final

    def _save_checkpoint(self):
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, f'{self.log_token}.pt')
        torch.save(self.model.state_dict(), path)
        torch.save(self.model.state_dict(),
                    os.path.join(self.config.save_dir, 'best_model.pt'))

    def _load_best_checkpoint(self):
        path = os.path.join(self.config.save_dir, f'{self.log_token}.pt')
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location='cpu'))


# ── Experiment configuration ─────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(BASE_DIR, 'report', 'figures')
RESULTS_DIR = os.path.join(BASE_DIR, 'report', 'results')

DATASET_CONFIGS = {
    'japan':            {'sim_mat': 'japan-adj',     'num_nodes': 47,  'horizons': [3, 5, 10, 15]},
    'region785':        {'sim_mat': 'region-adj',    'num_nodes': 10,  'horizons': [3, 5, 10, 15]},
    'state360':         {'sim_mat': 'state-adj-49',  'num_nodes': 49,  'horizons': [3, 5, 10, 15]},
    'australia-covid':  {'sim_mat': 'australia-adj',  'num_nodes': 8,   'horizons': [3, 7, 14]},
    'nhs_timeseries':   {'sim_mat': 'nhs-adj',       'num_nodes': 7,   'horizons': [3, 7, 14]},
    'ltla_timeseries':  {'sim_mat': 'ltla-adj',      'num_nodes': 372, 'horizons': [3, 7, 14]},
}

TRAIN_DEFAULTS = dict(
    epochs=1500, patience=100, lr=1e-3, weight_decay=5e-4,
    batch=32, window=20, dropout=0.2, num_scales=4,
    hidden_dim=32, attention_heads=4, bottleneck_dim=8,
)

SEEDS = [42, 30, 45, 123, 1000]
ABLATIONS = ['none', 'no_agam', 'no_mtfm', 'no_pprm']


# ── Single experiment ────────────────────────────────────────────────────────

def run_single_experiment(dataset, horizon, seed, ablation='none',
                          save_dir='save_all', verbose=True, force_cpu=False):
    cfg = DATASET_CONFIGS[dataset]
    args = Namespace(
        dataset=dataset, sim_mat=cfg['sim_mat'],
        window=TRAIN_DEFAULTS['window'], horizon=horizon,
        train=0.6, val=0.2, test=0.2,
        epochs=TRAIN_DEFAULTS['epochs'], batch=TRAIN_DEFAULTS['batch'],
        lr=TRAIN_DEFAULTS['lr'], weight_decay=TRAIN_DEFAULTS['weight_decay'],
        dropout=TRAIN_DEFAULTS['dropout'], patience=TRAIN_DEFAULTS['patience'],
        ablation=ablation, hidden_dim=TRAIN_DEFAULTS['hidden_dim'],
        attention_heads=TRAIN_DEFAULTS['attention_heads'],
        attention_regularization_weight=1e-5,
        num_scales=TRAIN_DEFAULTS['num_scales'], kernel_size=3,
        feature_channels=16, bottleneck_dim=TRAIN_DEFAULTS['bottleneck_dim'],
        use_adj_prior=True, adj_weight=0.1, use_graph_bias=True,
        adaptive=False, seed=seed, gpu=0,
        cuda=torch.cuda.is_available() and not force_cpu,
        save_dir=save_dir, mylog=True, highway_window=4,
        extra='', label='', pcc='',
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True

    data_loader = DataBasicLoader(args)
    model = MSAGATNet_Ablation(args, data_loader)
    model_name = 'MSAGAT-Net'

    if args.cuda:
        model.cuda()

    log_token = f"{model_name}.{dataset}.w-{args.window}.h-{horizon}.{ablation}.seed-{seed}.with_adj"

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {dataset} | h={horizon} | seed={seed} | ablation={ablation}")
        print(f"{'='*60}")

    trainer = Trainer(model, data_loader, args, log_token)
    final_metrics = trainer.train()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset_results_dir = os.path.join(RESULTS_DIR, dataset)
    os.makedirs(dataset_results_dir, exist_ok=True)

    results_csv = os.path.join(dataset_results_dir, f"final_metrics_{log_token}.csv")
    save_metrics(final_metrics.to_dict(), results_csv, dataset, args.window,
                 horizon, logger, model_name, ablation, seed, True)

    if verbose:
        print(f"Results saved to {results_csv}")
    return final_metrics.to_dict()


# ── Batch runners ────────────────────────────────────────────────────────────

def run_main_experiments(datasets, seeds, dry_run=False, force_cpu=False,
                         save_dir='save_all'):
    total = sum(len(DATASET_CONFIGS[d]['horizons']) for d in datasets) * len(seeds)
    done, failed = 0, 0

    for dataset in datasets:
        for horizon in DATASET_CONFIGS[dataset]['horizons']:
            for seed in seeds:
                done += 1
                print(f"\n[{done}/{total}] {dataset} h={horizon} seed={seed}")
                if dry_run:
                    continue
                try:
                    run_single_experiment(dataset, horizon, seed,
                                          save_dir=save_dir, force_cpu=force_cpu)
                except Exception as e:
                    failed += 1
                    print(f"  FAIL: {e}")

    print(f"\nMain experiments: {done-failed}/{total} completed, {failed} failed")


def run_ablation_experiments(datasets, seeds, dry_run=False, force_cpu=False,
                             save_dir='save_all'):
    ablation_horizons = [3, 7, 14]
    total = len(datasets) * len(ABLATIONS) * len(ablation_horizons)
    done, failed = 0, 0

    for dataset in datasets:
        for ablation in ABLATIONS:
            for horizon in ablation_horizons:
                done += 1
                print(f"\n[{done}/{total}] {dataset} h={horizon} ablation={ablation}")
                if dry_run:
                    continue
                try:
                    run_single_experiment(dataset, horizon, seeds[0], ablation=ablation,
                                          save_dir=save_dir, force_cpu=force_cpu)
                except Exception as e:
                    failed += 1
                    print(f"  FAIL: {e}")

    print(f"\nAblation experiments: {done-failed}/{total} completed, {failed} failed")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    parser = argparse.ArgumentParser(description='MSAGAT-Net Training')
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--experiment', choices=['main', 'ablation', 'all'], default='all')
    parser.add_argument('--dataset', type=str, default='japan')
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ablation', type=str, default='none')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--save_dir', type=str, default='save_all')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    if args.single:
        run_single_experiment(args.dataset, args.horizon, args.seed,
                              ablation=args.ablation, save_dir=args.save_dir,
                              force_cpu=args.cpu)
    else:
        datasets = args.datasets or list(DATASET_CONFIGS.keys())
        if args.experiment in ('main', 'all'):
            run_main_experiments(datasets, args.seeds, args.dry_run, args.cpu, args.save_dir)
        if args.experiment in ('ablation', 'all'):
            run_ablation_experiments(datasets, args.seeds, args.dry_run, args.cpu, args.save_dir)

    sys.exit(0)


if __name__ == '__main__':
    main()
