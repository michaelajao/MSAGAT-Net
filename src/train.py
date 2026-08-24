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
from .models import MSAGATNet_Ablation, pinball_loss

# CDC FluSight / COVID-19 Forecast Hub convention: 23 quantiles
# = median + 11 central intervals (Bracher et al. 2021; Cramer et al. 2022).
DEFAULT_QUANTILES = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                     0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9,
                     0.95, 0.975, 0.99]

# The log-growth decoder y = (anchor+1)*exp(g) - 1 is multiplicative, so a
# plausible growth rate applied to a large anchor can decode to a level far
# outside anything ever observed. Inverted levels are therefore capped at
# GROWTH_LEVEL_CAP x the per-node training maximum -- a train-only decoding
# constraint. The multiplier was selected once on the validation split
# (a single global value; per-dataset and per-cell selection both overfit
# validation and were worse on test).
GROWTH_LEVEL_CAP = 3.0

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
    r2: float
    r2_states: float
    var: float
    var_states: float
    peak_mae: float
    y_true: np.ndarray = field(repr=False)
    y_pred: np.ndarray = field(repr=False)
    y_pred_q: np.ndarray = field(default=None, repr=False)

    def to_dict(self) -> Dict:
        return {
            'mae': self.mae, 'std_MAE': self.mae_std,
            'rmse': self.rmse, 'rmse_states': self.rmse_states,
            'pcc': self.pcc, 'pcc_states': self.pcc_states,
            'R2': self.r2, 'R2_states': self.r2_states,
            'Var': self.var, 'Vars': self.var_states, 'Peak': self.peak_mae,
        }


# ── Core training / evaluation ──────────────────────────────────────────────

def train_epoch(model, data_loader, optimizer, batch_size, horizon, device,
                max_grad_norm=1.0, y_multi=None, growth=None, q_levels=None):
    model.train()
    total_loss, n_samples = 0.0, 0.0

    for inputs in data_loader.get_batches(data_loader.train, batch_size, shuffle=True):
        X, Y, index = inputs[0], inputs[1], inputs[2]
        optimizer.zero_grad()
        output, attn_reg_loss = model(X, index)
        if growth is not None:
            # Growth-space supervision: the model predicts the log-growth
            # ratio relative to the last observation instead of the level.
            target_last = growth[0][index].to(output.device)
            target = target_last.unsqueeze(1).expand(-1, horizon, -1)
        elif y_multi is not None:
            # Progressive supervision: slice j is trained toward the true
            # lead-(j+1) observation instead of the lead-h target repeated.
            target = y_multi[index].to(output.device)
            target_last = target[:, -1, :]
        else:
            target_last = Y
            target = Y.unsqueeze(1).expand(-1, horizon, -1)
        loss = nn.MSELoss()(output, target) + attn_reg_loss
        if q_levels is not None and model.last_quantiles is not None:
            loss = loss + pinball_loss(model.last_quantiles, target_last, q_levels)
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        n_samples += output.size(0) * data_loader.m

    return total_loss / n_samples


def evaluate(model, data_loader, batch_size, horizon, device,
             dataset='val', compute_pcc=True, y_multi=None, growth=None,
             q_levels=None, g_bounds=None, level_cap=None):
    model.eval()
    total_loss, n_samples = 0.0, 0.0
    y_true_list, y_pred_list, x_value_list, q_list = [], [], [], []

    data = data_loader.val if dataset == 'val' else data_loader.test

    with torch.no_grad():
        for inputs in data_loader.get_batches(data, batch_size, shuffle=False):
            X, Y, index = inputs[0], inputs[1], inputs[2]
            output, attn_reg_loss = model(X, index)
            if growth is not None:
                target_last = growth[0][index].to(output.device)
                target = target_last.unsqueeze(1).expand(-1, horizon, -1)
            elif y_multi is not None:
                target = y_multi[index].to(output.device)
                target_last = target[:, -1, :]
            else:
                target_last = Y
                target = Y.unsqueeze(1).expand(-1, horizon, -1)
            loss = nn.MSELoss()(output, target) + attn_reg_loss
            if q_levels is not None and model.last_quantiles is not None:
                loss = loss + pinball_loss(model.last_quantiles, target_last,
                                           q_levels)
            total_loss += loss.item()
            n_samples += output.size(0) * data_loader.m
            x_value_list.append(X.cpu())
            y_true_list.append(Y.cpu())
            y_pred_list.append(output.cpu())
            if q_levels is not None and model.last_quantiles is not None:
                q_list.append(model.last_quantiles.detach().cpu())

    x_value_mx = torch.cat(x_value_list)
    y_pred_mx = torch.cat(y_pred_list)[:, -1, :]
    y_true_mx = torch.cat(y_true_list)

    scale = data_loader.max - data_loader.min
    y_true_states = y_true_mx.numpy() * scale + data_loader.min
    if growth is not None:
        # Model output is a log-growth ratio; invert with per-sample anchors
        # (batches are un-shuffled, so split order is preserved). Predicted
        # growth is clipped to the training-observed range so exp() cannot
        # blow up on out-of-distribution logits.
        anchors = growth[1].numpy()
        g_pred = y_pred_mx.numpy()
        if g_bounds is not None:
            lo = g_bounds[0].numpy()[None, :]
            hi = g_bounds[1].numpy()[None, :]
            g_pred = np.clip(g_pred, lo, hi)
        y_pred_states = (anchors + 1.0) * np.exp(g_pred) - 1.0
        if level_cap is not None:
            y_pred_states = np.clip(y_pred_states, 0.0, level_cap[None, :])
    else:
        y_pred_states = y_pred_mx.numpy() * scale + data_loader.min

    y_pred_q_states = None
    if q_list:
        q_mx = torch.cat(q_list).numpy()          # [n, nodes, Q]
        if growth is not None:
            if g_bounds is not None:
                q_mx = np.clip(q_mx, lo[..., None], hi[..., None])
            y_pred_q_states = (anchors[..., None] + 1.0) * np.exp(q_mx) - 1.0
            if level_cap is not None:
                y_pred_q_states = np.clip(y_pred_q_states, 0.0,
                                          level_cap[None, :, None])
        else:
            y_pred_q_states = (q_mx * scale[None, :, None]
                               + data_loader.min[None, :, None])

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
        r2=r2, r2_states=r2_states, var=var, var_states=var_states,
        peak_mae=peak_mae_val, y_true=y_true_states, y_pred=y_pred_states,
        y_pred_q=y_pred_q_states,
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
        # Weight decay drove the learnable graph bias (u, v) to ~1e-36 on
        # trained checkpoints: once the softmax is flat the aggregation is a
        # uniform mean, so the gradient on a logit bias vanishes and decay wins.
        # Under attn_fix these attention-shaping parameters get their own
        # decay-free group.
        no_decay_keys = ('graph_attention.u', 'graph_attention.v',
                         'graph_attention.adj_scale',
                         'graph_attention.log_attn_temp')
        trainable = [(n, p) for n, p in model.named_parameters()
                     if p.requires_grad]
        exp = set(t for t in getattr(config, 'attn_exp', '').split(',') if t)
        lr_mult = 1.0
        for t in exp:
            if t.startswith('lrx'):
                lr_mult = float(t[3:])
        if getattr(config, 'attn_fix', False) or 'nodecay' in exp:
            shaped = [p for n, p in trainable if n in no_decay_keys]
            rest = [p for n, p in trainable if n not in no_decay_keys]
            groups = [{'params': rest,
                       'weight_decay': self.config.weight_decay},
                      {'params': shaped, 'weight_decay': 0.0,
                       'lr': self.config.lr * lr_mult}]
        elif lr_mult != 1.0:
            shaped = [p for n, p in trainable if n in no_decay_keys]
            rest = [p for n, p in trainable if n not in no_decay_keys]
            groups = [{'params': rest,
                       'weight_decay': self.config.weight_decay},
                      {'params': shaped,
                       'weight_decay': self.config.weight_decay,
                       'lr': self.config.lr * lr_mult}]
        else:
            groups = [{'params': [p for _, p in trainable],
                       'weight_decay': self.config.weight_decay}]
        self.optimizer = torch.optim.Adam(groups, lr=self.config.lr)

        # Progressive-refinement supervision: precompute per-lead targets once.
        self.y_multi_train = None
        self.y_multi_val = None
        if getattr(config, 'pprm_supervision', 'repeat') == 'multistep':
            self.y_multi_train = data_loader.multistep_targets(
                data_loader.train_set, self.horizon)
            self.y_multi_val = data_loader.multistep_targets(
                data_loader.valid_set, self.horizon)

        # Growth-space forecasting: precompute log-growth targets + anchors.
        self.growth_train = self.growth_val = self.growth_test = None
        self.g_bounds = None
        self.level_cap = None
        if getattr(config, 'target_space', 'level') == 'loggrowth':
            self.growth_train = data_loader.growth_targets(
                data_loader.train_set, self.horizon)
            self.growth_val = data_loader.growth_targets(
                data_loader.valid_set, self.horizon)
            self.growth_test = data_loader.growth_targets(
                data_loader.test_set, self.horizon)
            # Inversion guard: epidemic growth over a fixed lead is bounded;
            # clip predicted log-growth to the training-observed per-node range
            # (+/- 0.5 nats) so a single wild logit cannot detonate through
            # exp() at inversion time.
            g = self.growth_train[0]
            self.g_bounds = (g.min(dim=0).values - 0.5,
                             g.max(dim=0).values + 0.5)
            # data_loader.max is the per-node maximum over the raw training
            # window, so the cap uses no validation or test information.
            self.level_cap = GROWTH_LEVEL_CAP * np.asarray(data_loader.max,
                                                           dtype=np.float64)

        # Probabilistic output: quantile levels used by the pinball loss.
        self.q_levels = None
        if getattr(config, 'quantiles', None):
            self.q_levels = torch.tensor(sorted(config.quantiles),
                                         dtype=torch.float32,
                                         device=self.device)

        # Re-evaluate an existing checkpoint without retraining (used when an
        # inversion/metric change invalidates outputs but not weights).
        self.eval_only = getattr(config, 'eval_only', False)

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
        ckpt = os.path.join(self.config.save_dir, f'{self.log_token}.pt')
        if self.eval_only and os.path.exists(ckpt):
            print(f'Eval-only: loading {ckpt}')
            self._load_best_checkpoint()
            final = evaluate(self.model, self.data_loader,
                             self.config.batch_size, self.horizon, self.device,
                             dataset='test', growth=self.growth_test,
                             q_levels=self.q_levels, g_bounds=self.g_bounds,
                             level_cap=self.level_cap)
            print(f'Final  MAE {final.mae:.4f}  RMSE {final.rmse:.4f}  '
                  f'PCC {final.pcc:.4f}  R2 {final.r2:.4f}')
            return final

        print(f'Begin training  |  Parameters: '
              f'{sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}')

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()
            train_loss = train_epoch(
                self.model, self.data_loader, self.optimizer,
                self.config.batch_size, self.horizon, self.device,
                max_grad_norm=self.config.max_grad_norm,
                y_multi=self.y_multi_train, growth=self.growth_train,
                q_levels=self.q_levels)
            val_metrics = evaluate(
                self.model, self.data_loader, self.config.batch_size,
                self.horizon, self.device, dataset='val',
                y_multi=self.y_multi_val, growth=self.growth_val,
                q_levels=self.q_levels, g_bounds=self.g_bounds,
                level_cap=self.level_cap)

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
                    self.horizon, self.device, dataset='test',
                    growth=self.growth_test, q_levels=self.q_levels,
                    g_bounds=self.g_bounds, level_cap=self.level_cap)
                print(f'  TEST  MAE {test_metrics.mae:.4f}  RMSE {test_metrics.rmse:.4f}  '
                      f'PCC {test_metrics.pcc:.4f}  R2 {test_metrics.r2:.4f}')
            else:
                self.bad_counter += 1

            if self.bad_counter >= self.config.patience:
                print(f'Early stopping at epoch {epoch}')
                break

        self._load_best_checkpoint()
        final = evaluate(self.model, self.data_loader, self.config.batch_size,
                         self.horizon, self.device, dataset='test',
                         growth=self.growth_test, q_levels=self.q_levels,
                         g_bounds=self.g_bounds, level_cap=self.level_cap)
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
                          save_dir='save_all', verbose=True, force_cpu=False,
                          use_adj_prior=True, sim_mat=None,
                          save_predictions=True, pprm_supervision='repeat',
                          spatial_gate=False, target_space='level',
                          quantiles=None, eval_only=False, attn_fix=False,
                          attn_exp='', renewal=False, renewal_lag=0,
                          gi_fix=None):
    cfg = DATASET_CONFIGS[dataset]
    args = Namespace(
        dataset=dataset, sim_mat=sim_mat or cfg['sim_mat'],
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
        use_adj_prior=use_adj_prior, adj_weight=0.1, use_graph_bias=True,
        adaptive=False, seed=seed, gpu=0,
        cuda=torch.cuda.is_available() and not force_cpu,
        save_dir=save_dir, mylog=True, highway_window=4,
        extra='', label='', pcc='',
        pprm_supervision=pprm_supervision, spatial_gate=spatial_gate,
        target_space=target_space, quantiles=quantiles, eval_only=eval_only,
        attn_fix=attn_fix, attn_exp=attn_exp,
        renewal=renewal, renewal_lag=renewal_lag, gi_fix=gi_fix,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data_loader = DataBasicLoader(args)
    model = MSAGATNet_Ablation(args, data_loader)
    model_name = 'MSAGAT-Net'

    if args.cuda:
        model.cuda()

    adj_tag = 'with_adj' if use_adj_prior else 'no_adj'
    sim_tag = f".{sim_mat}" if sim_mat else ""
    variant_tag = ""
    if pprm_supervision != 'repeat':
        variant_tag += f".pprm-{pprm_supervision}"
    if spatial_gate:
        variant_tag += ".sgate"
    if target_space != 'level':
        variant_tag += f".{target_space}"
    if quantiles:
        variant_tag += ".quant"
    if attn_fix:
        variant_tag += ".attnfix"
    if attn_exp:
        variant_tag += ".exp-" + attn_exp.replace(',', '-')
    if renewal:
        variant_tag += f".renewal{renewal_lag or ''}"
    if gi_fix:
        variant_tag += f".gifix{gi_fix[0]:g}-{gi_fix[1]:g}"
    log_token = (f"{model_name}.{dataset}.w-{args.window}.h-{horizon}."
                 f"{ablation}.seed-{seed}.{adj_tag}{sim_tag}{variant_tag}")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {dataset} | h={horizon} | seed={seed} | ablation={ablation}"
              f" | adj={use_adj_prior} | sim_mat={sim_mat or 'default'}"
              f" | pprm={pprm_supervision} | gate={spatial_gate}")
        print(f"{'='*60}")

    trainer = Trainer(model, data_loader, args, log_token)
    final_metrics = trainer.train()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset_results_dir = os.path.join(RESULTS_DIR, dataset)
    os.makedirs(dataset_results_dir, exist_ok=True)

    if save_predictions:
        pred_dir = os.path.join(BASE_DIR, 'report', 'predictions', dataset)
        os.makedirs(pred_dir, exist_ok=True)
        payload = dict(
            y_true=final_metrics.y_true, y_pred=final_metrics.y_pred,
            model=model_name, dataset=dataset, horizon=horizon,
            window=args.window, seed=seed, ablation=ablation,
            use_adj=use_adj_prior, sim_mat=sim_mat or 'default',
            pprm_supervision=pprm_supervision, spatial_gate=spatial_gate,
            target_space=target_space, protocol='lead_h')
        if final_metrics.y_pred_q is not None:
            payload['y_pred_q'] = final_metrics.y_pred_q
            payload['quantile_levels'] = np.array(sorted(quantiles))
        # Validation-split predictions are the calibration set for conformal
        # intervals, so persist them alongside the test split.
        val_metrics = evaluate(trainer.model, data_loader, args.batch,
                               horizon, trainer.device, dataset='val',
                               growth=trainer.growth_val,
                               q_levels=trainer.q_levels,
                               g_bounds=trainer.g_bounds,
                               level_cap=trainer.level_cap)
        payload['y_true_val'] = val_metrics.y_true
        payload['y_pred_val'] = val_metrics.y_pred
        if val_metrics.y_pred_q is not None:
            payload['y_pred_q_val'] = val_metrics.y_pred_q
        np.savez_compressed(os.path.join(pred_dir, f"{log_token}.npz"),
                            **payload)

    model_tag = model_name + variant_tag
    results_csv = os.path.join(dataset_results_dir, f"final_metrics_{log_token}.csv")
    save_metrics(final_metrics.to_dict(), results_csv, dataset, args.window,
                 horizon, logger, model_tag, ablation, seed, use_adj_prior,
                 sim_mat=sim_mat or 'default')

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
    total = len(datasets) * len(ABLATIONS) * len(ablation_horizons) * len(seeds)
    done, failed = 0, 0

    for dataset in datasets:
        for ablation in ABLATIONS:
            for horizon in ablation_horizons:
                for seed in seeds:
                    done += 1
                    print(f"\n[{done}/{total}] {dataset} h={horizon} "
                          f"ablation={ablation} seed={seed}")
                    if dry_run:
                        continue
                    try:
                        run_single_experiment(dataset, horizon, seed, ablation=ablation,
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
    parser.add_argument('--use_adj_prior', action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument('--sim_mat', type=str, default=None,
                        help="override the dataset's adjacency file "
                             "(e.g. ltla-adj-200 for threshold sensitivity)")
    parser.add_argument('--pprm_supervision', choices=['repeat', 'multistep'],
                        default='repeat',
                        help='repeat: all refinement slices supervised with the '
                             'lead-h target; multistep: slice j supervised with '
                             'the true lead-(j+1) observation')
    parser.add_argument('--spatial_gate', action='store_true',
                        help='learnable gate blending spatial pathway output '
                             'with purely temporal features')
    parser.add_argument('--target_space', choices=['level', 'loggrowth'],
                        default='level',
                        help='loggrowth: predict log((y_t+1)/(y_anchor+1)) '
                             'instead of the level')
    parser.add_argument('--quantiles', nargs='*', type=float, default=None,
                        help='enable probabilistic quantile outputs; pass '
                             'levels or leave empty for the default set')
    parser.add_argument('--renewal', action='store_true',
                        help='renewal-equation decoder: backbone predicts '
                             'log R, a learned generation-interval kernel '
                             'supplies the convolution')
    parser.add_argument('--gi_fix', nargs=2, type=float, default=None,
                        metavar=('MEAN', 'SD'),
                        help='freeze the generation-interval kernel to a '
                             'discretised gamma with this mean and sd '
                             '(disables learning it)')
    parser.add_argument('--renewal_lag', type=int, default=0,
                        help='generation-interval kernel length (0 = auto)')
    parser.add_argument('--attn_exp', default='',
                        help='comma-separated attention-revival experiment '
                             'tokens, e.g. "nodecay,regpre" (see program.md)')
    parser.add_argument('--attn_fix', action='store_true',
                        help='learnable attention temperature + no weight decay '
                             'on the attention-shaping parameters')
    parser.add_argument('--eval_only', action='store_true',
                        help='re-evaluate an existing checkpoint without '
                             'retraining')
    args = parser.parse_args()

    quantiles = args.quantiles
    if quantiles is not None and len(quantiles) == 0:
        quantiles = DEFAULT_QUANTILES

    if args.single:
        run_single_experiment(args.dataset, args.horizon, args.seed,
                              ablation=args.ablation, save_dir=args.save_dir,
                              force_cpu=args.cpu,
                              use_adj_prior=args.use_adj_prior,
                              sim_mat=args.sim_mat,
                              pprm_supervision=args.pprm_supervision,
                              spatial_gate=args.spatial_gate,
                              target_space=args.target_space,
                              quantiles=quantiles,
                              eval_only=args.eval_only,
                              attn_fix=args.attn_fix,
                              attn_exp=args.attn_exp,
                              renewal=args.renewal,
                              renewal_lag=args.renewal_lag,
                              gi_fix=args.gi_fix)
    else:
        datasets = args.datasets or list(DATASET_CONFIGS.keys())
        if args.experiment in ('main', 'all'):
            run_main_experiments(datasets, args.seeds, args.dry_run, args.cpu, args.save_dir)
        if args.experiment in ('ablation', 'all'):
            run_ablation_experiments(datasets, args.seeds, args.dry_run, args.cpu, args.save_dir)

    sys.exit(0)


if __name__ == '__main__':
    main()
