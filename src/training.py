"""
Training Module for MSAGAT-Net

Provides training loops, evaluation functions, and the Trainer class for
orchestrating model training with logging, checkpointing, and early stopping.

Main Components:
    - train_epoch: Single epoch training function
    - evaluate: Model evaluation with comprehensive metrics
    - Trainer: High-level training orchestrator
"""

import os
import time
import shutil
import logging
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    explained_variance_score
)
from scipy.stats import pearsonr
from math import sqrt
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field

from .utils import peak_error

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    epochs: int = 1500
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 5e-4
    patience: int = 100
    max_grad_norm: float = 1.0
    save_dir: str = 'save'
    device: str = 'cpu'
    use_tensorboard: bool = True
    tensorboard_dir: str = 'tensorboard'


@dataclass 
class MetricsResult:
    """Container for evaluation metrics."""
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
        """Convert to dictionary (excluding arrays)."""
        return {
            'mae': self.mae,
            'std_MAE': self.mae_std,
            'rmse': self.rmse,
            'rmse_states': self.rmse_states,
            'pcc': self.pcc,
            'pcc_states': self.pcc_states,
            'MAPE': self.mape,
            'R2': self.r2,
            'R2_states': self.r2_states,
            'Var': self.var,
            'Vars': self.var_states,
            'Peak': self.peak_mae
        }


def train_epoch(model, data_loader, optimizer, batch_size: int, 
                horizon: int, device: torch.device,
                max_grad_norm: float = 1.0) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        data_loader: DataBasicLoader instance
        optimizer: PyTorch optimizer
        batch_size: Batch size
        horizon: Prediction horizon
        device: Torch device
        max_grad_norm: Maximum gradient norm for clipping (stabilizes training)
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    n_samples = 0.0

    for inputs in data_loader.get_batches(data_loader.train, batch_size, shuffle=True):
        X, Y, index = inputs[0], inputs[1], inputs[2]
        
        optimizer.zero_grad()
        output, attn_reg_loss = model(X, index)
        
        # Expand target for multi-horizon prediction
        y_expanded = Y.unsqueeze(1).expand(-1, horizon, -1)
        loss = nn.MSELoss()(output, y_expanded) + attn_reg_loss
        
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        
        n_samples += output.size(0) * data_loader.m
        
    return total_loss / n_samples


def evaluate(model, data_loader, batch_size: int, horizon: int,
             device: torch.device, dataset: str = 'val', 
             compute_pcc: bool = True) -> MetricsResult:
    """
    Evaluate model on a dataset split.
    
    Args:
        model: Neural network model
        data_loader: DataBasicLoader instance
        batch_size: Batch size
        horizon: Prediction horizon
        device: Torch device
        dataset: Which split to evaluate ('val' or 'test')
        compute_pcc: Whether to compute Pearson correlation
        
    Returns:
        MetricsResult with all evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    n_samples = 0.0
    
    y_true_list, y_pred_list, x_value_list = [], [], []
    
    # Select data split
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

    # Concatenate batches
    x_value_mx = torch.cat(x_value_list)
    y_pred_mx = torch.cat(y_pred_list)
    y_true_mx = torch.cat(y_true_list)
    
    # Use last horizon step for metrics
    y_pred_mx = y_pred_mx[:, -1, :]

    # Convert to original scale
    x_value_states = x_value_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    y_true_states = y_true_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min
    y_pred_states = y_pred_mx.numpy() * (data_loader.max - data_loader.min) + data_loader.min

    # Per-region metrics
    rmse_states = np.mean(np.sqrt(
        mean_squared_error(y_true_states, y_pred_states, multioutput='raw_values')
    ))
    raw_mae = mean_absolute_error(y_true_states, y_pred_states, multioutput='raw_values')
    std_mae = np.std(raw_mae)
    
    # Per-region PCC
    if compute_pcc:
        pcc_tmp = []
        for k in range(data_loader.m):
            # Check if either array is constant (no variance)
            if np.std(y_true_states[:, k]) < 1e-10 or np.std(y_pred_states[:, k]) < 1e-10:
                # Correlation undefined for constant arrays, use 0
                pcc_tmp.append(0.0)
            else:
                corr, _ = pearsonr(y_true_states[:, k], y_pred_states[:, k])
                pcc_tmp.append(corr)
        pcc_states = np.mean(np.array(pcc_tmp))
    else:
        pcc_states = 1.0
        
    r2_states = np.mean(r2_score(y_true_states, y_pred_states, multioutput='raw_values'))
    var_states = np.mean(explained_variance_score(y_true_states, y_pred_states, multioutput='raw_values'))

    # Flattened metrics
    y_true_flat = y_true_states.flatten()
    y_pred_flat = y_pred_states.flatten()
    
    rmse = sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    mape = np.mean(np.abs((y_pred_flat - y_true_flat) / (y_true_flat + 1e-5))) / 1e7
    
    if compute_pcc:
        # Check if either array is constant
        if np.std(y_true_flat) < 1e-10 or np.std(y_pred_flat) < 1e-10:
            pcc = 0.0
        else:
            pcc, _ = pearsonr(y_true_flat, y_pred_flat)
    else:
        pcc = 1.0
        
    r2 = r2_score(y_true_flat, y_pred_flat)
    var = explained_variance_score(y_true_flat, y_pred_flat)
    peak_mae_val = peak_error(y_true_states.copy(), y_pred_states.copy(), data_loader.peak_thold)

    return MetricsResult(
        loss=total_loss / n_samples,
        mae=mae,
        mae_std=std_mae,
        rmse=rmse,
        rmse_states=rmse_states,
        pcc=pcc,
        pcc_states=pcc_states,
        mape=mape,
        r2=r2,
        r2_states=r2_states,
        var=var,
        var_states=var_states,
        peak_mae=peak_mae_val,
        y_true=y_true_states,
        y_pred=y_pred_states
    )


class Trainer:
    """
    High-level training orchestrator for MSAGAT-Net models.
    
    Handles:
        - Training loop with early stopping
        - Validation monitoring
        - Model checkpointing
        - TensorBoard logging
        - Final evaluation
    
    Args:
        model: Neural network model
        data_loader: DataBasicLoader instance
        config: TrainingConfig or args namespace
        log_token: Identifier for logging/saving
    """
    
    def __init__(self, model, data_loader, config, log_token: str = 'model'):
        self.model = model
        self.data_loader = data_loader
        self.log_token = log_token
        
        # Handle both TrainingConfig and argparse namespace
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
        
        # Setup device
        if hasattr(config, 'cuda') and config.cuda:
            self.device = torch.device(f'cuda:{config.gpu}')
        else:
            self.device = torch.device('cpu')
            
        # Horizon from config
        self.horizon = config.horizon
        
        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # TensorBoard writer
        self.writer = None
        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join('tensorboard', log_token)
                os.makedirs(tb_dir, exist_ok=True)
                self.writer = SummaryWriter(tb_dir)
                logger.info(f'TensorBoard logging to {tb_dir}')
            except ImportError:
                logger.warning('TensorBoard not available')
        
        # Training state
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val = float('inf')
        self.best_epoch = 0
        self.bad_counter = 0
        
    def train(self) -> MetricsResult:
        """
        Run full training loop.
        
        Returns:
            MetricsResult from final test evaluation
        """
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        print('Begin training')
        print(f'Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}')
        
        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = train_epoch(
                self.model, self.data_loader, self.optimizer,
                self.config.batch_size, self.horizon, self.device,
                max_grad_norm=self.config.max_grad_norm
            )
            
            # Validate
            val_metrics = evaluate(
                self.model, self.data_loader, self.config.batch_size,
                self.horizon, self.device, dataset='val'
            )
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics.loss)
            
            # Log
            elapsed = time.time() - epoch_start
            print(f'Epoch {epoch:3d} | time: {elapsed:5.2f}s | '
                  f'train_loss: {train_loss:.8f} | val_loss: {val_metrics.loss:.8f}')
            
            if self.writer:
                self._log_tensorboard(epoch, train_loss, val_metrics)
            
            # Check for improvement
            if val_metrics.loss < self.best_val:
                self.best_val = val_metrics.loss
                self.best_epoch = epoch
                self.bad_counter = 0
                self._save_checkpoint()
                
                # Evaluate on test
                test_metrics = evaluate(
                    self.model, self.data_loader, self.config.batch_size,
                    self.horizon, self.device, dataset='test'
                )
                print(f'TEST MAE {test_metrics.mae:.4f} RMSE {test_metrics.rmse:.4f} '
                      f'PCC {test_metrics.pcc:.4f} R2 {test_metrics.r2:.4f}')
            else:
                self.bad_counter += 1
            
            # Early stopping
            if self.bad_counter >= self.config.patience:
                print(f'Early stopping at epoch {epoch}')
                break
        
        # Load best model and final evaluation
        self._load_best_checkpoint()
        final_metrics = evaluate(
            self.model, self.data_loader, self.config.batch_size,
            self.horizon, self.device, dataset='test'
        )
        
        print(f'\nFinal TEST - MAE: {final_metrics.mae:.4f}, RMSE: {final_metrics.rmse:.4f}, '
              f'PCC: {final_metrics.pcc:.4f}, R2: {final_metrics.r2:.4f}')
        
        if self.writer:
            self.writer.close()
            
        return final_metrics
    
    def _log_tensorboard(self, epoch: int, train_loss: float, metrics: MetricsResult):
        """Log metrics to TensorBoard."""
        self.writer.add_scalars('data/loss', {'train': train_loss, 'val': metrics.loss}, epoch)
        self.writer.add_scalars('data/mae', {'val': metrics.mae}, epoch)
        self.writer.add_scalars('data/rmse', {'val': metrics.rmse_states}, epoch)
        self.writer.add_scalars('data/pcc', {'val': metrics.pcc}, epoch)
        self.writer.add_scalars('data/pcc_states', {'val': metrics.pcc_states}, epoch)
        self.writer.add_scalars('data/R2', {'val': metrics.r2}, epoch)
        self.writer.add_scalars('data/R2_states', {'val': metrics.r2_states}, epoch)
        self.writer.add_scalars('data/var', {'val': metrics.var}, epoch)
        self.writer.add_scalars('data/var_states', {'val': metrics.var_states}, epoch)
        self.writer.add_scalars('data/peak_mae', {'val': metrics.peak_mae}, epoch)
    
    def _save_checkpoint(self):
        """Save model checkpoint."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        model_path = os.path.join(self.config.save_dir, f'{self.log_token}.pt')
        torch.save(self.model.state_dict(), model_path)
        
        # Backup best model - use torch.save instead of shutil.copy for Windows compatibility
        best_path = os.path.join(self.config.save_dir, 'best_model.pt')
        torch.save(self.model.state_dict(), best_path)
        print(f'Best validation epoch: {self.best_epoch}')
    
    def _load_best_checkpoint(self):
        """Load best model checkpoint."""
        model_path = os.path.join(self.config.save_dir, f'{self.log_token}.pt')
        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location='cpu')
            )
