#!/usr/bin/env python3
"""
Training CLI for MSAGAT-Net

Usage:
    python -m src.scripts.train --dataset japan --sim_mat japan-adj --horizon 5
    
    # With ablation
    python -m src.scripts.train --dataset japan --ablation no_agam
    
    # With GPU
    python -m src.scripts.train --dataset japan --cuda --gpu 0
"""

import os
import sys
import random
import logging
import argparse
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import DataBasicLoader
from src.models import MSTAGAT_Net, MSAGATNet_Ablation
from src.training import Trainer, evaluate
from src.utils import (
    visualize_matrices, 
    visualize_predictions, 
    plot_loss_curves, 
    save_metrics
)

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Output directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(BASE_DIR, 'report', 'figures')
RESULTS_DIR = os.path.join(BASE_DIR, 'report', 'results')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='MSAGAT-Net Training')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='region785',
                        help='Dataset name')
    parser.add_argument('--sim_mat', type=str, default='region-adj',
                        help='Adjacency matrix name')
    parser.add_argument('--window', type=int, default=20,
                        help='Input window size')
    parser.add_argument('--horizon', type=int, default=5,
                        help='Prediction horizon')
    
    # Data split arguments
    parser.add_argument('--train', type=float, default=0.5,
                        help='Training data fraction')
    parser.add_argument('--val', type=float, default=0.2,
                        help='Validation data fraction')
    parser.add_argument('--test', type=float, default=0.3,
                        help='Test data fraction')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=1500,
                        help='Maximum epochs')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--patience', type=int, default=100,
                        help='Early stopping patience')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='msagat',
                        choices=['msagat', 'ablation'],
                        help='Model type')
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'no_agam', 'no_mtfm', 'no_pprm'],
                        help='Ablation variant')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden dimension')
    parser.add_argument('--attention_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--attention_regularization_weight', type=float, default=1e-5,
                        help='Attention regularization weight')
    parser.add_argument('--num_scales', type=int, default=4,
                        help='Number of temporal scales')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Kernel size for convolutions')
    parser.add_argument('--feature_channels', type=int, default=16,
                        help='Feature channels')
    parser.add_argument('--bottleneck_dim', type=int, default=8,
                        help='Bottleneck dimension')
    
    # Device arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device id (default: 0 if --cuda flag is set)')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use CUDA if available (default: True)')
    parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                        help='Disable CUDA even if available')
    
    # Logging arguments
    parser.add_argument('--save_dir', type=str, default='save',
                        help='Model save directory')
    parser.add_argument('--mylog', action='store_false', default=True,
                        help='Enable TensorBoard logging')
    parser.add_argument('--record', type=str, default='',
                        help='Record file')
    
    # Optional arguments (for data loader compatibility)
    parser.add_argument('--extra', type=str, default='',
                        help='External data directory')
    parser.add_argument('--label', type=str, default='',
                        help='Label file')
    parser.add_argument('--pcc', type=str, default='',
                        help='PCC flag')
    parser.add_argument('--result', type=int, default=0)
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                        help='Start date for visualization')
    parser.add_argument('--no-figures', dest='save_figures', action='store_false', default=True,
                        help='Skip saving figures (useful for multi-seed runs)')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print('--------Parameters--------')
    print(args)
    print('--------------------------')
    
    # Set device
    # GPU setup - use GPU 0 by default if CUDA is available and requested
    if args.cuda and torch.cuda.is_available():
        if args.gpu is None:
            args.gpu = 0  # Default to GPU 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        torch.cuda.set_device(0)  # After setting CUDA_VISIBLE_DEVICES, use device 0
    else:
        args.cuda = False
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    
    logger.info(f'CUDA: {args.cuda}' + (f' (GPU {args.gpu})' if args.cuda else ''))
    
    # Model naming - include seed for multi-seed experiments
    model_name = 'MSTAGAT-Net'
    log_token = f"{model_name}.{args.dataset}.w-{args.window}.h-{args.horizon}.{args.ablation}.seed-{args.seed}"
    
    # Load data
    data_loader = DataBasicLoader(args)
    
    # Create model - USE ABLATION CLASS FOR ALL VARIANTS (including 'none')
    # This ensures fair comparison with identical architecture except ablated components
    logger.info(f'Using MSAGAT-Net with ablation: {args.ablation}')
    model = MSAGATNet_Ablation(args, data_loader)
    
    if args.cuda:
        model.cuda()
    
    device = torch.device(f'cuda:{args.gpu}') if args.cuda else torch.device('cpu')
    
    # Count parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'#params: {pytorch_total_params}')
    
    # Train
    trainer = Trainer(model, data_loader, args, log_token)
    final_metrics = trainer.train()
    
    # Create dataset-specific directories
    dataset_figures_dir = os.path.join(FIGURES_DIR, args.dataset)
    dataset_results_dir = os.path.join(RESULTS_DIR, args.dataset)
    os.makedirs(dataset_figures_dir, exist_ok=True)
    os.makedirs(dataset_results_dir, exist_ok=True)
    
    # Generate visualizations only if save_figures is True
    if args.save_figures:
        loss_fig_path = os.path.join(dataset_figures_dir, f"loss_curve_{log_token}.png")
        plot_loss_curves(trainer.train_losses, trainer.val_losses, loss_fig_path, args)
        logger.info(f"Loss curve saved to {loss_fig_path}")
        
        matrices_fig_path = os.path.join(dataset_figures_dir, f"matrices_{log_token}.png")
        visualize_matrices(data_loader, model, matrices_fig_path, device)
        logger.info(f"Matrices saved to {matrices_fig_path}")
        
        predictions_fig_path = os.path.join(dataset_figures_dir, f"predictions_{log_token}.png")
        visualize_predictions(final_metrics.y_true, final_metrics.y_pred, predictions_fig_path, logger=logger)
        logger.info(f"Predictions saved to {predictions_fig_path}")
    else:
        logger.info("Skipping figure generation (--no-figures)")
    
    # Save metrics in dataset folder (always save metrics)
    results_csv = os.path.join(dataset_results_dir, f"final_metrics_{log_token}.csv")
    save_metrics(
        final_metrics.to_dict(),
        results_csv, 
        args.dataset, 
        args.window, 
        args.horizon, 
        logger, 
        model_name, 
        args.ablation,
        args.seed
    )
    print(f"Saved final metrics to {results_csv}")
    
    # Record to file if requested
    if args.record:
        os.makedirs("result", exist_ok=True)
        with open("result/result.txt", "a", encoding="utf-8") as f:
            f.write(
                f'Model: {model_name}, dataset: {args.dataset}, '
                f'window: {args.window}, horizon: {args.horizon}, '
                f'seed: {args.seed}, ablation: {args.ablation}, '
                f'MAE: {final_metrics.mae:.4f}, RMSE: {final_metrics.rmse:.4f}, '
                f'PCC: {final_metrics.pcc:.4f}, lr: {args.lr}, dropout: {args.dropout}\n'
            )


if __name__ == '__main__':
    main()
