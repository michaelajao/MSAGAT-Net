#!/usr/bin/env python3
"""
Experiment Runner for MSAGAT-Net Paper

Consolidated training script that handles both single experiments and batch runs.

Usage:
    # Run all experiments (main + ablation)
    python -m src.scripts.experiments
    
    # Run specific datasets
    python -m src.scripts.experiments --datasets japan australia-covid
    
    # Run single experiment
    python -m src.scripts.experiments --single --dataset japan --horizon 5 --seed 42
    
    # Run main experiments only
    python -m src.scripts.experiments --experiment main
    
    # Run ablation experiments only  
    python -m src.scripts.experiments --experiment ablation
    
    # Dry run (show what would run)
    python -m src.scripts.experiments --dry-run
"""

import os
import sys
import random
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from argparse import Namespace

import numpy as np
import torch

# Suppress TensorFlow oneDNN warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU Performance Optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import DataBasicLoader
from src.models import MSAGATNet_Ablation
from src.training import Trainer
from src.utils import plot_loss_curves, save_metrics

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(BASE_DIR, 'report', 'figures')
RESULTS_DIR = os.path.join(BASE_DIR, 'report', 'results')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# OPTIMAL CONFIGURATIONS PER DATASET (from ablation studies)
# =============================================================================

DATASET_CONFIGS = {
    'japan': {
        'sim_mat': 'japan-adj',
        'num_nodes': 47,
        'horizons': [3, 5, 10, 15],
        'use_adj_prior': False,
        'use_graph_bias': False,
        'notes': 'Large graph - pure learned attention'
    },
    'australia-covid': {
        'sim_mat': 'australia-adj',
        'num_nodes': 8,
        'horizons': [3, 7, 14],
        'use_adj_prior': True,
        'use_graph_bias': True,
        'notes': 'Small graph - adjacency prior helps'
    },
    'nhs_timeseries': {
        'sim_mat': 'nhs-adj',
        'num_nodes': 7,
        'horizons': [3, 7, 14],
        'use_adj_prior': True,
        'use_graph_bias': True,
        'notes': 'Small graph - adjacency prior helps'
    },
    'spain-covid': {
        'sim_mat': 'spain-adj',
        'num_nodes': 17,
        'horizons': [3, 7, 14],
        'use_adj_prior': True,
        'use_graph_bias': True,
        'notes': 'Medium graph - adj helps for longer horizons'
    },
    'ltla_timeseries': {
        'sim_mat': 'ltla-adj',
        'num_nodes': 307,
        'horizons': [3, 7, 14],
        'use_adj_prior': False,
        'use_graph_bias': False,
        'notes': 'Very large graph - learned attention only'
    },
    'region785': {
        'sim_mat': 'region-adj',
        'num_nodes': 10,
        'horizons': [3, 5, 10, 15],
        'use_adj_prior': True,
        'use_graph_bias': True,
        'notes': 'Small graph - use adjacency prior'
    },
    'state360': {
        'sim_mat': 'state-adj-49',
        'num_nodes': 49,
        'horizons': [3, 5, 10, 15],
        'use_adj_prior': False,
        'use_graph_bias': False,
        'notes': 'Large graph - learned attention only'
    },
}

# Training hyperparameters
TRAIN_CONFIG = {
    'epochs': 1500,
    'patience': 100,
    'lr': 1e-3,
    'weight_decay': 5e-4,
    'batch': 64,
    'window': 20,
    'dropout': 0.2,
    'num_scales': 4,
    'hidden_dim': 32,
    'attention_heads': 4,
    'bottleneck_dim': 8,
}

# Seeds for reproducibility
SEEDS = [5, 30, 45, 123, 1000]

# Ablation configurations
ABLATIONS = ['none', 'no_agam', 'no_mtfm', 'no_pprm']


# =============================================================================
# SINGLE EXPERIMENT RUNNER
# =============================================================================

def run_single_experiment(
    dataset: str,
    horizon: int,
    seed: int,
    ablation: str = 'none',
    save_dir: str = 'save_all',
    use_adj_prior: Optional[bool] = None,
    use_graph_bias: Optional[bool] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Run a single training experiment.
    
    Args:
        dataset: Dataset name
        horizon: Prediction horizon
        seed: Random seed
        ablation: Ablation variant ('none', 'no_agam', 'no_mtfm', 'no_pprm')
        save_dir: Directory to save model checkpoints
        use_adj_prior: Override dataset config for adjacency prior
        use_graph_bias: Override dataset config for graph bias
        verbose: Print progress
        
    Returns:
        Dictionary of final metrics
    """
    config = DATASET_CONFIGS[dataset]
    
    # Use overrides or dataset defaults
    adj_prior = use_adj_prior if use_adj_prior is not None else config['use_adj_prior']
    graph_bias = use_graph_bias if use_graph_bias is not None else config['use_graph_bias']
    
    # Build args namespace
    args = Namespace(
        dataset=dataset,
        sim_mat=config['sim_mat'],
        window=TRAIN_CONFIG['window'],
        horizon=horizon,
        train=0.5,
        val=0.2,
        test=0.3,
        epochs=TRAIN_CONFIG['epochs'],
        batch=TRAIN_CONFIG['batch'],
        lr=TRAIN_CONFIG['lr'],
        weight_decay=TRAIN_CONFIG['weight_decay'],
        dropout=TRAIN_CONFIG['dropout'],
        patience=TRAIN_CONFIG['patience'],
        ablation=ablation,
        hidden_dim=TRAIN_CONFIG['hidden_dim'],
        attention_heads=TRAIN_CONFIG['attention_heads'],
        attention_regularization_weight=1e-5,
        num_scales=TRAIN_CONFIG['num_scales'],
        kernel_size=3,
        feature_channels=16,
        bottleneck_dim=TRAIN_CONFIG['bottleneck_dim'],
        use_adj_prior=adj_prior,
        adj_weight=0.1,
        use_graph_bias=graph_bias,
        adaptive=False,
        seed=seed,
        gpu=0,
        cuda=torch.cuda.is_available(),
        save_dir=save_dir,
        mylog=True,
        extra='',
        label='',
        pcc='',
    )
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.cuda.set_device(0)
    torch.backends.cudnn.deterministic = True
    
    # Model naming
    model_name = 'MSTAGAT-Net'
    log_token = f"{model_name}.{dataset}.w-{args.window}.h-{horizon}.{ablation}.seed-{seed}"
    if adj_prior:
        log_token += '.with_adj'
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training: {dataset} | h={horizon} | seed={seed} | ablation={ablation}")
        print(f"Settings: adj_prior={adj_prior}, graph_bias={graph_bias}")
        print(f"{'='*60}")
    
    # Load data
    data_loader = DataBasicLoader(args)
    
    # Create model
    model = MSAGATNet_Ablation(args, data_loader)
    if args.cuda:
        model.cuda()
    
    if verbose:
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {params:,}")
    
    # Train
    trainer = Trainer(model, data_loader, args, log_token)
    final_metrics = trainer.train()
    
    # Create output directories
    dataset_figures_dir = os.path.join(FIGURES_DIR, dataset)
    dataset_results_dir = os.path.join(RESULTS_DIR, dataset)
    os.makedirs(dataset_figures_dir, exist_ok=True)
    os.makedirs(dataset_results_dir, exist_ok=True)
    
    # Save loss curves (seed 5 only)
    if seed == 5:
        loss_fig_path = os.path.join(dataset_figures_dir, f"loss_curve_{log_token}.png")
        plot_loss_curves(trainer.train_losses, trainer.val_losses, loss_fig_path, args)
    
    # Save metrics
    results_csv = os.path.join(dataset_results_dir, f"final_metrics_{log_token}.csv")
    save_metrics(
        final_metrics.to_dict(),
        results_csv,
        dataset,
        args.window,
        horizon,
        logger,
        model_name,
        ablation,
        seed,
        adj_prior
    )
    
    if verbose:
        print(f"Results saved to {results_csv}")
    
    return final_metrics.to_dict()


# =============================================================================
# BATCH EXPERIMENT RUNNERS
# =============================================================================

def run_main_experiments(datasets: List[str], seeds: List[int], dry_run: bool = False):
    """Run main comparison experiments (Tables 1 & 2)."""
    
    print("\n" + "="*80)
    print("MAIN EXPERIMENTS: MSAGAT-Net with Optimal Settings")
    print("="*80)
    
    total = sum(len(DATASET_CONFIGS[d]['horizons']) for d in datasets) * len(seeds)
    completed = 0
    failed = 0
    
    for dataset in datasets:
        config = DATASET_CONFIGS[dataset]
        print(f"\n{dataset} ({config['num_nodes']} nodes): {config['notes']}")
        
        for horizon in config['horizons']:
            for seed in seeds:
                completed += 1
                print(f"\n[{completed}/{total}] {dataset} h={horizon} seed={seed}")
                
                if dry_run:
                    print("  [DRY RUN] Skipping...")
                    continue
                
                try:
                    run_single_experiment(
                        dataset=dataset,
                        horizon=horizon,
                        seed=seed,
                        ablation='none',
                        verbose=True
                    )
                    print(f"  [OK] Completed")
                except Exception as e:
                    failed += 1
                    print(f"  [FAIL] {e}")
    
    print(f"\n{'='*80}")
    print(f"MAIN EXPERIMENTS: {completed-failed}/{total} completed, {failed} failed")
    print(f"{'='*80}")


def run_ablation_experiments(datasets: List[str], seeds: List[int], dry_run: bool = False):
    """Run ablation study experiments (Tables 3 & 4)."""
    
    print("\n" + "="*80)
    print("ABLATION EXPERIMENTS: Component Contributions")
    print("="*80)
    
    ablation_horizons = [3, 7, 14]
    total = len(datasets) * len(ABLATIONS) * len(ablation_horizons) * len(seeds[:1])
    completed = 0
    failed = 0
    
    for dataset in datasets:
        config = DATASET_CONFIGS[dataset]
        print(f"\n{dataset} ({config['num_nodes']} nodes)")
        
        for ablation in ABLATIONS:
            for horizon in ablation_horizons:
                for seed in seeds[:1]:  # Single seed for ablation
                    completed += 1
                    print(f"\n[{completed}/{total}] {dataset} h={horizon} ablation={ablation}")
                    
                    if dry_run:
                        print("  [DRY RUN] Skipping...")
                        continue
                    
                    try:
                        run_single_experiment(
                            dataset=dataset,
                            horizon=horizon,
                            seed=seed,
                            ablation=ablation,
                            verbose=True
                        )
                        print(f"  [OK] Completed")
                    except Exception as e:
                        failed += 1
                        print(f"  [FAIL] {e}")
    
    print(f"\n{'='*80}")
    print(f"ABLATION EXPERIMENTS: {completed-failed}/{total} completed, {failed} failed")
    print(f"{'='*80}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MSAGAT-Net Experiment Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments
  python -m src.scripts.experiments
  
  # Run single experiment  
  python -m src.scripts.experiments --single --dataset japan --horizon 5 --seed 42
  
  # Run main experiments for specific datasets
  python -m src.scripts.experiments --experiment main --datasets japan australia-covid
  
  # Dry run (show what would run)
  python -m src.scripts.experiments --dry-run
        """
    )
    
    # Mode selection
    parser.add_argument('--single', action='store_true',
                        help='Run single experiment (requires --dataset, --horizon, --seed)')
    parser.add_argument('--experiment', choices=['main', 'ablation', 'all'],
                        default='all', help='Which batch experiments to run')
    
    # Single experiment args
    parser.add_argument('--dataset', type=str, default='japan',
                        help='Dataset name')
    parser.add_argument('--horizon', type=int, default=5,
                        help='Prediction horizon')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--ablation', type=str, default='none',
                        choices=['none', 'no_agam', 'no_mtfm', 'no_pprm'],
                        help='Ablation variant')
    
    # Batch experiment args
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to run (default: all)')
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                        help='Random seeds for batch runs')
    
    # Options
    parser.add_argument('--save_dir', type=str, default='save_all',
                        help='Model save directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would run without executing')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.single:
        # Run single experiment
        print("="*80)
        print("SINGLE EXPERIMENT MODE")
        print("="*80)
        
        if args.dataset not in DATASET_CONFIGS:
            print(f"Error: Unknown dataset '{args.dataset}'")
            print(f"Available: {list(DATASET_CONFIGS.keys())}")
            sys.exit(1)
        
        metrics = run_single_experiment(
            dataset=args.dataset,
            horizon=args.horizon,
            seed=args.seed,
            ablation=args.ablation,
            save_dir=args.save_dir,
            verbose=True
        )
        
        print(f"\nFinal Metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
    else:
        # Run batch experiments
        datasets = args.datasets or list(DATASET_CONFIGS.keys())
        datasets = [d for d in datasets if d in DATASET_CONFIGS]
        
        print("="*80)
        print("MSAGAT-Net Batch Experiments")
        print(f"Datasets: {datasets}")
        print(f"Seeds: {args.seeds}")
        print(f"Experiment: {args.experiment}")
        print("="*80)
        
        if args.experiment in ['main', 'all']:
            run_main_experiments(datasets, args.seeds, args.dry_run)
        
        if args.experiment in ['ablation', 'all']:
            run_ablation_experiments(datasets, args.seeds, args.dry_run)
        
        print("\n" + "="*80)
        print("EXPERIMENTS COMPLETE")
        print("="*80)
    
    sys.exit(0)


if __name__ == '__main__':
    main()
