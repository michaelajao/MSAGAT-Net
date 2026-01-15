#!/usr/bin/env python3
"""
Batch Experiment Runner for MSAGAT-Net

Runs training across multiple datasets, horizons, and ablation configurations.

Usage:
    python -m src.scripts.run_experiments
    python -m src.scripts.run_experiments --datasets japan region785
    python -m src.scripts.run_experiments --ablations none no_agam
"""

import os
import sys
import subprocess
import logging
import glob
import argparse
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICS_DIR = os.path.join(BASE_DIR, 'report', 'results')
SAVE_DIR = os.path.join(BASE_DIR, 'save_all')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
TRAIN_SCRIPT = os.path.join(BASE_DIR, 'src', 'scripts', 'train.py')

# Consolidated file names
ALL_RESULTS_CSV = "all_results.csv"
ALL_ABLATION_SUMMARY_CSV = "all_ablation_summary.csv"
ALL_ABLATION_REPORT_TXT = "all_ablation_report.txt"

# Default dataset configurations: (name, adjacency, horizons)
DEFAULT_DATASET_CONFIGS = [
    ('japan', 'japan-adj', [3, 5, 10, 15]),
    ('region785', 'region-adj', [3, 5, 10, 15]),
    ('state360', 'state-adj-49', [3, 5, 10, 15]),
    ('australia-covid', 'australia-adj', [3, 7, 14]),
    ('spain-covid', 'spain-adj', [3, 7, 14]),
    ('nhs_timeseries', 'nhs-adj', [3, 7, 14]),
    ('ltla_timeseries', 'ltla-adj', [3, 7, 14]),
]

ABLATIONS = ['none', 'no_agam', 'no_mtfm', 'no_pprm']
DEFAULT_WINDOW = 20


def setup_logging() -> logging.Logger:
    """Setup logging to file and console."""
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"run_experiments_{ts}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('run_experiments')
    logger.info(f"Logging to {log_file}")
    return logger


def load_metrics(dataset: str, window: int, horizon: int, 
                 ablation: str) -> Optional[pd.DataFrame]:
    """Load metrics from consolidated or individual CSV file in dataset folder."""
    # Dataset-specific folder
    dataset_results_dir = os.path.join(METRICS_DIR, dataset)
    
    # Try consolidated file in dataset folder first
    consolidated_path = os.path.join(dataset_results_dir, ALL_RESULTS_CSV)
    if os.path.exists(consolidated_path):
        df = pd.read_csv(consolidated_path)
        mask = (
            (df['dataset'] == dataset) & 
            (df['window'] == window) & 
            (df['horizon'] == horizon) & 
            (df['ablation'] == ablation)
        )
        if mask.any():
            return df[mask].iloc[[0]]
    
    # Fallback to individual files in dataset folder
    pattern = f"final_metrics_*{dataset}*w-{window}*h-{horizon}*{ablation}.csv"
    matches = glob.glob(os.path.join(dataset_results_dir, pattern))
    if matches:
        return pd.read_csv(matches[0])
    return None


def get_metric(df: pd.DataFrame, key: str) -> float:
    """Extract metric value from DataFrame."""
    if df is None:
        return float('nan')
    
    search = key.strip().lower().replace('-', '').replace('_', '')
    for col in df.columns:
        if col.strip().lower().replace('-', '').replace('_', '') == search:
            return df[col].iloc[0]
    return float('nan')


def generate_ablation_summary(dataset: str, window: int, horizon: int,
                              logger: logging.Logger) -> Optional[pd.DataFrame]:
    """Generate ablation study summary for a dataset/horizon combination."""
    records = {}
    
    # Load baseline (none) metrics
    base_df = load_metrics(dataset, window, horizon, 'none')
    if base_df is None:
        logger.warning(f"No baseline metrics for {dataset} h={horizon}")
        return None
    
    base_vals = {m: get_metric(base_df, m) for m in ['mae', 'rmse', 'pcc', 'r2']}
    
    # Process all ablations
    for abl in ABLATIONS:
        df = load_metrics(dataset, window, horizon, abl)
        if df is None:
            continue
            
        row = {m.upper(): get_metric(df, m) for m in ['mae', 'rmse', 'pcc', 'r2']}
        
        # Calculate percentage changes for ablation variants
        if abl != 'none':
            for m in ['mae', 'rmse', 'pcc', 'r2']:
                base_val = base_vals[m]
                val = get_metric(df, m)
                if base_val and not pd.isna(val):
                    row[f"{m.upper()}_CHANGE"] = 100 * (val - base_val) / base_val
        
        records[abl] = row
    
    if not records:
        return None
    
    # Create summary DataFrame
    summary = pd.DataFrame.from_dict(records, orient='index')
    summary['dataset'] = dataset
    summary['window'] = window
    summary['horizon'] = horizon
    summary['ablation'] = summary.index
    summary = summary.reset_index(drop=True)
    
    # Save to dataset-specific folder
    dataset_results_dir = os.path.join(METRICS_DIR, dataset)
    os.makedirs(dataset_results_dir, exist_ok=True)
    
    # Append to consolidated ablation summary CSV in dataset folder
    consolidated_csv = os.path.join(dataset_results_dir, ALL_ABLATION_SUMMARY_CSV)
    if os.path.exists(consolidated_csv):
        df_old = pd.read_csv(consolidated_csv)
        mask = ~(
            (df_old['dataset'] == dataset) & 
            (df_old['window'] == window) & 
            (df_old['horizon'] == horizon)
        )
        df_old = df_old[mask]
        summary = pd.concat([df_old, summary], ignore_index=True)
    
    summary = summary.sort_values(['dataset', 'window', 'horizon', 'ablation']).reset_index(drop=True)
    summary.to_csv(consolidated_csv, index=False)
    logger.info(f"Ablation summary appended to: {consolidated_csv}")
    
    return summary[summary['dataset'] == dataset]


def build_train_command(dataset: str, sim_mat: str, window: int, 
                        horizon: int, ablation: str, use_cuda: bool = True) -> List[str]:
    """Build training command with GPU support if available."""
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--dataset', dataset,
        '--sim_mat', sim_mat,
        '--window', str(window),
        '--horizon', str(horizon),
        '--ablation', ablation,
        '--save_dir', SAVE_DIR,
    ]
    
    # Add CUDA flag if requested (enabled by default in train.py)
    # User can disable with --no-cuda in train.py directly
    if not use_cuda:
        cmd.append('--no-cuda')
    
    return cmd


def run_experiment(dataset: str, sim_mat: str, window: int, horizon: int,
                   ablation: str, logger: logging.Logger, skip_existing: bool = True) -> bool:
    """Run a single experiment."""
    # Check if already completed
    if skip_existing:
        existing = load_metrics(dataset, window, horizon, ablation)
        if existing is not None:
            logger.info(f"Skipping {dataset} w={window} h={horizon} abl={ablation} (exists)")
            return True
    
    logger.info(f"Running: {dataset} w={window} h={horizon} abl={ablation}")
    
    cmd = build_train_command(dataset, sim_mat, window, horizon, ablation)
    
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Completed: {dataset} w={window} h={horizon} abl={ablation}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {dataset} w={window} h={horizon} abl={ablation}: {e}")
        return False


def run_all_experiments(configs: List[Tuple], ablations: List[str],
                        logger: logging.Logger, skip_existing: bool = True):
    """Run all experiment configurations."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    total = sum(len(horizons) * len(ablations) for _, _, horizons in configs)
    completed = 0
    failed = 0
    
    logger.info(f"Starting {total} experiments")
    
    for dataset, sim_mat, horizons in configs:
        for horizon in horizons:
            for ablation in ablations:
                success = run_experiment(
                    dataset, sim_mat, DEFAULT_WINDOW, horizon, ablation,
                    logger, skip_existing
                )
                if success:
                    completed += 1
                else:
                    failed += 1
            
            # Generate ablation summary after each horizon
            generate_ablation_summary(dataset, DEFAULT_WINDOW, horizon, logger)
    
    logger.info(f"Experiments completed: {completed}/{total}, failed: {failed}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run MSAGAT-Net experiments')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets to run (default: all)')
    parser.add_argument('--ablations', nargs='+', default=ABLATIONS,
                        help='Ablations to run')
    parser.add_argument('--horizons', nargs='+', type=int, default=None,
                        help='Override horizons for all datasets')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip existing experiments')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run of all experiments')
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    logger = setup_logging()
    
    # Filter configs if specific datasets requested
    if args.datasets:
        configs = [
            (d, s, h) for d, s, h in DEFAULT_DATASET_CONFIGS 
            if d in args.datasets
        ]
    else:
        configs = DEFAULT_DATASET_CONFIGS
    
    # Override horizons if specified
    if args.horizons:
        configs = [(d, s, args.horizons) for d, s, _ in configs]
    
    skip_existing = not args.force and args.skip_existing
    
    run_all_experiments(configs, args.ablations, logger, skip_existing)


if __name__ == '__main__':
    main()
