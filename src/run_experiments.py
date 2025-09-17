#!/usr/bin/env python3
"""Batch experiment runner for MSAGAT-Net training across datasets and ablations."""
import os
import sys
import subprocess
import logging
import glob
import pandas as pd
from datetime import datetime

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(BASE_DIR, 'report', 'results')
SAVE_DIR = os.path.join(BASE_DIR, 'save_all')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
TRAIN_SCRIPT = os.path.join(BASE_DIR, 'src', 'train.py')

# Dataset configurations
DATASET_CONFIGS = [
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

def setup_logging():
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

def find_metrics_file(dataset, window, horizon, ablation):
    """Find existing metrics file for given configuration."""
    pattern = f"final_metrics_*{dataset}*w-{window}*h-{horizon}*{ablation}.csv"
    matches = glob.glob(os.path.join(METRICS_DIR, pattern))
    return matches[0] if matches else None

def load_metrics(dataset, window, horizon, ablation):
    """Load metrics DataFrame from CSV file."""
    path = find_metrics_file(dataset, window, horizon, ablation)
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"Failed to read metrics CSV {path}: {e}")
        return None

def get_metric(df, key):
    """Extract metric value from DataFrame."""
    if df is None:
        return float('nan')
    
    search = key.strip().lower().replace('-', '').replace('_', '')
    for col in df.columns:
        if col.strip().lower().replace('-', '').replace('_', '') == search:
            return df[col].iloc[0]
    return float('nan')

def generate_ablation_summary(dataset, window, horizon):
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
    
    # Save summary
    summary = pd.DataFrame.from_dict(records, orient='index')
    os.makedirs(METRICS_DIR, exist_ok=True)
    out_path = os.path.join(METRICS_DIR, f"ablation_summary_{dataset}.w-{window}.h-{horizon}.csv")
    
    try:
        summary.to_csv(out_path)
        logger.info(f"Saved summary: {out_path}")
        
        # Generate text report
        report_lines = [
            f"Ablation Report for {dataset} (horizon={horizon})",
            "=" * 60,
            summary.to_string()
        ]
        report_path = os.path.join(METRICS_DIR, f"ablation_report_{dataset}.w-{window}.h-{horizon}.txt")
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))
        logger.info(f"Saved report: {report_path}")
        
    except Exception as e:
        logger.error(f"Failed to save summary: {e}")
    
    return summary

def build_train_command(dataset, sim_mat, window, horizon, ablation):
    """Build training command with essential arguments."""
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--dataset', dataset,
        '--sim_mat', sim_mat,
        '--window', str(window),
        '--horizon', str(horizon),
        '--ablation', ablation,
        '--save_dir', SAVE_DIR,
        '--epochs', '1500',
        '--batch', '32',
        '--lr', '1e-3',
        '--weight_decay', '5e-4',
        '--dropout', '0.2',
        '--mylog'
    ]
    
    # Add CUDA if available
    try:
        import torch
        if torch.cuda.is_available():
            cmd.extend(['--cuda', '--gpu', '0'])
    except ImportError:
        pass
    
    return cmd

def run_training(cmd):
    """Execute training command."""
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Training completed: {' '.join(cmd[:8])}...")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        return False

def run_all_experiments():
    """Run training for all dataset/horizon/ablation combinations."""
    if not os.path.exists(TRAIN_SCRIPT):
        logger.error(f"Training script not found: {TRAIN_SCRIPT}")
        return
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    for dataset, sim_mat, horizons in DATASET_CONFIGS:
        for horizon in horizons:
            for ablation in ABLATIONS:
                # Skip if metrics already exist
                if find_metrics_file(dataset, DEFAULT_WINDOW, horizon, ablation):
                    logger.info(f"Skipping existing: {dataset} h={horizon} ablation={ablation}")
                    continue
                
                cmd = build_train_command(dataset, sim_mat, DEFAULT_WINDOW, horizon, ablation)
                run_training(cmd)

def main():
    """Main execution function."""
    global logger
    logger = setup_logging()
    
    # Run all experiments
    logger.info("Starting batch experiments")
    run_all_experiments()
    
    # Generate summaries
    logger.info("Generating ablation summaries")
    for dataset, _, horizons in DATASET_CONFIGS:
        for horizon in horizons:
            generate_ablation_summary(dataset, DEFAULT_WINDOW, horizon)
    
    logger.info("All experiments complete")

if __name__ == '__main__':
    main()

