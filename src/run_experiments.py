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

# Consolidated file names
ALL_RESULTS_CSV = "all_results.csv"
ALL_ABLATION_SUMMARY_CSV = "all_ablation_summary.csv"
ALL_ABLATION_REPORT_TXT = "all_ablation_report.txt"

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
MODELS = ['msagat']
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

def find_metrics_file(dataset, window, horizon, ablation, model='msagat'):
    """Find existing metrics file for given configuration.
    First checks consolidated all_results.csv, then falls back to individual files.
    """
    # First check consolidated file
    consolidated_path = os.path.join(METRICS_DIR, ALL_RESULTS_CSV)
    if os.path.exists(consolidated_path):
        df = pd.read_csv(consolidated_path)
        mask = (
            (df['dataset'] == dataset) & 
            (df['window'] == window) & 
            (df['horizon'] == horizon) & 
            (df['ablation'] == ablation)
        )
        if mask.any():
            return consolidated_path  # Return consolidated path if entry exists
    
    # Fallback to individual files for backward compatibility
    if model == 'msagat':
        pattern = f"final_metrics_*{dataset}*w-{window}*h-{horizon}*{ablation}.csv"
    else:
        pattern = f"final_metrics_*{dataset}*w-{window}*h-{horizon}*{ablation}*{model}.csv"
    matches = glob.glob(os.path.join(METRICS_DIR, pattern))
    return matches[0] if matches else None

def load_metrics(dataset, window, horizon, ablation, model='msagat'):
    """Load metrics DataFrame from consolidated or individual CSV file."""
    # First try consolidated file
    consolidated_path = os.path.join(METRICS_DIR, ALL_RESULTS_CSV)
    if os.path.exists(consolidated_path):
        df = pd.read_csv(consolidated_path)
        mask = (
            (df['dataset'] == dataset) & 
            (df['window'] == window) & 
            (df['horizon'] == horizon) & 
            (df['ablation'] == ablation)
        )
        if mask.any():
            return df[mask].iloc[[0]]  # Return single row as DataFrame
    
    # Fallback to individual files
    path = find_metrics_file(dataset, window, horizon, ablation, model)
    if not path or not os.path.exists(path) or path == consolidated_path:
        return None
    return pd.read_csv(path)

def get_metric(df, key):
    """Extract metric value from DataFrame."""
    if df is None:
        return float('nan')
    
    search = key.strip().lower().replace('-', '').replace('_', '')
    for col in df.columns:
        if col.strip().lower().replace('-', '').replace('_', '') == search:
            return df[col].iloc[0]
    return float('nan')

def generate_ablation_summary(dataset, window, horizon, model='msagat'):
    """Generate ablation study summary for a dataset/horizon combination."""
    records = {}
    
    # Load baseline (none) metrics
    base_df = load_metrics(dataset, window, horizon, 'none', model)
    if base_df is None:
        logger.warning(f"No baseline metrics for {dataset} h={horizon} model={model}")
        return None
    
    base_vals = {m: get_metric(base_df, m) for m in ['mae', 'rmse', 'pcc', 'r2']}
    
    # Process all ablations
    for abl in ABLATIONS:
        df = load_metrics(dataset, window, horizon, abl, model)
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
    summary['model'] = model
    summary['ablation'] = summary.index
    summary = summary.reset_index(drop=True)
    
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    # Append to consolidated ablation summary CSV
    consolidated_csv = os.path.join(METRICS_DIR, ALL_ABLATION_SUMMARY_CSV)
    if os.path.exists(consolidated_csv):
        df_old = pd.read_csv(consolidated_csv)
        # Remove existing entries for this dataset/window/horizon
        mask = ~(
            (df_old['dataset'] == dataset) & 
            (df_old['window'] == window) & 
            (df_old['horizon'] == horizon)
        )
        df_old = df_old[mask]
        summary = pd.concat([df_old, summary], ignore_index=True)
    
    # Sort and save
    summary = summary.sort_values(['dataset', 'window', 'horizon', 'ablation']).reset_index(drop=True)
    summary.to_csv(consolidated_csv, index=False)
    logger.info(f"Ablation summary appended to: {consolidated_csv}")
    
    # Append to consolidated text report
    consolidated_txt = os.path.join(METRICS_DIR, ALL_ABLATION_REPORT_TXT)
    report_lines = [
        f"\n{'='*60}",
        f"Ablation Report for {dataset} (window={window}, horizon={horizon}, model={model})",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        summary[summary['dataset'] == dataset].to_string(),
        ""
    ]
    with open(consolidated_txt, 'a', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    logger.info(f"Ablation report appended to: {consolidated_txt}")
    
    return summary[summary['dataset'] == dataset]

def build_train_command(dataset, sim_mat, window, horizon, ablation, model='msagat'):
    """Build training command with essential arguments."""
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--dataset', dataset,
        '--sim_mat', sim_mat,
        '--window', str(window),
        '--horizon', str(horizon),
        '--ablation', ablation,
        '--model', model,
        '--save_dir', SAVE_DIR,
        '--epochs', '1500',
        '--batch', '32',
        '--lr', '1e-3',
        '--weight_decay', '5e-4',
        '--dropout', '0.2',
        '--hidden_dim', '32',
        '--attention_heads', '4',
        '--bottleneck_dim', '8',
        '--mylog'
    ]
    
    # Add CUDA if available
    import torch
    if torch.cuda.is_available():
        cmd.extend(['--cuda', '--gpu', '0'])
    
    return cmd

def run_training(cmd):
    """Execute training command."""
    subprocess.run(cmd, check=True)
    logger.info(f"Training completed: {' '.join(cmd[:8])}...")
    return True

def run_all_experiments():
    """Run training for all dataset/horizon/ablation/model combinations."""
    if not os.path.exists(TRAIN_SCRIPT):
        logger.error(f"Training script not found: {TRAIN_SCRIPT}")
        return
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    for dataset, sim_mat, horizons in DATASET_CONFIGS:
        for horizon in horizons:
            for model in MODELS:
                for ablation in ABLATIONS:
                    # Skip if metrics already exist
                    if find_metrics_file(dataset, DEFAULT_WINDOW, horizon, ablation, model):
                        logger.info(f"Skipping existing: {dataset} h={horizon} ablation={ablation} model={model}")
                        continue
                    
                    cmd = build_train_command(dataset, sim_mat, DEFAULT_WINDOW, horizon, ablation, model)
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
            generate_ablation_summary(dataset, DEFAULT_WINDOW, horizon, 'msagat')
    
    logger.info("All experiments complete")

if __name__ == '__main__':
    main()

