import os
import sys
import subprocess
import logging
import glob
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import central configuration
from config import (
    DATASET_CONFIGS, ABLATIONS, SEEDS, DEFAULT_WINDOW, TRAIN_CONFIG,
    get_dataset_list, ABLATION_DESCRIPTIONS
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
METRICS_DIR = os.path.join(BASE_DIR, 'report', 'results')
SAVE_DIR = os.path.join(BASE_DIR, 'save_all')
LOG_DIR = os.path.join(BASE_DIR, 'logs')
TRAIN_SCRIPT = os.path.join(BASE_DIR, 'src', 'train.py')

# Device configuration (CPU for small datasets - faster than GPU overhead)
DEVICE_FLAGS = []  # CPU mode - remove ['--cuda', '--gpu', '0'] for GPU

# Get dataset list for iteration (converts dict to list of tuples)
DATASET_LIST = get_dataset_list()

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(level=logging.INFO) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"run_experiments_{ts}.log")
    logging.basicConfig(
        level=level,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger('run_experiments')
    logger.info(f"Logging to {log_file}")
    return logger

logger = setup_logging()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# =============================================================================
# METRICS LOADING & SUMMARY
# =============================================================================

def normalize(name: str) -> str:
    return name.strip().lower().replace('-', '').replace('_', '')


def get_metric(df: pd.DataFrame, key: str) -> float:
    search = normalize(key)
    for col in df.columns:
        if normalize(col) == search:
            return df[col].iloc[0]
    return float('nan')


def find_metrics_file(dataset: str, window: int, horizon: int, ablation: str, seed: int = None) -> Optional[str]:
    """Find metrics file, optionally filtering by seed.
    
    Supports both old and new naming conventions:
    - Old: report/results/final_metrics_MSTAGAT-Net.{dataset}.w-{window}.h-{horizon}.{ablation}.seed-{seed}.csv
    - New: report/results/{dataset}/w-{window}.h-{horizon}.{ablation}.seed-{seed}.csv
    """
    # New pattern (organized by dataset folder)
    dataset_dir = os.path.join(METRICS_DIR, dataset)
    if seed is not None:
        new_pattern = os.path.join(dataset_dir, f"w-{window}.h-{horizon}.{ablation}.seed-{seed}.csv")
        if os.path.exists(new_pattern):
            return new_pattern
    else:
        new_pattern = os.path.join(dataset_dir, f"w-{window}.h-{horizon}.{ablation}.*.csv")
        matches = glob.glob(new_pattern)
        if matches:
            return matches[0]
    
    # Old pattern (fallback)
    if seed is not None:
        patterns = [
            f"final_metrics_*{dataset}*w-{window}*h-{horizon}*{ablation}*seed-{seed}.csv",
            f"final_metrics_*{dataset}*window-{window}*horizon-{horizon}*{ablation}*seed-{seed}.csv",
        ]
    else:
        patterns = [
            f"final_metrics_*{dataset}*w-{window}*h-{horizon}*{ablation}.csv",
            f"final_metrics_*{dataset}*window-{window}*horizon-{horizon}*{ablation}.csv",
        ]
    for pat in patterns:
        matches = glob.glob(os.path.join(METRICS_DIR, pat))
        if matches:
            return matches[0]
    return None


def load_metrics(dataset: str, window: int, horizon: int, ablation: str) -> Optional[pd.DataFrame]:
    path = find_metrics_file(dataset, window, horizon, ablation)
    if not path or not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        logger.error(f"Failed to read metrics CSV {path}: {e}")
        return None


def compute_ablation_summary(dataset: str, window: int, horizon: int) -> Optional[pd.DataFrame]:
    records = {}
    base = load_metrics(dataset, window, horizon, 'none')
    if base is None:
        logger.warning(f"No baseline metrics for {dataset} h={horizon}")
        return None
    base_vals = {m: get_metric(base, m) for m in ['mae','rmse','pcc','r2']}
    for abl in ABLATIONS:
        df = load_metrics(dataset, window, horizon, abl)
        if df is None:
            continue
        row = {m.upper(): get_metric(df, m) for m in ['mae','rmse','pcc','r2']}
        if abl != 'none':
            for m in ['mae','rmse','pcc','r2']:
                bv = base_vals[m]
                val = get_metric(df, m)
                if bv and not np.isnan(val):
                    row[f"{m.upper()}_CHANGE"] = 100*(val - bv)/bv
        records[abl] = row
    if not records:
        return None
    summary = pd.DataFrame.from_dict(records, orient='index')
    ensure_dir(METRICS_DIR)
    out = os.path.join(METRICS_DIR, f"ablation_summary_{dataset}.w-{window}.h-{horizon}.csv")
    try:
        # backup old
        if os.path.exists(out): shutil.copy2(out, out + '.bak')
        summary.to_csv(out)
        logger.info(f"Saved summary CSV: {out}")
    except Exception as e:
        logger.error(f"Failed to save summary CSV: {e}")
    return summary

# =============================================================================
# REPORT GENERATION
# =============================================================================

def generate_ablation_report(dataset: str, window: int, horizon: int) -> Optional[str]:
    """Generate comprehensive ablation report with metadata."""
    summary = compute_ablation_summary(dataset, window, horizon)
    if summary is None:
        return None
    
    txt = []
    txt.append("="*70)
    txt.append(f"MSAGAT-Net Ablation Study Report")
    txt.append("="*70)
    txt.append(f"Dataset:           {dataset}")
    txt.append(f"Window Size:       {window}")
    txt.append(f"Prediction Horizon: {horizon}")
    txt.append(f"Seeds Used:        {SEEDS}")
    txt.append(f"Ablation Variants: {ABLATIONS}")
    txt.append(f"Generated:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    txt.append("="*70)
    txt.append("")
    txt.append("PERFORMANCE SUMMARY (Averaged across seeds)")
    txt.append("-"*70)
    txt.append(summary.to_string())
    txt.append("")
    txt.append("COMPONENT DESCRIPTIONS:")
    txt.append("-"*70)
    txt.append("  none     = Full model with all components")
    txt.append("  no_agam  = Without Adaptive Graph Attention Module (LR-AGAM)")
    txt.append("  no_mtfm  = Without Multi-Scale Spatial Feature Module (MSSFM)")
    txt.append("  no_pprm  = Without Progressive Prediction Refinement Module (PPRM)")
    txt.append("")
    txt.append("METRICS EXPLANATION:")
    txt.append("-"*70)
    txt.append("  *_CHANGE = Percentage change when component is removed")
    txt.append("  Positive RMSE/MAE change = Performance degradation")
    txt.append("  Negative PCC/R2 change   = Performance degradation")
    txt.append("="*70)
    
    out = os.path.join(METRICS_DIR, f"ablation_report_{dataset}.w-{window}.h-{horizon}.txt")
    try:
        with open(out, 'w') as f:
            f.write("\n".join(txt))
        logger.info(f"Saved report TXT: {out}")
        return out
    except Exception as e:
        logger.error(f"Failed to save report: {e}")
        return None

# =============================================================================
# TRAINING EXECUTION
# =============================================================================

def validate_train_script() -> bool:
    if not os.path.exists(TRAIN_SCRIPT):
        logger.error(f"Training script missing: {TRAIN_SCRIPT}")
        return False
    return True


def build_cmd(dataset: str, sim_mat: str, window: int, horizon: int, ablation: str, seed: int) -> List[str]:
    """Build training command with optimal settings for each dataset."""
    # Get dataset-specific configuration
    dataset_cfg = DATASET_CONFIGS.get(dataset, {})
    use_adj_prior = dataset_cfg.get('use_adj_prior', False)
    use_graph_bias = dataset_cfg.get('use_graph_bias', True)
    
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--dataset', dataset,
        '--sim_mat', sim_mat,
        '--window', str(window),
        '--horizon', str(horizon),
        '--ablation', ablation,
        '--seed', str(seed),
        '--save_dir', SAVE_DIR,
    ]
    
    # Add graph structure options based on dataset config
    if use_adj_prior:
        cmd.append('--use_adj_prior')
    if not use_graph_bias:
        cmd.append('--no_graph_bias')
    
    return cmd + DEVICE_FLAGS


def run_train(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Trained: {' '.join(cmd)}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        return False


def run_all_experiments():
    """Run all experiments across datasets, horizons, ablations, and seeds."""
    if not validate_train_script():
        return
    ensure_dir(SAVE_DIR)
    
    total_experiments = sum(len(cfg['horizons']) for cfg in DATASET_CONFIGS.values()) * len(ABLATIONS) * len(SEEDS)
    completed = 0
    
    logger.info(f"Total experiments to run: {total_experiments}")
    logger.info(f"Datasets: {len(DATASET_CONFIGS)}, Ablations: {len(ABLATIONS)}, Seeds: {len(SEEDS)}")
    
    for dataset, sim_mat, horizons in DATASET_LIST:
        for h in horizons:
            for abl in ABLATIONS:
                for seed in SEEDS:
                    # Check if this specific seed's results exist
                    if find_metrics_file(dataset, DEFAULT_WINDOW, h, abl, seed):
                        logger.info(f"Skipping existing: {dataset} h={h} abl={abl} seed={seed}")
                        completed += 1
                        continue
                    
                    logger.info(f"Progress: {completed}/{total_experiments} - Running: {dataset} h={h} abl={abl} seed={seed}")
                    cmd = build_cmd(dataset, sim_mat, DEFAULT_WINDOW, h, abl, seed)
                    success = run_train(cmd)
                    if success:
                        completed += 1
                    logger.info(f"Completed: {completed}/{total_experiments}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Run training
    run_all_experiments()
    
    # Generate summaries and reports
    for dataset, _, horizons in DATASET_LIST:
        for h in horizons:
            generate_ablation_report(dataset, DEFAULT_WINDOW, h)
    
    logger.info("All experiments complete.")
    
    # Consolidate metrics
    logger.info("Consolidating metrics...")
    try:
        import subprocess
        consolidate_script = os.path.join(SCRIPT_DIR, 'consolidate_metrics.py')
        subprocess.run([sys.executable, consolidate_script], check=True)
        logger.info("Metrics consolidated successfully")
    except Exception as e:
        logger.error(f"Failed to consolidate metrics: {e}")
