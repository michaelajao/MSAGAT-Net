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

# Datasets and horizons configuration
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
DEVICE_FLAGS = ['--cuda', '--gpu', '0']

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


def find_metrics_file(dataset: str, window: int, horizon: int, ablation: str) -> Optional[str]:
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
    summary = compute_ablation_summary(dataset, window, horizon)
    if summary is None:
        return None
    txt = []
    txt.append(f"Ablation Report for {dataset} (horizon={horizon})")
    txt.append("="*60)
    txt.append(summary.to_string())
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


def build_cmd(dataset: str, sim_mat: str, window: int, horizon: int, ablation: str) -> List[str]:
    return [
        sys.executable, TRAIN_SCRIPT,
        '--dataset', dataset,
        '--sim_mat', sim_mat,
        '--window', str(window),
        '--horizon', str(horizon),
        '--ablation', ablation,
        '--save_dir', SAVE_DIR,
    ] + DEVICE_FLAGS


def run_train(cmd: List[str]) -> bool:
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Trained: {' '.join(cmd)}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        return False


def run_all_experiments():
    if not validate_train_script():
        return
    ensure_dir(SAVE_DIR)
    for dataset, sim_mat, horizons in DATASET_CONFIGS:
        for h in horizons:
            for abl in ABLATIONS:
                if find_metrics_file(dataset, DEFAULT_WINDOW, h, abl):
                    logger.info(f"Skipping existing: {dataset} h={h} ab={abl}")
                    continue
                cmd = build_cmd(dataset, sim_mat, DEFAULT_WINDOW, h, abl)
                run_train(cmd)

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Run training
    run_all_experiments()
    # Generate summaries and reports
    for dataset, _, horizons in DATASET_CONFIGS:
        for h in horizons:
            generate_ablation_report(dataset, DEFAULT_WINDOW, h)
    logger.info("All experiments complete.")

