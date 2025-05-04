"""
Comprehensive ablation pipeline for MSTAGAT-Net:
1) Trains the full model and ablated variants across multiple datasets and horizons
2) Evaluates performance and quantifies the importance of each model component
3) Generates visualizations comparing model variants
4) Creates detailed reports highlighting component contributions

This script orchestrates the complete evaluation pipeline, automatically 
detecting issues like non-functioning ablation parameters.
"""

import os
import sys
import subprocess
import logging
import argparse
import glob
import time
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# =============================================================================
#  CONFIGURATION SETUP
# =============================================================================

# Base paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)  # Parent directory of this script

# Directory structure
METRICS_DIR = os.path.join(BASE_DIR, 'report', 'results')
SAVE_DIR = os.path.join(BASE_DIR, 'save_all')  # Checkpoint folder
OUT_DIR = os.path.join(BASE_DIR, 'report', 'paper_figures')  # Visualization outputs
LOG_DIR = os.path.join(BASE_DIR, 'logs')  # Logging directory

# Training script
TRAIN_SCRIPT = os.path.join(BASE_DIR, 'src', 'train.py')

# Research configurations
DATASET_CONFIGS = [
    ('japan', 'japan-adj', [3, 5, 10, 15]),
    ('region785', 'region-adj', [3, 5, 10, 15]),
    ('state360', 'state-adj-49', [3, 5, 10, 15]),
    ('australia-covid', 'australia-adj', [3, 7, 14]),
    ('spain-covid', 'spain-adj', [3, 7, 14]),
    ('nhs_timeseries', 'nhs-adj', [3, 7, 14]),
    ('ltla_timeseries', 'ltla-adj', [3, 7, 14]),
]

# Default window size (used throughout the code)
DEFAULT_WINDOW = 20

# Model variants
ABLATIONS = ['none', 'no_agam', 'no_mtfm', 'no_pprm']

# Compute devices
DEVICES = [
    {'flags': ['--cuda', '--gpu', '0'], 'name': 'gpu'},
    # {'flags': [], 'name': 'cpu'}  # Uncomment to enable CPU training
]

# Default datasets and horizons for visualizations
DATASETS_FOR_PLOTS = ['japan', 'region785', 'nhs_timeseries', 'ltla_timeseries']
HORIZONS_FOR_PLOTS = [3, 5, 10, 15]

# Display names for metrics and components
METRICS_NAMES = {
    'RMSE': 'Root Mean Square Error',
    'MAE': 'Mean Absolute Error',
    'PCC': 'Pearson Correlation Coefficient',
    'R2': 'R-squared'
}

ABL_NAMES = {
    'none': 'Full Model',
    'no_agam': 'No LR‑AGAM',
    'no_mtfm': 'No MTFM',
    'no_pprm': 'No PPRM'
}

# Component full names and colors
COMP_FULL = {
    'agam': 'Low‑Rank Adaptive Graph\nAttention Module',
    'mtfm': 'Multi‑scale Temporal\nFusion Module',
    'pprm': 'Progressive Multi‑step\nPrediction Refinement Module'
}

COMP_COLORS = {
    'Low‑Rank Adaptive Graph\nAttention Module': '#1f77b4',  # Blue
    'Multi‑scale Temporal\nFusion Module': '#ff7f0e',  # Orange
    'Progressive Multi‑step\nPrediction Refinement Module': '#2ca02c'  # Green
}

# =============================================================================
#  LOGGING SETUP
# =============================================================================

def setup_logging(log_level=logging.INFO):
    """Configure logging with timestamped file output and console output."""
    os.makedirs(LOG_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"ablation_pipeline_{timestamp}.log")
    
    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s'
    ))
    
    # Configure console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s'
    ))
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create logger for this module
    logger = logging.getLogger('ablation_pipeline')
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

# Initialize logger
logger = setup_logging()

# =============================================================================
#  UTILITY FUNCTIONS
# =============================================================================

def normalize_metric_name(name: str) -> str:
    """Normalize metric names to a standard format for consistent handling."""
    if name is None:
        return None
        
    name = str(name).strip().lower()
    
    # Common variations of metric names
    name_map = {
        'root mean square error': 'rmse',
        'mean absolute error': 'mae', 
        'pearson correlation coefficient': 'pcc',
        'pearson correlation': 'pcc',
        'r squared': 'r2',
        'r-squared': 'r2',
        'r^2': 'r2',
        'coefficient of determination': 'r2'
    }
    
    return name_map.get(name, name)

def get_available_datasets() -> List[str]:
    """Get list of available datasets from the configuration."""
    return [config[0] for config in DATASET_CONFIGS]

def get_horizons_for_dataset(dataset: str) -> List[int]:
    """Get available horizons for the specified dataset."""
    for ds, _, horizons in DATASET_CONFIGS:
        if ds == dataset:
            return horizons
    return []

def get_similarity_matrix(dataset: str) -> Optional[str]:
    """Get the similarity matrix name for the specified dataset."""
    for ds, sim_mat, _ in DATASET_CONFIGS:
        if ds == dataset:
            return sim_mat
    return None

def ensure_dir_exists(directory: str) -> None:
    """Ensure the specified directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            raise

def backup_file(file_path: str) -> Optional[str]:
    """Create a backup of a file before modifying it."""
    if not os.path.exists(file_path):
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{file_path}.{timestamp}.bak"
    
    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
    except Exception as e:
        logger.warning(f"Failed to create backup of {file_path}: {str(e)}")
        return None

def get_file_timestamp(file_path: str) -> Optional[float]:
    """Get the last modified timestamp of a file."""
    if os.path.exists(file_path):
        return os.path.getmtime(file_path)
    return None

# =============================================================================
#  METRICS LOADING AND PROCESSING
# =============================================================================

def get_metric_value(df: pd.DataFrame, col_name: str) -> float:
    """
    Extract a metric value from a DataFrame with case-insensitive column matching.
    
    Args:
        df: DataFrame containing metrics
        col_name: Name of the metric to extract
        
    Returns:
        Metric value or NaN if not found
    """
    if df is None or len(df) == 0:
        return np.nan
    
    # Normalize the requested column name
    normalized_col = normalize_metric_name(col_name)
    
    # Create case-insensitive mapping of column names
    col_map = {normalize_metric_name(col): col for col in df.columns}
    
    # Try to find the column
    if normalized_col in col_map:
        return df[col_map[normalized_col]].iloc[0]
    
    # Try common variations
    variants = [
        col_name,                    # Original
        col_name.upper(),            # Uppercase
        col_name.lower(),            # Lowercase
        col_name.title(),            # Title case
        col_name.replace('_', ' ')   # Replace underscores with spaces
    ]
    
    for variant in variants:
        if variant in df.columns:
            return df[variant].iloc[0]
    
    # Log the missing column and return NaN
    logger.debug(f"Column '{col_name}' not found in DataFrame with columns: {list(df.columns)}")
    return np.nan

def find_metrics_file(dataset: str, window: int, horizon: int, ablation: str = 'none') -> Optional[str]:
    """
    Find the metrics file matching the given parameters using flexible pattern matching.
    
    Args:
        dataset: Dataset name
        window: Window size
        horizon: Forecast horizon
        ablation: Ablation type
        
    Returns:
        Path to the metrics file or None if not found
    """
    # Try multiple patterns to be flexible with file naming conventions
    patterns = [
        f"final_metrics_*{dataset}*w-{window}*h-{horizon}*{ablation}.csv",
        f"final_metrics_*{dataset}*w{window}*h{horizon}*{ablation}.csv",
        f"final_metrics_*{dataset}*window-{window}*horizon-{horizon}*{ablation}.csv",
        f"final_metrics_*{dataset}*window{window}*horizon{horizon}*{ablation}.csv",
        f"final_metrics_*{dataset}*w={window}*h={horizon}*{ablation}.csv",
        # Add more patterns if needed
    ]
    
    for pattern in patterns:
        paths = glob.glob(os.path.join(METRICS_DIR, pattern))
        if paths:
            path = paths[0]  # Take the first matching file
            logger.debug(f"Found metrics file: {os.path.basename(path)}")
            return path
    
    # Specific names for better error messages
    example1 = f"final_metrics_{dataset}.w-{window}.h-{horizon}.{ablation}.csv"
    example2 = f"final_metrics_MSTAGAT-Net.{dataset}.w-{window}.h-{horizon}.{ablation}.csv"
    
    logger.warning(f"Missing metrics file for {dataset}, w={window}, h={horizon}, ablation={ablation}")
    logger.warning(f"Tried patterns like {example1} and {example2}")
    return None

def load_metrics(dataset: str, window: int, horizon: int, ablation: str = 'none') -> Optional[pd.DataFrame]:
    """
    Load metrics from the appropriate CSV file.
    
    Args:
        dataset: Dataset name
        window: Window size
        horizon: Forecast horizon
        ablation: Ablation type
        
    Returns:
        DataFrame with metrics or None if file not found
    """
    path = find_metrics_file(dataset, window, horizon, ablation)
    
    if path:
        try:
            df = pd.read_csv(path)
            if len(df) > 1:
                logger.info(f"Metrics file has {len(df)} rows, using first row (index 0)")
            return df
        except Exception as e:
            logger.error(f"Error loading metrics file {path}: {str(e)}")
            return None
    else:
        return None

def find_summary_file(dataset: str, window: int, horizon: int) -> Optional[str]:
    """Find the ablation summary file for the given parameters."""
    # Try both naming patterns
    patterns = [
        f"ablation_summary_{dataset}.w-{window}.h-{horizon}.csv",
        f"ablation_summary_MSAGAT-Net.{dataset}.w-{window}.h-{horizon}.csv",
        f"ablation_summary_{dataset}.w{window}.h{horizon}.csv",
        f"ablation_summary_MSAGAT-Net.{dataset}.w{window}.h{horizon}.csv",
    ]
    
    for pattern in patterns:
        path = os.path.join(METRICS_DIR, pattern)
        if os.path.exists(path):
            return path
    
    return None

def load_summary(dataset: str, window: int, horizon: int) -> Optional[pd.DataFrame]:
    """
    Load an existing ablation summary or compute one on the fly.
    
    Args:
        dataset: Dataset name
        window: Window size
        horizon: Forecast horizon
        
    Returns:
        DataFrame with ablation summary or None if not available
    """
    path = find_summary_file(dataset, window, horizon)
    
    if path:
        try:
            return pd.read_csv(path, index_col=0)
        except Exception as e:
            logger.error(f"Error loading summary file {path}: {str(e)}")
            
    # Try to compute the summary on the fly
    logger.info(f"Computing ablation summary for {dataset}, w={window}, h={horizon}")
    summary = compute_ablation_summary(dataset, window, horizon)
    
    if summary is not None:
        logger.info(f"Successfully generated ablation summary")
        return summary
        
    logger.warning(f"Failed to compute ablation summary")
    return None

def compute_ablation_summary(dataset: str, window: int, horizon: int) -> Optional[pd.DataFrame]:
    """
    Compute summary of ablation results by comparing metrics across variants.
    
    Args:
        dataset: Dataset name
        window: Window size
        horizon: Forecast horizon
        
    Returns:
        DataFrame with ablation summary or None if insufficient data
    """
    metrics = {}
    
    # Load metrics for each ablation variant
    for ablation in ABLATIONS:
        df = load_metrics(dataset, window, horizon, ablation)
        if df is not None:
            metrics[ablation] = df
    
    if not metrics or 'none' not in metrics:
        logger.warning(f"Insufficient data to compute ablation summary for {dataset}, w={window}, h={horizon}")
        return None
    
    # Check for duplicated data (identical metrics files)
    identical_files = []
    for ablation in ABLATIONS:
        if ablation == 'none' or ablation not in metrics:
            continue
            
        # Compare key metrics to baseline
        all_identical = True
        for col in ['mae', 'rmse', 'pcc', 'r2']:
            abl_val = get_metric_value(metrics[ablation], col)
            base_val = get_metric_value(metrics['none'], col)
            if not np.isnan(abl_val) and not np.isnan(base_val) and abs(abl_val - base_val) > 1e-10:
                all_identical = False
                break
                
        if all_identical:
            identical_files.append(ablation)
            logger.warning(f"⚠️ The metrics for '{ablation}' are identical to 'none' for {dataset}, h={horizon}!")
            logger.warning(f"This suggests the ablation might not be working correctly.")
            
    if identical_files:
        logger.warning(f"Identical metrics were found for components: {', '.join(identical_files)}")
        logger.warning(f"This is likely due to a mismatch between CLI argument names and code implementation:")
        logger.warning(f"  - Check if 'no_mtfm' in CLI corresponds to 'no_dmtm' in ablation.py")
        logger.warning(f"  - Check if 'no_pprm' in CLI corresponds to 'no_ppm' in ablation.py") 
        logger.warning(f"Recommendation: Verify parameter names in ablation.py match those in train.py")
            
    # Organize performance metrics
    metric_cols = ['mae', 'rmse', 'pcc', 'r2']
    baseline_metrics = {col: get_metric_value(metrics['none'], col) for col in metric_cols}
    
    # Build summary DataFrame
    summary_data = {}
    for ablation, df in metrics.items():
        row_data = {}
        
        # Add raw metrics
        for col in metric_cols:
            row_data[col.upper()] = get_metric_value(df, col)
        
        # Compute percentage changes for non-baseline models
        if ablation != 'none':
            for col in metric_cols:
                value = get_metric_value(df, col)
                baseline = baseline_metrics[col]
                if not np.isnan(value) and not np.isnan(baseline) and baseline != 0:
                    pct_change = 100 * (value - baseline) / baseline
                    row_data[f"{col.upper()}_change"] = pct_change
                    
                    # Log very small changes that might be numerical errors
                    if abs(pct_change) < 1e-10:
                        logger.debug(f"Very small change detected for {dataset}, {ablation}, {col}: {pct_change}")
                    logger.info(f"{dataset}, h={horizon}, {ablation}, {col}: {value:.6f} vs baseline {baseline:.6f} = {pct_change:.2f}%")
        
        summary_data[ablation] = row_data
    
    # Create DataFrame from the summary data
    summary_df = pd.DataFrame(summary_data).T
    
    # Save summary to CSV
    # Use the same format as metrics files
    first_ablation = list(metrics.keys())[0]
    model_col_exists = 'model' in metrics[first_ablation].columns
    
    if model_col_exists and "MSAGAT-Net" in metrics[first_ablation]['model'].iloc[0]:
        # Use format with model name
        filename = f"ablation_summary_MSAGAT-Net.{dataset}.w-{window}.h-{horizon}.csv"
    else:
        # Use standard format
        filename = f"ablation_summary_{dataset}.w-{window}.h-{horizon}.csv"
    
    path = os.path.join(METRICS_DIR, filename)
    
    # Create backup of existing file
    backup_file(path)
    
    # Save the new summary
    try:
        summary_df.to_csv(path)
        logger.info(f"Saved ablation summary to {path}")
    except Exception as e:
        logger.error(f"Error saving ablation summary to {path}: {str(e)}")
    
    return summary_df

def check_ablation_issues(dataset: str, window: int, horizon: int) -> List[str]:
    """
    Comprehensive check for ablation study issues.
    
    Args:
        dataset: Dataset name
        window: Window size
        horizon: Forecast horizon
        
    Returns:
        List of issue descriptions
    """
    issues = []
    
    # 1. Check if all ablation metrics exist
    missing_ablations = []
    for ablation in ABLATIONS:
        df = load_metrics(dataset, window, horizon, ablation)
        if df is None:
            missing_ablations.append(ablation)
    
    if missing_ablations:
        issues.append(f"Missing metrics for ablations: {', '.join(missing_ablations)}")
    
    # 2. Check for identical metrics between ablations and baseline
    identical_ablations = []
    baseline_df = load_metrics(dataset, window, horizon, 'none')
    
    if baseline_df is not None:
        for ablation in [a for a in ABLATIONS if a != 'none']:
            df = load_metrics(dataset, window, horizon, ablation)
            if df is not None:
                identical = True
                for metric in ['rmse', 'mae', 'pcc', 'r2']:
                    abl_val = get_metric_value(df, metric)
                    base_val = get_metric_value(baseline_df, metric)
                    if not np.isnan(abl_val) and not np.isnan(base_val) and abs(abl_val - base_val) > 1e-10:
                        identical = False
                        break
                
                if identical:
                    identical_ablations.append(ablation)
    
    if identical_ablations:
        issues.append(f"Ablations with identical metrics to baseline: {', '.join(identical_ablations)}")
        issues.append("This suggests the ablation is not working correctly - check parameter names in train.py and ablation.py")
    
    # 3. Check for inconsistent improvement across metrics
    if baseline_df is not None:
        for ablation in [a for a in ABLATIONS if a != 'none' and a not in identical_ablations]:
            df = load_metrics(dataset, window, horizon, ablation)
            if df is None:
                continue
                
            # For each ablation, removing a component should generally worsen performance
            # Check if metrics move in contradictory directions
            contradictions = []
            expected_worse_metrics = ['rmse', 'mae']  # Should increase when component removed
            expected_better_metrics = ['pcc', 'r2']   # Should decrease when component removed
            
            # Check RMSE/MAE - should increase when component is removed
            for metric in expected_worse_metrics:
                abl_val = get_metric_value(df, metric)
                base_val = get_metric_value(baseline_df, metric)
                
                if not np.isnan(abl_val) and not np.isnan(base_val) and abl_val < base_val:
                    contradictions.append(f"{metric} improved when component was removed")
            
            # Check PCC/R2 - should decrease when component is removed
            for metric in expected_better_metrics:
                abl_val = get_metric_value(df, metric)
                base_val = get_metric_value(baseline_df, metric)
                
                if not np.isnan(abl_val) and not np.isnan(base_val) and abl_val > base_val:
                    contradictions.append(f"{metric} improved when component was removed")
            
            if contradictions:
                issues.append(f"Unexpected metric changes for {ablation}: {', '.join(contradictions)}")
                issues.append("This might indicate the component has negative impact or the ablation isn't working correctly")
    
    return issues

# =============================================================================
#  TRAINING FUNCTIONS
# =============================================================================

def validate_train_script() -> bool:
    """Validate that training script exists and is executable."""
    if not os.path.exists(TRAIN_SCRIPT):
        logger.error(f"Training script not found at: {TRAIN_SCRIPT}")
        return False
        
    if not os.access(TRAIN_SCRIPT, os.X_OK):
        # Not executable but exists
        logger.warning(f"Training script exists but may not be executable. Check permissions: {TRAIN_SCRIPT}")
        # We'll try to run it anyway
        
    return True

def build_train_command(
    dataset: str,
    horizon: int,
    window: int = DEFAULT_WINDOW,
    ablation: str = 'none',
    device_config: Dict[str, Any] = None
) -> List[str]:
    """
    Build command-line arguments for training.
    
    Args:
        dataset: Dataset name
        horizon: Forecast horizon
        window: Window size
        ablation: Ablation type
        device_config: Device configuration dictionary
        
    Returns:
        List of command arguments
    """
    # Get similarity matrix for dataset
    sim_mat = get_similarity_matrix(dataset)
    if sim_mat is None:
        logger.error(f"No similarity matrix found for dataset: {dataset}")
        return []
        
    # Use default device if none specified
    if device_config is None:
        device_config = DEVICES[0]
    
    # Build command
    cmd = [
        'python', TRAIN_SCRIPT,
        '--dataset', dataset,
        '--sim_mat', sim_mat,
        '--window', str(window),
        '--horizon', str(horizon),
        '--ablation', ablation,
        '--save_dir', SAVE_DIR
    ] + device_config['flags']
    
    return cmd

def run_train_command(cmd: List[str], timeout: int = 10800, ignore_errors: bool = False) -> bool:
    """
    Run a training command with proper error handling.
    
    Args:
        cmd: Command list to run
        timeout: Timeout in seconds (default: 3 hours)
        ignore_errors: Whether to continue on error
        
    Returns:
        True if command succeeded, False otherwise
    """
    if not cmd:
        logger.error("Empty command list provided")
        return False
        
    logger.info(f"Running training: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        logger.info(f"Command completed successfully: {' '.join(cmd)}")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout//3600} hours: {' '.join(cmd)}")
        if not ignore_errors:
            raise
        return False
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed with exit code {e.returncode}: {' '.join(cmd)}")
        logger.error(f"Output: {e.stdout}")
        logger.error(f"Error: {e.stderr}")
        if not ignore_errors:
            raise
        return False
        
    except Exception as e:
        logger.error(f"Error running command {' '.join(cmd)}: {str(e)}")
        if not ignore_errors:
            raise
        return False

def run_ablation_studies(
    datasets: List[str] = None,
    horizons: List[int] = None,
    ablations: List[str] = None,
    window: int = DEFAULT_WINDOW,
    devices: List[Dict[str, Any]] = None,
    ignore_errors: bool = False,
    max_concurrent: int = 1
) -> Dict[str, Any]:
    """
    Run ablation studies for specified datasets, horizons, and ablations.
    
    Args:
        datasets: List of datasets to use (default: all)
        horizons: List of horizons to use (default: all for each dataset)
        ablations: List of ablations to run (default: all)
        window: Window size to use
        devices: Device configurations to use (default: first device)
        ignore_errors: Whether to continue on error
        max_concurrent: Maximum number of concurrent runs (default: 1)
        
    Returns:
        Dictionary with results summary
    """
    # Validate training script
    if not validate_train_script():
        logger.error("Training script validation failed. Aborting ablation studies.")
        return {"status": "failed", "reason": "invalid_script"}
    
    # Use defaults if parameters not specified
    datasets = datasets or get_available_datasets()
    ablations = ablations or ABLATIONS
    devices = devices or [DEVICES[0]]
    
    # Create results tracking
    results = {
        "total": 0,
        "completed": 0,
        "failed": 0,
        "datasets": {},
        "status": "running"
    }
    
    # Create save directory if it doesn't exist
    ensure_dir_exists(SAVE_DIR)
    
    # Run training commands
    for dataset in datasets:
        dataset_horizons = horizons or get_horizons_for_dataset(dataset)
        
        if not dataset_horizons:
            logger.warning(f"No horizons found for dataset: {dataset}")
            continue
        
        results["datasets"][dataset] = {
            "horizons": {},
            "completed": 0,
            "failed": 0
        }
        
        for h in dataset_horizons:
            results["datasets"][dataset]["horizons"][h] = {
                "ablations": {},
                "completed": 0,
                "failed": 0
            }
            
            for ab in ablations:
                # Choose device - for now just use the first one
                device = devices[0]
                
                # Build and run training command
                cmd = build_train_command(
                    dataset=dataset,
                    horizon=h,
                    window=window,
                    ablation=ab,
                    device_config=device
                )
                
                results["total"] += 1
                
                # Check for existing metrics file to skip if already done
                if find_metrics_file(dataset, window, h, ab) is not None:
                    logger.info(f"Metrics file already exists for {dataset}, h={h}, ablation={ab}. Skipping.")
                    results["completed"] += 1
                    results["datasets"][dataset]["completed"] += 1
                    results["datasets"][dataset]["horizons"][h]["completed"] += 1
                    results["datasets"][dataset]["horizons"][h]["ablations"][ab] = "skipped"
                    continue
                
                # Run the command
                success = run_train_command(cmd, ignore_errors=ignore_errors)
                
                if success:
                    results["completed"] += 1
                    results["datasets"][dataset]["completed"] += 1
                    results["datasets"][dataset]["horizons"][h]["completed"] += 1
                    results["datasets"][dataset]["horizons"][h]["ablations"][ab] = "completed"
                else:
                    results["failed"] += 1
                    results["datasets"][dataset]["failed"] += 1
                    results["datasets"][dataset]["horizons"][h]["failed"] += 1
                    results["datasets"][dataset]["horizons"][h]["ablations"][ab] = "failed"
    
    # Update final status
    if results["failed"] == 0:
        results["status"] = "completed"
    else:
        results["status"] = "completed_with_errors"
    
    return results

# =============================================================================
#  VISUALIZATION FUNCTIONS
# =============================================================================

def setup_visualization_style():
    """Configure matplotlib and seaborn for consistent styling."""
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Configure matplotlib
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16
    
    # Use high-quality rendering
    plt.rcParams['figure.autolayout'] = True
    
    logger.debug("Visualization style configured")

def save_figure(fig, filename, output_dir=OUT_DIR, formats=None):
    """
    Save a figure to disk in multiple formats.
    
    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        output_dir: Output directory
        formats: List of formats to save (default: ['png', 'pdf'])
    """
    formats = formats or ['png', 'pdf']
    
    ensure_dir_exists(output_dir)
    
    for fmt in formats:
        path = os.path.join(output_dir, f"{filename}.{fmt}")
        try:
            fig.savefig(path, bbox_inches='tight', dpi=300)
            logger.debug(f"Saved figure: {path}")
        except Exception as e:
            logger.error(f"Error saving figure {path}: {str(e)}")

def generate_performance_table(
    datasets: List[str],
    window: int,
    horizons: List[int],
    output_dir: str = OUT_DIR
) -> Optional[pd.DataFrame]:
    """
    Generate performance summary table across datasets and horizons.
    
    Args:
        datasets: List of datasets to include
        window: Window size
        horizons: List of horizons to include
        output_dir: Output directory
        
    Returns:
        DataFrame with performance metrics or None if no data
    """
    rows = []
    
    for ds in datasets:
        for h in horizons:
            df = load_metrics(ds, window, h)
            if df is None:
                logger.warning(f"No metrics found for {ds}, h={h}")
                continue
                
            rows.append({
                'Dataset': ds,
                'Horizon': h,
                'RMSE': get_metric_value(df, 'rmse'),
                'MAE': get_metric_value(df, 'mae'),
                'PCC': get_metric_value(df, 'pcc'),
                'R2': get_metric_value(df, 'r2'),
            })
    
    if not rows:
        logger.warning("No performance data available.")
        return None
        
    perf_df = pd.DataFrame(rows)
    
    # Save to CSV
    try:
        csv_path = os.path.join(output_dir, f"performance_table_w{window}.csv")
        perf_df.to_csv(csv_path, index=False, float_format='%.4f')
        logger.info(f"Saved performance table to {csv_path}")
    except Exception as e:
        logger.error(f"Error saving performance table: {str(e)}")
    
    return perf_df

def generate_cross_horizon_performance(
    datasets: List[str],
    window: int,
    horizons: List[int],
    output_dir: str = OUT_DIR
) -> None:
    """
    Generate performance vs. horizon plots for key metrics.
    
    Args:
        datasets: List of datasets to include
        window: Window size
        horizons: List of horizons to include
        output_dir: Output directory
    """
    # Initialize plots
    setup_visualization_style()
    
    for metric in ['rmse', 'pcc']:
        plt.figure(figsize=(10, 6))
        
        # Plot data for each dataset
        for ds in datasets:
            # Collect values for this dataset across horizons
            values = []
            valid_horizons = []
            
            for h in horizons:
                df = load_metrics(ds, window, h)
                if df is not None:
                    val = get_metric_value(df, metric)
                    if not np.isnan(val):
                        values.append(val)
                        valid_horizons.append(h)
            
            # Only plot if we have data
            if values:
                plt.plot(valid_horizons, values, 'o-', label=ds, linewidth=2, markersize=8)
        
        # Set plot properties
        plt.title(f"{METRICS_NAMES[metric.upper()]} vs Horizon")
        plt.xlabel("Horizon (days)")
        plt.ylabel(METRICS_NAMES[metric.upper()])
        plt.xticks(horizons)
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend(loc='best')
        
        # Save figure
        filename = f"{metric}_vs_horizon_w{window}"
        save_figure(plt.gcf(), filename, output_dir)
        plt.close()
        
        logger.info(f"Saved cross-horizon {metric} plot")

def generate_component_importance_comparison(
    datasets: List[str],
    window: int,
    horizons: List[int],
    output_dir: str = OUT_DIR
) -> None:
    """
    Generate bar chart comparing component importance across horizons.
    
    Args:
        datasets: List of datasets to include
        window: Window size
        horizons: List of horizons to include
        output_dir: Output directory
    """
    # Collect data
    data = []
    
    for ds in datasets:
        for h in horizons:
            summary = load_summary(ds, window, h)
            if summary is None:
                continue
                
            # Extract component importance from each ablation
            for ab in ['no_agam', 'no_mtfm', 'no_pprm']:
                if ab in summary.index and 'RMSE_change' in summary.columns:
                    # Get component name
                    comp = ab.replace('no_', '')
                    
                    # Add data point
                    data.append({
                        'Horizon': h,
                        'Component': COMP_FULL[comp],
                        'Importance': abs(summary.loc[ab, 'RMSE_change']),
                        'Dataset': ds
                    })
    
    if not data:
        logger.warning("No component-importance data available.")
        return
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create the plot
    setup_visualization_style()
    plt.figure(figsize=(12, 8))
    
    # Create grouped barplot
    ax = sns.barplot(
        x='Horizon',
        y='Importance',
        hue='Component',
        data=df,
        palette=list(COMP_COLORS.values())
    )
    
    # Add values on top of bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)
    
    # Set plot properties
    plt.title("Component Importance Across Horizons", fontsize=16)
    plt.xlabel("Horizon (days)", fontsize=14)
    plt.ylabel("% RMSE Change When Removed", fontsize=14)
    plt.legend(title="Model Component", title_fontsize=12)
    
    # Save figure
    filename = f"component_importance_w{window}"
    save_figure(plt.gcf(), filename, output_dir)
    plt.close()
    
    logger.info("Saved component-importance chart")
    
    # Also create a faceted plot by dataset
    if len(datasets) > 1:
        plt.figure(figsize=(14, 8))
        g = sns.catplot(
            x='Horizon',
            y='Importance',
            hue='Component',
            col='Dataset',
            data=df,
            kind='bar',
            palette=list(COMP_COLORS.values()),
            height=4,
            aspect=1.2,
            col_wrap=min(3, len(datasets))
        )
        
        g.set_axis_labels("Horizon (days)", "% RMSE Change When Removed")
        g.set_titles("{col_name}")
        g.add_legend(title="Model Component")
        
        # Save figure
        filename = f"component_importance_by_dataset_w{window}"
        save_figure(g.fig, filename, output_dir)
        plt.close()
        
        logger.info("Saved component-importance by dataset chart")

def generate_ablation_impact_grid(
    dataset: str,
    window: int,
    horizons: List[int],
    output_dir: str = OUT_DIR
) -> None:
    """
    Generate heatmap grid showing impact of each ablation across horizons.
    
    Args:
        dataset: Dataset to visualize
        window: Window size
        horizons: List of horizons to include
        output_dir: Output directory
    """
    # Initialize plot
    setup_visualization_style()
    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 5), sharey=True)
    
    # Handle single horizon case
    if len(horizons) == 1:
        axes = [axes]
    
    # Process each horizon
    for i, h in enumerate(horizons):
        summary = load_summary(dataset, window, h)
        
        if summary is None:
            # No data for this horizon
            axes[i].text(0.5, 0.5, f"No data for h={h}", ha='center', va='center', fontsize=12)
            axes[i].axis('off')
            continue
        
        # Extract change metrics for heatmap
        heat = summary.filter(regex='change).drop('none_change', errors='ignore')
        
        # Replace row indices with readable names
        heat.index = [ABL_NAMES[idx] for idx in heat.index]
        
        # Check for components with zero impact
        zero_components = []
        for idx, row in heat.iterrows():
            if (abs(row) < 1e-10).all():
                zero_components.append(idx)
        
        # Log warning for zero-impact components
        if zero_components:
            logger.warning(f"⚠️ Components with no impact for {dataset}, h={h}: {', '.join(zero_components)}")
            logger.warning(f"This suggests the ablation is not working correctly - metrics are identical to baseline.")
        
        # Create the heatmap
        sns.heatmap(
            heat.T,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn_r',
            ax=axes[i],
            cbar=(i == len(horizons)-1)
        )
        
        # Add warning markers for zero-impact components
        for comp_name in zero_components:
            comp_idx = heat.index.get_loc(comp_name)
            for metric_idx in range(len(heat.columns)):
                axes[i].text(
                    comp_idx + 0.5,
                    metric_idx + 0.5,
                    'ZERO IMPACT!\nNeeds fixing',
                    ha='center',
                    va='center',
                    color='red',
                    fontweight='bold',
                    fontsize=8
                )
        
        # Set title for this subplot
        axes[i].set_title(f"{h}-day Horizon")
    
    # Set overall title
    plt.suptitle(f"Ablation Impact ({dataset})", fontsize=16, y=1.05)
    
    # Save figure
    filename = f"ablation_impact_{dataset}_w{window}"
    save_figure(fig, filename, output_dir)
    plt.close()
    
    logger.info(f"Saved ablation-impact grid for {dataset}")

def generate_performance_comparison_grid(
    dataset: str,
    window: int,
    horizons: List[int],
    output_dir: str = OUT_DIR
) -> None:
    """
    Generate grid of bar charts comparing performance metrics across ablations.
    
    Args:
        dataset: Dataset to visualize
        window: Window size
        horizons: List of horizons to include
        output_dir: Output directory
    """
    # Define metrics and ablations to include
    metrics = ['rmse', 'mae', 'pcc', 'r2']
    ablations = ['none', 'no_agam', 'no_mtfm', 'no_pprm']
    
    # Define color map for ablations
    colmap = {
        'none': '#2ca02c',      # Green - Full model
        'no_agam': '#d62728',   # Red
        'no_mtfm': '#ff7f0e',   # Orange
        'no_pprm': '#1f77b4'    # Blue
    }
    
    # Initialize plot
    setup_visualization_style()
    fig, axes = plt.subplots(
        len(metrics),
        len(horizons),
        figsize=(4*len(horizons), 3*len(metrics)),
        constrained_layout=True
    )
    
    # Create each subplot
    for i, metric in enumerate(metrics):
        for j, h in enumerate(horizons):
            # Get current axis
            ax = axes[i, j]
            
            # Collect data for this metric and horizon
            values, labels, colors = [], [], []
            
            for ablation in ablations:
                df = load_metrics(dataset, window, h, ablation)
                if df is not None:
                    value = get_metric_value(df, metric)
                    if not np.isnan(value):
                        values.append(value)
                        labels.append(ABL_NAMES[ablation])
                        colors.append(colmap[ablation])
            
            # Create bar chart
            if values:
                bars = ax.bar(labels, values, color=colors)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height,
                        f"{height:.3f}",
                        ha='center',
                        va='bottom',
                        fontsize=8
                    )
            
            # Set axis labels
            if j == 0:
                ax.set_ylabel(METRICS_NAMES[metric.upper()])
            if i == 0:
                ax.set_title(f"{h}-day")
            
            # Add grid
            ax.grid(alpha=0.3, linestyle='--')
            
            # Rotate x-axis labels if needed
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    
    # Set overall title
    plt.suptitle(f"Performance Comparison ({dataset})", fontsize=16)
    
    # Save figure
    filename = f"performance_comp_{dataset}_w{window}"
    save_figure(fig, filename, output_dir)
    plt.close()
    
    logger.info(f"Saved performance-comparison grid for {dataset}")

def generate_single_horizon_metrics(
    dataset: str,
    window: int,
    horizon: int,
    output_dir: str = OUT_DIR
) -> None:
    """
    Generate individual performance comparison charts for a specific horizon.
    
    Args:
        dataset: Dataset to visualize
        window: Window size
        horizon: Forecast horizon
        output_dir: Output directory
    """
    # Define metrics and ablations
    metrics = ['rmse', 'mae', 'pcc', 'r2']
    ablations = ['none', 'no_agam', 'no_mtfm', 'no_pprm']
    
    # Define color map for ablations
    colmap = {
        'none': '#2ca02c',      # Green - Full model
        'no_agam': '#d62728',   # Red
        'no_mtfm': '#ff7f0e',   # Orange
        'no_pprm': '#1f77b4'    # Blue
    }
    
    # Process each metric
    for metric in metrics:
        # Collect data
        values, labels, colors = [], [], []
        
        for ablation in ablations:
            df = load_metrics(dataset, window, horizon, ablation)
            if df is not None:
                value = get_metric_value(df, metric)
                if not np.isnan(value):
                    values.append(value)
                    labels.append(ABL_NAMES[ablation])
                    colors.append(colmap[ablation])
        
        # Skip if no data
        if not values:
            logger.warning(f"No data for {dataset}, h={horizon}, metric={metric}")
            continue
        
        # Create plot
        setup_visualization_style()
        plt.figure(figsize=(10, 6))
        
        # Create bar chart
        bars = plt.bar(labels, values, color=colors)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f"{height:.3f}",
                ha='center',
                va='bottom'
            )
        
        # Set plot properties
        plt.ylabel(METRICS_NAMES[metric.upper()])
        plt.title(f"{METRICS_NAMES[metric.upper()]} Comparison - {dataset} ({horizon}-day Horizon)")
        plt.grid(alpha=0.3, linestyle='--')
        
        # Save figure
        filename = f"{metric}_comparison_{dataset}_h{horizon}_w{window}"
        save_figure(plt.gcf(), filename, output_dir)
        plt.close()
        
        logger.info(f"Saved {metric} comparison for {dataset} h={horizon}")

def generate_component_contribution_charts(
    datasets: List[str],
    window: int,
    horizon: int,
    output_dir: str = OUT_DIR
) -> None:
    """
    Generate bar charts showing relative contribution of components for specific horizons.
    
    Args:
        datasets: List of datasets to include
        window: Window size
        horizon: Forecast horizon
        output_dir: Output directory
    """
    # Define metrics to visualize
    metrics = ['rmse', 'pcc']
    
    # Collect data for each metric
    data = {}
    
    for metric in metrics:
        metric_data = []
        
        for ds in datasets:
            summary = load_summary(ds, window, horizon)
            if summary is None:
                continue
            
            # Extract component contributions
            for ablation in ['no_agam', 'no_mtfm', 'no_pprm']:
                if ablation in summary.index and f"{metric.upper()}_change" in summary.columns:
                    # Get component name
                    comp = ablation.replace('no_', '')
                    
                    # Add data point
                    metric_data.append({
                        'Dataset': ds,
                        'Component': COMP_FULL[comp],
                        'Contribution': abs(summary.loc[ablation, f"{metric.upper()}_change"])
                    })
        
        # Skip if no data
        if not metric_data:
            logger.warning(f"No component contribution data for h={horizon}, metric={metric}")
            continue
        
        # Store data for this metric
        data[metric] = pd.DataFrame(metric_data)
    
    # Create visualizations for each metric
    for metric, df in data.items():
        if df.empty:
            continue
        
        # Create plot
        setup_visualization_style()
        plt.figure(figsize=(12, 8))
        
        # Create bar chart
        ax = sns.barplot(x='Component', y='Contribution', hue='Dataset', data=df)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', padding=3)
        
        # Set plot properties
        plt.title(f"Component Contribution to {METRICS_NAMES[metric.upper()]} - {horizon}-day Forecast")
        plt.ylabel(f"% {metric.upper()} Change When Removed")
        plt.xticks(rotation=0)
        plt.legend(title="Dataset")
        
        # Save figure
        filename = f"component_contribution_{metric}_{horizon}day_w{window}"
        save_figure(plt.gcf(), filename, output_dir)
        plt.close()
        
        logger.info(f"Saved component contribution chart for {metric}, h={horizon}")

def generate_relative_contribution_chart(
    datasets: List[str],
    window: int,
    horizon: int,
    output_dir: str = OUT_DIR
) -> None:
    """
    Generate visualization of relative contribution of each component at specific horizon.
    
    Args:
        datasets: List of datasets to include
        window: Window size
        horizon: Forecast horizon
        output_dir: Output directory
    """
    # Collect data
    all_data = []
    
    for ds in datasets:
        summary = load_summary(ds, window, horizon)
        if summary is None:
            continue
        
        # Extract component contributions
        for ablation in ['no_agam', 'no_mtfm', 'no_pprm']:
            if ablation in summary.index and 'RMSE_change' in summary.columns:
                # Get component name
                comp = ablation.replace('no_', '')
                
                # Add data point
                all_data.append({
                    'Dataset': ds,
                    'Component': COMP_FULL[comp],
                    'Color': COMP_COLORS[COMP_FULL[comp]],
                    'Contribution': abs(summary.loc[ablation, 'RMSE_change'])
                })
    
    # Skip if no data
    if not all_data:
        logger.warning(f"No relative contribution data for h={horizon}")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Create plot
    setup_visualization_style()
    plt.figure(figsize=(14, 8))
    
    # Create side-by-side plots
    # Left: Grouped bar chart
    ax1 = plt.subplot(1, 2, 1)
    sns.barplot(x='Dataset', y='Contribution', hue='Component', data=df, 
                palette=list(COMP_COLORS.values()), ax=ax1)
    plt.title(f"Component Contributions - {horizon}-day Forecast")
    plt.ylabel("% RMSE Degradation When Removed")
    plt.legend(title="Component")
    
    # Right: Pie chart of average contributions
    ax2 = plt.subplot(1, 2, 2)
    
    # Calculate average contribution by component
    avg_contribution = df.groupby('Component')['Contribution'].mean().reset_index()
    
    # Get colors for each component
    colors = [row['Color'] for _, row in avg_contribution.iterrows()]
    
    # Calculate percentages
    total = avg_contribution['Contribution'].sum()
    percentages = [100 * val / total for val in avg_contribution['Contribution']]
    
    # Create labels with component name and percentage
    labels = [f"{comp}\n({pct:.1f}%)" for comp, pct in zip(avg_contribution['Component'], percentages)]
    
    # Create pie chart
    ax2.pie(
        avg_contribution['Contribution'],
        labels=labels,
        colors=colors,
        autopct='',
        startangle=90,
        wedgeprops={'edgecolor': 'w'}
    )
    ax2.set_title(f"Average Relative Contribution - {horizon}-day")
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    filename = f"relative_contribution_{horizon}day_w{window}"
    save_figure(plt.gcf(), filename, output_dir)
    plt.close()
    
    logger.info(f"Saved relative contribution chart for h={horizon}")

def generate_ablation_report(
    dataset: str,
    window: int,
    horizon: int,
    output_dir: str = None
) -> Optional[str]:
    """
    Generate a detailed text report of the ablation study results.
    
    Args:
        dataset: Dataset to analyze
        window: Window size
        horizon: Forecast horizon
        output_dir: Output directory (default: METRICS_DIR)
        
    Returns:
        Path to the generated report or None if unsuccessful
    """
    # Default to metrics directory
    output_dir = output_dir or METRICS_DIR
    
    # Load summary data
    summary = load_summary(dataset, window, horizon)
    if summary is None:
        logger.warning(f"No summary data for {dataset}, w={window}, h={horizon}")
        return None
    
    # Check for zero impact components
    heat = summary.filter(regex='change).drop('none_change', errors='ignore')
    zero_components = []
    for idx, row in heat.iterrows():
        if (abs(row) < 1e-10).all():
            zero_components.append(ABL_NAMES[idx])
    
    # Ensure output directory exists
    ensure_dir_exists(output_dir)
    
    # Define report path
    report_path = os.path.join(output_dir, f"ablation_report_{dataset}.w-{window}.h-{horizon}.txt")
    
    # Write report
    try:
        with open(report_path, 'w') as f:
            f.write(f"MSAGAT-Net Ablation Study Report\n")
            f.write(f"==============================\n\n")
            
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Window Size: {window}\n")
            f.write(f"Forecast Horizon: {horizon}\n\n")
            
            f.write(f"Performance Metrics by Model Variant\n")
            f.write(f"-----------------------------------\n\n")
            metric_cols = [col for col in summary.columns if not col.endswith('_change')]
            f.write(f"{summary[metric_cols].to_string()}\n\n")
            
            change_cols = [col for col in summary.columns if col.endswith('_change')]
            if change_cols:
                f.write(f"Percentage Change from Full Model\n")
                f.write(f"--------------------------------\n\n")
                f.write(f"{summary[change_cols].to_string()}\n\n")
            
            f.write(f"Component Importance Analysis\n")
            f.write(f"----------------------------\n\n")
            
            # Map ablation types to component descriptions
            component_desc = {
                'no_agam': "Low‑Rank Adaptive Graph Attention Module (LR‑AGAM): Captures spatial dependencies between regions with linear complexity",
                'no_mtfm': "Multi‑scale Temporal Fusion Module (MTFM): Processes time-series patterns at different temporal resolutions",
                'no_pprm': "Progressive Multi‑step Prediction Refinement Module (PPRM): Enables region-aware multi-step forecasting with refinement"
            }
            
            # Calculate importance ranking based on RMSE degradation
            if 'RMSE_change' in summary.columns:
                importance = {}
                for ablation in summary.index:
                    if ablation != 'none' and ablation in component_desc and not pd.isna(summary.loc[ablation, 'RMSE_change']):
                        importance[ablation] = abs(summary.loc[ablation, 'RMSE_change'])
                
                # Sort by importance
                sorted_components = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                
                # Write importance ranking
                for i, (ablation, imp) in enumerate(sorted_components):
                    f.write(f"{i+1}. {component_desc[ablation]}\n")
                    f.write(f"   Impact when removed: {imp:.2f}% RMSE degradation\n\n")
            
            f.write(f"Conclusion\n")
            f.write(f"----------\n")
            
            # Check if we have ablation problems
            if zero_components:
                f.write(f"⚠️ ABLATION STUDY INCOMPLETE: Some components ({', '.join(zero_components)}) show no impact\n")
                f.write(f"when ablated, which suggests those ablations are not being properly implemented.\n")
                f.write(f"The ablation parameter names in train.py may not match those in ablation.py.\n")
                f.write(f"Please fix the ablation implementation and re-run the evaluation.\n\n")
            
            # Automatically generate conclusion
            if 'RMSE_change' in summary.columns and importance:
                # Get the most important non-zero component if possible
                valid_components = [(abl, imp) for abl, imp in sorted_components if abl not in zero_components]
                
                if valid_components:
                    most_important = valid_components[0][0].replace('no_', '')
                    f.write(f"The {COMP_FULL[most_important].replace('\\n', ' ')} has the most significant impact on model performance.\n")
                    f.write(f"Removing this component causes a {importance[valid_components[0][0]]:.2f}% degradation in RMSE.\n\n")
                    
                    if any(imp < 0 for abl, imp in valid_components):
                        better_ablations = [(abl, imp) for abl, imp in valid_components if imp < 0]
                        if better_ablations:
                            better_abl, better_imp = sorted(better_ablations, key=lambda x: x[1])[0]
                            comp_name = COMP_FULL[better_abl.replace('no_', '')].replace('\\n', ' ')
                            f.write(f"Interestingly, removing the {comp_name} component slightly improves performance by {abs(better_imp):.2f}%.\n")
                            f.write(f"This suggests potential optimization opportunities or redundancy in this component.\n\n")
                else:
                    f.write(f"⚠️ Unable to draw meaningful conclusions due to issues with the ablation implementation.\n\n")
                    
        logger.info(f"Generated ablation report: {report_path}")
        return report_path
        
    except Exception as e:
        logger.error(f"Error generating ablation report: {str(e)}")
        return None

# =============================================================================
#  ARGUMENT PARSING AND MAIN FUNCTION
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run MSTAGAT-Net ablation studies and generate visualizations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Main operation modes
    parser.add_argument('--only_visualize', action='store_true',
                        help='Skip training and only generate visualizations')
    
    # Dataset and configuration selection
    parser.add_argument('--datasets', nargs='+', choices=[d[0] for d in DATASET_CONFIGS],
                        help='Specific datasets to process (default: all)')
    parser.add_argument('--horizons', type=int, nargs='+',
                        help='Specific horizons to process (default: all appropriate for each dataset)')
    parser.add_argument('--window', type=int, default=DEFAULT_WINDOW,
                        help=f'Window size to use (default: {DEFAULT_WINDOW})')
    parser.add_argument('--ablations', nargs='+', choices=ABLATIONS,
                        help='Specific ablations to run (default: all)')
    
    # Error handling
    parser.add_argument('--ignore_missing', action='store_true',
                        help='Continue even if metrics files are missing')
    parser.add_argument('--continue_on_error', action='store_true',
                        help='Continue pipeline even if individual commands fail')
    
    # Output control
    parser.add_argument('--output_dir', type=str, default=OUT_DIR,
                        help='Directory for visualization outputs')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        default='INFO', help='Logging level')
    
    return parser.parse_args()

def validate_configuration():
    """Validate the configuration settings before running the pipeline."""
    issues = []
    
    # Check for train.py script
    if not os.path.exists(TRAIN_SCRIPT):
        issues.append(f"Training script not found at: {TRAIN_SCRIPT}")
    
    # Check that output directories can be created
    for directory in [METRICS_DIR, OUT_DIR, SAVE_DIR]:
        try:
            os.makedirs(directory, exist_ok=True)
        except PermissionError:
            issues.append(f"Permission denied when creating directory: {directory}")
        except Exception as e:
            issues.append(f"Error creating directory {directory}: {str(e)}")
    
    # Validate dataset configurations
    for ds_config in DATASET_CONFIGS:
        if len(ds_config) != 3:
            issues.append(f"Invalid dataset config: {ds_config}")
    
    if issues:
        for issue in issues:
            logger.error(issue)
        raise ValueError("Configuration validation failed. See errors above.")
    
    logger.info("Configuration validation successful.")

def main():
    """Main pipeline function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging level
    logger.setLevel(getattr(logging, args.log_level))
    
    # Print welcome message
    logger.info("=" * 80)
    logger.info("MSTAGAT-Net Ablation Pipeline")
    logger.info("=" * 80)
    
    # Validate configuration
    try:
        validate_configuration()
    except ValueError:
        logger.error("Configuration validation failed. Exiting.")
        return 1
    
    # Determine which datasets and horizons to process
    datasets_to_process = args.datasets or DATASETS_FOR_PLOTS
    
    # Show configuration summary
    logger.info("Pipeline Configuration:")
    logger.info(f"  Operation mode: {'Visualization only' if args.only_visualize else 'Full pipeline'}")
    logger.info(f"  Datasets: {', '.join(datasets_to_process)}")
    if args.horizons:
        logger.info(f"  Horizons: {', '.join(map(str, args.horizons))}")
    else:
        logger.info("  Horizons: Default for each dataset")
    logger.info(f"  Window size: {args.window}")
    logger.info(f"  Output directory: {args.output_dir}")
    
    # Phase 1: Training (if not skipped)
    if not args.only_visualize:
        logger.info("Starting training phase...")
        
        # Run ablation studies
        training_results = run_ablation_studies(
            datasets=datasets_to_process,
            horizons=args.horizons,
            ablations=args.ablations,
            window=args.window,
            ignore_errors=args.continue_on_error
        )
        
        logger.info(f"Training phase completed. Status: {training_results['status']}")
        logger.info(f"  Total runs: {training_results['total']}")
        logger.info(f"  Completed: {training_results['completed']}")
        logger.info(f"  Failed: {training_results['failed']}")
    else:
        logger.info("Skipping training phase (--only_visualize specified)")
    
    # Phase 2: Generate summaries and validate metrics
    logger.info("Generating ablation summaries and validating metrics...")
    
    validation_results = {}
    for dataset in datasets_to_process:
        validation_results[dataset] = {}
        
        dataset_horizons = args.horizons or get_horizons_for_dataset(dataset)
        for h in dataset_horizons:
            # Generate summary if needed
            summary = load_summary(dataset, args.window, h)
            
            # Check for issues
            issues = check_ablation_issues(dataset, args.window, h)
            validation_results[dataset][h] = {
                'summary_available': summary is not None,
                'issues': issues
            }
            
            # Log validation results
            if issues:
                logger.warning(f"Issues found for {dataset}, h={h}:")
                for issue in issues:
                    logger.warning(f"  - {issue}")
            else:
                logger.info(f"No issues found for {dataset}, h={h}")
    
    # Phase 3: Generate visualizations
    logger.info("Generating visualizations...")
    
    # Create output directory
    ensure_dir_exists(args.output_dir)
    
    # Initialize visualization style
    setup_visualization_style()
    
    # Overall performance table and cross-horizon plots
    generate_performance_table(
        datasets_to_process, args.window, HORIZONS_FOR_PLOTS, args.output_dir
    )
    
    generate_cross_horizon_performance(
        datasets_to_process, args.window, HORIZONS_FOR_PLOTS, args.output_dir
    )
    
    generate_component_importance_comparison(
        datasets_to_process, args.window, HORIZONS_FOR_PLOTS, args.output_dir
    )
    
    # Dataset-specific visualizations
    for dataset in datasets_to_process:
        dataset_horizons = args.horizons or [h for h in HORIZONS_FOR_PLOTS if h in get_horizons_for_dataset(dataset)]
        
        if not dataset_horizons:
            logger.warning(f"No valid horizons found for dataset: {dataset}")
            continue
        
        # Generate grid visualizations
        generate_ablation_impact_grid(
            dataset, args.window, dataset_horizons, args.output_dir
        )
        
        generate_performance_comparison_grid(
            dataset, args.window, dataset_horizons, args.output_dir
        )
        
        # Generate individual horizon metric charts
        for h in dataset_horizons:
            generate_single_horizon_metrics(dataset, args.window, h, args.output_dir)
    
    # Component contribution charts for specific horizons
    for h in [5, 10]:
        # Only generate if this horizon is available
        if h in HORIZONS_FOR_PLOTS:
            generate_component_contribution_charts(datasets_to_process, args.window, h, args.output_dir)
            generate_relative_contribution_chart(datasets_to_process, args.window, h, args.output_dir)
    
    # Phase 4: Generate reports
    logger.info("Generating reports...")
    
    for dataset in datasets_to_process:
        dataset_horizons = args.horizons or get_horizons_for_dataset(dataset)
        for h in dataset_horizons:
            generate_ablation_report(dataset, args.window, h)
    
    logger.info("=" * 80)
    logger.info("Pipeline complete!")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.exception("Unhandled exception in main pipeline:")
        sys.exit(1)