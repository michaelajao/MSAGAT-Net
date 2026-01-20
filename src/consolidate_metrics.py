#!/usr/bin/env python3
"""
Consolidate individual metrics CSV files into aggregated summary files.
Creates all_results.csv and all_ablation_summary.csv.
"""
import os
import glob
import pandas as pd
import numpy as np
from typing import List, Dict

# Import central configuration
from config import DATASET_CONFIGS, ABLATIONS, SEEDS, DEFAULT_WINDOW

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
METRICS_DIR = os.path.join(BASE_DIR, 'report', 'results')

# Dataset list (names only for filtering)
DATASET_LIST = list(DATASET_CONFIGS.keys())


def normalize(name: str) -> str:
    """Normalize column names for matching."""
    return name.strip().lower().replace('-', '').replace('_', '')


def extract_metadata_from_filename(filename: str) -> Dict:
    """Extract metadata from filename pattern."""
    # Example: final_metrics_MSTAGAT-Net.japan.w-20.h-5.none.seed-42.csv
    parts = filename.replace('.csv', '').split('.')
    metadata = {}
    
    for part in parts:
        if part.startswith('w-'):
            metadata['window'] = int(part.split('-')[1])
        elif part.startswith('h-'):
            metadata['horizon'] = int(part.split('-')[1])
        elif 'seed-' in part:
            metadata['seed'] = int(part.split('-')[1])
        elif part in ABLATIONS:
            metadata['ablation'] = part
        elif part in DATASET_LIST or any(ds in part for ds in DATASET_LIST):
            # Extract dataset name
            for ds in DATASET_LIST:
                if ds in part:
                    metadata['dataset'] = ds
                    break
    
    return metadata


def load_and_consolidate_results() -> pd.DataFrame:
    """Load all individual metrics CSV files and consolidate them."""
    all_records = []
    
    # Find all final_metrics CSV files
    pattern = os.path.join(METRICS_DIR, 'final_metrics_*.csv')
    csv_files = glob.glob(pattern)
    
    print(f"Found {len(csv_files)} metrics files")
    
    for csv_file in csv_files:
        try:
            # Extract metadata from filename
            basename = os.path.basename(csv_file)
            metadata = extract_metadata_from_filename(basename)
            
            # Skip if metadata incomplete
            if not all(k in metadata for k in ['dataset', 'window', 'horizon', 'ablation']):
                print(f"  Warning: Could not extract metadata from {basename}")
                continue
            
            # Load CSV
            df = pd.read_csv(csv_file)
            if len(df) == 0:
                continue
            
            # Extract metrics
            record = metadata.copy()
            
            # Map common metric names
            metric_map = {
                'mae': 'MAE',
                'rmse': 'RMSE',
                'pcc': 'PCC',
                'r2': 'R2',
                'mape': 'MAPE',
                'var': 'Var',
                'peak': 'Peak',
                'std_mae': 'std_MAE',
                'rmse_states': 'RMSE_states',
                'pcc_states': 'PCC_states',
                'r2_states': 'R2_states',
                'vars': 'Vars',
            }
            
            for col_key, col_name in metric_map.items():
                norm_key = normalize(col_key)
                for col in df.columns:
                    if normalize(col) == norm_key:
                        record[col_name] = df[col].iloc[0]
                        break
            
            all_records.append(record)
            
        except Exception as e:
            print(f"  Error processing {csv_file}: {e}")
            continue
    
    if not all_records:
        print("No records found!")
        return pd.DataFrame()
    
    # Create DataFrame
    df_all = pd.DataFrame(all_records)
    
    # Sort by dataset, horizon, ablation, seed
    sort_cols = [c for c in ['dataset', 'window', 'horizon', 'ablation', 'seed'] if c in df_all.columns]
    df_all = df_all.sort_values(sort_cols)
    
    return df_all


def compute_ablation_summaries(df_all: pd.DataFrame) -> pd.DataFrame:
    """Compute ablation study summaries across all seeds."""
    summaries = []
    
    # Group by dataset, window, horizon
    for (dataset, window, horizon), group in df_all.groupby(['dataset', 'window', 'horizon']):
        # Get baseline (full model) metrics across all seeds
        baseline_group = group[group['ablation'] == 'none']
        
        if len(baseline_group) == 0:
            continue
        
        # Compute mean baseline metrics
        baseline_metrics = {
            'MAE': baseline_group['MAE'].mean() if 'MAE' in baseline_group.columns else np.nan,
            'RMSE': baseline_group['RMSE'].mean() if 'RMSE' in baseline_group.columns else np.nan,
            'PCC': baseline_group['PCC'].mean() if 'PCC' in baseline_group.columns else np.nan,
            'R2': baseline_group['R2'].mean() if 'R2' in baseline_group.columns else np.nan,
        }
        
        # For each ablation
        for ablation in ABLATIONS:
            ab_group = group[group['ablation'] == ablation]
            
            if len(ab_group) == 0:
                continue
            
            record = {
                'dataset': dataset,
                'window': window,
                'horizon': horizon,
                'ablation': ablation,
            }
            
            # Compute mean metrics across seeds
            for metric in ['MAE', 'RMSE', 'PCC', 'R2']:
                if metric in ab_group.columns:
                    mean_val = ab_group[metric].mean()
                    std_val = ab_group[metric].std()
                    record[metric] = mean_val
                    record[f'{metric}_std'] = std_val
                    
                    # Compute percent change from baseline
                    if ablation != 'none' and not np.isnan(baseline_metrics[metric]) and baseline_metrics[metric] != 0:
                        pct_change = ((mean_val - baseline_metrics[metric]) / baseline_metrics[metric]) * 100
                        record[f'{metric}_CHANGE'] = pct_change
            
            summaries.append(record)
    
    if not summaries:
        return pd.DataFrame()
    
    df_summary = pd.DataFrame(summaries)
    
    # Sort
    sort_cols = [c for c in ['dataset', 'window', 'horizon', 'ablation'] if c in df_summary.columns]
    df_summary = df_summary.sort_values(sort_cols)
    
    return df_summary


def main():
    """Main consolidation function."""
    print("=" * 60)
    print("Consolidating Metrics CSVs")
    print("=" * 60)
    
    # Consolidate all results
    print("\n1. Loading and consolidating individual results...")
    df_all = load_and_consolidate_results()
    
    if len(df_all) == 0:
        print("No data to consolidate!")
        return
    
    print(f"   Consolidated {len(df_all)} records")
    
    # Save all results
    all_results_path = os.path.join(METRICS_DIR, 'all_results.csv')
    df_all.to_csv(all_results_path, index=False)
    print(f"   Saved: {all_results_path}")
    
    # Compute ablation summaries
    print("\n2. Computing ablation summaries...")
    df_summary = compute_ablation_summaries(df_all)
    
    if len(df_summary) > 0:
        summary_path = os.path.join(METRICS_DIR, 'all_ablation_summary.csv')
        df_summary.to_csv(summary_path, index=False)
        print(f"   Saved: {summary_path}")
        print(f"   Summary has {len(df_summary)} records")
    else:
        print("   No summary data to save")
    
    # Create per-dataset summaries (optional, for compatibility)
    print("\n3. Creating per-dataset summary files...")
    for dataset in DATASET_LIST:
        dataset_summary = df_summary[df_summary['dataset'] == dataset]
        
        if len(dataset_summary) == 0:
            continue
        
        # Create dataset directory
        dataset_dir = os.path.join(METRICS_DIR, dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Save dataset-specific all_results
        dataset_results = df_all[df_all['dataset'] == dataset]
        dataset_results_path = os.path.join(dataset_dir, 'all_results.csv')
        dataset_results.to_csv(dataset_results_path, index=False)
        
        # Save dataset-specific summary
        dataset_summary_path = os.path.join(dataset_dir, 'all_ablation_summary.csv')
        dataset_summary.to_csv(dataset_summary_path, index=False)
        
        print(f"   {dataset}: {len(dataset_results)} results, {len(dataset_summary)} summary records")
    
    print("\n" + "=" * 60)
    print("Consolidation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
