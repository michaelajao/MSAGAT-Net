#!/usr/bin/env python3
"""
Migration script to consolidate existing individual CSV files into single consolidated files.

This script:
1. Reads all existing final_metrics_*.csv files and combines them into all_results.csv
2. Reads all existing ablation_summary_*.csv files and combines them into all_ablation_summary.csv
3. Optionally removes the old individual files after successful migration

Run this once to migrate existing results to the new consolidated format.
"""
import os
import glob
import pandas as pd
from datetime import datetime

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(BASE_DIR, 'report', 'results')

# Consolidated file names
ALL_RESULTS_CSV = "all_results.csv"
ALL_ABLATION_SUMMARY_CSV = "all_ablation_summary.csv"
ALL_ABLATION_REPORT_TXT = "all_ablation_report.txt"


def extract_config_from_filename(filename):
    """Extract dataset, window, horizon, ablation from filename."""
    # Example: final_metrics_MSTAGAT-Net.japan.w-20.h-3.none.csv
    basename = os.path.basename(filename)
    parts = basename.replace('final_metrics_', '').replace('.csv', '').split('.')
    
    config = {'model': '', 'dataset': '', 'window': 0, 'horizon': 0, 'ablation': 'none'}
    
    for part in parts:
        if part.startswith('w-'):
            config['window'] = int(part[2:])
        elif part.startswith('h-'):
            config['horizon'] = int(part[2:])
        elif part in ['none', 'no_agam', 'no_mtfm', 'no_pprm']:
            config['ablation'] = part
        elif part.startswith('MSTAGAT') or part.startswith('msagat'):
            config['model'] = part
        else:
            config['dataset'] = part
    
    return config


def extract_summary_config_from_filename(filename):
    """Extract dataset, window, horizon from ablation summary filename."""
    # Example: ablation_summary_japan.w-20.h-3.csv
    basename = os.path.basename(filename)
    parts = basename.replace('ablation_summary_', '').replace('.csv', '').split('.')
    
    config = {'dataset': '', 'window': 0, 'horizon': 0}
    
    for part in parts:
        if part.startswith('w-'):
            config['window'] = int(part[2:])
        elif part.startswith('h-'):
            config['horizon'] = int(part[2:])
        else:
            config['dataset'] = part
    
    return config


def migrate_final_metrics(delete_old=False):
    """Migrate individual final_metrics files to consolidated all_results.csv."""
    pattern = os.path.join(METRICS_DIR, 'final_metrics_*.csv')
    files = glob.glob(pattern)
    
    if not files:
        print("No individual final_metrics files found to migrate.")
        return
    
    print(f"Found {len(files)} individual final_metrics files to migrate...")
    
    all_records = []
    for filepath in files:
        try:
            config = extract_config_from_filename(filepath)
            df = pd.read_csv(filepath)
            
            # Add config info if not present
            for key, value in config.items():
                if key not in df.columns or pd.isna(df[key].iloc[0]):
                    df[key] = value
            
            all_records.append(df)
            print(f"  Processed: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"  Error processing {filepath}: {e}")
    
    if all_records:
        consolidated = pd.concat(all_records, ignore_index=True)
        
        # Ensure required columns exist
        required_cols = ['model', 'dataset', 'window', 'horizon', 'ablation']
        for col in required_cols:
            if col not in consolidated.columns:
                consolidated[col] = ''
        
        # Sort and save
        consolidated = consolidated.sort_values(['dataset', 'window', 'horizon', 'ablation']).reset_index(drop=True)
        
        output_path = os.path.join(METRICS_DIR, ALL_RESULTS_CSV)
        consolidated.to_csv(output_path, index=False)
        print(f"\nConsolidated {len(all_records)} files into: {output_path}")
        print(f"Total records: {len(consolidated)}")
        
        if delete_old:
            for filepath in files:
                os.remove(filepath)
                print(f"  Deleted: {os.path.basename(filepath)}")
            print(f"Deleted {len(files)} old individual files.")


def migrate_ablation_summaries(delete_old=False):
    """Migrate individual ablation_summary files to consolidated all_ablation_summary.csv."""
    pattern = os.path.join(METRICS_DIR, 'ablation_summary_*.csv')
    files = glob.glob(pattern)
    
    if not files:
        print("No individual ablation_summary files found to migrate.")
        return
    
    print(f"\nFound {len(files)} individual ablation_summary files to migrate...")
    
    all_records = []
    for filepath in files:
        try:
            config = extract_summary_config_from_filename(filepath)
            df = pd.read_csv(filepath, index_col=0)
            
            # Add config info
            df['dataset'] = config['dataset']
            df['window'] = config['window']
            df['horizon'] = config['horizon']
            df['ablation'] = df.index
            df = df.reset_index(drop=True)
            
            all_records.append(df)
            print(f"  Processed: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"  Error processing {filepath}: {e}")
    
    if all_records:
        consolidated = pd.concat(all_records, ignore_index=True)
        
        # Sort and save
        consolidated = consolidated.sort_values(['dataset', 'window', 'horizon', 'ablation']).reset_index(drop=True)
        
        output_path = os.path.join(METRICS_DIR, ALL_ABLATION_SUMMARY_CSV)
        consolidated.to_csv(output_path, index=False)
        print(f"\nConsolidated {len(all_records)} files into: {output_path}")
        print(f"Total records: {len(consolidated)}")
        
        if delete_old:
            for filepath in files:
                os.remove(filepath)
                print(f"  Deleted: {os.path.basename(filepath)}")
            print(f"Deleted {len(files)} old individual files.")


def migrate_ablation_reports():
    """Migrate individual ablation report TXT files to consolidated file."""
    pattern = os.path.join(METRICS_DIR, 'ablation_report_*.txt')
    files = glob.glob(pattern)
    
    if not files:
        print("\nNo individual ablation_report files found to migrate.")
        return
    
    print(f"\nFound {len(files)} individual ablation_report files to migrate...")
    
    output_path = os.path.join(METRICS_DIR, ALL_ABLATION_REPORT_TXT)
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write(f"# Consolidated Ablation Reports\n")
        outfile.write(f"# Migrated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        outfile.write("=" * 60 + "\n\n")
        
        for filepath in sorted(files):
            try:
                with open(filepath, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write("\n\n")
                print(f"  Processed: {os.path.basename(filepath)}")
            except Exception as e:
                print(f"  Error processing {filepath}: {e}")
    
    print(f"\nConsolidated {len(files)} files into: {output_path}")


def main():
    """Run migration."""
    print("=" * 60)
    print("MSAGAT-Net Results Migration Script")
    print("=" * 60)
    print(f"Migrating files from: {METRICS_DIR}")
    print()
    
    # Ask user if they want to delete old files
    delete_old = input("Delete old individual files after migration? (y/N): ").strip().lower() == 'y'
    print()
    
    # Migrate all file types
    migrate_final_metrics(delete_old=delete_old)
    migrate_ablation_summaries(delete_old=delete_old)
    migrate_ablation_reports()
    
    print()
    print("=" * 60)
    print("Migration complete!")
    print("=" * 60)
    print("\nNew consolidated files:")
    print(f"  - {ALL_RESULTS_CSV}: All experiment metrics")
    print(f"  - {ALL_ABLATION_SUMMARY_CSV}: All ablation summaries")
    print(f"  - {ALL_ABLATION_REPORT_TXT}: All ablation reports")


if __name__ == '__main__':
    main()
