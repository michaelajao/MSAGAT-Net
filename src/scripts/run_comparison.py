#!/usr/bin/env python3
"""
Comprehensive Model Comparison: MSAGAT-Net vs EpiSIG-Net

Run experiments across multiple datasets, horizons, and seeds to compute
average metrics with standard deviations.

Datasets and horizons:
- Japan, USA (state360): horizons 2, 5, 10, 15
- LTLA, Australia, Spain: horizons 3, 7, 14

Seeds: 10 random seeds for statistical significance
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scripts.experiments import run_single_experiment, DATASET_CONFIGS

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Dataset-specific horizons
COMPARISON_CONFIG = {
    'japan': {
        'horizons': [2, 5, 10, 15],
        'sim_mat': 'japan-adj',
    },
    'state360': {
        'horizons': [2, 5, 10, 15],
        'sim_mat': 'state-adj-49',
    },
    'ltla_timeseries': {
        'horizons': [3, 7, 14],
        'sim_mat': 'ltla-adj',
    },
    'australia-covid': {
        'horizons': [3, 7, 14],
        'sim_mat': 'australia-adj',
    },
    'spain-covid': {
        'horizons': [3, 7, 14],
        'sim_mat': 'spain-adj',
    },
}

# 10 seeds for statistical significance
COMPARISON_SEEDS = [5, 10, 20, 30, 42, 45, 100, 123, 500, 1000]

# Models to compare
MODELS = ['msagat', 'episig']


def run_comparison_batch(
    datasets: List[str],
    models: List[str],
    seeds: List[int],
    force_cpu: bool = False,
    save_dir: str = 'save_comparison',
    results_file: str = 'comparison_results.json'
) -> Dict:
    """
    Run comprehensive comparison experiments.
    
    Returns:
        Dict with structure: {dataset: {model: {horizon: {metric: [values]}}}}
    """
    results = {}
    
    # Count total experiments
    total = 0
    for dataset in datasets:
        config = COMPARISON_CONFIG.get(dataset, {})
        horizons = config.get('horizons', [3, 7, 14])
        total += len(horizons) * len(seeds) * len(models)
    
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON")
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Seeds: {len(seeds)} seeds")
    print(f"Total experiments: {total}")
    print("="*80)
    
    completed = 0
    failed = 0
    
    for dataset in datasets:
        config = COMPARISON_CONFIG.get(dataset, {})
        horizons = config.get('horizons', [3, 7, 14])
        
        if dataset not in results:
            results[dataset] = {}
        
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"Horizons: {horizons}")
        print(f"{'='*60}")
        
        for model in models:
            if model not in results[dataset]:
                results[dataset][model] = {}
            
            for horizon in horizons:
                if horizon not in results[dataset][model]:
                    results[dataset][model][horizon] = {
                        'mae': [], 'rmse': [], 'pcc': [], 'r2': []
                    }
                
                for seed in seeds:
                    completed += 1
                    print(f"\n[{completed}/{total}] {dataset} | {model} | h={horizon} | seed={seed}")
                    
                    try:
                        metrics = run_single_experiment(
                            dataset=dataset,
                            horizon=horizon,
                            seed=seed,
                            ablation='none',
                            save_dir=save_dir,
                            force_cpu=force_cpu,
                            verbose=False,
                            model_type=model
                        )
                        
                        # Store metrics (note: R2 is capitalized in training.py)
                        results[dataset][model][horizon]['mae'].append(metrics['mae'])
                        results[dataset][model][horizon]['rmse'].append(metrics['rmse'])
                        results[dataset][model][horizon]['pcc'].append(metrics['pcc'])
                        results[dataset][model][horizon]['r2'].append(metrics['R2'])
                        
                        print(f"  [OK] MAE={metrics['mae']:.4f}, PCC={metrics['pcc']:.4f}, R2={metrics['r2']:.4f}")
                        
                    except Exception as e:
                        failed += 1
                        print(f"  [FAIL] {e}")
                
                # Save intermediate results
                save_results(results, results_file)
    
    print(f"\n{'='*80}")
    print(f"COMPARISON COMPLETE: {completed-failed}/{total} succeeded, {failed} failed")
    print(f"{'='*80}")
    
    return results


def convert_to_python_types(obj):
    """Convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_types(i) for i in obj]
    return obj


def save_results(results: Dict, filename: str):
    """Save results to JSON file."""
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'report', 'results'
    )
    os.makedirs(results_dir, exist_ok=True)
    
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(convert_to_python_types(results), f, indent=2)
    print(f"  Results saved to {filepath}")


def compute_statistics(results: Dict) -> Dict:
    """Compute mean and std for all metrics."""
    stats = {}
    
    for dataset, models_data in results.items():
        stats[dataset] = {}
        for model, horizons_data in models_data.items():
            stats[dataset][model] = {}
            for horizon, metrics in horizons_data.items():
                stats[dataset][model][horizon] = {}
                for metric_name, values in metrics.items():
                    if values:
                        stats[dataset][model][horizon][metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'n': len(values)
                        }
    
    return stats


def print_comparison_table(stats: Dict):
    """Print formatted comparison table."""
    print("\n" + "="*100)
    print("COMPARISON RESULTS (Mean ± Std)")
    print("="*100)
    
    for dataset in stats:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'='*80}")
        print(f"{'Model':<12} {'Horizon':>8} {'MAE':>18} {'RMSE':>18} {'PCC':>14} {'R2':>14}")
        print("-"*80)
        
        for model in sorted(stats[dataset].keys()):
            for horizon in sorted(stats[dataset][model].keys(), key=int):
                m = stats[dataset][model][horizon]
                mae_str = f"{m['mae']['mean']:.2f}±{m['mae']['std']:.2f}" if 'mae' in m else "N/A"
                rmse_str = f"{m['rmse']['mean']:.2f}±{m['rmse']['std']:.2f}" if 'rmse' in m else "N/A"
                pcc_str = f"{m['pcc']['mean']:.4f}±{m['pcc']['std']:.4f}" if 'pcc' in m else "N/A"
                r2_str = f"{m['r2']['mean']:.4f}±{m['r2']['std']:.4f}" if 'r2' in m else "N/A"
                
                print(f"{model:<12} {horizon:>8} {mae_str:>18} {rmse_str:>18} {pcc_str:>14} {r2_str:>14}")


def print_winner_summary(stats: Dict):
    """Print which model wins per dataset/horizon."""
    print("\n" + "="*80)
    print("WINNER SUMMARY (based on PCC)")
    print("="*80)
    
    for dataset in stats:
        print(f"\n{dataset}:")
        for horizon in sorted(set(h for m in stats[dataset].values() for h in m.keys()), key=int):
            models_pcc = {}
            for model in stats[dataset]:
                if horizon in stats[dataset][model] and 'pcc' in stats[dataset][model][horizon]:
                    models_pcc[model] = stats[dataset][model][horizon]['pcc']['mean']
            
            if models_pcc:
                winner = max(models_pcc, key=models_pcc.get)
                diff = models_pcc.get('msagat', 0) - models_pcc.get('episig', 0)
                print(f"  h={horizon:>2}: {winner.upper():<10} (MSAGAT: {models_pcc.get('msagat', 0):.4f}, EpiSIG: {models_pcc.get('episig', 0):.4f}, diff: {diff:+.4f})")


def parse_args():
    parser = argparse.ArgumentParser(description='Run comprehensive model comparison')
    parser.add_argument('--datasets', nargs='+', default=list(COMPARISON_CONFIG.keys()),
                        help='Datasets to compare')
    parser.add_argument('--models', nargs='+', default=MODELS,
                        help='Models to compare')
    parser.add_argument('--seeds', nargs='+', type=int, default=COMPARISON_SEEDS,
                        help='Random seeds')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training')
    parser.add_argument('--save_dir', type=str, default='save_comparison',
                        help='Directory to save model checkpoints')
    parser.add_argument('--results', type=str, default='comparison_results.json',
                        help='Results JSON filename')
    parser.add_argument('--analyze-only', action='store_true',
                        help='Only analyze existing results (no training)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'report', 'results'
    )
    results_path = os.path.join(results_dir, args.results)
    
    if args.analyze_only:
        # Load existing results
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
            print(f"Loaded results from {results_path}")
        else:
            print(f"No results file found at {results_path}")
            sys.exit(1)
    else:
        # Run experiments
        results = run_comparison_batch(
            datasets=args.datasets,
            models=args.models,
            seeds=args.seeds,
            force_cpu=args.cpu,
            save_dir=args.save_dir,
            results_file=args.results
        )
    
    # Compute statistics and print
    stats = compute_statistics(results)
    print_comparison_table(stats)
    print_winner_summary(stats)
    
    # Save final statistics
    stats_file = args.results.replace('.json', '_stats.json')
    stats_path = os.path.join(results_dir, stats_file)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        return obj
    
    with open(stats_path, 'w') as f:
        json.dump(convert_numpy(stats), f, indent=2)
    print(f"\nStatistics saved to {stats_path}")


if __name__ == '__main__':
    main()
