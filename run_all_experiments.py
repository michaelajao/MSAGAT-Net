"""
Run all experiments for paper comparison:
  MSAGAT-Net v2, EpiGNN, Cola-GNN, DCRNN, LSTNet, CNNRNN-Res.
All models use seed=42, 60/20/20 split, GPU, window=20.
Results logged to doc/paper_results_final.csv.

Usage:
    python run_all_experiments.py --model msagat
    python run_all_experiments.py --model epignn
    python run_all_experiments.py --model colagnn
    python run_all_experiments.py --model dcrnn
    python run_all_experiments.py --model lstnet
    python run_all_experiments.py --model cnnrnn_res
    python run_all_experiments.py --model all
    python run_all_experiments.py --model msagat --datasets japan region785
"""
import subprocess
import sys
import os
import csv
import time
import re

SEED = 42

# Paper datasets and horizons (Spain removed - only 122 timesteps, unusable)
DATASETS = {
    'japan':            {'sim_mat': 'japan-adj',      'horizons': [3, 5, 10, 15]},
    'region785':        {'sim_mat': 'region-adj',     'horizons': [3, 5, 10, 15]},
    'state360':         {'sim_mat': 'state-adj-49',   'horizons': [3, 5, 10, 15]},
    'australia-covid':  {'sim_mat': 'australia-adj',  'horizons': [3, 7, 14]},
    'ltla_timeseries':  {'sim_mat': 'ltla-adj',       'horizons': [3, 7, 14]},
    'nhs_timeseries':   {'sim_mat': 'nhs-adj',        'horizons': [3, 7, 14]},
}

ALL_MODELS = ['msagat', 'epignn', 'colagnn', 'dcrnn', 'lstnet', 'cnnrnn_res']

# Project roots
MSAGAT_ROOT = os.path.dirname(os.path.abspath(__file__))
EPIGNN_ROOT = os.path.join(os.path.dirname(MSAGAT_ROOT), 'EpiGNN')
COLAGNN_ROOT = os.path.join(os.path.dirname(MSAGAT_ROOT), 'colagnn')
RESULTS_FILE = os.path.join(MSAGAT_ROOT, 'doc', 'paper_results_final.csv')


def parse_msagat_output(output):
    """Parse: Final TEST - MAE: 333.60, RMSE: 1133.24, PCC: 0.8629, R2: 0.6950"""
    metrics = {}
    for line in output.split('\n'):
        if 'Final TEST' in line:
            for m in re.finditer(r'(MAE|RMSE|PCC|R2):\s*([\d.eE+-]+)', line):
                metrics[m.group(1).lower()] = float(m.group(2))
            break
    return metrics


def parse_baseline_output(output):
    """Parse: TEST MAE 459.38 std 123.45 RMSE 1115.63 RMSEs 234.56 PCC 0.8531 ..."""
    metrics = {}
    for line in output.split('\n'):
        if line.strip().startswith('TEST MAE'):
            tokens = line.split()
            for i, tok in enumerate(tokens):
                if tok == 'MAE' and i + 1 < len(tokens):
                    metrics['mae'] = float(tokens[i + 1])
                elif tok == 'RMSE' and i + 1 < len(tokens) and (i == 0 or tokens[i-1] != 'std'):
                    if 'rmse' not in metrics:
                        metrics['rmse'] = float(tokens[i + 1])
                elif tok == 'PCC' and i + 1 < len(tokens):
                    if 'pcc' not in metrics:
                        metrics['pcc'] = float(tokens[i + 1])
                elif tok == 'R2' and i + 1 < len(tokens):
                    if 'r2' not in metrics:
                        metrics['r2'] = float(tokens[i + 1])
            break
    return metrics


def build_command(model_name, dataset, sim_mat, horizon, seed):
    """Build the command and working directory for each model."""

    if model_name == 'msagat':
        cmd = [
            sys.executable, '-m', 'src.scripts.experiments',
            '--single', '--dataset', dataset, '--horizon', str(horizon),
            '--seed', str(seed), '--model', 'msagat'
        ]
        return cmd, MSAGAT_ROOT

    elif model_name == 'epignn':
        cmd = [
            sys.executable, 'src/train.py',
            '--dataset', dataset, '--sim_mat', sim_mat,
            '--horizon', str(horizon), '--seed', str(seed),
            '--cuda',
            '--train', '0.6', '--val', '0.2', '--test', '0.2',
            '--record', 'yes'
        ]
        return cmd, EPIGNN_ROOT

    elif model_name == 'colagnn':
        cmd = [
            sys.executable, 'src/train.py',
            '--model', 'cola_gnn',
            '--dataset', dataset, '--sim_mat', sim_mat,
            '--horizon', str(horizon), '--seed', str(seed),
            '--cuda',
            '--train', '0.6', '--val', '0.2', '--test', '0.2',
        ]
        return cmd, COLAGNN_ROOT

    elif model_name == 'dcrnn':
        cmd = [
            sys.executable, 'src/train.py',
            '--model', 'dcrnn',
            '--dataset', dataset, '--sim_mat', sim_mat,
            '--horizon', str(horizon), '--seed', str(seed),
            '--cuda',
            '--train', '0.6', '--val', '0.2', '--test', '0.2',
        ]
        return cmd, COLAGNN_ROOT

    elif model_name == 'lstnet':
        cmd = [
            sys.executable, 'src/train.py',
            '--model', 'lstnet',
            '--dataset', dataset, '--sim_mat', sim_mat,
            '--horizon', str(horizon), '--seed', str(seed),
            '--cuda',
            '--train', '0.6', '--val', '0.2', '--test', '0.2',
        ]
        return cmd, COLAGNN_ROOT

    elif model_name == 'cnnrnn_res':
        cmd = [
            sys.executable, 'src/train.py',
            '--model', 'CNNRNN_Res',
            '--dataset', dataset, '--sim_mat', sim_mat,
            '--horizon', str(horizon), '--seed', str(seed),
            '--cuda',
            '--train', '0.6', '--val', '0.2', '--test', '0.2',
        ]
        return cmd, COLAGNN_ROOT

    return None, None


def run_experiment(model_name, dataset, sim_mat, horizon, seed):
    """Run a single experiment and return metrics dict."""

    cmd, cwd = build_command(model_name, dataset, sim_mat, horizon, seed)
    if cmd is None:
        return {}

    print(f"    CMD: {' '.join(cmd)}")
    print(f"    CWD: {cwd}")

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=cwd,
        timeout=3600  # 1 hour max per experiment
    )

    output = result.stdout + '\n' + result.stderr

    if model_name == 'msagat':
        return parse_msagat_output(output)
    else:
        return parse_baseline_output(output)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run all paper experiments')
    parser.add_argument('--model', choices=ALL_MODELS + ['all'], default='all',
                        help='Which model(s) to run')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets (default: all)')
    args = parser.parse_args()

    models = ALL_MODELS if args.model == 'all' else [args.model]
    ds_filter = args.datasets or list(DATASETS.keys())
    datasets = {k: v for k, v in DATASETS.items() if k in ds_filter}

    # Prepare CSV - start fresh
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    write_header = not os.path.exists(RESULTS_FILE)

    total = sum(len(v['horizons']) for v in datasets.values()) * len(models)
    done = 0
    failed = 0

    print("=" * 70)
    print(f"PAPER EXPERIMENTS  |  seed={SEED}  |  split=60/20/20  |  GPU")
    print(f"Models: {models}")
    print(f"Datasets: {list(datasets.keys())}")
    print(f"Total experiments: {total}")
    print("=" * 70)

    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['model', 'dataset', 'horizon', 'mae', 'rmse', 'pcc', 'r2'])

        for model_name in models:
            print(f"\n{'#'*70}")
            print(f"# MODEL: {model_name.upper()}")
            print(f"{'#'*70}")

            for ds_name, ds_cfg in datasets.items():
                for horizon in ds_cfg['horizons']:
                    done += 1
                    print(f"\n[{done}/{total}] {model_name.upper()} | {ds_name} | h={horizon}")

                    start = time.time()
                    metrics = run_experiment(
                        model_name, ds_name, ds_cfg['sim_mat'], horizon, SEED
                    )
                    elapsed = time.time() - start

                    if metrics:
                        print(f"    RMSE={metrics.get('rmse','?')}  PCC={metrics.get('pcc','?')}  "
                              f"MAE={metrics.get('mae','?')}  R2={metrics.get('r2','?')}  "
                              f"({elapsed:.0f}s)")
                        writer.writerow([
                            model_name, ds_name, horizon,
                            f"{metrics.get('mae', ''):.4f}" if metrics.get('mae') else '',
                            f"{metrics.get('rmse', ''):.4f}" if metrics.get('rmse') else '',
                            f"{metrics.get('pcc', ''):.4f}" if metrics.get('pcc') else '',
                            f"{metrics.get('r2', ''):.4f}" if metrics.get('r2') else '',
                        ])
                    else:
                        failed += 1
                        print(f"    FAILED ({elapsed:.0f}s)")
                        writer.writerow([model_name, ds_name, horizon, 'FAILED', '', '', ''])
                    f.flush()

    print(f"\n{'='*70}")
    print(f"COMPLETE: {done-failed}/{total} succeeded, {failed} failed")
    print(f"Results: {RESULTS_FILE}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
