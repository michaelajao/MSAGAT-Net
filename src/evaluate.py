"""
Evaluation module for MSAGAT-Net.

Generates publication-ready figures and aggregates multi-seed results.

Usage:
    python -m src.evaluate                          # all figures
    python -m src.evaluate --figures                 # figures only
    python -m src.evaluate --aggregate               # aggregate results only
    python -m src.evaluate --aggregate --format latex
"""

import os
import sys
import glob
import argparse
from argparse import Namespace
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from .data import DataBasicLoader
from .models import MSAGATNet_Ablation
from .utils import visualize_matrices, visualize_predictions, visualize_predictions_summary

# ── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_DIR = os.path.join(BASE_DIR, 'report', 'results')
OUT_DIR = os.path.join(BASE_DIR, 'report', 'figures', 'paper')
MODEL_DIR = os.path.join(BASE_DIR, 'save_all')
DEFAULT_WINDOW = 20
DEFAULT_SEED = 42

# ── Dataset configs ──────────────────────────────────────────────────────────

DATASET_CONFIGS_FULL = {
    'japan':           {'horizons': [3,5,10,15], 'sim_mat': 'japan-adj',    'use_adj_prior': True, 'use_graph_bias': True},
    'region785':       {'horizons': [3,5,10,15], 'sim_mat': 'region-adj',   'use_adj_prior': True, 'use_graph_bias': True},
    'state360':        {'horizons': [3,5,10,15], 'sim_mat': 'state-adj-49', 'use_adj_prior': True, 'use_graph_bias': True},
    'nhs_timeseries':  {'horizons': [3,7,14],    'sim_mat': 'nhs-adj',      'use_adj_prior': True, 'use_graph_bias': True},
    'ltla_timeseries': {'horizons': [3,7,14],    'sim_mat': 'ltla-adj',     'use_adj_prior': True, 'use_graph_bias': True},
    'australia-covid': {'horizons': [3,7,14],    'sim_mat': 'australia-adj', 'use_adj_prior': True, 'use_graph_bias': True},
}

DATASET_CONFIGS = [(name, cfg['horizons']) for name, cfg in DATASET_CONFIGS_FULL.items()]
ABLATIONS = ['none', 'no_agam', 'no_mtfm', 'no_pprm']
METRICS = ['rmse', 'mae', 'pcc', 'r2']
METRIC_NAMES = {'rmse': 'RMSE', 'mae': 'MAE', 'pcc': 'PCC', 'r2': r'R$^2$'}
COMP_NAMES = {'agam': 'EAGAM', 'mtfm': 'MSSFM', 'pprm': 'PPRM'}
DATASET_NAMES = {
    'japan': 'Japan', 'region785': 'US Region', 'state360': 'US States',
    'nhs_timeseries': 'NHS (UK)', 'ltla_timeseries': 'LTLA (UK)',
    'australia-covid': 'Australia',
}


# ── Utilities ────────────────────────────────────────────────────────────────

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def normalize(name):
    return name.strip().lower().replace('-', '').replace('_', '')

def get_metric(df, key):
    search = normalize(key)
    for col in df.columns:
        if normalize(col) == search:
            return df[col].iloc[0]
    return float('nan')

def load_metrics(dataset, horizon, ablation):
    for base in [os.path.join(METRICS_DIR, dataset), METRICS_DIR]:
        csv = os.path.join(base, 'all_results.csv')
        if os.path.exists(csv):
            df = pd.read_csv(csv)
            mask = ((df['dataset'] == dataset) & (df['window'] == DEFAULT_WINDOW) &
                    (df['horizon'] == horizon) & (df['ablation'] == ablation) &
                    (df['seed'] == DEFAULT_SEED) & (df['model'] == 'MSAGAT-Net'))
            if mask.any():
                return df[mask].iloc[[0]]
    return None


# ── Plot style ───────────────────────────────────────────────────────────────

def setup_style():
    plt.rcdefaults()
    colors = ["#0173B2", "#DE8F05", "#029E73", "#D55E00", "#CC78BC", "#CA9161", "#949494"]
    sns.set_palette(colors)
    sns.set_style('ticks', {'axes.grid': False})
    plt.rcParams.update({
        'figure.dpi': 300, 'savefig.dpi': 300, 'figure.figsize': (7, 5),
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'legend.frameon': False, 'lines.linewidth': 1.5, 'lines.markersize': 6,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.facecolor': 'white', 'figure.facecolor': 'white',
    })

def save_fig(fig, name):
    ensure_dir(OUT_DIR)
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), bbox_inches='tight', dpi=300, facecolor='white')
    print(f"  Saved: {name}.png")
    plt.close(fig)


# ── Figure generators ────────────────────────────────────────────────────────

def fig_performance_vs_horizon():
    setup_style()
    markers = ['o', 's', 'D', '^', 'v', 'p', 'h']
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
    colors = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161', '#949494']

    for metric in ['rmse', 'pcc']:
        fig, ax = plt.subplots(figsize=(7, 5))
        for idx, (ds, horizons) in enumerate(DATASET_CONFIGS):
            vals, hs = [], []
            for h in horizons:
                df = load_metrics(ds, h, 'none')
                if df is not None:
                    val = get_metric(df, metric)
                    if not np.isnan(val):
                        vals.append(val); hs.append(h)
            if vals:
                ax.plot(hs, vals, marker=markers[idx], linestyle=linestyles[idx],
                        color=colors[idx], linewidth=2, markersize=8,
                        markeredgecolor='black', markeredgewidth=0.5,
                        label=DATASET_NAMES.get(ds, ds))
        ax.set_xlabel('Prediction Horizon (days)')
        ax.set_ylabel(METRIC_NAMES[metric])
        ax.legend(loc='best')
        all_h = sorted(set(h for _, horizons in DATASET_CONFIGS for h in horizons))
        ax.set_xticks(all_h)
        save_fig(fig, f'fig1_{metric}_vs_horizon')


def fig_ablation_study():
    setup_style()
    ab_colors = {'none': '#0173B2', 'no_agam': '#DE8F05', 'no_mtfm': '#029E73', 'no_pprm': '#D55E00'}
    ab_labels = {'none': 'Full Model', 'no_agam': 'w/o EAGAM', 'no_mtfm': 'w/o MSSFM', 'no_pprm': 'w/o PPRM'}

    for ds, horizons in [('japan', [3, 7, 14])]:
        for h in horizons:
            vals, labels, colors = [], [], []
            for ab in ABLATIONS:
                df = load_metrics(ds, h, ab)
                if df is not None:
                    rmse = get_metric(df, 'rmse')
                    if not np.isnan(rmse):
                        vals.append(rmse)
                        labels.append(ab_labels.get(ab, ab))
                        colors.append(ab_colors.get(ab, '#333'))
            if len(vals) < 2:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(labels, vals, color=colors, edgecolor='black', linewidth=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
            if vals:
                ax.axhline(y=vals[0], color='navy', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_ylabel('RMSE')
            ax.set_title(f'{DATASET_NAMES.get(ds, ds)} - Horizon {h}')
            ax.set_ylim(0, max(vals) * 1.15)
            plt.xticks(rotation=15, ha='right')
            plt.tight_layout()
            save_fig(fig, f'fig2_ablation_{ds}_h{h}')


def fig_component_impact():
    setup_style()
    for ds, horizons in [('japan', [3, 7, 14])]:
        data = []
        for h in horizons:
            full_df = load_metrics(ds, h, 'none')
            if full_df is None:
                continue
            baseline = get_metric(full_df, 'rmse')
            if np.isnan(baseline) or baseline == 0:
                continue
            for ab in ['no_agam', 'no_mtfm', 'no_pprm']:
                ab_df = load_metrics(ds, h, ab)
                if ab_df is not None:
                    val = get_metric(ab_df, 'rmse')
                    if not np.isnan(val):
                        comp = ab.replace('no_', '')
                        data.append({'Horizon': f'H{h}', 'Component': COMP_NAMES.get(comp, comp),
                                     'RMSE Increase (%)': ((val - baseline) / baseline) * 100})
        if not data:
            continue
        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(8, 5))
        comp_colors = {'EAGAM': '#0173B2', 'MSSFM': '#DE8F05', 'PPRM': '#029E73'}
        x = np.arange(len(df['Horizon'].unique()))
        width, hlist = 0.25, df['Horizon'].unique()
        for i, comp in enumerate(['EAGAM', 'MSSFM', 'PPRM']):
            vals = [df[(df['Horizon'] == h) & (df['Component'] == comp)]['RMSE Increase (%)'].values[0]
                    if len(df[(df['Horizon'] == h) & (df['Component'] == comp)]) > 0 else 0 for h in hlist]
            bars = ax.bar(x + i * width, vals, width, label=comp, color=comp_colors[comp],
                          edgecolor='black', linewidth=0.8)
            for bar, val in zip(bars, vals):
                if val != 0:
                    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                            f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.set_xlabel('Prediction Horizon')
        ax.set_ylabel('RMSE Increase (%)')
        ax.set_title(f'{DATASET_NAMES.get(ds, ds)} - Component Impact')
        ax.set_xticks(x + width)
        ax.set_xticklabels(hlist)
        ax.legend(title='Removed Component')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(min(ymin, -5), max(ymax, 10))
        plt.tight_layout()
        save_fig(fig, f'fig6_component_impact_{ds}')


# ── Diagnostic figures (attention matrices + predictions) ────────────────────

def _create_data_args(dataset, horizon, window=20):
    cfg = DATASET_CONFIGS_FULL[dataset]
    return Namespace(dataset=dataset, sim_mat=cfg['sim_mat'], window=window,
                     horizon=horizon, train=0.6, val=0.2, test=0.2,
                     cuda=False, save_dir=MODEL_DIR, extra='', label='', pcc='')

def _create_model(dataset, loader, device, ablation='none'):
    cfg = DATASET_CONFIGS_FULL[dataset]
    model_args = Namespace(
        window=loader.P, horizon=loader.h, hidden_dim=32, kernel_size=3,
        bottleneck_dim=8, attention_heads=4, dropout=0.2,
        use_graph_bias=cfg['use_graph_bias'], use_adj_prior=cfg['use_adj_prior'],
        attention_regularization_weight=1e-5, num_scales=4, feature_channels=16,
        adaptive=False, ablation=ablation)
    class D:
        def __init__(s, m, adj): s.m, s.adj = m, adj
    return MSAGATNet_Ablation(model_args, D(loader.m, getattr(loader, 'adj', None))).to(device)

def _model_path(dataset, horizon, ablation, seed):
    cfg = DATASET_CONFIGS_FULL[dataset]
    name = f"MSAGAT-Net.{dataset}.w-{DEFAULT_WINDOW}.h-{horizon}.{ablation}.seed-{seed}"
    if cfg['use_adj_prior']:
        name += ".with_adj"
    return os.path.join(MODEL_DIR, name + ".pt")


def fig_diagnostic_matrices():
    print("\nGenerating diagnostic matrix visualisations...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rep = {'japan': 5, 'region785': 5, 'state360': 5,
           'australia-covid': 7, 'nhs_timeseries': 7, 'ltla_timeseries': 7}

    for dataset, horizons in DATASET_CONFIGS:
        horizon = rep.get(dataset, horizons[0])
        args = _create_data_args(dataset, horizon)
        loader = DataBasicLoader(args)
        path = _model_path(dataset, horizon, 'none', DEFAULT_SEED)
        if not os.path.exists(path):
            print(f"  Skipping {dataset}: model not found")
            continue
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        model = _create_model(dataset, loader, device)
        model.load_state_dict(state)
        model.eval()
        out_dir = os.path.join(BASE_DIR, 'report', 'figures', dataset)
        ensure_dir(out_dir)
        save_path = os.path.join(out_dir, f'matrices_{dataset}_h{horizon}.png')
        visualize_matrices(loader, model, save_path, device)
        print(f"  Saved: {save_path}")


def fig_diagnostic_predictions():
    print("\nGenerating diagnostic prediction plots...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rep = {'japan': 5, 'region785': 5, 'state360': 5,
           'australia-covid': 7, 'nhs_timeseries': 7, 'ltla_timeseries': 7}

    for dataset, horizons in DATASET_CONFIGS:
        horizon = rep.get(dataset, horizons[0])
        display = DATASET_NAMES.get(dataset, dataset)
        path = _model_path(dataset, horizon, 'none', DEFAULT_SEED)
        if not os.path.exists(path):
            print(f"  Skipping {dataset}: model not found")
            continue
        args = _create_data_args(dataset, horizon)
        loader = DataBasicLoader(args)
        ckpt = torch.load(path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        model = _create_model(dataset, loader, device)
        model.load_state_dict(state)
        model.eval()

        X_test = loader.test[0].clone().detach().to(device)
        idx = torch.arange(len(X_test)).to(device)
        with torch.no_grad():
            y_pred, _ = model(X_test, idx)
        y_pred = y_pred.cpu().numpy()
        y_test = loader.test[1].numpy() if isinstance(loader.test[1], torch.Tensor) else loader.test[1]
        if y_pred.ndim == 3: y_pred = y_pred[:, -1, :]
        if y_test.ndim == 3: y_test = y_test[:, -1, :]

        out_dir = os.path.join(BASE_DIR, 'report', 'figures', dataset)
        ensure_dir(out_dir)
        n_regions = min(6, loader.m)
        visualize_predictions(y_test, y_pred, os.path.join(out_dir, f'predictions_{dataset}_h{horizon}.png'),
                              regions=n_regions, dataset_name=f'{display} (H={horizon})')
        visualize_predictions_summary(y_test, y_pred, os.path.join(out_dir, f'predictions_summary_{dataset}_h{horizon}.png'),
                                       dataset_name=f'{display} (H={horizon})')
        print(f"  Saved predictions for {dataset}")


# ── Result aggregation ───────────────────────────────────────────────────────

AGG_METRICS = ['mae', 'rmse', 'pcc', 'R2']

def find_seed_results(dataset, horizon, ablation):
    csv_path = os.path.join(METRICS_DIR, dataset, 'all_results.csv')
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    return df[(df['horizon'] == horizon) & (df['ablation'] == ablation)]

def aggregate_metrics(df):
    result = {'n_seeds': len(df), 'seeds': list(df['seed'].values) if 'seed' in df.columns else []}
    for metric in AGG_METRICS:
        col = next((c for c in df.columns if c.lower() == metric.lower()), None)
        if col:
            vals = df[col].values
            result.update({f'{metric}_mean': np.mean(vals), f'{metric}_std': np.std(vals)})
    return result

def run_aggregation(datasets, horizons, ablations, fmt='markdown'):
    all_results = []
    for dataset in datasets:
        records = []
        for h in horizons:
            for ab in ablations:
                df = find_seed_results(dataset, h, ab)
                if df.empty or 'seed' not in df.columns or len(df) < 2:
                    continue
                agg = aggregate_metrics(df)
                agg.update(dataset=dataset, horizon=h, ablation=ab)
                records.append(agg)
        if not records:
            print(f"No multi-seed results for {dataset}")
            continue
        rdf = pd.DataFrame(records)
        all_results.append(rdf)

        if fmt in ('markdown', 'both'):
            print(f"\n## {dataset} (mean +/- std)")
            print("| Horizon | Ablation | RMSE | PCC | R2 | Seeds |")
            print("|---------|----------|------|-----|----|-------|")
            for _, r in rdf.iterrows():
                print(f"| {int(r['horizon'])} | {r['ablation']} "
                      f"| {r.get('rmse_mean',0):.2f}+/-{r.get('rmse_std',0):.2f} "
                      f"| {r.get('pcc_mean',0):.4f}+/-{r.get('pcc_std',0):.4f} "
                      f"| {r.get('R2_mean',0):.4f}+/-{r.get('R2_std',0):.4f} "
                      f"| {int(r['n_seeds'])} |")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out = os.path.join(METRICS_DIR, 'aggregated_multiseed_results.csv')
        combined.to_csv(out, index=False)
        print(f"\nSaved: {out}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='MSAGAT-Net Evaluation & Figures')
    parser.add_argument('--figures', action='store_true', help='Generate publication figures')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate multi-seed results')
    parser.add_argument('--datasets', nargs='+', default=list(DATASET_CONFIGS_FULL.keys()))
    parser.add_argument('--format', choices=['markdown', 'latex', 'both'], default='markdown')
    args = parser.parse_args()

    run_all = not args.figures and not args.aggregate

    if args.figures or run_all:
        ensure_dir(OUT_DIR)
        print("Generating publication figures...")
        fig_performance_vs_horizon()
        fig_ablation_study()
        fig_component_impact()
        fig_diagnostic_matrices()
        fig_diagnostic_predictions()

    if args.aggregate or run_all:
        print("\nAggregating results...")
        all_horizons = sorted(set(h for cfg in DATASET_CONFIGS_FULL.values() for h in cfg['horizons']))
        run_aggregation(args.datasets, all_horizons, ABLATIONS, args.format)


if __name__ == '__main__':
    main()
