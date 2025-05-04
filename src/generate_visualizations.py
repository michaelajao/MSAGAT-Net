import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
METRICS_DIR = os.path.join(BASE_DIR, 'report', 'results')
OUT_DIR = os.path.join(BASE_DIR, 'report', 'paper_figures')
DEFAULT_WINDOW = 20

# Dataset configurations: (name, horizons)
DATASET_CONFIGS = [
    ('japan', [3, 5, 10, 15]),
    ('region785', [3, 5, 10, 15]),
    ('nhs_timeseries', [3, 7, 14]),
    ('ltla_timeseries', [3, 7, 14]),
]

# Ablation variants
ABLATIONS = ['none', 'no_agam', 'no_mtfm', 'no_pprm']
# Metrics
METRICS = ['rmse', 'mae', 'pcc', 'r2']
METRIC_NAMES = {
    'rmse': 'Root Mean Square Error',
    'mae': 'Mean Absolute Error',
    'pcc': 'Pearson Correlation Coefficient',
    'r2':  'R-squared'
}

# Component full names for legends
COMP_FULL = {
    'agam': 'Low-Rank Adaptive Graph Attention Module',
    'mtfm': 'Multi-scale Temporal Fusion Module',
    'pprm': 'Progressive Prediction Refinement Module'
}

# =============================================================================
# UTILITIES
# =============================================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize(name: str) -> str:
    return name.strip().lower().replace('-', '').replace('_', '')


def get_metric(df: pd.DataFrame, key: str) -> float:
    search = normalize(key)
    for col in df.columns:
        if normalize(col) == search:
            return df[col].iloc[0]
    return float('nan')


def find_metrics_file(dataset: str, horizon: int, ablation: str) -> str:
    patterns = [
        f"final_metrics_*{dataset}*w-{DEFAULT_WINDOW}*h-{horizon}*{ablation}.csv",
        f"final_metrics_*{dataset}*horizon-{horizon}*{ablation}.csv",
    ]
    for pat in patterns:
        matches = glob.glob(os.path.join(METRICS_DIR, pat))
        if matches:
            return matches[0]
    return None


def load_metrics(dataset: str, horizon: int, ablation: str) -> pd.DataFrame:
    path = find_metrics_file(dataset, horizon, ablation)
    return pd.read_csv(path) if path and os.path.exists(path) else None


def load_summary(dataset: str, horizon: int) -> pd.DataFrame:
    path = os.path.join(METRICS_DIR, f"ablation_summary_{dataset}.w-{DEFAULT_WINDOW}.h-{horizon}.csv")
    return pd.read_csv(path, index_col=0) if os.path.exists(path) else None

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def setup_style():
    sns.set_style('whitegrid')
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.autolayout'] = True


def save_fig(fig, name: str):
    ensure_dir(OUT_DIR)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches='tight')


def performance_table():
    rows = []
    for ds, horizons in DATASET_CONFIGS:
        for h in horizons:
            df = load_metrics(ds, h, 'none')
            if df is None: continue
            rows.append({
                'Dataset': ds,
                'Horizon': h,
                **{METRIC_NAMES[m]: get_metric(df, m) for m in METRICS}
            })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(OUT_DIR, 'performance_table.csv'), index=False)
    return table


def cross_horizon_performance():
    setup_style()
    for m in ['rmse', 'pcc']:
        fig, ax = plt.subplots()
        for ds, horizons in DATASET_CONFIGS:
            vals, hs = [], []
            for h in horizons:
                df = load_metrics(ds, h, 'none')
                if df is None: continue
                val = get_metric(df, m)
                if not np.isnan(val): vals.append(val); hs.append(h)
            if vals: ax.plot(hs, vals, 'o-', label=ds)
        ax.set_title(f"{METRIC_NAMES[m]} vs Horizon")
        ax.set_xlabel('Horizon')
        ax.set_ylabel(METRIC_NAMES[m])
        ax.legend()
        save_fig(fig, f"{m}_vs_horizon")
        plt.close(fig)


def component_importance_comparison():
    setup_style()
    data = []
    for ds, horizons in DATASET_CONFIGS:
        for h in horizons:
            sum_df = load_summary(ds, h)
            if sum_df is None: continue
            for ab in ABLATIONS[1:]:
                col = 'RMSE_CHANGE'
                if col in sum_df.columns and ab in sum_df.index:
                    comp = ab.replace('no_', '')
                    data.append({'Dataset': ds, 'Horizon': h,
                                 'Component': COMP_FULL[comp],
                                 'Importance': abs(sum_df.loc[ab, col])})
    df = pd.DataFrame(data)
    fig, ax = plt.subplots()
    sns.barplot(x='Horizon', y='Importance', hue='Component', data=df, ax=ax)
    ax.set_ylabel('% RMSE Change')
    ax.set_title('Component Importance Across Horizons')
    save_fig(fig, 'component_importance_comparison')
    plt.close(fig)


def single_horizon_metrics():
    setup_style()
    for ds, horizons in DATASET_CONFIGS:
        for h in horizons:
            fig, ax = plt.subplots()
            vals, labels = [], []
            for ab in ABLATIONS:
                df = load_metrics(ds, h, ab)
                if df is None: continue
                val = get_metric(df, 'rmse')
                if not np.isnan(val): vals.append(val); labels.append(ab)
            if vals:
                ax.bar(labels, vals)
                ax.set_title(f'RMSE Comparison {ds} h={h}')
                save_fig(fig, f'rmse_comp_{ds}_h{h}')
            plt.close(fig)


def component_contribution_charts():
    setup_style()
    metrics = ['rmse', 'pcc']
    for m in metrics:
        data = []
        for ds, horizons in DATASET_CONFIGS:
            for h in horizons:
                sum_df = load_summary(ds, h)
                if sum_df is None: continue
                for ab in ABLATIONS[1:]:
                    col = f'{m.upper()}_CHANGE'
                    if col in sum_df.columns and ab in sum_df.index:
                        comp = ab.replace('no_', '')
                        data.append({'Dataset': ds, 'Component': COMP_FULL[comp], 'Contribution': abs(sum_df.loc[ab,col])})
        df = pd.DataFrame(data)
        fig, ax = plt.subplots()
        sns.barplot(x='Component', y='Contribution', hue='Dataset', data=df, ax=ax)
        ax.set_title(f'Component Contribution to {METRIC_NAMES[m]}')
        save_fig(fig, f'component_contribution_{m}')
        plt.close(fig)

# =============================================================================
# MAIN
# =============================================================================

def main():
    ensure_dir(OUT_DIR)
    performance_table()
    cross_horizon_performance()
    component_importance_comparison()
    single_horizon_metrics()
    component_contribution_charts()

if __name__ == '__main__':
    main()
