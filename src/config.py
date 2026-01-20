"""
MSAGAT-Net Configuration File

Central configuration for all experiments, ensuring consistency across
training, evaluation, and visualization scripts.
"""

# =============================================================================
# OPTIMAL CONFIGURATIONS PER DATASET (from ablation studies)
# =============================================================================

DATASET_CONFIGS = {
    'japan': {
        'sim_mat': 'japan-adj',
        'num_nodes': 47,
        'horizons': [3, 5, 10, 15],
        'use_adj_prior': False,
        'use_graph_bias': False,
        'notes': 'Large graph - pure learned attention (best performance)'
    },
    'australia-covid': {
        'sim_mat': 'australia-adj',
        'num_nodes': 8,
        'horizons': [3, 7, 14],
        'use_adj_prior': True,
        'use_graph_bias': True,
        'notes': 'Small graph - adjacency prior helps'
    },
    'nhs_timeseries': {
        'sim_mat': 'nhs-adj',
        'num_nodes': 7,
        'horizons': [3, 7, 14],
        'use_adj_prior': True,
        'use_graph_bias': True,
        'notes': 'Small graph - adjacency prior helps'
    },
    'spain-covid': {
        'sim_mat': 'spain-adj',
        'num_nodes': 17,
        'horizons': [3, 7, 14],
        'use_adj_prior': True,
        'use_graph_bias': True,
        'notes': 'Medium graph - adj helps for longer horizons'
    },
    'ltla_timeseries': {
        'sim_mat': 'ltla-adj',
        'num_nodes': 307,
        'horizons': [3, 7, 14],
        'use_adj_prior': False,
        'use_graph_bias': False,
        'notes': 'Very large graph - learned attention only'
    },
    'region785': {
        'sim_mat': 'region-adj',
        'num_nodes': 10,
        'horizons': [3, 5, 10, 15],
        'use_adj_prior': True,
        'use_graph_bias': True,
        'notes': 'Small graph - use adjacency prior'
    },
    'state360': {
        'sim_mat': 'state-adj-49',
        'num_nodes': 49,
        'horizons': [3, 5, 10, 15],
        'use_adj_prior': False,
        'use_graph_bias': False,
        'notes': 'Large graph - learned attention only'
    },
}

# =============================================================================
# TRAINING HYPERPARAMETERS (Optimal from experiments)
# =============================================================================

TRAIN_CONFIG = {
    'epochs': 1500,
    'patience': 100,
    'lr': 1e-3,
    'weight_decay': 5e-4,
    'batch': 32,
    'window': 20,
    'dropout': 0.2,
}

# =============================================================================
# MODEL HYPERPARAMETERS (Optimal from experiments)
# =============================================================================

MODEL_CONFIG = {
    'hidden_dim': 32,
    'attention_heads': 4,
    'num_scales': 4,
    'kernel_size': 3,
    'feature_channels': 16,
    'bottleneck_dim': 8,
    'attention_regularization_weight': 1e-5,
}

# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

# Seeds for reproducibility and statistical robustness
SEEDS = [5, 30, 45, 123, 1000]

# First seed used for visualization (to avoid too many figures)
FIRST_SEED = SEEDS[0]  # 5

# Ablation configurations (no_mtfm kept for backward compatibility but MSSFM removed from main model)
ABLATIONS = ['none', 'no_agam', 'no_pprm']

# Default window size
DEFAULT_WINDOW = TRAIN_CONFIG['window']

# =============================================================================
# ABLATION DESCRIPTIONS
# =============================================================================

ABLATION_DESCRIPTIONS = {
    'none': 'Full model with all components',
    'no_agam': 'Without Low-rank Adaptive Graph Attention Module (LR-AGAM)',
    'no_pprm': 'Without GRU-based Progressive Prediction Refinement Module (PPRM)',
}

# Module full names for documentation
MODULE_FULL_NAMES = {
    'tfem': 'Temporal Feature Extraction Module (TFEM)',
    'agam': 'Low-rank Adaptive Graph Attention Module (LR-AGAM)', 
    'pprm': 'GRU-based Progressive Prediction Refinement Module (PPRM)',
}

# =============================================================================
# DISPLAY NAMES (for publications/reports)
# =============================================================================

DATASET_DISPLAY_NAMES = {
    'japan': 'Japan',
    'region785': 'US Region',
    'state360': 'US State',
    'nhs_timeseries': 'NHS',
    'ltla_timeseries': 'LTLA',
    'australia-covid': 'Australia',
    'spain-covid': 'Spain',
}

COMPONENT_DISPLAY_NAMES = {
    'agam': 'LR-AGAM',
    'mtfm': 'MSSFM',
    'pprm': 'PPRM',
}

METRIC_DISPLAY_NAMES = {
    'rmse': 'RMSE',
    'mae': 'MAE',
    'pcc': 'PCC',
    'r2': 'R²',
    'mape': 'MAPE',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_dataset_list():
    """Return list of (dataset, sim_mat, horizons) tuples for iteration."""
    return [
        (name, cfg['sim_mat'], cfg['horizons'])
        for name, cfg in DATASET_CONFIGS.items()
    ]

def get_dataset_config(dataset: str) -> dict:
    """Get configuration for a specific dataset."""
    return DATASET_CONFIGS.get(dataset, {})

def get_sim_mat(dataset: str) -> str:
    """Get similarity matrix name for a dataset."""
    return DATASET_CONFIGS.get(dataset, {}).get('sim_mat', f'{dataset}-adj')

def get_horizons(dataset: str) -> list:
    """Get prediction horizons for a dataset."""
    return DATASET_CONFIGS.get(dataset, {}).get('horizons', [3, 5, 10, 15])
