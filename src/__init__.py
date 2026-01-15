"""
MSAGAT-Net: Multi-Scale Temporal-Adaptive Graph Attention Network

A deep learning framework for spatio-temporal forecasting using graph attention
mechanisms and multi-scale temporal convolutions.

Package Structure:
    src/
    ├── __init__.py      # This file - package exports
    ├── models.py        # Neural network architectures  
    ├── data.py          # Data loading and preprocessing
    ├── training.py      # Training loops and evaluation
    ├── utils.py         # Visualization and utilities
    ├── optimize.py      # Hyperparameter optimization
    └── scripts/         # CLI entry points
        ├── train.py           # Training script
        ├── run_experiments.py # Batch experiments
        └── generate_figures.py # Publication figures

Usage:
    # As a package
    from src import MSTAGAT_Net, DataBasicLoader, Trainer
    
    # CLI training
    python -m src.scripts.train --dataset japan --horizon 5
    
    # Batch experiments
    python -m src.scripts.run_experiments --datasets japan region785
    
    # Generate figures
    python -m src.scripts.generate_figures
    
    # Hyperparameter optimization  
    python -m src.optimize --dataset japan --trials 50
"""

# Lazy imports to avoid circular dependencies and speed up import time
__version__ = "1.0.0"
__author__ = "MSAGAT-Net Team"

__all__ = [
    # Models
    "MSTAGAT_Net",
    "MSAGATNet_Ablation",
    "SpatialAttentionModule",
    "MultiScaleTemporalModule",
    "HorizonPredictor",
    "DepthwiseSeparableConv1D",
    # Ablation components
    "SimpleGraphConvolutionalLayer",
    "SingleScaleTemporalModule", 
    "DirectPredictionModule",
    # Data
    "DataBasicLoader",
    # Training
    "train_epoch",
    "evaluate",
    "Trainer",
    "TrainingConfig",
    "MetricsResult",
    # Utils
    "get_laplace_matrix",
    "peak_error",
    "visualize_matrices",
    "visualize_predictions",
    "plot_loss_curves",
    "save_metrics",
    "setup_visualization_style",
    # Constants
    "HIDDEN_DIM",
    "ATTENTION_HEADS",
    "DROPOUT",
    "NUM_TEMPORAL_SCALES",
    "KERNEL_SIZE",
    "FEATURE_CHANNELS",
    "BOTTLENECK_DIM",
]


def __getattr__(name):
    """Lazy loading of module attributes."""
    if name in (
        "MSTAGAT_Net", "MSAGATNet_Ablation", "SpatialAttentionModule",
        "MultiScaleTemporalModule", "HorizonPredictor", "DepthwiseSeparableConv1D",
        "SimpleGraphConvolutionalLayer", "SingleScaleTemporalModule", "DirectPredictionModule",
        "HIDDEN_DIM", "ATTENTION_HEADS", "DROPOUT", "NUM_TEMPORAL_SCALES",
        "KERNEL_SIZE", "FEATURE_CHANNELS", "BOTTLENECK_DIM"
    ):
        from . import models
        return getattr(models, name)
    
    elif name == "DataBasicLoader":
        from .data import DataBasicLoader
        return DataBasicLoader
    
    elif name in ("train_epoch", "evaluate", "Trainer", "TrainingConfig", "MetricsResult"):
        from . import training
        return getattr(training, name)
    
    elif name in (
        "get_laplace_matrix", "peak_error", "visualize_matrices",
        "visualize_predictions", "plot_loss_curves", "save_metrics",
        "setup_visualization_style"
    ):
        from . import utils
        return getattr(utils, name)
    
    raise AttributeError(f"module 'src' has no attribute '{name}'")
