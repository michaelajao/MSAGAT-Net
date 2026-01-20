"""
MSAGAT-Net: Multi-Scale Temporal-Adaptive Graph Attention Network

A deep learning framework for spatio-temporal forecasting using graph attention
mechanisms and multi-scale spatial convolutions.

Package Structure:
    src/
    ├── __init__.py      # This file
    ├── models.py        # Neural network architectures  
    ├── data.py          # Data loading and preprocessing
    ├── training.py      # Training loops and evaluation
    ├── utils.py         # Visualization and utilities
    ├── optimize.py      # Hyperparameter optimization
    └── scripts/         # CLI entry points
        ├── experiments.py      # Batch experiments
        ├── generate_figures.py # Publication figures
        └── aggregate_results.py # Results aggregation

Usage:
    # As a package
    from src.models import MSTAGAT_Net
    from src.data import DataBasicLoader
    from src.training import Trainer
    
    # Batch experiments
    python -m src.scripts.experiments --datasets japan region785
    
    # Generate figures
    python -m src.scripts.generate_figures
    
    # Hyperparameter optimization  
    python -m src.optimize --dataset japan --trials 50
"""

__version__ = "1.0.0"
