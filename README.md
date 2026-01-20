# MSTAGAT-Net: Multi-Scale Temporal Adaptive Graph Attention Network

A lightweight spatiotemporal deep learning framework for epidemic forecasting across multiple geographical regions.

## Overview

MSTAGAT-Net combines efficient graph attention mechanisms for spatial dependencies with multi-scale temporal processing for accurate time-series forecasting. The model achieves state-of-the-art performance with only **~23.6K parameters**.

## Architecture

MSTAGAT-Net consists of three key components:

1. **LR-AGAM (Low-Rank Adaptive Graph Attention Module)**: Captures spatial dependencies between regions with linear complexity O(N)
2. **MTFM (Multi-Scale Temporal Fusion Module)**: Processes temporal patterns at different resolutions using dilated convolutions
3. **PPRM (Progressive Prediction Refinement Module)**: Enables region-aware multi-step forecasting with adaptive refinement

### Key Features

- **Linear-time Attention**: O(N) complexity using ELU+1 kernel trick
- **Low-Rank Factorization**: Parameter-efficient representations
- **Multi-Scale Processing**: Captures temporal patterns at multiple resolutions
- **Adaptive Graph Learning**: Optional adjacency prior for small graphs
- **Regularized Attention**: L1 regularization for interpretable patterns

## Installation

```bash
git clone https://github.com/yourusername/MSAGAT-Net.git
cd MSAGAT-Net
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (for visualization)

## Quick Start

### Single Training Run

```bash
python src/train.py \
  --dataset japan \
  --sim_mat japan-adj \
  --window 20 \
  --horizon 5 \
  --cuda --gpu 0 \
  --seed 42
```

### Run All Experiments

Run the complete experiment pipeline (7 datasets × 4 ablations × 5 seeds):

```bash
python src/run_full_pipeline.py
```

This will:
1. Train all model configurations
2. Generate ablation reports
3. Consolidate metrics
4. Create publication-ready figures

## Datasets

| Dataset | Regions | Horizons | Description |
|---------|---------|----------|-------------|
| Japan | 47 | 3, 5, 10, 15 | COVID-19 cases by prefecture |
| Australia | 8 | 3, 7, 14 | COVID-19 cases by state |
| Spain | 17 | 3, 7, 14 | COVID-19 cases by region |
| NHS | 7 | 3, 7, 14 | UK NHS regional data |
| LTLA | 307 | 3, 7, 14 | UK local authority data |
| US Region | 10 | 3, 5, 10, 15 | US regional influenza |
| US State | 49 | 3, 5, 10, 15 | US state-level influenza |

## Configuration

All settings are centralized in `src/config.py`.

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 32 | Hidden layer dimension |
| `attention_heads` | 4 | Number of attention heads |
| `num_scales` | 4 | Temporal scales in MTFM |
| `kernel_size` | 3 | Convolution kernel size |
| `bottleneck_dim` | 8 | Low-rank projection dimension |
| `feature_channels` | 16 | Feature extraction channels |
| `dropout` | 0.2 | Dropout rate |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 1500 | Maximum training epochs |
| `patience` | 100 | Early stopping patience |
| `lr` | 1e-3 | Learning rate |
| `weight_decay` | 5e-4 | L2 regularization |
| `batch` | 32 | Batch size |

### Graph Structure Options

```bash
# Use adjacency prior (recommended for small graphs < 20 nodes)
python src/train.py --dataset australia-covid --use_adj_prior --cuda --gpu 0

# Disable learnable graph bias (for very large graphs)
python src/train.py --dataset ltla_timeseries --no_graph_bias --cuda --gpu 0
```

**Optimal settings by dataset size:**
- **Small graphs** (< 20 nodes): `--use_adj_prior` (Australia, NHS, Spain, US Region)
- **Large graphs** (> 40 nodes): Pure learned attention (Japan, US State, LTLA)

## Ablation Studies

Run ablation experiments to evaluate component contributions:

```bash
# Full model
python src/train.py --dataset japan --ablation none --cuda --gpu 0

# Without LR-AGAM (spatial attention)
python src/train.py --dataset japan --ablation no_agam --cuda --gpu 0

# Without MTFM (multi-scale temporal)
python src/train.py --dataset japan --ablation no_mtfm --cuda --gpu 0

# Without PPRM (progressive prediction)
python src/train.py --dataset japan --ablation no_pprm --cuda --gpu 0
```

## Project Structure

```
MSAGAT-Net/
├── data/                    # Datasets and adjacency matrices
├── src/
│   ├── config.py           # Central configuration
│   ├── model.py            # MSTAGAT-Net architecture
│   ├── ablation.py         # Ablation model variants
│   ├── data.py             # Data loading utilities
│   ├── utils.py            # Helper functions
│   ├── train.py            # Training script
│   ├── run_experiments.py  # Batch experiment runner
│   ├── run_full_pipeline.py # Master pipeline script
│   ├── consolidate_metrics.py # Results aggregation
│   └── generate_visualizations.py # Figure generation
├── save_all/               # Saved model checkpoints
├── report/
│   ├── results/            # Metrics and reports
│   └── figures/            # Generated visualizations
├── logs/                   # Training logs
└── tensorboard/            # TensorBoard logs
```

## Output Files

After running experiments:

- **Models**: `save_all/MSTAGAT-Net.{dataset}.w-{window}.h-{horizon}.{ablation}.seed-{seed}.pt`
- **Metrics**: `report/results/final_metrics_*.csv`
- **Reports**: `report/results/ablation_report_*.txt`
- **Consolidated**: `report/results/all_results.csv`, `all_ablation_summary.csv`
- **Figures**: `report/figures/paper/` (publication-ready)
- **Diagnostics**: `report/figures/{dataset}/` (per-dataset visualizations)

## Visualization

Generate publication-ready figures:

```bash
python src/generate_visualizations.py
```

Outputs include:
- Performance vs. horizon plots
- Ablation study bar charts
- Component importance heatmaps
- Attention matrix visualizations
- Prediction comparisons

## Citation

```bibtex
@article{mstagat2026,
  title={MSTAGAT-Net: Multi-Scale Temporal Adaptive Graph Attention Network for Epidemic Forecasting},
  author={Your Name},
  journal={Journal Name},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.