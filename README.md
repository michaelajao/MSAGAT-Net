# MSAGAT-Net: Multi-Scale Adaptive Graph Attention Network for Epidemic Forecasting

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A computationally efficient spatiotemporal deep learning framework for epidemic forecasting that achieves **O(N) linear complexity** while maintaining state-of-the-art accuracy across diverse epidemic scenarios including influenza, COVID-19, and ICU bed occupancy.

---

## Architecture

MSAGAT-Net integrates four core components:

### 1. Temporal Feature Extraction Module (TFEM)
- Depthwise separable convolutions for parameter efficiency
- Low-rank bottleneck projections (d → d_bottle → d_hidden)

### 2. Efficient Adaptive Graph Attention Module (EAGAM)
- Scaled dot-product softmax attention with low-rank QKV projections
- Additive structural bias: learnable low-rank graph bias (U@V) added directly to attention scores
- Optional adjacency prior with learnable scale for soft, self-regulating structure guidance
- L1 regularisation on attention weights to promote sparsity

### 3. Multi-Scale Spatial Feature Module (MSSFM)
- Multi-hop graph convolutions using powers of the normalised adjacency
- Adaptive hop depth to prevent oversmoothing on small graphs
- Locality-biased fusion weights to blend multi-hop features

### 4. Progressive Prediction Refinement Module (PPRM)
- Learnable decay for trend extrapolation with adaptive refinement gating
- Highway connection blends model forecasts with recent history

---

## Datasets

The repository includes **6 epidemic forecasting benchmarks**:

| Dataset | Description | Regions | Time Series | Adjacency |
|---------|-------------|---------|-------------|-----------|
| **Japan** | Prefecture-level influenza | 47 | `japan.txt` | `japan-adj.txt` |
| **Region785** | US regional data | 10 | `region785.txt` | `region-adj.txt` |
| **State360** | US state-level | 49 | `state360.txt` | `state-adj-49.txt` |
| **Australia COVID** | State-level COVID-19 | 8 | `australia-covid.txt` | `australia-adj.txt` |
| **LTLA-COVID** | UK local authority COVID-19 | 372 | `ltla_timeseries.txt` | `ltla-adj.txt` |
| **NHS-ICUBeds** | England ICU bed occupancy | 7 | `nhs_timeseries.txt` | `nhs-adj.txt` |

All datasets are located in `data/`.

---

## Usage

### Train a single model

```bash
python -m src.train --single --dataset japan --horizon 5 --seed 42
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `japan` | Dataset name |
| `--horizon` | `5` | Forecast horizon in days |
| `--seed` | `42` | Random seed |
| `--ablation` | `none` | Ablation variant (`none`, `no_agam`, `no_mtfm`, `no_pprm`) |
| `--save_dir` | `save_all` | Directory for model checkpoints |
| `--cpu` | — | Force CPU training |

### Run ablation studies

```bash
python -m src.train --single --dataset japan --ablation no_agam --horizon 7 --seed 42
python -m src.train --single --dataset japan --ablation no_mtfm --horizon 7 --seed 42
python -m src.train --single --dataset japan --ablation no_pprm --horizon 7 --seed 42
```

### Generate figures

```bash
# All publication figures and diagnostic plots
python -m src.evaluate

# Figures only
python -m src.evaluate --figures

# Aggregate multi-seed results
python -m src.evaluate --aggregate

# Aggregate in LaTeX table format
python -m src.evaluate --aggregate --format latex
```

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim` | 32 | Hidden feature dimension |
| `attention_heads` | 4 | Number of attention heads |
| `bottleneck_dim` | 8 | Low-rank projection bottleneck |
| `num_scales` | 4 | Spatial scales (hop depths 1, 2, 4, 8) |
| `kernel_size` | 3 | Convolution kernel size |
| `feature_channels` | 16 | Feature extractor output channels |
| `dropout` | 0.2 | Dropout probability |
| `lr` | 1e-3 | Learning rate |
| `weight_decay` | 5e-4 | L2 weight decay |
| `batch_size` | 32 | Training batch size |
| `patience` | 100 | Early stopping patience |
| `window` | 20 | Lookback window (days) |

---

## Metrics

All evaluations report:
- **RMSE** — Root Mean Squared Error
- **MAE** — Mean Absolute Error
- **PCC** — Pearson Correlation Coefficient
- **R²** — Coefficient of Determination

---

## Citation

```bibtex
@article{ajaoolarinoye2026msagat,
  title={MSAGAT-Net: Multi-Scale Adaptive Graph Attention Network for
         Efficient Spatiotemporal Epidemic Forecasting},
  author={Ajao-Olarinoye, Michael and others},
  journal={Under Review},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License.
