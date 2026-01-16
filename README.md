# MSAGAT-Net: Multi-Scale Adaptive Graph Attention Network for Epidemic Forecasting

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A computationally efficient spatiotemporal deep learning framework for epidemic forecasting that achieves **O(N) linear complexity** while maintaining state-of-the-art accuracy across diverse epidemic scenarios including influenza, COVID-19, and ICU bed occupancy.

---

## 📖 Overview

MSAGAT-Net addresses three fundamental challenges in spatiotemporal epidemic forecasting:
1. **Computational scalability** - Quadratic O(N²) attention complexity limits real-world deployment
2. **Multi-scale temporal dynamics** - Epidemics exhibit patterns from daily fluctuations to seasonal waves  
3. **Multi-horizon forecast stability** - Error accumulation degrades long-range predictions

**Key Innovation**: Reduces graph attention complexity from **O(N²) to O(N)** using linearised attention with low-rank projections, enabling real-time surveillance across thousands of regions.

### Publication

**MSAGAT-Net: Multi-Scale Temporal Adaptive Graph Attention for Efficient Spatiotemporal Epidemic Forecasting**  
*Under Review, 2026*

---

## 🏗️ Architecture

MSAGAT-Net integrates four core components:

### 1. **Efficient Feature Extraction**
- Depthwise separable convolutions for parameter efficiency
- Low-rank bottleneck projections (d → d_bottle → d_hidden)

### 2. **Adaptive Graph Attention Module (AGAM)**
- **Linear O(N) complexity** via ELU+1 kernel trick and low-rank factorization
- **Novel graph bias message passing**: Learnable spatial structure (U⊗V) integrated directly into forward computation
- Multi-head attention with learnable L1 regularization for sparsity

### 3. **Multi-Scale Temporal Feature Module (MTFM)**
- Parallel dilated convolutions at 3 scales (dilation rates: 1, 2, 4)
- Adaptive scale fusion with learnable weights
- Captures short-term fluctuations and long-term trends simultaneously

### 4. **Progressive Prediction Refinement Module (PPRM)**
- Stabilises multi-horizon forecasts by combining model predictions with trend extrapolation
- Residual connections prevent gradient degradation

---

## 📊 Datasets

The repository includes **7 epidemic forecasting benchmarks** (2 novel):

| Dataset | Description | Regions | Time Series | Adjacency |
|---------|-------------|---------|-------------|-----------|
| **Japan** | Prefecture-level influenza | 47 | `japan.txt` | `japan-adj.txt` |
| **Region785** | US regional data | 785 | `region785.txt` | `region-adj.txt` |
| **State360** | US state-level | 50 | `state360.txt` | `state-adj-49.txt` |
| **Australia COVID** | State-level COVID-19 | 8 | `australia-covid.txt` | `australia-adj.txt` |
| **Spain COVID** | Regional COVID-19 | 17 | `spain-covid.txt` | `spain-adj.txt` |
| **LTLA-COVID** 🆕 | UK local authority COVID-19 | 317 | `ltla_timeseries.txt` | `ltla-adj.txt` |
| **NHS-ICUBeds** 🆕 | England ICU bed occupancy | 142 | `nhs_timeseries.txt` | `nhs-adj.txt` |

All datasets located in [`data/`](data/)

---

## 🚀 Installation

### Prerequisites
- Python 3.11+
- CUDA 11.8+ (for GPU acceleration)
- Conda (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/michaelajao/MSAGAT-Net.git
cd MSAGAT-Net

# Create environment
conda create -n dl_env python=3.11
conda activate dl_env

# Install PyTorch (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install numpy pandas scipy scikit-learn matplotlib seaborn optuna tensorboard
```

---

## 🎯 Usage

### 1. Train Single Model

```bash
python src/scripts/train.py \
  --dataset japan \
  --sim_mat japan-adj \
  --window 20 \
  --horizon 5 \
  --model msagat \
  --hidden_dim 32 \
  --attention_heads 4 \
  --bottleneck_dim 8 \
  --num_scales 3 \
  --epochs 1500 \
  --batch 32 \
  --lr 1e-3 \
  --patience 100 \
  --cuda --gpu 0 \
  --save_dir save_all \
  --mylog
```

**Key arguments:**
- `--dataset`: Dataset name (japan, region785, ltla_timeseries, etc.)
- `--sim_mat`: Adjacency matrix file (without .txt extension)
- `--window`: Lookback window size (default: 20)
- `--horizon`: Forecast horizon (3, 5, 7, 10, 14, or 15 days)
- `--ablation`: Ablation variant (none, no_agam, no_mtfm, no_pprm)

### 2. Run Ablation Studies

```bash
# Full model
python src/scripts/train.py --dataset japan --ablation none --horizon 3

# Without AGAM (spatial attention)
python src/scripts/train.py --dataset japan --ablation no_agam --horizon 3

# Without MTFM (multi-scale temporal)
python src/scripts/train.py --dataset japan --ablation no_mtfm --horizon 3

# Without PPRM (progressive refinement)
python src/scripts/train.py --dataset japan --ablation no_pprm --horizon 3
```

### 3. Batch Experiments

Run comprehensive experiments across all datasets, horizons, and ablations:

```bash
python src/scripts/run_experiments.py
```

This executes **~500 training runs** (7 datasets × multiple horizons × 4 ablations × 5 random seeds) and automatically skips completed experiments.

**Filtering options:****
```bash
# Run only specific datasets
python src/scripts/run_experiments.py --datasets japan ltla_timeseries

# Run only specific ablations
python src/scripts/run_experiments.py --ablations none no_agam
```

### 4. Generate Figures

Create publication-ready visualizations:

```bash
conda activate dl_env
cd src/scripts
python generate_figures.py
```

Generates comprehensive analysis visualizations including performance comparisons, ablation studies, and component importance heatmaps.

---

## 📁 Project Structure

```
MSAGAT-Net/
├── data/                          # Epidemic time series + adjacency matrices
│   ├── japan.txt, japan-adj.txt
│   ├── ltla_timeseries.txt, ltla-adj.txt
│   └── ...
├── src/
│   ├── models.py                  # MSAGAT-Net architecture + ablation variants
│   ├── data.py                    # DataLoader, preprocessing, train/val/test splits
│   ├── training.py                # Training loop, early stopping, checkpointing
│   ├── utils.py                   # Metrics (RMSE, MAE, PCC, R²)
│   └── scripts/
│       ├── train.py               # Single experiment training script
│       ├── run_experiments.py     # Batch experiment runner
│       ├── aggregate_results.py   # Consolidate results across runs
│       └── generate_figures.py    # Publication figure generation
├── save_all/                      # Trained model checkpoints (.pt files)
├── report/
│   ├── results/                   # Per-dataset aggregated results
│   │   ├── japan/
│   │   │   ├── all_results.csv           # All runs (seed-level)
│   │   │   └── all_ablation_summary.csv  # Mean across seeds
│   │   └── ...
│   └── figures/paper/             # Generated publication figures (.png)
└── README.md                      # This file
```

---

## 🔬 Model Hyperparameters

**Default Configuration** (empirically optimized):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim` | 32 | Hidden feature dimension |
| `attention_heads` | 4 | Number of attention heads |
| `bottleneck_dim` | 8 | Low-rank projection bottleneck |
| `num_scales` | 3 | Temporal scales (dilations: 1,2,4) |
| `kernel_size` | 3 | Convolution kernel size |
| `feature_channels` | 16 | Feature extractor output channels |
| `dropout` | 0.2 | Dropout probability |
| `attention_regularization_weight` | 1e-5 | L1 regularization on graph bias (learnable) |
| `lr` | 1e-3 | Learning rate |
| `weight_decay` | 5e-4 | L2 weight decay |
| `batch_size` | 32 | Training batch size |
| `patience` | 100 | Early stopping patience |

---

## 📊 Evaluation Metrics

All experiments track:
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error  
- **PCC**: Pearson Correlation Coefficient
- **R²**: Coefficient of Determination

---

## 📝 Citation

If you use MSAGAT-Net in your research, please cite:

```bibtex
@article{ajaoolarinoye2026msagat,
  title={MSAGAT-Net: Multi-Scale Temporal Adaptive Graph Attention for Efficient Spatiotemporal Epidemic Forecasting},
  author={Ajao-Olarinoye, Michael and others},
  journal={Under Review},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License.

<!-- ---

## 🙏 Acknowledgments

This research was supported by Coventry University. We thank the public health authorities for providing open epidemic surveillance data. -->

<!-- ---

## 📧 Contact

For questions or collaborations:
- **GitHub Issues**: [https://github.com/michaelajao/MSAGAT-Net/issues](https://github.com/michaelajao/MSAGAT-Net/issues)
- **Email**: See paper for contact details -->