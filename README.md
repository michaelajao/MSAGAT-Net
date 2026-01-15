# MSAGAT-Net: Multi-Scale Temporal-Adaptive Graph Attention Network

A sophisticated spatiotemporal deep learning framework designed for epidemic forecasting across multiple geographical regions. MSAGAT-Net combines efficient graph attention mechanisms with multi-scale temporal processing to achieve state-of-the-art forecasting accuracy with linear computational complexity.

## 🚀 Key Features

- **Linear Attention Complexity**: O(N) instead of O(N²) using ELU+1 kernel trick
- **Low-Rank Factorization**: Efficient parameter usage with factorized attention
- **Multi-Scale Temporal Processing**: Captures patterns at different time scales via dilated convolutions
- **Learnable Regularization**: Adaptive L1 regularization weight for graph structure
- **Automated Hyperparameter Optimization**: Built-in Optuna-based optimization
- **Comprehensive Evaluation**: Extensive metrics and visualization tools
- **Production Ready**: Clean, modular code with extensive documentation

## 🏗️ Model Architecture

MSAGAT-Net consists of three key components:

1. **Adaptive Graph Attention Module (AGAM)**
   - Low-rank factorized attention with learnable graph bias (U⊗V)
   - Linear complexity O(N) using ELU+1 kernel linearization
   - Integrated adjacency information via O(N) message passing
   - Learnable regularization weight for graph structure sparsity

2. **Multi-Scale Temporal Fusion Module (MTFM)**
   - Dilated convolutions at multiple scales (dilation rates: 1, 2, 4)
   - Low-rank temporal fusion for efficient multi-scale integration
   - Captures both short-term and long-term temporal dependencies

3. **Progressive Prediction Refinement Module (PPRM)**
   - Multi-step forecasting with iterative refinement
   - Skip connections from input to output
   - Fixed decay rate for prediction horizon weighting

## 📊 Performance Highlights

Our experiments across multiple epidemic datasets demonstrate:

- **State-of-the-art Accuracy**: R² = 0.7284, RMSE = 1073.91, PCC = 0.8560 on Japan COVID-19 (Horizon=5)
- **Linear Complexity**: O(N) attention mechanism enables scalability to large graphs
- **Parameter Efficiency**: Low-rank factorization reduces parameters while maintaining accuracy
- **Strong Correlation**: High PCC and R² scores indicate excellent capture of epidemic trends
- **Computational Efficiency**: Significantly faster than quadratic attention methods

## 📊 Available Datasets

The repository includes multiple epidemic forecasting datasets:

| Dataset | Description | Nodes | File | Adjacency |
|---------|-------------|-------|------|-----------|
| Japan COVID-19 | Prefecture-level data | 47 | `japan.txt` | `japan-adj.txt` |
| Australia COVID-19 | State-level data | 8 | `australia-covid.txt` | `australia-adj.txt` |
| Spain COVID-19 | Regional data | 17 | `spain-covid.txt` | `spain-adj.txt` |
| UK NHS | Regional health data | 142 | `nhs_timeseries.txt` | `nhs-adj.txt` |
| US States | State-level data | 49/50 | `state360.txt` | `state-adj-49.txt` |
| US Regions | Regional data | 785 | `region785.txt` | `region-adj.txt` |
| LTLA | Local authority data | - | `ltla_timeseries.txt` | `ltla-adj.txt` |

## 🔧 Model Configuration

### Core Hyperparameters

| Parameter | Description | Default | Optimized Range |
|-----------|-------------|---------|-----------------|
| `--hidden_dim` | Hidden dimension size | 32 | [16, 32, 64] |
| `--attention_heads` | Number of attention heads | 4 | [2, 4, 8] |
| `--bottleneck_dim` | Low-rank projection dimension | 8 | [4, 8, 16, 32] |
| `--num_scales` | Temporal scales count | 4 | [2, 6] |
| `--kernel_size` | Convolution kernel size | 3 | [3, 5, 7] |
| `--feature_channels` | Feature extractor channels | 16 | [8, 16, 32] |
| `--dropout` | Dropout probability | 0.2 | [0.1, 0.5] |
| `--attention_regularization_weight` | Attention L1 regularization | 1e-5 | [1e-6, 1e-4] |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--lr` | Learning rate | 1e-3 |
| `--weight_decay` | L2 regularization | 5e-4 |
| `--batch` | Batch size | 32 |
| `--epochs` | Maximum epochs | 1500 |
| `--patience` | Early stopping patience | 100 |
| `--window` | Input sequence length | 20 |
| `--horizon` | Prediction horizon | 5 |

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/MSAGAT-Net.git
cd MSAGAT-Net

# Create conda environment
conda create -n msagat python=3.11
conda activate msagat

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
pip install optuna scikit-learn scipy pandas matplotlib seaborn tensorboard
```

## 🎯 Quick Start

### Basic Training

Train MSAGAT-Net with optimal hyperparameters:

```bash
python src/train.py \
  --dataset japan \
  --sim_mat japan-adj \
  --window 20 \
  --horizon 5 \
  --model msagat \
  --cuda \
  --gpu 0 \
  --epochs 1500 \
  --batch 32 \
  --lr 1e-3 \
  --hidden_dim 32 \
  --attention_heads 4 \
  --bottleneck_dim 8 \
  --save_dir save \
  --mylog
```

### Hyperparameter Optimization

Automatically search for optimal hyperparameters using Optuna:

```bash
python src/optimize.py \
  --trials 50 \
  --dataset japan \
  --sim-mat japan-adj \
  --window 20 \
  --horizon 5 \
  --model-type msagat \
  --gpu 0
```

The optimization process explores learning rate, dropout, architecture dimensions, and regularization weights, saving results to `optim_results/`.

### Batch Experiments

Run comprehensive experiments across all 7 datasets, multiple horizons, and ablation variants:

```bash
python src/run_experiments.py
```

This executes 96 training runs (7 datasets × varying horizons × 4 ablations), automatically skipping completed experiments. Results are saved to `save_all/` and `report/results/`.

## Model Variants and Ablation Studies

We conducted comprehensive ablation studies to evaluate the contribution of each major component to the model's performance. The ablation variants include:

- **Full Model**: The complete MSAGAT-Net with all components
- **No EAGAM**: Removes the Efficient Adaptive Graph Attention Module
- **No DMTM**: Removes the Dilated Multi-Scale Temporal Module
- **No PPM**: Removes the Progressive Prediction Module

![Component Importance](report/figures/component_importance_comparison_w20.png)

Our analysis shows that each component contributes significantly to the model's overall performance, with EAGAM showing the most substantial impact on spatial-aware forecasting accuracy.

### Running Ablation Studies

```bash
# Run the full model
python src/train_ablation.py --dataset japan --sim_mat japan-adj --window 20 --horizon 5 --ablation none

# Run without Efficient Adaptive Graph Attention Module
python src/train_ablation.py --dataset japan --sim_mat japan-adj --window 20 --horizon 5 --ablation no_eagam

# Run without Dilated Multi-Scale Temporal Module
python src/train_ablation.py --dataset japan --sim_mat japan-adj --window 20 --horizon 5 --ablation no_dmtm

# Run without Progressive Prediction Module
python src/train_ablation.py --dataset japan --sim_mat japan-adj --window 20 --horizon 5 --ablation no_ppm
```

### Analyzing Ablation Results

After running ablation variants, analyze the results:

```bash
# Analyze all available ablation results automatically
python src/analyze_ablations.py --results_dir results --figures_dir report/figures
```

This generates:
- Comparison plots showing the impact of each ablation on model performance
- Component importance visualizations
- Detailed reports summarizing the findings

## Hyperparameter Optimization

MSAGATNet includes an Optuna-based hyperparameter optimization framework:

```bash
python src/optimize.py --dataset japan --sim-mat japan-adj --window 20 --horizon 5 --trials 100
```

The optimization process:
1. Automatically explores different hyperparameter combinations
2. Trains and evaluates models with early stopping
3. Tracks metrics including RMSE, PCC, and parameter efficiency
4. Saves detailed trial histories and best model checkpoints

## Visualizing Results

Generate publication-quality visualizations:

```bash
python generate_paper_visualizations.py --results_dir results --output_dir report/figures
```

This script produces:
1. Performance comparison plots across datasets and forecast horizons
2. Component importance visualizations showing which model parts contribute most

## 🧪 Experimental Results

### Performance on Japan COVID-19 Dataset (Horizon=5)

Our optimized model achieves state-of-the-art performance:

| Metric | Value |
|--------|-------|
| **R²** | 0.7284 |
| **RMSE** | 1073.91 |
| **MAE** | 704.32 |
| **PCC** | 0.8560 |

**Optimal Hyperparameters**:
- Hidden dimension: 32
- Attention heads: 4
- Bottleneck dimension: 8
- Temporal scales: 4 (dilations: 1, 2, 4)
- Kernel size: 3
- Feature channels: 16
- Dropout: 0.2
- Learning rate: 1e-3
- Weight decay: 5e-4

### Cross-Dataset Performance

MSAGAT-Net demonstrates strong generalization across diverse epidemic datasets:

- **Japan COVID-19** (47 prefectures): Superior performance on prefecture-level forecasting
- **US States** (50 states): Scalable to larger geographical regions
- **UK NHS** (142 regions): Handles high-dimensional spatial data efficiently
- **Spain, Australia**: Robust across different epidemic characteristics

Detailed results and ablation study findings are available in `report/results/`

The framework tracks comprehensive evaluation metrics:

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **PCC**: Pearson Correlation Coefficient  
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination
- **Peak Error**: Maximum absolute error across regions

All metrics are computed both globally and per-region for detailed analysis.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{msagatnet2026,
  title={MSAGAT-Net: Multi-Scale Temporal-Adaptive Graph Attention Network for Epidemic Forecasting},
  author={Author Names},
  journal={Under Review},
  year={2026}
}
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

This research was supported by [Funding Agency]. We thank [Collaborators] for valuable discussions and feedback