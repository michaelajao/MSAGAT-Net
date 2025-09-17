# MSAGAT-Net: Multi-Scale Temporal-Adaptive Graph Attention Network

A sophisticated spatiotemporal deep learning framework designed for epidemic forecasting across multiple geographical regions. MSAGAT-Net combines efficient graph attention mechanisms with multi-scale temporal processing to achieve state-of-the-art forecasting accuracy with linear computational complexity.

## 🚀 Key Features

- **Two Model Variants**: Standard MSAGAT-Net and Linformer-based implementation
- **Linear Attention Complexity**: O(N) instead of O(N²) for large-scale applications  
- **Multi-Scale Temporal Processing**: Captures patterns at different time scales
- **Automated Hyperparameter Optimization**: Built-in Optuna-based optimization
- **Comprehensive Evaluation**: Extensive metrics and visualization tools
- **Production Ready**: Clean, modular code with extensive documentation

## 🏗️ Model Architecture

### Standard MSAGAT-Net

1. **Spatial Attention Module**: Low-rank graph attention with learnable bias and O(N) complexity
2. **Multi-Scale Temporal Module**: Dilated convolutions for temporal pattern extraction  
3. **Progressive Prediction Module**: Multi-step forecasting with adaptive refinement

### Linformer MSAGAT-Net

1. **Linformer Attention**: True O(N) complexity with E/F projection matrices from Wang et al. (2020)
2. **Enhanced Spatial Processing**: Projected attention in lower-dimensional space (k << N)
3. **Efficient Implementation**: Maintains accuracy while reducing computational cost

## 📊 Performance Highlights

- **Linear-time Attention Complexity**: Uses linearized attention with the ELU+1 kernel trick to achieve O(N) complexity
- **Low-Rank Factorization**: Employs parameter-efficient factorized representations in attention and projection layers
- **Multi-Scale Processing**: Captures temporal patterns at different resolutions with dilated convolutions
- **Adaptive Fusion**: Automatically learns optimal weights for combining multi-scale features
- **Regularized Graph Learning**: Incorporates L1 regularization for interpretable attention patterns

## Performance Highlights

Our experiments across multiple epidemic datasets show:

- **Superior Forecasting Accuracy**: Lower RMSE and MAE than state-of-the-art models across diverse datasets
- **Strong Correlation**: Higher Pearson Correlation Coefficient (PCC) and R² scores, indicating better capture of trend dynamics
- **Computational Efficiency**: Linear-time complexity with respect to the number of regions
- **Parameter Efficiency**: Significantly fewer parameters than comparable models through factorization techniques

![Performance Across Horizons](report/figures/RMSE_vs_horizon_w20.png)

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

Train the standard MSAGAT-Net model:

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

Train the Linformer variant:

```bash
python src/train.py \
  --dataset japan \
  --sim_mat japan-adj \
  --window 20 \
  --horizon 5 \
  --model linformer \
  --cuda \
  --gpu 0 \
  --epochs 1500 \
  --batch 32 \
  --lr 1e-3 \
  --hidden_dim 32 \
  --attention_heads 2 \
  --bottleneck_dim 16 \
  --save_dir save
```

### Hyperparameter Optimization

Find optimal parameters automatically using Optuna:

```bash
# Optimize MSAGAT-Net
python src/optimize.py \
  --trials 50 \
  --dataset japan \
  --sim-mat japan-adj \
  --window 20 \
  --horizon 5 \
  --model msagat \
  --gpu 0

# Optimize Linformer variant  
python src/optimize.py \
  --trials 50 \
  --dataset japan \
  --sim-mat japan-adj \
  --window 20 \
  --horizon 5 \
  --model linformer \
  --gpu 0
```

### Batch Experiments

Run systematic experiments across multiple configurations:

```bash
# Run experiments across all datasets and horizons
python src/run_experiments.py
```

```bash
python src/train.py   --dataset japan   --sim_mat japan-adj   --window 20   --horizon 5   --cuda   --seed 42   --batch 32   --epochs 1500   --lr 1e-3   --weight_decay 5e-4   --dropout 0.20   --patience 100   --lr_patience 20   --lr_factor 0.5   --attention_regularization_weight 1e-4   --num_scales 6   --kernel_size 9   --feature_channels 64   --bottleneck_dim 8
```

```bash
python src/train.py \
  --dataset japan \
  --sim_mat japan-adj \
  --window 20 \
  --horizon 5 \
  --cuda \
  --seed 42 \
  --batch 32 \
  --epochs 1500 \
  --lr 1e-3 \
  --weight_decay 5e-4 \
  --dropout 0.20 \
  --patience 100 \
  --attention_heads 4 \
  --attention_regularization_weight 1e-5 \
  --num_scales 4 \
  --kernel_size 3 \
  --feature_channels 16 \
  --bottleneck_dim 8
```

```bash
python src/train.py \
  --dataset japan \
  --sim_mat japan-adj \
  --window 20 \
  --horizon 5 \
  --train 0.5 \
  --val   0.2 \
  --test  0.3 \
  --cuda \
  --seed   42 \
  --batch  32 \
  --epochs 1500 \
  --lr              1e-3 \
  --weight_decay    5e-4 \
  --dropout         0.20 \
  --patience        100 \
  --attention_heads                  4 \
  --attention_regularization_weight  1e-5 \
  --num_scales      4 \
  --kernel_size     3 \
  --feature_channels 16 \
  --bottleneck_dim  8
```


```bash
python src/train.py   --dataset japan   --sim_mat japan-adj   --window 20   --horizon 5  --epochs 1500   --batch 32   --lr 0.001   --weight_decay 5e-4   --dropout 0.2   --patience 100   --hidden_dim 32   --attention_heads 4   --attention_regularization_weight 1e-5   --num_scales 4   --kernel_size 3   --feature_channels 16   --bottleneck_dim 8   --seed 42   --gpu 0   --model msagat   --use_adjacency   --save_dir save   --mylog
```

```bash
python src/train.py \
  --dataset region785 \
  --sim_mat region-adj \
  --window 20 \
  --horizon 5 \
  --train 0.5 \
  --val 0.2 \
  --test 0.3 \
  --epochs 1500 \
  --batch 32 \
  --lr 0.004865348445118787 \
  --weight_decay 1.001411958893656e-05 \
  --dropout 0.2234645558140993 \
  --patience 100 \
  --hidden_dim 32 \
  --attention_heads 8 \
  --attention_regularization_weight 3.963070437166054e-05 \
  --num_scales 5 \
  --kernel_size 9 \
  --feature_channels 8 \
  --bottleneck_dim 12 \
  --seed 42 \
  --gpu 0 \
  --save_dir save \
  --mylog
```
```bash
python src/train.py \
  --dataset japan \
  --sim_mat japan-adj \
  --window 20 \
  --horizon 5 \
  --train 0.5 \
  --val 0.2 \
  --test 0.3 \
  --epochs 1500 \
  --batch 32 \
  --lr 0.001893 \
  --weight_decay 6.72e-5 \
  --dropout 0.318 \
  --patience 100 \
  --hidden_dim 16 \
  --attention_heads 4 \
  --attention_regularization_weight 3.15e-4 \
  --num_scales 4 \
  --kernel_size 5 \
  --feature_channels 12 \
  --bottleneck_dim 8 \
  --seed 42 \
  --gpu 0 \
  --model msagat \
  --use_adjacency \
  --save_dir save \
  --mylog
```

```bash
python src/train.py   --dataset japan  --sim_mat japan-adj   --window 20   --horizon 3   --epochs 1500   --batch 32   --lr 0.005   --weight_decay 6.72e-5   --dropout 0.2   --patience 100   --hidden_dim 16   --attention_heads 4   --num_scales 4   --kernel_size 5   --feature_channels 12   --attention_regularization_weight 3.15e-4   --bottleneck_dim 12   --seed 42   --gpu 0   --model enhanced_msagat   --highway_window 3   --save_dir save   --mylog
```



### Prediction

Generate forecasts with a trained model:

```bash
python src/predict.py \
  --dataset japan \
  --sim_mat japan-adj \
  --window 20 \
  --horizon 5 \
  --model_path save/MSAGATNet.japan.w-20.h-5.pt \
  --output_dir predictions
```

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

## 🧪 Optimization Results

Recent hyperparameter optimization revealed optimal configurations for both model variants:

### MSAGAT-Net (Japan Dataset, Horizon=5)
- **Best Validation RMSE**: 989.07
- **Best Test RMSE**: 1171.08
- **Optimal Configuration**:
  - `hidden_dim`: 16
  - `attention_heads`: 8  
  - `bottleneck_dim`: 8
  - `num_scales`: 5
  - `kernel_size`: 5
  - `dropout`: 0.255
  - `lr`: 0.000579
  - `weight_decay`: 2.38e-05

### Linformer MSAGAT-Net (Japan Dataset, Horizon=5)
- **Best Validation RMSE**: 960.69
- **Best Test RMSE**: 1069.27  
- **Optimal Configuration**:
  - `hidden_dim`: 32
  - `attention_heads`: 2
  - `bottleneck_dim`: 16
  - `num_scales`: 3
  - `kernel_size`: 3
  - `dropout`: 0.137
  - `lr`: 0.000933
  - `weight_decay`: 1.19e-05

The Linformer variant shows **superior performance** with lower RMSE while maintaining computational efficiency.

## Extended Model Variants

In addition to the original MSAGAT-Net architecture, we have developed and implemented several enhanced models that address specific challenges in spatiotemporal forecasting:

### LocationAwareMSAGAT_Net

An extension of the original model that explicitly incorporates geospatial information through a dedicated location-aware attention mechanism. This variant achieves improved performance on datasets with strong spatial dependencies.

```bash
# Train the LocationAwareMSAGAT_Net model
python src/train.py \
  --dataset region785 \
  --sim_mat region-adj \
  --window 20 \
  --horizon 5 \
  --model location_aware \
  --use_adjacency \
  --cuda \
  --gpu 0
```

### DynaGraphNet

DynaGraphNet provides dynamic graph structure learning capabilities, adaptively discovering relationships between nodes based on their feature representations rather than relying solely on predefined adjacency matrices.

Key features:
- Dynamic graph structure inference from node features
- Unified relational attention for capturing both spatial and temporal dependencies
- Optional autoregressive connections for small graphs with strong temporal patterns

```bash
# Train the DynaGraphNet model
python src/train.py \
  --dataset region785 \
  --sim_mat region-adj \
  --window 20 \
  --horizon 5 \
  --model dynagraph \
  --cuda \
  --gpu 0

# With autoregressive connection for small graphs
python src/train.py \
  --dataset japan \
  --sim_mat japan-adj \
  --window 20 \
  --horizon 5 \
  --model dynagraph \
  --autoregressive \
  --cuda \
  --gpu 0
```

### AFGNet (Adaptive Fusion Graph Network)

AFGNet combines the strengths of both static and dynamic graph approaches through an adaptive fusion mechanism. This model is particularly effective for datasets where both predefined relationships and learned patterns contribute to forecasting accuracy.

Key features:
- Efficient feature extraction with depthwise separable convolutions
- Dynamic graph structure inference
- Adaptive fusion of static and dynamic graph structures
- Multi-scale temporal pattern encoding
- Enhanced prediction module with adaptive blending

```bash
# Train the AFGNet model
python src/train.py \
  --dataset japan \
  --sim_mat japan-adj \
  --window 20 \
  --horizon 5 \
  --model afgnet \
  --cuda \
  --gpu 0

# With static adjacency matrix
python src/train.py \
  --dataset japan \
  --sim_mat japan-adj \
  --window 20 \
  --horizon 5 \
  --model afgnet \
  --use_adjacency \
  --cuda \
  --gpu 0
```

## Model Comparison

Each model variant offers unique strengths for different forecasting scenarios:

1. **MSAGAT-Net (Original)**: Best for general forecasting with efficient parameter usage
2. **LocationAwareMSAGAT_Net**: Optimal for datasets with strong geographic dependencies
3. **DynaGraphNet**: Excels when relationships between nodes evolve over time
4. **AFGNet**: Provides the most flexible approach by adaptively blending static and dynamic graph structures

You can compare model performance across different datasets using:

```bash
# Run a comprehensive model comparison
python compare_models.py --datasets japan,region785 --horizons 5,10,15 --output_dir report/figures
```

## Visualizing Model Outputs

All model variants provide rich visualizations of their learned attention patterns, making it easier to interpret how they capture spatial relationships:

```bash
# Generate paper-quality visualizations
python generate_paper_visualizations.py --results_dir results --output_dir report/figures
```

The visualization outputs include:
- Learned attention/graph structures
- Comparisons with geographic adjacency 
- Prediction accuracy across different time horizons

Check the `report/figures` directory for generated visualizations after training.

## 📁 Project Structure

```
MSAGAT-Net/
├── data/                   # Epidemic datasets and adjacency matrices
├── optim_results/          # Hyperparameter optimization outputs  
├── report/
│   ├── figures/           # Generated visualizations
│   └── results/           # Evaluation metrics and results
├── save/                  # Model checkpoints
├── save_all/              # All model variants checkpoints
├── src/                   # Source code
│   ├── model.py           # Standard MSAGAT-Net implementation
│   ├── model_true_linformer.py  # Linformer MSAGAT-Net implementation
│   ├── train.py           # Unified training script
│   ├── optimize.py        # Hyperparameter optimization (Optuna)
│   ├── run_experiments.py # Batch experiment runner
│   ├── ablation.py        # Ablation study models
│   ├── data.py            # Data loading utilities
│   └── utils.py           # Helper functions
├── tensorboard/           # TensorBoard training logs
└── old/                   # Legacy code and results
```

## 📈 Evaluation Metrics

The framework tracks comprehensive evaluation metrics:

- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **PCC**: Pearson Correlation Coefficient  
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of Determination
- **Peak Error**: Maximum absolute error across regions

All metrics are computed both globally and per-region for detailed analysis.

## Citation

If you use this code in your research, please cite our work:

```bibtex
@article{msagatnet2025,
  title={MSAGATNet: Multi-Scale Adaptive Graph Attention Network for Epidemic Forecasting},
  author={Author Names},
  journal={Journal Name},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.