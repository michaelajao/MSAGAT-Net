# MSAGATNet: Multi-Scale Adaptive Graph Attention Network for Epidemic Forecasting

MSAGATNet is a lightweight spatiotemporal deep learning framework specifically designed for epidemic forecasting across multiple geographical regions. The model combines efficient graph attention for spatial relationships with multi-scale temporal processing for accurate time-series forecasting.

## Model Architecture

MSAGATNet consists of four key components:

1. **Efficient Adaptive Graph Attention Module**: Captures spatial dependencies between regions with linear complexity O(N) instead of quadratic O(N²)
2. **Dilated Multi-Scale Temporal Module**: Processes time-series patterns at different temporal resolutions
3. **Depthwise Separable Convolutions**: Provides parameter-efficient feature extraction
4. **Attention-Based Prediction Module**: Enables region-aware multi-step forecasting

## Key Features

- **Linear-time Attention Complexity**: Uses linearized attention with the ELU+1 kernel trick to achieve O(N) complexity
- **Low-Rank Factorization**: Employs parameter-efficient factorized representations in attention and projection layers
- **Multi-Scale Processing**: Captures temporal patterns at different resolutions with dilated convolutions
- **Adaptive Fusion**: Automatically learns optimal weights for combining multi-scale features
- **Regularized Graph Learning**: Incorporates L1 regularization for interpretable attention patterns

## Datasets

The repository includes several epidemic datasets:
- Japan COVID-19 dataset (`japan.txt`) with adjacency information (`japan-adj.txt`)
- Spain COVID-19 dataset (`spain-covid.txt`) with adjacency information (`spain-adj.txt`)
- Australia COVID-19 dataset (`australia-covid.txt`) with adjacency information (`australia-adj.txt`)
- UK NHS regions dataset (`nhs_timeseries.txt`) with adjacency information (`nhs-adj.txt`)
- US state-level dataset (`state360.txt`) with adjacency information (`state-adj-49.txt`, `state-adj-50.txt`)

## Usage

### Training

Train the model with custom parameters:

```bash
python src/train.py --dataset japan --sim_mat japan-adj --window 20 --hidden_dim 16 --attn_heads 16 --low_rank_dim 6 --num_scales 5 --kernel_size 3 --temp_conv_out_channels 12 --dropout 0.249 --attention_reg_weight 1e-3 --lr 0.001 --weight_decay 5e-4 --cuda --mylog --horizon 3
```

### Hyperparameter Optimization

MSAGATNet includes an Optuna-based hyperparameter optimization framework to find optimal model configurations for different datasets and forecasting tasks:

```bash
python src/optimize.py --dataset japan --sim-mat japan-adj --window 20 --horizon 5 --trials 100 --max-params 50000
```

The optimization process:
1. Automatically explores different hyperparameter combinations
2. Trains and evaluates models with early stopping
3. Tracks metrics including RMSE, PCC, and parameter efficiency
4. Saves detailed trial histories and best model checkpoints

#### Optimization Parameters

- `--dataset`: Name of the dataset (e.g., `japan`, `region785`, `state360`)
- `--sim-mat`: Name of the adjacency matrix file (e.g., `japan-adj`)
- `--window`: Input window size (default: `20`)
- `--horizon`: Prediction horizon (default: `5`) 
- `--trials`: Number of optimization trials to run (default: `100`)
- `--max-params`: Maximum parameter count threshold (default: `50000`)
- `--seed`: Random seed for reproducibility (default: `42`)
- `--study-name`: Custom study name (default: auto-generated)
- `--continue-from`: Path to continue from a previous study
- `--parallel`: Number of parallel optimization processes (default: `1`) 
- `--gpu`: GPU index to use (default: `0`)

#### Optimization Output

The optimization process produces:
- Trial logs in `optim_results/logs/`
- Best model checkpoints in `optim_results/models/`
- Trial history in JSON format in `optim_results/`
- Study results in CSV format for further analysis

### Parameters

- `--dataset`: Name of the dataset (default: `region785`)
- `--sim_mat`: Name of the adjacency matrix file (default: `region-adj`)
- `--window`: Input window size (default: `20`)
- `--horizon`: Prediction horizon (default: `5`)
- `--hidden_dim`: Hidden dimension size (default: `32`)
- `--attn_heads`: Number of attention heads (default: `4`)
- `--low_rank_dim`: Dimension for low-rank decompositions (default: `8`)
- `--num_scales`: Number of temporal scales in multi-scale module (default: `4`)
- `--kernel_size`: Size of temporal convolution kernel (default: `3`)
- `--temp_conv_out_channels`: Output channels for temporal convolution (default: `16`)
- `--dropout`: Dropout rate (default: `0.355`)
- `--attention_reg_weight`: Weight for attention regularization (default: `1e-5`)
- `--lr`: Learning rate (default: `0.001`)
- `--weight_decay`: Weight decay (default: `5e-4`)
- `--cuda`: Use GPU acceleration if available
- `--mylog`: Enable TensorBoard logging

## Output

Training produces:
- Model checkpoints in `save` directory
- Performance metrics in `results/metrics_MSAGATNet.csv`
- TensorBoard logs in `tensorboard` directory
- Visualization figures in `report/figures`

## Performance Metrics

Performance is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Pearson Correlation Coefficient (PCC)
- R² Score
- Explained Variance
- Peak MAE (for epidemic peak prediction accuracy)

## Requirements

- Python 3.7+
- PyTorch 1.7+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- TensorBoard

## after optimization
```bash
python src/train.py \
  --dataset region785 \
  --sim_mat region-adj \
  --window 20 \
  --horizon 5 \
  --epochs 1500 \
  --batch 32 \
  --lr 0.0001 \
  --weight_decay 1e-5 \
  --dropout 0.287 \
  --hidden_dim 24 \
  --attn_heads 4 \
  --num_scales 5 \
  --kernel_size 7 \
  --temp_conv_out_channels 16 \
  --low_rank_dim 6 \
  --attention_reg_weight 0.001 \
  --seed 42 \
  --cuda \
  --gpu 0 \
  --patience 100 \
  --mylog \
  --save_dir save
```

## Project Structure

```
MSAGAT-Net/
├── data/                   # Epidemic datasets and adjacency matrices
├── optim_results/          # Hyperparameter optimization outputs
│   ├── logs/               # Trial logs
│   ├── models/             # Saved model checkpoints from trials
│   └── *.json              # Trial history files
├── report/                 # Report and visualization outputs
│   └── figures/            # Generated plots and visualizations
├── results/                # Evaluation metrics and results
├── save/                   # Saved model checkpoints
├── src/                    # Source code
│   ├── data.py             # Data loading and preprocessing
│   ├── model.py            # MSAGATNet model implementation
│   ├── optimize.py         # Hyperparameter optimization script
│   ├── train.py            # Training and evaluation script
│   └── utils.py            # Utility functions
└── tensorboard/            # TensorBoard logs
```

## Citation

If you use this code in your research, please cite our work:

```bibtex
@article{msagatnet2023,
  title={MSAGATNet: Multi-Scale Adaptive Graph Attention Network for Epidemic Forecasting},
  author={Author Names},
  journal={Journal Name},
  year={2023}
}
```