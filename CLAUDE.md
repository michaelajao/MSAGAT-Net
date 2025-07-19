# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSAGAT-Net (Multi-Scale Adaptive Graph Attention Network) is a PyTorch-based deep learning framework for spatiotemporal epidemic forecasting across multiple geographical regions.

## Key Commands

### Training Models

```bash
# Basic training
python src/train.py --dataset japan --sim_mat japan-adj --window 20 --horizon 5 --cuda --mylog

# Training with specific hyperparameters (example)
python src/train.py \
  --dataset japan \
  --sim_mat japan-adj \
  --window 20 \
  --horizon 5 \
  --batch 32 \
  --epochs 1500 \
  --lr 0.001 \
  --weight_decay 5e-4 \
  --dropout 0.2 \
  --patience 100 \
  --hidden_dim 16 \
  --attention_heads 4 \
  --attention_regularization_weight 1e-5 \
  --num_scales 4 \
  --kernel_size 3 \
  --feature_channels 16 \
  --bottleneck_dim 8 \
  --seed 42 \
  --gpu 0 \
  --save_dir save \
  --mylog
```

### Running Ablation Studies

```bash
# Full model
python src/train.py --dataset japan --sim_mat japan-adj --window 20 --horizon 5 --ablation none

# Without Adaptive Graph Attention Module
python src/train.py --dataset japan --sim_mat japan-adj --window 20 --horizon 5 --ablation no_agam

# Without Multi-scale Temporal Feature Module
python src/train.py --dataset japan --sim_mat japan-adj --window 20 --horizon 5 --ablation no_mtfm

# Without Progressive Prediction Refinement Module
python src/train.py --dataset japan --sim_mat japan-adj --window 20 --horizon 5 --ablation no_pprm
```

### Running Experiments and Visualizations

```bash
# Run multiple experiments
python src/run_experiments.py

# Generate visualizations
python src/generate_visualizations.py
```

## Architecture

### Core Model Components (src/model.py)

1. **SpatialAttentionModule**: Efficient graph attention with linear complexity O(N) using low-rank decomposition
2. **MultiScaleTemporalModule**: Dilated convolutions at multiple scales for temporal pattern recognition
3. **HorizonPredictor**: Multi-step prediction module with refinement capabilities

### Model Variants (src/ablation.py)

Implements ablation study variants that selectively disable components:
- Full model (none)
- No LR-AGAM (Low-rank Adaptive Graph Attention Module)
- No MTFM (Multi-scale Temporal Feature Module)
- No PPRM (Progressive Prediction Refinement Module)

### Data Handling (src/data.py)

- **DataBasicLoader**: Handles epidemic time series data loading, preprocessing, and train/val/test splitting
- Supports adjacency matrix loading for spatial relationships
- Implements sliding window approach for sequence generation

## Important Implementation Details

### Device Handling
- Use `--cuda` flag to enable GPU training
- Use `--gpu <id>` to specify GPU device (e.g., `--gpu 0`)
- Code automatically falls back to CPU if CUDA is unavailable

### TensorBoard Logging
- Enabled by default with `--mylog` flag
- Logs are saved to `tensorboard/` directory
- Tracks training/validation losses and metrics

### Model Checkpointing
- Models are saved to `save/` directory by default
- Checkpoint naming: `{model_name}.{dataset}.w-{window}.h-{horizon}.{ablation}.pt`

### Evaluation Metrics
The model tracks multiple metrics:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- PCC (Pearson Correlation Coefficient)
- R² Score
- Peak Error

### Dataset Format
- Time series data files: `{dataset_name}.txt`
- Adjacency matrices: `{dataset_name}-adj.txt`
- Data should be in `data/` directory

## Dependencies

The project requires:
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib
- TensorBoard

Note: No requirements.txt file exists in the repository. Install dependencies manually based on imports in the code.

## Testing

No formal test suite exists. Model validation is performed during training through:
- Train/validation/test splits
- Early stopping based on validation loss
- Comprehensive metric tracking