"""
MSAGAT-Net Ablation Study Components

This module contains alternative implementations of MSAGAT-Net's key components
for ablation studies. It includes:

1. SimpleGraphConvolutionalLayer: Standard GCN layer (ablation for EfficientAdaptiveGraphAttentionModule)
2. SingleScaleTemporalModule: Single-scale convolution (ablation for DilatedMultiScaleTemporalModule)
3. DirectPredictionModule: Direct prediction (ablation for ProgressivePredictionModule)
4. MSAGATNet_Ablation: Model class with configurable components for ablation studies

These components are used to systematically evaluate the contribution of each
module to the overall model performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

# import model constants and modules directly
from model import (
    HIDDEN_DIM,
    ATTENTION_HEADS,
    ATTENTION_REG_WEIGHT_INIT,  # Updated from ATTENTION_REG_WEIGHT to ATTENTION_REG_WEIGHT_INIT
    DROPOUT,
    NUM_TEMPORAL_SCALES,
    KERNEL_SIZE,
    FEATURE_CHANNELS,
    BOTTLENECK_DIM,
    SpatialAttentionModule,
    MultiScaleTemporalModule,
    HorizonPredictor,
    DepthwiseSeparableConv1D
)

# =============================================================================
# 1. Simple Graph Convolutional Layer (for LR-AGAM ablation)
# =============================================================================
class SimpleGraphConvolutionalLayer(nn.Module):
    """Standard GCN layer with fixed adjacency matrix.
    
    Used as an ablation replacement for the EfficientAdaptiveGraphAttentionModule.
    This layer performs standard graph convolution without adaptive attention mechanisms.
    
    Args:
        hidden_dim (int): Dimension of hidden representations
        num_nodes (int): Number of nodes in the graph
        dropout (float, optional): Dropout rate. Defaults to DROPOUT
    
    Shape:
        - Input: (batch_size, num_nodes, hidden_dim)
        - Output: (batch_size, num_nodes, hidden_dim), scalar_loss
    """
    def __init__(self, hidden_dim, num_nodes, dropout=DROPOUT, **kwargs):
        super(SimpleGraphConvolutionalLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Fixed adjacency matrix (identity by default)
        self.register_buffer('adj_matrix', torch.eye(num_nodes))
        
    def forward(self, x, mask=None):
        """Forward pass of the GCN layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, hidden_dim)
            mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.
            
        Returns:
            tuple: (output tensor of shape (batch_size, num_nodes, hidden_dim), attention regularization loss)
        """
        # x: [B, N, H]
        batch_size = x.size(0)
        num_nodes = x.size(1)
        
        # Apply first linear transform
        x = self.linear1(x)
        
        # Ensure adj_matrix has the correct size
        if self.adj_matrix.size(0) != num_nodes or self.adj_matrix.size(1) != num_nodes:
            # If the adjacency matrix doesn't match the input size, use identity matrix
            adj_matrix = torch.eye(num_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # Otherwise use the stored adjacency matrix
            adj_matrix = self.adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
            
        # Apply GCN operation: X' = AXW
        x = torch.bmm(adj_matrix, x)  # Aggregate neighbor features
        x = F.relu(x)                  # Apply non-linearity
        x = self.dropout(x)            # Apply dropout for regularization
        x = self.linear2(x)            # Second linear transformation
        x = self.norm(x)               # Layer normalization
        
        # Store empty attention for visualization purposes
        self.attn = [torch.eye(num_nodes, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)]
        
        # Return zero for attention regularization loss (not used in GCN)
        return x, 0.0

# =============================================================================
# 2. Single-scale Temporal Module (for DMTFM ablation)
# =============================================================================
class SingleScaleTemporalModule(nn.Module):
    """Single-scale temporal convolution for ablation study.
    
    Used as an ablation replacement for the DilatedMultiScaleTemporalModule.
    This module processes temporal features at a single scale with a fixed kernel size.
    
    Args:
        hidden_dim (int): Dimension of hidden representations
        dropout (float, optional): Dropout rate. Defaults to DROPOUT
    
    Shape:
        - Input: (batch_size, num_nodes, hidden_dim)
        - Output: (batch_size, num_nodes, hidden_dim)
    """
    def __init__(self, hidden_dim, num_scales=NUM_TEMPORAL_SCALES, kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super(SingleScaleTemporalModule, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Single-scale convolution without dilation
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Simple output projection
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """Forward pass of the single-scale temporal module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, hidden_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_nodes, hidden_dim)
        """
        # x: [B, N, H] -> transpose to [B, H, N] for conv1d
        x = x.transpose(1, 2)  # Conv1d expects channels dimension second
        
        # Apply single-scale convolution
        feat = self.conv(x)  # [B, H, N]
        
        # Transpose back to [B, N, H]
        feat = feat.transpose(1, 2)
        
        # Apply output projection
        out = self.fusion(feat)
        
        return out

# =============================================================================
# 3. Direct Prediction Module (for PPRM ablation)
# =============================================================================
class DirectPredictionModule(nn.Module):
    """Direct multi-step prediction for ablation study.
    
    Used as an ablation replacement for the ProgressivePredictionModule.
    This module directly predicts all future time steps without any refinement mechanism.
    
    Args:
        hidden_dim (int): Dimension of hidden representations
        horizon (int): Prediction horizon
        dropout (float, optional): Dropout rate. Defaults to DROPOUT
    
    Shape:
        - Input: (batch_size, num_nodes, hidden_dim), (batch_size, num_nodes)
        - Output: (batch_size, num_nodes, horizon)
    """
    def __init__(self, hidden_dim, horizon, low_rank_dim=BOTTLENECK_DIM, dropout=DROPOUT):
        super(DirectPredictionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # Direct predictor for all forecast steps
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, horizon)
        )

    def forward(self, x, last_step=None):
        """Forward pass of the direct prediction module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_nodes, hidden_dim)
            last_step (torch.Tensor): Last observation tensor of shape (batch_size, num_nodes)
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size, num_nodes, horizon)
        """
        # Simple direct prediction (without refinement)
        preds = self.predictor(x)  # [B, N, horizon]
        return preds

# =============================================================================
# 4. Overall Model: MSAGAT-Net with Ablation Options
# =============================================================================
class MSAGATNet_Ablation(nn.Module):
    """MSAGAT-Net: Multi-Scale Adaptive Graph Attention Network with ablation options.
    
    This model allows for systematic ablation studies by replacing key components with simpler alternatives.
    
    Args:
        args (Namespace): Arguments containing model hyperparameters
        data (Dataset): Dataset object containing input data
    
    Shape:
        - Input: (batch_size, window, num_nodes)
        - Output: (batch_size, horizon, num_nodes), scalar_loss
    """
    def __init__(self, args, data):
        super(MSAGATNet_Ablation, self).__init__()
        self.m = data.m  # Number of nodes
        self.window = args.window  # Input window size
        self.horizon = args.horizon  # Prediction horizon
        self.ablation = getattr(args, 'ablation', 'none')

        # Model dimensions
        self.hidden_dim = getattr(args, 'hidden_dim', HIDDEN_DIM)
        self.kernel_size = getattr(args, 'kernel_size', KERNEL_SIZE)
        self.low_rank_dim = getattr(args, 'bottleneck_dim', BOTTLENECK_DIM)

        # Feature extraction
        self.temp_conv = DepthwiseSeparableConv1D(
            in_channels=1,
            out_channels=FEATURE_CHANNELS,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # Feature processing and compression (same for all ablations)
        self.feature_process_low = nn.Linear(FEATURE_CHANNELS * self.window, self.low_rank_dim)
        self.feature_process_high = nn.Linear(self.low_rank_dim, self.hidden_dim)
        self.feature_norm = nn.LayerNorm(self.hidden_dim)
        self.feature_act = nn.ReLU()
        
        # Spatial component: choose full attention or GCN
        if self.ablation == 'no_agam':
            # Replace Efficient Adaptive Graph Attention with simple GCN
            self.graph_attention = SimpleGraphConvolutionalLayer(
                self.hidden_dim,
                num_nodes=self.m,
                dropout=getattr(args, 'dropout', DROPOUT)
            )
            # Store adjacency matrix from data if available
            if hasattr(data, 'adj'):
                self.graph_attention.adj_matrix = data.adj
        else:
            # Original: Efficient Adaptive Graph Attention Module
            self.graph_attention = SpatialAttentionModule(
                hidden_dim=self.hidden_dim,
                num_nodes=self.m,
                dropout=getattr(args, 'dropout', DROPOUT),
                attention_heads=getattr(args, 'attention_heads', ATTENTION_HEADS),
                attention_regularization_weight=getattr(args, 'attention_regularization_weight', ATTENTION_REG_WEIGHT_INIT),
                bottleneck_dim=self.low_rank_dim
            )
        
        # Temporal component: choose multi-scale or single-scale
        if self.ablation == 'no_dmtm':
            # Replace Dilated Multi-Scale Temporal with single-scale
            self.temporal_module = SingleScaleTemporalModule(
                self.hidden_dim,
                kernel_size=self.kernel_size,
                dropout=getattr(args, 'dropout', DROPOUT)
            )
        else:
            # Original: Dilated Multi-Scale Temporal Module
            self.temporal_module = MultiScaleTemporalModule(
                hidden_dim=self.hidden_dim,
                num_scales=getattr(args, 'num_scales', NUM_TEMPORAL_SCALES),
                kernel_size=self.kernel_size,
                dropout=getattr(args, 'dropout', DROPOUT)
            )
        
        # Prediction component: choose progressive or direct
        if self.ablation == 'no_ppm':
            # Replace Progressive Prediction with direct multi-step
            self.prediction_module = DirectPredictionModule(
                hidden_dim=self.hidden_dim,
                horizon=self.horizon,
                low_rank_dim=self.low_rank_dim,
                dropout=getattr(args, 'dropout', DROPOUT)
            )
        else:
            # Original: Progressive Prediction Module
            self.prediction_module = HorizonPredictor(
                hidden_dim=self.hidden_dim,
                horizon=self.horizon,
                bottleneck_dim=self.low_rank_dim,
                dropout=getattr(args, 'dropout', DROPOUT)
            )

    def forward(self, x, idx=None):
        """Forward pass of the overall model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, window, num_nodes)
            idx (torch.Tensor, optional): Index tensor. Defaults to None
            
        Returns:
            tuple: (predictions tensor of shape (batch_size, horizon, num_nodes), attention regularization loss)
        """
        B, T, N = x.shape
        x_last = x[:, -1, :]  # Last observed values
        
        # Reshape for temporal processing
        x_temp = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)
        
        # Extract temporal features
        temp_features = self.temp_conv(x_temp)
        temp_features = temp_features.view(B, N, -1)
        
        # Process features with dimension reduction
        features = self.feature_process_low(temp_features)
        features = self.feature_process_high(features)
        features = self.feature_norm(features)
        features = self.feature_act(features)
        
        # Apply graph attention for spatial dependencies
        graph_features, attn_reg_loss = self.graph_attention(features)
        
        # Process temporal patterns
        fusion_features = self.temporal_module(graph_features)
        
        # Generate predictions
        predictions = self.prediction_module(fusion_features, x_last)
        predictions = predictions.transpose(1, 2)  # [B, horizon, N]
        
        return predictions, attn_reg_loss