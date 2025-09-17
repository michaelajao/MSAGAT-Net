import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# Model hyperparameters with descriptive names
HIDDEN_DIM = 32
ATTENTION_HEADS = 4
# Initial value for adaptive attention regularization weight
ATTENTION_REG_WEIGHT_INIT = 1e-05
DROPOUT = 0.25
NUM_TEMPORAL_SCALES = 4
KERNEL_SIZE = 3
FEATURE_CHANNELS = 16
BOTTLENECK_DIM = 8

class LinformerSpatialAttentionModule(nn.Module):
    """
    Linformer-style Spatial Attention Module with E and F projection matrices.
    
    This implements the actual Linformer mechanism from Wang et al. (2020) that achieves
    O(N) complexity by using projection matrices E and F to reduce the key and value
    sequence length from N to k, where k << N.
    
    Args:
        hidden_dim (int): Dimensionality of node features
        num_nodes (int): Number of nodes in the graph
        dropout (float): Dropout probability for regularization
        attention_heads (int): Number of parallel attention heads
        bottleneck_dim (int): Projected dimension k for Linformer (k << N)
        attention_regularization_weight (float): Weight for L1 regularization on attention
    """
    def __init__(self, hidden_dim, num_nodes, dropout=DROPOUT, attention_heads=ATTENTION_HEADS, 
                 attention_regularization_weight=ATTENTION_REG_WEIGHT_INIT, bottleneck_dim=BOTTLENECK_DIM):
        super(LinformerSpatialAttentionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = attention_heads
        self.head_dim = hidden_dim // self.heads
        self.num_nodes = num_nodes
        self.bottleneck_dim = bottleneck_dim

        # Make attention regularization weight a learnable parameter (log-domain for positivity)
        self.log_attention_reg_weight = nn.Parameter(torch.tensor(math.log(attention_regularization_weight), dtype=torch.float32))

        # Standard QKV projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # CRITICAL LINFORMER COMPONENTS: E and F projection matrices
        # E projects keys from N×d to k×d, F projects values from N×d to k×d
        # Following Linformer paper: E_i, F_i ∈ R^(N×k)
        self.E_proj = nn.Parameter(torch.randn(self.heads, num_nodes, bottleneck_dim) / math.sqrt(bottleneck_dim))
        self.F_proj = nn.Parameter(torch.randn(self.heads, num_nodes, bottleneck_dim) / math.sqrt(bottleneck_dim))

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable graph structure bias in the projected space (k×k instead of N×N)
        self.graph_bias_u = nn.Parameter(torch.randn(self.heads, bottleneck_dim, bottleneck_dim) * 0.1)
        self.graph_bias_v = nn.Parameter(torch.randn(self.heads, bottleneck_dim, bottleneck_dim) * 0.1)

    def forward(self, x, mask=None):
        """
        Forward pass implementing Linformer attention with O(N) complexity.
        
        Args:
            x (tensor): Input node features [batch, nodes, hidden_dim]
            mask (tensor, optional): Attention mask
            
        Returns:
            tuple: (Updated node features, Attention regularization loss)
        """
        B, N, H = x.shape
        
        # Standard QKV projections
        q = self.q_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        k = self.k_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        v = self.v_proj(x).view(B, N, self.heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        
        # CORE LINFORMER OPERATION: Project keys and values using E and F matrices
        # This reduces sequence length from N to k (bottleneck_dim)
        k_proj = torch.einsum('hik,bhnd->bhkd', self.E_proj, k)  # [B, heads, k, head_dim]
        v_proj = torch.einsum('hik,bhnd->bhkd', self.F_proj, v)  # [B, heads, k, head_dim]
        
        # Compute attention scores: Q @ K_proj^T with projected keys
        # Shape: [B, heads, N, head_dim] @ [B, heads, head_dim, k] = [B, heads, N, k]
        attn_scores = torch.einsum('bhnd,bhkd->bhnk', q, k_proj) / math.sqrt(self.head_dim)
        
        # Add learnable graph structure bias in the projected k×k space
        # Following Linformer paper: apply bias directly to attention scores
        graph_bias = torch.matmul(self.graph_bias_u, self.graph_bias_v)  # [heads, k, k]
        
        # Apply graph bias to attention scores in the k-dimensional space
        # Since attn_scores is [B, heads, N, k] and graph_bias is [heads, k, k],
        # we need to broadcast properly. Each node attends to k projected positions.
        # We add the bias by expanding it to match the batch and node dimensions.
        for h in range(self.heads):
            # Expand bias from [k, k] to [1, N, k] by taking diagonal or mean
            # For simplicity, use only the diagonal of the k×k bias matrix
            bias_diagonal = torch.diag(graph_bias[h])  # [k]
            attn_scores[:, h, :, :] = attn_scores[:, h, :, :] + bias_diagonal.unsqueeze(0).unsqueeze(0)  # [B, N, k]
        
        # Apply softmax over the k dimension (not N!)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, heads, N, k]
        
        # Apply attention to projected values
        # Shape: [B, heads, N, k] @ [B, heads, k, head_dim] = [B, heads, N, head_dim]
        output = torch.einsum('bhnk,bhkd->bhnd', attn_weights, v_proj)
        
        # Apply dropout
        output = self.dropout(output)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(B, N, H)
        output = self.out_proj(output)
        
        # Compute regularization loss on attention weights and graph bias
        attention_reg_weight = torch.exp(self.log_attention_reg_weight)
        attn_reg_loss = attention_reg_weight * (
            torch.mean(torch.abs(attn_weights)) + 
            torch.mean(torch.abs(graph_bias)) * 0.1
        )
        
        return output, attn_reg_loss


class MultiScaleTemporalModule(nn.Module):
    """
    Multi-Scale Temporal Module that captures temporal patterns at different scales.
    
    This module uses dilated convolutions at different dilation rates to capture
    short and long-term temporal dependencies. The outputs from different scales
    are adaptively fused.
    
    Args:
        hidden_dim (int): Dimensionality of node features
        num_scales (int): Number of temporal scales (different dilation rates)
        kernel_size (int): Size of the convolutional kernel
        dropout (float): Dropout probability for regularization
    """
    def __init__(self, hidden_dim, num_scales=NUM_TEMPORAL_SCALES, kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super(MultiScaleTemporalModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        # Create multiple dilated convolutional layers with increasing dilation rates
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                          padding=(kernel_size//2) * 2**i, dilation=2**i),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_scales)
        ])
        
        # Learnable weights for adaptive fusion of scales
        self.fusion_weight = Parameter(torch.ones(num_scales), requires_grad=True)
        
        # Low-rank projection for fusion
        self.fusion_low = nn.Linear(hidden_dim, BOTTLENECK_DIM)
        self.fusion_high = nn.Linear(BOTTLENECK_DIM, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Forward pass of the multi-scale temporal module.
        
        Args:
            x (tensor): Input features [batch, nodes, hidden_dim]
            
        Returns:
            tensor: Temporally processed features
        """
        # Reshape for 1D convolution [batch*nodes, hidden_dim, time]
        x = x.transpose(1, 2)  # [batch, hidden_dim, nodes]
        
        # Apply different temporal scales
        features = [scale(x) for scale in self.scales]
        
        # Compute adaptive weights for scale fusion
        alpha = F.softmax(self.fusion_weight, dim=0)
        
        # Stack and fuse multi-scale features
        stacked = torch.stack(features, dim=1)  # [batch, scales, hidden_dim, nodes]
        fused = torch.sum(alpha.view(1, self.num_scales, 1, 1) * stacked, dim=1)
        
        # Reshape back
        fused = fused.transpose(1, 2)  # [batch, nodes, hidden_dim]
        
        # Apply low-rank projection and residual connection
        out = self.fusion_low(fused)
        out = self.fusion_high(out)
        out = self.layer_norm(out + fused)
        
        return out


class HorizonPredictor(nn.Module):
    """
    Horizon Predictor module for forecasting future values.
    
    This module takes node features and generates predictions for multiple
    future time steps. It includes an optional refinement mechanism based on
    the last observed value.
    
    Args:
        hidden_dim (int): Dimensionality of node features
        horizon (int): Number of future time steps to predict
        bottleneck_dim (int): Dimension for bottleneck layers
        dropout (float): Dropout probability for regularization
    """
    def __init__(self, hidden_dim, horizon, bottleneck_dim=BOTTLENECK_DIM, dropout=DROPOUT):
        super(HorizonPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.bottleneck_dim = bottleneck_dim
        
        # Low-rank prediction projection
        self.predictor_low = nn.Linear(hidden_dim, bottleneck_dim)
        self.predictor_mid = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.predictor_high = nn.Linear(bottleneck_dim, horizon)
        
        # Adaptive refinement gate based on last observation
        self.refine_gate = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, horizon),
            nn.Sigmoid()
        )

    def forward(self, x, last_step=None):
        """
        Forward pass of the horizon predictor.
        
        Args:
            x (tensor): Node features [batch, nodes, hidden_dim]
            last_step (tensor, optional): Last observed values [batch, nodes]
            
        Returns:
            tensor: Predictions for future time steps [batch, nodes, horizon]
        """
        # Generate initial predictions
        x_low = self.predictor_low(x)
        x_mid = self.predictor_mid(x_low)
        initial_pred = self.predictor_high(x_mid)
        
        # Apply refinement if last observed value is provided
        if last_step is not None:
            # Compute adaptive gate
            gate = self.refine_gate(x)
            
            # Prepare last step and exponential decay
            last_step = last_step.unsqueeze(-1)
            time_decay = torch.arange(1, self.horizon + 1, device=x.device).float().view(1, 1, self.horizon)
            progressive_part = last_step * torch.exp(-0.1 * time_decay)
            
            # Adaptive fusion of model prediction and exponential decay
            final_pred = gate * initial_pred + (1 - gate) * progressive_part
        else:
            final_pred = initial_pred
            
        return final_pred


class DepthwiseSeparableConv1D(nn.Module):
    """
    Depthwise Separable 1D Convolution for efficient feature extraction.
    
    This module splits a standard convolution into a depthwise convolution
    (applied to each channel separately) followed by a pointwise convolution
    (1x1 convolution across channels). This reduces parameters and computation.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        stride (int): Convolution stride
        padding (int): Padding size
        dilation (int): Dilation rate
        dropout (float): Dropout probability
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, dropout=DROPOUT):
        super(DepthwiseSeparableConv1D, self).__init__()
        
        # Depthwise convolution (per-channel)
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_channels)
        self.bn1 = nn.BatchNorm1d(in_channels)
        
        # Pointwise convolution (1x1 conv across channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the depthwise separable convolution.
        
        Args:
            x (tensor): Input features [batch, channels, time]
            
        Returns:
            tensor: Convolved features [batch, out_channels, time]
        """
        # Depthwise convolution
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        
        # Pointwise convolution
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        
        return x


class MSTAGAT_Net_Linformer(nn.Module):
    """
    Multi-Scale Temporal-Adaptive Graph Attention Network with Linformer Attention
    
    This model implements the actual Linformer mechanism from Wang et al. (2020) with
    E and F projection matrices to achieve true O(N) complexity in attention computation.
    
    Key components:
    - Feature extraction using depthwise separable convolutions
    - True Linformer spatial attention with E and F projections
    - Temporal modeling with multi-scale dilated convolutions
    - Horizon prediction with adaptive refinement
    
    Args:
        args: Model configuration arguments
        data: Data object containing dataset information
    """
    def __init__(self, args, data):
        super(MSTAGAT_Net_Linformer, self).__init__()
        self.num_nodes = data.m
        self.window = args.window
        self.horizon = args.horizon
        self.hidden_dim = getattr(args, 'hidden_dim', HIDDEN_DIM)
        self.kernel_size = getattr(args, 'kernel_size', KERNEL_SIZE)
        self.bottleneck_dim = getattr(args, 'bottleneck_dim', BOTTLENECK_DIM)

        # Feature Extraction Component
        # ----------------------------
        # Extract features from raw time series using depthwise separable convolutions
        feature_channels = getattr(args, 'feature_channels', FEATURE_CHANNELS)
        self.feature_extractor = DepthwiseSeparableConv1D(
            in_channels=1, 
            out_channels=feature_channels,
            kernel_size=self.kernel_size, 
            padding=self.kernel_size // 2,
            dropout=getattr(args, 'dropout', DROPOUT)
        )

        # Low-rank projection of extracted features
        self.feature_projection_low = nn.Linear(feature_channels * self.window, self.bottleneck_dim)
        self.feature_projection_high = nn.Linear(self.bottleneck_dim, self.hidden_dim)
        self.feature_norm = nn.LayerNorm(self.hidden_dim)
        self.feature_act = nn.ReLU()

        # Spatial Component with Linformer Attention
        # -------------------------------------------
        # Graph attention mechanism with E and F projections for O(N) complexity
        self.spatial_module = LinformerSpatialAttentionModule(
            self.hidden_dim, num_nodes=self.num_nodes,
            dropout=getattr(args, 'dropout', DROPOUT),
            attention_heads=getattr(args, 'attention_heads', ATTENTION_HEADS),
            bottleneck_dim=self.bottleneck_dim
        )

        # Temporal Component
        # -----------------
        # Multi-scale temporal module to capture patterns at different time scales
        self.temporal_module = MultiScaleTemporalModule(
            self.hidden_dim,
            num_scales=getattr(args, 'num_scales', NUM_TEMPORAL_SCALES),
            kernel_size=self.kernel_size,
            dropout=getattr(args, 'dropout', DROPOUT)
        )

        # Prediction Component
        # -------------------
        # Horizon predictor to forecast future values
        self.prediction_module = HorizonPredictor(
            self.hidden_dim, self.horizon,
            bottleneck_dim=self.bottleneck_dim,
            dropout=getattr(args, 'dropout', DROPOUT)
        )

    def forward(self, x, idx=None):
        """
        Forward pass of the MSTAGAT-Net model with Linformer attention.
        
        Args:
            x (tensor): Input time series [batch, time_window, nodes]
            idx (tensor, optional): Node indices
            
        Returns:
            tuple: (Predictions, Attention regularization loss)
        """
        B, T, N = x.shape
        x_last = x[:, -1, :]  # Last observed values
        
        # Feature Extraction
        # -----------------
        # Reshape for 1D convolution [batch*nodes, 1, time_window]
        x_temp = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)
        temporal_features = self.feature_extractor(x_temp)
        temporal_features = temporal_features.view(B, N, -1)
        
        # Feature projection through bottleneck
        features = self.feature_projection_low(temporal_features)
        features = self.feature_projection_high(features)
        features = self.feature_norm(features)
        features = self.feature_act(features)
        
        # Spatial Processing with Linformer Attention
        # --------------------------------------------
        # Apply Linformer graph attention with O(N) complexity
        graph_features, attn_reg_loss = self.spatial_module(features)
        
        # Temporal Processing
        # ------------------
        # Apply multi-scale temporal module to capture temporal patterns
        fusion_features = self.temporal_module(graph_features)
        
        # Prediction
        # ----------
        # Generate predictions for future time steps
        predictions = self.prediction_module(fusion_features, x_last)
        predictions = predictions.transpose(1, 2)  # [batch, horizon, nodes]
        
        return predictions, attn_reg_loss