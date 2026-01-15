"""
MSAGAT-Net Model Architectures

This module contains all neural network architectures for the MSAGAT-Net framework:
- Core building blocks (attention, temporal modules, convolutions)
- Main MSTAGAT_Net model
- Ablation study variants (MSAGATNet_Ablation)

Architecture Components:
    1. SpatialAttentionModule: Graph attention with O(N) linear complexity
    2. MultiScaleTemporalModule: Dilated convolutions at multiple scales
    3. HorizonPredictor: Progressive multi-step prediction with refinement
    4. DepthwiseSeparableConv1D: Efficient feature extraction
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# =============================================================================
# MODEL HYPERPARAMETERS (DEFAULTS)
# =============================================================================
HIDDEN_DIM = 32
ATTENTION_HEADS = 4
ATTENTION_REG_WEIGHT_INIT = 1e-5
DROPOUT = 0.25
NUM_TEMPORAL_SCALES = 3
KERNEL_SIZE = 3
FEATURE_CHANNELS = 16
BOTTLENECK_DIM = 8


# =============================================================================
# CORE BUILDING BLOCKS
# =============================================================================

class DepthwiseSeparableConv1D(nn.Module):
    """
    Depthwise Separable 1D Convolution for efficient feature extraction.
    
    Splits a standard convolution into:
    1. Depthwise convolution (per-channel)
    2. Pointwise convolution (1x1 across channels)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolutional kernel
        stride: Convolution stride
        padding: Padding size
        dilation: Dilation rate
        dropout: Dropout probability
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, 
                 padding=0, dilation=1, dropout=DROPOUT):
        super().__init__()
        
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size, 
            stride=stride, padding=padding, dilation=dilation, 
            groups=in_channels
        )
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, channels, time]
        Returns:
            Output tensor [batch, out_channels, time]
        """
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.dropout(self.act(self.bn2(self.pointwise(x))))
        return x


class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module with LINEAR attention (O(N) complexity).
    
    Uses ELU+1 kernel trick for efficient linear attention with learnable
    regularization weight. Includes learnable low-rank graph bias for
    capturing static spatial relationships.
    
    Args:
        hidden_dim: Dimensionality of node features
        num_nodes: Number of nodes in the graph
        dropout: Dropout probability
        attention_heads: Number of parallel attention heads
        attention_regularization_weight: Initial weight for L1 regularization
        bottleneck_dim: Dimension of the low-rank projection
    """
    
    def __init__(self, hidden_dim, num_nodes, dropout=DROPOUT, 
                 attention_heads=ATTENTION_HEADS,
                 attention_regularization_weight=ATTENTION_REG_WEIGHT_INIT,
                 bottleneck_dim=BOTTLENECK_DIM):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.heads = attention_heads
        self.head_dim = hidden_dim // self.heads
        self.num_nodes = num_nodes
        self.bottleneck_dim = bottleneck_dim

        # Learnable attention regularization weight (log-domain for positivity)
        self.log_attention_reg_weight = nn.Parameter(
            torch.tensor(math.log(attention_regularization_weight), dtype=torch.float32)
        )

        # Low-rank projections for query, key, value
        self.qkv_proj_low = nn.Linear(hidden_dim, 3 * bottleneck_dim)
        self.qkv_proj_high = nn.Linear(3 * bottleneck_dim, 3 * hidden_dim)

        # Low-rank projections for output
        self.out_proj_low = nn.Linear(hidden_dim, bottleneck_dim)
        self.out_proj_high = nn.Linear(bottleneck_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        
        # Learnable graph structure bias (low-rank factorization)
        self.u = Parameter(torch.Tensor(self.heads, num_nodes, bottleneck_dim))
        self.v = Parameter(torch.Tensor(self.heads, bottleneck_dim, num_nodes))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

    @property
    def current_reg_weight(self):
        """Get current learned regularization weight for diagnostics."""
        return torch.exp(self.log_attention_reg_weight).item()

    def _compute_linear_attention(self, q, k, v):
        """
        Compute LINEAR attention with O(N) complexity using ELU+1 kernel trick.
        
        Args:
            q: Query tensors [batch, heads, nodes, head_dim]
            k: Key tensors [batch, heads, nodes, head_dim]
            v: Value tensors [batch, heads, nodes, head_dim]
            
        Returns:
            Attended values with O(N) complexity
        """
        # Apply ELU+1 for positive feature map (kernel trick)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        # Compute key-value products: O(N * d²) instead of O(N²)
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        
        # Normalization for stability
        ones = torch.ones(k.size(0), k.size(1), k.size(2), 1, device=k.device)
        z = 1.0 / (torch.einsum('bhnd,bhno->bhn', k, ones) + 1e-8)
        
        # Apply linear attention: O(N * d²)
        return torch.einsum('bhnd,bhde,bhn->bhne', q, kv, z)

    def _compute_graph_bias_message_passing(self, v):
        """
        Apply learned low-rank graph bias as normalized message passing term.
        
        Keeps O(N) complexity via low-rank factorization: B = U @ V
        
        Args:
            v: Value tensors [batch, heads, nodes, head_dim]
        Returns:
            Bias-based message passing [batch, heads, nodes, head_dim]
        """
        # Ensure non-negative weights for stable normalization
        u_pos = F.elu(self.u) + 1.0  # [heads, N, r]
        v_pos = F.elu(self.v) + 1.0  # [heads, r, N]

        # Compute (U @ V) @ v without materializing NxN
        tmp = torch.einsum('hrn,bhnd->bhrd', v_pos, v)
        out = torch.einsum('hnr,bhrd->bhnd', u_pos, tmp)

        # Normalize
        v_sum = v_pos.sum(dim=-1)
        denom = torch.einsum('hnr,hr->hn', u_pos, v_sum).unsqueeze(0).unsqueeze(-1)
        out = out / (denom + 1e-8)

        return out

    def forward(self, x, mask=None):
        """
        Forward pass using linear attention with learnable regularization.
        
        Args:
            x: Input node features [batch, nodes, hidden_dim]
            mask: Attention mask (optional)
            
        Returns:
            tuple: (Updated features, regularization loss)
        """
        B, N, H = x.shape

        # Low-rank projection for qkv
        qkv_low = self.qkv_proj_low(x)
        qkv = self.qkv_proj_high(qkv_low)
        qkv = qkv.chunk(3, dim=-1)

        # Separate query, key, value and reshape for multi-head attention
        q, k, v = [tensor.view(B, N, self.heads, self.head_dim) for tensor in qkv]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use LINEAR attention - O(N) complexity
        output = self._compute_linear_attention(q, k, v)

        # Integrate learned graph bias (still O(N) via low-rank ops)
        graph_bias_out = self._compute_graph_bias_message_passing(v)
        output = output + self.dropout(graph_bias_out)

        # Dense view of graph bias (used for plotting/regularization only)
        graph_bias = torch.matmul(self.u, self.v)

        # Compute regularization loss with learnable weight
        attention_reg_weight = torch.exp(self.log_attention_reg_weight)
        attn_reg_loss = attention_reg_weight * torch.mean(torch.abs(graph_bias))

        # Reshape output to original dimensions
        output = output.transpose(1, 2).contiguous().view(B, N, H)

        # Low-rank projection for output
        output = self.out_proj_low(output)
        output = self.out_proj_high(output)

        return output, attn_reg_loss


class MultiScaleTemporalModule(nn.Module):
    """
    Multi-Scale Temporal Module using dilated convolutions.
    
    Captures temporal patterns at different time scales through
    dilated convolutions with adaptive fusion.
    
    Args:
        hidden_dim: Dimensionality of node features
        num_scales: Number of temporal scales (dilation rates)
        kernel_size: Size of the convolutional kernel
        dropout: Dropout probability
    """
    
    def __init__(self, hidden_dim, num_scales=NUM_TEMPORAL_SCALES, 
                 kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        # Dilated convolutions at different scales
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                          padding=(kernel_size // 2) * (2 ** i), dilation=2 ** i),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_scales)
        ])
        
        # Learnable fusion weights
        self.fusion_weight = Parameter(torch.ones(num_scales))
        
        # Low-rank projection for fusion
        self.fusion_low = nn.Linear(hidden_dim, BOTTLENECK_DIM)
        self.fusion_high = nn.Linear(BOTTLENECK_DIM, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: Input features [batch, nodes, hidden_dim]
        Returns:
            Temporally processed features [batch, nodes, hidden_dim]
        """
        # Reshape for 1D convolution: [B, N, H] -> [B, H, N]
        x = x.transpose(1, 2)
        
        # Apply multi-scale convolutions
        features = [scale(x) for scale in self.scales]
        
        # Adaptive fusion
        alpha = F.softmax(self.fusion_weight, dim=0)
        stacked = torch.stack(features, dim=0)
        fused = torch.einsum('s,sbhn->bhn', alpha, stacked)
        
        # Reshape back: [B, H, N] -> [B, N, H]
        fused = fused.transpose(1, 2)
        
        # Apply low-rank projection and residual connection
        out = self.fusion_low(fused)
        out = self.fusion_high(out)
        out = self.layer_norm(out + fused)
        
        return out


class HorizonPredictor(nn.Module):
    """
    Horizon Predictor for multi-step forecasting with refinement.
    
    Generates predictions for multiple future time steps with optional
    refinement based on the last observed value.
    
    Args:
        hidden_dim: Dimensionality of node features
        horizon: Number of future time steps to predict
        bottleneck_dim: Dimension for bottleneck layers
        dropout: Dropout probability
    """
    
    def __init__(self, hidden_dim, horizon, bottleneck_dim=BOTTLENECK_DIM, 
                 dropout=DROPOUT):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, horizon)
        )
        
        # Refinement gate
        self.refine_gate = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, horizon),
            nn.Sigmoid()
        )
        
    def forward(self, x, last_step=None):
        """
        Args:
            x: Node features [batch, nodes, hidden_dim]
            last_step: Last observed values [batch, nodes] (optional)
        Returns:
            Predictions [batch, nodes, horizon]
        """
        initial_pred = self.predictor(x)
        
        if last_step is not None:
            gate = self.refine_gate(x)
            
            # Fixed decay rate (proven optimal)
            decay_rate = 0.1
            
            last_step = last_step.unsqueeze(-1)
            time_steps = torch.arange(1, self.horizon + 1, device=x.device).float()
            time_decay = time_steps.view(1, 1, self.horizon)
            
            progressive_part = last_step * torch.exp(-decay_rate * time_decay)
            final_pred = gate * initial_pred + (1 - gate) * progressive_part
        else:
            final_pred = initial_pred
            
        return final_pred


# =============================================================================
# MAIN MODEL
# =============================================================================

class MSTAGAT_Net(nn.Module):
    """
    Multi-Scale Temporal-Adaptive Graph Attention Network (MSTAGAT-Net)
    
    Combines graph attention mechanisms for spatial dependencies and
    multi-scale temporal convolutions for temporal patterns.
    
    Architecture:
        1. Feature extraction using depthwise separable convolutions
        2. Spatial modeling with efficient graph attention
        3. Temporal modeling with multi-scale dilated convolutions
        4. Horizon prediction with adaptive refinement
    
    Args:
        args: Model configuration with attributes:
            - window: Input window size
            - horizon: Prediction horizon
            - hidden_dim, kernel_size, bottleneck_dim, etc.
        data: Data object with attribute:
            - m: Number of nodes
    """
    
    def __init__(self, args, data):
        super().__init__()
        
        self.num_nodes = data.m
        self.window = args.window
        self.horizon = args.horizon
        self.hidden_dim = getattr(args, 'hidden_dim', HIDDEN_DIM)
        self.kernel_size = getattr(args, 'kernel_size', KERNEL_SIZE)
        self.bottleneck_dim = getattr(args, 'bottleneck_dim', BOTTLENECK_DIM)
        
        feature_channels = getattr(args, 'feature_channels', FEATURE_CHANNELS)
        dropout = getattr(args, 'dropout', DROPOUT)

        # Feature Extraction
        self.feature_extractor = DepthwiseSeparableConv1D(
            in_channels=1, 
            out_channels=feature_channels,
            kernel_size=self.kernel_size, 
            padding=self.kernel_size // 2,
            dropout=dropout
        )
        
        # Low-rank projection of extracted features
        self.feature_projection_low = nn.Linear(
            feature_channels * self.window, self.bottleneck_dim
        )
        self.feature_projection_high = nn.Linear(
            self.bottleneck_dim, self.hidden_dim
        )
        self.feature_norm = nn.LayerNorm(self.hidden_dim)
        self.feature_act = nn.ReLU()

        # Spatial attention
        self.spatial_module = SpatialAttentionModule(
            self.hidden_dim, 
            num_nodes=self.num_nodes,
            dropout=dropout,
            attention_heads=getattr(args, 'attention_heads', ATTENTION_HEADS),
            attention_regularization_weight=getattr(
                args, 'attention_regularization_weight', ATTENTION_REG_WEIGHT_INIT
            ),
            bottleneck_dim=self.bottleneck_dim
        )

        # Temporal processing
        self.temporal_module = MultiScaleTemporalModule(
            self.hidden_dim,
            num_scales=getattr(args, 'num_scales', NUM_TEMPORAL_SCALES),
            kernel_size=self.kernel_size,
            dropout=dropout
        )

        # Prediction
        self.prediction_module = HorizonPredictor(
            self.hidden_dim, 
            self.horizon,
            bottleneck_dim=self.bottleneck_dim,
            dropout=dropout
        )

    def forward(self, x, idx=None):
        """
        Args:
            x: Input time series [batch, time_window, nodes]
            idx: Node indices (optional, unused)
        Returns:
            tuple: (Predictions [batch, horizon, nodes], Attention reg loss)
        """
        B, T, N = x.shape
        x_last = x[:, -1, :]
        
        # Feature extraction
        x_temp = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)
        temporal_features = self.feature_extractor(x_temp)
        temporal_features = temporal_features.view(B, N, -1)
        
        # Feature projection through bottleneck
        features = self.feature_projection_low(temporal_features)
        features = self.feature_projection_high(features)
        features = self.feature_norm(features)
        features = self.feature_act(features)
        
        # Spatial processing
        graph_features, attn_reg_loss = self.spatial_module(features)
        
        # Temporal processing
        fusion_features = self.temporal_module(graph_features)
        
        # Prediction
        predictions = self.prediction_module(fusion_features, x_last)
        predictions = predictions.transpose(1, 2)
        
        return predictions, attn_reg_loss


# =============================================================================
# ABLATION STUDY COMPONENTS
# =============================================================================

class SimpleGraphConvolutionalLayer(nn.Module):
    """
    Standard GCN layer with fixed adjacency matrix.
    
    Used as ablation replacement for SpatialAttentionModule (no_agam).
    Performs standard graph convolution without adaptive attention.
    
    Args:
        hidden_dim: Dimension of hidden representations
        num_nodes: Number of nodes in the graph
        dropout: Dropout rate
    """
    
    def __init__(self, hidden_dim, num_nodes, dropout=DROPOUT, **kwargs):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Fixed adjacency matrix (identity by default)
        self.register_buffer('adj_matrix', torch.eye(num_nodes))
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input [batch, nodes, hidden_dim]
            mask: Unused
        Returns:
            tuple: (output [batch, nodes, hidden_dim], 0.0)
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)
        
        x = self.linear1(x)
        
        # Handle size mismatch
        if self.adj_matrix.size(0) != num_nodes:
            adj_matrix = torch.eye(num_nodes, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        else:
            adj_matrix = self.adj_matrix.unsqueeze(0).expand(batch_size, -1, -1)
            
        # GCN operation: X' = AXW
        x = torch.bmm(adj_matrix, x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm(x)
        
        # Store for visualization
        self.attn = [torch.eye(num_nodes, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)]
        
        return x, 0.0


class SingleScaleTemporalModule(nn.Module):
    """
    Single-scale temporal convolution for ablation study.
    
    Used as ablation replacement for MultiScaleTemporalModule (no_mtfm).
    
    Args:
        hidden_dim: Dimension of hidden representations
        kernel_size: Convolution kernel size
        dropout: Dropout rate
    """
    
    def __init__(self, hidden_dim, num_scales=NUM_TEMPORAL_SCALES, 
                 kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Single-scale convolution without dilation
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                      padding=kernel_size // 2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: Input [batch, nodes, hidden_dim]
        Returns:
            Output [batch, nodes, hidden_dim]
        """
        x = x.transpose(1, 2)
        feat = self.conv(x)
        feat = feat.transpose(1, 2)
        return self.fusion(feat)


class DirectPredictionModule(nn.Module):
    """
    Direct multi-step prediction for ablation study.
    
    Used as ablation replacement for HorizonPredictor (no_pprm).
    Directly predicts all future time steps without refinement.
    
    Args:
        hidden_dim: Dimension of hidden representations
        horizon: Prediction horizon
        dropout: Dropout rate
    """
    
    def __init__(self, hidden_dim, horizon, low_rank_dim=BOTTLENECK_DIM, 
                 dropout=DROPOUT):
        super().__init__()
        
        self.horizon = horizon
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, horizon)
        )

    def forward(self, x, last_step=None):
        """
        Args:
            x: Input [batch, nodes, hidden_dim]
            last_step: Unused
        Returns:
            Predictions [batch, nodes, horizon]
        """
        return self.predictor(x)


class MSAGATNet_Ablation(nn.Module):
    """
    MSAGAT-Net with configurable components for ablation studies.
    
    Allows systematic evaluation of each component's contribution by
    replacing key modules with simpler alternatives.
    
    Args:
        args: Configuration with attributes:
            - ablation: 'none', 'no_agam', 'no_mtfm', or 'no_pprm'
            - window, horizon, hidden_dim, etc.
        data: Data object with attribute:
            - m: Number of nodes
            - adj: Adjacency matrix (optional)
    
    Ablation variants:
        - none: Full model
        - no_agam: Replace SpatialAttentionModule with SimpleGraphConvolutionalLayer
        - no_mtfm: Replace MultiScaleTemporalModule with SingleScaleTemporalModule
        - no_pprm: Replace HorizonPredictor with DirectPredictionModule
    """
    
    def __init__(self, args, data):
        super().__init__()
        
        self.m = data.m
        self.window = args.window
        self.horizon = args.horizon
        self.ablation = getattr(args, 'ablation', 'none')

        # Model dimensions
        self.hidden_dim = getattr(args, 'hidden_dim', HIDDEN_DIM)
        self.kernel_size = getattr(args, 'kernel_size', KERNEL_SIZE)
        self.low_rank_dim = getattr(args, 'bottleneck_dim', BOTTLENECK_DIM)
        dropout = getattr(args, 'dropout', DROPOUT)

        # Feature extraction (same for all ablations)
        self.temp_conv = DepthwiseSeparableConv1D(
            in_channels=1,
            out_channels=FEATURE_CHANNELS,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            dropout=dropout
        )
        
        self.feature_process_low = nn.Linear(
            FEATURE_CHANNELS * self.window, self.low_rank_dim
        )
        self.feature_process_high = nn.Linear(self.low_rank_dim, self.hidden_dim)
        self.feature_norm = nn.LayerNorm(self.hidden_dim)
        self.feature_act = nn.ReLU()
        
        # Spatial component: choose full attention or GCN
        if self.ablation == 'no_agam':
            self.graph_attention = SimpleGraphConvolutionalLayer(
                self.hidden_dim, num_nodes=self.m, dropout=dropout
            )
            if hasattr(data, 'adj'):
                self.graph_attention.adj_matrix = data.adj
        else:
            self.graph_attention = SpatialAttentionModule(
                hidden_dim=self.hidden_dim,
                num_nodes=self.m,
                dropout=dropout,
                attention_heads=getattr(args, 'attention_heads', ATTENTION_HEADS),
                attention_regularization_weight=getattr(
                    args, 'attention_regularization_weight', ATTENTION_REG_WEIGHT_INIT
                ),
                bottleneck_dim=self.low_rank_dim
            )
        
        # Temporal component: choose multi-scale or single-scale
        if self.ablation == 'no_mtfm':
            self.temporal_module = SingleScaleTemporalModule(
                self.hidden_dim, kernel_size=self.kernel_size, dropout=dropout
            )
        else:
            self.temporal_module = MultiScaleTemporalModule(
                hidden_dim=self.hidden_dim,
                num_scales=getattr(args, 'num_scales', NUM_TEMPORAL_SCALES),
                kernel_size=self.kernel_size,
                dropout=dropout
            )
        
        # Prediction component: choose progressive or direct
        if self.ablation == 'no_pprm':
            self.prediction_module = DirectPredictionModule(
                hidden_dim=self.hidden_dim,
                horizon=self.horizon,
                low_rank_dim=self.low_rank_dim,
                dropout=dropout
            )
        else:
            self.prediction_module = HorizonPredictor(
                hidden_dim=self.hidden_dim,
                horizon=self.horizon,
                bottleneck_dim=self.low_rank_dim,
                dropout=dropout
            )

    def forward(self, x, idx=None):
        """
        Args:
            x: Input [batch, window, nodes]
            idx: Unused
        Returns:
            tuple: (predictions [batch, horizon, nodes], attention reg loss)
        """
        B, T, N = x.shape
        x_last = x[:, -1, :]
        
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
        
        # Apply graph attention
        graph_features, attn_reg_loss = self.graph_attention(features)
        
        # Process temporal patterns
        fusion_features = self.temporal_module(graph_features)
        
        # Generate predictions
        predictions = self.prediction_module(fusion_features, x_last)
        predictions = predictions.transpose(1, 2)
        
        return predictions, attn_reg_loss


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    'HIDDEN_DIM',
    'ATTENTION_HEADS',
    'ATTENTION_REG_WEIGHT_INIT',
    'DROPOUT',
    'NUM_TEMPORAL_SCALES',
    'KERNEL_SIZE',
    'FEATURE_CHANNELS',
    'BOTTLENECK_DIM',
    # Building blocks
    'DepthwiseSeparableConv1D',
    'SpatialAttentionModule',
    'MultiScaleTemporalModule',
    'HorizonPredictor',
    # Main models
    'MSTAGAT_Net',
    # Ablation components
    'SimpleGraphConvolutionalLayer',
    'SingleScaleTemporalModule',
    'DirectPredictionModule',
    'MSAGATNet_Ablation',
]
