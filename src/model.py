import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# Model configuration parameters
# HIDDEN_DIM = 16
# ATTENTION_HEADS = 4
# ATTENTION_REG_WEIGHT = 1e-05
# DROPOUT = 0.13813778561947987
# NUM_SCALES = 3
# KERNEL_SIZE = 3
# TEMP_CONV_OUT_CHANNELS = 16
# LOW_RANK_DIM = 8

# Model configuration parameters
HIDDEN_DIM = 32
ATTENTION_HEADS = 4
ATTENTION_REG_WEIGHT = 1e-05
DROPOUT = 0.355
NUM_SCALES = 4
KERNEL_SIZE = 3
TEMP_CONV_OUT_CHANNELS = 16
LOW_RANK_DIM = 8


class EfficientAdaptiveGraphAttentionModule(nn.Module):
    """
    Efficient Adaptive Graph Attention Module with linear attention and low-rank decomposition.
    
    This is NOT a standard graph attention module, but a specialized version that uses:
    1. Linear attention computation (ELU+1 kernel trick) for O(N) complexity
    2. Low-rank factorization for QKV projections
    3. Factorized adjacency matrix (U×V^T) for memory efficiency
    4. L1 regularization for interpretable attention patterns
    """

    def __init__(self, hidden_dim, num_nodes, dropout=DROPOUT, attn_heads=ATTENTION_HEADS, 
                 attn_reg_weight=ATTENTION_REG_WEIGHT, low_rank_dim=LOW_RANK_DIM):
        super(EfficientAdaptiveGraphAttentionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = attn_heads
        self.head_dim = hidden_dim // self.heads
        self.num_nodes = num_nodes
        self.attn_reg_weight = attn_reg_weight
        self.low_rank_dim = low_rank_dim
        
        # Low-rank projections for query, key, value
        self.qkv_proj_low = nn.Linear(hidden_dim, 3 * low_rank_dim)
        self.qkv_proj_high = nn.Linear(3 * low_rank_dim, 3 * hidden_dim)
        
        # Output projection with low-rank decomposition
        self.out_proj_low = nn.Linear(hidden_dim, low_rank_dim)
        self.out_proj_high = nn.Linear(low_rank_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        
        # Learnable adjacency factors (U * V^T for memory efficiency)
        self.learnable_adj_u = Parameter(torch.Tensor(self.heads, num_nodes, low_rank_dim))
        self.learnable_adj_v = Parameter(torch.Tensor(self.heads, low_rank_dim, num_nodes))
        nn.init.xavier_uniform_(self.learnable_adj_u)
        nn.init.xavier_uniform_(self.learnable_adj_v)

    def _linearized_attention(self, q, k, v):
        """
        Implements attention with linear complexity using kernel feature maps.
        """
        # Apply ELU+1 for positive feature mapping
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        # Linear attention computation (O(N) complexity)
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        
        # Compute normalization factor
        ones = torch.ones(k.size(0), k.size(1), k.size(2), 1, device=k.device)
        z = 1.0 / (torch.einsum('bhnd,bhno->bhn', k, ones) + 1e-8)
        
        # Apply normalized attention
        output = torch.einsum('bhnd,bhde,bhn->bhne', q, kv, z)
        
        return output
    
    def forward(self, x, mask=None):
        """
        Forward pass of the graph attention module.
        
        Args:
            x: Input tensor [batch_size, num_nodes, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, attention regularization loss)
        """
        B, N, H = x.shape
        
        # Low-rank QKV projection
        qkv_low = self.qkv_proj_low(x)
        qkv = self.qkv_proj_high(qkv_low)
        
        # Split into query, key, value
        qkv = qkv.chunk(3, dim=-1)
        q, k, v = [x.view(B, N, self.heads, self.head_dim) for x in qkv]
        
        # Rearrange for attention computation
        q = q.transpose(1, 2)  # [B, heads, N, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute linearized attention
        output = self._linearized_attention(q, k, v)
        
        # Compute factorized adjacency bias
        adj_bias = torch.matmul(self.learnable_adj_u, self.learnable_adj_v)
        
        # Store attention for regularization
        self.attn = F.softmax(torch.einsum('bhnd,bhmd->bhnm', q, k) / math.sqrt(self.head_dim) + adj_bias, dim=-1)
        
        # Compute regularization loss for sparsity
        attn_reg_loss = self.attn_reg_weight * torch.mean(torch.abs(self.attn))
        
        # Restore original shape
        output = output.transpose(1, 2).contiguous().view(B, N, H)
        
        # Low-rank output projection
        output = self.out_proj_low(output)
        output = self.out_proj_high(output)
        
        return output, attn_reg_loss


class DilatedMultiScaleTemporalModule(nn.Module):
    """
    Dilated Multi-Scale Temporal Module for processing time-series data at different resolutions.
    
    Uses parallel dilated convolutions to capture temporal patterns at various scales
    with an adaptive fusion mechanism.
    """

    def __init__(self, hidden_dim, num_scales=NUM_SCALES, kernel_size=KERNEL_SIZE, 
                 dropout=DROPOUT):
        super(DilatedMultiScaleTemporalModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales

        # Multi-scale dilated convolutions
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                          padding=(kernel_size//2) * 2**i, dilation=2**i),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for i in range(num_scales)
        ])
        
        # Adaptive fusion weights
        self.fusion_weight = Parameter(torch.ones(num_scales), requires_grad=True)
        
        # Fusion layer
        self.fusion_low = nn.Linear(hidden_dim, LOW_RANK_DIM)
        self.fusion_high = nn.Linear(LOW_RANK_DIM, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Forward pass of the temporal module.
        
        Args:
            x: Input tensor [batch_size, num_nodes, hidden_dim]
            
        Returns:
            Processed tensor with multi-scale temporal features
        """
        # Transpose for temporal convolution: [B, N, H] -> [B, H, N]
        x = x.transpose(1, 2)
        
        # Process at different temporal scales
        features = []
        for scale in self.scales:
            feat = scale(x)
            features.append(feat)
        
        # Compute adaptive fusion weights
        alpha = F.softmax(self.fusion_weight, dim=0)
        
        # Weight and combine features from different scales
        stacked = torch.stack(features, dim=1)
        fused = torch.sum(alpha.view(1, self.num_scales, 1, 1) * stacked, dim=1)
        
        # Restore dimensions and apply fusion
        fused = fused.transpose(1, 2)
        out = self.fusion_low(fused)
        out = self.fusion_high(out)
        out = self.layer_norm(out + fused)  # Add residual connection
        
        return out


class ProgressivePredictionModule(nn.Module):
    """
    Progressive Prediction Module for multi-step forecasting with refinement.
    
    Generates initial predictions and progressively refines them using
    a gating mechanism that incorporates recent observations with time decay.
    """

    def __init__(self, hidden_dim, horizon, low_rank_dim=LOW_RANK_DIM, dropout=DROPOUT):
        super(ProgressivePredictionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.low_rank_dim = low_rank_dim

        # Predictor with factorization for efficiency
        self.predictor_low = nn.Linear(hidden_dim, low_rank_dim)
        self.predictor_mid = nn.Sequential(
            nn.LayerNorm(low_rank_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.predictor_high = nn.Linear(low_rank_dim, horizon)
        
        # Refinement gate
        self.refine_gate = nn.Sequential(
            nn.Linear(hidden_dim, low_rank_dim),
            nn.ReLU(),
            nn.Linear(low_rank_dim, horizon),
            nn.Sigmoid()
        )
        
    def forward(self, x, last_step=None):
        """
        Forward pass of the prediction module.
        
        Args:
            x: Input tensor [batch_size, num_nodes, hidden_dim]
            last_step: Last observation [batch_size, num_nodes]
            
        Returns:
            Predictions tensor [batch_size, num_nodes, horizon]
        """
        # Generate initial multi-step prediction
        x_low = self.predictor_low(x)
        x_mid = self.predictor_mid(x_low)
        initial_pred = self.predictor_high(x_mid)
        
        if last_step is not None:
            # Compute refinement gate
            gate = self.refine_gate(x)
            
            # Prepare for refinement
            last_step = last_step.unsqueeze(-1)
            
            # Apply time-based exponential decay
            time_decay = torch.arange(1, self.horizon + 1, device=x.device).float()
            time_decay = time_decay.view(1, 1, self.horizon)
            
            # Generate decay component
            progressive_part = last_step * torch.exp(-0.1 * time_decay)
            
            # Combine predictions with adaptive gating
            final_pred = gate * initial_pred + (1 - gate) * progressive_part
        else:
            final_pred = initial_pred
        
        return final_pred


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable 1D Convolution for efficient temporal processing.
    
    Reduces parameters by separating the depthwise convolution (spatial) 
    and pointwise convolution (channel mixing) operations while 
    maintaining representational capacity.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, 
                 dropout=DROPOUT):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                  padding=padding, dilation=dilation, groups=in_channels)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass of the separable convolution.
        
        Args:
            x: Input tensor [batch_size, in_channels, sequence_length]
            
        Returns:
            Processed tensor [batch_size, out_channels, sequence_length]
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


class MSAGATNet(nn.Module):
    """
    Multi-Scale Adaptive Graph Attention Spatiotemporal Network (MSAGAT-Net) for epidemic forecasting.
    
    A lightweight model that combines efficient graph attention for spatial relationships
    with multi-scale temporal processing for time-series forecasting.
    
    Key components:
    1. Efficient Adaptive Graph Attention Module for spatial dependencies with linear complexity
    2. Dilated Multi-Scale Temporal Module for time-series patterns at different resolutions
    3. Depthwise Separable Convolutions for parameter-efficient feature extraction
    4. Progressive Prediction Module with time-decay refinement for accurate forecasting
    """
    
    def __init__(self, args, data):
        super(MSAGATNet, self).__init__()
        self.m = data.m  # Number of nodes
        self.window = args.window  # Input window size
        self.horizon = args.horizon  # Prediction horizon

        # Model dimensions
        self.hidden_dim = getattr(args, 'hidden_dim', HIDDEN_DIM)
        self.kernel_size = getattr(args, 'kernel_size', KERNEL_SIZE)
        self.low_rank_dim = getattr(args, 'low_rank_dim', LOW_RANK_DIM)

        # Temporal feature extraction
        self.temp_conv = DepthwiseSeparableConv1d(
            in_channels=1, 
            out_channels=TEMP_CONV_OUT_CHANNELS,
            kernel_size=self.kernel_size, 
            padding=self.kernel_size // 2,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # Feature processing and compression
        self.feature_process_low = nn.Linear(TEMP_CONV_OUT_CHANNELS * self.window, self.low_rank_dim)
        self.feature_process_high = nn.Linear(self.low_rank_dim, self.hidden_dim)
        self.feature_norm = nn.LayerNorm(self.hidden_dim)
        self.feature_act = nn.ReLU()
        
        # Main components
        self.graph_attention = EfficientAdaptiveGraphAttentionModule(
            self.hidden_dim, num_nodes=self.m,
            dropout=getattr(args, 'dropout', DROPOUT),
            attn_heads=getattr(args, 'attn_heads', ATTENTION_HEADS),
            low_rank_dim=self.low_rank_dim
        )
        
        self.temporal_module = DilatedMultiScaleTemporalModule(
            self.hidden_dim,
            num_scales=getattr(args, 'num_scales', NUM_SCALES),
            kernel_size=self.kernel_size,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        self.prediction_module = ProgressivePredictionModule(
            self.hidden_dim, self.horizon,
            low_rank_dim=self.low_rank_dim,
            dropout=getattr(args, 'dropout', DROPOUT)
        )

    def forward(self, x, idx=None):
        """
        Forward pass of the MSAGAT-Net model.
        
        Args:
            x: Input tensor [batch_size, window, num_nodes]
            idx: Optional index tensor
            
        Returns:
            Tuple of (predictions, attention_regularization_loss)
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