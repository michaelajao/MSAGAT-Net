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
import numpy as np
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
DROPOUT = 0.2
NUM_TEMPORAL_SCALES = 4
KERNEL_SIZE = 3
FEATURE_CHANNELS = 16
BOTTLENECK_DIM = 8

# Graph size thresholds for adaptive configuration
SMALL_GRAPH_THRESHOLD = 20   # <= 20 nodes: use adj prior + graph bias
LARGE_GRAPH_THRESHOLD = 40   # >= 40 nodes: disable graph bias for efficiency


def get_adaptive_config(num_nodes):
    """
    Get adaptive hyperparameters based on graph size.
    
    Rationale from experiments:
    - Small graphs (<20 nodes): Benefit from adjacency prior and graph bias
    - Large graphs (>40 nodes): Pure learned attention performs better
    - Very large graphs (>200 nodes): Need larger hidden dimensions
    
    Args:
        num_nodes: Number of nodes in the graph
        
    Returns:
        dict: Recommended hyperparameters for the graph size
    """
    config = {
        'hidden_dim': HIDDEN_DIM,
        'bottleneck_dim': BOTTLENECK_DIM,
        'use_graph_bias': True,
        'use_adj_prior': False,
    }
    
    if num_nodes <= SMALL_GRAPH_THRESHOLD:
        # Small graphs: Use all priors
        config['use_graph_bias'] = True
        config['use_adj_prior'] = True
    elif num_nodes >= LARGE_GRAPH_THRESHOLD:
        # Large graphs: Learned attention only
        config['use_graph_bias'] = False
        config['use_adj_prior'] = False
    
    # Scale hidden dim for very large graphs
    if num_nodes > 200:
        config['hidden_dim'] = 64
        config['bottleneck_dim'] = 16
    elif num_nodes > 100:
        config['hidden_dim'] = 48
        config['bottleneck_dim'] = 12
        
    return config

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
    Low-rank Adaptive Graph Attention Module (LR-AGAM).
    
    Captures node relationships in a graph structure using an efficient
    attention mechanism with low-rank decomposition. Computes attention 
    scores between nodes and updates node representations accordingly.
    
    Args:
        hidden_dim: Dimensionality of node features
        num_nodes: Number of nodes in the graph
        dropout: Dropout probability for regularization
        attention_heads: Number of parallel attention heads
        attention_regularization_weight: Weight for L1 regularization on attention
        bottleneck_dim: Dimension of the low-rank projection
        use_adj_prior: Whether to use adjacency matrix as prior for attention
        use_graph_bias: Whether to use learnable graph structure bias
        adj_matrix: Predefined adjacency matrix [num_nodes, num_nodes] (optional)
    """
    
    def __init__(self, hidden_dim, num_nodes, dropout=DROPOUT, 
                 attention_heads=ATTENTION_HEADS,
                 attention_regularization_weight=ATTENTION_REG_WEIGHT_INIT,
                 bottleneck_dim=BOTTLENECK_DIM,
                 use_adj_prior=False,
                 use_graph_bias=True,
                 adj_matrix=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.heads = attention_heads
        self.head_dim = hidden_dim // self.heads
        self.num_nodes = num_nodes
        self.bottleneck_dim = bottleneck_dim
        self.use_adj_prior = use_adj_prior
        self.use_graph_bias = use_graph_bias

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
        
        # Learnable graph structure bias (low-rank) - only if use_graph_bias is True
        if self.use_graph_bias:
            self.u = Parameter(torch.Tensor(self.heads, num_nodes, bottleneck_dim))
            self.v = Parameter(torch.Tensor(self.heads, bottleneck_dim, num_nodes))
            nn.init.xavier_uniform_(self.u)
            nn.init.xavier_uniform_(self.v)
        
        # Register adjacency matrix as buffer if using adj_prior
        if self.use_adj_prior and adj_matrix is not None:
            if isinstance(adj_matrix, np.ndarray):
                adj_matrix = torch.from_numpy(adj_matrix).float()
            # Normalize adjacency matrix and expand for heads
            adj_norm = adj_matrix / (adj_matrix.sum(dim=-1, keepdim=True) + 1e-8)
            self.register_buffer('adj_prior', adj_norm.unsqueeze(0).expand(self.heads, -1, -1).clone())
        else:
            self.register_buffer('adj_prior', None)

    @property
    def current_reg_weight(self):
        """Get current learned regularization weight for diagnostics."""
        return torch.exp(self.log_attention_reg_weight).item()

    def _compute_attention(self, q, k, v):
        """
        Compute attention scores and apply them to values.
        
        Args:
            q: Query tensors [batch, heads, nodes, head_dim]
            k: Key tensors [batch, heads, nodes, head_dim]
            v: Value tensors [batch, heads, nodes, head_dim]
            
        Returns:
            Attended values
        """
        # Apply ELU activation + 1 for stability
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        # Compute key-value products
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        
        # Normalize keys for stability
        ones = torch.ones(k.size(0), k.size(1), k.size(2), 1, device=k.device)
        z = 1.0 / (torch.einsum('bhnd,bhno->bhn', k, ones) + 1e-8)
        
        # Apply attention mechanism
        return torch.einsum('bhnd,bhde,bhn->bhne', q, kv, z)

    def forward(self, x, mask=None):
        """
        Forward pass of the spatial attention module.
        
        Args:
            x: Input node features [batch, nodes, hidden_dim]
            mask: Attention mask (optional)
            
        Returns:
            tuple: (Updated node features, Attention regularization loss)
        """
        B, N, H = x.shape

        # Low-rank projection for qkv
        qkv_low = self.qkv_proj_low(x)
        qkv = self.qkv_proj_high(qkv_low)
        qkv = qkv.chunk(3, dim=-1)

        # Separate query, key, value and reshape for multi-head attention
        q, k, v = [tensor.view(B, N, self.heads, self.head_dim) for tensor in qkv]
        q = q.transpose(1, 2)  # [B, heads, N, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attended values
        output = self._compute_attention(q, k, v)
        
        # Compute attention scores
        attn_scores = torch.einsum('bhnd,bhmd->bhnm', q, k) / math.sqrt(self.head_dim)
        
        # Add graph structure bias if enabled
        if self.use_graph_bias:
            adj_bias = torch.matmul(self.u, self.v)
            attn_scores = attn_scores + adj_bias
        
        # Add adjacency prior if enabled
        if self.use_adj_prior and self.adj_prior is not None:
            # Scale adjacency prior contribution
            attn_scores = attn_scores + self.adj_prior.unsqueeze(0) * 0.5
        
        # Apply softmax to get attention weights
        self.attn = F.softmax(attn_scores, dim=-1)
        
        # Compute regularization loss on attention weights using learned parameter
        attention_reg_weight = torch.exp(self.log_attention_reg_weight)
        attn_reg_loss = attention_reg_weight * torch.mean(torch.abs(self.attn))

        # Reshape output to original dimensions
        output = output.transpose(1, 2).contiguous().view(B, N, H)

        # Low-rank projection for output
        output = self.out_proj_low(output)
        output = self.out_proj_high(output)

        return output, attn_reg_loss


class MultiScaleSpatialModule(nn.Module):
    """
    Multi-Scale Spatial Feature Module (MSSFM) using dilated convolutions.
    
    Despite the historical naming, this module operates on the NODE (spatial) dimension.
    It uses dilated convolutions at different dilation rates to capture local and global
    spatial dependencies between nodes. The outputs from different scales are adaptively fused.
    
    Args:
        hidden_dim: Dimensionality of node features  
        num_scales: Number of spatial scales (different dilation rates)
        kernel_size: Size of the convolutional kernel
        dropout: Dropout probability
    """
    
    def __init__(self, hidden_dim, num_scales=NUM_TEMPORAL_SCALES, 
                 kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        # Create multiple dilated convolutional layers with increasing dilation rates
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                          padding=(kernel_size // 2) * (2 ** i), dilation=2 ** i),
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
        Apply multi-scale spatial convolutions over the node dimension.
        
        Args:
            x: Input features [batch, nodes, hidden_dim]
        Returns:
            Spatially processed features [batch, nodes, hidden_dim]
        """
        # Reshape for 1D convolution: [batch, hidden_dim, nodes]
        x = x.transpose(1, 2)
        
        # Apply different spatial scales (dilated convolutions along node dimension)
        features = [scale(x) for scale in self.scales]
        
        # Compute adaptive weights for scale fusion
        alpha = F.softmax(self.fusion_weight, dim=0)
        
        # Stack and fuse multi-scale features
        stacked = torch.stack(features, dim=1)  # [batch, scales, hidden_dim, nodes]
        fused = torch.sum(alpha.view(1, self.num_scales, 1, 1) * stacked, dim=1)
        
        # Reshape back: [batch, nodes, hidden_dim]
        fused = fused.transpose(1, 2)
        
        # Apply low-rank projection and residual connection
        out = self.fusion_low(fused)
        out = self.fusion_high(out)
        out = self.layer_norm(out + fused)
        
        return out


class HorizonPredictor(nn.Module):
    """
    Progressive Prediction Refinement Module (PPRM).
    
    Takes node features and generates predictions for multiple future time steps.
    Includes an adaptive refinement mechanism that blends model predictions with
    exponentially decayed extrapolations from the last observed value.
    
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
        Args:
            x: Node features [batch, nodes, hidden_dim]
            last_step: Last observed values [batch, nodes] (optional)
        Returns:
            Predictions [batch, nodes, horizon]
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


# =============================================================================
# MAIN MODEL
# =============================================================================

class MSTAGAT_Net(nn.Module):
    """
    Multi-Scale Temporal-Adaptive Graph Attention Network (MSTAGAT-Net)
    
    A spatiotemporal forecasting model with four key components:
    - TFEM: Temporal Feature Extraction Module (depthwise separable convolutions along time)
    - LR-AGAM: Low-rank Adaptive Graph Attention Module (spatial attention across nodes)
    - MSSFM: Multi-Scale Spatial Feature Module (dilated convolutions along nodes)
    - PPRM: Progressive Prediction Refinement Module (horizon prediction with adaptive blending)
    
    Data flow:
    1. Input [B, T, N] → TFEM extracts temporal features → [B, N, D]
    2. LR-AGAM computes attention across nodes → [B, N, D]
    3. MSSFM captures multi-scale spatial patterns → [B, N, D]
    4. PPRM generates predictions → [B, H, N]
    
    Uses adaptive configuration based on graph size:
    - Small graphs (<=20 nodes): Use adjacency prior + graph bias
    - Large graphs (>=40 nodes): Learned attention only (no graph bias)
    
    Args:
        args: Model configuration with attributes:
            - window: Input window size
            - horizon: Prediction horizon
            - hidden_dim, kernel_size, bottleneck_dim, etc.
            - use_adj_prior: Whether to use adjacency as prior
            - use_graph_bias: Whether to use learnable graph structure bias
        data: Data object with attributes:
            - m: Number of nodes
            - adj: Adjacency matrix (optional)
    """
    
    def __init__(self, args, data):
        super().__init__()
        
        self.num_nodes = data.m
        self.window = args.window
        self.horizon = args.horizon
        self.hidden_dim = getattr(args, 'hidden_dim', HIDDEN_DIM)
        self.kernel_size = getattr(args, 'kernel_size', KERNEL_SIZE)
        self.bottleneck_dim = getattr(args, 'bottleneck_dim', BOTTLENECK_DIM)
        
        # Apply adaptive configuration based on graph size
        adaptive_config = get_adaptive_config(self.num_nodes)
        
        # Graph structure options - use adaptive config as defaults, allow explicit overrides
        self.use_adj_prior = getattr(args, 'use_adj_prior', adaptive_config['use_adj_prior'])
        self.use_graph_bias = getattr(args, 'use_graph_bias', adaptive_config['use_graph_bias'])
        
        # Get adjacency matrix if available
        adj_matrix = getattr(data, 'adj', None)
        if adj_matrix is not None and not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

        # Feature Extraction Component (TFEM)
        self.feature_channels = getattr(args, 'feature_channels', FEATURE_CHANNELS)
        self.feature_extractor = DepthwiseSeparableConv1D(
            in_channels=1, 
            out_channels=self.feature_channels,
            kernel_size=self.kernel_size, 
            padding=self.kernel_size // 2,
            dropout=getattr(args, 'dropout', DROPOUT)
        )

        # Low-rank projection of extracted features
        self.feature_projection_low = nn.Linear(
            self.feature_channels * self.window, self.bottleneck_dim
        )
        self.feature_projection_high = nn.Linear(
            self.bottleneck_dim, self.hidden_dim
        )
        self.feature_norm = nn.LayerNorm(self.hidden_dim)
        self.feature_act = nn.ReLU()

        # Spatial Component (LR-AGAM)
        self.spatial_module = SpatialAttentionModule(
            self.hidden_dim, 
            num_nodes=self.num_nodes,
            dropout=getattr(args, 'dropout', DROPOUT),
            attention_heads=getattr(args, 'attention_heads', ATTENTION_HEADS),
            bottleneck_dim=self.bottleneck_dim,
            use_adj_prior=self.use_adj_prior,
            use_graph_bias=self.use_graph_bias,
            adj_matrix=adj_matrix if self.use_adj_prior else None
        )

        # Multi-Scale Spatial Feature Component (MSSFM)
        self.temporal_module = MultiScaleSpatialModule(
            self.hidden_dim,
            num_scales=getattr(args, 'num_scales', NUM_TEMPORAL_SCALES),
            kernel_size=self.kernel_size,
            dropout=getattr(args, 'dropout', DROPOUT)
        )

        # Prediction Component (PPRM)
        self.prediction_module = HorizonPredictor(
            self.hidden_dim, 
            self.horizon,
            bottleneck_dim=self.bottleneck_dim,
            dropout=getattr(args, 'dropout', DROPOUT)
        )

    def forward(self, x, idx=None):
        """
        Forward pass of the MSTAGAT-Net model.
        
        Args:
            x: Input time series [batch, time_window, nodes]
            idx: Node indices (optional, unused)
        Returns:
            tuple: (Predictions [batch, horizon, nodes], Attention reg loss)
        """
        B, T, N = x.shape
        x_last = x[:, -1, :]  # Last observed values
        
        # Feature Extraction
        x_temp = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)
        temporal_features = self.feature_extractor(x_temp)
        temporal_features = temporal_features.view(B, N, -1)
        
        # Feature projection through bottleneck
        features = self.feature_projection_low(temporal_features)
        features = self.feature_projection_high(features)
        features = self.feature_norm(features)
        features = self.feature_act(features)
        
        # Spatial Processing (LR-AGAM)
        graph_features, attn_reg_loss = self.spatial_module(features)
        
        # Multi-Scale Spatial Feature Processing (MSSFM)
        fusion_features = self.temporal_module(graph_features)
        
        # Prediction (PPRM)
        predictions = self.prediction_module(fusion_features, x_last)
        predictions = predictions.transpose(1, 2)  # [batch, horizon, nodes]
        
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


class SingleScaleSpatialModule(nn.Module):
    """
    Single-scale spatial convolution for ablation study.
    
    Used as ablation replacement for MultiScaleSpatialModule (no_mtfm).
    Operates on node dimension with a single dilation rate.
    
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
        Apply single-scale spatial convolution over node dimension.
        
        Args:
            x: Input [batch, nodes, hidden_dim]
        Returns:
            Output [batch, nodes, hidden_dim]
        """
        # Reshape for 1D convolution over nodes: [B, N, H] -> [B, H, N]
        x = x.transpose(1, 2)
        feat = self.conv(x)
        # Reshape back: [B, H, N] -> [B, N, H]
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
            - use_adj_prior: Whether to use adjacency as prior (default: False)
            - adj_weight: Weight for adjacency prior (default: 0.1)
        data: Data object with attribute:
            - m: Number of nodes
            - adj: Adjacency matrix (optional)
    
    Ablation variants:
        - none: Full model (with optional adjacency prior)
        - no_agam: Replace SpatialAttentionModule with SimpleGraphConvolutionalLayer (uses adj if available)
        - no_mtfm: Replace MultiScaleSpatialModule with SingleScaleSpatialModule
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
        
        # Optional adjacency matrix prior
        use_adj_prior = getattr(args, 'use_adj_prior', False)
        adj_matrix = getattr(data, 'adj', None) if use_adj_prior else None

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
            # Ablation: Use simple GCN with fixed adjacency (requires adj matrix)
            self.graph_attention = SimpleGraphConvolutionalLayer(
                self.hidden_dim, num_nodes=self.m, dropout=dropout
            )
            if hasattr(data, 'adj'):
                self.graph_attention.adj_matrix = data.adj
        else:
            # Full model: Use adaptive attention with optional adjacency prior
            use_graph_bias = getattr(args, 'use_graph_bias', True)
            self.graph_attention = SpatialAttentionModule(
                hidden_dim=self.hidden_dim,
                num_nodes=self.m,
                dropout=dropout,
                attention_heads=getattr(args, 'attention_heads', ATTENTION_HEADS),
                bottleneck_dim=self.low_rank_dim,
                use_adj_prior=use_adj_prior,
                use_graph_bias=use_graph_bias,
                adj_matrix=adj_matrix
            )
        
        # Spatial refinement component: choose multi-scale or single-scale
        if self.ablation == 'no_mtfm':
            self.spatial_refinement_module = SingleScaleSpatialModule(
                self.hidden_dim, kernel_size=self.kernel_size, dropout=dropout
            )
        else:
            self.spatial_refinement_module = MultiScaleSpatialModule(
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
        
        # Apply multi-scale spatial refinement
        fusion_features = self.spatial_refinement_module(graph_features)
        
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
    'MultiScaleSpatialModule',
    'MultiScaleTemporalModule',  # Deprecated alias for backward compatibility
    'HorizonPredictor',
    # Main models
    'MSTAGAT_Net',
    # Ablation components
    'SimpleGraphConvolutionalLayer',
    'SingleScaleSpatialModule',
    'SingleScaleTemporalModule',  # Deprecated alias for backward compatibility
    'DirectPredictionModule',
    'MSAGATNet_Ablation',
]

# Backward compatibility aliases
MultiScaleTemporalModule = MultiScaleSpatialModule
SingleScaleTemporalModule = SingleScaleSpatialModule
