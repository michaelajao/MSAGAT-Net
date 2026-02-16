"""
MSAGAT-Net Model Architectures

This module contains all neural network architectures for the MSAGAT-Net framework:
- Core building blocks (attention, spatial modules, convolutions)
- Main MSAGAT_Net model
- Ablation study variants (MSAGATNet_Ablation)

Architecture Components:
    1. SpatialAttentionModule: Scaled dot-product attention with additive structural bias
    2. MultiScaleSpatialModule: Multi-hop graph convolutions with adaptive fusion
    3. HorizonPredictor: Progressive multi-step prediction with learnable decay
    4. DepthwiseSeparableConv1D: Efficient temporal feature extraction
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
NUM_SPATIAL_SCALES = 4
KERNEL_SIZE = 3
FEATURE_CHANNELS = 16
BOTTLENECK_DIM = 8
HIGHWAY_WINDOW = 4  # For autoregressive component (critical for stable forecasting!)


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
    Efficient Adaptive Graph Attention Module (EAGAM).
    
    Graph-structure-aware multi-head attention with low-rank decomposition.
    Unlike standard content-only attention, this module integrates graph
    topology directly into the attention computation:
    
    attention = softmax(QK^T/sqrt(d) + graph_bias + adj_prior_blend) @ V
    
    Key design:
    - Low-rank QKV projections: hidden_dim -> bottleneck_dim -> hidden_dim
    - Graph bias: Learnable low-rank U @ V captures latent node relationships
    - Adjacency prior: Geographic/known structure blended with learnable weight and scale
    - Residual connection with LayerNorm for stable training
    
    Args:
        hidden_dim: Dimensionality of node features
        num_nodes: Number of nodes in the graph
        dropout: Dropout probability for regularization
        attention_heads: Number of parallel attention heads
        attention_regularization_weight: Weight for L1 regularization on attention
        bottleneck_dim: Dimension of the low-rank projection
        adj_matrix: Predefined adjacency matrix [num_nodes, num_nodes] (optional)
    """
    
    def __init__(self, hidden_dim, num_nodes, dropout=DROPOUT, 
                 attention_heads=ATTENTION_HEADS,
                 attention_regularization_weight=ATTENTION_REG_WEIGHT_INIT,
                 bottleneck_dim=BOTTLENECK_DIM,
                 adj_matrix=None):
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
        
        # Residual connection normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Learnable graph structure bias (low-rank): U @ V produces [heads, N, N]
        self.u = Parameter(torch.Tensor(self.heads, num_nodes, bottleneck_dim))
        self.v = Parameter(torch.Tensor(self.heads, bottleneck_dim, num_nodes))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)
        
        # Adjacency prior with learnable additive scale
        if adj_matrix is not None:
            if isinstance(adj_matrix, np.ndarray):
                adj_matrix = torch.from_numpy(adj_matrix).float()
            adj_matrix = adj_matrix.detach().cpu().float()
            adj_norm = adj_matrix / (adj_matrix.sum(dim=-1, keepdim=True) + 1e-8)
            self.register_buffer('adj_prior', adj_norm)
            # Learnable positive scale for additive adjacency bias
            # softplus(1.0) ≈ 1.31, a moderate structural nudge
            self.adj_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.register_buffer('adj_prior', None)
            self.adj_scale = None

    @property
    def current_reg_weight(self):
        """Get current learned regularization weight for diagnostics."""
        return torch.exp(self.log_attention_reg_weight).item()

    def forward(self, x, mask=None):
        """
        Forward pass with unified graph-structure-aware attention.
        
        Graph bias and adjacency prior directly influence the attention weights
        used for value aggregation, ensuring graph topology affects predictions.
        
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
        q, k, v = [tensor.view(B, N, self.heads, self.head_dim).transpose(1, 2) for tensor in qkv]

        # Compute content-based attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add learnable graph structure bias (low-rank U @ V)
        adj_bias = torch.matmul(self.u, self.v)  # [heads, N, N]
        attn_scores = attn_scores + adj_bias.unsqueeze(0)
        
        # Add adjacency prior as additive structural bias (self-regulating:
        # dense graphs -> near-uniform prior -> no effect on softmax;
        # sparse graphs -> peaked prior -> meaningful structural guidance)
        if self.adj_prior is not None:
            scale = F.softplus(self.adj_scale)
            adj_expanded = self.adj_prior.unsqueeze(0).unsqueeze(0).expand(B, self.heads, -1, -1)
            attn_scores = attn_scores + scale * adj_expanded
        
        # Softmax attention -> value aggregation (graph structure directly affects output)
        self.attn = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(self.attn)
        output = torch.matmul(attn_weights, v)  # [B, heads, N, head_dim]
        
        # Compute regularization loss on attention weights
        attention_reg_weight = torch.exp(self.log_attention_reg_weight)
        attn_reg_loss = attention_reg_weight * torch.mean(torch.abs(self.attn))

        # Reshape output to original dimensions
        output = output.transpose(1, 2).contiguous().view(B, N, H)

        # Low-rank projection for output
        output = self.out_proj_low(output)
        output = self.out_proj_high(output)
        
        # Residual connection with layer normalization
        output = self.norm(output + x)

        return output, attn_reg_loss


class MultiScaleSpatialModule(nn.Module):
    """
    Multi-Scale Spatial Feature Module (MSSFM) using multi-hop graph convolutions.
    
    Captures spatial dependencies at different graph diffusion scales using
    powers of the normalized adjacency matrix. Each scale corresponds to a
    different hop count in the graph, from self-features (0-hop) to multi-hop
    neighborhood aggregation.
    
    Hop depth adapts to graph size to prevent oversmoothing: small graphs use
    fewer hops, large graphs use more. Fusion weights are initialized to favor
    locality (lower hops weighted higher).
    
    Scales:
        - Scale 0: Self-features (identity / 0-hop)
        - Scale 1: 1-hop neighbors (direct connections)
        - Scale 2: 2-hop neighbors (neighbors of neighbors)
        - Scale 3: 3-hop neighbors (broader spatial context)
    
    Args:
        hidden_dim: Dimensionality of node features
        num_nodes: Number of nodes in the graph
        num_scales: Maximum number of spatial scales (hop counts)
        dropout: Dropout probability
        adj_matrix: Adjacency matrix [num_nodes, num_nodes] (optional)
    """
    
    def __init__(self, hidden_dim, num_nodes, num_scales=NUM_SPATIAL_SCALES, 
                 dropout=DROPOUT, adj_matrix=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Adaptive scale count: limit hops for small graphs to prevent oversmoothing
        # For 8 nodes: min(4, max(2, 1)) = 2 hops
        # For 17 nodes: min(4, max(2, 3)) = 3 hops
        # For 47 nodes: min(4, max(2, 9)) = 4 hops
        self.num_scales = min(num_scales, max(2, num_nodes // 5))
        
        # Graph convolution transform for each scale
        self.scale_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(self.num_scales)
        ])
        
        # Learnable fusion weights - initialized to favor locality (lower hops weighted higher)
        init_weights = torch.exp(-0.5 * torch.arange(self.num_scales).float())
        self.fusion_weight = Parameter(init_weights, requires_grad=True)
        
        # Low-rank projection for fusion
        self.fusion_low = nn.Linear(hidden_dim, BOTTLENECK_DIM)
        self.fusion_high = nn.Linear(BOTTLENECK_DIM, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Pre-compute multi-hop adjacency powers
        self._build_hop_matrices(num_nodes, self.num_scales, adj_matrix)
    
    def _build_hop_matrices(self, num_nodes, num_scales, adj_matrix):
        """Pre-compute normalized adjacency matrix powers for multi-hop aggregation."""
        if adj_matrix is not None:
            if isinstance(adj_matrix, np.ndarray):
                adj_matrix = torch.from_numpy(adj_matrix).float()
            elif isinstance(adj_matrix, torch.Tensor):
                adj_matrix = adj_matrix.detach().cpu().float()
            else:
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            
            # Add self-loops and normalize: A_hat = D^{-1}(A + I)
            adj_hat = adj_matrix + torch.eye(num_nodes)
            adj_hat = adj_hat / (adj_hat.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Scale 0: Identity (self-features)
            self.register_buffer('adj_hop_0', torch.eye(num_nodes))
            
            # Scale 1..k: Successive powers of normalized adjacency
            current = adj_hat
            for i in range(1, num_scales):
                self.register_buffer(f'adj_hop_{i}', current.clone())
                current = torch.matmul(current, adj_hat)
        else:
            # No adjacency: all scales default to identity (self-loop only)
            for i in range(num_scales):
                self.register_buffer(f'adj_hop_{i}', torch.eye(num_nodes))

    def forward(self, x):
        """
        Apply multi-hop graph convolutions and adaptively fuse scales.
        
        Args:
            x: Input features [batch, nodes, hidden_dim]
        Returns:
            Spatially processed features [batch, nodes, hidden_dim]
        """
        B, N, H = x.shape
        
        # Apply graph convolution at each scale (hop count)
        features = []
        for i in range(self.num_scales):
            adj_k = getattr(self, f'adj_hop_{i}')
            # Graph convolution: X' = A^k @ X @ W
            aggregated = torch.matmul(adj_k.unsqueeze(0).expand(B, -1, -1), x)
            transformed = self.scale_transforms[i](aggregated)
            features.append(transformed)
        
        # Compute adaptive weights for scale fusion
        alpha = F.softmax(self.fusion_weight, dim=0)
        
        # Stack and fuse multi-scale features
        stacked = torch.stack(features, dim=0)  # [scales, batch, nodes, hidden_dim]
        fused = torch.sum(alpha.view(self.num_scales, 1, 1, 1) * stacked, dim=0)
        
        # Apply low-rank projection and residual connection
        out = self.fusion_low(fused)
        out = self.fusion_high(out)
        out = self.layer_norm(out + x)
        
        return out


class HorizonPredictor(nn.Module):
    """
    Progressive Prediction Refinement Module (PPRM).
    
    Takes node features and generates predictions for multiple future time steps.
    Includes an adaptive refinement mechanism that blends model predictions with
    exponentially decayed extrapolations from the last observed value, where the
    decay rate is learned (not fixed).
    
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
        
        # Learnable decay rate (log-domain for positivity, initialized to ~0.1)
        self.log_decay = nn.Parameter(torch.tensor(-2.3))
        
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
            
            # Prepare last step and learnable decay
            last_step = last_step.unsqueeze(-1)
            decay_rate = torch.exp(self.log_decay)
            time_decay = torch.arange(1, self.horizon + 1, device=x.device).float().view(1, 1, self.horizon)
            progressive_part = last_step * torch.exp(-decay_rate * time_decay)
            
            # Adaptive fusion of model prediction and decay extrapolation
            final_pred = gate * initial_pred + (1 - gate) * progressive_part
        else:
            final_pred = initial_pred
            
        return final_pred


# =============================================================================
# MAIN MODEL
# =============================================================================

class MSAGAT_Net(nn.Module):
    """
    MSAGAT-Net (Multi-Scale Adaptive Graph Attention Network)
    
    A spatiotemporal forecasting model with four key components:
    - TFEM: Temporal Feature Extraction Module (depthwise separable convolutions along time)
    - EAGAM: Efficient Adaptive Graph Attention Module (graph-structure-aware attention)
    - MSSFM: Multi-Scale Spatial Feature Module (multi-hop graph convolutions)
    - PPRM: Progressive Prediction Refinement Module (horizon prediction with learnable decay)
    
    Data flow:
    1. Input [B, T, N] -> TFEM extracts temporal features -> [B, N, D]
    2. LR-AGAM computes graph-structure-aware attention -> [B, N, D] (residual)
    3. MSSFM captures multi-hop spatial patterns -> [B, N, D] (residual)
    4. PPRM generates predictions -> [B, H, N]
    5. Highway connection blends with autoregressive component
    
    Args:
        args: Model configuration with attributes:
            - window: Input window size
            - horizon: Prediction horizon
            - hidden_dim, kernel_size, bottleneck_dim, etc.
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
        
        # Get adjacency matrix if available (always passed to modules)
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

        # Spatial Component (EAGAM) - graph structure always used
        self.spatial_module = SpatialAttentionModule(
            self.hidden_dim, 
            num_nodes=self.num_nodes,
            dropout=getattr(args, 'dropout', DROPOUT),
            attention_heads=getattr(args, 'attention_heads', ATTENTION_HEADS),
            bottleneck_dim=self.bottleneck_dim,
            adj_matrix=adj_matrix
        )

        # Multi-Scale Spatial Feature Component (MSSFM) - multi-hop graph convolutions
        self.spatial_refinement_module = MultiScaleSpatialModule(
            self.hidden_dim,
            num_nodes=self.num_nodes,
            num_scales=getattr(args, 'num_scales', NUM_SPATIAL_SCALES),
            dropout=getattr(args, 'dropout', DROPOUT),
            adj_matrix=adj_matrix
        )

        # Prediction Component (PPRM)
        self.prediction_module = HorizonPredictor(
            self.hidden_dim, 
            self.horizon,
            bottleneck_dim=self.bottleneck_dim,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # Highway/Autoregressive Connection
        self.highway_window = min(getattr(args, 'highway_window', HIGHWAY_WINDOW), self.window)
        if self.highway_window > 0:
            self.highway = nn.Linear(self.highway_window, self.horizon)
        else:
            self.highway = None
        self.highway_ratio = nn.Parameter(torch.tensor(0.5))
        
        # Initialize weights properly (Xavier - used by all successful models)
        self._init_weights()
        
    # Parameters with special initialization that must NOT be overwritten
    _PRESERVE_PARAMS = {
        'fusion_weight',      # MSSFM locality-biased init: exp(-0.5 * k)
        'log_decay',          # PPRM learnable decay: -2.3 -> ~0.1
        'adj_scale',          # EAGAM adjacency scale: 1.0
        'highway_ratio',      # Highway blend: 0.5
        'log_attention_reg_weight',  # Attention reg: log(1e-5)
    }

    def _init_weights(self):
        """Initialize weights using Xavier uniform, preserving special inits."""
        for name, p in self.named_parameters():
            # Skip parameters with carefully designed initializations
            if any(pname in name for pname in self._PRESERVE_PARAMS):
                continue
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1 and p.size(0) > 0:
                if 'bias' in name:
                    nn.init.zeros_(p)
                else:
                    stdv = 1. / math.sqrt(p.size(0))
                    p.data.uniform_(-stdv, stdv)
            # Skip 0-dimensional tensors (scalars)

    def forward(self, x, idx=None):
        """
        Forward pass of the MSAGAT-Net model.
        
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
        
        # Spatial Processing (EAGAM)
        graph_features, attn_reg_loss = self.spatial_module(features)
        
        # Multi-Scale Spatial Feature Processing (MSSFM)
        fusion_features = self.spatial_refinement_module(graph_features)
        
        # Prediction (PPRM)
        model_pred = self.prediction_module(fusion_features, x_last)
        model_pred = model_pred.transpose(1, 2)  # [batch, horizon, nodes]
        
        # Highway/Autoregressive connection
        if self.highway is not None and self.highway_window > 0:
            # Get recent observations
            z = x[:, -self.highway_window:, :]  # [B, hw, N]
            z = z.permute(0, 2, 1).contiguous()  # [B, N, hw]
            z = z.view(B * N, self.highway_window)  # [B*N, hw]
            z = self.highway(z)  # [B*N, H]
            z = z.view(B, N, self.horizon)  # [B, N, H]
            z = z.transpose(1, 2)  # [B, H, N]
            
            # Blend with sigmoid-gated ratio
            ratio = torch.sigmoid(self.highway_ratio)
            predictions = ratio * model_pred + (1 - ratio) * z
        else:
            predictions = model_pred
        
        return predictions, attn_reg_loss


# =============================================================================
# ABLATION STUDY COMPONENTS
# =============================================================================

class IdentitySpatialModule(nn.Module):
    """
    Identity pass-through for ablation of SpatialAttentionModule (no_agam).
    
    Simply passes features through with a LayerNorm for training stability.
    No graph convolution, no attention -- tests the marginal contribution of
    the EAGAM component by removing ALL spatial attention.
    
    Args:
        hidden_dim: Dimension of hidden representations
        num_nodes: Number of nodes (unused, kept for interface compatibility)
        dropout: Dropout rate (unused)
    """
    
    def __init__(self, hidden_dim, num_nodes, dropout=DROPOUT, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_nodes = num_nodes
        
    def forward(self, x, mask=None):
        """
        Identity pass-through with normalization.
        
        Args:
            x: Input [batch, nodes, hidden_dim]
            mask: Unused
        Returns:
            tuple: (normalized input [batch, nodes, hidden_dim], 0.0)
        """
        # Store identity attention for visualization compatibility
        B, N, _ = x.shape
        self.attn = [torch.eye(N, device=x.device).unsqueeze(0).repeat(B, 1, 1)]
        return self.norm(x), 0.0


class IdentityMultiScaleModule(nn.Module):
    """
    Identity pass-through for ablation of MultiScaleSpatialModule (no_mtfm).
    
    Simply passes features through with a LayerNorm for training stability.
    No multi-hop graph convolution, no multi-scale fusion -- tests the marginal
    contribution of the MSSFM component.
    
    Args:
        hidden_dim: Dimension of hidden representations
        kernel_size: Unused (kept for interface compatibility)
        dropout: Unused
    """
    
    def __init__(self, hidden_dim, num_scales=NUM_SPATIAL_SCALES, 
                 kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        Identity pass-through with normalization.
        
        Args:
            x: Input [batch, nodes, hidden_dim]
        Returns:
            Output [batch, nodes, hidden_dim]
        """
        return self.norm(x)


class DirectPredictionModule(nn.Module):
    """
    Minimal direct prediction for ablation of HorizonPredictor (no_pprm).
    
    Single linear projection from hidden_dim to horizon, with no progressive 
    refinement, no learnable decay, no gating. Tests the marginal contribution
    of the PPRM component by using the simplest possible predictor.
    
    Args:
        hidden_dim: Dimension of hidden representations
        horizon: Prediction horizon
        dropout: Unused (kept for interface compatibility)
    """
    
    def __init__(self, hidden_dim, horizon, low_rank_dim=BOTTLENECK_DIM, 
                 dropout=DROPOUT):
        super().__init__()
        self.horizon = horizon
        # Single linear projection -- the minimal possible predictor
        self.predictor = nn.Linear(hidden_dim, horizon)

    def forward(self, x, last_step=None):
        """
        Args:
            x: Input [batch, nodes, hidden_dim]
            last_step: Unused (no refinement with last observed value)
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
        - none: Full model (graph-structure-aware attention + multi-hop graph conv)
        - no_agam: Replace SpatialAttentionModule with identity pass-through
        - no_mtfm: Replace MultiScaleSpatialModule with identity pass-through
        - no_pprm: Replace HorizonPredictor with single linear projection
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
        
        # Adjacency matrix (always loaded when available)
        adj_matrix = getattr(data, 'adj', None)
        if adj_matrix is not None and not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

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
        
        # Spatial component: choose full attention or identity pass-through
        if self.ablation == 'no_agam':
            self.graph_attention = IdentitySpatialModule(
                self.hidden_dim, num_nodes=self.m, dropout=dropout
            )
        else:
            self.graph_attention = SpatialAttentionModule(
                hidden_dim=self.hidden_dim,
                num_nodes=self.m,
                dropout=dropout,
                attention_heads=getattr(args, 'attention_heads', ATTENTION_HEADS),
                bottleneck_dim=self.low_rank_dim,
                adj_matrix=adj_matrix
            )
        
        # Spatial refinement: choose multi-hop graph conv or identity pass-through
        if self.ablation == 'no_mtfm':
            self.spatial_refinement_module = IdentityMultiScaleModule(
                self.hidden_dim, kernel_size=self.kernel_size, dropout=dropout
            )
        else:
            self.spatial_refinement_module = MultiScaleSpatialModule(
                hidden_dim=self.hidden_dim,
                num_nodes=self.m,
                num_scales=getattr(args, 'num_scales', NUM_SPATIAL_SCALES),
                dropout=dropout,
                adj_matrix=adj_matrix
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
        
        # Highway/Autoregressive Connection (same as main model for fair comparison)
        self.highway_window = min(getattr(args, 'highway_window', HIGHWAY_WINDOW), self.window)
        if self.highway_window > 0:
            self.highway = nn.Linear(self.highway_window, self.horizon)
        else:
            self.highway = None
        self.highway_ratio = nn.Parameter(torch.tensor(0.5))
        
        # Initialize weights
        self._init_weights()
        
    # Parameters with special initialization that must NOT be overwritten
    _PRESERVE_PARAMS = {
        'fusion_weight',      # MSSFM locality-biased init: exp(-0.5 * k)
        'log_decay',          # PPRM learnable decay: -2.3 -> ~0.1
        'adj_scale',          # EAGAM adjacency scale: 1.0
        'highway_ratio',      # Highway blend: 0.5
        'log_attention_reg_weight',  # Attention reg: log(1e-5)
    }

    def _init_weights(self):
        """Initialize weights using Xavier uniform, preserving special inits."""
        for name, p in self.named_parameters():
            # Skip parameters with carefully designed initializations
            if any(pname in name for pname in self._PRESERVE_PARAMS):
                continue
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1 and p.size(0) > 0:
                if 'bias' in name:
                    nn.init.zeros_(p)
                else:
                    stdv = 1. / math.sqrt(p.size(0))
                    p.data.uniform_(-stdv, stdv)
            # Skip 0-dimensional tensors (scalars)

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
        
        # Generate model predictions
        model_pred = self.prediction_module(fusion_features, x_last)
        model_pred = model_pred.transpose(1, 2)
        
        # Highway/Autoregressive connection
        if self.highway is not None and self.highway_window > 0:
            z = x[:, -self.highway_window:, :]
            z = z.permute(0, 2, 1).contiguous()
            z = z.view(B * N, self.highway_window)
            z = self.highway(z)
            z = z.view(B, N, self.horizon)
            z = z.transpose(1, 2)
            
            ratio = torch.sigmoid(self.highway_ratio)
            predictions = ratio * model_pred + (1 - ratio) * z
        else:
            predictions = model_pred
        
        return predictions, attn_reg_loss


# =============================================================================
# PAPER-CONSISTENT MODULE ALIASES
# =============================================================================
# These aliases map paper acronyms to their implementation classes,
# making the code easier to follow alongside the manuscript.

TFEM = DepthwiseSeparableConv1D    # Temporal Feature Extraction Module
EAGAM = SpatialAttentionModule     # Efficient Adaptive Graph Attention Module
MSSFM = MultiScaleSpatialModule    # Multi-Scale Spatial Feature Module
PPRM = HorizonPredictor            # Progressive Prediction Refinement Module
