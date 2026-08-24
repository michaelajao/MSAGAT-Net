"""
MSAGAT-Net Model Architectures

This module contains all neural network architectures for the MSAGAT-Net framework:
- Core building blocks (attention, spatial modules, convolutions, quantile head)
- MSAGATNet_Ablation: the model used for all experiments; its `ablation`
  argument selects the full model ('none') or a component-ablated variant

Architecture Components:
    1. SpatialAttentionModule: Scaled dot-product attention with additive structural bias
    2. MultiScaleSpatialModule: Multi-hop graph convolutions with adaptive fusion
    3. HorizonPredictor: Progressive multi-step prediction with learnable decay
    4. DepthwiseSeparableConv1D: Efficient temporal feature extraction
"""

import math
import re
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

class QuantileHead(nn.Module):
    """Monotone quantile forecasts around the lead-h point forecast.

    Predicts positive increments that are cumulatively summed away from the
    median on both sides, so quantile crossing is impossible by construction.
    The median quantile is tied to the point forecast, which keeps the point
    metrics of the probabilistic model identical in expectation to the
    deterministic path.
    """

    def __init__(self, hidden_dim, quantile_levels, dropout=DROPOUT):
        super().__init__()
        levels = sorted(float(q) for q in quantile_levels)
        self.register_buffer('levels', torch.tensor(levels, dtype=torch.float32))
        self.median_idx = min(range(len(levels)),
                              key=lambda i: abs(levels[i] - 0.5))
        n_offsets = len(levels) - 1
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_offsets),
        )

    def forward(self, features, point):
        """
        Args:
            features: [batch, nodes, hidden] spatial features
            point: [batch, nodes] lead-h point forecast (model output space)
        Returns:
            quantiles [batch, nodes, n_levels], monotone along the last axis.
        """
        inc = F.softplus(self.proj(features))          # [B, N, Q-1] positive
        m = self.median_idx
        parts = []
        if m > 0:
            low = inc[..., :m]
            low = torch.flip(torch.cumsum(torch.flip(low, [-1]), -1), [-1])
            parts.append(point.unsqueeze(-1) - low)
        parts.append(point.unsqueeze(-1))
        if m < inc.shape[-1]:
            up = torch.cumsum(inc[..., m:], -1)
            parts.append(point.unsqueeze(-1) + up)
        return torch.cat(parts, dim=-1)


def pinball_loss(quantile_pred, target, levels):
    """Mean pinball (quantile) loss.

    Args:
        quantile_pred: [batch, nodes, n_levels]
        target: [batch, nodes]
        levels: [n_levels] tensor of quantile levels in (0, 1)
    """
    diff = target.unsqueeze(-1) - quantile_pred
    return torch.mean(torch.maximum(levels * diff, (levels - 1.0) * diff))

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
        attention_regularization_weight: Weight for entropy regularization on attention
        bottleneck_dim: Dimension of the low-rank projection
        adj_matrix: Predefined adjacency matrix [num_nodes, num_nodes] (optional)
    """
    
    def __init__(self, hidden_dim, num_nodes, dropout=DROPOUT, 
                 attention_heads=ATTENTION_HEADS,
                 attention_regularization_weight=ATTENTION_REG_WEIGHT_INIT,
                 bottleneck_dim=BOTTLENECK_DIM,
                 adj_matrix=None, attn_fix=False, attn_exp=''):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.heads = attention_heads
        self.head_dim = hidden_dim // self.heads
        self.num_nodes = num_nodes
        self.bottleneck_dim = bottleneck_dim

        # Attention-revival experiment tokens (program.md). Composable:
        #   nodecay   - u/v/adj_scale/temp excluded from weight decay (train.py)
        #   temp      - learnable pre-softmax logit temperature
        #   regpre    - replace the inert post-softmax L1 with a pre-softmax L1
        #               on the structural bias U@V (has gradient; direction 1b)
        #   regent    - explicit row-entropy penalty (direction 10 -- forces
        #               non-uniformity by construction, weakest evidence)
        #   initN     - scale u/v initialisation by N (direction 3)
        #   lrxN      - u/v et al. get N x learning rate (train.py, direction 4)
        #   rankN     - low-rank dim of the U@V graph bias (direction 6)
        #   adjstd    - standardise the adjacency term per row (direction 7)
        #   scorenorm - normalise the combined logits (direction 8)
        #   multgate  - structure gates content multiplicatively (direction 9)
        self.attn_exp = set(t for t in attn_exp.split(',') if t)

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
        bias_rank = bottleneck_dim
        for t in self.attn_exp:
            m = re.fullmatch(r'rank(\d+)', t)
            if m:
                bias_rank = int(m.group(1))
        self.bias_rank = bias_rank
        self.u = Parameter(torch.Tensor(self.heads, num_nodes, bias_rank))
        self.v = Parameter(torch.Tensor(self.heads, bias_rank, num_nodes))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

        # Measured on trained checkpoints: the summed attention logits have a
        # per-row sd of ~0.004 on a 372-node graph, where a selective softmax
        # needs O(1) -- so attention degenerates to a uniform mean. attn_fix
        # adds a learnable temperature that can amplify whatever spread the
        # logits carry; u/v are additionally held out of weight decay, which
        # had driven them to ~1e-36.
        self.attn_fix = attn_fix
        use_temp = attn_fix or 'temp' in self.attn_exp
        self.log_attn_temp = nn.Parameter(torch.zeros(1)) if use_temp else None

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

    def apply_init_scale(self):
        """Scale u/v per an initN token.

        Must be called *after* the parent's _init_weights(), which re-applies
        xavier_uniform_ to every parameter with dim >= 2 -- including u and v --
        and would otherwise silently undo the scaling.
        """
        for t in self.attn_exp:
            m = re.fullmatch(r'init(\d+)', t)
            if m:
                with torch.no_grad():
                    self.u.mul_(float(m.group(1)))
                    self.v.mul_(float(m.group(1)))

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
        if 'multgate' in self.attn_exp:
            # Direction 9: structure gates content multiplicatively rather than
            # being added to it, so the bias modulates evidence instead of
            # competing with it for logit magnitude.
            attn_scores = attn_scores * torch.sigmoid(adj_bias).unsqueeze(0)
        else:
            attn_scores = attn_scores + adj_bias.unsqueeze(0)
        
        # Add adjacency prior as additive structural bias (self-regulating:
        # dense graphs -> near-uniform prior -> no effect on softmax;
        # sparse graphs -> peaked prior -> meaningful structural guidance)
        if self.adj_prior is not None:
            scale = F.softplus(self.adj_scale)
            adj_expanded = self.adj_prior.unsqueeze(0).unsqueeze(0).expand(B, self.heads, -1, -1)
            if 'adjstd' in self.attn_exp:
                # Softmax is shift-invariant along the row, so only a term's
                # WITHIN-row variation can influence attention -- a near-constant
                # row contributes nothing however large its scale. Standardising
                # puts the prior's variation on the same footing as the learned
                # terms instead of leaving it at the mercy of graph density.
                a = adj_expanded
                adj_expanded = ((a - a.mean(-1, keepdim=True))
                                / (a.std(-1, keepdim=True) + 1e-8))
            attn_scores = attn_scores + scale * adj_expanded

        if 'scorenorm' in self.attn_exp:
            # Direction 8: normalise the combined logits so no single term can
            # set the softmax temperature by magnitude alone.
            attn_scores = ((attn_scores - attn_scores.mean(-1, keepdim=True))
                           / (attn_scores.std(-1, keepdim=True) + 1e-8))
        
        # Scale the summed logits before the softmax so the module can learn
        # to be selective rather than being pinned at uniform.
        if self.log_attn_temp is not None:
            attn_scores = attn_scores * torch.exp(self.log_attn_temp)

        # Softmax attention -> value aggregation (graph structure directly affects output)
        self.attn = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(self.attn)
        output = torch.matmul(attn_weights, v)  # [B, heads, N, head_dim]
        
        # Attention regularisation. The historical term -- L1 on the row-wise
        # softmax output -- is provably inert: rows are non-negative and sum
        # to 1, so mean|A| = 1/N exactly, the gradient w.r.t. the attention is
        # zero, and the learnable weight lambda decays to kill even the
        # constant. Experiment modes replace it with terms that carry
        # gradient (fixed lambda = 1e-3 so it cannot self-annihilate):
        #   regpre: L1 on the pre-softmax structural bias U@V. Uniform
        #           downward pressure that the task gradient must overcome,
        #           yielding sparse surviving structure -- the mechanism the
        #           original penalty was intended to provide.
        #   regent: normalised row entropy of the attention itself. Forces
        #           non-uniformity by construction; direction 10, marked.
        if 'regpre' in self.attn_exp:
            attn_reg_loss = 1e-3 * adj_bias.abs().mean()
        elif 'regent' in self.attn_exp:
            ent = -(self.attn * torch.log(self.attn.clamp_min(1e-12))).sum(-1)
            attn_reg_loss = 1e-3 * (ent / math.log(N)).mean()
        else:
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
                adj_matrix=adj_matrix,
                attn_fix=getattr(args, 'attn_fix', False),
                attn_exp=getattr(args, 'attn_exp', '')
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

        # Optional learnable gate on the spatial pathway. Cross-scale ablations
        # show the spatial modules can hurt on very small graphs (N<=8); the
        # gate lets the model attenuate spatial mixing where the graph carries
        # no usable signal, instead of being forced through it.
        # sigmoid(0) = 0.5 starts the blend balanced.
        if getattr(args, 'spatial_gate', False):
            self.spatial_gate = nn.Parameter(torch.tensor(0.0))
        else:
            self.spatial_gate = None

        # Optional probabilistic output: monotone quantile forecasts for the
        # lead-h step, centred on the point forecast. self.last_quantiles is
        # populated on every forward pass when enabled.
        q_levels = getattr(args, 'quantiles', None)
        if q_levels:
            self.quantile_head = QuantileHead(self.hidden_dim, q_levels,
                                              dropout=dropout)
        else:
            self.quantile_head = None
        self.last_quantiles = None

        # ---- Renewal decoder (optional) -----------------------------------
        # Reparameterises the forecast as a differentiable, spatially-coupled
        # renewal equation:
        #     I_i(t+h) = R_i(t) * SUM_tau alpha_tau * SUM_j w_ij I_j(t-tau)
        # The backbone predicts log R (transmissibility); the convolution over
        # a LEARNED generation-interval kernel alpha supplies the rest, so the
        # network no longer has to learn the epidemic's magnitude dynamics.
        # alpha is a softmax over lags, hence a proper distribution summing to
        # 1 -- which also makes the convolution commute with the affine min-max
        # normalisation, so alpha stays interpretable as a generation interval
        # on the raw incidence scale.
        self.renewal = bool(getattr(args, 'renewal', False))
        if self.renewal:
            max_lag = int(getattr(args, 'renewal_lag', 0) or 0)
            if max_lag <= 0:
                max_lag = min(14, self.window)
            # The kernel starts at tau=1, never tau=0 -- the standard
            # epidemiological convention (Cori et al. 2013 set w_0 = 0: in
            # discrete time nobody is infected at zero delay).
            #
            # It also removes the decoder's degenerate solution. With mass
            # allowed at tau=0, alpha can collapse to a delta there and
            # Lambda becomes SUM_j w_ij I_j(t) -- a purely spatial, zero-delay
            # aggregate carrying no temporal structure at all, which guts the
            # generation-interval reading while leaving RMSE untouched. (It
            # degenerates further, to offset == 0 and a model identical to the
            # direct log-growth decoder, only if the coupling also approaches
            # the identity; row-softmax makes that unlikely but not
            # impossible.) Starting at tau=1 forces alpha to be a genuine
            # DELAY distribution, which is what the claim rests on.
            # `renewal_lag0` restores tau=0 so the failure can be shown as an
            # ablation rather than merely asserted.
            _exp = set(t for t in getattr(args, 'attn_exp', '').split(',') if t)
            self.renewal_from_zero = 'renewal_lag0' in _exp
            span = self.window - (0 if self.renewal_from_zero else 1)
            self.renewal_lag = max(1, min(max_lag, span))
            self.log_alpha = Parameter(torch.zeros(self.renewal_lag))
            # Direction 2 (residual variant): a free scalar on the renewal
            # offset, initialised to 1.0 so training starts as pure renewal.
            # The learned value IS the experiment's answer -- it measures how
            # much renewal structure the model actually wants. gamma -> 0 means
            # it prefers the direct decoder; gamma ~ 1 means the convolution
            # earns its place. Softer and far more informative than an on/off
            # ablation.
            self.renewal_gamma = (Parameter(torch.ones(1))
                                  if 'renewres' in _exp else None)
            # ITERATED renewal (`reniter`). The single-convolution decoder is
            # wrong for h-step-ahead forecasting: the infections generating the
            # target occur in the unobserved gap between t and t+h, so a
            # convolution over history older than h is a stale-delay kernel,
            # not the renewal equation. Measured consequence: 6 configs x 5
            # cells = 30 comparisons, 0 wins, and the learned residual weight
            # collapsed to ~0 at h=3 where the gap swallows the whole kernel.
            #
            # Rolling the equation forward instead --
            #     I(t+s) = R(t+s) * SUM_tau alpha_tau * SUM_j w_ij I_j(t+s-tau)
            # for s = 1..h, feeding each prediction back -- means the kernel
            # always spans the tau=1..L most recent values (observed, then
            # predicted). Only here is alpha_tau a genuine generation interval.
            self.renewal_iter = 'reniter' in _exp

            # Decisive experiment (adversarial-priority-check.md rec 3): freeze
            # alpha to a FIXED literature generation interval instead of
            # learning it. `gi_fix=(mean, sd)` discretises a gamma over
            # tau=1..L and disables the gradient, so the learned-vs-fixed
            # comparison isolates what learning the kernel actually buys.
            # `gi_uniform` is the third arm: no GI shape at all.
            gi = getattr(args, 'gi_fix', None)
            self.gi_fixed = None
            if gi or 'giunif' in _exp:
                lags = torch.arange(1, self.renewal_lag + 1, dtype=torch.float32)
                if 'giunif' in _exp:
                    w = torch.ones_like(lags)
                    self.gi_fixed = ('uniform', 0.0, 0.0)
                else:
                    mean, sd = float(gi[0]), float(gi[1])
                    shape = (mean / sd) ** 2
                    scale = sd ** 2 / mean
                    # unnormalised gamma pdf on the lag grid
                    w = (lags ** (shape - 1)) * torch.exp(-lags / scale)
                    self.gi_fixed = ('gamma', mean, sd)
                w = w / w.sum()
                with torch.no_grad():
                    self.log_alpha.copy_(torch.log(w.clamp_min(1e-12)))
                self.log_alpha.requires_grad_(False)
            # R is bounded so h compounding steps cannot explode:
            # exp(+-1.5) => R in [0.22, 4.48], a plausible reproduction range.
            self.renewal_logR_clamp = 1.5
            dmax = torch.as_tensor(data.max, dtype=torch.float32)
            dmin = torch.as_tensor(data.min, dtype=torch.float32)
            self.register_buffer('dat_max', dmax)
            self.register_buffer('dat_min', dmin)
        else:
            self.renewal_lag = 0

        # Initialize weights
        self._init_weights()
        if hasattr(self.graph_attention, 'apply_init_scale'):
            self.graph_attention.apply_init_scale()
        
    # Parameters with special initialization that must NOT be overwritten
    _PRESERVE_PARAMS = {
        'fusion_weight',      # MSSFM locality-biased init: exp(-0.5 * k)
        'log_decay',          # PPRM learnable decay: -2.3 -> ~0.1
        'adj_scale',          # EAGAM adjacency scale: 1.0
        'highway_ratio',      # Highway blend: 0.5
        'spatial_gate',       # Spatial pathway gate: sigmoid(0) = 0.5
        'log_attention_reg_weight',  # Attention reg: log(1e-5)
        'log_attn_temp',      # EAGAM logit temperature: exp(0) = 1
        'log_alpha',          # Renewal generation interval: uniform over lags
        'renewal_gamma',      # Renewal residual weight: starts at 1.0
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

        # Gated blend of spatial pathway vs purely temporal features
        if self.spatial_gate is not None:
            g = torch.sigmoid(self.spatial_gate)
            fusion_features = g * fusion_features + (1 - g) * features

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

        if self.renewal:
            # Treat the backbone output as log R and add the renewal offset.
            # x[:, -1, :] is rawdat[idx-h] once denormalised -- exactly the
            # anchor growth_targets() uses -- so the sum below is the model's
            # log-growth prediction under the same definition, and no
            # evaluation code needs to change.
            scale = (self.dat_max - self.dat_min).clamp_min(1e-8)
            raw = (x * scale + self.dat_min).clamp_min(0.0)      # [B, T, N]
            W = self._renewal_coupling()                          # [N, N]
            mixed = torch.matmul(raw, W.transpose(0, 1))          # [B, T, N]
            alpha = torch.softmax(self.log_alpha, dim=0)          # [L]

            if self.renewal_iter:
                # Roll the renewal equation forward h steps. The backbone's
                # [B, h, N] output is read as log R at each step.
                L = self.renewal_lag
                buf = mixed[:, -L:, :].flip(1)                    # tau=1..L
                anchor = raw[:, -1, :]                            # rawdat[idx-h]
                logR = predictions.clamp(-self.renewal_logR_clamp,
                                         self.renewal_logR_clamp)
                cur = None
                for step in range(self.horizon):
                    lam = (buf * alpha.view(1, -1, 1)).sum(dim=1)  # [B, N]
                    cur = torch.exp(logR[:, step, :]) * lam        # I(t+step+1)
                    nxt = torch.matmul(cur, W.transpose(0, 1))     # mix forward
                    buf = torch.cat([nxt.unsqueeze(1), buf[:, :-1, :]], dim=1)
                g_out = torch.log((cur + 1.0) / (anchor + 1.0))
                if self.renewal_gamma is not None:
                    g_out = self.renewal_gamma * g_out
                predictions = g_out.unsqueeze(1).expand(-1, self.horizon, -1)
                if self.quantile_head is not None:
                    self.last_quantiles = self.quantile_head(
                        fusion_features, predictions[:, -1, :])
                return predictions, attn_reg_loss
            if self.renewal_from_zero:
                lags = mixed[:, -self.renewal_lag:, :].flip(1)     # tau = 0..L-1
            else:
                end = mixed.shape[1] - 1                           # drop tau=0
                lags = mixed[:, end - self.renewal_lag:end, :].flip(1)  # tau = 1..L
            lam = (lags * alpha.view(1, -1, 1)).sum(dim=1)        # [B, N]
            anchor = raw[:, -1, :]                                # [B, N]
            offset = torch.log((lam + 1.0) / (anchor + 1.0))      # [B, N]
            if self.renewal_gamma is not None:
                offset = self.renewal_gamma * offset
            predictions = predictions + offset.unsqueeze(1)

        if self.quantile_head is not None:
            self.last_quantiles = self.quantile_head(
                fusion_features, predictions[:, -1, :])

        return predictions, attn_reg_loss

    def _renewal_coupling(self):
        """Row-stochastic spatial coupling w_ij for the renewal convolution.

        Combines the density-invariant static geographic prior with the learned
        low-rank term, then row-softmaxes so each row is a proper distribution
        over source regions -- required for the renewal reading of the operator.
        """
        sa = self.graph_attention
        logits = None
        if getattr(sa, 'u', None) is not None and getattr(sa, 'v', None) is not None:
            logits = torch.matmul(sa.u, sa.v).mean(0)             # [N, N]
        prior = getattr(sa, 'adj_prior', None)
        if prior is not None:
            # Row-normalising a dense graph drives the prior's within-row
            # spread to ~0 (measured: 0.005 on 372-node LTLA vs 0.197 on
            # 7-node NHS), and softmax only sees within-row variation --
            # standardising restores geography at every graph size.
            std = (prior - prior.mean(-1, keepdim=True)) / (
                prior.std(-1, keepdim=True) + 1e-8)
            scale = F.softplus(sa.adj_scale) if sa.adj_scale is not None else 1.0
            logits = std * scale if logits is None else logits + std * scale
        if logits is None:
            n = self.m
            return torch.full((n, n), 1.0 / n, device=self.log_alpha.device)
        return torch.softmax(logits, dim=-1)


# =============================================================================
# PAPER-CONSISTENT MODULE ALIASES
# =============================================================================
# These aliases map paper acronyms to their implementation classes,
# making the code easier to follow alongside the manuscript.

TFEM = DepthwiseSeparableConv1D    # Temporal Feature Extraction Module
EAGAM = SpatialAttentionModule     # Efficient Adaptive Graph Attention Module
MSSFM = MultiScaleSpatialModule    # Multi-Scale Spatial Feature Module
PPRM = HorizonPredictor            # Progressive Prediction Refinement Module
