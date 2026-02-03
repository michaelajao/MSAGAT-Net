"""
EpiSIG-Net v2: Enhanced Serial Interval Graph Network

Hybrid model combining:
1. EpiSIG-Net's Serial Interval Graph (epidemiological novelty)
2. MSAGAT-Net's Linear Attention O(N) (efficiency on large graphs)  
3. MSAGAT-Net's DepthwiseSeparableConv1D (efficient feature extraction)
4. Highway/Autoregressive connections (stable forecasting)

============================================================================
KEY IMPROVEMENTS OVER EpiSIG-Net v1
============================================================================

1. LINEAR ATTENTION: O(N) instead of O(N²) - better on large graphs (LTLA, USA)
2. DEPTHWISE SEPARABLE CONVS: More efficient temporal feature extraction
3. LOW-RANK PROJECTIONS: Reduces parameters while maintaining expressivity
4. BETTER LONG-HORIZON: Enhanced prediction module with refinement

============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
HIDDEN_DIM = 32
DROPOUT = 0.2
KERNEL_SIZE = 3
MAX_LAG = 7  # Serial interval range
ATTENTION_HEADS = 4
BOTTLENECK_DIM = 8
HIGHWAY_WINDOW = 4
FEATURE_CHANNELS = 16


# =============================================================================
# COMPONENT 1: DEPTHWISE SEPARABLE TEMPORAL ENCODER (from MSAGAT-Net)
# =============================================================================

class DepthwiseSeparableConv1D(nn.Module):
    """
    Efficient feature extraction using depthwise separable convolutions.
    More parameter efficient than standard convolutions.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, dropout=DROPOUT):
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
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.dropout(self.act(self.bn2(self.pointwise(x))))
        return x


class EfficientTemporalEncoder(nn.Module):
    """
    Multi-scale temporal feature extraction using depthwise separable convolutions.
    More efficient than dilated convolutions for feature extraction.
    """
    
    def __init__(self, window, hidden_dim=HIDDEN_DIM, 
                 feature_channels=FEATURE_CHANNELS, dropout=DROPOUT):
        super().__init__()
        
        self.window = window
        self.hidden_dim = hidden_dim
        
        # Depthwise separable convolution for efficient feature extraction
        self.feature_conv = DepthwiseSeparableConv1D(
            in_channels=1,
            out_channels=feature_channels,
            kernel_size=KERNEL_SIZE,
            padding=KERNEL_SIZE // 2,
            dropout=dropout
        )
        
        # Multi-scale convolutions with different dilations
        self.scale_convs = nn.ModuleList([
            DepthwiseSeparableConv1D(
                in_channels=feature_channels,
                out_channels=feature_channels,
                kernel_size=KERNEL_SIZE,
                padding=(KERNEL_SIZE // 2) * (2 ** i),
                dilation=2 ** i,
                dropout=dropout
            )
            for i in range(3)  # 3 scales like EpiSIG
        ])
        
        # Learnable scale fusion
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Low-rank projection to hidden_dim
        self.proj_low = nn.Linear(feature_channels * window, BOTTLENECK_DIM)
        self.proj_high = nn.Linear(BOTTLENECK_DIM, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        """
        Args:
            x: [batch, window, nodes]
        Returns:
            features: [batch, nodes, hidden_dim]
        """
        B, T, N = x.shape
        
        # Process each node's time series: [B, T, N] -> [B*N, 1, T]
        x_temp = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)
        
        # Initial feature extraction
        features = self.feature_conv(x_temp)  # [B*N, feature_channels, T]
        
        # Multi-scale processing
        scale_outputs = [conv(features) for conv in self.scale_convs]
        
        # Adaptive fusion
        weights = F.softmax(self.scale_weights, dim=0)
        fused = sum(w * out for w, out in zip(weights, scale_outputs))
        
        # Reshape and project: [B*N, feature_channels, T] -> [B, N, hidden_dim]
        fused = fused.view(B, N, -1)
        out = self.proj_low(fused)
        out = self.proj_high(out)
        out = self.norm(out)
        
        return out


# =============================================================================
# COMPONENT 2: SERIAL INTERVAL GRAPH (Novel - from EpiSIG-Net)
# =============================================================================

class SerialIntervalGraphV2(nn.Module):
    """
    Serial Interval Graph with Linear Attention.
    
    NOVEL: Epidemiological propagation delay modeling with O(N) complexity.
    
    Combines:
    - Learnable generation interval distribution (epidemiological)
    - Linear attention mechanism (efficient)
    - Low-rank projections (compact)
    """
    
    def __init__(self, num_nodes, hidden_dim=HIDDEN_DIM, max_lag=MAX_LAG,
                 num_heads=ATTENTION_HEADS, bottleneck_dim=BOTTLENECK_DIM,
                 adj_matrix=None, dropout=DROPOUT):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.bottleneck_dim = bottleneck_dim
        
        # =====================================================================
        # LEARNABLE GENERATION INTERVAL DISTRIBUTION (Epidemiological novelty)
        # =====================================================================
        # Initialize with exponential decay (typical for many diseases)
        init_weights = torch.exp(-torch.arange(max_lag + 1).float() / 3.0)
        init_weights = init_weights / init_weights.sum()
        self.delay_logits = nn.Parameter(torch.log(init_weights + 1e-8))
        
        # =====================================================================
        # LINEAR ATTENTION (O(N) complexity from MSAGAT-Net)
        # =====================================================================
        # Low-rank QKV projections
        self.qkv_proj_low = nn.Linear(hidden_dim, 3 * bottleneck_dim)
        self.qkv_proj_high = nn.Linear(3 * bottleneck_dim, 3 * hidden_dim)
        
        # Output projections
        self.out_proj_low = nn.Linear(hidden_dim, bottleneck_dim)
        self.out_proj_high = nn.Linear(bottleneck_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # =====================================================================
        # GEOGRAPHIC PRIOR (optional)
        # =====================================================================
        if adj_matrix is not None:
            if not isinstance(adj_matrix, torch.Tensor):
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            adj_norm = adj_matrix / (adj_matrix.sum(dim=-1, keepdim=True) + 1e-8)
            self.register_buffer('adj_prior', adj_norm)
            self.geo_weight = nn.Parameter(torch.tensor(0.3))
        else:
            self.register_buffer('adj_prior', None)
            self.geo_weight = None
            
        # =====================================================================
        # DELAY SIGNAL PROJECTION
        # =====================================================================
        # Projects delay-weighted signal to feature space
        self.delay_proj = nn.Linear(1, hidden_dim)
        
    def get_delay_weights(self):
        """Get normalized delay weights (generation interval distribution)."""
        return F.softmax(self.delay_logits, dim=0)
    
    def _linear_attention(self, q, k, v):
        """
        Linear attention with O(N) complexity.
        Uses ELU + 1 kernel for positive features.
        """
        # Apply ELU + 1 for positive features
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        # Compute key-value products: [B, H, D, D] 
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        
        # Normalize for stability
        k_sum = k.sum(dim=2, keepdim=True) + 1e-8  # [B, H, 1, D]
        
        # Compute output: [B, H, N, D]
        out = torch.einsum('bhnd,bhde->bhne', q, kv)
        out = out / (torch.einsum('bhnd,bhod->bhn', q, k_sum).unsqueeze(-1) + 1e-8)
        
        return out
    
    def forward(self, x, features):
        """
        Apply Serial Interval Graph with linear attention.
        
        Args:
            x: Raw time series [batch, window, nodes]
            features: Node features [batch, nodes, hidden_dim]
            
        Returns:
            updated_features: [batch, nodes, hidden_dim]
        """
        B, T, N = x.shape
        
        # =====================================================================
        # STEP 1: Compute delay-weighted historical signal
        # =====================================================================
        delay_weights = self.get_delay_weights()  # [max_lag + 1]
        
        delayed_signal = torch.zeros(B, N, device=x.device)
        for tau in range(min(self.max_lag + 1, T)):
            lagged_value = x[:, -(tau + 1), :]  # [B, N]
            delayed_signal = delayed_signal + delay_weights[tau] * lagged_value
        
        # Project delay signal to feature space
        delay_features = self.delay_proj(delayed_signal.unsqueeze(-1))  # [B, N, D]
        
        # =====================================================================
        # STEP 2: Combine features with delay information
        # =====================================================================
        combined_features = features + delay_features * 0.1
        
        # =====================================================================
        # STEP 3: Linear attention for spatial aggregation
        # =====================================================================
        # Low-rank QKV projection
        qkv_low = self.qkv_proj_low(combined_features)
        qkv = self.qkv_proj_high(qkv_low)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply linear attention
        attended = self._linear_attention(q, k, v)
        
        # =====================================================================
        # STEP 4: Incorporate geographic prior if available
        # =====================================================================
        if self.adj_prior is not None:
            geo_blend = torch.sigmoid(self.geo_weight)
            # Standard attention for geographic prior
            geo_attn = self.adj_prior.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
            v_geo = torch.einsum('bhnm,bhmd->bhnd', geo_attn, v)
            attended = (1 - geo_blend) * attended + geo_blend * v_geo
        
        # =====================================================================
        # STEP 5: Output projection
        # =====================================================================
        attended = attended.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)
        attended = self.dropout(attended)
        
        # Low-rank output projection
        out = self.out_proj_low(attended)
        out = self.out_proj_high(out)
        
        # Residual connection and normalization
        out = self.norm(out + features)
        
        return out
    
    def get_interpretable_outputs(self):
        """Return interpretable outputs for analysis."""
        delay_weights = self.get_delay_weights().detach().cpu()
        peak_delay = torch.argmax(delay_weights).item()
        
        return {
            'delay_weights': delay_weights,
            'peak_delay': peak_delay,
            'interpretation': f"Peak transmission delay: {peak_delay} time steps"
        }


# =============================================================================
# COMPONENT 3: ENHANCED PREDICTOR (Better for long horizons)
# =============================================================================

class EnhancedPredictor(nn.Module):
    """
    Prediction module with multi-head output and adaptive refinement.
    Better for long horizons.
    """
    
    def __init__(self, hidden_dim, horizon, bottleneck_dim=BOTTLENECK_DIM, 
                 dropout=DROPOUT):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # Multi-scale prediction heads (short, medium, long term)
        self.short_head = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, min(horizon, 3))
        )
        
        self.medium_head = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, min(horizon, 7))
        ) if horizon > 3 else None
        
        self.long_head = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(), 
            nn.Linear(bottleneck_dim, horizon)
        ) if horizon > 7 else None
        
        # Final fusion
        num_heads = 1 + (1 if horizon > 3 else 0) + (1 if horizon > 7 else 0)
        self.fusion = nn.Linear(num_heads, 1) if num_heads > 1 else None
        
        # Learnable decay for refinement
        self.log_decay = nn.Parameter(torch.tensor(-2.3))  # ~0.1
        
        # Refinement gate
        self.refine_gate = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, horizon),
            nn.Sigmoid()
        )
        
    def forward(self, features, last_value):
        """
        Args:
            features: [batch, nodes, hidden_dim]
            last_value: [batch, nodes] - last observed value
            
        Returns:
            predictions: [batch, nodes, horizon]
        """
        B, N, D = features.shape
        
        # Generate predictions from each head
        preds = []
        
        # Short-term
        short_pred = self.short_head(features)  # [B, N, min(H, 3)]
        if short_pred.size(-1) < self.horizon:
            short_pred = F.pad(short_pred, (0, self.horizon - short_pred.size(-1)))
        preds.append(short_pred)
        
        # Medium-term
        if self.medium_head is not None:
            med_pred = self.medium_head(features)
            if med_pred.size(-1) < self.horizon:
                med_pred = F.pad(med_pred, (0, self.horizon - med_pred.size(-1)))
            preds.append(med_pred)
        
        # Long-term
        if self.long_head is not None:
            long_pred = self.long_head(features)
            preds.append(long_pred)
        
        # Fuse predictions
        if len(preds) > 1:
            stacked = torch.stack(preds, dim=-1)  # [B, N, H, num_heads]
            raw_pred = self.fusion(stacked).squeeze(-1)  # [B, N, H]
        else:
            raw_pred = preds[0]
        
        # Adaptive refinement with decay extrapolation
        gate = self.refine_gate(features)  # [B, N, H]
        
        decay_rate = torch.exp(self.log_decay)
        time_steps = torch.arange(1, self.horizon + 1, device=features.device).float()
        decay_curve = torch.exp(-decay_rate * time_steps).view(1, 1, -1)
        decay_pred = last_value.unsqueeze(-1) * decay_curve
        
        # Blend
        final_pred = gate * raw_pred + (1 - gate) * decay_pred
        
        return final_pred


# =============================================================================
# MAIN MODEL: EpiSIG-Net v2
# =============================================================================

class EpiSIGNetV2(nn.Module):
    """
    EpiSIG-Net v2: Enhanced Serial Interval Graph Network
    
    Combines:
    - Efficient depthwise separable temporal encoding
    - Serial Interval Graph with linear O(N) attention
    - Multi-scale prediction with adaptive refinement
    - Highway connections for stability
    
    ~30-40K parameters (lightweight like Cola-GNN)
    """
    
    def __init__(self, args, data):
        super().__init__()
        
        self.num_nodes = data.m
        self.window = args.window
        self.horizon = args.horizon
        self.hidden_dim = getattr(args, 'hidden_dim', HIDDEN_DIM)
        
        # Get adjacency matrix
        adj_matrix = getattr(data, 'adj', None)
        if adj_matrix is not None and not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        
        # =====================================================================
        # TEMPORAL ENCODER (Depthwise Separable - from MSAGAT)
        # =====================================================================
        self.temporal_encoder = EfficientTemporalEncoder(
            window=self.window,
            hidden_dim=self.hidden_dim,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # =====================================================================
        # SERIAL INTERVAL GRAPH (Novel - with Linear Attention)
        # =====================================================================
        self.sig_module = SerialIntervalGraphV2(
            num_nodes=self.num_nodes,
            hidden_dim=self.hidden_dim,
            max_lag=MAX_LAG,
            adj_matrix=adj_matrix,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # =====================================================================
        # PREDICTOR (Enhanced for long horizons)
        # =====================================================================
        self.predictor = EnhancedPredictor(
            hidden_dim=self.hidden_dim,
            horizon=self.horizon,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # =====================================================================
        # HIGHWAY CONNECTION (Critical for stable forecasting)
        # =====================================================================
        self.highway_window = min(getattr(args, 'highway_window', HIGHWAY_WINDOW), self.window)
        if self.highway_window > 0:
            self.highway = nn.Linear(self.highway_window, self.horizon)
        else:
            self.highway = None
        self.highway_ratio = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform."""
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1 and p.size(0) > 0:
                if 'bias' in name:
                    nn.init.zeros_(p)
                else:
                    stdv = 1. / math.sqrt(p.size(0))
                    p.data.uniform_(-stdv, stdv)
    
    def forward(self, x, idx=None):
        """
        Forward pass.
        
        Args:
            x: [batch, window, nodes]
            idx: Optional node indices
            
        Returns:
            predictions: [batch, horizon, nodes]
            reg_loss: Regularization loss (placeholder for compatibility)
        """
        B, T, N = x.shape
        x_last = x[:, -1, :]
        
        # Temporal encoding
        features = self.temporal_encoder(x)  # [B, N, D]
        
        # Serial Interval Graph processing
        features = self.sig_module(x, features)  # [B, N, D]
        
        # Prediction
        predictions = self.predictor(features, x_last)  # [B, N, H]
        
        # Highway connection
        if self.highway is not None:
            highway_input = x[:, -self.highway_window:, :].permute(0, 2, 1)  # [B, N, hw]
            highway_out = self.highway(highway_input)  # [B, N, H]
            
            ratio = torch.sigmoid(self.highway_ratio)
            predictions = ratio * predictions + (1 - ratio) * highway_out
        
        # Reshape to [B, H, N]
        predictions = predictions.transpose(1, 2)
        
        # Return dummy reg_loss for compatibility
        reg_loss = torch.tensor(0.0, device=x.device)
        
        return predictions, reg_loss
    
    def get_interpretable_outputs(self):
        """Get interpretable outputs from the Serial Interval Graph."""
        return self.sig_module.get_interpretable_outputs()


# =============================================================================
# ABLATION: EpiSIG-Net v2 without SIG (for fair comparison)
# =============================================================================

class EpiSIGNetV2_NoSIG(nn.Module):
    """
    Ablation variant without the Serial Interval Graph.
    Uses standard linear attention instead.
    """
    
    def __init__(self, args, data):
        super().__init__()
        
        self.num_nodes = data.m
        self.window = args.window
        self.horizon = args.horizon
        self.hidden_dim = getattr(args, 'hidden_dim', HIDDEN_DIM)
        
        adj_matrix = getattr(data, 'adj', None)
        if adj_matrix is not None and not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        
        # Temporal encoder (same)
        self.temporal_encoder = EfficientTemporalEncoder(
            window=self.window,
            hidden_dim=self.hidden_dim,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # Standard attention (no SIG - ablation)
        from src.models import SpatialAttentionModule
        self.spatial_attn = SpatialAttentionModule(
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            adj_matrix=adj_matrix,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # Predictor (same)
        self.predictor = EnhancedPredictor(
            hidden_dim=self.hidden_dim,
            horizon=self.horizon,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # Highway (same)
        self.highway_window = min(getattr(args, 'highway_window', HIGHWAY_WINDOW), self.window)
        if self.highway_window > 0:
            self.highway = nn.Linear(self.highway_window, self.horizon)
        else:
            self.highway = None
        self.highway_ratio = nn.Parameter(torch.tensor(0.5))
        
        self._init_weights()
        
    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif p.dim() == 1 and p.size(0) > 0:
                if 'bias' in name:
                    nn.init.zeros_(p)
                else:
                    stdv = 1. / math.sqrt(p.size(0))
                    p.data.uniform_(-stdv, stdv)
    
    def forward(self, x, idx=None):
        B, T, N = x.shape
        x_last = x[:, -1, :]
        
        features = self.temporal_encoder(x)
        features, reg_loss = self.spatial_attn(features)
        predictions = self.predictor(features, x_last)
        
        if self.highway is not None:
            highway_input = x[:, -self.highway_window:, :].permute(0, 2, 1)
            highway_out = self.highway(highway_input)
            ratio = torch.sigmoid(self.highway_ratio)
            predictions = ratio * predictions + (1 - ratio) * highway_out
        
        predictions = predictions.transpose(1, 2)
        
        return predictions, reg_loss
