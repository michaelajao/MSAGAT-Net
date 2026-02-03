"""
EpiSIG-Net v3: Optimal Serial Interval Graph Network

The sweet spot combining:
1. Depthwise separable convs (efficient temporal encoding - from MSAGAT-Net)
2. Standard softmax attention (expressive spatial modeling - from EpiSIG v1)
3. Serial Interval Graph (novel epidemiological contribution)
4. Highway connections (stable long-horizon forecasting)

Target: ~20-30K parameters with strong performance across all graph sizes.

============================================================================
DESIGN RATIONALE
============================================================================

Why this combination works:

1. EFFICIENT TEMPORAL: Depthwise separable convs extract temporal features
   with fewer parameters than standard convolutions while maintaining quality.

2. EXPRESSIVE SPATIAL: Standard softmax attention (not linear) captures
   complex spatial dependencies better than O(N) approximations.

3. EPIDEMIOLOGICAL NOVELTY: Serial Interval Graph models propagation delays -
   the key differentiator from all baselines.

4. STABILITY: Highway connections ensure stable training and good performance
   on long horizons.

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
FEATURE_CHANNELS = 16
HIGHWAY_WINDOW = 4


# =============================================================================
# COMPONENT 1: EFFICIENT TEMPORAL ENCODER (from MSAGAT-Net)
# =============================================================================

class DepthwiseSeparableConv1D(nn.Module):
    """
    Depthwise separable convolution for efficient feature extraction.
    Reduces parameters while maintaining expressivity.
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
    Multi-scale temporal feature extraction with depthwise separable convolutions.
    Efficient and effective for capturing temporal patterns.
    """
    
    def __init__(self, window, hidden_dim=HIDDEN_DIM, 
                 feature_channels=FEATURE_CHANNELS, dropout=DROPOUT):
        super().__init__()
        
        self.window = window
        self.hidden_dim = hidden_dim
        
        # Initial feature extraction
        self.feature_conv = DepthwiseSeparableConv1D(
            in_channels=1,
            out_channels=feature_channels,
            kernel_size=KERNEL_SIZE,
            padding=KERNEL_SIZE // 2,
            dropout=dropout
        )
        
        # Multi-scale processing with different dilations (3 scales like EpiSIG v1)
        self.scale_convs = nn.ModuleList([
            DepthwiseSeparableConv1D(
                in_channels=feature_channels,
                out_channels=feature_channels,
                kernel_size=KERNEL_SIZE,
                padding=(KERNEL_SIZE // 2) * (2 ** i),
                dilation=2 ** i,
                dropout=dropout
            )
            for i in range(3)
        ])
        
        # Learnable scale fusion weights
        self.scale_weights = nn.Parameter(torch.ones(3) / 3)
        
        # Project to hidden dimension
        self.proj = nn.Sequential(
            nn.Linear(feature_channels * window, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, window, nodes]
        Returns:
            features: [batch, nodes, hidden_dim]
        """
        B, T, N = x.shape
        
        # Reshape for convolution: [B*N, 1, T]
        x_temp = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)
        
        # Extract features
        features = self.feature_conv(x_temp)  # [B*N, feature_channels, T]
        
        # Multi-scale processing
        scale_outputs = [conv(features) for conv in self.scale_convs]
        
        # Adaptive fusion
        weights = F.softmax(self.scale_weights, dim=0)
        fused = sum(w * out for w, out in zip(weights, scale_outputs))
        
        # Reshape and project: [B*N, feature_channels, T] -> [B, N, hidden_dim]
        fused = fused.view(B, N, -1)
        out = self.proj(fused)
        
        return out


# =============================================================================
# COMPONENT 2: SERIAL INTERVAL GRAPH (Novel - with Standard Attention)
# =============================================================================

class SerialIntervalGraphV3(nn.Module):
    """
    Serial Interval Graph with standard softmax attention.
    
    NOVEL CONTRIBUTION: Epidemiological propagation delay modeling.
    
    This module learns the generation interval distribution (time between
    successive infections) and uses it to model delayed spatial dependencies.
    
    Key difference from v2: Uses standard O(N²) softmax attention for better
    expressivity on smaller/medium graphs where N² is still tractable.
    """
    
    def __init__(self, num_nodes, hidden_dim=HIDDEN_DIM, max_lag=MAX_LAG,
                 num_heads=ATTENTION_HEADS, adj_matrix=None, dropout=DROPOUT):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # =====================================================================
        # LEARNABLE GENERATION INTERVAL DISTRIBUTION (Epidemiological novelty)
        # =====================================================================
        # α_τ represents the probability that transmission occurs at delay τ
        # Initialize with exponential decay (COVID-19: τ ~ 5-7 days)
        init_weights = torch.exp(-torch.arange(max_lag + 1).float() / 3.0)
        init_weights = init_weights / init_weights.sum()
        self.delay_logits = nn.Parameter(torch.log(init_weights + 1e-8))
        
        # =====================================================================
        # STANDARD MULTI-HEAD ATTENTION (Expressive)
        # =====================================================================
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
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
        # DELAY SIGNAL EMBEDDING
        # =====================================================================
        # Projects the delay-weighted signal into feature space
        self.delay_proj = nn.Linear(1, hidden_dim)
        
    def get_delay_weights(self):
        """Get normalized delay weights (generation interval distribution)."""
        return F.softmax(self.delay_logits, dim=0)
    
    def forward(self, x, features):
        """
        Apply Serial Interval Graph with standard attention.
        
        Args:
            x: Raw time series [batch, window, nodes]
            features: Node features from temporal encoder [batch, nodes, hidden_dim]
            
        Returns:
            updated_features: [batch, nodes, hidden_dim]
        """
        B, T, N = x.shape
        
        # =====================================================================
        # STEP 1: Compute delay-weighted historical signal (NOVEL)
        # =====================================================================
        # This is the key epidemiological innovation
        delay_weights = self.get_delay_weights()  # [max_lag + 1]
        
        # Compute weighted sum of lagged values
        delayed_signal = torch.zeros(B, N, device=x.device)
        for tau in range(min(self.max_lag + 1, T)):
            lagged_value = x[:, -(tau + 1), :]  # [B, N]
            delayed_signal = delayed_signal + delay_weights[tau] * lagged_value
        
        # Project to feature space
        delay_features = self.delay_proj(delayed_signal.unsqueeze(-1))  # [B, N, D]
        
        # =====================================================================
        # STEP 2: Combine with temporal features
        # =====================================================================
        # Blend delay information with learned temporal features
        combined_features = features + delay_features * 0.2
        
        # =====================================================================
        # STEP 3: Standard multi-head attention (Expressive)
        # =====================================================================
        # Compute Q, K, V
        qkv = self.qkv_proj(combined_features)  # [B, N, 3*D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # =====================================================================
        # STEP 4: Blend with geographic prior if available
        # =====================================================================
        if self.adj_prior is not None:
            geo_blend = torch.sigmoid(self.geo_weight)
            # Expand adjacency for heads: [N, N] -> [B, heads, N, N]
            adj_expanded = self.adj_prior.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
            attn_scores = (1 - geo_blend) * attn_scores + geo_blend * adj_expanded * 5.0
        
        # Softmax attention
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attended = torch.matmul(attn_weights, V)  # [B, heads, N, head_dim]
        attended = attended.transpose(1, 2).reshape(B, N, self.hidden_dim)
        
        # =====================================================================
        # STEP 5: Output projection with residual
        # =====================================================================
        out = self.out_proj(attended)
        out = self.dropout(out)
        out = self.norm(out + features)
        
        return out
    
    def get_interpretable_outputs(self):
        """Return interpretable epidemiological parameters."""
        delay_weights = self.get_delay_weights().detach().cpu()
        peak_delay = torch.argmax(delay_weights).item()
        
        return {
            'delay_weights': delay_weights,
            'peak_delay': peak_delay,
            'mean_delay': (delay_weights * torch.arange(len(delay_weights)).float()).sum().item(),
            'interpretation': f"Peak transmission delay: {peak_delay} time steps"
        }


# =============================================================================
# COMPONENT 3: IMPROVED PREDICTOR
# =============================================================================

class ImprovedPredictor(nn.Module):
    """
    Improved prediction module with GRU and adaptive refinement.
    Better for long horizons while keeping parameters reasonable.
    """
    
    def __init__(self, hidden_dim, horizon, dropout=DROPOUT):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # GRU for autoregressive prediction
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )
        
        # Project features to GRU hidden state
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Learnable decay for refinement
        self.log_decay = nn.Parameter(torch.tensor(-2.3))  # ~0.1 initial
        
        # Adaptive refinement gate
        self.refine_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, horizon),
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
        
        # Initialize GRU hidden state from features
        h0 = self.hidden_proj(features.view(B * N, D))
        h0 = h0.unsqueeze(0)  # [1, B*N, hidden_dim//2]
        
        # Autoregressive generation
        current_input = last_value.reshape(B * N, 1, 1)
        predictions = []
        hidden = h0
        
        for t in range(self.horizon):
            gru_out, hidden = self.gru(current_input, hidden)
            pred_t = self.output_proj(gru_out.squeeze(1))  # [B*N, 1]
            predictions.append(pred_t)
            current_input = pred_t.unsqueeze(1)
        
        # Stack: [B*N, H] -> [B, N, H]
        gru_pred = torch.cat(predictions, dim=-1)
        gru_pred = gru_pred.view(B, N, self.horizon)
        
        # Adaptive refinement with decay extrapolation
        gate = self.refine_gate(features)  # [B, N, H]
        
        # Learned decay curve
        decay_rate = torch.exp(self.log_decay)
        time_steps = torch.arange(1, self.horizon + 1, device=features.device).float()
        decay_curve = torch.exp(-decay_rate * time_steps).view(1, 1, -1)
        decay_pred = last_value.unsqueeze(-1) * decay_curve
        
        # Blend GRU prediction with decay
        final_pred = gate * gru_pred + (1 - gate) * decay_pred
        
        return final_pred


# =============================================================================
# MAIN MODEL: EpiSIG-Net v3
# =============================================================================

class EpiSIGNetV3(nn.Module):
    """
    EpiSIG-Net v3: Optimal Serial Interval Graph Network
    
    Architecture:
    1. Efficient temporal encoder (depthwise separable convs)
    2. Serial Interval Graph with standard attention
    3. Improved GRU predictor with refinement
    4. Highway connections for stability
    
    Target: ~20-30K parameters
    Novel contribution: Serial Interval Graph (epidemiological propagation delays)
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
        # TEMPORAL ENCODER (Depthwise Separable - Efficient)
        # =====================================================================
        self.temporal_encoder = EfficientTemporalEncoder(
            window=self.window,
            hidden_dim=self.hidden_dim,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # =====================================================================
        # SERIAL INTERVAL GRAPH (Novel - with Standard Attention)
        # =====================================================================
        self.sig_module = SerialIntervalGraphV3(
            num_nodes=self.num_nodes,
            hidden_dim=self.hidden_dim,
            max_lag=MAX_LAG,
            adj_matrix=adj_matrix,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # =====================================================================
        # PREDICTOR (Improved for long horizons)
        # =====================================================================
        self.predictor = ImprovedPredictor(
            hidden_dim=self.hidden_dim,
            horizon=self.horizon,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # =====================================================================
        # HIGHWAY CONNECTION (Critical for stability)
        # =====================================================================
        self.highway_window = min(getattr(args, 'highway_window', HIGHWAY_WINDOW), self.window)
        if self.highway_window > 0:
            self.highway = nn.Linear(self.highway_window, self.horizon)
        else:
            self.highway = None
        self.highway_ratio = nn.Parameter(torch.tensor(0.5))
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using Xavier uniform (like Cola-GNN/EpiGNN)."""
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
            reg_loss: Regularization loss (0 for compatibility)
        """
        B, T, N = x.shape
        x_last = x[:, -1, :]
        
        # =====================================================================
        # 1. TEMPORAL ENCODING (Efficient depthwise separable convs)
        # =====================================================================
        features = self.temporal_encoder(x)  # [B, N, D]
        
        # =====================================================================
        # 2. SERIAL INTERVAL GRAPH (Novel epidemiological component)
        # =====================================================================
        features = self.sig_module(x, features)  # [B, N, D]
        
        # =====================================================================
        # 3. PREDICTION (Improved for long horizons)
        # =====================================================================
        predictions = self.predictor(features, x_last)  # [B, N, H]
        
        # =====================================================================
        # 4. HIGHWAY CONNECTION (Stability)
        # =====================================================================
        if self.highway is not None:
            highway_input = x[:, -self.highway_window:, :].permute(0, 2, 1)
            highway_out = self.highway(highway_input)  # [B, N, H]
            
            ratio = torch.sigmoid(self.highway_ratio)
            predictions = ratio * predictions + (1 - ratio) * highway_out
        
        # Reshape to [B, H, N]
        predictions = predictions.transpose(1, 2)
        
        # Return dummy reg_loss for compatibility
        reg_loss = torch.tensor(0.0, device=x.device)
        
        return predictions, reg_loss
    
    def get_interpretable_outputs(self):
        """Get interpretable epidemiological parameters."""
        return self.sig_module.get_interpretable_outputs()


# =============================================================================
# ABLATION: EpiSIG-Net v3 without SIG
# =============================================================================

class EpiSIGNetV3_NoSIG(nn.Module):
    """
    Ablation variant without Serial Interval Graph.
    Uses standard attention without epidemiological delays.
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
        
        # Standard graph attention (NO delay modeling - ablation)
        self.spatial_attn = SimpleGraphAttention(
            hidden_dim=self.hidden_dim,
            num_heads=ATTENTION_HEADS,
            adj_matrix=adj_matrix,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # Predictor (same)
        self.predictor = ImprovedPredictor(
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
        features = self.spatial_attn(features)
        predictions = self.predictor(features, x_last)
        
        if self.highway is not None:
            highway_input = x[:, -self.highway_window:, :].permute(0, 2, 1)
            highway_out = self.highway(highway_input)
            ratio = torch.sigmoid(self.highway_ratio)
            predictions = ratio * predictions + (1 - ratio) * highway_out
        
        predictions = predictions.transpose(1, 2)
        reg_loss = torch.tensor(0.0, device=x.device)
        
        return predictions, reg_loss


class SimpleGraphAttention(nn.Module):
    """Simple multi-head graph attention (standard baseline)."""
    
    def __init__(self, hidden_dim=HIDDEN_DIM, num_heads=ATTENTION_HEADS,
                 adj_matrix=None, dropout=DROPOUT):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Geographic prior
        if adj_matrix is not None:
            if not isinstance(adj_matrix, torch.Tensor):
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            adj_norm = adj_matrix / (adj_matrix.sum(dim=-1, keepdim=True) + 1e-8)
            self.register_buffer('adj_prior', adj_norm)
            self.geo_weight = nn.Parameter(torch.tensor(0.3))
        else:
            self.register_buffer('adj_prior', None)
            self.geo_weight = None
        
    def forward(self, x):
        """
        Args:
            x: [batch, nodes, hidden_dim]
        Returns:
            [batch, nodes, hidden_dim]
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv_proj(x)  # [B, N, 3*D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add geographic prior
        if self.adj_prior is not None:
            geo_blend = torch.sigmoid(self.geo_weight)
            adj_expanded = self.adj_prior.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
            attn_scores = (1 - geo_blend) * attn_scores + geo_blend * adj_expanded * 5.0
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).reshape(B, N, D)
        
        # Output projection with residual
        out = self.out_proj(out)
        out = self.norm(out + x)
        
        return out
