"""
EpiSIG-Net v4: Adaptive Serial Interval Graph Network

Goal: Win on ALL datasets (large graphs like Japan, small graphs like Australia)

Key Design Decisions (from papers analysis):
============================================================================

1. CONTRIBUTION OPTIONS (not all epidemiological):
   - Serial Interval Graph (epidemiological - delays)
   - Adaptive Graph Learning (architectural - data-driven)
   - Cross-scale Feature Fusion (architectural - multi-resolution)
   
2. WHY v1 beats v3 on large graphs:
   - v1 has MORE capacity (dilated convs, extra graph attention)
   - v3 is too lightweight for complex patterns
   
3. WHY MSAGAT beats others on small graphs:
   - Depthwise separable prevents overfitting
   - Simpler model for simpler data

SOLUTION: Adaptive capacity model that scales with graph size/complexity

Architecture:
============================================================================
1. Hybrid Temporal Encoder: Depthwise separable + dilated convolutions
2. Adaptive Spatial Module: SIG + optional extra graph attention
3. Multi-scale Feature Fusion: Capture both local and global patterns
4. Smart Highway: Adaptive blending based on learned reliability

Target: ~30-40K parameters (between v3 and v1)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
HIDDEN_DIM = 40  # Balanced for capacity and efficiency
DROPOUT = 0.2
KERNEL_SIZE = 3
MAX_LAG = 7
ATTENTION_HEADS = 4
FEATURE_CHANNELS = 20  # Balanced
HIGHWAY_WINDOW = 4


# =============================================================================
# COMPONENT 1: HYBRID TEMPORAL ENCODER
# =============================================================================
# Combines efficiency of depthwise separable with capacity of dilated convs

class DepthwiseSeparableConv1D(nn.Module):
    """Efficient feature extraction (good for small graphs)."""
    
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
        self.act = nn.GELU()  # GELU often better than ReLU
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.dropout(self.act(self.bn2(self.pointwise(x))))
        return x


class DilatedConvBlock(nn.Module):
    """Higher capacity temporal processing (good for large graphs)."""
    
    def __init__(self, in_channels, out_channels, kernel_size=KERNEL_SIZE, 
                 dilation=1, dropout=DROPOUT):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        out = self.dropout(out)
        
        if self.residual is not None:
            out = out + self.residual(x)
        elif out.shape == x.shape:
            out = out + x
            
        return out


class HybridTemporalEncoder(nn.Module):
    """
    Hybrid encoder combining depthwise separable (efficiency) and dilated (capacity).
    
    Architecture:
    1. Depthwise separable for initial features (efficient)
    2. Multi-scale dilated convolutions (capacity)
    3. Cross-scale fusion
    """
    
    def __init__(self, window, hidden_dim=HIDDEN_DIM, 
                 feature_channels=FEATURE_CHANNELS, dropout=DROPOUT):
        super().__init__()
        
        self.window = window
        self.hidden_dim = hidden_dim
        
        # Stage 1: Efficient initial features (depthwise separable)
        self.initial_conv = DepthwiseSeparableConv1D(
            in_channels=1,
            out_channels=feature_channels,
            kernel_size=KERNEL_SIZE,
            padding=KERNEL_SIZE // 2,
            dropout=dropout
        )
        
        # Stage 2: Multi-scale dilated convolutions for capacity
        self.dilated_blocks = nn.ModuleList([
            DilatedConvBlock(
                feature_channels if i == 0 else feature_channels,
                feature_channels,
                dilation=2 ** i,
                dropout=dropout
            )
            for i in range(3)  # dilation 1, 2, 4
        ])
        
        # Stage 3: Cross-scale fusion (NOVEL - architectural contribution)
        # Learn how to combine different temporal scales
        self.scale_attention = nn.Sequential(
            nn.Linear(feature_channels * 3, feature_channels),
            nn.GELU(),
            nn.Linear(feature_channels, 3),
            nn.Softmax(dim=-1)
        )
        
        # Final projection
        self.proj = nn.Sequential(
            nn.Linear(feature_channels * window, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: [batch, window, nodes]
        Returns:
            features: [batch, nodes, hidden_dim]
        """
        B, T, N = x.shape
        
        # Reshape: [B, T, N] -> [B*N, 1, T]
        x_temp = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)
        
        # Initial efficient features
        features = self.initial_conv(x_temp)  # [B*N, C, T]
        
        # Multi-scale dilated processing
        scale_outputs = []
        for block in self.dilated_blocks:
            features = block(features)
            scale_outputs.append(features)
        
        # Stack scales: [B*N, 3, C, T]
        stacked = torch.stack(scale_outputs, dim=1)
        
        # Compute scale attention weights from global features
        # [B*N, C, T] -> [B*N, C] (global avg pool over time)
        global_feats = [out.mean(dim=-1) for out in scale_outputs]  # List of [B*N, C]
        global_concat = torch.cat(global_feats, dim=-1)  # [B*N, 3*C]
        scale_weights = self.scale_attention(global_concat)  # [B*N, 3]
        
        # Weighted fusion: [B*N, 3, C, T] * [B*N, 3, 1, 1] -> [B*N, C, T]
        scale_weights = scale_weights.unsqueeze(-1).unsqueeze(-1)  # [B*N, 3, 1, 1]
        fused = (stacked * scale_weights).sum(dim=1)  # [B*N, C, T]
        
        # Reshape and project
        fused = fused.view(B, N, -1)  # [B, N, C*T]
        out = self.proj(fused)  # [B, N, D]
        
        return out


# =============================================================================
# COMPONENT 2: ENHANCED SERIAL INTERVAL GRAPH
# =============================================================================
# Keeps the epidemiological novelty but adds more expressive spatial modeling

class EnhancedSerialIntervalGraph(nn.Module):
    """
    Enhanced Serial Interval Graph with multi-head attention.
    
    NOVEL CONTRIBUTION: Epidemiological propagation delay modeling
    ENHANCEMENT: More expressive spatial attention for large graphs
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
        init_weights = torch.exp(-torch.arange(max_lag + 1).float() / 3.0)
        init_weights = init_weights / init_weights.sum()
        self.delay_logits = nn.Parameter(torch.log(init_weights + 1e-8))
        
        # =====================================================================
        # DELAY EMBEDDING (Efficient)
        # =====================================================================
        # Project delay-weighted signal into feature space
        self.delay_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # =====================================================================
        # MULTI-HEAD ATTENTION (More expressive for large graphs)
        # =====================================================================
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # =====================================================================
        # GEOGRAPHIC PRIOR (Optional)
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
        
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # =====================================================================
        # FEEDFORWARD (Efficient)
        # =====================================================================
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def get_delay_weights(self):
        """Get normalized delay weights (generation interval distribution)."""
        return F.softmax(self.delay_logits, dim=0)
    
    def forward(self, x, features):
        """
        Args:
            x: Raw time series [batch, window, nodes]
            features: Node features [batch, nodes, hidden_dim]
            
        Returns:
            updated_features: [batch, nodes, hidden_dim]
        """
        B, T, N = x.shape
        
        # =====================================================================
        # STEP 1: Compute delay-weighted historical signal (NOVEL)
        # =====================================================================
        delay_weights = self.get_delay_weights()
        
        delayed_signal = torch.zeros(B, N, device=x.device)
        for tau in range(min(self.max_lag + 1, T)):
            lagged_value = x[:, -(tau + 1), :]
            delayed_signal = delayed_signal + delay_weights[tau] * lagged_value
        
        # Encode delay signal
        delay_features = self.delay_encoder(delayed_signal.unsqueeze(-1))  # [B, N, D]
        
        # =====================================================================
        # STEP 2: Combine features with delay information
        # =====================================================================
        combined = features + delay_features * 0.3  # Blend
        
        # =====================================================================
        # STEP 3: Multi-head self-attention
        # =====================================================================
        qkv = self.qkv_proj(combined)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Blend with geographic prior
        if self.adj_prior is not None:
            geo_blend = torch.sigmoid(self.geo_weight)
            adj_expanded = self.adj_prior.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
            attn_scores = (1 - geo_blend) * attn_scores + geo_blend * adj_expanded * 5.0
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).reshape(B, N, self.hidden_dim)
        
        # Output projection with residual
        out = self.out_proj(attended)
        out = self.dropout(out)
        out = self.norm1(out + features)  # Residual
        
        # =====================================================================
        # STEP 4: Feedforward with residual
        # =====================================================================
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)
        
        return out
    
    def get_interpretable_outputs(self):
        """Return interpretable epidemiological parameters."""
        delay_weights = self.get_delay_weights().detach().cpu()
        peak_delay = torch.argmax(delay_weights).item()
        mean_delay = (delay_weights * torch.arange(len(delay_weights)).float()).sum().item()
        
        return {
            'delay_weights': delay_weights,
            'peak_delay': peak_delay,
            'mean_delay': mean_delay,
            'interpretation': f"Peak transmission delay: {peak_delay} days, Mean: {mean_delay:.1f} days"
        }


# =============================================================================
# COMPONENT 3: ADDITIONAL GRAPH ATTENTION (Like v1)
# =============================================================================
# This extra layer helps on large graphs (Japan)

class GraphAttentionLayer(nn.Module):
    """
    Additional graph attention for more spatial expressivity.
    Critical for large graphs like Japan.
    """
    
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
            self.geo_weight = nn.Parameter(torch.tensor(0.2))
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
        
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if self.adj_prior is not None:
            geo_blend = torch.sigmoid(self.geo_weight)
            adj_expanded = self.adj_prior.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, -1, -1)
            attn_scores = (1 - geo_blend) * attn_scores + geo_blend * adj_expanded * 5.0
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        out = self.norm(out + x)
        
        return out


# =============================================================================
# COMPONENT 4: ADAPTIVE PREDICTOR
# =============================================================================

class AdaptivePredictor(nn.Module):
    """
    Adaptive prediction module.
    Uses both GRU (for complex patterns) and direct projection (for simple patterns).
    """
    
    def __init__(self, hidden_dim, horizon, dropout=DROPOUT):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # GRU for autoregressive prediction (complex patterns)
        self.gru = nn.GRU(
            input_size=1,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True
        )
        
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim // 2)
        
        self.gru_output = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Direct projection (simple patterns)
        self.direct_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon)
        )
        
        # Adaptive gate: learn when to use GRU vs direct
        self.pred_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Decay for extrapolation
        self.log_decay = nn.Parameter(torch.tensor(-2.3))
        
    def forward(self, features, last_value):
        """
        Args:
            features: [batch, nodes, hidden_dim]
            last_value: [batch, nodes]
            
        Returns:
            predictions: [batch, nodes, horizon]
        """
        B, N, D = features.shape
        
        # Method 1: GRU autoregressive
        h0 = self.hidden_proj(features.view(B * N, D)).unsqueeze(0)
        current_input = last_value.reshape(B * N, 1, 1)
        gru_preds = []
        hidden = h0
        
        for t in range(self.horizon):
            gru_out, hidden = self.gru(current_input, hidden)
            pred_t = self.gru_output(gru_out.squeeze(1))
            gru_preds.append(pred_t)
            current_input = pred_t.unsqueeze(1)
        
        gru_pred = torch.cat(gru_preds, dim=-1).view(B, N, self.horizon)
        
        # Method 2: Direct projection
        direct_pred = self.direct_pred(features)  # [B, N, H]
        
        # Adaptive blending
        gate = self.pred_gate(features)  # [B, N, 1]
        
        # Blend: GRU for complex, direct for simple
        blended_pred = gate * gru_pred + (1 - gate) * direct_pred
        
        # Add decay-based extrapolation for stability
        decay_rate = torch.exp(self.log_decay)
        time_steps = torch.arange(1, self.horizon + 1, device=features.device).float()
        decay_curve = torch.exp(-decay_rate * time_steps).view(1, 1, -1)
        decay_pred = last_value.unsqueeze(-1) * decay_curve
        
        # Small contribution from decay for stability
        final_pred = 0.9 * blended_pred + 0.1 * decay_pred
        
        return final_pred


# =============================================================================
# MAIN MODEL: EpiSIG-Net v4
# =============================================================================

class EpiSIGNetV4(nn.Module):
    """
    EpiSIG-Net v4: Adaptive Serial Interval Graph Network
    
    Goal: Win on ALL datasets (small, medium, and large graphs)
    
    Key Improvements over v3:
    1. Hybrid temporal encoder (efficiency + capacity)
    2. Enhanced SIG with feedforward
    3. Additional graph attention layer (critical for large graphs)
    4. Adaptive predictor (GRU + direct)
    
    Target: ~35-45K parameters
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
        # 1. HYBRID TEMPORAL ENCODER (Efficiency + Capacity)
        # =====================================================================
        self.temporal_encoder = HybridTemporalEncoder(
            window=self.window,
            hidden_dim=self.hidden_dim,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # =====================================================================
        # 2. ENHANCED SERIAL INTERVAL GRAPH (Novel + Expressive)
        # =====================================================================
        self.sig_module = EnhancedSerialIntervalGraph(
            num_nodes=self.num_nodes,
            hidden_dim=self.hidden_dim,
            max_lag=MAX_LAG,
            adj_matrix=adj_matrix,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # =====================================================================
        # 3. ADDITIONAL GRAPH ATTENTION (For large graphs like Japan)
        # =====================================================================
        self.graph_attention = GraphAttentionLayer(
            hidden_dim=self.hidden_dim,
            num_heads=ATTENTION_HEADS,
            adj_matrix=adj_matrix,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # =====================================================================
        # 4. ADAPTIVE PREDICTOR
        # =====================================================================
        self.predictor = AdaptivePredictor(
            hidden_dim=self.hidden_dim,
            horizon=self.horizon,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # =====================================================================
        # 5. HIGHWAY CONNECTION (Stability)
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
            reg_loss: Regularization loss (0)
        """
        B, T, N = x.shape
        x_last = x[:, -1, :]
        
        # 1. Temporal encoding (hybrid)
        features = self.temporal_encoder(x)  # [B, N, D]
        
        # 2. Serial Interval Graph (novel epidemiological component)
        features = self.sig_module(x, features)  # [B, N, D]
        
        # 3. Additional graph attention (helps large graphs)
        features = self.graph_attention(features)  # [B, N, D]
        
        # 4. Adaptive prediction
        predictions = self.predictor(features, x_last)  # [B, N, H]
        
        # 5. Highway connection
        if self.highway is not None:
            highway_input = x[:, -self.highway_window:, :].permute(0, 2, 1)
            highway_out = self.highway(highway_input)
            
            ratio = torch.sigmoid(self.highway_ratio)
            predictions = ratio * predictions + (1 - ratio) * highway_out
        
        # Reshape to [B, H, N]
        predictions = predictions.transpose(1, 2)
        
        reg_loss = torch.tensor(0.0, device=x.device)
        
        return predictions, reg_loss
    
    def get_interpretable_outputs(self):
        """Get interpretable epidemiological parameters."""
        return self.sig_module.get_interpretable_outputs()


# =============================================================================
# ABLATION: Without SIG
# =============================================================================

class EpiSIGNetV4_NoSIG(nn.Module):
    """Ablation variant without Serial Interval Graph."""
    
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
        self.temporal_encoder = HybridTemporalEncoder(
            window=self.window,
            hidden_dim=self.hidden_dim,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # Two graph attention layers (replaces SIG + attention)
        self.graph_attention1 = GraphAttentionLayer(
            hidden_dim=self.hidden_dim,
            num_heads=ATTENTION_HEADS,
            adj_matrix=adj_matrix,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        self.graph_attention2 = GraphAttentionLayer(
            hidden_dim=self.hidden_dim,
            num_heads=ATTENTION_HEADS,
            adj_matrix=adj_matrix,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # Predictor (same)
        self.predictor = AdaptivePredictor(
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
        features = self.graph_attention1(features)
        features = self.graph_attention2(features)
        predictions = self.predictor(features, x_last)
        
        if self.highway is not None:
            highway_input = x[:, -self.highway_window:, :].permute(0, 2, 1)
            highway_out = self.highway(highway_input)
            ratio = torch.sigmoid(self.highway_ratio)
            predictions = ratio * predictions + (1 - ratio) * highway_out
        
        predictions = predictions.transpose(1, 2)
        reg_loss = torch.tensor(0.0, device=x.device)
        
        return predictions, reg_loss


# =============================================================================
# UTILITY
# =============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    from argparse import Namespace
    
    args = Namespace(
        window=20,
        horizon=7,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        highway_window=HIGHWAY_WINDOW,
        cuda=False,
        save_dir='test'
    )
    
    class MockData:
        m = 47  # Japan prefectures
        adj = None
    
    data = MockData()
    
    # Test v4
    model = EpiSIGNetV4(args, data)
    params = count_parameters(model)
    print(f"EpiSIG-Net v4 parameters: {params:,}")
    
    # Test forward pass
    x = torch.randn(4, 20, 47)
    pred, loss = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    
    # Test ablation
    model_ablation = EpiSIGNetV4_NoSIG(args, data)
    print(f"EpiSIG-Net v4 (No SIG) parameters: {count_parameters(model_ablation):,}")
    
    # Compare with different graph sizes
    for n_nodes in [8, 17, 47, 50]:
        data.m = n_nodes
        model = EpiSIGNetV4(args, data)
        print(f"Nodes={n_nodes}: {count_parameters(model):,} params")
