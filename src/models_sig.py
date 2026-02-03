"""
EpiSIG-Net: Epidemic Serial Interval Graph Network

A lightweight (~40-50K params) spatiotemporal epidemic forecasting model with
ONE focused novel contribution: Serial Interval Graph (SIG) for explicit
propagation delay modeling.

============================================================================
ARCHITECTURE DESIGN (Following EpiGNN/Cola-GNN principles)
============================================================================

1. Temporal Feature Extraction: Dilated convolutions (proven, like Cola-GNN)
2. Spatial Processing: Simple graph attention (proven)
3. Novel Component: Serial Interval Graph - models propagation delays
4. Prediction: Direct prediction with refinement (simple)

============================================================================
NOVEL CONTRIBUTION: SERIAL INTERVAL GRAPH (SIG)
============================================================================

Epidemiological Principle:
- Disease transmission has inherent delays (serial interval)
- The serial interval (time between symptom onset in successive cases) is a
  key epidemiological parameter (COVID-19: 4-7 days, Flu: 2-3 days)
- NO existing model explicitly incorporates these delays in graph learning

Mathematical Formulation:
    Influence_ij(t) = Σ_τ α_τ · x_j(t-τ) · A_ij
    
Where:
- τ ranges over delay values (0 to max_lag)
- α_τ is a learnable delay weight (generation interval distribution)
- x_j(t-τ) is the value in region j at time t-τ
- A_ij is the learned spatial adjacency

The delay weights α_τ are learnable and interpretable as the generation
interval distribution for the disease being modeled.

============================================================================
DIFFERENTIATION FROM COMPETITORS
============================================================================

vs. EpiGNN: EpiGNN models transmission risk but at the SAME timestep.
           We model transmission with EXPLICIT TIME DELAYS.

vs. Cola-GNN: Cola-GNN's cross-location attention is instantaneous.
             Our SIG captures lagged spatial dependencies.

vs. DCRNN: DCRNN's diffusion is over spatial hops, not temporal delays.
          We diffuse over both space AND time.

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
NUM_DILATIONS = 3  # Like Cola-GNN
MAX_LAG = 7  # Serial interval range
ATTENTION_HEADS = 4
HIGHWAY_WINDOW = 4  # For autoregressive component (like Cola-GNN/EpiGNN)


# =============================================================================
# COMPONENT 1: TEMPORAL FEATURE EXTRACTION (Dilated Convolutions)
# =============================================================================

class DilatedTemporalBlock(nn.Module):
    """
    Dilated temporal convolution block for multi-scale temporal feature extraction.
    Similar to Cola-GNN's temporal processing.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=KERNEL_SIZE, 
                 dilation=1, dropout=DROPOUT):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection if dimensions match
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x):
        """
        Args:
            x: [batch, channels, time]
        Returns:
            [batch, out_channels, time]
        """
        out = self.conv(x)
        out = self.bn(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        if self.residual is not None:
            out = out + self.residual(x)
        elif out.shape == x.shape:
            out = out + x
            
        return out


class TemporalEncoder(nn.Module):
    """
    Multi-scale temporal feature extraction using stacked dilated convolutions.
    Extracts features at multiple temporal resolutions.
    """
    
    def __init__(self, window, hidden_dim=HIDDEN_DIM, num_dilations=NUM_DILATIONS,
                 dropout=DROPOUT):
        super().__init__()
        
        self.window = window
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(window, hidden_dim)
        
        # Stacked dilated convolutions with increasing dilation rates
        self.dilated_blocks = nn.ModuleList([
            DilatedTemporalBlock(
                hidden_dim if i > 0 else 1,
                hidden_dim,
                dilation=2**i,
                dropout=dropout
            )
            for i in range(num_dilations)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * window, hidden_dim),
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
        
        # Process each node's time series
        # [B, T, N] -> [B*N, 1, T]
        x_temp = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)
        
        # Apply dilated convolutions
        for block in self.dilated_blocks:
            x_temp = block(x_temp)
        
        # [B*N, hidden_dim, T] -> [B, N, hidden_dim * T] -> [B, N, hidden_dim]
        x_temp = x_temp.view(B, N, -1)
        features = self.output_proj(x_temp)
        
        return features


# =============================================================================
# COMPONENT 2: SERIAL INTERVAL GRAPH (NOVEL CONTRIBUTION)
# =============================================================================

class SerialIntervalGraph(nn.Module):
    """
    Serial Interval Graph (SIG) - The core novel contribution.
    
    Models epidemic propagation delays by learning a generation interval
    distribution and applying it to spatial message passing.
    
    NOVEL: First model to explicitly incorporate epidemiological propagation
    delays into graph-based spatial learning.
    
    Key insight: Disease doesn't spread instantaneously. A case in region A
    at time t affects region B at time t+τ, where τ follows the generation
    interval distribution.
    
    Args:
        num_nodes: Number of spatial locations
        hidden_dim: Feature dimension
        max_lag: Maximum delay to consider (default: 7 days)
        adj_matrix: Optional predefined adjacency matrix
    """
    
    def __init__(self, num_nodes, hidden_dim=HIDDEN_DIM, max_lag=MAX_LAG, 
                 adj_matrix=None):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag
        
        # =====================================================================
        # LEARNABLE GENERATION INTERVAL DISTRIBUTION
        # =====================================================================
        # Initialize with exponential decay (typical for many diseases)
        # α_τ represents the probability that transmission occurs at delay τ
        init_weights = torch.exp(-torch.arange(max_lag + 1).float() / 3.0)
        init_weights = init_weights / init_weights.sum()  # Normalize
        self.delay_logits = nn.Parameter(torch.log(init_weights + 1e-8))
        
        # =====================================================================
        # SPATIAL AFFINITY LEARNING
        # =====================================================================
        # Learn spatial relationships between regions
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # =====================================================================
        # OPTIONAL GEOGRAPHIC PRIOR
        # =====================================================================
        if adj_matrix is not None:
            if not isinstance(adj_matrix, torch.Tensor):
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            # Normalize adjacency
            adj_norm = adj_matrix / (adj_matrix.sum(dim=-1, keepdim=True) + 1e-8)
            self.register_buffer('adj_prior', adj_norm)
            # Learnable blend weight
            self.geo_weight = nn.Parameter(torch.tensor(0.3))
        else:
            self.register_buffer('adj_prior', None)
            self.geo_weight = None
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def get_delay_weights(self):
        """Get normalized delay weights (generation interval distribution)."""
        return F.softmax(self.delay_logits, dim=0)
    
    def forward(self, x, features):
        """
        Apply Serial Interval Graph message passing.
        
        Args:
            x: Raw time series [batch, window, nodes]
            features: Node features [batch, nodes, hidden_dim]
            
        Returns:
            updated_features: [batch, nodes, hidden_dim]
        """
        B, T, N = x.shape
        
        # =====================================================================
        # STEP 1: Compute delay-weighted historical values
        # =====================================================================
        # Get generation interval distribution
        delay_weights = self.get_delay_weights()  # [max_lag + 1]
        
        # Compute weighted sum of lagged values for each node
        # This represents "incoming infection pressure" from past
        delayed_signal = torch.zeros(B, N, device=x.device)
        
        for tau in range(min(self.max_lag + 1, T)):
            # x[:, -(tau+1), :] is the value at time t-tau
            lagged_value = x[:, -(tau + 1), :]  # [B, N]
            delayed_signal = delayed_signal + delay_weights[tau] * lagged_value
        
        # =====================================================================
        # STEP 2: Compute spatial affinity (learned adjacency)
        # =====================================================================
        Q = self.query_proj(features)  # [B, N, D]
        K = self.key_proj(features)    # [B, N, D]
        V = self.value_proj(features)  # [B, N, D]
        
        # Compute attention scores
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        
        # Blend with geographic prior if available
        if self.adj_prior is not None:
            geo_blend = torch.sigmoid(self.geo_weight)
            # Expand adj_prior for batch: [N, N] -> [B, N, N]
            adj_expanded = self.adj_prior.unsqueeze(0).expand(B, -1, -1)
            attn_scores = (1 - geo_blend) * attn_scores + geo_blend * adj_expanded * 5.0
        
        # Normalize to get attention weights (spatial affinity)
        spatial_attn = F.softmax(attn_scores, dim=-1)  # [B, N, N]
        
        # =====================================================================
        # STEP 3: Propagate delayed signal through spatial graph
        # =====================================================================
        # delayed_signal: [B, N] - historical infection pressure per node
        # spatial_attn: [B, N, N] - how much each node influences others
        
        # Propagate: each node receives delayed signal from neighbors
        # [B, N, N] @ [B, N, 1] -> [B, N, 1]
        propagated_signal = torch.bmm(spatial_attn, delayed_signal.unsqueeze(-1)).squeeze(-1)
        
        # =====================================================================
        # STEP 4: Combine propagated signal with features
        # =====================================================================
        # Standard attention-based feature update
        attended_features = torch.bmm(spatial_attn, V)  # [B, N, D]
        
        # Add propagated delayed signal as additional information
        # Scale propagated_signal to match feature dimension
        signal_embedding = propagated_signal.unsqueeze(-1) * 0.1  # [B, N, 1]
        
        # Combine: features + attended + signal
        combined = features + attended_features + signal_embedding
        
        # Project output
        output = self.output_proj(combined)
        
        return output
    
    def get_interpretable_outputs(self):
        """
        Return interpretable outputs for analysis.
        
        Returns:
            dict with:
            - delay_weights: Learned generation interval distribution
            - peak_delay: Most likely transmission delay
        """
        delay_weights = self.get_delay_weights().detach().cpu()
        peak_delay = torch.argmax(delay_weights).item()
        
        return {
            'delay_weights': delay_weights,
            'peak_delay': peak_delay,
            'interpretation': f"Peak transmission delay: {peak_delay} time steps"
        }


# =============================================================================
# COMPONENT 3: SIMPLE GRAPH ATTENTION (Standard, like EpiGNN)
# =============================================================================

class SimpleGraphAttention(nn.Module):
    """
    Simple multi-head graph attention for spatial feature aggregation.
    Standard component, similar to EpiGNN's spatial processing.
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM, num_heads=ATTENTION_HEADS, 
                 dropout=DROPOUT):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        
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
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        out = torch.matmul(attn_weights, V)  # [B, heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        
        # Output projection and residual
        out = self.out_proj(out)
        out = self.norm(out + x)
        
        return out


# =============================================================================
# COMPONENT 4: PREDICTION HEAD (Simple, like EpiGNN)
# =============================================================================

class PredictionHead(nn.Module):
    """
    Simple prediction head with optional refinement.
    Direct prediction without complex multi-head strategies.
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM, horizon=7, dropout=DROPOUT):
        super().__init__()
        
        self.horizon = horizon
        
        # Simple MLP predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon)
        )
        
        # Refinement gate (blend with last observation)
        self.refine_gate = nn.Sequential(
            nn.Linear(hidden_dim, horizon),
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
        # Direct prediction
        pred = self.predictor(features)  # [B, N, H]
        
        # Refinement with last observation
        gate = self.refine_gate(features)  # [B, N, H]
        last_expanded = last_value.unsqueeze(-1).expand(-1, -1, self.horizon)
        
        # Blend: model prediction + extrapolation from last value
        output = gate * pred + (1 - gate) * last_expanded
        
        return output


# =============================================================================
# MAIN MODEL: EpiSIG-Net
# =============================================================================

class EpiSIGNet(nn.Module):
    """
    EpiSIG-Net: Epidemic Serial Interval Graph Network
    
    A lightweight spatiotemporal model with ONE focused novel contribution:
    Serial Interval Graph (SIG) for explicit propagation delay modeling.
    
    Architecture (similar to EpiGNN/Cola-GNN):
    1. Temporal Encoder: Dilated convolutions for temporal features
    2. Serial Interval Graph: NOVEL - delay-aware spatial message passing
    3. Graph Attention: Standard spatial feature refinement
    4. Prediction Head: Simple direct prediction with refinement
    5. Highway Connection: Autoregressive component (like Cola-GNN/EpiGNN)
    
    Total Parameters: ~40-50K (comparable to EpiGNN/Cola-GNN)
    
    Args:
        args: Configuration with window, horizon, hidden_dim, etc.
        data: Data object with m (nodes), adj (optional)
    """
    
    def __init__(self, args, data):
        super().__init__()
        
        self.num_nodes = data.m
        self.window = args.window
        self.horizon = args.horizon
        self.hidden_dim = getattr(args, 'hidden_dim', HIDDEN_DIM)
        
        # Get adjacency matrix if available
        adj_matrix = getattr(data, 'adj', None)
        if adj_matrix is not None and not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        
        # Component 1: Temporal Feature Extraction
        self.temporal_encoder = TemporalEncoder(
            self.window, 
            self.hidden_dim,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # Component 2: Serial Interval Graph (NOVEL)
        self.sig = SerialIntervalGraph(
            self.num_nodes,
            self.hidden_dim,
            max_lag=getattr(args, 'max_lag', MAX_LAG),
            adj_matrix=adj_matrix
        )
        
        # Component 3: Standard Graph Attention
        self.graph_attention = SimpleGraphAttention(
            self.hidden_dim,
            num_heads=getattr(args, 'attention_heads', ATTENTION_HEADS),
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # Component 4: Prediction Head
        self.prediction_head = PredictionHead(
            self.hidden_dim,
            self.horizon,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # =====================================================================
        # Component 5: Highway/Autoregressive Connection (like Cola-GNN/EpiGNN)
        # Critical for all successful epidemic forecasting models!
        # =====================================================================
        self.highway_window = min(getattr(args, 'highway_window', HIGHWAY_WINDOW), self.window)
        if self.highway_window > 0:
            self.highway = nn.Linear(self.highway_window, self.horizon)
        else:
            self.highway = None
        
        # Ratio for blending highway with model output
        self.highway_ratio = nn.Parameter(torch.tensor(0.5))
        
        # Initialize weights properly (Xavier - used by all successful models)
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
            # Skip 0-dimensional tensors (scalars like highway_ratio)
        
    def forward(self, x, idx=None):
        """
        Forward pass.
        
        Args:
            x: Input time series [batch, window, nodes]
            idx: Unused (for compatibility)
            
        Returns:
            predictions: [batch, horizon, nodes]
            aux_loss: Auxiliary loss (0.0 for this model)
        """
        B, T, N = x.shape
        x_last = x[:, -1, :]  # Last observed values
        
        # 1. Extract temporal features
        features = self.temporal_encoder(x)  # [B, N, D]
        
        # 2. Apply Serial Interval Graph (NOVEL)
        features = self.sig(x, features)  # [B, N, D]
        
        # 3. Apply standard graph attention
        features = self.graph_attention(features)  # [B, N, D]
        
        # 4. Predict from model
        model_pred = self.prediction_head(features, x_last)  # [B, N, H]
        model_pred = model_pred.transpose(1, 2)  # [B, H, N]
        
        # 5. Highway/Autoregressive connection (like Cola-GNN/EpiGNN)
        # This is CRITICAL for stable forecasting
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
        
        return predictions, torch.tensor(0.0, device=x.device)
    
    def get_interpretable_outputs(self, x):
        """
        Get interpretable epidemiological outputs.
        
        Returns:
            dict with learned generation interval distribution
        """
        return self.sig.get_interpretable_outputs()


# =============================================================================
# ABLATION VARIANT (For proving SIG contribution)
# =============================================================================

class EpiSIGNet_NoSIG(nn.Module):
    """
    EpiSIG-Net without Serial Interval Graph (ablation baseline).
    Used to prove the contribution of the SIG component.
    Has same highway connection for fair comparison.
    """
    
    def __init__(self, args, data):
        super().__init__()
        
        self.num_nodes = data.m
        self.window = args.window
        self.horizon = args.horizon
        self.hidden_dim = getattr(args, 'hidden_dim', HIDDEN_DIM)
        
        # Same components but NO SIG
        self.temporal_encoder = TemporalEncoder(
            self.window, 
            self.hidden_dim,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # Standard graph attention (replaces SIG)
        self.graph_attention1 = SimpleGraphAttention(
            self.hidden_dim,
            num_heads=getattr(args, 'attention_heads', ATTENTION_HEADS),
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        self.graph_attention2 = SimpleGraphAttention(
            self.hidden_dim,
            num_heads=getattr(args, 'attention_heads', ATTENTION_HEADS),
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        self.prediction_head = PredictionHead(
            self.hidden_dim,
            self.horizon,
            dropout=getattr(args, 'dropout', DROPOUT)
        )
        
        # Highway connection (same as main model for fair comparison)
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
            # Skip 0-dimensional tensors (scalars)
        
    def forward(self, x, idx=None):
        B, T, N = x.shape
        x_last = x[:, -1, :]
        
        features = self.temporal_encoder(x)
        features = self.graph_attention1(features)
        features = self.graph_attention2(features)
        model_pred = self.prediction_head(features, x_last)
        model_pred = model_pred.transpose(1, 2)
        
        # Highway connection
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
        
        return predictions, torch.tensor(0.0, device=x.device)


# =============================================================================
# UTILITY: Count parameters
# =============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    class MockArgs:
        window = 20
        horizon = 7
        hidden_dim = 32
        dropout = 0.2
        
    class MockData:
        m = 47  # Japan prefectures
        adj = None
    
    args = MockArgs()
    data = MockData()
    
    model = EpiSIGNet(args, data)
    print(f"EpiSIG-Net parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(4, 20, 47)
    pred, loss = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    
    # Test interpretable outputs
    interp = model.get_interpretable_outputs(x)
    print(f"Delay weights: {interp['delay_weights']}")
    print(f"Peak delay: {interp['peak_delay']}")
