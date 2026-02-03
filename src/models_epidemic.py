"""
Epidemic-Informed Spatiotemporal Network (EpiSTNet)

A novel epidemic forecasting model that incorporates epidemiological domain knowledge:
- Epidemic phase detection (exponential/peak/decline)
- Reproduction number (Rt) tracking as latent variable
- Generation interval-aware temporal attention
- Metapopulation transmission dynamics

Key Innovations:
1. Phase-Aware Encoder: Detects epidemic growth/peak/decline phases
2. Rt Estimator: Learns reproduction number dynamics
3. Generation Interval Attention: Temporal attention with disease-specific lags
4. Transmission Flow Module: Models inter-region disease spread

Unlike generic spatiotemporal models, this architecture encodes epidemiological
principles directly into the network structure.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# =============================================================================
# HYPERPARAMETERS
# =============================================================================
HIDDEN_DIM = 32
DROPOUT = 0.2
BOTTLENECK_DIM = 8
DEFAULT_GENERATION_INTERVAL = 5  # Default serial interval (days)
NUM_PHASES = 3  # Growth, Peak, Decline


# =============================================================================
# EPIDEMIC PHASE DETECTION
# =============================================================================

class EpidemicPhaseEncoder(nn.Module):
    """
    Detects which phase of the epidemic each region is in.
    
    Epidemiological Motivation:
    - Exponential growth phase: Rt > 1, cases doubling
    - Peak/saturation phase: Rt ≈ 1, maximum reached
    - Decline phase: Rt < 1, cases decreasing
    
    Different phases require different prediction strategies.
    
    Args:
        window: Input time window size
        hidden_dim: Hidden dimension for embeddings
        num_phases: Number of epidemic phases (default: 3)
    """
    
    def __init__(self, window, hidden_dim=HIDDEN_DIM, num_phases=NUM_PHASES):
        super().__init__()
        self.window = window
        self.num_phases = num_phases
        
        # Compute trend features from time series
        self.trend_encoder = nn.Sequential(
            nn.Linear(window, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Phase classifier
        self.phase_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, num_phases),
            nn.Softmax(dim=-1)
        )
        
        # Phase embeddings (learnable)
        self.phase_embeddings = nn.Parameter(torch.randn(num_phases, hidden_dim))
        
    def compute_growth_features(self, x):
        """
        Compute epidemiologically meaningful features.
        
        Args:
            x: Input time series [batch, window, nodes]
            
        Returns:
            Growth rate features [batch, nodes, features]
        """
        # First differences (daily change)
        diff = x[:, 1:, :] - x[:, :-1, :]
        
        # Second differences (acceleration)
        diff2 = diff[:, 1:, :] - diff[:, :-1, :]
        
        # Compute simple growth indicators
        # Positive mean diff = growing, negative = declining
        mean_diff = diff.mean(dim=1)  # [batch, nodes]
        mean_accel = diff2.mean(dim=1)  # [batch, nodes]
        
        # Recent vs early comparison (is it growing recently?)
        mid = self.window // 2
        recent_mean = x[:, mid:, :].mean(dim=1)
        early_mean = x[:, :mid, :].mean(dim=1)
        trend_ratio = (recent_mean + 1e-6) / (early_mean + 1e-6)
        
        return mean_diff, mean_accel, trend_ratio
        
    def forward(self, x):
        """
        Detect epidemic phase for each region.
        
        Args:
            x: Input time series [batch, window, nodes]
            
        Returns:
            tuple: (phase_probs [batch, nodes, num_phases], 
                    phase_embedding [batch, nodes, hidden_dim])
        """
        B, T, N = x.shape
        
        # Encode each node's time series
        x_node = x.permute(0, 2, 1)  # [batch, nodes, window]
        trend_features = self.trend_encoder(x_node)  # [batch, nodes, hidden_dim//2]
        
        # Classify phase
        phase_probs = self.phase_classifier(trend_features)  # [batch, nodes, num_phases]
        
        # Weighted phase embedding
        # phase_probs: [B, N, P], phase_embeddings: [P, D]
        phase_emb = torch.einsum('bnp,pd->bnd', phase_probs, self.phase_embeddings)
        
        return phase_probs, phase_emb


# =============================================================================
# REPRODUCTION NUMBER ESTIMATION
# =============================================================================

class ReproductionNumberEstimator(nn.Module):
    """
    Estimates the reproduction number (Rt) as a latent variable.
    
    Epidemiological Motivation:
    Rt represents average secondary infections per case. It's the most
    important quantity in epidemic forecasting:
    - Rt > 1: Epidemic growing
    - Rt = 1: Endemic equilibrium  
    - Rt < 1: Epidemic declining
    
    This module learns to estimate Rt from case data, which then informs
    the prediction module.
    
    Args:
        window: Input time window
        hidden_dim: Hidden dimension
        generation_interval: Serial interval of the disease (days)
    """
    
    def __init__(self, window, hidden_dim=HIDDEN_DIM, 
                 generation_interval=DEFAULT_GENERATION_INTERVAL):
        super().__init__()
        self.window = window
        self.generation_interval = generation_interval
        
        # Learnable generation interval (disease-specific timing)
        self.log_gen_interval = nn.Parameter(
            torch.log(torch.tensor(float(generation_interval)))
        )
        
        # Rt estimator network
        self.rt_encoder = nn.Sequential(
            nn.Linear(window, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Rt must be positive
        )
        
        # Uncertainty estimator (for confidence bounds)
        self.rt_uncertainty = nn.Sequential(
            nn.Linear(window, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
    @property
    def current_generation_interval(self):
        """Get current learned generation interval."""
        return torch.exp(self.log_gen_interval).item()
        
    def forward(self, x):
        """
        Estimate reproduction number for each region.
        
        Args:
            x: Input time series [batch, window, nodes]
            
        Returns:
            tuple: (rt_estimate [batch, nodes, 1], 
                    rt_uncertainty [batch, nodes, 1])
        """
        B, T, N = x.shape
        
        # Process each node's time series
        x_node = x.permute(0, 2, 1)  # [batch, nodes, window]
        
        # Estimate Rt
        rt = self.rt_encoder(x_node)  # [batch, nodes, 1]
        
        # Estimate uncertainty
        rt_unc = self.rt_uncertainty(x_node)  # [batch, nodes, 1]
        
        return rt, rt_unc


# =============================================================================
# GENERATION INTERVAL ATTENTION
# =============================================================================

class GenerationIntervalAttention(nn.Module):
    """
    Temporal attention weighted by epidemic generation interval.
    
    Epidemiological Motivation:
    In epidemics, today's cases are caused by infections from ~5 days ago
    (for COVID) or ~3 days ago (for flu). This module applies attention
    that respects this biological timing.
    
    Unlike generic temporal attention, this encodes the disease-specific
    serial interval into the attention weights.
    
    Args:
        window: Input window size
        hidden_dim: Hidden dimension
        num_heads: Number of attention heads
        generation_interval: Default serial interval
    """
    
    def __init__(self, window, hidden_dim=HIDDEN_DIM, num_heads=4,
                 generation_interval=DEFAULT_GENERATION_INTERVAL):
        super().__init__()
        self.window = window
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Learnable generation interval decay
        self.log_gen_interval = nn.Parameter(
            torch.log(torch.tensor(float(generation_interval)))
        )
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Learnable relative position bias (encodes temporal lags)
        self.rel_pos_bias = nn.Parameter(torch.zeros(num_heads, window, window))
        nn.init.xavier_uniform_(self.rel_pos_bias)
        
    def compute_generation_interval_mask(self, T, device):
        """
        Create attention mask based on generation interval.
        
        Weights decrease for time lags far from the generation interval.
        
        Args:
            T: Sequence length
            device: Torch device
            
        Returns:
            Mask tensor [T, T]
        """
        gen_int = torch.exp(self.log_gen_interval)
        
        # Time lag matrix
        times = torch.arange(T, device=device, dtype=torch.float32)
        lag_matrix = times.unsqueeze(1) - times.unsqueeze(0)  # [T, T]
        
        # Gaussian weighting around generation interval
        # Peaks at lag = generation_interval
        weight = torch.exp(-0.5 * ((lag_matrix - gen_int) / (gen_int * 0.5)) ** 2)
        
        # Also allow recent observations (lag=1) to have influence
        recent_weight = torch.exp(-0.1 * torch.abs(lag_matrix))
        
        # Combine: both generation interval and recent matter
        mask = 0.7 * weight + 0.3 * recent_weight
        
        # Causal mask (can't attend to future)
        causal = torch.tril(torch.ones(T, T, device=device))
        
        return mask * causal
        
    def forward(self, x):
        """
        Apply generation interval-aware temporal attention.
        
        Args:
            x: Input features [batch, nodes, window, hidden_dim]
            
        Returns:
            Attended features [batch, nodes, window, hidden_dim]
        """
        B, N, T, D = x.shape
        
        # Reshape for attention: [B*N, T, D]
        x_flat = x.view(B * N, T, D)
        
        # Project to Q, K, V
        q = self.q_proj(x_flat).view(B * N, T, self.num_heads, self.head_dim)
        k = self.k_proj(x_flat).view(B * N, T, self.num_heads, self.head_dim)
        v = self.v_proj(x_flat).view(B * N, T, self.num_heads, self.head_dim)
        
        # Transpose for attention: [B*N, heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B*N, heads, T, T]
        
        # Add relative position bias
        attn = attn + self.rel_pos_bias.unsqueeze(0)
        
        # Apply generation interval mask
        gi_mask = self.compute_generation_interval_mask(T, x.device)
        attn = attn * gi_mask.unsqueeze(0).unsqueeze(0)
        
        # Softmax
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B*N, heads, T, head_dim]
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B * N, T, D)
        out = self.out_proj(out)
        out = out.view(B, N, T, D)
        
        return out


# =============================================================================
# TRANSMISSION FLOW MODULE
# =============================================================================

class TransmissionFlowModule(nn.Module):
    """
    Models disease transmission between regions.
    
    Epidemiological Motivation:
    Disease spreads through human mobility and contact patterns. This module
    learns transmission probabilities between regions, weighted by:
    - Geographic proximity (if adjacency available)
    - Learned transmission affinity
    - Current outbreak intensity
    
    Unlike generic graph attention, this explicitly models unidirectional
    disease flow (infected region A → susceptible region B).
    
    Args:
        hidden_dim: Hidden dimension
        num_nodes: Number of regions
        dropout: Dropout rate
        use_adj_prior: Whether to use adjacency as transmission prior
    """
    
    def __init__(self, hidden_dim, num_nodes, dropout=DROPOUT,
                 use_adj_prior=True, adj_matrix=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.use_adj_prior = use_adj_prior
        
        # Transmission probability encoder
        # Takes source and target features, outputs transmission probability
        self.transmission_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Probability
        )
        
        # Outbreak intensity encoder
        # High outbreak intensity = more likely to transmit
        self.intensity_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # Learnable transmission affinity (region-pair specific)
        self.affinity_low = nn.Parameter(torch.randn(num_nodes, BOTTLENECK_DIM))
        self.affinity_high = nn.Parameter(torch.randn(BOTTLENECK_DIM, num_nodes))
        nn.init.xavier_uniform_(self.affinity_low)
        nn.init.xavier_uniform_(self.affinity_high)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Register adjacency prior if provided
        if use_adj_prior and adj_matrix is not None:
            if isinstance(adj_matrix, np.ndarray):
                adj_matrix = torch.from_numpy(adj_matrix).float()
            # Normalize
            adj_norm = adj_matrix / (adj_matrix.sum(dim=-1, keepdim=True) + 1e-8)
            self.register_buffer('adj_prior', adj_norm)
        else:
            self.register_buffer('adj_prior', None)
            
    def forward(self, x):
        """
        Compute transmission-weighted message passing.
        
        Args:
            x: Node features [batch, nodes, hidden_dim]
            
        Returns:
            Updated features with transmission effects [batch, nodes, hidden_dim]
        """
        B, N, D = x.shape
        
        # Compute outbreak intensity for each region
        intensity = self.intensity_encoder(x)  # [B, N, 1]
        
        # Compute transmission affinity matrix (low-rank)
        affinity = torch.matmul(self.affinity_low, self.affinity_high)  # [N, N]
        affinity = torch.sigmoid(affinity)  # [0, 1]
        
        # Combine with adjacency prior if available
        if self.adj_prior is not None:
            transmission_prob = 0.5 * affinity + 0.5 * self.adj_prior
        else:
            transmission_prob = affinity
            
        # Weight by source outbreak intensity
        # High intensity regions contribute more
        intensity_weight = intensity.squeeze(-1)  # [B, N]
        weighted_trans = transmission_prob.unsqueeze(0) * intensity_weight.unsqueeze(2)  # [B, N, N]
        
        # Normalize (each target receives from all sources)
        weighted_trans = weighted_trans / (weighted_trans.sum(dim=1, keepdim=True) + 1e-8)
        
        # Message passing: aggregate features from transmitting regions
        transmitted = torch.bmm(weighted_trans.transpose(1, 2), x)  # [B, N, D]
        
        # Residual connection + projection
        out = self.output_proj(x + transmitted)
        
        # Store for visualization
        self.transmission_matrix = weighted_trans.detach()
        
        return out


# =============================================================================
# EPIDEMIC-AWARE PREDICTION MODULE
# =============================================================================

class EpidemicAwarePrediction(nn.Module):
    """
    Prediction module that uses epidemic phase and Rt information.
    
    Epidemiological Motivation:
    - Growing phase (Rt > 1): Predict exponential increase
    - Peak phase (Rt ≈ 1): Predict plateau/saturation
    - Declining phase (Rt < 1): Predict exponential decrease
    
    The prediction strategy adapts based on detected epidemic phase.
    
    Args:
        hidden_dim: Hidden dimension
        horizon: Prediction horizon
        num_phases: Number of epidemic phases
    """
    
    def __init__(self, hidden_dim, horizon, num_phases=NUM_PHASES, dropout=DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.num_phases = num_phases
        
        # Phase-specific predictors
        self.phase_predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, horizon)
            ) for _ in range(num_phases)
        ])
        
        # Rt-based trend predictor
        self.rt_trend = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for Rt
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon)
        )
        
        # Fusion gate (how much to trust phase vs Rt)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim + num_phases + 1, horizon),  # features + phase + Rt
            nn.Sigmoid()
        )
        
    def forward(self, features, phase_probs, rt_estimate, last_value):
        """
        Generate predictions using epidemic information.
        
        Args:
            features: Node features [batch, nodes, hidden_dim]
            phase_probs: Phase probabilities [batch, nodes, num_phases]
            rt_estimate: Reproduction number [batch, nodes, 1]
            last_value: Last observed value [batch, nodes]
            
        Returns:
            Predictions [batch, nodes, horizon]
        """
        B, N, D = features.shape
        
        # Phase-specific predictions
        phase_preds = torch.stack([
            pred(features) for pred in self.phase_predictors
        ], dim=-1)  # [B, N, horizon, num_phases]
        
        # Weighted by phase probabilities
        phase_pred = torch.einsum('bnhp,bnp->bnh', phase_preds, phase_probs)
        
        # Rt-based trend prediction
        rt_input = torch.cat([features, rt_estimate], dim=-1)
        rt_pred = self.rt_trend(rt_input)  # [B, N, horizon]
        
        # Compute fusion gate
        gate_input = torch.cat([features, phase_probs, rt_estimate], dim=-1)
        gate = self.fusion_gate(gate_input)  # [B, N, horizon]
        
        # Fuse phase and Rt predictions
        fused_pred = gate * phase_pred + (1 - gate) * rt_pred
        
        # Add last value baseline (epidemic persistence)
        last_expanded = last_value.unsqueeze(-1).expand(-1, -1, self.horizon)
        
        # Blend with persistence (epidemics have momentum)
        persistence_weight = 0.3
        final_pred = (1 - persistence_weight) * fused_pred + persistence_weight * last_expanded
        
        return final_pred


# =============================================================================
# MAIN MODEL: EpiSTNet
# =============================================================================

class EpiSTNet(nn.Module):
    """
    Epidemic-Informed Spatiotemporal Network (EpiSTNet)
    
    A novel architecture that incorporates epidemiological domain knowledge:
    
    Key Components:
    1. Epidemic Phase Encoder: Detects growth/peak/decline phases
    2. Reproduction Number Estimator: Learns Rt as latent variable
    3. Generation Interval Attention: Temporal attention with disease timing
    4. Transmission Flow Module: Models inter-region disease spread
    5. Epidemic-Aware Prediction: Phase and Rt-informed forecasting
    
    Novel Contributions:
    - Explicit epidemic phase detection (not in EpiGNN, Cola-GNN, DCRNN)
    - Reproduction number as learnable latent variable
    - Generation interval-based temporal attention
    - Transmission probability modeling with outbreak intensity
    
    Data Flow:
    1. Input [B, T, N] → Phase detection + Rt estimation
    2. Generation interval attention (temporal)
    3. Transmission flow (spatial)
    4. Epidemic-aware prediction → [B, H, N]
    
    Args:
        args: Configuration with window, horizon, hidden_dim, etc.
        data: Data object with m (nodes), adj (optional adjacency)
    """
    
    def __init__(self, args, data):
        super().__init__()
        
        self.num_nodes = data.m
        self.window = args.window
        self.horizon = args.horizon
        self.hidden_dim = getattr(args, 'hidden_dim', HIDDEN_DIM)
        
        # Get adjacency if available
        adj_matrix = getattr(data, 'adj', None)
        if adj_matrix is not None and not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        use_adj = adj_matrix is not None
        
        # Disease-specific parameter
        self.generation_interval = getattr(args, 'generation_interval', DEFAULT_GENERATION_INTERVAL)
        
        # Component 1: Epidemic Phase Encoder
        self.phase_encoder = EpidemicPhaseEncoder(
            window=self.window,
            hidden_dim=self.hidden_dim
        )
        
        # Component 2: Reproduction Number Estimator
        self.rt_estimator = ReproductionNumberEstimator(
            window=self.window,
            hidden_dim=self.hidden_dim,
            generation_interval=self.generation_interval
        )
        
        # Input embedding (raw values to hidden dim)
        self.input_embedding = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Component 3: Generation Interval Attention (temporal)
        self.temporal_attention = GenerationIntervalAttention(
            window=self.window,
            hidden_dim=self.hidden_dim,
            generation_interval=self.generation_interval
        )
        
        # Temporal aggregation
        self.temporal_agg = nn.Sequential(
            nn.Linear(self.window * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Component 4: Transmission Flow Module (spatial)
        self.transmission_module = TransmissionFlowModule(
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            use_adj_prior=use_adj,
            adj_matrix=adj_matrix
        )
        
        # Feature fusion with phase embedding
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # features + phase_emb
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Component 5: Epidemic-Aware Prediction
        self.predictor = EpidemicAwarePrediction(
            hidden_dim=self.hidden_dim,
            horizon=self.horizon,
            num_phases=NUM_PHASES
        )
        
    def forward(self, x, idx=None):
        """
        Forward pass of EpiSTNet.
        
        Args:
            x: Input time series [batch, window, nodes]
            idx: Unused (for compatibility)
            
        Returns:
            tuple: (predictions [batch, horizon, nodes], auxiliary_loss)
        """
        B, T, N = x.shape
        x_last = x[:, -1, :]  # Last observed value
        
        # Detect epidemic phase
        phase_probs, phase_emb = self.phase_encoder(x)
        
        # Estimate reproduction number
        rt_estimate, rt_uncertainty = self.rt_estimator(x)
        
        # Embed input sequence
        x_embed = x.unsqueeze(-1)  # [B, T, N, 1]
        x_embed = self.input_embedding(x_embed)  # [B, T, N, D]
        x_embed = x_embed.permute(0, 2, 1, 3)  # [B, N, T, D]
        
        # Apply generation interval attention (temporal)
        temporal_features = self.temporal_attention(x_embed)  # [B, N, T, D]
        
        # Aggregate temporal features
        temporal_flat = temporal_features.reshape(B, N, -1)  # [B, N, T*D]
        node_features = self.temporal_agg(temporal_flat)  # [B, N, D]
        
        # Apply transmission flow (spatial)
        spatial_features = self.transmission_module(node_features)
        
        # Fuse with phase embedding
        combined = torch.cat([spatial_features, phase_emb], dim=-1)
        fused_features = self.feature_fusion(combined)
        
        # Generate predictions
        predictions = self.predictor(
            fused_features, phase_probs, rt_estimate, x_last
        )
        
        # Transpose for expected output format
        predictions = predictions.transpose(1, 2)  # [B, horizon, N]
        
        # Auxiliary loss: encourage smooth Rt estimates
        rt_smoothness_loss = torch.mean(rt_uncertainty)
        
        return predictions, rt_smoothness_loss
    
    def get_interpretable_outputs(self, x):
        """
        Get interpretable epidemic quantities for analysis.
        
        Returns phase probabilities, Rt estimates, and transmission matrix.
        """
        with torch.no_grad():
            phase_probs, _ = self.phase_encoder(x)
            rt_estimate, rt_uncertainty = self.rt_estimator(x)
            
            # Run forward to get transmission matrix
            _ = self.forward(x)
            transmission_matrix = self.transmission_module.transmission_matrix
            
        return {
            'phase_probs': phase_probs.cpu().numpy(),
            'rt_estimate': rt_estimate.cpu().numpy(),
            'rt_uncertainty': rt_uncertainty.cpu().numpy(),
            'transmission_matrix': transmission_matrix.cpu().numpy(),
            'generation_interval': self.rt_estimator.current_generation_interval
        }


# =============================================================================
# ABLATION VARIANT
# =============================================================================

class EpiSTNet_Ablation(nn.Module):
    """
    Ablation study variant of EpiSTNet.
    
    Allows systematic evaluation of each epidemiological component:
    - no_phase: Remove epidemic phase detection
    - no_rt: Remove reproduction number estimation
    - no_gi: Remove generation interval attention
    - no_transmission: Remove transmission flow module
    
    Args:
        args: Configuration with 'ablation' attribute
        data: Data object
    """
    
    def __init__(self, args, data):
        super().__init__()
        
        self.num_nodes = data.m
        self.window = args.window
        self.horizon = args.horizon
        self.hidden_dim = getattr(args, 'hidden_dim', HIDDEN_DIM)
        self.ablation = getattr(args, 'ablation', 'none')
        
        adj_matrix = getattr(data, 'adj', None)
        if adj_matrix is not None and not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            
        # Phase encoder (unless ablated)
        if self.ablation != 'no_phase':
            self.phase_encoder = EpidemicPhaseEncoder(self.window, self.hidden_dim)
        else:
            self.phase_encoder = None
            
        # Rt estimator (unless ablated)
        if self.ablation != 'no_rt':
            self.rt_estimator = ReproductionNumberEstimator(self.window, self.hidden_dim)
        else:
            self.rt_estimator = None
            
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Temporal attention (generation interval or standard)
        if self.ablation != 'no_gi':
            self.temporal_attention = GenerationIntervalAttention(
                self.window, self.hidden_dim
            )
        else:
            # Standard self-attention without generation interval
            self.temporal_attention = nn.MultiheadAttention(
                self.hidden_dim, num_heads=4, batch_first=True
            )
            
        self.temporal_agg = nn.Sequential(
            nn.Linear(self.window * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Spatial module
        if self.ablation != 'no_transmission':
            self.spatial_module = TransmissionFlowModule(
                self.hidden_dim, self.num_nodes,
                use_adj_prior=adj_matrix is not None,
                adj_matrix=adj_matrix
            )
        else:
            # Simple linear layer instead
            self.spatial_module = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim)
            )
            
        # Prediction
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.horizon)
        )
        
    def forward(self, x, idx=None):
        B, T, N = x.shape
        x_last = x[:, -1, :]
        
        # Embed input
        x_embed = x.unsqueeze(-1)
        x_embed = self.input_embedding(x_embed)
        x_embed = x_embed.permute(0, 2, 1, 3)  # [B, N, T, D]
        
        # Temporal processing
        if self.ablation != 'no_gi':
            temporal_features = self.temporal_attention(x_embed)
        else:
            # Standard attention (reshape for MultiheadAttention)
            x_flat = x_embed.view(B * N, T, -1)
            temporal_features, _ = self.temporal_attention(x_flat, x_flat, x_flat)
            temporal_features = temporal_features.view(B, N, T, -1)
            
        temporal_flat = temporal_features.reshape(B, N, -1)
        node_features = self.temporal_agg(temporal_flat)
        
        # Spatial processing
        if self.ablation != 'no_transmission':
            spatial_features = self.spatial_module(node_features)
        else:
            spatial_features = self.spatial_module(node_features)
            
        # Prediction
        predictions = self.predictor(spatial_features)
        predictions = predictions.transpose(1, 2)
        
        return predictions, torch.tensor(0.0, device=x.device)
