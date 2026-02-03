"""
EpiDelay-Net: Epidemic Delay-Aware Spatiotemporal Network (Lightweight Version)

A lightweight epidemic forecasting model that fills genuine gaps in the literature
by explicitly modeling epidemic propagation dynamics.

============================================================================
NOVEL CONTRIBUTIONS (Not in PISID, HeatGNN, EISTGNN, EpiGNN, Cola-GNN)
============================================================================

1. Serial Interval Graph: Explicit delay modeling in graph structure
2. Lead-Lag Attention: Cross-temporal regional relationships  
3. Reproduction Number Predictor: Rt as prediction driver
4. Wave Phase Encoder: Explicit epidemic phase detection

============================================================================
DESIGN PRINCIPLE: LIGHTWEIGHT ARCHITECTURE
============================================================================

- Similar parameter count to MSAGAT-Net (~30-50K parameters)
- Hidden dim: 32 (same as baseline)
- Bottleneck dim: 8 (same as baseline)
- No heavy transformers or complex attention
- Efficient implementations throughout

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# =============================================================================
# HYPERPARAMETERS (Matching MSAGAT-Net for fair comparison)
# =============================================================================
HIDDEN_DIM = 32
DROPOUT = 0.2
BOTTLENECK_DIM = 8
NUM_PHASES = 3  # Growth, Peak, Decline
DEFAULT_SERIAL_INTERVAL = 5
MAX_LAG = 5  # Reduced for efficiency


# =============================================================================
# COMPONENT 1: MOMENTUM ENCODER (Lightweight)
# =============================================================================

class MomentumEncoder(nn.Module):
    """
    Encodes epidemic momentum: velocity and acceleration of the epidemic curve.
    Lightweight version using simple projections.
    """
    
    def __init__(self, window, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.window = window
        self.hidden_dim = hidden_dim
        
        # Single projection combining all momentum features
        # Input: raw values (window) + velocity (window-1) + acceleration (window-2)
        total_features = window + (window - 1) + (window - 2)
        
        self.encoder = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input time series [batch, window, nodes]
            
        Returns:
            momentum_features: [batch, nodes, hidden_dim]
            velocity: [batch, nodes] - most recent velocity
            acceleration: [batch, nodes] - most recent acceleration
        """
        B, T, N = x.shape
        
        # Compute velocity and acceleration
        velocity_full = x[:, 1:, :] - x[:, :-1, :]  # [B, T-1, N]
        accel_full = velocity_full[:, 1:, :] - velocity_full[:, :-1, :]  # [B, T-2, N]
        
        # Get most recent values
        current_velocity = velocity_full[:, -1, :]  # [B, N]
        current_accel = accel_full[:, -1, :]  # [B, N]
        
        # Concatenate all features per node: [B, N, total_features]
        x_t = x.transpose(1, 2)  # [B, N, T]
        vel_t = velocity_full.transpose(1, 2)  # [B, N, T-1]
        acc_t = accel_full.transpose(1, 2)  # [B, N, T-2]
        
        combined = torch.cat([x_t, vel_t, acc_t], dim=-1)  # [B, N, total]
        
        # Encode
        momentum_features = self.encoder(combined)  # [B, N, D]
        
        return momentum_features, current_velocity, current_accel


# =============================================================================
# COMPONENT 2: WAVE PHASE ENCODER (Lightweight)
# =============================================================================

class WavePhaseEncoder(nn.Module):
    """
    Detects epidemic wave phase (growth/peak/decline) using velocity/acceleration.
    Lightweight version with minimal parameters.
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM, num_phases=NUM_PHASES):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_phases = num_phases
        
        # Simple phase classifier
        self.phase_classifier = nn.Sequential(
            nn.Linear(hidden_dim + 2, BOTTLENECK_DIM),  # +2 for vel, acc
            nn.ReLU(),
            nn.Linear(BOTTLENECK_DIM, num_phases)
        )
        
        # Learnable phase embeddings
        self.phase_embeddings = nn.Parameter(torch.randn(num_phases, hidden_dim) * 0.1)
        
    def forward(self, momentum_features, velocity, acceleration):
        """
        Args:
            momentum_features: [batch, nodes, hidden_dim]
            velocity: [batch, nodes]
            acceleration: [batch, nodes]
            
        Returns:
            phase_probs: [batch, nodes, num_phases]
            phase_embedding: [batch, nodes, hidden_dim]
        """
        B, N, D = momentum_features.shape
        
        # Normalize velocity and acceleration
        vel_norm = velocity / (velocity.abs().max(dim=-1, keepdim=True)[0] + 1e-8)
        acc_norm = acceleration / (acceleration.abs().max(dim=-1, keepdim=True)[0] + 1e-8)
        
        # Classify phase
        classifier_input = torch.cat([
            momentum_features, 
            vel_norm.unsqueeze(-1), 
            acc_norm.unsqueeze(-1)
        ], dim=-1)
        
        phase_logits = self.phase_classifier(classifier_input)
        phase_probs = F.softmax(phase_logits, dim=-1)
        
        # Weighted phase embedding
        phase_embedding = torch.einsum('bnp,pd->bnd', phase_probs, self.phase_embeddings)
        
        return phase_probs, phase_embedding


# =============================================================================
# COMPONENT 3: REPRODUCTION NUMBER PREDICTOR (Lightweight)
# =============================================================================

class ReproductionNumberPredictor(nn.Module):
    """
    Predicts Rt using ratio method refined by neural network.
    Lightweight version with minimal layers.
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM, serial_interval=DEFAULT_SERIAL_INTERVAL):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.serial_interval = serial_interval
        
        # Simple Rt refiner
        self.rt_refiner = nn.Sequential(
            nn.Linear(hidden_dim + 1, BOTTLENECK_DIM),
            nn.ReLU(),
            nn.Linear(BOTTLENECK_DIM, 1),
            nn.Softplus()  # Rt must be positive
        )
        
        # Rt embedding
        self.rt_embed = nn.Linear(1, hidden_dim)
        
    def forward(self, x, momentum_features):
        """
        Args:
            x: Raw time series [batch, window, nodes]
            momentum_features: [batch, nodes, hidden_dim]
            
        Returns:
            rt_estimate: [batch, nodes]
            rt_embedding: [batch, nodes, hidden_dim]
        """
        B, T, N = x.shape
        
        # Ratio-based Rt estimate
        current = x[:, -1, :]
        tau = min(self.serial_interval, T - 1)
        delayed = x[:, -1 - tau, :]
        ratio = torch.clamp(current / (delayed + 1e-8), 0.0, 10.0)
        
        # Refine
        refiner_input = torch.cat([momentum_features, ratio.unsqueeze(-1)], dim=-1)
        rt_estimate = self.rt_refiner(refiner_input).squeeze(-1)
        
        # Embedding
        rt_embedding = self.rt_embed(rt_estimate.unsqueeze(-1))
        
        return rt_estimate, rt_embedding


# =============================================================================
# COMPONENT 4: SERIAL INTERVAL GRAPH (Lightweight)
# =============================================================================

class SerialIntervalGraph(nn.Module):
    """
    Constructs delay-aware graph modeling epidemic propagation with time delays.
    Lightweight version using efficient operations.
    
    NOVEL: First model to explicitly incorporate propagation delays in graph structure.
    """
    
    def __init__(self, num_nodes, hidden_dim=HIDDEN_DIM, max_lag=MAX_LAG, adj_matrix=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag
        
        # Learnable delay weights (generation interval distribution)
        init_weights = torch.exp(-torch.arange(max_lag).float() / 2.0)
        self.delay_weights = nn.Parameter(init_weights / init_weights.sum())
        
        # Simple spatial affinity (low-rank)
        self.spatial_proj = nn.Linear(hidden_dim, BOTTLENECK_DIM)
        
        # Optional adjacency prior
        if adj_matrix is not None:
            if not isinstance(adj_matrix, torch.Tensor):
                adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
            adj_matrix = adj_matrix / (adj_matrix.sum(dim=-1, keepdim=True) + 1e-8)
            self.register_buffer('adj_prior', adj_matrix)
            self.adj_weight = nn.Parameter(torch.tensor(0.3))
        else:
            self.register_buffer('adj_prior', None)
            self.adj_weight = None
            
    def forward(self, x, features):
        """
        Args:
            x: Raw time series [batch, window, nodes]
            features: Node features [batch, nodes, hidden_dim]
            
        Returns:
            delay_features: [batch, nodes, hidden_dim] - aggregated delay features
        """
        B, T, N = x.shape
        
        # Compute spatial affinity (low-rank)
        proj = self.spatial_proj(features)  # [B, N, bottleneck]
        spatial_affinity = torch.bmm(proj, proj.transpose(1, 2))  # [B, N, N]
        spatial_affinity = spatial_affinity / math.sqrt(BOTTLENECK_DIM)
        
        # Add adjacency prior if available
        if self.adj_prior is not None:
            blend = torch.sigmoid(self.adj_weight)
            spatial_affinity = (1 - blend) * spatial_affinity + blend * self.adj_prior
        
        spatial_affinity = F.softmax(spatial_affinity, dim=-1)
        
        # Compute delay-weighted features (efficient version)
        delay_weights = F.softmax(self.delay_weights, dim=0)
        
        # Weighted sum of lagged infection values
        weighted_values = torch.zeros(B, N, device=x.device)
        for tau in range(min(self.max_lag, T - 1)):
            lagged = x[:, -(tau + 1), :]  # [B, N]
            weighted_values = weighted_values + delay_weights[tau] * lagged
        
        # Apply spatial affinity and aggregate features
        # [B, N, N] x [B, N] -> [B, N]
        propagated = torch.bmm(spatial_affinity, weighted_values.unsqueeze(-1)).squeeze(-1)
        
        # Combine with features
        delay_features = features + propagated.unsqueeze(-1) * 0.1  # Residual style
        
        return delay_features


# =============================================================================
# COMPONENT 5: LEAD-LAG ATTENTION (Lightweight)
# =============================================================================

class LeadLagAttention(nn.Module):
    """
    Models lead-lag relationships between regions using cross-correlation.
    Lightweight single-head attention version.
    
    NOVEL: First model to capture asynchronous regional dynamics via cross-correlation.
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM, max_lag=MAX_LAG):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag
        
        # Simple attention projections
        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def compute_max_cross_correlation(self, x):
        """Compute max cross-correlation across lags (efficient version)."""
        B, T, N = x.shape
        
        # Normalize
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-8
        x_norm = (x - x_mean) / x_std
        
        # Only compute for a few key lags for efficiency
        key_lags = [0, 1, 3]  # Current, 1-step, 3-step
        max_corr = torch.zeros(B, N, N, device=x.device)
        
        for lag in key_lags:
            if lag == 0:
                corr = torch.bmm(x_norm.transpose(1, 2), x_norm) / T
            elif lag < T:
                x_current = x_norm[:, lag:, :]
                x_lagged = x_norm[:, :-lag, :]
                corr = torch.bmm(x_current.transpose(1, 2), x_lagged) / (T - lag)
            else:
                continue
            max_corr = torch.maximum(max_corr, corr)
            
        return max_corr
        
    def forward(self, x, features):
        """
        Args:
            x: Raw time series [batch, window, nodes]
            features: Node features [batch, nodes, hidden_dim]
            
        Returns:
            attended_features: [batch, nodes, hidden_dim]
        """
        B, N, D = features.shape
        
        # Cross-correlation bias
        max_corr = self.compute_max_cross_correlation(x)  # [B, N, N]
        
        # QKV projection
        qkv = self.qkv_proj(features)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Attention with correlation bias
        attn_scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(D)
        attn_scores = attn_scores + max_corr * 0.5  # Add correlation bias
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention
        attended = torch.bmm(attn_weights, v)
        output = self.out_proj(attended)
        output = self.norm(output + features)  # Residual
        
        return output


# =============================================================================
# COMPONENT 6: RT-CONDITIONED PREDICTOR (Lightweight)
# =============================================================================

class RtConditionedPredictor(nn.Module):
    """
    Makes predictions conditioned on Rt using soft gating.
    Lightweight version with shared backbone and different heads.
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM, horizon=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # features + rt_emb
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        
        # Three lightweight prediction heads
        self.growth_head = nn.Linear(hidden_dim, horizon)
        self.equil_head = nn.Linear(hidden_dim, horizon)
        self.decline_head = nn.Linear(hidden_dim, horizon)
        
        # Rt-based gate
        self.rt_gate = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 3),
            nn.Softmax(dim=-1)
        )
        
        # Refinement gate
        self.refine_gate = nn.Sequential(
            nn.Linear(hidden_dim, horizon),
            nn.Sigmoid()
        )
        
    def forward(self, features, rt_embedding, rt_estimate, last_values):
        """
        Args:
            features: [batch, nodes, hidden_dim]
            rt_embedding: [batch, nodes, hidden_dim]
            rt_estimate: [batch, nodes]
            last_values: [batch, nodes]
            
        Returns:
            predictions: [batch, nodes, horizon]
        """
        B, N, D = features.shape
        
        # Combine features
        combined = torch.cat([features, rt_embedding], dim=-1)
        backbone_out = self.backbone(combined)
        
        # Get predictions from each head
        growth_pred = self.growth_head(backbone_out)
        equil_pred = self.equil_head(backbone_out)
        decline_pred = self.decline_head(backbone_out)
        
        # Stack and gate by Rt
        all_preds = torch.stack([growth_pred, equil_pred, decline_pred], dim=2)
        gate_weights = self.rt_gate(rt_estimate.unsqueeze(-1))
        weighted_pred = (all_preds * gate_weights.unsqueeze(-1)).sum(dim=2)
        
        # Refinement with last observation
        refine = self.refine_gate(backbone_out)
        last_expanded = last_values.unsqueeze(-1).expand(-1, -1, self.horizon)
        final_pred = refine * weighted_pred + (1 - refine) * last_expanded
        
        return final_pred


# =============================================================================
# MAIN MODEL: EpiDelay-Net (Lightweight)
# =============================================================================

class EpiDelayNet(nn.Module):
    """
    Epidemic Delay-Aware Spatiotemporal Network (EpiDelay-Net) - Lightweight Version
    
    A lightweight (~30-50K params) epidemic forecasting model with novel contributions:
    
    1. Serial Interval Graph: Explicit delay modeling
    2. Lead-Lag Attention: Cross-temporal regional relationships
    3. Rt Predictor: Reproduction number as prediction driver
    4. Wave Phase Encoder: Explicit phase detection
    
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
        self.serial_interval = getattr(args, 'serial_interval', DEFAULT_SERIAL_INTERVAL)
        
        # Get adjacency matrix
        adj_matrix = getattr(data, 'adj', None)
        if adj_matrix is not None and not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)
        
        # Components
        self.momentum_encoder = MomentumEncoder(self.window, self.hidden_dim)
        self.phase_encoder = WavePhaseEncoder(self.hidden_dim)
        self.rt_predictor = ReproductionNumberPredictor(self.hidden_dim, self.serial_interval)
        self.delay_graph = SerialIntervalGraph(self.num_nodes, self.hidden_dim, MAX_LAG, adj_matrix)
        self.lead_lag_attn = LeadLagAttention(self.hidden_dim, MAX_LAG)
        
        # Feature fusion (lightweight)
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        # Predictor
        self.predictor = RtConditionedPredictor(self.hidden_dim, self.horizon)
        
    def forward(self, x, idx=None):
        """
        Forward pass.
        
        Args:
            x: Input [batch, window, nodes]
            idx: Unused
            
        Returns:
            predictions: [batch, horizon, nodes]
            aux_loss: Small auxiliary loss
        """
        B, T, N = x.shape
        x_last = x[:, -1, :]
        
        # 1. Encode momentum
        momentum_features, velocity, acceleration = self.momentum_encoder(x)
        
        # 2. Detect phase
        phase_probs, phase_embedding = self.phase_encoder(momentum_features, velocity, acceleration)
        
        # 3. Predict Rt
        rt_estimate, rt_embedding = self.rt_predictor(x, momentum_features)
        
        # 4. Apply serial interval graph
        delay_features = self.delay_graph(x, momentum_features)
        
        # 5. Apply lead-lag attention
        attended_features = self.lead_lag_attn(x, delay_features)
        
        # 6. Fuse features
        combined = torch.cat([attended_features, phase_embedding], dim=-1)
        fused = self.fusion(combined)
        
        # 7. Predict with Rt conditioning
        predictions = self.predictor(fused, rt_embedding, rt_estimate, x_last)
        predictions = predictions.transpose(1, 2)  # [B, H, N]
        
        # Auxiliary loss (encourage diverse phases)
        phase_entropy = -torch.sum(phase_probs * torch.log(phase_probs + 1e-8), dim=-1)
        aux_loss = 0.01 * torch.mean(phase_entropy)
        
        return predictions, aux_loss
        
    def get_interpretable_outputs(self, x):
        """Get interpretable epidemiological outputs."""
        B, T, N = x.shape
        
        momentum_features, velocity, acceleration = self.momentum_encoder(x)
        phase_probs, _ = self.phase_encoder(momentum_features, velocity, acceleration)
        rt_estimate, _ = self.rt_predictor(x, momentum_features)
        
        delay_weights = F.softmax(self.delay_graph.delay_weights, dim=0)
        
        return {
            'rt_estimate': rt_estimate.detach(),
            'phase_probs': phase_probs.detach(),
            'delay_weights': delay_weights.detach(),
            'velocity': velocity.detach(),
            'acceleration': acceleration.detach()
        }


# =============================================================================
# ABLATION VARIANT
# =============================================================================

class EpiDelayNet_Ablation(nn.Module):
    """
    EpiDelay-Net with configurable components for ablation studies.
    
    Ablation Options:
        - 'none': Full model
        - 'no_delay': Remove Serial Interval Graph
        - 'no_leadlag': Remove Lead-Lag Attention  
        - 'no_rt': Remove Rt conditioning
        - 'no_phase': Remove Wave Phase Encoder
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
        
        # Always use momentum encoder
        self.momentum_encoder = MomentumEncoder(self.window, self.hidden_dim)
        
        # Conditional components
        self.use_phase = self.ablation != 'no_phase'
        self.use_rt = self.ablation != 'no_rt'
        self.use_delay = self.ablation != 'no_delay'
        self.use_leadlag = self.ablation != 'no_leadlag'
        
        if self.use_phase:
            self.phase_encoder = WavePhaseEncoder(self.hidden_dim)
        if self.use_rt:
            self.rt_predictor = ReproductionNumberPredictor(self.hidden_dim)
        if self.use_delay:
            self.delay_graph = SerialIntervalGraph(self.num_nodes, self.hidden_dim, MAX_LAG, adj_matrix)
        if self.use_leadlag:
            self.lead_lag_attn = LeadLagAttention(self.hidden_dim)
        
        # Fusion and prediction
        fusion_dim = self.hidden_dim
        if self.use_phase:
            fusion_dim += self.hidden_dim
            
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )
        
        if self.use_rt:
            self.predictor = RtConditionedPredictor(self.hidden_dim, self.horizon)
        else:
            self.predictor = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
                nn.Linear(self.hidden_dim, self.horizon)
            )
            
    def forward(self, x, idx=None):
        B, T, N = x.shape
        x_last = x[:, -1, :]
        
        momentum_features, velocity, acceleration = self.momentum_encoder(x)
        features = momentum_features
        
        if self.use_delay:
            features = self.delay_graph(x, features)
        if self.use_leadlag:
            features = self.lead_lag_attn(x, features)
            
        if self.use_phase:
            _, phase_embedding = self.phase_encoder(momentum_features, velocity, acceleration)
            features = torch.cat([features, phase_embedding], dim=-1)
            
        fused = self.fusion(features)
        
        if self.use_rt:
            rt_estimate, rt_embedding = self.rt_predictor(x, momentum_features)
            predictions = self.predictor(fused, rt_embedding, rt_estimate, x_last)
        else:
            predictions = self.predictor(fused)
            
        predictions = predictions.transpose(1, 2) if predictions.dim() == 3 else predictions.transpose(1, 2)
        
        return predictions, torch.tensor(0.0, device=x.device)
