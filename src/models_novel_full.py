"""
EpiDelay-Net Full: Epidemic Delay-Aware Spatiotemporal Network (Full Version)

The complete version with all components at full capacity:
- Separate projections for value, velocity, acceleration
- Multi-head lead-lag attention with lag embeddings
- Rt uncertainty estimation
- Phase transition prediction
- Richer feature representations

"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# =============================================================================
# HYPERPARAMETERS (Full version - larger capacity)
# =============================================================================
HIDDEN_DIM = 32
DROPOUT = 0.2
BOTTLENECK_DIM = 16  # Larger bottleneck
NUM_PHASES = 3
DEFAULT_SERIAL_INTERVAL = 5
MAX_LAG = 7  # More lags
NUM_HEADS = 4  # Multi-head attention


# =============================================================================
# COMPONENT 1: MOMENTUM ENCODER (Full)
# =============================================================================

class MomentumEncoder(nn.Module):
    """
    Full momentum encoder with separate projections for each derivative.
    """
    
    def __init__(self, window, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.window = window
        self.hidden_dim = hidden_dim
        
        # Separate projections for each component
        self.value_proj = nn.Sequential(
            nn.Linear(window, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        
        self.velocity_proj = nn.Sequential(
            nn.Linear(window - 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        
        self.accel_proj = nn.Sequential(
            nn.Linear(window - 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT)
        )
        
        # Multi-layer fusion
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x):
        B, T, N = x.shape
        
        # Transpose for per-node processing
        x_t = x.transpose(1, 2)  # [B, N, T]
        
        # Compute derivatives
        velocity_full = x[:, 1:, :] - x[:, :-1, :]
        velocity_t = velocity_full.transpose(1, 2)
        
        accel_full = velocity_full[:, 1:, :] - velocity_full[:, :-1, :]
        accel_t = accel_full.transpose(1, 2)
        
        # Current values for phase detection
        current_velocity = velocity_full[:, -1, :]
        current_accel = accel_full[:, -1, :]
        
        # Separate projections
        value_emb = self.value_proj(x_t)
        velocity_emb = self.velocity_proj(velocity_t)
        accel_emb = self.accel_proj(accel_t)
        
        # Fuse
        combined = torch.cat([value_emb, velocity_emb, accel_emb], dim=-1)
        momentum_features = self.fusion(combined)
        
        return momentum_features, current_velocity, current_accel


# =============================================================================
# COMPONENT 2: WAVE PHASE ENCODER (Full)
# =============================================================================

class WavePhaseEncoder(nn.Module):
    """
    Full wave phase encoder with transition prediction.
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM, num_phases=NUM_PHASES):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_phases = num_phases
        
        # Deeper phase classifier
        self.phase_classifier = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_phases)
        )
        
        # Learnable phase embeddings
        self.phase_embeddings = nn.Parameter(torch.randn(num_phases, hidden_dim) * 0.1)
        
        # Phase transition predictor
        self.transition_predictor = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, momentum_features, velocity, acceleration):
        B, N, D = momentum_features.shape
        
        # Normalize
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
        
        # Predict transition probability
        transition_prob = self.transition_predictor(classifier_input).squeeze(-1)
        
        return phase_probs, phase_embedding, transition_prob


# =============================================================================
# COMPONENT 3: REPRODUCTION NUMBER PREDICTOR (Full)
# =============================================================================

class ReproductionNumberPredictor(nn.Module):
    """
    Full Rt predictor with uncertainty estimation.
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM, serial_interval=DEFAULT_SERIAL_INTERVAL):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.serial_interval = serial_interval
        
        # Deeper Rt refiner
        self.rt_refiner = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # Uncertainty estimator
        self.rt_uncertainty = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()
        )
        
        # Richer Rt embedding
        self.rt_embed = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),  # Rt + uncertainty
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
    def forward(self, x, momentum_features):
        B, T, N = x.shape
        
        # Ratio-based Rt estimate
        current = x[:, -1, :]
        tau = min(self.serial_interval, T - 1)
        delayed = x[:, -1 - tau, :]
        ratio = torch.clamp(current / (delayed + 1e-8), 0.0, 10.0)
        
        # Refine
        refiner_input = torch.cat([momentum_features, ratio.unsqueeze(-1)], dim=-1)
        rt_estimate = self.rt_refiner(refiner_input).squeeze(-1)
        
        # Uncertainty
        rt_unc = self.rt_uncertainty(refiner_input).squeeze(-1)
        
        # Embedding (includes uncertainty)
        rt_with_unc = torch.stack([rt_estimate, rt_unc], dim=-1)
        rt_embedding = self.rt_embed(rt_with_unc)
        
        return rt_estimate, rt_unc, rt_embedding


# =============================================================================
# COMPONENT 4: SERIAL INTERVAL GRAPH (Full)
# =============================================================================

class SerialIntervalGraph(nn.Module):
    """
    Full serial interval graph with separate key/query projections.
    """
    
    def __init__(self, num_nodes, hidden_dim=HIDDEN_DIM, max_lag=MAX_LAG, adj_matrix=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.max_lag = max_lag
        
        # Learnable delay weights
        init_weights = torch.exp(-torch.arange(max_lag).float() / 2.0)
        self.delay_weights = nn.Parameter(init_weights / init_weights.sum())
        
        # Separate key/query projections
        self.spatial_key = nn.Linear(hidden_dim, BOTTLENECK_DIM)
        self.spatial_query = nn.Linear(hidden_dim, BOTTLENECK_DIM)
        
        # Value projection
        self.spatial_value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Adjacency prior
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
        B, T, N = x.shape
        
        # Compute spatial affinity
        keys = self.spatial_key(features)
        queries = self.spatial_query(features)
        values = self.spatial_value(features)
        
        spatial_affinity = torch.bmm(queries, keys.transpose(1, 2))
        spatial_affinity = spatial_affinity / math.sqrt(BOTTLENECK_DIM)
        
        # Add adjacency prior
        if self.adj_prior is not None:
            blend = torch.sigmoid(self.adj_weight)
            spatial_affinity = (1 - blend) * spatial_affinity + blend * self.adj_prior
        
        spatial_affinity = F.softmax(spatial_affinity, dim=-1)
        
        # Delay-weighted influence
        delay_weights = F.softmax(self.delay_weights, dim=0)
        
        weighted_values = torch.zeros(B, N, device=x.device)
        for tau in range(min(self.max_lag, T - 1)):
            lagged = x[:, -(tau + 1), :]
            weighted_values = weighted_values + delay_weights[tau] * lagged
        
        # Propagate
        propagated = torch.bmm(spatial_affinity, values)
        
        # Add delay signal
        delay_signal = weighted_values.unsqueeze(-1) * 0.1
        output = propagated + delay_signal
        
        return self.out_proj(output)


# =============================================================================
# COMPONENT 5: LEAD-LAG ATTENTION (Full - Multi-head)
# =============================================================================

class LeadLagAttention(nn.Module):
    """
    Full multi-head lead-lag attention with lag embeddings.
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM, num_heads=NUM_HEADS, max_lag=MAX_LAG):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.max_lag = max_lag
        
        # Multi-head projections
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Lag embeddings
        self.lag_embedding = nn.Parameter(torch.randn(max_lag, hidden_dim) * 0.1)
        
        # Lag scorer
        self.lag_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_lag)
        )
        
        # Output
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(DROPOUT)
        
    def compute_cross_correlation(self, x):
        B, T, N = x.shape
        
        # Normalize
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True) + 1e-8
        x_norm = (x - x_mean) / x_std
        
        cross_corr = []
        for lag in range(self.max_lag):
            if lag == 0:
                corr = torch.bmm(x_norm.transpose(1, 2), x_norm) / T
            elif lag < T:
                x_current = x_norm[:, lag:, :]
                x_lagged = x_norm[:, :-lag, :]
                corr = torch.bmm(x_current.transpose(1, 2), x_lagged) / (T - lag)
            else:
                corr = torch.zeros(B, N, N, device=x.device)
            cross_corr.append(corr)
            
        return torch.stack(cross_corr, dim=-1)  # [B, N, N, max_lag]
        
    def forward(self, x, features):
        B, N, D = features.shape
        
        # Cross-correlation
        cross_corr = self.compute_cross_correlation(x)
        
        # Find optimal lag per region pair
        max_corr, optimal_lag = cross_corr.max(dim=-1)  # [B, N, N]
        
        # Multi-head attention
        Q = self.query_proj(features).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(features).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(features).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add cross-correlation bias
        attn_scores = attn_scores + max_corr.unsqueeze(1) * 0.5
        
        # Softmax and apply
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attended = torch.matmul(attn_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(B, N, D)
        
        output = self.out_proj(attended)
        output = self.norm(output + features)
        
        return output, attn_weights.mean(dim=1)


# =============================================================================
# COMPONENT 6: RT-CONDITIONED PREDICTOR (Full)
# =============================================================================

class RtConditionedPredictor(nn.Module):
    """
    Full Rt-conditioned predictor with separate deep heads.
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM, horizon=7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        
        # Three separate deep prediction heads
        self.growth_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, horizon)
        )
        
        self.equilibrium_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, horizon)
        )
        
        self.decline_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, horizon)
        )
        
        # Rt-based gate
        self.rt_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softmax(dim=-1)
        )
        
        # Refinement gate
        self.refine_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon),
            nn.Sigmoid()
        )
        
    def forward(self, features, rt_embedding, rt_estimate, last_values):
        B, N, D = features.shape
        
        # Combine features
        combined = torch.cat([features, rt_embedding], dim=-1)
        
        # Get predictions from each head
        growth_pred = self.growth_head(combined)
        equil_pred = self.equilibrium_head(combined)
        decline_pred = self.decline_head(combined)
        
        # Stack and gate
        all_preds = torch.stack([growth_pred, equil_pred, decline_pred], dim=2)
        gate_weights = self.rt_gate(rt_estimate.unsqueeze(-1))
        weighted_pred = (all_preds * gate_weights.unsqueeze(-1)).sum(dim=2)
        
        # Refinement
        refine = self.refine_gate(combined)
        last_expanded = last_values.unsqueeze(-1).expand(-1, -1, self.horizon)
        final_pred = refine * weighted_pred + (1 - refine) * last_expanded
        
        return final_pred


# =============================================================================
# MAIN MODEL: EpiDelay-Net Full
# =============================================================================

class EpiDelayNetFull(nn.Module):
    """
    Full EpiDelay-Net with all components at maximum capacity.
    
    Compared to lightweight version:
    - Separate projections for value/velocity/acceleration
    - Multi-head lead-lag attention
    - Rt uncertainty estimation
    - Phase transition prediction
    - Deeper prediction heads
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
        
        # Components (Full versions)
        self.momentum_encoder = MomentumEncoder(self.window, self.hidden_dim)
        self.phase_encoder = WavePhaseEncoder(self.hidden_dim)
        self.rt_predictor = ReproductionNumberPredictor(self.hidden_dim, self.serial_interval)
        self.delay_graph = SerialIntervalGraph(self.num_nodes, self.hidden_dim, MAX_LAG, adj_matrix)
        self.lead_lag_attn = LeadLagAttention(self.hidden_dim, NUM_HEADS, MAX_LAG)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Predictor
        self.predictor = RtConditionedPredictor(self.hidden_dim, self.horizon)
        
    def forward(self, x, idx=None):
        B, T, N = x.shape
        x_last = x[:, -1, :]
        
        # 1. Encode momentum
        momentum_features, velocity, acceleration = self.momentum_encoder(x)
        
        # 2. Detect phase
        phase_probs, phase_embedding, transition_prob = self.phase_encoder(
            momentum_features, velocity, acceleration
        )
        
        # 3. Predict Rt with uncertainty
        rt_estimate, rt_uncertainty, rt_embedding = self.rt_predictor(x, momentum_features)
        
        # 4. Apply serial interval graph
        delay_features = self.delay_graph(x, momentum_features)
        
        # 5. Apply lead-lag attention
        attended_features, lead_lag_matrix = self.lead_lag_attn(x, delay_features)
        
        # 6. Fuse features
        combined = torch.cat([attended_features, phase_embedding, rt_embedding], dim=-1)
        fused = self.fusion(combined)
        
        # 7. Predict with Rt conditioning
        predictions = self.predictor(fused, rt_embedding, rt_estimate, x_last)
        predictions = predictions.transpose(1, 2)
        
        # Auxiliary loss
        phase_entropy = -torch.sum(phase_probs * torch.log(phase_probs + 1e-8), dim=-1)
        aux_loss = 0.01 * torch.mean(phase_entropy) + 0.01 * torch.mean(rt_uncertainty)
        
        return predictions, aux_loss
        
    def get_interpretable_outputs(self, x):
        B, T, N = x.shape
        
        momentum_features, velocity, acceleration = self.momentum_encoder(x)
        phase_probs, _, transition_prob = self.phase_encoder(momentum_features, velocity, acceleration)
        rt_estimate, rt_uncertainty, _ = self.rt_predictor(x, momentum_features)
        
        delay_weights = F.softmax(self.delay_graph.delay_weights, dim=0)
        
        return {
            'rt_estimate': rt_estimate.detach(),
            'rt_uncertainty': rt_uncertainty.detach(),
            'phase_probs': phase_probs.detach(),
            'transition_prob': transition_prob.detach(),
            'delay_weights': delay_weights.detach(),
            'velocity': velocity.detach(),
            'acceleration': acceleration.detach()
        }
