import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# Model hyperparameters with descriptive names
HIDDEN_DIM = 32
ATTENTION_HEADS = 4
# Initial value for adaptive attention regularization weight
ATTENTION_REG_WEIGHT_INIT = 1e-05
DROPOUT = 0.25
NUM_TEMPORAL_SCALES = 4
KERNEL_SIZE = 3
FEATURE_CHANNELS = 16
BOTTLENECK_DIM = 8

class SpatialAttentionModule(nn.Module):
    """
    Low-rank Adaptive Graph Attention Module (LR-AGAM).
    
    Captures node relationships in a graph structure using an efficient
    attention mechanism with low-rank decomposition. Computes attention 
    scores between nodes and updates node representations accordingly.
    
    Args:
        hidden_dim (int): Dimensionality of node features
        num_nodes (int): Number of nodes in the graph
        dropout (float): Dropout probability for regularization
        attention_heads (int): Number of parallel attention heads
        attention_regularization_weight (float): Weight for L1 regularization on attention
        bottleneck_dim (int): Dimension of the low-rank projection
        use_adj_prior (bool): Whether to use adjacency matrix as prior for attention
        use_graph_bias (bool): Whether to use learnable graph structure bias
        adj_matrix (tensor, optional): Predefined adjacency matrix [num_nodes, num_nodes]
    """
    def __init__(self, hidden_dim, num_nodes, dropout=DROPOUT, attention_heads=ATTENTION_HEADS, 
                 attention_regularization_weight=ATTENTION_REG_WEIGHT_INIT, bottleneck_dim=BOTTLENECK_DIM,
                 use_adj_prior=False, use_graph_bias=True, adj_matrix=None):
        super(SpatialAttentionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = attention_heads
        self.head_dim = hidden_dim // self.heads
        self.num_nodes = num_nodes
        self.bottleneck_dim = bottleneck_dim
        self.use_adj_prior = use_adj_prior
        self.use_graph_bias = use_graph_bias

        # Make attention regularization weight a learnable parameter (log-domain for positivity)
        self.log_attention_reg_weight = nn.Parameter(torch.tensor(math.log(attention_regularization_weight), dtype=torch.float32))

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
            # Normalize adjacency matrix and expand for heads
            adj_norm = adj_matrix / (adj_matrix.sum(dim=-1, keepdim=True) + 1e-8)
            self.register_buffer('adj_prior', adj_norm.unsqueeze(0).expand(self.heads, -1, -1).clone())
        else:
            self.register_buffer('adj_prior', None)

    def _compute_attention(self, q, k, v):
        """
        Compute attention scores and apply them to values.
        
        Args:
            q (tensor): Query tensors [batch, heads, nodes, head_dim]
            k (tensor): Key tensors [batch, heads, nodes, head_dim]
            v (tensor): Value tensors [batch, heads, nodes, head_dim]
            
        Returns:
            tensor: Attended values
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
            x (tensor): Input node features [batch, nodes, hidden_dim]
            mask (tensor, optional): Attention mask
            
        Returns:
            tuple: (Updated node features, Attention regularization loss)
        """
        B, N, H = x.shape
        
        # Low-rank projection for qkv
        qkv_low = self.qkv_proj_low(x)
        qkv = self.qkv_proj_high(qkv_low)
        qkv = qkv.chunk(3, dim=-1)
        
        # Separate query, key, value and reshape for multi-head attention
        q, k, v = [x.view(B, N, self.heads, self.head_dim) for x in qkv]
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


class MultiScaleTemporalModule(nn.Module):
    """
    Multi-Scale Spatial Feature Module that captures spatial patterns at different scales.
    
    Despite the class name, this module operates on the NODE (spatial) dimension, not time.
    It uses dilated convolutions at different dilation rates to capture local and global
    spatial dependencies between nodes. The outputs from different scales are adaptively fused.
    
    Args:
        hidden_dim (int): Dimensionality of node features
        num_scales (int): Number of spatial scales (different dilation rates)
        kernel_size (int): Size of the convolutional kernel
        dropout (float): Dropout probability for regularization
    """
    def __init__(self, hidden_dim, num_scales=NUM_TEMPORAL_SCALES, kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super(MultiScaleTemporalModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        
        # Create multiple dilated convolutional layers with increasing dilation rates
        self.scales = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                          padding=(kernel_size//2) * 2**i, dilation=2**i),
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
        Forward pass of the multi-scale spatial feature module.
        
        Args:
            x (tensor): Input features [batch, nodes, hidden_dim]
            
        Returns:
            tensor: Spatially processed features with multi-scale information
        """
        # Reshape for 1D convolution [batch, hidden_dim, nodes]
        x = x.transpose(1, 2)  # [batch, hidden_dim, nodes]
        
        # Apply different spatial scales (dilated convolutions along node dimension)
        features = [scale(x) for scale in self.scales]
        
        # Compute adaptive weights for scale fusion
        alpha = F.softmax(self.fusion_weight, dim=0)
        
        # Stack and fuse multi-scale features
        stacked = torch.stack(features, dim=1)  # [batch, scales, hidden_dim, nodes]
        fused = torch.sum(alpha.view(1, self.num_scales, 1, 1) * stacked, dim=1)
        
        # Reshape back
        fused = fused.transpose(1, 2)  # [batch, nodes, hidden_dim]
        
        # Apply low-rank projection and residual connection
        out = self.fusion_low(fused)
        out = self.fusion_high(out)
        out = self.layer_norm(out + fused)
        
        return out


class HorizonPredictor(nn.Module):
    """
    Progressive Prediction Refinement Module (PPRM) with GRU.
    
    Uses a GRU to autoregressively generate predictions for multiple future time steps.
    The GRU learns temporal dynamics for the prediction horizon, with each step
    conditioned on the previous prediction. Includes adaptive refinement that
    blends GRU outputs with learned decay from the last observation.
    
    Args:
        hidden_dim (int): Dimensionality of node features
        horizon (int): Number of future time steps to predict
        bottleneck_dim (int): Dimension for bottleneck layers
        dropout (float): Dropout probability for regularization
    """
    def __init__(self, hidden_dim, horizon, bottleneck_dim=BOTTLENECK_DIM, dropout=DROPOUT):
        super(HorizonPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.bottleneck_dim = bottleneck_dim
        
        # GRU for autoregressive prediction
        self.gru = nn.GRU(
            input_size=1,  # Previous prediction value
            hidden_size=bottleneck_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0  # Single layer, no dropout needed
        )
        
        # Project node features to GRU hidden state
        self.hidden_proj = nn.Linear(hidden_dim, bottleneck_dim)
        
        # Output projection from GRU hidden to prediction
        self.output_proj = nn.Linear(bottleneck_dim, 1)
        
        # Learnable decay rate (replaces fixed 0.1)
        self.log_decay = nn.Parameter(torch.tensor(-2.3))  # ~0.1 initial
        
        # Adaptive refinement gate
        self.refine_gate = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, horizon),
            nn.Sigmoid()
        )

    def forward(self, x, last_step=None):
        """
        Forward pass of the GRU-based horizon predictor.
        
        Args:
            x (tensor): Node features [batch, nodes, hidden_dim]
            last_step (tensor, optional): Last observed values [batch, nodes]
            
        Returns:
            tensor: Predictions for future time steps [batch, nodes, horizon]
        """
        B, N, D = x.shape
        
        # Initialize GRU hidden state from node features
        # [batch, nodes, hidden_dim] -> [batch*nodes, bottleneck_dim]
        h0 = self.hidden_proj(x.view(B * N, D))
        h0 = h0.unsqueeze(0)  # [1, batch*nodes, bottleneck_dim]
        
        # Start with last observation as first input
        if last_step is not None:
            current_input = last_step.reshape(B * N, 1, 1)  # [batch*nodes, 1, 1]
        else:
            current_input = torch.zeros(B * N, 1, 1, device=x.device)
        
        # Autoregressive prediction
        predictions = []
        hidden = h0
        for t in range(self.horizon):
            gru_out, hidden = self.gru(current_input, hidden)
            pred_t = self.output_proj(gru_out.squeeze(1))  # [batch*nodes, 1]
            predictions.append(pred_t)
            current_input = pred_t.unsqueeze(1)  # [batch*nodes, 1, 1]
        
        # Stack predictions: [batch*nodes, horizon]
        gru_pred = torch.cat(predictions, dim=-1)
        gru_pred = gru_pred.view(B, N, self.horizon)
        
        # Apply adaptive refinement if last observation provided
        if last_step is not None:
            gate = self.refine_gate(x)  # [batch, nodes, horizon]
            
            # Learned decay extrapolation
            decay_rate = torch.exp(self.log_decay)
            time_steps = torch.arange(1, self.horizon + 1, device=x.device).float()
            decay_curve = torch.exp(-decay_rate * time_steps).view(1, 1, self.horizon)
            decay_pred = last_step.unsqueeze(-1) * decay_curve
            
            # Blend GRU prediction with decay extrapolation
            final_pred = gate * gru_pred + (1 - gate) * decay_pred
        else:
            final_pred = gru_pred
            
        return final_pred


class DepthwiseSeparableConv1D(nn.Module):
    """
    Depthwise Separable 1D Convolution for Temporal Feature Extraction Module (TFEM).
    
    This module splits a standard convolution into a depthwise convolution
    (applied to each channel separately) followed by a pointwise convolution
    (1x1 convolution across channels). Used to process the input time window
    and extract temporal features efficiently.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        stride (int): Convolution stride
        padding (int): Padding size
        dilation (int): Dilation rate
        dropout (float): Dropout probability
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, dropout=DROPOUT):
        super(DepthwiseSeparableConv1D, self).__init__()
        
        # Depthwise convolution (per-channel)
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dilation=dilation, groups=in_channels)
        self.bn1 = nn.BatchNorm1d(in_channels)
        
        # Pointwise convolution (1x1 conv across channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the depthwise separable convolution.
        
        Args:
            x (tensor): Input features [batch, channels, time]
            
        Returns:
            tensor: Convolved features [batch, out_channels, time]
        """
        # Depthwise convolution
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        
        # Pointwise convolution
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act(x)
        x = self.dropout(x)
        
        return x


class MSTAGAT_Net(nn.Module):
    """
    Multi-Scale Temporal-Adaptive Graph Attention Network (MSTAGAT-Net)
    
    A spatiotemporal forecasting model with three key components:
    - TFEM: Temporal Feature Extraction Module (depthwise separable convolutions along time)
    - LR-AGAM: Low-rank Adaptive Graph Attention Module (spatial attention across nodes)
    - PPRM: Progressive Prediction Refinement Module (GRU-based autoregressive prediction)
    
    Data flow:
    1. Input [B, T, N] → TFEM extracts temporal features → [B, N, D]
    2. LR-AGAM computes attention across nodes → [B, N, D]
    3. PPRM generates predictions autoregressively → [B, H, N]
    
    Key components:
    - Temporal feature extraction using depthwise separable convolutions along TIME
    - Spatial modeling with graph attention across NODES
    - Autoregressive prediction with GRU and adaptive refinement
    
    Args:
        args: Model configuration arguments
        data: Data object containing dataset information
    """
    def __init__(self, args, data):
        super(MSTAGAT_Net, self).__init__()
        self.num_nodes = data.m
        self.window = args.window
        self.horizon = args.horizon
        self.hidden_dim = getattr(args, 'hidden_dim', HIDDEN_DIM)
        self.kernel_size = getattr(args, 'kernel_size', KERNEL_SIZE)
        self.bottleneck_dim = getattr(args, 'bottleneck_dim', BOTTLENECK_DIM)
        
        # Graph structure options
        self.use_adj_prior = getattr(args, 'use_adj_prior', False)
        self.use_graph_bias = getattr(args, 'use_graph_bias', True)
        
        # Get adjacency matrix if available
        adj_matrix = getattr(data, 'adj', None)
        if adj_matrix is not None and not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

        # Feature Extraction Component
        # ----------------------------
        # Extract features from raw time series using depthwise separable convolutions
        self.feature_channels = getattr(args, 'feature_channels', FEATURE_CHANNELS)
        self.feature_extractor = DepthwiseSeparableConv1D(
            in_channels=1, 
            out_channels=self.feature_channels,
            kernel_size=self.kernel_size, 
            padding=self.kernel_size // 2,
            dropout=getattr(args, 'dropout', DROPOUT)
        )

        # Low-rank projection of extracted features
        self.feature_projection_low = nn.Linear(self.feature_channels * self.window, self.bottleneck_dim)
        self.feature_projection_high = nn.Linear(self.bottleneck_dim, self.hidden_dim)
        self.feature_norm = nn.LayerNorm(self.hidden_dim)
        self.feature_act = nn.ReLU()

        # Spatial Component (LR-AGAM)
        # -----------------------------
        # Graph attention mechanism to capture spatial dependencies between nodes
        self.spatial_module = SpatialAttentionModule(
            self.hidden_dim, num_nodes=self.num_nodes,
            dropout=getattr(args, 'dropout', DROPOUT),
            attention_heads=getattr(args, 'attention_heads', ATTENTION_HEADS),
            bottleneck_dim=self.bottleneck_dim,
            use_adj_prior=self.use_adj_prior,
            use_graph_bias=self.use_graph_bias,
            adj_matrix=adj_matrix
        )

        # Prediction Component (PPRM)
        # ---------------------------
        # GRU-based horizon predictor with adaptive refinement
        self.prediction_module = HorizonPredictor(
            self.hidden_dim, self.horizon,
            bottleneck_dim=self.bottleneck_dim,
            dropout=getattr(args, 'dropout', DROPOUT)
        )

    def forward(self, x, idx=None):
        """
        Forward pass of the MSTAGAT-Net model.
        
        Args:
            x (tensor): Input time series [batch, time_window, nodes]
            idx (tensor, optional): Node indices
            
        Returns:
            tuple: (Predictions, Attention regularization loss)
        """
        B, T, N = x.shape
        x_last = x[:, -1, :]  # Last observed values
        
        # Feature Extraction (TFEM)
        # -------------------------
        # Reshape for 1D convolution [batch*nodes, 1, time_window]
        x_temp = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)
        temporal_features = self.feature_extractor(x_temp)
        temporal_features = temporal_features.view(B, N, -1)
        
        # Feature projection through bottleneck
        features = self.feature_projection_low(temporal_features)
        features = self.feature_projection_high(features)
        features = self.feature_norm(features)
        features = self.feature_act(features)
        
        # Spatial Processing (LR-AGAM)
        # ----------------------------
        # Apply graph attention to capture spatial dependencies
        graph_features, attn_reg_loss = self.spatial_module(features)
        
        # Prediction (PPRM)
        # -----------------
        # Generate predictions using GRU-based autoregressive predictor
        predictions = self.prediction_module(graph_features, x_last)
        predictions = predictions.transpose(1, 2)  # [batch, horizon, nodes]
        
        return predictions, attn_reg_loss