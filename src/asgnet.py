# -*- coding: utf-8 -*-
"""
ASGNet: Adaptive Spatiotemporal Graph Network without bottleneck layers
for efficient spatiotemporal forecasting
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# Default hyperparameters
DROPOUT = 0.2
ATTENTION_HEADS = 8
NUM_TEMPORAL_SCALES = 5
KERNEL_SIZE = 3


class MultiScaleTemporalProcessor(nn.Module):
    """
    Processes time series data at multiple temporal scales with learnable importance weights.
    No bottleneck in feature aggregation.
    """
    def __init__(self, input_dim, hidden_dim, num_scales=NUM_TEMPORAL_SCALES, kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super(MultiScaleTemporalProcessor, self).__init__()
        self.num_scales = num_scales
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Learnable scale importance weights
        self.scale_weights = Parameter(torch.ones(num_scales))
        
        # Multi-scale dilated convolutions with batch normalization
        for i in range(num_scales):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation // 2
            self.convs.append(
                nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        # Direct projection without bottleneck
        self.projection = nn.Sequential(
            nn.Linear(num_scales * hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Input x: (Batch * Nodes, Input_Dim, Window_Size)
        multi_scale_features = []
        
        # Generate normalized scale importance
        scale_importance = F.softmax(self.scale_weights, dim=0)
        
        # Process input at multiple scales
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            # Apply convolution and batch normalization
            conv_out = conv(x)
            conv_out = bn(conv_out)
            conv_out = F.relu(conv_out)
            
            # Aggregate temporal dimension with adaptive importance
            pooled = F.adaptive_avg_pool1d(conv_out, 1).squeeze(-1)
            multi_scale_features.append(pooled * scale_importance[i])
            
        # Concatenate features from all scales
        concatenated = torch.cat(multi_scale_features, dim=1)
        
        # Apply final projection
        output = self.projection(concatenated)
        return output


class DirectAttention(nn.Module):
    """
    Optimized attention mechanism with no bottleneck layers.
    Uses full-sized projections for higher expressivity.
    """
    def __init__(self, hidden_dim, num_nodes, heads=ATTENTION_HEADS, dropout=DROPOUT):
        super(DirectAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.num_nodes = num_nodes
        
        # Learnable sparsity threshold per head
        self.sparsity_threshold = Parameter(torch.zeros(heads))
        
        # Learnable regularization weights
        self.l1_reg_weight = Parameter(torch.tensor(1e-4))
        
        # Direct QKV projections without bottleneck
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        
        # Direct output projection without bottleneck
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        # Learnable graph structure (full rank)
        self.use_graph_structure = True
        if self.use_graph_structure:
            # Use direct parameterization instead of low-rank factorization
            self.attention_bias = Parameter(torch.Tensor(heads, num_nodes, num_nodes))
            nn.init.zeros_(self.attention_bias)  # Initialize with zeros
            
            # Learnable scalar for static adjacency bias
            self.adj_bias_scale = Parameter(torch.tensor(0.1))
        
    def forward(self, x, adj=None):
        # Input: (batch_size, num_nodes, hidden_dim)
        residual = x
        x = self.layer_norm(x)
        
        B, N, C = x.shape
        
        # Compute QKV projections
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each shape: (B, heads, N, head_dim)
        
        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add learnable graph structure if enabled
        if self.use_graph_structure:
            # Add learned attention bias directly (not factorized)
            attn_scores = attn_scores + self.attention_bias.unsqueeze(0)
            
            # Add static adjacency bias if provided
            if adj is not None:
                if adj.dim() == 2:
                    adj_bias = adj.unsqueeze(0).unsqueeze(1).expand(B, self.heads, N, N)
                else:
                    adj_bias = adj.unsqueeze(1).expand(B, self.heads, N, N)
                    
                # Apply bias with learnable scale
                adj_bias = torch.where(adj_bias > 0, torch.zeros_like(adj_bias), 
                                      torch.full_like(adj_bias, -1e9))
                attn_scores = attn_scores + self.adj_bias_scale * adj_bias
        
        # Apply adaptive sparsity using vectorized operations
        sparse_attn = torch.zeros_like(attn_scores).fill_(-float('inf'))
        
        # Get sparsity threshold for each head (sigmoid to keep between 0-1)
        thresholds = torch.sigmoid(self.sparsity_threshold)
        
        for h in range(self.heads):
            # Calculate k for this head - using a fixed percentage of nodes
            k_value = max(1, int(N * (1.0 - thresholds[h].item())))
            
            # For each head and batch item, get top-k values and indices
            top_values, top_indices = torch.topk(attn_scores[:, h], k=k_value, dim=-1)
            
            # Create sparse attention for this head
            batch_range = torch.arange(B, device=x.device)[:, None, None]
            node_range = torch.arange(N, device=x.device)[None, :, None]
            # Use advanced indexing to place values
            sparse_attn[batch_range, h, node_range, top_indices] = top_values
        
        # Apply softmax
        attn_weights = F.softmax(sparse_attn, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Calculate L1 regularization with learnable weight
        l1_weight = F.softplus(self.l1_reg_weight)
        reg_loss = l1_weight * torch.mean(torch.abs(attn_weights))
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attended = attended.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        output = self.out_proj(attended)
        output = self.dropout(output)
        
        # Add residual connection
        output = output + residual
        
        return output, reg_loss


class SpatiotemporalFusion(nn.Module):
    """
    Fuses spatial and temporal features with learnable importance weights.
    No bottleneck in fusion layer.
    """
    def __init__(self, hidden_dim, dropout=DROPOUT):
        super(SpatiotemporalFusion, self).__init__()
        
        # Learnable balance between spatial and temporal features
        self.spatial_importance = Parameter(torch.ones(hidden_dim))
        self.temporal_importance = Parameter(torch.ones(hidden_dim))
        
        # Direct fusion without bottleneck
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, spatial_features, temporal_features):
        # spatial_features: (batch_size, num_nodes, hidden_dim)
        # temporal_features: (batch_size, num_nodes, hidden_dim)
        
        # Calculate normalized importance weights
        spatial_weight = torch.sigmoid(self.spatial_importance)
        temporal_weight = torch.sigmoid(self.temporal_importance)
        
        # Normalize weights to sum to approximately 1
        total = spatial_weight + temporal_weight
        spatial_weight = spatial_weight / total
        temporal_weight = temporal_weight / total
        
        # Apply weights
        weighted_spatial = spatial_features * spatial_weight
        weighted_temporal = temporal_features * temporal_weight
        
        # Concatenate weighted features
        combined = torch.cat([weighted_spatial, weighted_temporal], dim=-1)
        
        # Apply fusion layer
        output = self.fusion(combined)
        
        return output


class DirectForecaster(nn.Module):
    """
    Direct forecaster module without bottleneck layers.
    """
    def __init__(self, input_dim, horizon, num_nodes, dropout=DROPOUT):
        super(DirectForecaster, self).__init__()
        
        self.horizon = horizon
        self.num_nodes = num_nodes
        
        # Direct prediction path without bottleneck
        self.prediction = nn.Sequential(
            nn.Linear(input_dim, input_dim),  # Same size hidden layer
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, horizon)
        )
        
        # Highway gate (learnable weight for highway connection)
        self.highway_gate = Parameter(torch.tensor(0.5))
        
    def forward(self, x, recent_inputs=None):
        # x: (batch_size, num_nodes, input_dim)
        B, N, _ = x.shape
        
        # Reshape for batch norm
        x_reshaped = x.view(B * N, -1)
        
        # Apply prediction layers
        pred = self.prediction(x_reshaped)
        
        # Reshape back
        pred = pred.view(B, N, self.horizon).permute(0, 2, 1)  # (B, horizon, N)
        
        # Apply highway connection if recent inputs provided
        if recent_inputs is not None:
            # Ensure highway_gate is between 0 and 1
            gate = torch.sigmoid(self.highway_gate)
            
            # Scale predictions: (1-gate)*pred + gate*highway
            pred = (1 - gate) * pred + gate * recent_inputs
            
        return pred


class LinearAttention(nn.Module):
    """
    Linear attention mechanism with O(N) complexity instead of O(N²).
    Direct projections without bottleneck.
    """
    def __init__(self, hidden_dim, num_nodes, heads=ATTENTION_HEADS, dropout=DROPOUT):
        super(LinearAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        
        # Direct QKV projections without bottleneck
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        
        # Direct output projection without bottleneck
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x, adj=None):
        # Input: (batch_size, num_nodes, hidden_dim)
        residual = x
        x = self.layer_norm(x)
        
        B, N, C = x.shape
        
        # QKV projections
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each shape: (B, heads, N, head_dim)
        
        # Apply ELU + 1 kernel for linear attention
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        
        # Linear attention: compute KV first, then multiply by Q
        kv = torch.einsum('bhnd,bhne->bhde', k, v)
        
        # Normalize by the sum of keys
        k_sum = k.sum(dim=2, keepdim=True)
        
        # Compute attention output
        output = torch.einsum('bhnd,bhde->bhne', q, kv)
        
        # Normalize by the sum of queries times sum of keys
        normalizer = torch.einsum('bhnd,bhd->bhn', q, k_sum.squeeze(-2))
        output = output / (normalizer.unsqueeze(-1) + 1e-8)
        
        # Reshape and apply output projection
        output = output.permute(0, 2, 1, 3).contiguous().view(B, N, C)
        output = self.out_proj(output)
        output = self.dropout(output)
        
        # Add residual connection
        output = output + residual
        
        # Return zero regularization loss for API compatibility
        return output, torch.tensor(0.0, device=output.device)


class ASGNet(nn.Module):
    """
    ASGNet: Adaptive Spatiotemporal Graph Network without bottleneck layers
    for efficient spatiotemporal forecasting tasks.
    """
    def __init__(self, args, data_loader):
        super(ASGNet, self).__init__()
        self.num_nodes = data_loader.m
        self.window = args.window
        self.horizon = args.horizon
        self.hidden_dim = args.hidden_dim
        self.highway_window = getattr(args, 'highway_window', 0)
        
        # Check if linear attention is requested
        self.use_linear_attention = getattr(args, 'use_linear_attention', False)
        
        # 1. Temporal Module
        self.temporal_module = MultiScaleTemporalProcessor(
            input_dim=1,  # Assuming raw time series
            hidden_dim=self.hidden_dim,
            num_scales=args.num_scales,
            kernel_size=args.kernel_size,
            dropout=args.dropout
        )
        
        # 2. Spatial Attention Module - choose implementation
        if self.use_linear_attention:
            self.spatial_module = LinearAttention(
                hidden_dim=self.hidden_dim,
                num_nodes=self.num_nodes,
                heads=args.attention_heads,
                dropout=args.dropout
            )
        else:
            self.spatial_module = DirectAttention(
                hidden_dim=self.hidden_dim,
                num_nodes=self.num_nodes,
                heads=args.attention_heads,
                dropout=args.dropout
            )
        
        # 3. Optional Spatiotemporal Fusion Module
        self.use_st_fusion = getattr(args, 'use_st_fusion', True)
        if self.use_st_fusion:
            self.fusion_module = SpatiotemporalFusion(
                hidden_dim=self.hidden_dim,
                dropout=args.dropout
            )
        
        # 4. Forecasting Module
        self.forecaster = DirectForecaster(
            input_dim=self.hidden_dim,
            horizon=self.horizon,
            num_nodes=self.num_nodes,
            dropout=args.dropout
        )
        
        # Optional Highway Component
        if self.highway_window > 0:
            self.highway_conv = nn.Conv2d(
                self.highway_window, self.horizon, kernel_size=(1, 1)
            )
        
        # Register static adjacency matrix if provided
        static_adj = getattr(data_loader, 'adj', None)
        if static_adj is not None and isinstance(static_adj, torch.Tensor):
            self.register_buffer('static_adj', static_adj)
            print(f"Registered static_adj buffer with shape: {self.static_adj.shape}")
        else:
            self.static_adj = None
            print("No static_adj registered.")
    
    def forward(self, x, index=None):
        # x: (batch_size, window, num_nodes)
        B, W, N = x.shape
        
        # Store inputs for potential highway connection
        highway_input = None
        if self.highway_window > 0 and self.highway_window <= W:
            highway_input = x[:, -self.highway_window:, :]
        
        # 1. Temporal Processing
        # Reshape for temporal module: (B, W, N) -> (B*N, 1, W)
        x_temporal = x.permute(0, 2, 1).reshape(B * N, 1, W)
        temporal_features = self.temporal_module(x_temporal)
        temporal_features = temporal_features.view(B, N, self.hidden_dim)
        
        # 2. Spatial Processing 
        spatial_features, attn_reg_loss = self.spatial_module(
            temporal_features, adj=self.static_adj
        )
        
        # 3. Optional Spatiotemporal Fusion
        if self.use_st_fusion:
            features = self.fusion_module(spatial_features, temporal_features)
        else:
            features = spatial_features
        
        # 4. Generate Predictions
        if self.highway_window > 0 and highway_input is not None:
            # Process highway connection
            # Shape: (B, highway_window, N) -> (B, highway_window, 1, N)
            z = highway_input.unsqueeze(2)
            highway_pred = self.highway_conv(z).squeeze(2)  # (B, horizon, N)
            
            # Pass both to forecaster for adaptive combination
            predictions = self.forecaster(features, highway_pred)
        else:
            predictions = self.forecaster(features)
        
        return predictions, attn_reg_loss