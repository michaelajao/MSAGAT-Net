# -*- coding: utf-8 -*-
"""
Enhanced MSAGAT-Net model incorporating static adjacency bias and highway connections.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# Default values (consider moving to args or config)
DROPOUT = 0.2
ATTENTION_HEADS = 8
ATTENTION_REG_WEIGHT = 1e-3
BOTTLENECK_DIM = 6
NUM_TEMPORAL_SCALES = 5
KERNEL_SIZE = 3

# --- Helper Modules ---

class DepthwiseSeparableConv1D(nn.Module):
    """Depthwise Separable 1D Convolution."""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, dilation=1):
        super(DepthwiseSeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size,
                                   padding=padding, dilation=dilation, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.relu(x)

class MultiScaleTemporalModule(nn.Module):
    """
    Multi-Scale Temporal Module adapted for (B*N, C_in, L) input.
    Outputs aggregated temporal features (B*N, hidden_dim).
    """
    def __init__(self, input_dim, hidden_dim, num_scales=NUM_TEMPORAL_SCALES, kernel_size=KERNEL_SIZE, dropout=DROPOUT):
        super(MultiScaleTemporalModule, self).__init__()
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        self.convs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        # Input dim is typically 1 for raw time series per node
        for i in range(num_scales):
            dilation = 2**i
            # Calculate padding for 'same' convolution
            padding = (kernel_size - 1) * dilation // 2
            # Use standard Conv1d as input channel is 1
            self.convs.append(
                nn.Conv1d(input_dim, hidden_dim, kernel_size, padding=padding, dilation=dilation)
            )

        # Adaptive fusion layer
        self.fusion_layer = nn.Linear(num_scales * hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input x: (Batch * Nodes, Input_Dim, Window_Size)
        outputs = []
        for conv in self.convs:
            out = self.relu(conv(x))
            # Aggregate time dimension (using adaptive average pooling)
            out = F.adaptive_avg_pool1d(out, 1).squeeze(-1) # Output: (B*N, hidden_dim)
            outputs.append(out)

        # Concatenate features from different scales
        concatenated = torch.cat(outputs, dim=1) # (B*N, num_scales * hidden_dim)
        fused = self.relu(self.fusion_layer(concatenated)) # (B*N, hidden_dim)
        return self.dropout(fused)

class SpatialAttentionModule(nn.Module):
    """
    Spatial Attention Module with optional static adjacency bias.
    """
    def __init__(self, hidden_dim, num_nodes, dropout=DROPOUT, attention_heads=ATTENTION_HEADS,
                 attention_regularization_weight=ATTENTION_REG_WEIGHT, bottleneck_dim=BOTTLENECK_DIM):
        super(SpatialAttentionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.heads = attention_heads
        # Ensure head_dim calculation is valid
        if hidden_dim % self.heads != 0:
             raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by attention_heads ({self.heads})")
        self.head_dim = hidden_dim // self.heads
        self.num_nodes = num_nodes
        self.attention_regularization_weight = attention_regularization_weight
        self.bottleneck_dim = bottleneck_dim

        # Low-rank projections for query, key, value
        self.qkv_proj_low = nn.Linear(hidden_dim, 3 * bottleneck_dim)
        self.qkv_proj_high = nn.Linear(3 * bottleneck_dim, 3 * hidden_dim)

        # Low-rank projections for output
        self.out_proj_low = nn.Linear(hidden_dim, bottleneck_dim)
        self.out_proj_high = nn.Linear(bottleneck_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim) # Add LayerNorm

        # Learnable graph structure bias (low-rank)
        self.u = Parameter(torch.Tensor(self.heads, num_nodes, bottleneck_dim))
        self.v = Parameter(torch.Tensor(self.heads, bottleneck_dim, num_nodes))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

        # Learnable scalar for static adjacency bias
        self.adj_bias_scale = Parameter(torch.tensor(0.0)) # Initialize with zero bias

    def _compute_attention(self, q, k, v, adj_bias=None):
        """Computes attention scores and applies attention to values."""
        # Apply ELU activation + 1 for stability (Linear Attention Kernel)
        # q = F.elu(q) + 1.0
        # k = F.elu(k) + 1.0

        # --- Revisit Standard Attention with Bias (for comparison/simplicity if linear fails) ---
        # Calculate QK^T scores
        attn_scores = torch.einsum('bhnd,bhmd->bhnm', q, k) / math.sqrt(self.head_dim) # (B, H, N, N)

        # Add learnable low-rank graph bias
        graph_bias = torch.einsum('hnb,hbm->hnm', self.u, self.v) # (H, N, N)
        attn_scores = attn_scores + graph_bias.unsqueeze(0) # Add bias across batch dim (B, H, N, N)

        # Add optional static adjacency matrix bias
        if adj_bias is not None:
            # adj_bias is already (B, H, N, N) with large negative for non-edges
            attn_scores = attn_scores + self.adj_bias_scale * adj_bias

        # Normalize attention scores using softmax
        attn_probs = torch.softmax(attn_scores, dim=-1) # (B, H, N, N)
        attn_probs = self.dropout(attn_probs) # Apply dropout to attention weights

        # Apply attention to values (AttnProbs * V)
        output = torch.einsum('bhnm,bhmd->bhnd', attn_probs, v) # (B, H, N, D_head)

        # --- REMOVED INCORRECT REGULARIZATION ---
        # attn_reg_loss = self.attention_regularization_weight * torch.mean(torch.sum(torch.abs(attn_probs), dim=-1))
        attn_reg_loss = torch.tensor(0.0, device=output.device) # Set loss to zero

        # Store attn_probs for potential visualization later if needed
        # Use try-except to handle cases where model might not be training (e.g., during eval)
        try:
            self.attn_probs_viz = attn_probs.detach() # Detach to prevent gradient issues if only for viz
        except Exception: # Catch potential errors if attribute setting fails outside training
            pass


        return output, attn_reg_loss


    def forward(self, x, adj=None, mask=None):
        """
        Forward pass for the Spatial Attention Module.

        Args:
            x (torch.Tensor): Input node features (batch_size, num_nodes, hidden_dim).
            adj (torch.Tensor, optional): Static adjacency matrix (num_nodes, num_nodes)
                                          or (batch_size, num_nodes, num_nodes). Assumes 1 for edge, 0 otherwise.
            mask (torch.Tensor, optional): Attention mask. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output node features and attention regularization loss.
        """
        B, N, C = x.shape # Batch, Nodes, Channels (hidden_dim)
        x_residual = x # Store for residual connection

        # Apply LayerNorm before attention
        x = self.layer_norm(x)

        # Apply low-rank projections for Q, K, V
        qkv_low = self.qkv_proj_low(x)
        qkv = self.qkv_proj_high(qkv_low).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # Shape: (B, heads, N, head_dim)

        # Prepare adjacency matrix bias if provided
        adj_bias = None
        if adj is not None:
            if adj.dim() == 2: # Static adj (N, N) -> expand
                adj_bias = adj.unsqueeze(0).unsqueeze(1).expand(B, self.heads, N, N)
            elif adj.dim() == 3: # Batch-specific adj (B, N, N) -> expand
                adj_bias = adj.unsqueeze(1).expand(B, self.heads, N, N)
            else:
                 raise ValueError(f"Adjacency matrix has unexpected dimension: {adj.dim()}")

            # Set non-edges to a large negative value (-inf can cause NaN with autocast/float16)
            # Use a large negative number instead.
            adj_bias = torch.where(adj_bias > 0, torch.zeros_like(adj_bias), torch.full_like(adj_bias, -1e9))


        # Compute attention (Using standard softmax attention with bias for now)
        attn_output, attn_reg_loss = self._compute_attention(q, k, v, adj_bias=adj_bias)

        # Reshape and apply output projection
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(B, N, C) # (B, N, hidden_dim)
        out_low = self.out_proj_low(attn_output)
        output = self.out_proj_high(out_low)

        # Apply dropout and residual connection
        output = self.dropout(output)
        output = output + x_residual # Add residual connection

        return output, attn_reg_loss


class HorizonPredictor(nn.Module):
    """Predicts multiple future steps using node features."""
    def __init__(self, input_dim, horizon, num_nodes, bottleneck_dim=BOTTLENECK_DIM):
        super(HorizonPredictor, self).__init__()
        self.horizon = horizon
        self.num_nodes = num_nodes
        self.input_dim = input_dim

        # Use bottleneck for efficiency
        self.projection1 = nn.Linear(input_dim, bottleneck_dim)
        # Output dimension needs to be horizon * num_nodes if predicting all nodes at once
        # Or just horizon if predicting per node independently
        # Let's predict per node: output dim = horizon
        self.projection2 = nn.Linear(bottleneck_dim, horizon)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Input x: (batch_size, num_nodes, input_dim)
        B, N, _ = x.shape
        x = self.relu(self.projection1(x)) # (B, N, bottleneck_dim)
        output = self.projection2(x)      # (B, N, horizon)

        # Reshape to desired output: (B, horizon, N)
        output = output.permute(0, 2, 1)
        return output

# --- Main Enhanced Model ---

class EnhancedMSAGAT_Net(nn.Module):
    """
    Enhanced Multi-Scale Temporal Attention Graph Network (MSAGAT-Net).
    - Temporal processing first, then spatial.
    - Optional static adjacency bias in spatial attention.
    - Optional highway connection.
    """
    def __init__(self, args, data_loader):
        super(EnhancedMSAGAT_Net, self).__init__()
        self.num_nodes = data_loader.m
        self.window = args.window
        self.horizon = args.horizon
        self.dropout_rate = args.dropout
        self.hidden_dim = args.hidden_dim
        self.input_dim = 1 # Assuming input feature dim per node is 1

        # Highway window argument
        self.highway_window = getattr(args, 'highway_window', 0) # Default to 0

        # Temporal Module (Processes raw time series per node)
        self.temporal_module = MultiScaleTemporalModule(
            input_dim=self.input_dim, # Raw time series feature dim
            hidden_dim=self.hidden_dim,
            num_scales=args.num_scales,
            kernel_size=args.kernel_size,
            dropout=self.dropout_rate
        )

        # Spatial Attention Module
        self.spatial_attention = SpatialAttentionModule(
            hidden_dim=self.hidden_dim,
            num_nodes=self.num_nodes,
            dropout=self.dropout_rate,
            attention_heads=args.attention_heads,
            attention_regularization_weight=args.attention_regularization_weight,
            bottleneck_dim=args.bottleneck_dim
        )

        # Horizon Predictor Module
        self.horizon_predictor = HorizonPredictor(
            input_dim=self.hidden_dim, # Takes output of spatial module
            horizon=self.horizon,
            num_nodes=self.num_nodes,
            bottleneck_dim=args.bottleneck_dim
        )

        # Optional Highway component (Conv2d to map input window slice to output horizon)
        if self.highway_window > 0:
            # Input channels = highway_window, Output channels = horizon
            # Kernel size (1, 1) acts like a per-node linear layer across time steps
            self.highway_conv = nn.Conv2d(self.highway_window, self.horizon, kernel_size=(1, 1))

        # Store static adjacency matrix if provided by data_loader
        # Ensure it's registered as a buffer if it's a tensor
        static_adj_tensor = getattr(data_loader, 'adj', None)
        if static_adj_tensor is not None and isinstance(static_adj_tensor, torch.Tensor):
             self.register_buffer('static_adj', static_adj_tensor)
             print(f"Registered static_adj buffer with shape: {self.static_adj.shape}")
        else:
             self.static_adj = None
             print("No static_adj registered.")


    def forward(self, x, index=None):
        """
        Forward pass of the Enhanced MSTAGAT-Net model.

        Args:
            x (torch.Tensor): Input time series data (batch_size, window, num_nodes).
                              Assumes input feature dim is 1.
            index (torch.Tensor, optional): Batch indices (unused in this version).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predictions (B, H, N) and attention regularization loss.
        """
        B, W, N = x.shape
        if N != self.num_nodes:
            raise ValueError(f"Input node dimension ({N}) does not match model's num_nodes ({self.num_nodes})")

        # --- 1. Temporal Processing ---
        # Reshape input for temporal module: (B, W, N) -> (B, N, W) -> (B*N, 1, W)
        x_permuted = x.permute(0, 2, 1) # (B, N, W)
        x_temporal_input = x_permuted.reshape(B * N, self.input_dim, W)

        # Apply temporal module per node
        # Output: (B*N, hidden_dim) - aggregated temporal features
        h_temporal_agg = self.temporal_module(x_temporal_input)

        # Reshape back: (B*N, hidden_dim) -> (B, N, hidden_dim)
        h_temporal_agg = h_temporal_agg.view(B, N, self.hidden_dim)

        # --- 2. Spatial Processing ---
        # Apply spatial attention on the aggregated temporal features
        # Pass static_adj if it exists and is registered
        h_spatial, attn_reg_loss = self.spatial_attention(h_temporal_agg, adj=self.static_adj)
        # Output h_spatial: (B, N, hidden_dim)

        # --- 3. Prediction ---
        predictions = self.horizon_predictor(h_spatial) # Output: (B, horizon, N)

        # --- 4. Optional Highway Component ---
        if self.highway_window > 0 and self.highway_window <= W:
            # Take last 'highway_window' inputs: x shape is (B, W, N)
            z = x[:, -self.highway_window:, :] # Shape: (B, highway_window, N)

            # Prepare for Conv2d: (B, C_in, H_in, W_in)
            # Treat highway_window as input channels, H=1, W=N
            z = z.unsqueeze(2) # Shape: (B, highway_window, 1, N)

            # Apply highway convolution
            highway_pred = self.highway_conv(z) # Output: (B, horizon, 1, N)
            highway_pred = highway_pred.squeeze(2) # Output: (B, horizon, N)

            # Add highway component to the main predictions
            predictions = predictions + highway_pred

        # Ensure predictions have the correct shape
        if predictions.shape != (B, self.horizon, N):
             raise RuntimeError(f"Prediction shape mismatch: Expected {(B, self.horizon, N)}, Got {predictions.shape}")


        return predictions, attn_reg_loss
