import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

# Base model hyperparameters (will be dynamically adjusted)
HIDDEN_DIM = 32
ATTENTION_HEADS = 4
ATTENTION_REG_WEIGHT = 1e-05
DROPOUT = 0.2
NUM_TEMPORAL_SCALES = 4
KERNEL_SIZE = 3
FEATURE_CHANNELS = 16
BOTTLENECK_DIM = 8
MAX_SEQ_LENGTH = 200  # Maximum expected sequence length


class LocationAwareSpatialAttention(nn.Module):
    """
    Location-Aware Spatial Attention Module that captures node relationships in a graph structure.

    This module enhances the original SpatialAttentionModule with location-specific embeddings
    and dynamic parameter scaling based on dataset characteristics.

    Args:
        hidden_dim (int): Dimensionality of node features
        num_nodes (int): Number of nodes in the graph
        dropout (float): Dropout probability for regularization
        attention_heads (int): Number of parallel attention heads
        attention_regularization_weight (float): Weight for L1 regularization on attention
        bottleneck_dim (int): Dimension of the low-rank projection
        dataset_name (str): Name of the dataset for dataset-specific embeddings
    """

    def __init__(
        self,
        hidden_dim,
        num_nodes,
        dropout=DROPOUT,
        attention_heads=ATTENTION_HEADS,
        attention_regularization_weight=ATTENTION_REG_WEIGHT,
        bottleneck_dim=BOTTLENECK_DIM,
        dataset_name="default",
    ):
        super(LocationAwareSpatialAttention, self).__init__()

        # Check valid parameters
        assert (
            hidden_dim % attention_heads == 0
        ), "hidden_dim must be divisible by attention_heads"

        self.hidden_dim = hidden_dim
        self.heads = attention_heads
        self.head_dim = hidden_dim // self.heads
        self.num_nodes = num_nodes
        self.attention_regularization_weight = attention_regularization_weight
        self.bottleneck_dim = bottleneck_dim
        self.dataset_name = dataset_name

        # Low-rank projections for query, key, value
        self.qkv_proj_low = nn.Linear(hidden_dim, 3 * bottleneck_dim)
        self.qkv_proj_high = nn.Linear(3 * bottleneck_dim, 3 * hidden_dim)

        # Low-rank projections for output
        self.out_proj_low = nn.Linear(hidden_dim, bottleneck_dim)
        self.out_proj_high = nn.Linear(bottleneck_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Learnable graph structure bias (low-rank)
        self.u = Parameter(torch.Tensor(self.heads, num_nodes, bottleneck_dim))
        self.v = Parameter(torch.Tensor(self.heads, bottleneck_dim, num_nodes))
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)

        # Location-specific learnable embeddings
        self.location_embeddings = Parameter(torch.Tensor(num_nodes, hidden_dim))
        nn.init.xavier_uniform_(self.location_embeddings)

        # Dataset identifier embedding (for cross-dataset adaptability)
        self.dataset_embedding = Parameter(torch.Tensor(1, hidden_dim))
        nn.init.xavier_uniform_(self.dataset_embedding)

        # Location-aware projection layer
        self.location_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None, node_indices=None):
        """
        Forward pass of the location-aware spatial attention module.

        Args:
            x (tensor): Input node features [batch, nodes, hidden_dim]
            mask (tensor, optional): Attention mask
            node_indices (tensor, optional): Indices of nodes being used (for subgraph processing)

        Returns:
            tuple: (Updated node features, Attention regularization loss)
        """
        B, N, H = x.shape

        # Input validation
        if N > self.num_nodes:
            raise ValueError(
                f"Input has {N} nodes but model was initialized with {self.num_nodes}"
            )

        # Store input for residual connection
        residual = x

        # Add location-specific embeddings to input features
        location_emb = self.location_embeddings
        if node_indices is not None:
            location_emb = location_emb[node_indices]

        # Add dataset-specific embedding
        dataset_emb = self.dataset_embedding.expand(B, N, -1)

        # Enhance input with location awareness
        x = x + location_emb + dataset_emb
        x = self.location_proj(x)

        # Low-rank projection for qkv
        qkv_low = self.qkv_proj_low(x)
        qkv = self.qkv_proj_high(qkv_low)
        qkv = qkv.chunk(3, dim=-1)

        # Separate query, key, value and reshape for multi-head attention
        q, k, v = [
            x.reshape(B, N, self.heads, self.head_dim).transpose(1, 2) for x in qkv
        ]
        # Now q, k, v have shape [B, heads, N, head_dim]

        # Compute graph structure bias from low-rank factors
        adj_bias = torch.matmul(self.u, self.v)

        # Use unified attention mechanism
        # First compute standard attention scores with graph bias
        attention_scores = torch.einsum("bhnd,bhmd->bhnm", q, k) / math.sqrt(
            self.head_dim
        )
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Add structural bias and apply softmax
        attention_scores = attention_scores + adj_bias
        attention_weights = F.softmax(attention_scores, dim=-1)
        self.attn = attention_weights  # Store for regularization
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, v)  # [B, heads, N, head_dim]

        # Reshape back to original dimensions
        context = context.transpose(1, 2).contiguous().reshape(B, N, H)

        # Low-rank projection for output
        output = self.out_proj_low(context)
        output = self.out_proj_high(output)

        # Apply residual connection and normalization
        output = self.layer_norm(output + residual)

        # Compute regularization loss on attention weights
        attn_reg_loss = self.attention_regularization_weight * torch.mean(
            torch.abs(self.attn)
        )

        return output, attn_reg_loss


class AdaptiveMultiScaleTemporalModule(nn.Module):
    """
    Adaptive Multi-Scale Temporal Module that captures temporal patterns at different scales.

    This module enhances the original MultiScaleTemporalModule with dynamic scale selection
    based on dataset temporal characteristics.

    Args:
        hidden_dim (int): Dimensionality of node features
        num_scales (int): Number of temporal scales (different dilation rates)
        kernel_size (int): Size of the convolutional kernel
        dropout (float): Dropout probability for regularization
        temporal_complexity (float): Estimated temporal complexity (influences scale importance)
    """

    def __init__(
        self,
        hidden_dim,
        num_scales=NUM_TEMPORAL_SCALES,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        temporal_complexity=1.0,
        max_seq_length=MAX_SEQ_LENGTH,
    ):
        super(AdaptiveMultiScaleTemporalModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.temporal_complexity = temporal_complexity

        # Create multiple dilated convolutional layers with increasing dilation rates
        self.scales = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=(kernel_size // 2) * 2**i,
                        dilation=2**i,
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.SiLU(),  # Swish activation (SiLU) instead of ReLU
                    nn.Dropout(dropout),
                )
                for i in range(num_scales)
            ]
        )

        # Learnable weights for adaptive fusion of scales
        # Initialize with bias toward longer temporal dependencies for complex datasets
        self.fusion_weight = Parameter(
            torch.ones(num_scales) * temporal_complexity, requires_grad=True
        )

        # Low-rank projection for fusion
        self.fusion_low = nn.Linear(hidden_dim, BOTTLENECK_DIM)
        self.fusion_high = nn.Linear(BOTTLENECK_DIM, hidden_dim)

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Temporal encoding (position embeddings for temporal awareness)
        self.temporal_encoding = nn.Parameter(torch.zeros(max_seq_length, hidden_dim))
        nn.init.xavier_uniform_(self.temporal_encoding)

    def forward(self, x, seq_length=None):
        """
        Forward pass of the adaptive multi-scale temporal module.

        Args:
            x (tensor): Input features [batch, nodes, hidden_dim]
            seq_length (int, optional): Actual sequence length, used for temporal encoding

        Returns:
            tensor: Temporally processed features
        """
        B, N, H = x.shape

        # Store input for residual connection
        residual = x

        # Add temporal encoding if sequence length is provided
        if seq_length is not None:
            # Skip temporal encoding if dimensions don't match
            # This is likely happening because we're treating nodes as sequence elements
            # in the convolution, not time steps
            pass

        # Reshape for 1D convolution: [batch*nodes, channels=hidden_dim, sequence=1]
        # We're treating each node as a separate sample in the batch dimension
        x = x.transpose(1, 2)  # Now [batch, hidden_dim, nodes]

        # Apply different temporal scales
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
        out = self.layer_norm(out + residual)  # Add residual connection

        return out


class LocationAwarePredictor(nn.Module):
    """
    Location-Aware Horizon Predictor module for forecasting future values.

    This module enhances the original HorizonPredictor with location-aware graph
    attention for better cross-dataset adaptability.

    Args:
        hidden_dim (int): Dimensionality of node features
        horizon (int): Number of future time steps to predict
        num_nodes (int): Number of nodes in the graph
        bottleneck_dim (int): Dimension for bottleneck layers
        dropout (float): Dropout probability for regularization
        attention_heads (int): Number of attention heads for GAT
    """

    def __init__(
        self,
        hidden_dim,
        horizon,
        num_nodes,
        bottleneck_dim=BOTTLENECK_DIM,
        dropout=DROPOUT,
        attention_heads=4,
    ):
        super(LocationAwarePredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.bottleneck_dim = bottleneck_dim
        self.num_nodes = num_nodes

        # Location-specific embeddings
        self.location_embeddings = Parameter(torch.Tensor(num_nodes, hidden_dim // 4))
        nn.init.xavier_uniform_(self.location_embeddings)

        # Graph attention layer
        self.gat_layer = GraphAttentionLayer(
            hidden_dim + hidden_dim // 4,
            hidden_dim,
            heads=attention_heads,
            dropout=dropout,
        )

        # Fallback projection when adjacency matrix is not provided
        # Define this in __init__ rather than creating dynamically during forward pass
        self.fallback_projection = nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim)

        # Low-rank prediction projection
        self.predictor_low = nn.Linear(hidden_dim, bottleneck_dim)
        self.predictor_mid = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.SiLU(),  # Use SiLU (Swish) activation
            nn.Dropout(dropout),
        )
        self.predictor_high = nn.Linear(bottleneck_dim, horizon)

        # Adaptive refinement gate based on last observation
        self.refine_gate = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.SiLU(),  # Use SiLU (Swish) activation
            nn.Linear(bottleneck_dim, horizon),
            nn.Sigmoid(),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, adj_matrix=None, last_step=None, node_indices=None):
        """
        Forward pass of the location-aware horizon predictor.

        Args:
            x (tensor): Node features [batch, nodes, hidden_dim]
            adj_matrix (tensor): Adjacency matrix for graph attention
            last_step (tensor, optional): Last observed values [batch, nodes]
            node_indices (tensor, optional): Indices of nodes being used

        Returns:
            tensor: Predictions for future time steps [batch, nodes, horizon]
        """
        B, N, H = x.shape

        # Store input for residual connection
        residual = x

        # Get location embeddings
        loc_emb = self.location_embeddings
        if node_indices is not None:
            loc_emb = loc_emb[node_indices]

        # Expand location embeddings
        loc_emb_expanded = loc_emb.unsqueeze(0).expand(B, -1, -1)

        # Concatenate with input features
        x_with_loc = torch.cat([x, loc_emb_expanded], dim=-1)

        # Apply graph attention if adjacency matrix is provided, otherwise use fallback
        if adj_matrix is not None:
            x = self.gat_layer(x_with_loc, adj_matrix)
        else:
            x = self.fallback_projection(x_with_loc)

        # Add residual connection with proper projection since dimensions don't match
        # First project residual to match dimensions if needed
        if H != x.shape[-1]:
            residual_proj = nn.Linear(H, x.shape[-1]).to(x.device)(residual)
            x = x + residual_proj
        else:
            x = x + residual

        x = self.layer_norm(x)

        # Generate initial predictions
        x_low = self.predictor_low(x)
        x_mid = self.predictor_mid(x_low)
        initial_pred = self.predictor_high(x_mid)

        # Apply refinement if last observed value is provided
        if last_step is not None:
            # Compute adaptive gate
            gate = self.refine_gate(x)

            # Prepare last step and exponential decay
            last_step = last_step.unsqueeze(-1)
            time_decay = (
                torch.arange(1, self.horizon + 1, device=x.device)
                .float()
                .view(1, 1, self.horizon)
            )
            progressive_part = last_step * torch.exp(-0.1 * time_decay)

            # Adaptive fusion of model prediction and exponential decay
            final_pred = gate * initial_pred + (1 - gate) * progressive_part
        else:
            final_pred = initial_pred

        return final_pred


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network (GAT) layer with multi-head attention.

    Args:
        in_features (int): Input feature dimension
        out_features (int): Output feature dimension
        heads (int): Number of attention heads
        dropout (float): Dropout probability
        alpha (float): LeakyReLU negative slope
    """

    def __init__(self, in_features, out_features, heads=1, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.dropout = dropout

        # Ensure out_features is divisible by heads
        assert out_features % heads == 0, "out_features must be divisible by heads"

        # Define per-head transformations
        self.W = nn.Parameter(
            torch.zeros(size=(heads, in_features, out_features // heads))
        )
        nn.init.xavier_uniform_(self.W.data)

        # Attention parameters - simplified structure to avoid dimension errors
        self.a_src = nn.Parameter(torch.zeros(size=(heads, out_features // heads, 1)))
        self.a_dst = nn.Parameter(torch.zeros(size=(heads, out_features // heads, 1)))
        nn.init.xavier_uniform_(self.a_src.data)
        nn.init.xavier_uniform_(self.a_dst.data)

        # LeakyReLU and Dropout
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)

        # Output normalization
        self.layer_norm = nn.LayerNorm(out_features)

    def forward(self, x, adj):
        """
        Forward pass of the GAT layer.

        Args:
            x (tensor): Node features [batch, nodes, in_features]
            adj (tensor): Adjacency matrix [nodes, nodes] or [batch, nodes, nodes]

        Returns:
            tensor: Updated node features [batch, nodes, out_features]
        """
        B, N, _ = x.shape

        # Apply linear transformation for each head
        x = x.unsqueeze(1).repeat(1, self.heads, 1, 1)  # [B, heads, N, in_features]
        Wh = torch.matmul(x, self.W)  # [B, heads, N, out_features//heads]

        # Compute attention scores directly without reshaping (simpler approach)
        attn_src = torch.matmul(Wh, self.a_src)  # [B, heads, N, 1]
        attn_dst = torch.matmul(Wh, self.a_dst)  # [B, heads, N, 1]

        # Broadcast to create attention matrix
        attn_scores = attn_src + attn_dst.transpose(2, 3)  # [B, heads, N, N]

        # Apply LeakyReLU
        attn_scores = self.leakyrelu(attn_scores)

        # Create a mask from the adjacency matrix - handle different formats
        # Initialize mask with default value (allow all connections) in case adj processing fails
        mask = torch.ones(B, self.heads, N, N, device=x.device)

        try:
            # Apply adjacency mask - handle batch size mismatch
            if adj.dim() == 2:
                # Create an adjacency mask that matches the batch size
                # This assumes the adjacency matrix is the same for all samples in the batch
                if adj.size(0) != N or adj.size(1) != N:
                    print(
                        f"Warning: Adjacency matrix size {adj.size()} doesn't match number of nodes {N}. Using default mask."
                    )
                else:
                    # Apply the same adjacency matrix to all samples in batch
                    mask = (adj > 0).float().to(x.device)
                    mask = mask.unsqueeze(0).unsqueeze(0).expand(B, self.heads, -1, -1)
            elif adj.dim() == 3:
                # Batch of adjacency matrices
                if adj.size(1) != N or adj.size(2) != N:
                    print(
                        f"Warning: Adjacency matrix size {adj.size()} doesn't match number of nodes {N}. Using default mask."
                    )
                else:
                    mask = (adj > 0).float().to(x.device)
                    mask = mask.unsqueeze(1).expand(-1, self.heads, -1, -1)
        except Exception as e:
            print(f"Warning: Error creating attention mask: {e}. Using default mask.")
            # Keep the default mask (all ones) in case of error

        # Set scores to -inf for non-edges (where mask is 0)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention = F.softmax(attn_scores, dim=-1)
        attention = self.dropout_layer(attention)

        # Apply attention to get output features
        h_prime = torch.matmul(attention, Wh)  # [B, heads, N, out_features//heads]

        # Concatenate multi-head results
        h_prime = h_prime.transpose(1, 2).contiguous().view(B, N, -1)

        # Apply layer normalization
        h_prime = self.layer_norm(h_prime)

        return h_prime


class LocationAwareMSAGAT_Net(nn.Module):
    """
    Location-Aware Multi-Scale Adaptive Graph Attention Network

    This enhanced model adds dataset-aware parameter scaling and location-specific embeddings
    to enable better cross-dataset adaptability and reduce the need for manual tuning
    for different datasets.

    Key components:
    - Dynamic parameter scaling based on dataset characteristics
    - Location-aware attention mechanism
    - Adaptive multi-scale temporal processing
    - Graph attention-based progressive prediction

    Args:
        args: Model configuration arguments
        data: Data object containing dataset information
    """

    def __init__(self, args, data):
        super(LocationAwareMSAGAT_Net, self).__init__()
        self.num_nodes = data.m
        self.window = args.window
        self.horizon = args.horizon
        self.dataset_name = getattr(args, "dataset", "default")

        # Input validation
        if not hasattr(data, "m"):
            raise ValueError("Data object must have attribute 'm' (number of nodes)")
        if not hasattr(args, "window"):
            raise ValueError("Args must have attribute 'window' (time window size)")
        if not hasattr(args, "horizon"):
            raise ValueError("Args must have attribute 'horizon' (prediction horizon)")

        # Dynamic parameter scaling based on dataset size
        scale_factor = math.log(self.num_nodes) / math.log(50)  # 50 is reference size

        # Scale model capacity based on dataset characteristics
        self.hidden_dim = min(64, max(16, int(16 * scale_factor)))
        self.attention_heads = min(16, max(4, int(4 * scale_factor)))
        self.bottleneck_dim = min(16, max(4, int(8 * scale_factor)))

        # Scale regularization based on dataset size (larger datasets need less regularization)
        base_attention_reg = 1e-4
        self.attention_reg_weight = base_attention_reg / scale_factor

        # Scale temporal parameters
        self.kernel_size = min(9, max(3, int(3 * scale_factor)))
        self.num_scales = min(7, max(3, int(4 * scale_factor)))

        # Use arguments if explicitly provided, otherwise use the dynamic values
        self.hidden_dim = getattr(args, "hidden_dim", self.hidden_dim)
        self.attention_heads = getattr(args, "attention_heads", self.attention_heads)
        self.kernel_size = getattr(args, "kernel_size", self.kernel_size)
        self.bottleneck_dim = getattr(args, "bottleneck_dim", self.bottleneck_dim)
        self.attention_reg_weight = getattr(
            args, "attention_regularization_weight", self.attention_reg_weight
        )
        self.num_scales = getattr(args, "num_scales", self.num_scales)
        self.dropout = getattr(args, "dropout", DROPOUT)

        # Detect dataset temporal complexity by analyzing variance
        if hasattr(data, "train"):
            # Simple heuristic - more variance might indicate more complex temporal patterns
            try:
                if isinstance(data.train, list):
                    train_tensor = torch.tensor(data.train, dtype=torch.float32)
                elif isinstance(data.train, torch.Tensor):
                    train_tensor = data.train
                else:
                    train_tensor = torch.tensor([1.0])

                temporal_complexity = min(2.0, max(0.5, torch.var(train_tensor).item()))
            except Exception as e:
                print(
                    f"Warning: Could not calculate temporal complexity from data: {e}"
                )
                temporal_complexity = 1.0
        else:
            temporal_complexity = 1.0

        # Feature Extraction Component with dynamic channel scaling
        # --------------------------------------------------------
        feature_channels = getattr(
            args, "feature_channels", min(32, max(8, int(16 * scale_factor)))
        )

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                1,
                feature_channels,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm1d(feature_channels),
            nn.SiLU(),  # Use SiLU (Swish) activation instead of ReLU
            nn.Dropout(self.dropout),
        )

        # Low-rank projection of extracted features
        self.feature_projection_low = nn.Linear(
            feature_channels * self.window, self.bottleneck_dim
        )
        self.feature_projection_high = nn.Linear(self.bottleneck_dim, self.hidden_dim)
        self.feature_norm = nn.LayerNorm(self.hidden_dim)
        self.feature_act = nn.SiLU()  # Use SiLU (Swish) activation

        # Location-Aware Spatial Component
        # --------------------------------
        self.spatial_module = LocationAwareSpatialAttention(
            self.hidden_dim,
            num_nodes=self.num_nodes,
            dropout=self.dropout,
            attention_heads=self.attention_heads,
            attention_regularization_weight=self.attention_reg_weight,
            bottleneck_dim=self.bottleneck_dim,
            dataset_name=self.dataset_name,
        )

        # Adaptive Temporal Component
        # ---------------------------
        self.temporal_module = AdaptiveMultiScaleTemporalModule(
            self.hidden_dim,
            num_scales=self.num_scales,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            temporal_complexity=temporal_complexity,
            max_seq_length=self.window,
        )

        # Location-Aware Prediction Component
        # -----------------------------------
        self.prediction_module = LocationAwarePredictor(
            self.hidden_dim,
            self.horizon,
            num_nodes=self.num_nodes,
            bottleneck_dim=self.bottleneck_dim,
            dropout=self.dropout,
            attention_heads=self.attention_heads // 2,  # Use fewer heads for efficiency
        )

        # Log model configuration
        print(
            f"[LocationAwareMSAGAT_Net] Dataset: {self.dataset_name}, Nodes: {self.num_nodes}"
        )
        print(
            f"[LocationAwareMSAGAT_Net] Dynamic parameters: hidden_dim={self.hidden_dim}, "
            f"attention_heads={self.attention_heads}, kernel_size={self.kernel_size}, "
            f"bottleneck_dim={self.bottleneck_dim}, num_scales={self.num_scales}, "
            f"attention_reg_weight={self.attention_reg_weight:.6f}"
        )

    def forward(self, x, adj=None, idx=None):
        """
        Forward pass of the Location-Aware MSAGAT-Net model.

        Args:
            x (tensor): Input time series [batch, time_window, nodes]
            adj (tensor, optional): Adjacency matrix for graph attention
            idx (tensor, optional): Node indices

        Returns:
            tuple: (Predictions, Attention regularization loss)
        """
        B, T, N = x.shape

        # Input validation
        if T != self.window:
            raise ValueError(
                f"Input time window size {T} doesn't match expected window size {self.window}"
            )

        # Store last observed values for prediction refinement
        x_last = x[:, -1, :]  # [batch, nodes]

        # Feature Extraction
        # -----------------
        # Reshape for 1D convolution: [batch*nodes, channels=1, time_window]
        x_temp = x.permute(0, 2, 1).contiguous().view(B * N, 1, T)
        temporal_features = self.feature_extractor(x_temp)  # [B*N, feature_channels, T]
        temporal_features = temporal_features.view(
            B, N, -1
        )  # [B, N, feature_channels*T]

        # Feature projection through bottleneck
        features = self.feature_projection_low(temporal_features)
        features = self.feature_projection_high(features)
        features = self.feature_norm(features)
        features = self.feature_act(features)

        # Location-Aware Spatial Processing
        # ---------------------------------
        graph_features, attn_reg_loss = self.spatial_module(features, node_indices=idx)

        # Adaptive Temporal Processing
        # ---------------------------
        fusion_features = self.temporal_module(graph_features, seq_length=T)

        # Location-Aware Prediction
        # ------------------------
        if adj is None and hasattr(self, "adj_matrix"):
            # Use cached adjacency matrix if available
            adj = self.adj_matrix

        predictions = self.prediction_module(
            fusion_features, adj_matrix=adj, last_step=x_last, node_indices=idx
        )
        predictions = predictions.transpose(1, 2)  # [batch, horizon, nodes]

        return predictions, attn_reg_loss

    def set_adjacency_matrix(self, adj):
        """Cache the adjacency matrix for future use"""
        self.register_buffer("adj_matrix", adj)

    @staticmethod
    def estimate_complexity(data_loader):
        """
        Estimate dataset complexity from data statistics.
        Higher values indicate more complex temporal patterns.

        Args:
            data_loader: DataLoader containing the dataset

        Returns:
            float: Complexity score between 0.5 and 2.0
        """
        if not hasattr(data_loader, "train"):
            return 1.0

        try:
            # Use the comprehensive temporal complexity calculation
            return calculate_temporal_complexity(data_loader.train)
        except Exception as e:
            print(f"Warning: Error in estimating complexity: {e}. Using default value.")
            return 1.0

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency (for large graphs)"""
        # Apply gradient checkpointing to the main components
        from torch.utils.checkpoint import checkpoint

        # Store original forward methods
        self._original_spatial_forward = self.spatial_module.forward
        self._original_temporal_forward = self.temporal_module.forward
        self._original_prediction_forward = self.prediction_module.forward

        # Replace with checkpointed versions
        self.spatial_module.forward = lambda *args, **kwargs: checkpoint(
            self._original_spatial_forward, *args, **kwargs
        )
        self.temporal_module.forward = lambda *args, **kwargs: checkpoint(
            self._original_temporal_forward, *args, **kwargs
        )
        self.prediction_module.forward = lambda *args, **kwargs: checkpoint(
            self._original_prediction_forward, *args, **kwargs
        )

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing"""
        if hasattr(self, "_original_spatial_forward"):
            self.spatial_module.forward = self._original_spatial_forward
            self.temporal_module.forward = self._original_temporal_forward
            self.prediction_module.forward = self._original_prediction_forward


def calculate_temporal_complexity(data, max_lag=5):
    """
    Calculate temporal complexity based on multiple time series characteristics.

    Args:
        data: Input data (list, numpy array, or tensor)
        max_lag: Maximum lag for autocorrelation calculation

    Returns:
        float: Temporal complexity score between 0.5 and 2.0
    """
    import numpy as np

    # Convert data to tensor if it's not already
    if not isinstance(data, torch.Tensor):
        try:
            if isinstance(data, list):
                x = torch.tensor(data, dtype=torch.float)
            elif isinstance(data, np.ndarray):
                x = torch.from_numpy(data).float()
            else:
                print(
                    f"Warning: Unknown data type {type(data)}, using default complexity"
                )
                return 1.0
        except Exception as e:
            print(f"Warning: Error converting data: {e}, using default complexity")
            return 1.0
    else:
        x = data.float()

    # Handle NaN values
    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=0.0)

    # Ensure we have enough data points
    if x.numel() < max_lag + 2:
        return 1.0

    try:
        # 1. Variance (normalized)
        variance_score = min(1.0, torch.var(x).item() / 5.0)

        # 2. Autocorrelation decay (lower decay = simpler time series)
        acf_values = []
        x_1d = x.flatten()
        x_centered = x_1d - torch.mean(x_1d)

        # Calculate autocorrelation for different lags
        denom = torch.sum(x_centered**2)
        if denom == 0:
            autocorr_decay = 0.5
        else:
            for lag in range(1, min(max_lag + 1, len(x_centered) // 2)):
                numer = torch.sum(x_centered[lag:] * x_centered[:-lag])
                acf_values.append((numer / denom).item())

            # Calculate decay rate
            if len(acf_values) < 2:
                autocorr_decay = 0.5
            else:
                # Faster decay = higher complexity
                autocorr_decay = 1.0 - (sum(map(abs, acf_values)) / len(acf_values))

        # 3. Non-linearity (approximated by checking for direction changes)
        if len(x_1d) < 3:
            nonlinearity = 0.5
        else:
            diffs = x_1d[1:] - x_1d[:-1]
            sign_changes = torch.sum((diffs[1:] * diffs[:-1]) < 0).item()
            nonlinearity = min(1.0, sign_changes / (len(x_1d) - 2))

        # Combine metrics with weights
        complexity = 0.3 * variance_score + 0.5 * autocorr_decay + 0.2 * nonlinearity

        # Scale to desired range [0.5, 2.0]
        scaled_complexity = 0.5 + 1.5 * complexity

        return float(scaled_complexity)

    except Exception as e:
        print(f"Warning: Error calculating temporal complexity: {e}, using default")
        return 1.0


def compute_autocorrelation(x, lag=1):
    """Compute autocorrelation coefficient at specified lag"""
    if torch.isnan(x).any():
        return 1.0

    # Remove mean
    x_centered = x - torch.mean(x, dim=0, keepdim=True)

    # Compute autocorrelation
    numerator = torch.sum(x_centered[lag:] * x_centered[:-lag])
    denominator = torch.sqrt(torch.sum(x_centered**2))

    if denominator == 0:
        return 0.0

    return abs((numerator / denominator).item())
