import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.checkpoint import checkpoint

# Base hyperparameters
BASE_CHANNELS = 32
ATTENTION_HEADS = 4
FUSION_ALPHA = 1e-05
DROPOUT_RATE = 0.2
TEMPORAL_LEVELS = 4
FILTER_SIZE = 3
INPUT_CHANNELS = 16
COMPRESSION_DIM = 8


class GraphInferenceModule(nn.Module):
    """
    Graph Inference Module that dynamically discovers relationships between nodes
    based on their feature representations.

    Args:
        feature_dim (int): Input feature dimension
        projection_dim (int): Internal projection dimension
        dropout (float): Dropout rate for regularization
    """

    def __init__(self, feature_dim, projection_dim=None, dropout=DROPOUT_RATE):
        super(GraphInferenceModule, self).__init__()
        self.feature_dim = feature_dim
        self.projection_dim = projection_dim or feature_dim // 2

        # Affinity projection networks
        self.origin_projection = nn.Linear(feature_dim, self.projection_dim)
        self.target_projection = nn.Linear(feature_dim, self.projection_dim)

        # Post-processing components
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.scaling = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, features):
        """
        Generate graph structure from node features.

        Args:
            features (tensor): Node features [batch, nodes, feature_dim]

        Returns:
            tensor: Inferred adjacency matrix [batch, nodes, nodes]
        """
        batch_size, num_nodes = features.size(0), features.size(1)

        # Project features for affinity calculation
        origin_embed = self.activation(self.origin_projection(features))
        target_embed = self.activation(self.target_projection(features))

        # Apply dropout for regularization
        origin_embed = self.dropout(origin_embed)
        target_embed = self.dropout(target_embed)

        # Calculate affinity matrix
        affinity = torch.bmm(origin_embed, target_embed.transpose(1, 2))

        # Apply scaling and normalization
        affinity = affinity * self.scaling
        affinity = F.normalize(affinity, p=2, dim=-1)

        # Add self-connections
        identity = (
            torch.eye(num_nodes, device=features.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        adjacency = affinity + identity

        return adjacency


class AdaptiveMultiHeadAttention(nn.Module):
    """
    Adaptive Multi-Head Attention with support for multiple graph structures.

    This module fuses information from static and dynamic graph structures
    while applying multi-head attention mechanisms.

    Args:
        feature_dim (int): Feature dimension
        num_nodes (int): Number of nodes in the graph
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
        fusion_weight (float): Weight for attention regularization
        compression_dim (int): Dimension for internal compression
    """

    def __init__(
        self,
        feature_dim,
        num_nodes,
        num_heads=ATTENTION_HEADS,
        dropout=DROPOUT_RATE,
        fusion_weight=FUSION_ALPHA,
        compression_dim=COMPRESSION_DIM,
    ):
        super(AdaptiveMultiHeadAttention, self).__init__()

        # Validate dimensions
        assert (
            feature_dim % num_heads == 0
        ), "feature_dim must be divisible by num_heads"

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.num_nodes = num_nodes
        self.fusion_weight = fusion_weight
        self.compression_dim = compression_dim

        # Efficient parameter compression
        self.qkv_compress = nn.Linear(feature_dim, 3 * compression_dim)
        self.qkv_expand = nn.Linear(3 * compression_dim, 3 * feature_dim)

        # Output transformation with compression
        self.output_compress = nn.Linear(feature_dim, compression_dim)
        self.output_expand = nn.Linear(compression_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

        # Low-rank graph bias components
        self.bias_factor1 = Parameter(
            torch.Tensor(num_heads, num_nodes, compression_dim)
        )
        self.bias_factor2 = Parameter(
            torch.Tensor(num_heads, compression_dim, num_nodes)
        )
        nn.init.xavier_uniform_(self.bias_factor1)
        nn.init.xavier_uniform_(self.bias_factor2)

        # Graph fusion parameter
        self.graph_fusion = nn.Parameter(torch.FloatTensor([0.5]))

    def forward(self, features, fixed_graph=None, learned_graph=None):
        """
        Process features using adaptive multi-head attention.

        Args:
            features (tensor): Input features [batch, nodes, feature_dim]
            fixed_graph (tensor, optional): Fixed graph structure
            learned_graph (tensor, optional): Dynamically learned graph

        Returns:
            tuple: (Updated features, Attention regularization value)
        """
        batch_size, num_nodes, feature_dim = features.shape
        device = features.device

        # Save input for residual connection
        residual = features

        # Efficient parameter compression and expansion
        qkv_compressed = self.qkv_compress(features)
        qkv = self.qkv_expand(qkv_compressed)

        # Split into query, key, value
        qkv = qkv.chunk(3, dim=-1)
        query, key, value = [
            x.reshape(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(
                1, 2
            )
            for x in qkv
        ]

        # Calculate attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )

        # Generate structural bias matrix (already head-specific)
        structure_bias = torch.matmul(self.bias_factor1, self.bias_factor2)

        # Add structure bias
        attention_scores = attention_scores + structure_bias

        # Incorporate graph structures if provided
        if fixed_graph is not None or learned_graph is not None:
            # Handle and validate graphs
            try:
                # Ensure graphs are on the same device as features
                if fixed_graph is not None:
                    fixed_graph = fixed_graph.to(device)

                    # Check dimensions
                    if (
                        fixed_graph.dim() == 2
                        and fixed_graph.size(0) == num_nodes
                        and fixed_graph.size(1) == num_nodes
                    ):
                        # Convert to batch format
                        fixed_graph = fixed_graph.unsqueeze(0).expand(
                            batch_size, -1, -1
                        )

                    # Ensure it matches the expected dimensions
                    if (
                        fixed_graph.size(1) != num_nodes
                        or fixed_graph.size(2) != num_nodes
                    ):
                        print(
                            f"Warning: Fixed graph dimensions {fixed_graph.size()} don't match feature dimensions {num_nodes}. Ignoring fixed graph."
                        )
                        fixed_graph = None

                if learned_graph is not None:
                    learned_graph = learned_graph.to(device)

                    # Check and reshape if needed
                    if learned_graph.dim() == 3 and learned_graph.size(1) != num_nodes:
                        print(
                            f"Warning: Learned graph dimensions {learned_graph.size()} don't match feature dimensions {num_nodes}. Ignoring learned graph."
                        )
                        learned_graph = None

                # Prepare combined graph
                combined_graph = None

                if fixed_graph is not None and learned_graph is not None:
                    # Adaptive fusion of fixed and learned graphs
                    fusion_ratio = torch.sigmoid(self.graph_fusion)
                    combined_graph = (
                        fusion_ratio * learned_graph + (1 - fusion_ratio) * fixed_graph
                    )
                elif fixed_graph is not None:
                    combined_graph = fixed_graph
                elif learned_graph is not None:
                    combined_graph = learned_graph

                # Apply graph as bias if available
                if combined_graph is not None:
                    # Expand to match attention heads dimensions [batch, heads, nodes, nodes]
                    if combined_graph.dim() == 3:  # [batch, nodes, nodes]
                        combined_graph = combined_graph.unsqueeze(1)
                        combined_graph = combined_graph.expand(
                            -1, self.num_heads, -1, -1
                        )

                    # Add graph as bias to attention scores
                    attention_scores = attention_scores + combined_graph
            except Exception as e:
                print(
                    f"Error applying graph structures: {e}. Continuing without graph bias."
                )

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Store for regularization
        self.attention = attention_weights

        # Apply dropout
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).reshape(batch_size, num_nodes, feature_dim)

        # Apply efficient compression/expansion for output
        output = self.output_compress(context)
        output = self.output_expand(output)

        # Add residual connection and normalize
        output = self.norm(output + residual)

        # Calculate regularization loss
        reg_loss = self.fusion_weight * torch.mean(torch.abs(self.attention))

        return output, reg_loss


class MultiScaleTemporalEncoder(nn.Module):
    """
    Multi-Scale Temporal Encoder that captures patterns at different time scales.

    Uses dilated convolutions at multiple rates to capture both short and
    long-range temporal dependencies with an adaptive fusion mechanism.

    Args:
        feature_dim (int): Feature dimension
        num_levels (int): Number of temporal scales
        filter_size (int): Size of convolutional filters
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        feature_dim,
        num_levels=TEMPORAL_LEVELS,
        filter_size=FILTER_SIZE,
        dropout=DROPOUT_RATE,
    ):
        super(MultiScaleTemporalEncoder, self).__init__()
        self.feature_dim = feature_dim
        self.num_levels = num_levels

        # Create multiple temporal levels with increasing receptive fields
        self.levels = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        feature_dim,
                        feature_dim,
                        kernel_size=filter_size,
                        padding=(filter_size // 2) * 2**i,
                        dilation=2**i,
                    ),
                    nn.BatchNorm1d(feature_dim),
                    nn.SiLU(),  # SiLU/Swish activation for better gradient properties
                    nn.Dropout(dropout),
                )
                for i in range(num_levels)
            ]
        )

        # Adaptive fusion weights
        self.fusion_weights = Parameter(torch.ones(num_levels), requires_grad=True)

        # Efficient compression for fusion
        self.compress = nn.Linear(feature_dim, COMPRESSION_DIM)
        self.expand = nn.Linear(COMPRESSION_DIM, feature_dim)

        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features):
        """
        Process features through multiple temporal scales.

        Args:
            features (tensor): Input features [batch, nodes, feature_dim]

        Returns:
            tensor: Temporally processed features
        """
        # Save for residual connection
        residual = features

        # Prepare for 1D convolution
        features = features.transpose(1, 2)  # [batch, feature_dim, nodes]

        # Process at different temporal scales
        multi_scale_features = [level(features) for level in self.levels]

        # Calculate adaptive fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)

        # Stack and weight the multi-scale features
        stacked = torch.stack(
            multi_scale_features, dim=1
        )  # [batch, scales, feature_dim, nodes]
        fused = torch.sum(weights.view(1, self.num_levels, 1, 1) * stacked, dim=1)

        # Reshape back
        fused = fused.transpose(1, 2)  # [batch, nodes, feature_dim]

        # Apply efficient compression/expansion
        output = self.compress(fused)
        output = self.expand(output)

        # Add residual connection and normalize
        output = self.norm(output + residual)

        return output


class HorizonForecastModule(nn.Module):
    """
    Horizon Forecast Module that predicts multiple future time steps.

    Features an adaptive blend of model predictions and exponential decay
    based on the last observed values.

    Args:
        feature_dim (int): Feature dimension
        horizon (int): Number of future time steps to predict
        compression_dim (int): Dimension for internal compression
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        feature_dim,
        horizon,
        compression_dim=COMPRESSION_DIM,
        dropout=DROPOUT_RATE,
    ):
        super(HorizonForecastModule, self).__init__()
        self.feature_dim = feature_dim
        self.horizon = horizon
        self.compression_dim = compression_dim

        # Multi-stage prediction pathway
        self.compress = nn.Linear(feature_dim, compression_dim)
        self.process = nn.Sequential(
            nn.LayerNorm(compression_dim), nn.SiLU(), nn.Dropout(dropout)
        )
        self.forecast = nn.Linear(compression_dim, horizon)

        # Adaptive blending mechanism
        self.blend_gate = nn.Sequential(
            nn.Linear(feature_dim, compression_dim),
            nn.SiLU(),
            nn.Linear(compression_dim, horizon),
            nn.Sigmoid(),
        )

    def forward(self, features, last_values=None):
        """
        Generate forecasts for future time steps.

        Args:
            features (tensor): Node features [batch, nodes, feature_dim]
            last_values (tensor, optional): Last observed values [batch, nodes]

        Returns:
            tensor: Forecasts for future time steps [batch, nodes, horizon]
        """
        # Generate model predictions
        compressed = self.compress(features)
        processed = self.process(compressed)
        forecasts = self.forecast(processed)

        # Apply adaptive blending if last values are provided
        if last_values is not None:
            # Calculate blend ratio
            blend_ratio = self.blend_gate(features)

            # Prepare last values with exponential decay
            last_values = last_values.unsqueeze(-1)
            decay_factor = (
                torch.arange(1, self.horizon + 1, device=features.device)
                .float()
                .view(1, 1, self.horizon)
            )
            trend_forecast = last_values * torch.exp(-0.1 * decay_factor)

            # Blend model and trend forecasts
            forecasts = blend_ratio * forecasts + (1 - blend_ratio) * trend_forecast

        return forecasts


class EfficientFeatureExtractor(nn.Module):
    """
    Efficient Feature Extractor using depthwise separable convolutions.

    Significantly reduces parameter count while maintaining expressiveness.

    Args:
        in_channels (int): Input channels
        out_channels (int): Output channels
        kernel_size (int): Convolution kernel size
        dropout (float): Dropout rate
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=INPUT_CHANNELS,
        kernel_size=FILTER_SIZE,
        dropout=DROPOUT_RATE,
    ):
        super(EfficientFeatureExtractor, self).__init__()

        # Depthwise convolution (acts on each channel separately)
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
        )

        # Pointwise convolution (1x1 conv for channel mixing)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        # Normalization and activation
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Extract features from input time series.

        Args:
            x (tensor): Input data [batch*nodes, in_channels, time_steps]

        Returns:
            tensor: Extracted features [batch*nodes, out_channels, time_steps]
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DirectPathway(nn.Module):
    """
    Direct Pathway that provides a shortcut from input to output.

    Particularly effective for small graphs with strong autoregressive patterns.

    Args:
        window_size (int): Size of the input time window
        dropout (float): Dropout rate
    """

    def __init__(self, window_size, hidden_dim=None, dropout=DROPOUT_RATE):
        super(DirectPathway, self).__init__()
        self.window_size = window_size
        hidden_dim = hidden_dim or max(4, window_size // 2)

        # Two-stage projection for better expressivity
        self.project = nn.Sequential(
            nn.Linear(window_size, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        Process input time series directly to output.

        Args:
            x (tensor): Input time series [batch, time_steps, nodes]

        Returns:
            tensor: Direct outputs [batch, nodes]
        """
        # Extract recent history
        recent = x[:, -self.window_size :, :]

        # Reshape for processing each node separately
        batch_size, _, num_nodes = recent.shape
        recent = recent.permute(0, 2, 1)  # [batch, nodes, window]
        recent = recent.reshape(-1, self.window_size)

        # Project to output
        output = self.project(recent).view(batch_size, num_nodes)

        return output


class AFGNet(nn.Module):
    """
    Adaptive Fusion Graph Network (AFGNet) for spatio-temporal forecasting.

    Combines dynamic graph learning, multi-scale temporal encoding,
    and adaptive fusion mechanisms for effective forecasting on graphs
    of various sizes.

    Key features:
    - Efficient feature extraction
    - Dynamic graph structure inference
    - Adaptive multi-head attention with graph fusion
    - Multi-scale temporal pattern encoding
    - Direct autoregressive pathway for small graphs
    - Memory-efficient implementation

    Args:
        args: Configuration arguments
        data: Data object containing dataset information
    """

    def __init__(self, args, data):
        super(AFGNet, self).__init__()
        # Core parameters
        self.num_nodes = data.m
        self.window = args.window
        self.horizon = args.horizon

        # Model dimensions
        self.feature_dim = getattr(args, "hidden_dim", BASE_CHANNELS)
        self.num_heads = getattr(args, "attention_heads", ATTENTION_HEADS)
        self.filter_size = getattr(args, "kernel_size", FILTER_SIZE)
        self.compression_dim = getattr(args, "bottleneck_dim", COMPRESSION_DIM)
        self.dropout_rate = getattr(args, "dropout", DROPOUT_RATE)

        # Memory efficiency option
        self.use_checkpointing = getattr(args, "use_checkpointing", False)

        # Static graph structure if available
        self.use_static_graph = getattr(args, "use_adjacency", False)
        self.static_graph = (
            getattr(data, "adj", None) if self.use_static_graph else None
        )

        # 1. Feature Extraction
        self.feature_extractor = EfficientFeatureExtractor(
            in_channels=1,
            out_channels=INPUT_CHANNELS,
            kernel_size=self.filter_size,
            dropout=self.dropout_rate,
        )

        # Feature transformation from extraction to model dimension
        self.feature_transform = nn.Sequential(
            nn.Linear(INPUT_CHANNELS * self.window, self.compression_dim),
            nn.LayerNorm(self.compression_dim),
            nn.SiLU(),
            nn.Linear(self.compression_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(self.dropout_rate),
        )

        # 2. Graph Inference
        self.graph_inference = GraphInferenceModule(
            feature_dim=self.feature_dim,
            projection_dim=self.compression_dim,
            dropout=self.dropout_rate,
        )

        # 3. Adaptive Multi-Head Attention
        self.attention = AdaptiveMultiHeadAttention(
            feature_dim=self.feature_dim,
            num_nodes=self.num_nodes,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            fusion_weight=FUSION_ALPHA,
            compression_dim=self.compression_dim,
        )

        # 4. Multi-Scale Temporal Encoding
        self.temporal_encoder = MultiScaleTemporalEncoder(
            feature_dim=self.feature_dim,
            num_levels=getattr(args, "num_scales", TEMPORAL_LEVELS),
            filter_size=self.filter_size,
            dropout=self.dropout_rate,
        )

        # 5. Direct Pathway (for small graphs)
        self.use_direct_pathway = getattr(args, "autoregressive", False)
        if self.use_direct_pathway:
            self.direct_pathway = DirectPathway(
                window_size=min(self.window, 10),
                hidden_dim=self.compression_dim,
                dropout=self.dropout_rate,
            )

        # 6. Horizon Forecasting
        self.forecaster = HorizonForecastModule(
            feature_dim=self.feature_dim,
            horizon=self.horizon,
            compression_dim=self.compression_dim,
            dropout=self.dropout_rate,
        )

        # Initialize weights for better convergence
        self.apply(self._init_weights)

        # Log model configuration
        print(
            f"AFGNet: nodes={self.num_nodes}, window={self.window}, "
            f"horizon={self.horizon}, feature_dim={self.feature_dim}"
        )

    def _init_weights(self, module):
        """Initialize weights with optimized values for better training"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, external_graph=None, node_indices=None):
        """
        Forward pass of AFGNet.

        Args:
            x (tensor): Input time series [batch, time_steps, nodes]
            external_graph (tensor, optional): External graph structure
            node_indices (tensor, optional): Selected node indices

        Returns:
            tuple: (Forecasts, Attention regularization loss)
        """
        batch_size, time_steps, num_nodes = x.shape

        # Store last observed values for forecasting
        last_values = x[:, -1, :]  # [batch, nodes]

        # STAGE 1: FEATURE EXTRACTION
        # --------------------------
        # Reshape for temporal convolution
        x_reshaped = (
            x.permute(0, 2, 1).contiguous().view(batch_size * num_nodes, 1, time_steps)
        )
        temporal_features = self.feature_extractor(x_reshaped)

        # Reshape and transform features
        embed_dim = temporal_features.size(1)
        node_features = temporal_features.view(
            batch_size, num_nodes, embed_dim * time_steps
        )
        node_features = self.feature_transform(node_features)

        # STAGE 2: GRAPH INFERENCE & FUSION
        # --------------------------------
        # Determine which graph structures to use
        if self.use_checkpointing and self.training:
            inferred_graph = checkpoint(self.graph_inference, node_features)
        else:
            inferred_graph = self.graph_inference(node_features)

        # Select appropriate static graph
        static_graph = (
            external_graph if external_graph is not None else self.static_graph
        )

        # STAGE 3: SPATIO-TEMPORAL PROCESSING
        # ----------------------------------
        # Apply attention with graph structures
        if self.use_checkpointing and self.training:
            enhanced_features, attn_reg_loss = checkpoint(
                self.attention, node_features, static_graph, inferred_graph
            )
            temporal_features = checkpoint(self.temporal_encoder, enhanced_features)
        else:
            enhanced_features, attn_reg_loss = self.attention(
                node_features, static_graph, inferred_graph
            )
            temporal_features = self.temporal_encoder(enhanced_features)

        # STAGE 4: FORECASTING
        # -------------------
        # Generate forecasts
        forecasts = self.forecaster(temporal_features, last_values)

        # Apply direct pathway for small graphs if enabled
        if self.use_direct_pathway:
            direct_output = self.direct_pathway(x)
            # Add to first forecast step
            forecasts[:, :, 0] = forecasts[:, :, 0] + direct_output

        # Transpose for standard output format [batch, horizon, nodes]
        forecasts = forecasts.transpose(1, 2)

        return forecasts, attn_reg_loss
        # Return the final forecasts and attention regularization loss
