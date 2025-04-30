import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class TemporalFeatureEncoder(nn.Module):
    """
    Encodes raw time series data into meaningful feature representations
    by applying 1D convolution to capture local temporal patterns.

    Args:
        in_channels (int): Number of input channels (typically 1 for univariate time series)
        feature_channels (int): Number of feature channels to extract
        kernel_size (int): Size of the convolutional kernel
        dropout (float): Dropout rate for regularization
    """

    def __init__(self, in_channels=1, feature_channels=16, kernel_size=3, dropout=0.2):
        super(TemporalFeatureEncoder, self).__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.batch_norm = nn.BatchNorm1d(feature_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of temporal feature encoder.

        Args:
            x (tensor): Input time series [batch*nodes, channels, time_window]

        Returns:
            tensor: Encoded features [batch*nodes, feature_channels, time_window]
        """
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class DynamicGraphGenerator(nn.Module):
    """
    Dynamically generates graph adjacency matrices from node features,
    allowing the model to infer relationships beyond static topology.

    Inspired by graph learning in EpiGNN but implemented with a simpler approach.

    Args:
        hidden_dim (int): Dimensionality of node features
        projection_dim (int): Dimensionality for edge projection
        dropout (float): Dropout probability
    """

    def __init__(self, hidden_dim, projection_dim=None, dropout=0.2):
        super(DynamicGraphGenerator, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection_dim = projection_dim or hidden_dim // 2

        # Projections for source and destination nodes
        self.src_projection = nn.Linear(hidden_dim, self.projection_dim)
        self.dst_projection = nn.Linear(hidden_dim, self.projection_dim)

        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        # Edge scaling parameter
        self.edge_scale = nn.Parameter(torch.ones(1))

    def forward(self, node_features):
        """
        Generate adjacency matrix from node features.

        Args:
            node_features (tensor): Node features [batch, nodes, hidden_dim]

        Returns:
            tensor: Generated adjacency matrix [batch, nodes, nodes]
        """
        batch_size, num_nodes = node_features.size(0), node_features.size(1)

        # Project features for source and destination representation
        src_embedding = self.activation(self.src_projection(node_features))
        dst_embedding = self.activation(self.dst_projection(node_features))

        # Apply dropout
        src_embedding = self.dropout(src_embedding)
        dst_embedding = self.dropout(dst_embedding)

        # Compute edge weights through dot product
        adjacency = torch.bmm(src_embedding, dst_embedding.transpose(1, 2))

        # Scale and normalize
        adjacency = adjacency * self.edge_scale
        adjacency = F.normalize(adjacency, p=2, dim=-1)

        # Add self-connections
        identity = (
            torch.eye(num_nodes, device=node_features.device)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        adjacency = adjacency + identity

        return adjacency


class RelationalAttention(nn.Module):
    """
    Unified module for capturing both spatial and temporal dependencies
    through multi-head attention mechanism.

    Args:
        embed_dim (int): Dimension of node embeddings
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """

    def __init__(self, embed_dim, num_heads=4, dropout=0.2):
        super(RelationalAttention, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adjacency=None):
        """
        Process node features with attention, optionally using adjacency as mask.

        Args:
            x (tensor): Node features [batch, nodes, embed_dim]
            adjacency (tensor, optional): Adjacency matrix [batch, nodes, nodes]

        Returns:
            tensor: Updated node features [batch, nodes, embed_dim]
        """
        # Store input for residual connection
        residual = x
        batch_size, num_nodes = x.shape[0], x.shape[1]

        # Create attention mask from adjacency if provided
        attn_mask = None
        if adjacency is not None:
            # Convert adjacency to boolean mask
            # In PyTorch 1.9+, MultiheadAttention with batch_first=True expects
            # attn_mask to be (batch_size*num_heads, tgt_len, src_len)
            # We'll convert our mask to the right format

            # First convert to binary mask (1 = keep, 0 = mask)
            mask = (adjacency > 0).float()

            # For additive mask format: 0 = keep, -inf = mask
            # Invert and set masked positions to -inf
            mask = (1.0 - mask) * -1e9

            # Handle head dimension by just using a regular mask
            # which will be broadcasted to all heads
            attn_mask = mask

            # Convert to correct format for additive mask (not key_padding_mask)
            # No head dimension needed - nn.MultiheadAttention handles this internally

        # Apply multi-head attention
        # Since we're using additive style masks, set need_weights=False
        x, _ = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=None,  # Don't use the built-in mask handling
            key_padding_mask=None,
            need_weights=False,
        )

        # Apply custom masking using adjacency matrix
        if adjacency is not None:
            # Create custom attention matrix by simple softmax on node pairs
            # This isn't exactly the same as multi-head attention but provides a practical workaround
            attn_weights = torch.bmm(x, x.transpose(1, 2)) / math.sqrt(x.size(-1))

            # Apply mask
            mask = (adjacency <= 0).float() * -1e9
            attn_weights = attn_weights + mask

            # Apply softmax
            attn_weights = F.softmax(attn_weights, dim=-1)

            # Apply attention
            x = torch.bmm(attn_weights, x)

        # Apply dropout and residual connection
        x = self.dropout(x)
        x = self.layer_norm(x + residual)

        return x


class AutoregressiveConnector(nn.Module):
    """
    Direct pathway from input to output that incorporates autoregressive patterns,
    particularly effective for small graphs with strong temporal dependencies.

    Inspired by highway connections in EpiGNN.

    Args:
        window_size (int): Size of the lookback window
        hidden_dim (int): Hidden dimension size for intermediate projection
        dropout (float): Dropout probability
    """

    def __init__(self, window_size, hidden_dim=None, dropout=0.2):
        super(AutoregressiveConnector, self).__init__()

        self.window_size = window_size
        hidden_dim = hidden_dim or window_size // 2

        self.projector = nn.Sequential(
            nn.Linear(window_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        """
        Apply autoregressive connection to input time series.

        Args:
            x (tensor): Input time series [batch, time_window, nodes]

        Returns:
            tensor: Direct projection output [batch, nodes]
        """
        # Extract recent history (last window_size values)
        recent = x[:, -self.window_size :, :]

        # Transpose for per-node processing
        recent = recent.permute(0, 2, 1)  # [batch, nodes, window]

        # Reshape and apply projection
        batch_size, num_nodes = recent.size(0), recent.size(1)
        recent_flat = recent.reshape(-1, self.window_size)
        output_flat = self.projector(recent_flat)

        # Reshape back to [batch, nodes]
        output = output_flat.view(batch_size, num_nodes)

        return output


class HorizonPredictor(nn.Module):
    """
    Generates predictions for future time steps based on node embeddings.

    Args:
        hidden_dim (int): Input embedding dimension
        horizon (int): Number of future time steps to predict
        activation (bool): Whether to apply activation to output
    """

    def __init__(self, hidden_dim, horizon, activation=False):
        super(HorizonPredictor, self).__init__()

        self.predictor = nn.Linear(hidden_dim, horizon)
        self.use_activation = activation
        self.activation = nn.ReLU() if activation else None

    def forward(self, x):
        """
        Generate predictions for future time steps.

        Args:
            x (tensor): Node embeddings [batch, nodes, hidden_dim]

        Returns:
            tensor: Predictions [batch, nodes, horizon]
        """
        predictions = self.predictor(x)

        if self.use_activation:
            predictions = self.activation(predictions)

        return predictions


class DynaGraphNet(nn.Module):
    """
    Dynamic Graph Network for spatio-temporal forecasting.

    Combines dynamic graph learning with attention-based processing for efficient
    modeling of both spatial and temporal dependencies in time series data.

    Key components:
    1. Temporal feature encoding through 1D convolution
    2. Dynamic graph structure generation based on node features
    3. Unified relational attention for spatio-temporal dependencies
    4. Optional autoregressive connection for small graphs
    5. Horizon prediction for future time steps

    Args:
        args: Configuration arguments
        data: Data object containing dataset information
    """

    def __init__(self, args, data):
        super(DynaGraphNet, self).__init__()

        # Core parameters
        self.num_nodes = getattr(data, "m", 100)  # Default to 100 if not specified
        self.window = getattr(args, "window", 12)
        self.horizon = getattr(args, "horizon", 12)

        # Model dimensions
        self.hidden_dim = getattr(args, "hidden_dim", 32)
        self.feature_channels = getattr(args, "feature_channels", 16)
        self.attention_heads = getattr(args, "attention_heads", 4)
        self.dropout_rate = getattr(args, "dropout", 0.2)

        # Check model configuration
        assert self.window > 0, "Window size must be positive"
        assert self.horizon > 0, "Prediction horizon must be positive"
        assert (
            self.hidden_dim % self.attention_heads == 0
        ), "Hidden dimension must be divisible by attention heads"

        # 1. Temporal Feature Encoding
        self.temporal_encoder = TemporalFeatureEncoder(
            in_channels=1,
            feature_channels=self.feature_channels,
            kernel_size=getattr(args, "kernel_size", 3),
            dropout=self.dropout_rate,
        )

        # Feature projection after temporal encoding
        self.feature_projector = nn.Sequential(
            nn.Linear(self.feature_channels * self.window, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
        )

        # 2. Dynamic Graph Generation
        self.graph_generator = DynamicGraphGenerator(
            hidden_dim=self.hidden_dim,
            projection_dim=self.hidden_dim // 2,
            dropout=self.dropout_rate,
        )

        # 3. Relational Attention Layers
        self.num_layers = getattr(args, "num_layers", 1)
        self.attention_layers = nn.ModuleList(
            [
                RelationalAttention(
                    embed_dim=self.hidden_dim,
                    num_heads=self.attention_heads,
                    dropout=self.dropout_rate,
                )
                for _ in range(self.num_layers)
            ]
        )

        # 4. Autoregressive Connection (optional)
        self.use_autoregressive = getattr(args, "autoregressive", False)
        if self.use_autoregressive:
            self.autoregressive = AutoregressiveConnector(
                window_size=min(self.window, 10),  # Use at most 10 steps
                hidden_dim=self.hidden_dim // 2,
                dropout=self.dropout_rate,
            )

        # 5. Horizon Prediction
        self.horizon_predictor = HorizonPredictor(
            hidden_dim=self.hidden_dim, horizon=self.horizon
        )

        # Initialize weights
        self.apply(self._init_weights)

        # Report model configuration
        print(
            f"DynaGraphNet: nodes={self.num_nodes}, "
            f"window={self.window}, horizon={self.horizon}, "
            f"hidden_dim={self.hidden_dim}, layers={self.num_layers}"
        )

    def _init_weights(self, module):
        """Initialize weights for better convergence"""
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

    def forward(self, x, index=None):
        """
        Forward pass of DynaGraphNet.

        Args:
            x (tensor): Input time series [batch, time_window, nodes]
            index (tensor, optional): Batch indices for reference

        Returns:
            tuple: (Predictions [batch, horizon, nodes], regularization loss)
        """
        batch_size, seq_len, num_nodes = x.shape

        # Validate input dimensions
        if seq_len != self.window:
            raise ValueError(f"Expected sequence length {self.window}, got {seq_len}")

        # Store last observed values for autoregressive connection
        x_last = x[:, -1, :]  # [batch, nodes]

        # 1. Extract temporal features
        # Reshape for 1D convolution [batch*nodes, 1, window]
        x_temp = (
            x.permute(0, 2, 1).contiguous().view(batch_size * num_nodes, 1, seq_len)
        )
        temporal_features = self.temporal_encoder(x_temp)

        # Flatten and reshape back to [batch, nodes, features*window]
        temp_dim = temporal_features.size(1)
        node_features = temporal_features.view(
            batch_size, num_nodes, temp_dim * seq_len
        )

        # Project to hidden dimension
        node_features = self.feature_projector(node_features)

        # 2. Generate dynamic graph structure
        adjacency = self.graph_generator(node_features)

        # Track attention weights for regularization (optional)
        self.attention_weights = adjacency

        # Small regularization loss on adjacency matrix for sparsity
        # This matches what other models are doing with attn_reg_loss
        reg_loss = torch.mean(torch.abs(adjacency)) * 1e-4

        # 3. Process through attention layers
        for attention_layer in self.attention_layers:
            node_features = attention_layer(node_features, adjacency)

        # 4. Generate predictions for future time steps
        predictions = self.horizon_predictor(node_features)  # [batch, nodes, horizon]

        # 5. Apply autoregressive connection if enabled
        if self.use_autoregressive:
            ar_output = self.autoregressive(x)  # [batch, nodes]

            # Add autoregressive output to first prediction step
            predictions[:, :, 0] = predictions[:, :, 0] + ar_output

        # Transpose to match expected output format [batch, horizon, nodes]
        predictions = predictions.transpose(1, 2)

        # Return predictions and regularization loss for API consistency
        return predictions, reg_loss

    def calculate_loss(self, pred, target, criterion=None):
        """
        Calculate loss between predictions and targets.

        Args:
            pred (tensor): Model predictions
            target (tensor): Ground truth values
            criterion (function, optional): Loss function

        Returns:
            tensor: Computed loss
        """
        if criterion is None:
            criterion = nn.MSELoss()

        return criterion(pred, target)
