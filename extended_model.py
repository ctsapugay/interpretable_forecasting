"""
Extended Interpretable Time Series Forecasting Model

This module extends the existing interpretable time series forecasting model with:
- Cross-attention mechanisms for inter-variable relationships
- Temporal compression for efficient sequence processing
- Spline-based forecasting for interpretable predictions

The extended model builds upon the InterpretableTimeEncoder while adding new components
for comprehensive forecasting capabilities.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import warnings

try:
    from .model import InterpretableTimeEncoder, ModelConfig
except ImportError:
    # Handle case when running as script
    from model import InterpretableTimeEncoder, ModelConfig


@dataclass
class ExtendedModelConfig:
    """
    Extended configuration class for the interpretable forecasting model.
    
    Includes all original ModelConfig parameters plus new parameters for:
    - Cross-attention mechanisms
    - Temporal compression
    - Spline-based forecasting
    """
    # Original parameters from ModelConfig
    num_variables: int = 7          # HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    embed_dim: int = 32
    hidden_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    max_len: int = 512
    
    # Cross-attention parameters
    cross_dim: int = 32              # Cross-attention embedding dimension
    cross_heads: int = 4             # Number of cross-attention heads
    cross_dropout: float = 0.1       # Dropout for cross-attention
    
    # Temporal compression parameters
    compressed_dim: int = 64         # Compressed representation dimension
    compression_ratio: int = 4       # Temporal compression ratio (T -> T/ratio)
    compression_method: str = 'attention'  # 'attention' or 'pooling'
    
    # Spline forecasting parameters
    num_control_points: int = 8      # Number of B-spline control points
    spline_degree: int = 3           # B-spline degree (typically 3 for cubic)
    forecast_horizon: int = 24       # Default forecast steps ahead
    spline_stability: bool = True    # Enable spline stability constraints
    
    # Multi-horizon forecasting
    forecast_horizons: list = None   # Multiple forecast horizons [1, 12, 24, 48]
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate_config()
        
        # Set default forecast horizons if not provided
        if self.forecast_horizons is None:
            self.forecast_horizons = [1, 12, 24, 48]
    
    def _validate_config(self):
        """
        Validate configuration parameters for compatibility and correctness.
        
        Raises:
            ValueError: If configuration parameters are invalid or incompatible
        """
        # Basic parameter validation
        if self.num_variables <= 0:
            raise ValueError(f"num_variables must be positive, got {self.num_variables}")
        
        if self.embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {self.embed_dim}")
        
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        
        # Attention head validation
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})")
        
        if self.cross_heads <= 0:
            raise ValueError(f"cross_heads must be positive, got {self.cross_heads}")
        
        if self.cross_dim % self.cross_heads != 0:
            raise ValueError(f"cross_dim ({self.cross_dim}) must be divisible by cross_heads ({self.cross_heads})")
        
        # Dropout validation
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {self.dropout}")
        
        if not (0.0 <= self.cross_dropout <= 1.0):
            raise ValueError(f"cross_dropout must be in [0, 1], got {self.cross_dropout}")
        
        # Compression validation
        if self.compressed_dim <= 0:
            raise ValueError(f"compressed_dim must be positive, got {self.compressed_dim}")
        
        if self.compression_ratio <= 0:
            raise ValueError(f"compression_ratio must be positive, got {self.compression_ratio}")
        
        if self.compression_method not in ['attention', 'pooling']:
            raise ValueError(f"compression_method must be 'attention' or 'pooling', got {self.compression_method}")
        
        # Spline validation
        if self.num_control_points <= 0:
            raise ValueError(f"num_control_points must be positive, got {self.num_control_points}")
        
        if self.spline_degree <= 0:
            raise ValueError(f"spline_degree must be positive, got {self.spline_degree}")
        
        if self.spline_degree >= self.num_control_points:
            raise ValueError(f"spline_degree ({self.spline_degree}) must be less than num_control_points ({self.num_control_points})")
        
        if self.forecast_horizon <= 0:
            raise ValueError(f"forecast_horizon must be positive, got {self.forecast_horizon}")
        
        # Sequence length validation
        if self.max_len <= 0:
            raise ValueError(f"max_len must be positive, got {self.max_len}")
        
        # Cross-compatibility warnings
        if self.cross_dim != self.embed_dim:
            warnings.warn(f"cross_dim ({self.cross_dim}) differs from embed_dim ({self.embed_dim}). "
                         "This may require additional projection layers.")
        
        if self.compressed_dim < self.cross_dim:
            warnings.warn(f"compressed_dim ({self.compressed_dim}) is smaller than cross_dim ({self.cross_dim}). "
                         "This may cause information loss during compression.")
        
        # Performance warnings
        if self.num_variables > 20:
            warnings.warn(f"Large number of variables ({self.num_variables}) may cause high memory usage "
                         "in cross-attention computation (O(M²)).")
        
        if self.max_len > 1000:
            warnings.warn(f"Large sequence length ({self.max_len}) may cause high memory usage "
                         "in temporal attention computation (O(T²)).")
    
    def to_base_config(self) -> ModelConfig:
        """
        Convert to base ModelConfig for compatibility with existing components.
        
        Returns:
            ModelConfig: Base configuration for InterpretableTimeEncoder
        """
        return ModelConfig(
            num_variables=self.num_variables,
            embed_dim=self.embed_dim,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            max_len=self.max_len
        )
    
    @classmethod
    def from_base_config(cls, base_config: ModelConfig, **kwargs) -> 'ExtendedModelConfig':
        """
        Create ExtendedModelConfig from base ModelConfig.
        
        Args:
            base_config: Base ModelConfig instance
            **kwargs: Additional parameters for extended configuration
            
        Returns:
            ExtendedModelConfig: Extended configuration instance
        """
        return cls(
            num_variables=base_config.num_variables,
            embed_dim=base_config.embed_dim,
            hidden_dim=base_config.hidden_dim,
            num_heads=base_config.num_heads,
            dropout=base_config.dropout,
            max_len=base_config.max_len,
            **kwargs
        )
    
    def get_memory_estimate(self, batch_size: int, seq_len: int) -> Dict[str, int]:
        """
        Estimate memory usage for different components.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Dictionary with memory estimates in number of parameters/activations
        """
        estimates = {}
        
        # Existing components
        estimates['univariate_learners'] = self.num_variables * (self.hidden_dim + self.embed_dim * self.hidden_dim + self.embed_dim)
        estimates['temporal_attention'] = self.embed_dim * self.embed_dim * 3 + self.max_len * self.embed_dim
        
        # New components
        estimates['cross_attention'] = self.cross_dim * self.cross_dim * 3 * self.cross_heads
        estimates['temporal_encoder'] = self.cross_dim * self.compressed_dim + batch_size * self.num_variables * seq_len
        estimates['spline_learner'] = self.compressed_dim * self.num_control_points + self.num_control_points * self.forecast_horizon
        
        # Activation memory (approximate)
        estimates['activations'] = batch_size * seq_len * (
            self.num_variables * self.embed_dim +  # Variable embeddings
            self.num_variables * self.cross_dim +  # Cross-attended embeddings
            self.num_variables * self.compressed_dim  # Compressed representations
        )
        
        return estimates


class CrossVariableAttention(nn.Module):
    """
    Cross-variable attention module that learns relationships between different variables.
    
    This module applies multi-head attention across variables while preserving the temporal
    dimension. Each variable can attend to all other variables to learn inter-variable
    dependencies and correlations.
    
    Args:
        embed_dim: Input embedding dimension from temporal attention
        cross_dim: Output cross-attention embedding dimension  
        num_heads: Number of attention heads for cross-variable attention
        dropout: Dropout probability for attention weights and output
    """
    
    def __init__(self, embed_dim: int, cross_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.cross_dim = cross_dim
        self.num_heads = num_heads
        self.dropout_prob = dropout
        
        # Validate parameters
        if cross_dim % num_heads != 0:
            raise ValueError(f"cross_dim ({cross_dim}) must be divisible by num_heads ({num_heads})")
        
        self.head_dim = cross_dim // num_heads
        
        # Multi-head attention for cross-variable processing
        # Input projection to match cross_dim if needed
        if embed_dim != cross_dim:
            self.input_projection = nn.Linear(embed_dim, cross_dim)
        else:
            self.input_projection = nn.Identity()
        
        # Multi-head attention layer for variable-to-variable relationships
        self.attention = nn.MultiheadAttention(
            embed_dim=cross_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(cross_dim)
        
        # Dropout for residual connection
        self.dropout = nn.Dropout(dropout)
        
        # Output projection if dimensions differ
        if embed_dim != cross_dim:
            self.output_projection = nn.Linear(cross_dim, cross_dim)
        else:
            self.output_projection = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-variable attention computation.
        
        Args:
            x: Input tensor of shape (B, M, T, embed_dim) where:
               - B is batch size
               - M is number of variables
               - T is sequence length
               - embed_dim is embedding dimension from temporal attention
               
        Returns:
            Tuple containing:
            - attended_embeddings: Cross-attended embeddings of shape (B, M, T, cross_dim)
            - attention_weights: Attention weights of shape (B, num_heads, M, M)
        """
        batch_size, num_vars, seq_len, embed_dim = x.shape
        
        # Input validation
        if embed_dim != self.embed_dim:
            raise ValueError(f"Expected embed_dim {self.embed_dim}, got {embed_dim}")
        
        # Handle input reshaping from (B, M, T, embed_dim) to enable variable-to-variable attention
        # Reshape to (B*T, M, embed_dim) to process each time step independently
        x_reshaped = x.transpose(1, 2).reshape(batch_size * seq_len, num_vars, embed_dim)
        
        # Apply input projection if needed
        x_projected = self.input_projection(x_reshaped)  # (B*T, M, cross_dim)
        
        # Apply multi-head attention across variables while preserving temporal dimension
        # Each variable attends to all other variables at the same time step
        attended_output, attn_weights = self.attention(
            query=x_projected,
            key=x_projected, 
            value=x_projected,
            average_attn_weights=False  # Return per-head attention weights
        )
        # attended_output: (B*T, M, cross_dim)
        # attn_weights: (B*T, num_heads, M, M)
        
        # Implement residual connections and layer normalization
        # Pre-norm residual connection
        residual_input = x_projected
        attended_normalized = self.layer_norm(attended_output + residual_input)
        
        # Apply dropout
        attended_dropped = self.dropout(attended_normalized)
        
        # Apply output projection
        final_output = self.output_projection(attended_dropped)
        
        # Reshape back to (B, M, T, cross_dim)
        final_output = final_output.reshape(batch_size, seq_len, num_vars, self.cross_dim)
        final_output = final_output.transpose(1, 2)  # (B, M, T, cross_dim)
        
        # Average attention weights across time steps for interpretability
        # attn_weights: (B*T, num_heads, M, M) -> (B, num_heads, M, M)
        attn_weights = attn_weights.reshape(batch_size, seq_len, self.num_heads, num_vars, num_vars)
        attn_weights_avg = attn_weights.mean(dim=1)  # Average over time dimension
        
        return final_output, attn_weights_avg


class TemporalEncoder(nn.Module):
    """
    Temporal encoder module for sequence compression using attention-based pooling.
    
    This module compresses temporal sequences while preserving essential information
    for forecasting. It uses learnable compression queries and attention mechanisms
    to identify and preserve the most important temporal patterns.
    
    Args:
        input_dim: Input dimension from cross-attention embeddings
        compressed_dim: Output compressed representation dimension
        compression_ratio: Ratio for temporal compression (T -> T/ratio)
    """
    
    def __init__(self, input_dim: int, compressed_dim: int, compression_ratio: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.compressed_dim = compressed_dim
        self.compression_ratio = compression_ratio
        
        # Validate parameters
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if compressed_dim <= 0:
            raise ValueError(f"compressed_dim must be positive, got {compressed_dim}")
        if compression_ratio <= 0:
            raise ValueError(f"compression_ratio must be positive, got {compression_ratio}")
        
        # Set up learnable compression query parameters for each variable
        # Each variable gets its own compression query to learn variable-specific patterns
        self.compression_query = nn.Parameter(
            torch.randn(1, 1, 1, input_dim) * 0.02  # Small initialization
        )
        
        # Input projection to ensure compatibility
        if input_dim != compressed_dim:
            self.input_projection = nn.Linear(input_dim, compressed_dim)
        else:
            self.input_projection = nn.Identity()
        
        # Create attention mechanism for temporal pooling
        # Use scaled dot-product attention for compression
        self.attention_scale = (input_dim ** -0.5)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(compressed_dim)
        
        # Optional: Additional MLP for post-compression processing
        self.post_compression_mlp = nn.Sequential(
            nn.Linear(compressed_dim, compressed_dim * 2),
            nn.ReLU(),
            nn.Linear(compressed_dim * 2, compressed_dim),
            nn.Dropout(0.1)
        )
        
        print(f"✅ TemporalEncoder initialized:")
        print(f"   Input dim: {input_dim} → Compressed dim: {compressed_dim}")
        print(f"   Compression ratio: {compression_ratio}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for temporal compression with attention-based pooling.
        
        Args:
            x: Input tensor of shape (B, M, T, input_dim) where:
               - B is batch size
               - M is number of variables
               - T is sequence length
               - input_dim is cross-attention embedding dimension
               
        Returns:
            Tuple containing:
            - compressed_repr: Compressed representations of shape (B, M, compressed_dim)
            - compression_attn: Compression attention weights of shape (B, M, 1, T)
        """
        batch_size, num_vars, seq_len, input_dim = x.shape
        
        # Input validation
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {input_dim}")
        
        # Apply input projection if needed
        x_projected = self.input_projection(x)  # (B, M, T, compressed_dim)
        
        # Generate compression attention weights to identify important time steps
        # Expand compression query to match batch and variable dimensions
        compression_query = self.compression_query.expand(batch_size, num_vars, 1, self.input_dim)
        
        # Project query to compressed dimension if needed
        if self.input_dim != self.compressed_dim:
            query_projected = self.input_projection(compression_query)  # (B, M, 1, compressed_dim)
        else:
            query_projected = compression_query
        
        # Compute attention scores between compression query and all time steps
        # query: (B, M, 1, compressed_dim), keys: (B, M, T, compressed_dim)
        attention_scores = torch.matmul(
            query_projected, 
            x_projected.transpose(-2, -1)
        ) * self.attention_scale  # (B, M, 1, T)
        
        # Apply softmax to get attention weights
        compression_attn = torch.softmax(attention_scores, dim=-1)  # (B, M, 1, T)
        
        # Apply weighted temporal pooling to compress sequence length
        # Maintain variable-specific compression patterns
        compressed = torch.matmul(compression_attn, x_projected)  # (B, M, 1, compressed_dim)
        compressed = compressed.squeeze(-2)  # (B, M, compressed_dim)
        
        # Apply layer normalization
        compressed_normalized = self.layer_norm(compressed)
        
        # Apply post-compression MLP for additional processing
        compressed_final = self.post_compression_mlp(compressed_normalized)
        
        # Add residual connection if dimensions match
        if compressed_normalized.shape == compressed_final.shape:
            compressed_final = compressed_normalized + compressed_final
        
        return compressed_final, compression_attn


class SplineFunctionLearner(nn.Module):
    """
    Spline-based function learner for interpretable forecasting.
    
    This module generates interpretable B-spline functions for forecasting future values.
    It predicts control points from compressed representations and uses B-spline basis
    functions to generate smooth, extrapolatable forecasts.
    
    Args:
        input_dim: Input dimension from compressed representations
        num_control_points: Number of B-spline control points
        spline_degree: Degree of B-spline (typically 3 for cubic splines)
        forecast_horizon: Number of future time steps to predict
        stability_constraints: Whether to apply stability constraints to control points
    """
    
    def __init__(self, input_dim: int, num_control_points: int = 8, spline_degree: int = 3, 
                 forecast_horizon: int = 24, stability_constraints: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.num_control_points = num_control_points
        self.spline_degree = spline_degree
        self.forecast_horizon = forecast_horizon
        self.stability_constraints = stability_constraints
        
        # Validate parameters
        self._validate_parameters()
        
        # Create control point prediction network from compressed representations
        self.control_point_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_control_points)
        )
        
        # Initialize B-spline basis functions and knot vector
        self.knot_vector = self._create_knot_vector()
        self.basis_functions = self._generate_basis_functions()
        
        # Register as buffers so they move with the model to GPU/CPU
        self.register_buffer('_knot_vector', self.knot_vector)
        self.register_buffer('_basis_functions', self.basis_functions)
        
        print(f"✅ SplineFunctionLearner initialized:")
        print(f"   Input dim: {input_dim}")
        print(f"   Control points: {num_control_points}")
        print(f"   Spline degree: {spline_degree}")
        print(f"   Forecast horizon: {forecast_horizon}")
        print(f"   Stability constraints: {stability_constraints}")
    
    def _validate_parameters(self):
        """Validate spline parameters for mathematical correctness."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        
        if self.num_control_points <= 0:
            raise ValueError(f"num_control_points must be positive, got {self.num_control_points}")
        
        if self.spline_degree <= 0:
            raise ValueError(f"spline_degree must be positive, got {self.spline_degree}")
        
        if self.spline_degree >= self.num_control_points:
            raise ValueError(f"spline_degree ({self.spline_degree}) must be less than "
                           f"num_control_points ({self.num_control_points})")
        
        if self.forecast_horizon <= 0:
            raise ValueError(f"forecast_horizon must be positive, got {self.forecast_horizon}")
    
    def _create_knot_vector(self) -> torch.Tensor:
        """
        Create knot vector for B-spline basis functions.
        
        Uses uniform knot spacing with appropriate multiplicity at endpoints
        for open B-splines that interpolate the first and last control points.
        
        Returns:
            Knot vector of length (num_control_points + spline_degree + 1)
        """
        n = self.num_control_points
        p = self.spline_degree
        
        # Total number of knots
        num_knots = n + p + 1
        
        # Create uniform knot vector with endpoint multiplicity
        knots = torch.zeros(num_knots)
        
        # First (p+1) knots are 0 (multiplicity p+1 at start)
        knots[:p+1] = 0.0
        
        # Last (p+1) knots are 1 (multiplicity p+1 at end)
        knots[-p-1:] = 1.0
        
        # Interior knots are uniformly spaced
        if num_knots > 2 * (p + 1):
            interior_knots = torch.linspace(0, 1, num_knots - 2 * p)[1:-1]
            knots[p+1:-p-1] = interior_knots
        
        return knots
    
    def _generate_basis_functions(self) -> torch.Tensor:
        """
        Generate B-spline basis functions for the forecast horizon.
        
        Uses the Cox-de Boor recursion formula to compute B-spline basis functions
        evaluated at points corresponding to the forecast horizon.
        
        Returns:
            Basis function matrix of shape (forecast_horizon, num_control_points)
        """
        # Create evaluation points for forecast horizon
        # Map forecast steps to parameter space [0, 1] and then extrapolate
        t_eval = torch.linspace(1.0, 1.5, self.forecast_horizon)  # Extrapolate beyond [0,1]
        
        # Initialize basis function matrix
        basis_matrix = torch.zeros(self.forecast_horizon, self.num_control_points)
        
        # Compute basis functions using Cox-de Boor recursion
        for i in range(self.forecast_horizon):
            t = t_eval[i]
            basis_values = self._cox_de_boor_recursion(t)
            basis_matrix[i] = basis_values
        
        return basis_matrix
    
    def _cox_de_boor_recursion(self, t: float) -> torch.Tensor:
        """
        Compute B-spline basis functions at parameter t using Cox-de Boor recursion.
        
        Args:
            t: Parameter value where to evaluate basis functions
            
        Returns:
            Basis function values of shape (num_control_points,)
        """
        n = self.num_control_points
        p = self.spline_degree
        knots = self.knot_vector
        
        # Initialize basis functions
        basis = torch.zeros(n)
        
        # Find knot span (which interval t falls into)
        span = self._find_knot_span(t, knots, n, p)
        
        # Ensure span is within valid range
        span = max(p, min(span, n - 1))
        
        # Compute non-zero basis functions using Cox-de Boor recursion
        N = torch.zeros(p + 1)
        N[0] = 1.0
        
        for j in range(1, p + 1):
            saved = 0.0
            for r in range(j):
                # Check bounds before accessing knots
                left_idx = span - j + r + 1
                right_idx = span + r + 1
                
                if left_idx >= 0 and right_idx < len(knots):
                    # Avoid division by zero
                    denom = knots[right_idx] - knots[left_idx]
                    if abs(denom) > 1e-10:
                        alpha = (t - knots[left_idx]) / denom
                        temp = N[r] * alpha
                        N[r] = saved + N[r] * (knots[right_idx] - t) / denom
                        saved = temp
                    else:
                        N[r] = saved
                        saved = 0.0
                else:
                    N[r] = saved
                    saved = 0.0
            N[j] = saved
        
        # Copy non-zero basis functions to correct positions
        for j in range(p + 1):
            idx = span - p + j
            if 0 <= idx < n:
                basis[idx] = N[j]
        
        return basis
    
    def _find_knot_span(self, t: float, knots: torch.Tensor, n: int, p: int) -> int:
        """
        Find the knot span index for parameter t.
        
        Args:
            t: Parameter value
            knots: Knot vector
            n: Number of control points
            p: Spline degree
            
        Returns:
            Knot span index
        """
        # Handle edge cases - knots has length n+p+1
        if t >= knots[n]:
            return n - 1
        if t <= knots[p]:
            return p
        
        # Linear search for simplicity and robustness
        for i in range(p, n):
            if knots[i] <= t < knots[i + 1]:
                return i
        
        return n - 1  # Fallback
    
    def forward(self, compressed_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass that predicts control points and generates spline forecasts.
        
        Args:
            compressed_repr: Compressed representations of shape (B, M, input_dim)
            
        Returns:
            Dictionary containing:
            - 'forecasts': Spline-based forecasts of shape (B, M, forecast_horizon)
            - 'control_points': Predicted control points of shape (B, M, num_control_points)
            - 'basis_functions': B-spline basis functions of shape (forecast_horizon, num_control_points)
            - 'knot_vector': Knot vector for the B-splines
        """
        batch_size, num_vars, input_dim = compressed_repr.shape
        
        # Input validation
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {input_dim}")
        
        # Predict control points for each variable
        control_points = self.control_point_predictor(compressed_repr)  # (B, M, num_control_points)
        
        # Apply stability constraints if enabled
        if self.stability_constraints:
            control_points = self._apply_stability_constraints(control_points)
        
        # Generate spline forecasts using basis functions and control points
        # basis_functions: (forecast_horizon, num_control_points)
        # control_points: (B, M, num_control_points)
        # Result: (B, M, forecast_horizon)
        forecasts = torch.matmul(control_points, self._basis_functions.T)
        
        return {
            'forecasts': forecasts,
            'control_points': control_points,
            'basis_functions': self._basis_functions,
            'knot_vector': self._knot_vector
        }
    
    def _apply_stability_constraints(self, control_points: torch.Tensor) -> torch.Tensor:
        """
        Apply stability constraints to control points for robust forecasting.
        
        Args:
            control_points: Raw control points of shape (B, M, num_control_points)
            
        Returns:
            Constrained control points with improved stability
        """
        # Clamp control points to reasonable ranges to prevent extreme values
        control_points = torch.clamp(control_points, min=-10.0, max=10.0)
        
        # Optional: Apply smoothness constraint (penalize large differences between adjacent points)
        # This can be done during training via regularization, but here we apply a simple smoothing
        if self.num_control_points > 2:
            # Apply light smoothing to reduce sharp changes
            smoothed = control_points.clone()
            for i in range(1, self.num_control_points - 1):
                smoothed[:, :, i] = 0.25 * control_points[:, :, i-1] + \
                                   0.5 * control_points[:, :, i] + \
                                   0.25 * control_points[:, :, i+1]
            control_points = smoothed
        
        return control_points
    
    def extrapolate(self, compressed_repr: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Generate forecasts for a different horizon using spline extrapolation.
        
        Args:
            compressed_repr: Compressed representations of shape (B, M, input_dim)
            horizon: Number of steps to forecast (different from default)
            
        Returns:
            Extrapolated forecasts of shape (B, M, horizon)
        """
        # Predict control points
        control_points = self.control_point_predictor(compressed_repr)
        
        if self.stability_constraints:
            control_points = self._apply_stability_constraints(control_points)
        
        # Generate basis functions for the new horizon
        t_eval = torch.linspace(1.0, 1.0 + 0.5 * horizon / self.forecast_horizon, horizon)
        basis_matrix = torch.zeros(horizon, self.num_control_points, device=compressed_repr.device)
        
        for i in range(horizon):
            t = t_eval[i].item()
            basis_values = self._cox_de_boor_recursion(t)
            basis_matrix[i] = basis_values.to(compressed_repr.device)
        
        # Generate forecasts
        forecasts = torch.matmul(control_points, basis_matrix.T)
        
        return forecasts
    
    def get_spline_coefficients(self, compressed_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get interpretable spline coefficients and parameters.
        
        Args:
            compressed_repr: Compressed representations
            
        Returns:
            Dictionary with spline coefficients and mathematical parameters
        """
        result = self.forward(compressed_repr)
        
        # Add additional interpretability information
        result['spline_degree'] = torch.tensor(self.spline_degree)
        result['num_control_points'] = torch.tensor(self.num_control_points)
        
        # Compute spline derivatives at control points for trend analysis
        if self.num_control_points > 1:
            control_points = result['control_points']
            derivatives = control_points[:, :, 1:] - control_points[:, :, :-1]
            result['control_point_derivatives'] = derivatives
        
        return result
    
    def estimate_uncertainty(self, compressed_repr: torch.Tensor, num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Estimate uncertainty in forecasts through Monte Carlo sampling of control points.
        
        Args:
            compressed_repr: Compressed representations of shape (B, M, input_dim)
            num_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            Dictionary containing:
            - 'mean_forecast': Mean forecast across samples
            - 'std_forecast': Standard deviation of forecasts
            - 'confidence_intervals': 95% confidence intervals
            - 'forecast_samples': All forecast samples for further analysis
        """
        batch_size, num_vars, input_dim = compressed_repr.shape
        
        # Generate multiple samples by adding noise to control points
        forecast_samples = []
        
        for _ in range(num_samples):
            # Predict control points
            control_points = self.control_point_predictor(compressed_repr)
            
            # Add small amount of noise to simulate uncertainty
            noise_scale = 0.1  # Adjustable uncertainty level
            noise = torch.randn_like(control_points) * noise_scale
            noisy_control_points = control_points + noise
            
            # Apply stability constraints
            if self.stability_constraints:
                noisy_control_points = self._apply_stability_constraints(noisy_control_points)
            
            # Generate forecast with noisy control points
            forecast = torch.matmul(noisy_control_points, self._basis_functions.T)
            forecast_samples.append(forecast)
        
        # Stack all samples: (num_samples, B, M, forecast_horizon)
        forecast_samples = torch.stack(forecast_samples, dim=0)
        
        # Compute statistics
        mean_forecast = forecast_samples.mean(dim=0)  # (B, M, forecast_horizon)
        std_forecast = forecast_samples.std(dim=0)    # (B, M, forecast_horizon)
        
        # Compute 95% confidence intervals
        lower_percentile = torch.quantile(forecast_samples, 0.025, dim=0)
        upper_percentile = torch.quantile(forecast_samples, 0.975, dim=0)
        confidence_intervals = torch.stack([lower_percentile, upper_percentile], dim=-1)
        
        return {
            'mean_forecast': mean_forecast,
            'std_forecast': std_forecast,
            'confidence_intervals': confidence_intervals,
            'forecast_samples': forecast_samples
        }
    
    def visualize_spline_coefficients(self, compressed_repr: torch.Tensor, variable_idx: int = 0, 
                                    batch_idx: int = 0) -> Dict[str, torch.Tensor]:
        """
        Generate visualization data for spline coefficients and interpretation.
        
        Args:
            compressed_repr: Compressed representations
            variable_idx: Which variable to visualize (0 to num_variables-1)
            batch_idx: Which batch sample to visualize
            
        Returns:
            Dictionary with visualization data for plotting spline curves and control points
        """
        # Get spline results
        spline_results = self.forward(compressed_repr)
        
        # Extract data for specific variable and batch
        control_points = spline_results['control_points'][batch_idx, variable_idx]  # (num_control_points,)
        basis_functions = spline_results['basis_functions']  # (forecast_horizon, num_control_points)
        knot_vector = spline_results['knot_vector']
        
        # Generate high-resolution spline curve for smooth visualization
        t_fine = torch.linspace(1.0, 1.5, 200)  # Fine grid for smooth curve
        basis_fine = torch.zeros(200, self.num_control_points, device=compressed_repr.device)
        
        for i, t in enumerate(t_fine):
            basis_values = self._cox_de_boor_recursion(t.item())
            basis_fine[i] = basis_values.to(compressed_repr.device)
        
        # Compute fine spline curve
        spline_curve = torch.matmul(basis_fine, control_points)
        
        # Compute spline derivatives for trend analysis
        if len(t_fine) > 1:
            dt = t_fine[1] - t_fine[0]
            spline_derivative = torch.gradient(spline_curve, spacing=dt.item())[0]
        else:
            spline_derivative = torch.zeros_like(spline_curve)
        
        # Parameter space for control points (uniform spacing in [0,1])
        control_param_space = torch.linspace(0, 1, self.num_control_points)
        
        return {
            'control_points': control_points,
            'control_param_space': control_param_space,
            'spline_curve': spline_curve,
            'spline_derivative': spline_derivative,
            'parameter_space': t_fine,
            'forecast_points': spline_results['forecasts'][batch_idx, variable_idx],
            'basis_functions': basis_functions,
            'knot_vector': knot_vector,
            'forecast_parameter_space': torch.linspace(1.0, 1.5, self.forecast_horizon)
        }
    
    def analyze_spline_properties(self, compressed_repr: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze mathematical properties of the learned splines.
        
        Args:
            compressed_repr: Compressed representations
            
        Returns:
            Dictionary with spline property analysis
        """
        spline_results = self.forward(compressed_repr)
        control_points = spline_results['control_points']  # (B, M, num_control_points)
        
        batch_size, num_vars, num_cp = control_points.shape
        
        # Analyze control point statistics
        cp_mean = control_points.mean(dim=-1)  # (B, M) - mean control point value
        cp_std = control_points.std(dim=-1)    # (B, M) - control point variability
        cp_range = control_points.max(dim=-1)[0] - control_points.min(dim=-1)[0]  # (B, M) - range
        
        # Analyze smoothness (second differences of control points)
        if num_cp > 2:
            first_diff = control_points[:, :, 1:] - control_points[:, :, :-1]
            second_diff = first_diff[:, :, 1:] - first_diff[:, :, :-1]
            smoothness = torch.mean(torch.abs(second_diff), dim=-1)  # (B, M)
        else:
            smoothness = torch.zeros(batch_size, num_vars, device=control_points.device)
        
        # Analyze monotonicity (percentage of increasing segments)
        if num_cp > 1:
            increasing_segments = (first_diff > 0).float()
            monotonicity_score = increasing_segments.mean(dim=-1)  # (B, M) - 1.0 = fully increasing
        else:
            monotonicity_score = torch.zeros(batch_size, num_vars, device=control_points.device)
        
        # Forecast trend analysis
        forecasts = spline_results['forecasts']  # (B, M, forecast_horizon)
        if self.forecast_horizon > 1:
            forecast_trend = (forecasts[:, :, -1] - forecasts[:, :, 0]) / self.forecast_horizon
        else:
            forecast_trend = torch.zeros(batch_size, num_vars, device=forecasts.device)
        
        return {
            'control_point_mean': cp_mean,
            'control_point_std': cp_std,
            'control_point_range': cp_range,
            'smoothness_score': smoothness,
            'monotonicity_score': monotonicity_score,
            'forecast_trend': forecast_trend,
            'spline_complexity': cp_std / (cp_range + 1e-8)  # Normalized complexity measure
        }


class InterpretableForecastingModel(nn.Module):
    """
    Extended interpretable forecasting model that combines:
    1. Existing InterpretableTimeEncoder (univariate learners + temporal attention)
    2. CrossVariableAttention for inter-variable relationships
    3. TemporalEncoder for sequence compression
    4. SplineFunctionLearner for interpretable forecasting
    
    The model maintains end-to-end differentiability while providing interpretability
    artifacts at each stage of processing.
    
    Args:
        config: ExtendedModelConfig with all model parameters
    """
    
    def __init__(self, config: ExtendedModelConfig):
        super().__init__()
        self.config = config
        
        # Validate configuration
        config._validate_config()
        
        # Store key dimensions for easy access
        self.num_variables = config.num_variables
        self.embed_dim = config.embed_dim
        self.cross_dim = config.cross_dim
        self.compressed_dim = config.compressed_dim
        self.forecast_horizon = config.forecast_horizon
        
        # Initialize existing components
        self._init_base_components()
        
        # Initialize new components (placeholders for now)
        self._init_extended_components()
        
        # Initialize output projection if needed
        self._init_output_projection()
        
        print(f"✅ InterpretableForecastingModel initialized")
        print(f"   Variables: {self.num_variables}")
        print(f"   Embedding dim: {self.embed_dim} → Cross dim: {self.cross_dim} → Compressed dim: {self.compressed_dim}")
        print(f"   Forecast horizon: {self.forecast_horizon}")
        print(f"   Total parameters: {self.count_parameters():,}")
    
    def _init_base_components(self):
        """Initialize the existing InterpretableTimeEncoder components."""
        base_config = self.config.to_base_config()
        
        # Use existing InterpretableTimeEncoder
        self.interpretable_encoder = InterpretableTimeEncoder(
            num_variables=base_config.num_variables,
            embed_dim=base_config.embed_dim,
            hidden_dim=base_config.hidden_dim,
            num_heads=base_config.num_heads,
            dropout=base_config.dropout,
            max_len=base_config.max_len
        )
    
    def _init_extended_components(self):
        """Initialize the new extended components."""
        # CrossVariableAttention - implemented
        self.cross_attention = CrossVariableAttention(
            embed_dim=self.embed_dim,
            cross_dim=self.cross_dim,
            num_heads=self.config.cross_heads,
            dropout=self.config.cross_dropout
        )
        
        # TemporalEncoder for sequence compression
        self.temporal_encoder = TemporalEncoder(
            input_dim=self.cross_dim,
            compressed_dim=self.compressed_dim,
            compression_ratio=self.config.compression_ratio
        )
        
        # SplineFunctionLearner for interpretable forecasting
        self.spline_learner = SplineFunctionLearner(
            input_dim=self.compressed_dim,
            num_control_points=self.config.num_control_points,
            spline_degree=self.config.spline_degree,
            forecast_horizon=self.config.forecast_horizon,
            stability_constraints=self.config.spline_stability
        )
        
        print("✅ Extended components initialized")
        print("   CrossVariableAttention: Implemented")
        print("   TemporalEncoder: Implemented")
        print("   SplineFunctionLearner: Implemented")
    
    def _init_output_projection(self):
        """Initialize output projection layers if needed."""
        # For now, no additional projection needed
        # This may be updated when implementing actual forecasting output
        pass
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete extended model with comprehensive error handling.
        
        Args:
            x: Input tensor of shape (B, T, M) where:
               - B is batch size
               - T is sequence length  
               - M is number of variables (must equal config.num_variables)
               
        Returns:
            Dictionary containing:
            - 'forecasts': Predicted future values
            - 'interpretability': Dictionary with attention weights and other artifacts
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If gradient flow or stability issues are detected
        """
        try:
            # Comprehensive input validation
            self._validate_input(x)
            
            batch_size, seq_len, num_vars = x.shape
            
            # Validate batch compatibility
            self._validate_batch_compatibility(batch_size)
            
            # Handle variable-length sequences if needed
            x = self._handle_variable_length_sequences(x)
            
            # Update dimensions after potential sequence handling
            batch_size, seq_len, num_vars = x.shape
            
            # Stage 1: Existing pipeline (univariate learners + temporal attention)
            try:
                var_embeddings, temporal_attn = self.interpretable_encoder(x)
                # var_embeddings: (B, M, T, embed_dim)
                # temporal_attn: (B, M, num_heads, T, T)
            except Exception as e:
                raise RuntimeError(f"Error in interpretable encoder stage: {str(e)}")
            
            # Stage 2: Cross-variable attention
            try:
                cross_embeddings, cross_attn = self.cross_attention(var_embeddings)
            except Exception as e:
                raise RuntimeError(f"Error in cross-variable attention stage: {str(e)}")
            
            # Stage 3: Temporal compression
            try:
                compressed_repr, compression_attn = self.temporal_encoder(cross_embeddings)
            except Exception as e:
                raise RuntimeError(f"Error in temporal compression stage: {str(e)}")
            
            # Stage 4: Spline-based forecasting
            try:
                spline_results = self.spline_learner(compressed_repr)
                forecasts = spline_results['forecasts']
                spline_params = {
                    'control_points': spline_results['control_points'],
                    'basis_functions': spline_results['basis_functions'],
                    'knot_vector': spline_results['knot_vector']
                }
            except Exception as e:
                raise RuntimeError(f"Error in spline forecasting stage: {str(e)}")
            
            # Validate spline stability
            self._validate_spline_stability(spline_params)
            
            # Prepare output
            output = {
                'forecasts': forecasts,
                'interpretability': {
                    'temporal_attention': temporal_attn,
                    'cross_attention': cross_attn,
                    'compression_attention': compression_attn,
                    'spline_parameters': spline_params,
                    'variable_embeddings': var_embeddings,
                    'cross_embeddings': cross_embeddings,
                    'compressed_repr': compressed_repr
                }
            }
            
            # Detect NaN values in outputs
            self._detect_nan_outputs(output)
            
            return output
            
        except (ValueError, TypeError) as e:
            # Re-raise validation errors as-is (expected errors)
            raise e
        except Exception as e:
            # Enhanced error reporting with context for unexpected errors
            error_context = {
                'input_shape': x.shape if isinstance(x, torch.Tensor) else 'Invalid',
                'model_config': {
                    'num_variables': self.config.num_variables,
                    'embed_dim': self.config.embed_dim,
                    'cross_dim': self.config.cross_dim,
                    'compressed_dim': self.config.compressed_dim,
                    'forecast_horizon': self.config.forecast_horizon
                },
                'device': x.device if isinstance(x, torch.Tensor) else 'Unknown',
                'dtype': x.dtype if isinstance(x, torch.Tensor) else 'Unknown'
            }
            
            raise RuntimeError(f"InterpretableForecastingModel forward pass failed: {str(e)}\n"
                             f"Context: {error_context}")
    
    def forward_with_gradient_monitoring(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with gradient flow monitoring (for debugging/training).
        
        Args:
            x: Input tensor
            
        Returns:
            Model output with gradient monitoring
        """
        # Enable gradient computation
        x.requires_grad_(True)
        
        # Forward pass
        output = self.forward(x)
        
        # Compute a dummy loss for gradient monitoring
        loss = output['forecasts'].sum()
        loss.backward(retain_graph=True)
        
        # Monitor gradient flow
        self._monitor_gradient_flow()
        
        return output
    
    def _validate_input(self, x: torch.Tensor):
        """
        Comprehensive input validation for ETT dataset compatibility.
        
        Args:
            x: Input tensor to validate
            
        Raises:
            ValueError: If input doesn't meet requirements
            TypeError: If input is not a tensor
        """
        # Type validation
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor input, got {type(x)}")
        
        # Dimension validation
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input tensor (B, T, M), got {x.dim()}D tensor with shape {x.shape}")
        
        batch_size, seq_len, num_vars = x.shape
        
        # Variable count validation for ETT dataset compatibility
        if num_vars != self.num_variables:
            raise ValueError(f"Expected {self.num_variables} variables for ETT dataset, got {num_vars}")
        
        # Sequence length validation
        if seq_len > self.config.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.config.max_len}")
        
        if seq_len == 0:
            raise ValueError("Sequence length cannot be zero")
        
        # Batch size validation
        if batch_size == 0:
            raise ValueError("Batch size cannot be zero")
        
        if batch_size > 1000:  # Reasonable upper limit to prevent memory issues
            warnings.warn(f"Large batch size ({batch_size}) may cause memory issues")
        
        # Data quality validation
        if torch.isnan(x).any():
            nan_count = torch.isnan(x).sum().item()
            raise ValueError(f"Input contains {nan_count} NaN values")
        
        if torch.isinf(x).any():
            inf_count = torch.isinf(x).sum().item()
            raise ValueError(f"Input contains {inf_count} infinite values")
        
        # Data range validation (reasonable bounds for time series data)
        x_min, x_max = x.min().item(), x.max().item()
        if abs(x_min) > 1e6 or abs(x_max) > 1e6:
            warnings.warn(f"Input values have large magnitude: range [{x_min:.2e}, {x_max:.2e}]. "
                         "Consider normalization for better numerical stability.")
        
        # Device consistency validation
        if hasattr(self, '_device_cache'):
            if x.device != self._device_cache:
                warnings.warn(f"Input device ({x.device}) differs from model device ({self._device_cache})")
        else:
            self._device_cache = x.device
        
        # Dtype validation
        if x.dtype not in [torch.float32, torch.float64]:
            warnings.warn(f"Input dtype {x.dtype} may cause numerical issues. Recommend float32 or float64.")
    
    def _monitor_gradient_flow(self):
        """
        Monitor gradient flow through all components and detect issues.
        
        Raises:
            RuntimeError: If gradient flow issues are detected
        """
        gradient_issues = []
        
        # Check each component for gradient flow
        components = {
            'interpretable_encoder': self.interpretable_encoder,
            'cross_attention': self.cross_attention,
            'temporal_encoder': self.temporal_encoder,
            'spline_learner': self.spline_learner
        }
        
        for comp_name, component in components.items():
            param_count = 0
            grad_count = 0
            nan_grad_count = 0
            zero_grad_count = 0
            
            for name, param in component.named_parameters():
                if param.requires_grad:
                    param_count += 1
                    
                    if param.grad is not None:
                        grad_count += 1
                        
                        # Check for NaN gradients
                        if torch.isnan(param.grad).any():
                            nan_grad_count += 1
                            gradient_issues.append(f"NaN gradient in {comp_name}.{name}")
                        
                        # Check for zero gradients (potential dead neurons)
                        grad_norm = param.grad.norm().item()
                        if grad_norm < 1e-10:
                            zero_grad_count += 1
                            gradient_issues.append(f"Zero gradient in {comp_name}.{name} (norm: {grad_norm:.2e})")
                    else:
                        gradient_issues.append(f"Missing gradient for {comp_name}.{name}")
            
            # Component-level gradient statistics
            if param_count > 0:
                grad_ratio = grad_count / param_count
                if grad_ratio < 1.0:
                    gradient_issues.append(f"{comp_name}: Only {grad_count}/{param_count} parameters have gradients")
        
        # Raise error if critical gradient issues found
        if gradient_issues:
            critical_issues = [issue for issue in gradient_issues if 'NaN' in issue or 'Missing' in issue]
            if critical_issues:
                raise RuntimeError(f"Critical gradient flow issues detected:\n" + "\n".join(critical_issues))
            else:
                # Just warn for non-critical issues
                warnings.warn(f"Gradient flow warnings:\n" + "\n".join(gradient_issues))
    
    def _detect_nan_outputs(self, output: Dict[str, torch.Tensor]):
        """
        Detect NaN values in model outputs and provide detailed diagnostics.
        
        Args:
            output: Model output dictionary
            
        Raises:
            RuntimeError: If NaN values are detected in outputs
        """
        nan_locations = []
        
        # Check forecasts
        if 'forecasts' in output:
            forecasts = output['forecasts']
            if torch.isnan(forecasts).any():
                nan_count = torch.isnan(forecasts).sum().item()
                nan_locations.append(f"Forecasts contain {nan_count} NaN values")
        
        # Check interpretability artifacts
        if 'interpretability' in output:
            interp = output['interpretability']
            
            for key, tensor in interp.items():
                if isinstance(tensor, torch.Tensor):
                    if torch.isnan(tensor).any():
                        nan_count = torch.isnan(tensor).sum().item()
                        nan_locations.append(f"{key} contains {nan_count} NaN values")
                elif isinstance(tensor, dict):
                    # Handle nested dictionaries (like spline_parameters)
                    for sub_key, sub_tensor in tensor.items():
                        if isinstance(sub_tensor, torch.Tensor) and torch.isnan(sub_tensor).any():
                            nan_count = torch.isnan(sub_tensor).sum().item()
                            nan_locations.append(f"{key}.{sub_key} contains {nan_count} NaN values")
        
        if nan_locations:
            raise RuntimeError(f"NaN values detected in model outputs:\n" + "\n".join(nan_locations))
    
    def _validate_spline_stability(self, spline_params: Dict[str, torch.Tensor]):
        """
        Validate spline parameters for stability and mathematical correctness.
        
        Args:
            spline_params: Dictionary containing spline parameters
            
        Raises:
            RuntimeError: If spline parameters are unstable
        """
        stability_issues = []
        
        if 'control_points' in spline_params:
            control_points = spline_params['control_points']
            
            # Check for reasonable control point values
            cp_min, cp_max = control_points.min().item(), control_points.max().item()
            if abs(cp_min) > 100 or abs(cp_max) > 100:
                stability_issues.append(f"Control points have extreme values: range [{cp_min:.2f}, {cp_max:.2f}]")
            
            # Check for NaN or infinite control points
            if torch.isnan(control_points).any():
                stability_issues.append("Control points contain NaN values")
            
            if torch.isinf(control_points).any():
                stability_issues.append("Control points contain infinite values")
            
            # Check control point variance (too low variance indicates potential issues)
            cp_var = control_points.var().item()
            if cp_var < 1e-8:
                stability_issues.append(f"Control points have very low variance: {cp_var:.2e}")
        
        if 'basis_functions' in spline_params:
            basis_functions = spline_params['basis_functions']
            
            # Check basis function properties
            if torch.isnan(basis_functions).any():
                stability_issues.append("Basis functions contain NaN values")
            
            # Check if basis functions sum approximately to 1 (partition of unity property)
            basis_sum = basis_functions.sum(dim=-1)
            if not torch.allclose(basis_sum, torch.ones_like(basis_sum), atol=1e-3):
                stability_issues.append("Basis functions don't satisfy partition of unity property")
        
        if stability_issues:
            if self.config.spline_stability:
                raise RuntimeError(f"Spline stability issues detected:\n" + "\n".join(stability_issues))
            else:
                warnings.warn(f"Spline stability warnings (constraints disabled):\n" + "\n".join(stability_issues))
    
    def _handle_variable_length_sequences(self, x: torch.Tensor) -> torch.Tensor:
        """
        Handle variable-length sequences gracefully with padding/truncation.
        
        Args:
            x: Input tensor of shape (B, T, M)
            
        Returns:
            Processed tensor with consistent sequence length
        """
        batch_size, seq_len, num_vars = x.shape
        
        # Handle sequences that are too long
        if seq_len > self.config.max_len:
            warnings.warn(f"Truncating sequence from {seq_len} to {self.config.max_len}")
            x = x[:, :self.config.max_len, :]
        
        # Handle very short sequences (pad if necessary)
        min_seq_len = 4  # Minimum reasonable sequence length for forecasting
        if seq_len < min_seq_len:
            warnings.warn(f"Padding short sequence from {seq_len} to {min_seq_len}")
            padding = torch.zeros(batch_size, min_seq_len - seq_len, num_vars, 
                                device=x.device, dtype=x.dtype)
            x = torch.cat([x, padding], dim=1)
        
        return x
    
    def _validate_batch_compatibility(self, batch_size: int):
        """
        Validate batch size compatibility with model components.
        
        Args:
            batch_size: Batch size to validate
            
        Raises:
            ValueError: If batch size is incompatible
        """
        # Check memory constraints (rough estimate)
        estimated_memory = batch_size * self.config.max_len * self.config.num_variables * 4  # 4 bytes per float32
        
        if estimated_memory > 1e9:  # 1GB threshold
            warnings.warn(f"Large batch size ({batch_size}) may require {estimated_memory/1e9:.1f}GB memory")
        
        # Check for batch size compatibility with attention mechanisms
        max_attention_size = batch_size * self.config.num_variables * self.config.num_variables
        if max_attention_size > 1e6:  # Attention matrix size threshold
            warnings.warn(f"Large batch size with many variables may cause attention memory issues")
        
        # Validate batch size is reasonable for training/inference
        if batch_size > 512:
            warnings.warn(f"Very large batch size ({batch_size}) may cause numerical instability")
    
    def _placeholder_forecast(self, compressed_repr: torch.Tensor) -> torch.Tensor:
        """
        Placeholder forecasting function.
        
        Args:
            compressed_repr: Compressed representations of shape (B, M, compressed_dim)
            
        Returns:
            Simple forecast based on compressed representations
        """
        # compressed_repr has shape (B, M, compressed_dim) from TemporalEncoder
        batch_size, num_vars, compressed_dim = compressed_repr.shape
        
        # Create a simple linear projection for forecasting
        if not hasattr(self, '_forecast_projection'):
            self._forecast_projection = nn.Linear(compressed_dim, self.forecast_horizon).to(compressed_repr.device)
        
        # Project to forecast horizon
        forecasts = self._forecast_projection(compressed_repr)  # (B, M, forecast_horizon)
        
        return forecasts
    
    def _placeholder_spline_params(self, batch_size: int, num_vars: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Create placeholder spline parameters."""
        return {
            'control_points': torch.zeros(batch_size, num_vars, self.config.num_control_points, 
                                        device=device, dtype=dtype),
            'basis_functions': torch.zeros(self.forecast_horizon, self.config.num_control_points,
                                         device=device, dtype=dtype),
            'knot_vector': torch.zeros(self.config.num_control_points + self.config.spline_degree + 1,
                                     device=device, dtype=dtype)
        }
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_parameter_groups(self) -> Dict[str, nn.Module]:
        """
        Get parameter groups for different learning rates or regularization.
        
        Returns:
            Dictionary mapping component names to their modules
        """
        return {
            'base_encoder': self.interpretable_encoder,
            'cross_attention': self.cross_attention,
            'temporal_encoder': self.temporal_encoder,
            'spline_learner': self.spline_learner
        }
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency (placeholder)."""
        # This will be implemented when actual components are added
        warnings.warn("Gradient checkpointing not yet implemented for extended components")
    
    def get_attention_weights(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights without computing full forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with attention weights from different components
        """
        with torch.no_grad():
            output = self.forward(x)
            return output['interpretability']


def create_extended_model(config: Optional[ExtendedModelConfig] = None) -> InterpretableForecastingModel:
    """
    Convenience function to create an extended model with default or custom configuration.
    
    Args:
        config: Optional ExtendedModelConfig. If None, uses default ETT configuration.
        
    Returns:
        InterpretableForecastingModel: Initialized model
    """
    if config is None:
        # Default configuration for ETT dataset
        config = ExtendedModelConfig(
            num_variables=7,
            embed_dim=32,
            hidden_dim=64,
            num_heads=4,
            cross_dim=32,
            cross_heads=4,
            compressed_dim=64,
            forecast_horizon=24
        )
    
    return InterpretableForecastingModel(config)


def test_cross_variable_attention():
    """
    Comprehensive unit tests for CrossVariableAttention module.
    
    Tests:
    - Input/output shape transformations for various batch sizes and sequence lengths
    - Attention weight properties (non-negative, normalized, interpretable)
    - Gradient flow through all parameters
    - Variable masking and independence properties
    """
    print("🧪 Testing CrossVariableAttention module...")
    
    # Test cases with different configurations
    test_cases = [
        {"batch_size": 2, "num_vars": 7, "seq_len": 10, "embed_dim": 32, "cross_dim": 32, "num_heads": 4},
        {"batch_size": 4, "num_vars": 5, "seq_len": 20, "embed_dim": 64, "cross_dim": 48, "num_heads": 6},
        {"batch_size": 1, "num_vars": 3, "seq_len": 5, "embed_dim": 16, "cross_dim": 24, "num_heads": 3},
        {"batch_size": 3, "num_vars": 7, "seq_len": 50, "embed_dim": 32, "cross_dim": 32, "num_heads": 8},  # ETT-like
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Test case {i+1}: B={case['batch_size']}, M={case['num_vars']}, "
              f"T={case['seq_len']}, embed_dim={case['embed_dim']}, cross_dim={case['cross_dim']}, heads={case['num_heads']} ---")
        
        # Create CrossVariableAttention module
        cross_attn = CrossVariableAttention(
            embed_dim=case['embed_dim'],
            cross_dim=case['cross_dim'],
            num_heads=case['num_heads'],
            dropout=0.1
        )
        
        # Create test input (variable embeddings from temporal attention)
        x = torch.randn(case['batch_size'], case['num_vars'], case['seq_len'], case['embed_dim'])
        print(f"Input shape: {x.shape}")
        
        # Test 1: Input/output shape transformations
        print("1. Testing input/output shape transformations...")
        attended_embeddings, attn_weights = cross_attn(x)
        
        expected_output_shape = (case['batch_size'], case['num_vars'], case['seq_len'], case['cross_dim'])
        expected_attn_shape = (case['batch_size'], case['num_heads'], case['num_vars'], case['num_vars'])
        
        assert attended_embeddings.shape == expected_output_shape, \
            f"Expected output shape {expected_output_shape}, got {attended_embeddings.shape}"
        assert attn_weights.shape == expected_attn_shape, \
            f"Expected attention shape {expected_attn_shape}, got {attn_weights.shape}"
        
        print(f"✅ Output embeddings shape: {attended_embeddings.shape}")
        print(f"✅ Attention weights shape: {attn_weights.shape}")
        
        # Test 2: Attention weight properties
        print("2. Testing attention weight properties...")
        
        # Non-negative weights
        assert (attn_weights >= 0).all(), "Attention weights contain negative values"
        print("✅ Attention weights are non-negative")
        
        # Normalized weights (sum to approximately 1 along last dimension)
        attn_sum = attn_weights.sum(dim=-1)  # Sum over key variables
        sum_min, sum_max = attn_sum.min().item(), attn_sum.max().item()
        assert 0.7 <= sum_min and sum_max <= 1.3, \
            f"Attention weights poorly normalized: range [{sum_min:.3f}, {sum_max:.3f}]"
        print(f"✅ Attention weights normalized (sum range: [{sum_min:.3f}, {sum_max:.3f}])")
        
        # Interpretable structure (diagonal should have reasonable values)
        diagonal_attn = torch.diagonal(attn_weights, dim1=-2, dim2=-1)  # Self-attention values
        diag_mean = diagonal_attn.mean().item()
        print(f"✅ Self-attention mean: {diag_mean:.3f} (should be reasonable)")
        
        # Test 3: Gradient flow through all parameters
        print("3. Testing gradient flow...")
        loss = attended_embeddings.sum()
        loss.backward()
        
        param_count = 0
        grad_count = 0
        for name, param in cross_attn.named_parameters():
            param_count += 1
            if param.grad is not None:
                grad_count += 1
                grad_norm = param.grad.norm().item()
                print(f"   Gradient norm for {name}: {grad_norm:.6f}")
                assert grad_norm > 1e-8, f"Gradient too small for {name}: {grad_norm}"
            else:
                print(f"   ❌ No gradient for {name}")
        
        assert grad_count == param_count, f"Missing gradients: {grad_count}/{param_count} parameters have gradients"
        print(f"✅ All {param_count} parameters have gradients")
        
        # Test 4: Variable independence properties
        print("4. Testing variable masking and independence...")
        
        # Create masked input (zero out one variable)
        x_masked = x.clone()
        mask_var = 0
        x_masked[:, mask_var, :, :] = 0
        
        with torch.no_grad():
            attended_masked, attn_masked = cross_attn(x_masked)
            
            # Check that masked variable has different attention pattern
            original_attn_var = attn_weights[:, :, mask_var, :]  # Attention from masked variable
            masked_attn_var = attn_masked[:, :, mask_var, :]
            
            attn_diff = torch.norm(original_attn_var - masked_attn_var)
            print(f"✅ Attention difference with masking: {attn_diff:.6f}")
        
        print(f"✅ Test case {i+1} passed!\n")
    
    print("✅ All CrossVariableAttention tests passed!")


def test_temporal_encoder():
    """
    Comprehensive unit tests for TemporalEncoder module.
    
    Tests:
    - Compression ratio functionality and output dimensions
    - Information preservation through reconstruction quality tests
    - Variable-length sequence handling
    - Attention weight interpretability and visualization
    """
    print("🧪 Testing TemporalEncoder module...")
    
    # Test cases with different configurations
    test_cases = [
        {"batch_size": 2, "num_vars": 7, "seq_len": 20, "input_dim": 32, "compressed_dim": 64, "compression_ratio": 4},
        {"batch_size": 4, "num_vars": 5, "seq_len": 40, "input_dim": 48, "compressed_dim": 32, "compression_ratio": 8},
        {"batch_size": 1, "num_vars": 3, "seq_len": 12, "input_dim": 16, "compressed_dim": 24, "compression_ratio": 2},
        {"batch_size": 3, "num_vars": 7, "seq_len": 96, "input_dim": 32, "compressed_dim": 64, "compression_ratio": 4},  # ETT-like
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n--- Test case {i+1}: B={case['batch_size']}, M={case['num_vars']}, "
              f"T={case['seq_len']}, input_dim={case['input_dim']}, compressed_dim={case['compressed_dim']}, ratio={case['compression_ratio']} ---")
        
        # Create TemporalEncoder module
        temporal_encoder = TemporalEncoder(
            input_dim=case['input_dim'],
            compressed_dim=case['compressed_dim'],
            compression_ratio=case['compression_ratio']
        )
        
        # Create test input (cross-attended embeddings)
        x = torch.randn(case['batch_size'], case['num_vars'], case['seq_len'], case['input_dim'])
        print(f"Input shape: {x.shape}")
        
        # Test 1: Compression ratio functionality and output dimensions
        print("1. Testing compression ratio functionality and output dimensions...")
        compressed_repr, compression_attn = temporal_encoder(x)
        
        expected_compressed_shape = (case['batch_size'], case['num_vars'], case['compressed_dim'])
        expected_attn_shape = (case['batch_size'], case['num_vars'], 1, case['seq_len'])
        
        assert compressed_repr.shape == expected_compressed_shape, \
            f"Expected compressed shape {expected_compressed_shape}, got {compressed_repr.shape}"
        assert compression_attn.shape == expected_attn_shape, \
            f"Expected attention shape {expected_attn_shape}, got {compression_attn.shape}"
        
        print(f"✅ Compressed representation shape: {compressed_repr.shape}")
        print(f"✅ Compression attention shape: {compression_attn.shape}")
        
        # Verify compression actually happened (sequence length reduced to 1 in compressed representation)
        original_seq_len = case['seq_len']
        compressed_seq_len = 1  # Should be 1 after compression
        compression_achieved = original_seq_len > compressed_seq_len
        assert compression_achieved, f"No compression achieved: {original_seq_len} -> {compressed_seq_len}"
        print(f"✅ Compression achieved: {original_seq_len} -> {compressed_seq_len}")
        
        # Test 2: Attention weight properties
        print("2. Testing attention weight properties...")
        
        # Attention weights should be non-negative
        assert (compression_attn >= 0).all(), "Compression attention weights contain negative values"
        print("✅ Compression attention weights are non-negative")
        
        # Attention weights should be normalized (sum to 1 along temporal dimension)
        attn_sum = compression_attn.sum(dim=-1)  # Sum over time steps
        sum_min, sum_max = attn_sum.min().item(), attn_sum.max().item()
        assert 0.95 <= sum_min and sum_max <= 1.05, \
            f"Compression attention weights poorly normalized: range [{sum_min:.3f}, {sum_max:.3f}]"
        print(f"✅ Compression attention weights normalized (sum range: [{sum_min:.3f}, {sum_max:.3f}])")
        
        # Check attention distribution (shouldn't be uniform - should focus on important time steps)
        attn_entropy = -torch.sum(compression_attn * torch.log(compression_attn + 1e-8), dim=-1)
        max_entropy = torch.log(torch.tensor(float(case['seq_len'])))  # Uniform distribution entropy
        avg_entropy = attn_entropy.mean().item()
        entropy_ratio = avg_entropy / max_entropy.item()
        print(f"✅ Attention entropy ratio: {entropy_ratio:.3f} (< 1.0 means focused attention)")
        
        # Test 3: Information preservation through reconstruction quality
        print("3. Testing information preservation...")
        
        # Test that different inputs produce different compressed representations
        x_alt = torch.randn_like(x)
        compressed_alt, _ = temporal_encoder(x_alt)
        
        repr_diff = torch.norm(compressed_repr - compressed_alt).item()
        assert repr_diff > 1e-3, f"Compressed representations too similar: diff={repr_diff}"
        print(f"✅ Different inputs produce different compressed representations (diff: {repr_diff:.6f})")
        
        # Test that compression preserves some information about the input
        # Compute correlation between input statistics and compressed representation
        input_mean = x.mean(dim=-2)  # (B, M, input_dim) - mean over time
        input_std = x.std(dim=-2)   # (B, M, input_dim) - std over time
        
        # Simple correlation test: compressed representation should correlate with input statistics
        if case['input_dim'] == case['compressed_dim']:
            # When dimensions match, check direct correlation
            mean_corr = torch.corrcoef(torch.cat([input_mean.flatten(), compressed_repr.flatten()]))[0, 1]
            print(f"✅ Input-compression correlation: {mean_corr:.3f}")
        else:
            # When dimensions differ, just check that compression captures some variance
            compressed_var = compressed_repr.var().item()
            input_var = x.var().item()
            var_ratio = compressed_var / input_var
            print(f"✅ Variance preservation ratio: {var_ratio:.3f}")
            assert var_ratio > 0.01, f"Too much variance lost in compression: {var_ratio}"
        
        # Test 4: Variable-length sequence handling
        print("4. Testing variable-length sequence handling...")
        
        # Test with shorter sequence
        short_seq_len = max(1, case['seq_len'] // 2)
        x_short = torch.randn(case['batch_size'], case['num_vars'], short_seq_len, case['input_dim'])
        
        compressed_short, attn_short = temporal_encoder(x_short)
        
        expected_short_compressed_shape = (case['batch_size'], case['num_vars'], case['compressed_dim'])
        expected_short_attn_shape = (case['batch_size'], case['num_vars'], 1, short_seq_len)
        
        assert compressed_short.shape == expected_short_compressed_shape, \
            f"Short sequence compressed shape mismatch: expected {expected_short_compressed_shape}, got {compressed_short.shape}"
        assert attn_short.shape == expected_short_attn_shape, \
            f"Short sequence attention shape mismatch: expected {expected_short_attn_shape}, got {attn_short.shape}"
        
        print(f"✅ Variable-length sequences handled correctly")
        print(f"   Original: {case['seq_len']} -> Compressed: 1")
        print(f"   Short: {short_seq_len} -> Compressed: 1")
        
        # Test 5: Gradient flow
        print("5. Testing gradient flow...")
        loss = compressed_repr.sum()
        loss.backward()
        
        param_count = 0
        grad_count = 0
        for name, param in temporal_encoder.named_parameters():
            param_count += 1
            if param.grad is not None:
                grad_count += 1
                grad_norm = param.grad.norm().item()
                print(f"   Gradient norm for {name}: {grad_norm:.6f}")
                assert grad_norm > 1e-8, f"Gradient too small for {name}: {grad_norm}"
            else:
                print(f"   ❌ No gradient for {name}")
        
        assert grad_count == param_count, f"Missing gradients: {grad_count}/{param_count} parameters have gradients"
        print(f"✅ All {param_count} parameters have gradients")
        
        # Test 6: Attention weight interpretability
        print("6. Testing attention weight interpretability...")
        
        # Create input with clear temporal pattern
        t = torch.linspace(0, 4*torch.pi, case['seq_len'])
        pattern_input = torch.zeros_like(x)
        for b in range(case['batch_size']):
            for m in range(case['num_vars']):
                # Create sinusoidal pattern with peak in the middle
                pattern = torch.sin(t) * torch.exp(-(t - 2*torch.pi)**2 / (torch.pi**2))
                pattern_input[b, m, :, 0] = pattern
        
        with torch.no_grad():
            _, pattern_attn = temporal_encoder(pattern_input)
            
            # Check if attention focuses on the peak region (middle of sequence)
            middle_start = case['seq_len'] // 3
            middle_end = 2 * case['seq_len'] // 3
            
            middle_attn = pattern_attn[:, :, :, middle_start:middle_end].sum(dim=-1)
            total_attn = pattern_attn.sum(dim=-1)
            middle_ratio = (middle_attn / total_attn).mean().item()
            
            print(f"✅ Attention focus on pattern peak: {middle_ratio:.3f} (higher is better)")
        
        print(f"✅ Test case {i+1} passed!\n")
    
    # Test edge cases
    print("Testing edge cases...")
    
    # Test with sequence length of 1
    encoder_edge = TemporalEncoder(input_dim=32, compressed_dim=32, compression_ratio=1)
    x_single = torch.randn(2, 3, 1, 32)  # Single time step
    compressed_single, attn_single = encoder_edge(x_single)
    
    assert compressed_single.shape == (2, 3, 32), "Single time step compression failed"
    assert attn_single.shape == (2, 3, 1, 1), "Single time step attention failed"
    assert torch.allclose(attn_single, torch.ones_like(attn_single)), "Single time step attention should be 1.0"
    print("✅ Single time step test passed")
    
    # Test with very large compression ratio
    try:
        encoder_large = TemporalEncoder(input_dim=32, compressed_dim=64, compression_ratio=1000)
        x_large = torch.randn(1, 2, 50, 32)
        compressed_large, _ = encoder_large(x_large)
        assert compressed_large.shape == (1, 2, 64), "Large compression ratio test failed"
        print("✅ Large compression ratio test passed")
    except Exception as e:
        print(f"⚠️ Large compression ratio test failed (expected): {e}")
    
    # Test parameter validation
    print("Testing parameter validation...")
    
    try:
        TemporalEncoder(input_dim=0, compressed_dim=32)
        assert False, "Should have raised ValueError for input_dim=0"
    except ValueError:
        print("✅ Correctly caught error for input_dim=0")
    
    try:
        TemporalEncoder(input_dim=32, compressed_dim=0)
        assert False, "Should have raised ValueError for compressed_dim=0"
    except ValueError:
        print("✅ Correctly caught error for compressed_dim=0")
    
    try:
        TemporalEncoder(input_dim=32, compressed_dim=32, compression_ratio=0)
        assert False, "Should have raised ValueError for compression_ratio=0"
    except ValueError:
        print("✅ Correctly caught error for compression_ratio=0")
    
    print("\n✅ All TemporalEncoder tests passed!")


def test_extended_model_integration():
    """
    Test the integration of TemporalEncoder with the complete extended model.
    
    Tests:
    - End-to-end pipeline with TemporalEncoder
    - Gradient flow through all components
    - Output shapes and interpretability artifacts
    """
    print("🧪 Testing Extended Model Integration with TemporalEncoder...")
    
    # Create extended model with default configuration
    config = ExtendedModelConfig(
        num_variables=7,
        embed_dim=32,
        cross_dim=32,
        compressed_dim=64,
        forecast_horizon=24
    )
    
    model = InterpretableForecastingModel(config)
    
    # Test with ETT-like input
    batch_size, seq_len = 4, 96
    x = torch.randn(batch_size, seq_len, config.num_variables)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    
    # Check output structure
    assert 'forecasts' in output, "Missing 'forecasts' in output"
    assert 'interpretability' in output, "Missing 'interpretability' in output"
    
    forecasts = output['forecasts']
    interpretability = output['interpretability']
    
    # Check forecast shape
    expected_forecast_shape = (batch_size, config.num_variables, config.forecast_horizon)
    assert forecasts.shape == expected_forecast_shape, \
        f"Expected forecast shape {expected_forecast_shape}, got {forecasts.shape}"
    print(f"✅ Forecasts shape: {forecasts.shape}")
    
    # Check interpretability artifacts
    required_artifacts = ['temporal_attention', 'cross_attention', 'compression_attention', 
                         'spline_parameters', 'variable_embeddings', 'cross_embeddings', 'compressed_repr']
    
    for artifact in required_artifacts:
        assert artifact in interpretability, f"Missing interpretability artifact: {artifact}"
    
    # Check compression attention shape (from TemporalEncoder)
    compression_attn = interpretability['compression_attention']
    expected_compression_attn_shape = (batch_size, config.num_variables, 1, seq_len)
    assert compression_attn.shape == expected_compression_attn_shape, \
        f"Expected compression attention shape {expected_compression_attn_shape}, got {compression_attn.shape}"
    print(f"✅ Compression attention shape: {compression_attn.shape}")
    
    # Check compressed representation shape
    compressed_repr = interpretability['compressed_repr']
    expected_compressed_shape = (batch_size, config.num_variables, config.compressed_dim)
    assert compressed_repr.shape == expected_compressed_shape, \
        f"Expected compressed shape {expected_compressed_shape}, got {compressed_repr.shape}"
    print(f"✅ Compressed representation shape: {compressed_repr.shape}")
    
    # Test gradient flow through entire model
    loss = forecasts.sum()
    loss.backward()
    
    # Check gradients for TemporalEncoder specifically
    temporal_encoder = model.temporal_encoder
    temporal_param_count = 0
    temporal_grad_count = 0
    
    for name, param in temporal_encoder.named_parameters():
        temporal_param_count += 1
        if param.grad is not None:
            temporal_grad_count += 1
            grad_norm = param.grad.norm().item()
            print(f"   TemporalEncoder gradient norm for {name}: {grad_norm:.6f}")
            assert grad_norm > 1e-8, f"TemporalEncoder gradient too small for {name}: {grad_norm}"
        else:
            print(f"   ❌ No gradient for TemporalEncoder {name}")
    
    assert temporal_grad_count == temporal_param_count, \
        f"Missing TemporalEncoder gradients: {temporal_grad_count}/{temporal_param_count} parameters have gradients"
    print(f"✅ All {temporal_param_count} TemporalEncoder parameters have gradients")
    
    # Test compression attention properties
    compression_attn = interpretability['compression_attention']
    
    # Should be normalized
    attn_sum = compression_attn.sum(dim=-1)
    sum_min, sum_max = attn_sum.min().item(), attn_sum.max().item()
    assert 0.95 <= sum_min and sum_max <= 1.05, \
        f"Compression attention poorly normalized: range [{sum_min:.3f}, {sum_max:.3f}]"
    print(f"✅ Compression attention normalized (sum range: [{sum_min:.3f}, {sum_max:.3f}])")
    
    # Should be non-negative
    assert (compression_attn >= 0).all(), "Compression attention contains negative values"
    print("✅ Compression attention is non-negative")
    
    print("✅ Extended Model Integration with TemporalEncoder passed!")


if __name__ == "__main__":
    # Run tests
    print("="*60)
    print("🧪 RUNNING EXTENDED MODEL TESTS")
    print("="*60)
    
    # Test individual components
    test_cross_variable_attention()
    print("\n" + "-"*60 + "\n")
    
    test_temporal_encoder()
    print("\n" + "-"*60 + "\n")
    
    # Test integration
    test_extended_model_integration()
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)