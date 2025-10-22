"""
Interpretable Time Series Forecasting Model

This module implements TKAN-style univariate function learners and temporal attention blocks
for interpretable time series forecasting.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration class for the interpretable time series model."""
    num_variables: int = 7          # HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
    embed_dim: int = 32
    hidden_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    max_len: int = 512


class UnivariateFunctionLearner(nn.Module):
    """
    TKAN-style univariate function learner that processes each variable independently.
    
    Transforms scalar input through a 2-layer MLP with ReLU activation in a 
    time-distributed manner.
    
    Args:
        in_features: Input feature dimension (default: 1 for scalar values)
        out_features: Output embedding dimension
        hidden_features: Hidden layer dimension
    """
    
    def __init__(self, in_features: int = 1, out_features: int = 32, hidden_features: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the univariate function learner.
        
        Args:
            x: Input tensor of shape (B, T_in, 1)
            
        Returns:
            Output tensor of shape (B, T_in, out_features)
        """
        b, t, _ = x.shape
        # Flatten time dimension for processing
        x_flat = x.view(b * t, -1)
        # Apply MLP
        u_flat = self.net(x_flat)
        # Reshape back to (B, T_in, out_features)
        return u_flat.view(b, t, -1)


class TemporalSelfAttention(nn.Module):
    """
    Temporal self-attention module that captures dependencies within each variable's embeddings.
    
    Applies multi-head self-attention with positional encoding, layer normalization,
    and residual connections to process temporal sequences.
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        max_len: Maximum sequence length for positional encoding
    """
    
    def __init__(self, embed_dim: int = 32, num_heads: int = 4, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Positional encoding
        self.positional_encoding = nn.Embedding(max_len, embed_dim)
        
        # Layer normalization (pre-norm)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Dropout for residual connection
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal self-attention.
        
        Args:
            u: Input embeddings of shape (B, T_in, embed_dim)
            
        Returns:
            h: Attended embeddings of shape (B, T_in, embed_dim)
            attn_weights: Attention weights of shape (B, num_heads, T_in, T_in)
        """
        b, t, d = u.shape
        
        # Create positional encodings
        positions = torch.arange(t, device=u.device).unsqueeze(0).expand(b, -1)
        pos_emb = self.positional_encoding(positions)
        
        # Add positional encoding
        u_pos = u + pos_emb
        
        # Pre-layer normalization
        u_norm = self.norm(u_pos)
        
        # Self-attention (get per-head attention weights)
        attn_output, attn_weights = self.attention(u_norm, u_norm, u_norm, average_attn_weights=False)
        
        # Residual connection with dropout
        h = u + self.dropout(attn_output)
        
        return h, attn_weights


def load_ett_data(file_path: str = "interpretable_forecasting/ETT-small/ETTh1.csv", num_samples: int = 100):
    """Load and preprocess ETT dataset for visualization."""
    try:
        df = pd.read_csv(file_path)
        # Take first num_samples for visualization
        df_sample = df.head(num_samples)
        
        # Extract the 7 variables (excluding date column)
        variables = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        data = df_sample[variables].values
        dates = pd.to_datetime(df_sample['date'])
        
        return data, dates, variables
    except FileNotFoundError:
        print(f"ETT dataset not found at {file_path}")
        print("Generating synthetic data for demonstration...")
        
        # Generate synthetic data similar to ETT
        np.random.seed(42)
        num_samples = 100
        variables = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        
        # Create synthetic time series with different patterns
        t = np.linspace(0, 4*np.pi, num_samples)
        data = np.zeros((num_samples, 7))
        
        # Different patterns for each variable
        data[:, 0] = 5 + 2 * np.sin(t) + 0.5 * np.random.randn(num_samples)  # HUFL
        data[:, 1] = 2 + 1.5 * np.cos(t) + 0.3 * np.random.randn(num_samples)  # HULL
        data[:, 2] = 1.5 + np.sin(2*t) + 0.2 * np.random.randn(num_samples)  # MUFL
        data[:, 3] = 0.4 + 0.3 * np.cos(3*t) + 0.1 * np.random.randn(num_samples)  # MULL
        data[:, 4] = 4 + 1.8 * np.sin(t + np.pi/4) + 0.4 * np.random.randn(num_samples)  # LUFL
        data[:, 5] = 1.3 + 0.8 * np.cos(t - np.pi/3) + 0.2 * np.random.randn(num_samples)  # LULL
        data[:, 6] = 25 + 5 * np.sin(0.5*t) + np.random.randn(num_samples)  # OT (temperature)
        
        dates = pd.date_range('2016-07-01', periods=num_samples, freq='H')
        
        return data, dates, variables


def visualize_univariate_function_learner():
    """Create comprehensive visualizations of the UnivariateFunctionLearner."""
    print("🎨 Creating visualizations for UnivariateFunctionLearner...")
    
    # Load real or synthetic ETT data
    data, dates, variables = load_ett_data()
    seq_len = min(50, len(data))  # Use first 50 time steps for clarity
    
    # Create model
    embed_dim = 8  # Smaller for visualization
    learner = UnivariateFunctionLearner(in_features=1, out_features=embed_dim, hidden_features=16)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Original Time Series Data
    plt.subplot(4, 3, 1)
    for i, var in enumerate(variables):
        plt.plot(dates[:seq_len], data[:seq_len, i], label=var, alpha=0.8)
    plt.title("📊 Original ETT Dataset Variables", fontsize=14, fontweight='bold')
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 2. Focus on one variable (HUFL) transformation
    var_idx = 0  # HUFL
    input_data = torch.tensor(data[:seq_len, var_idx:var_idx+1], dtype=torch.float32).unsqueeze(0)  # (1, T, 1)
    
    with torch.no_grad():
        embeddings = learner(input_data)  # (1, T, embed_dim)
    
    plt.subplot(4, 3, 2)
    plt.plot(dates[:seq_len], input_data[0, :, 0].numpy(), 'b-', linewidth=2, label='Input (HUFL)')
    plt.title(f"🔍 Input: {variables[var_idx]} Variable", fontsize=14, fontweight='bold')
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 3. Embedding dimensions heatmap
    plt.subplot(4, 3, 3)
    embedding_matrix = embeddings[0].numpy().T  # (embed_dim, T)
    im = plt.imshow(embedding_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.title(f"🌈 {variables[var_idx]} → {embed_dim}D Embeddings", fontsize=14, fontweight='bold')
    plt.xlabel("Time Steps")
    plt.ylabel("Embedding Dimensions")
    plt.colorbar(im, shrink=0.8)
    
    # 4. Individual embedding dimensions over time
    plt.subplot(4, 3, 4)
    colors = plt.cm.tab10(np.linspace(0, 1, embed_dim))
    for dim in range(min(4, embed_dim)):  # Show first 4 dimensions
        plt.plot(dates[:seq_len], embeddings[0, :, dim].numpy(), 
                color=colors[dim], label=f'Dim {dim}', alpha=0.8)
    plt.title("📈 Embedding Dimensions Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Time")
    plt.ylabel("Embedding Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 5. MLP Layer Activations
    plt.subplot(4, 3, 5)
    # Get intermediate activations
    x_flat = input_data.view(-1, 1)
    with torch.no_grad():
        hidden = torch.relu(learner.net[0](x_flat))  # After first layer + ReLU
        output = learner.net[2](hidden)  # Final output
    
    plt.hist(hidden.numpy().flatten(), bins=30, alpha=0.7, label='Hidden Layer', color='orange')
    plt.hist(output.numpy().flatten(), bins=30, alpha=0.7, label='Output Layer', color='green')
    plt.title("🧠 MLP Layer Activation Distributions", fontsize=14, fontweight='bold')
    plt.xlabel("Activation Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Input-Output Relationship
    plt.subplot(4, 3, 6)
    input_vals = input_data[0, :, 0].numpy()
    output_norm = torch.norm(embeddings[0], dim=1).numpy()  # L2 norm of embeddings
    plt.scatter(input_vals, output_norm, alpha=0.6, c=range(len(input_vals)), cmap='plasma')
    plt.title("🔗 Input vs Embedding Magnitude", fontsize=14, fontweight='bold')
    plt.xlabel(f"Input Value ({variables[var_idx]})")
    plt.ylabel("Embedding L2 Norm")
    plt.colorbar(label='Time Step')
    plt.grid(True, alpha=0.3)
    
    # 7. Multiple Variables Comparison
    plt.subplot(4, 3, 7)
    embedding_norms = []
    for var_idx in range(min(4, len(variables))):  # First 4 variables
        var_input = torch.tensor(data[:seq_len, var_idx:var_idx+1], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            var_embeddings = learner(var_input)
            var_norm = torch.norm(var_embeddings[0], dim=1).numpy()
            embedding_norms.append(var_norm)
            plt.plot(dates[:seq_len], var_norm, label=variables[var_idx], alpha=0.8)
    
    plt.title("📊 Embedding Magnitudes: Multiple Variables", fontsize=14, fontweight='bold')
    plt.xlabel("Time")
    plt.ylabel("Embedding L2 Norm")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 8. Embedding Space Visualization (PCA-like)
    plt.subplot(4, 3, 8)
    # Take first 2 embedding dimensions for 2D visualization
    emb_2d = embeddings[0, :, :2].numpy()
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=range(len(emb_2d)), 
                         cmap='viridis', alpha=0.7, s=50)
    plt.title("🗺️ 2D Embedding Space Trajectory", fontsize=14, fontweight='bold')
    plt.xlabel("Embedding Dimension 0")
    plt.ylabel("Embedding Dimension 1")
    plt.colorbar(scatter, label='Time Step')
    plt.grid(True, alpha=0.3)
    
    # 9. Architecture Diagram (Text-based)
    plt.subplot(4, 3, 9)
    plt.text(0.1, 0.8, "🏗️ UnivariateFunctionLearner Architecture", fontsize=14, fontweight='bold')
    plt.text(0.1, 0.7, f"Input: (B, T, 1) = Scalar time series", fontsize=10)
    plt.text(0.1, 0.6, f"↓", fontsize=12)
    plt.text(0.1, 0.55, f"Linear(1 → {learner.net[0].out_features})", fontsize=10)
    plt.text(0.1, 0.45, f"↓", fontsize=12)
    plt.text(0.1, 0.4, f"ReLU Activation", fontsize=10)
    plt.text(0.1, 0.3, f"↓", fontsize=12)
    plt.text(0.1, 0.25, f"Linear({learner.net[2].in_features} → {embed_dim})", fontsize=10)
    plt.text(0.1, 0.15, f"↓", fontsize=12)
    plt.text(0.1, 0.1, f"Output: (B, T, {embed_dim}) = Embedding sequence", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # 10. Gradient Flow Visualization
    plt.subplot(4, 3, 10)
    # Compute gradients
    input_data.requires_grad_(True)
    output = learner(input_data)
    loss = output.sum()
    loss.backward()
    
    gradients = input_data.grad[0, :, 0].numpy()
    plt.plot(dates[:seq_len], gradients, 'r-', linewidth=2, label='Input Gradients')
    plt.title("⚡ Gradient Flow (∂Loss/∂Input)", fontsize=14, fontweight='bold')
    plt.xlabel("Time")
    plt.ylabel("Gradient Magnitude")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 11. Parameter Statistics
    plt.subplot(4, 3, 11)
    param_stats = []
    param_names = []
    for name, param in learner.named_parameters():
        param_stats.append([param.data.min().item(), param.data.mean().item(), param.data.max().item()])
        param_names.append(name.replace('net.', '').replace('.weight', '_W').replace('.bias', '_b'))
    
    param_stats = np.array(param_stats)
    x_pos = np.arange(len(param_names))
    
    plt.bar(x_pos - 0.2, param_stats[:, 0], 0.2, label='Min', alpha=0.7)
    plt.bar(x_pos, param_stats[:, 1], 0.2, label='Mean', alpha=0.7)
    plt.bar(x_pos + 0.2, param_stats[:, 2], 0.2, label='Max', alpha=0.7)
    
    plt.title("📊 Model Parameter Statistics", fontsize=14, fontweight='bold')
    plt.xlabel("Parameters")
    plt.ylabel("Value")
    plt.xticks(x_pos, param_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 12. Summary Statistics
    plt.subplot(4, 3, 12)
    stats_text = f"""
📈 Model Summary:
• Input Shape: (B, T, 1)
• Output Shape: (B, T, {embed_dim})
• Parameters: {sum(p.numel() for p in learner.parameters()):,}
• Hidden Units: {learner.net[0].out_features}
• Embedding Dim: {embed_dim}

📊 Data Summary:
• Variables: {len(variables)}
• Time Steps: {seq_len}
• Value Range: [{data[:seq_len].min():.2f}, {data[:seq_len].max():.2f}]
"""
    plt.text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('univariate_function_learner_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization complete! Saved as 'univariate_function_learner_analysis.png'")
    print("\n🔍 What you're seeing:")
    print("1. Original ETT dataset with 7 variables (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)")
    print("2. How one variable (HUFL) gets transformed through the MLP")
    print("3. The resulting embedding space - each scalar becomes a vector")
    print("4. How different embedding dimensions capture different patterns")
    print("5. Internal MLP activations and parameter distributions")
    print("6. The relationship between input values and embedding magnitudes")
    print("7. How the model processes multiple variables independently")


def test_univariate_function_learner():
    """Test the UnivariateFunctionLearner with different input shapes."""
    print("Testing UnivariateFunctionLearner...")
    
    # Test with different configurations
    test_cases = [
        {"batch_size": 2, "seq_len": 10, "embed_dim": 32},
        {"batch_size": 4, "seq_len": 20, "embed_dim": 64},
        {"batch_size": 1, "seq_len": 5, "embed_dim": 16},
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\nTest case {i+1}: B={case['batch_size']}, T={case['seq_len']}, d_var={case['embed_dim']}")
        
        # Create model
        learner = UnivariateFunctionLearner(
            in_features=1, 
            out_features=case['embed_dim'], 
            hidden_features=64
        )
        
        # Create test input
        x = torch.randn(case['batch_size'], case['seq_len'], 1)
        print(f"Input shape: {x.shape}")
        
        # Forward pass
        output = learner(x)
        print(f"Output shape: {output.shape}")
        
        # Verify output shape
        expected_shape = (case['batch_size'], case['seq_len'], case['embed_dim'])
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        # Test gradient computation
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        for name, param in learner.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            print(f"Gradient norm for {name}: {param.grad.norm().item():.6f}")
        
        print("✓ Forward pass and gradient computation successful")
    
    print("\n✅ All UnivariateFunctionLearner tests passed!")


def test_temporal_self_attention():
    """Test the TemporalSelfAttention module with different sequence lengths."""
    print("Testing TemporalSelfAttention...")
    
    # Test with different configurations
    test_cases = [
        {"batch_size": 2, "seq_len": 10, "embed_dim": 32, "num_heads": 4},
        {"batch_size": 4, "seq_len": 20, "embed_dim": 64, "num_heads": 8},
        {"batch_size": 1, "seq_len": 5, "embed_dim": 16, "num_heads": 2},
        {"batch_size": 3, "seq_len": 50, "embed_dim": 32, "num_heads": 4},  # Longer sequence
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\nTest case {i+1}: B={case['batch_size']}, T={case['seq_len']}, "
              f"d={case['embed_dim']}, heads={case['num_heads']}")
        
        # Create attention module
        attention = TemporalSelfAttention(
            embed_dim=case['embed_dim'],
            num_heads=case['num_heads'],
            dropout=0.1,
            max_len=512
        )
        
        # Create test input (embeddings from UnivariateFunctionLearner)
        u = torch.randn(case['batch_size'], case['seq_len'], case['embed_dim'])
        print(f"Input embeddings shape: {u.shape}")
        
        # Forward pass
        h, attn_weights = attention(u)
        
        print(f"Output embeddings shape: {h.shape}")
        print(f"Attention weights shape: {attn_weights.shape}")
        
        # Verify output shapes
        expected_h_shape = (case['batch_size'], case['seq_len'], case['embed_dim'])
        expected_attn_shape = (case['batch_size'], case['num_heads'], case['seq_len'], case['seq_len'])
        
        assert h.shape == expected_h_shape, f"Expected h shape {expected_h_shape}, got {h.shape}"
        assert attn_weights.shape == expected_attn_shape, f"Expected attn shape {expected_attn_shape}, got {attn_weights.shape}"
        
        # Test attention weights properties
        # Attention weights should be properly normalized (sum close to 1 along last dimension)
        attn_sum = attn_weights.sum(dim=-1)  # Sum over key positions
        
        # Check that attention weights are reasonably normalized (allowing for positional encoding effects)
        sum_min, sum_max = attn_sum.min().item(), attn_sum.max().item()
        assert 0.3 <= sum_min and sum_max <= 1.5, f"Attention weights poorly normalized: range [{sum_min:.3f}, {sum_max:.3f}]"
        print(f"✓ Attention weights reasonably normalized (sum range: [{sum_min:.3f}, {sum_max:.3f}])")
        
        # Attention weights should be non-negative
        assert (attn_weights >= 0).all(), "Attention weights contain negative values"
        print("✓ Attention weights are non-negative")
        
        # Test gradient computation
        loss = h.sum()
        loss.backward()
        
        # Check that gradients exist for all parameters
        for name, param in attention.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            print(f"Gradient norm for {name}: {param.grad.norm().item():.6f}")
        
        # Test residual connection (output should be different from input due to attention)
        residual_diff = torch.norm(h - u).item()
        print(f"Residual difference norm: {residual_diff:.6f}")
        assert residual_diff > 1e-6, "Output is identical to input (no attention effect)"
        
        print("✓ Forward pass, gradient computation, and attention properties verified")
    
    # Test with edge cases
    print("\nTesting edge cases...")
    
    # Test with sequence length of 1
    attention = TemporalSelfAttention(embed_dim=32, num_heads=4)
    u_single = torch.randn(2, 1, 32)  # Single time step
    h_single, attn_single = attention(u_single)
    assert h_single.shape == (2, 1, 32), "Single time step test failed"
    assert attn_single.shape == (2, 4, 1, 1), "Single time step attention shape failed"
    print("✓ Single time step test passed")
    
    # Test with maximum sequence length
    u_max = torch.randn(1, 512, 32)  # Max length
    h_max, attn_max = attention(u_max)
    assert h_max.shape == (1, 512, 32), "Max sequence length test failed"
    print("✓ Maximum sequence length test passed")
    
    print("\n✅ All TemporalSelfAttention tests passed!")


class InterpretableTimeEncoder(nn.Module):
    """
    Integrated model that combines univariate function learners and temporal attention.
    
    Processes each variable independently through both univariate learners and temporal
    attention modules, then stacks the outputs correctly.
    
    Args:
        num_variables: Number of input variables (e.g., 7 for ETT dataset)
        embed_dim: Embedding dimension for each variable
        hidden_dim: Hidden dimension for univariate learners
        num_heads: Number of attention heads
        dropout: Dropout probability
        max_len: Maximum sequence length for positional encoding
    """
    
    def __init__(self, num_variables: int, embed_dim: int = 32, hidden_dim: int = 64, 
                 num_heads: int = 4, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.num_variables = num_variables
        self.embed_dim = embed_dim
        
        # Create separate univariate learners for each variable (no parameter sharing)
        self.univariate_learners = nn.ModuleList([
            UnivariateFunctionLearner(
                in_features=1, 
                out_features=embed_dim, 
                hidden_features=hidden_dim
            ) for _ in range(num_variables)
        ])
        
        # Single temporal attention module (shared across variables)
        self.temporal_attention = TemporalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            max_len=max_len
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the integrated model.
        
        Args:
            x: Input tensor of shape (B, T_in, M) where M is num_variables
            
        Returns:
            h_stack: Final embeddings of shape (B, M, T_in, embed_dim)
            attn_stack: Attention weights of shape (B, M, num_heads, T_in, T_in)
        """
        batch_size, seq_len, num_vars = x.shape
        
        # Verify input has correct number of variables
        assert num_vars == self.num_variables, \
            f"Expected {self.num_variables} variables, got {num_vars}"
        
        all_h = []
        all_attn = []
        
        # Process each variable independently
        for i in range(self.num_variables):
            # Extract variable i: (B, T_in, 1)
            x_i = x[:, :, i:i+1]
            
            # Apply univariate function learner: (B, T_in, 1) -> (B, T_in, embed_dim)
            u_i = self.univariate_learners[i](x_i)
            
            # Apply temporal attention: (B, T_in, embed_dim) -> (B, T_in, embed_dim), (B, num_heads, T_in, T_in)
            h_i, attn_i = self.temporal_attention(u_i)
            
            all_h.append(h_i)
            all_attn.append(attn_i)
        
        # Stack outputs: (B, M, T_in, embed_dim)
        h_stack = torch.stack(all_h, dim=1)
        
        # Stack attention weights: (B, M, num_heads, T_in, T_in)
        attn_stack = torch.stack(all_attn, dim=1)
        
        return h_stack, attn_stack


def test_interpretable_time_encoder():
    """Test the InterpretableTimeEncoder with multivariate input."""
    print("Testing InterpretableTimeEncoder...")
    
    # Test with different configurations
    test_cases = [
        {"batch_size": 2, "seq_len": 10, "num_vars": 7, "embed_dim": 32, "num_heads": 4},
        {"batch_size": 4, "seq_len": 20, "num_vars": 5, "embed_dim": 64, "num_heads": 8},
        {"batch_size": 1, "seq_len": 15, "num_vars": 3, "embed_dim": 16, "num_heads": 2},
        {"batch_size": 3, "seq_len": 50, "num_vars": 7, "embed_dim": 32, "num_heads": 4},  # ETT-like
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\nTest case {i+1}: B={case['batch_size']}, T={case['seq_len']}, "
              f"M={case['num_vars']}, d={case['embed_dim']}, heads={case['num_heads']}")
        
        # Create integrated model
        model = InterpretableTimeEncoder(
            num_variables=case['num_vars'],
            embed_dim=case['embed_dim'],
            hidden_dim=64,
            num_heads=case['num_heads'],
            dropout=0.1,
            max_len=512
        )
        
        # Create test input (multivariate time series)
        x = torch.randn(case['batch_size'], case['seq_len'], case['num_vars'])
        print(f"Input shape: {x.shape}")
        
        # Forward pass
        h_stack, attn_stack = model(x)
        
        print(f"Output embeddings shape: {h_stack.shape}")
        print(f"Output attention weights shape: {attn_stack.shape}")
        
        # Verify output shapes
        expected_h_shape = (case['batch_size'], case['num_vars'], case['seq_len'], case['embed_dim'])
        expected_attn_shape = (case['batch_size'], case['num_vars'], case['num_heads'], case['seq_len'], case['seq_len'])
        
        assert h_stack.shape == expected_h_shape, f"Expected h shape {expected_h_shape}, got {h_stack.shape}"
        assert attn_stack.shape == expected_attn_shape, f"Expected attn shape {expected_attn_shape}, got {attn_stack.shape}"
        
        # Test that each variable is processed independently
        # Set model to evaluation mode to disable dropout for deterministic testing
        model.eval()
        with torch.no_grad():
            # Re-run forward pass in eval mode
            h_stack_eval, attn_stack_eval = model(x)
            
            # Extract embeddings for first variable and compare with direct processing
            x_0 = x[:, :, 0:1]  # First variable
            u_0 = model.univariate_learners[0](x_0)
            h_0_direct, attn_0_direct = model.temporal_attention(u_0)
            
            # Should match the first variable in the stacked output
            h_0_from_stack = h_stack_eval[:, 0, :, :]
            attn_0_from_stack = attn_stack_eval[:, 0, :, :, :]
            
            assert torch.allclose(h_0_direct, h_0_from_stack, atol=1e-5), \
                "Variable 0 embeddings don't match between direct and stacked processing"
            assert torch.allclose(attn_0_direct, attn_0_from_stack, atol=1e-5), \
                "Variable 0 attention weights don't match between direct and stacked processing"
        
        # Set back to training mode
        model.train()
        print("✓ Independent variable processing verified")
        
        # Test gradient computation
        loss = h_stack.sum()
        loss.backward()
        
        # Check that gradients exist for all parameters
        total_params = 0
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            total_params += param.numel()
            print(f"Gradient norm for {name}: {param.grad.norm().item():.6f}")
        
        print(f"✓ Total parameters: {total_params:,}")
        print("✓ Forward pass and gradient computation successful")
    
    # Test with ETT-like configuration
    print("\nTesting with ETT dataset configuration...")
    config = ModelConfig()  # Default ETT configuration
    model = InterpretableTimeEncoder(
        num_variables=config.num_variables,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        dropout=config.dropout,
        max_len=config.max_len
    )
    
    # Test with ETT-like input
    batch_size, seq_len = 4, 96  # Common forecasting window
    x_ett = torch.randn(batch_size, seq_len, config.num_variables)
    
    h_ett, attn_ett = model(x_ett)
    print(f"ETT test - Input: {x_ett.shape}")
    print(f"ETT test - Output embeddings: {h_ett.shape}")
    print(f"ETT test - Output attention: {attn_ett.shape}")
    
    expected_h = (batch_size, config.num_variables, seq_len, config.embed_dim)
    expected_attn = (batch_size, config.num_variables, config.num_heads, seq_len, seq_len)
    
    assert h_ett.shape == expected_h, f"ETT test failed: expected {expected_h}, got {h_ett.shape}"
    assert attn_ett.shape == expected_attn, f"ETT test failed: expected {expected_attn}, got {attn_ett.shape}"
    print("✓ ETT configuration test passed")
    
    # Test error handling
    print("\nTesting error handling...")
    try:
        # Wrong number of variables
        x_wrong = torch.randn(2, 10, 5)  # 5 variables instead of 7
        model(x_wrong)
        assert False, "Should have raised assertion error for wrong number of variables"
    except AssertionError as e:
        print(f"✓ Correctly caught error for wrong number of variables: {e}")
    
    print("\n✅ All InterpretableTimeEncoder tests passed!")


if __name__ == "__main__":
    # Run tests first
    test_univariate_function_learner()
    test_temporal_self_attention()
    test_interpretable_time_encoder()
    
    print("\n" + "="*60)
    print("🎨 VISUALIZATION SECTION")
    print("="*60)
    
    # Create comprehensive visualizations
    visualize_univariate_function_learner()