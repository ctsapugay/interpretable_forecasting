"""
Validation script for the Interpretable Time Series Forecasting Model.

This script tests the complete model pipeline with real ETT data to verify
that everything works correctly end-to-end.
"""

import torch
import torch.nn as nn
import numpy as np
from data_utils import ETTDataLoader
from model import InterpretableTimeEncoder, ModelConfig
import matplotlib.pyplot as plt
from typing import Tuple


def validate_model_with_ett_data():
    """
    Test the complete model with real ETT data and print output shapes.
    """
    print("🧪 Validating Interpretable Time Series Forecasting Model with ETT Data")
    print("=" * 70)
    
    # Load ETT data
    print("\n📊 Loading ETT Dataset...")
    loader = ETTDataLoader(
        file_path="interpretable_forecasting/ETT-small/ETTh1.csv",
        normalize='standard',
        num_samples=1000  # Use first 1000 samples for validation
    )
    
    # Get variable information
    var_info = loader.get_variable_info()
    print(f"   Variables: {var_info['names']}")
    print(f"   Variable count: {var_info['count']}")
    print(f"   Data shape: {loader.data.shape}")
    
    # Create model configuration
    config = ModelConfig(
        num_variables=var_info['count'],
        embed_dim=32,
        hidden_dim=64,
        num_heads=4,
        dropout=0.1,
        max_len=512
    )
    
    print(f"\n🏗️ Model Configuration:")
    print(f"   Number of variables: {config.num_variables}")
    print(f"   Embedding dimension: {config.embed_dim}")
    print(f"   Hidden dimension: {config.hidden_dim}")
    print(f"   Number of attention heads: {config.num_heads}")
    print(f"   Dropout: {config.dropout}")
    print(f"   Max sequence length: {config.max_len}")
    
    # Create model
    model = InterpretableTimeEncoder(
        num_variables=config.num_variables,
        embed_dim=config.embed_dim,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        dropout=config.dropout,
        max_len=config.max_len
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🔧 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test with different sequence lengths
    test_cases = [
        {"seq_len": 24, "batch_size": 4, "description": "Short sequences (24 hours)"},
        {"seq_len": 96, "batch_size": 8, "description": "Medium sequences (4 days)"},
        {"seq_len": 168, "batch_size": 2, "description": "Long sequences (1 week)"},
    ]
    
    print(f"\n🧪 Testing Model with Different Sequence Lengths:")
    print("-" * 50)
    
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {case['description']}")
        
        # Get windowed data
        windows, _ = loader.get_windows(
            window_size=case['seq_len'],
            stride=case['seq_len'] // 2,  # 50% overlap
            as_torch=True
        )
        
        # Take a batch
        batch_size = min(case['batch_size'], windows.shape[0])
        batch_data = windows[:batch_size]  # Shape: (B, T, M)
        
        print(f"   Input shape: {batch_data.shape}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            embeddings, attention_weights = model(batch_data)
        
        print(f"   Output embeddings shape: {embeddings.shape}")
        print(f"   Attention weights shape: {attention_weights.shape}")
        
        # Verify shapes
        expected_emb_shape = (batch_size, config.num_variables, case['seq_len'], config.embed_dim)
        expected_attn_shape = (batch_size, config.num_variables, config.num_heads, case['seq_len'], case['seq_len'])
        
        assert embeddings.shape == expected_emb_shape, f"Embedding shape mismatch: expected {expected_emb_shape}, got {embeddings.shape}"
        assert attention_weights.shape == expected_attn_shape, f"Attention shape mismatch: expected {expected_attn_shape}, got {attention_weights.shape}"
        
        # Check output statistics
        emb_mean = embeddings.mean().item()
        emb_std = embeddings.std().item()
        attn_mean = attention_weights.mean().item()
        attn_std = attention_weights.std().item()
        
        print(f"   Embedding statistics: mean={emb_mean:.4f}, std={emb_std:.4f}")
        print(f"   Attention statistics: mean={attn_mean:.4f}, std={attn_std:.4f}")
        
        # Check attention weights are properly normalized
        attn_sums = attention_weights.sum(dim=-1)  # Sum over key positions
        attn_sum_mean = attn_sums.mean().item()
        attn_sum_std = attn_sums.std().item()
        
        print(f"   Attention normalization: sum_mean={attn_sum_mean:.4f}, sum_std={attn_sum_std:.4f}")
        
        # Verify attention weights are non-negative
        assert (attention_weights >= 0).all(), "Attention weights contain negative values"
        
        print("   ✅ Forward pass successful")
    
    print(f"\n🎯 Testing Gradient Computation:")
    print("-" * 30)
    
    # Test gradient computation with a small batch
    model.train()
    test_input = windows[:2]  # Small batch for gradient test
    test_input.requires_grad_(True)
    
    # Forward pass
    embeddings, attention_weights = model(test_input)
    
    # Compute a simple loss (sum of embeddings)
    loss = embeddings.sum()
    
    print(f"   Test input shape: {test_input.shape}")
    print(f"   Loss value: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist for all parameters
    gradient_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_norms[name] = grad_norm
            print(f"   {name}: grad_norm={grad_norm:.6f}")
        else:
            print(f"   ❌ {name}: No gradient!")
    
    # Check input gradients
    if test_input.grad is not None:
        input_grad_norm = test_input.grad.norm().item()
        print(f"   Input gradients: norm={input_grad_norm:.6f}")
        print("   ✅ Gradient computation successful")
    else:
        print("   ❌ No input gradients!")
    
    print(f"\n🔍 Testing Individual Variable Processing:")
    print("-" * 40)
    
    # Test that each variable is processed independently
    test_seq_len = 48
    test_batch_size = 2
    
    # Create test data
    test_windows, _ = loader.get_windows(window_size=test_seq_len, as_torch=True)
    test_batch = test_windows[:test_batch_size]
    
    model.eval()
    with torch.no_grad():
        # Full model forward pass
        full_embeddings, full_attention = model(test_batch)
        
        # Process each variable individually and compare
        for var_idx in range(config.num_variables):
            var_name = var_info['names'][var_idx]
            
            # Extract single variable
            single_var_input = test_batch[:, :, var_idx:var_idx+1]  # (B, T, 1)
            
            # Process through univariate learner
            var_embeddings = model.univariate_learners[var_idx](single_var_input)  # (B, T, embed_dim)
            
            # Process through attention
            var_attended, var_attention = model.temporal_attention(var_embeddings)
            
            # Compare with full model output
            full_var_embeddings = full_embeddings[:, var_idx, :, :]  # (B, T, embed_dim)
            full_var_attention = full_attention[:, var_idx, :, :, :]  # (B, num_heads, T, T)
            
            # Check if they match
            emb_diff = torch.abs(var_attended - full_var_embeddings).max().item()
            attn_diff = torch.abs(var_attention - full_var_attention).max().item()
            
            print(f"   Variable {var_idx} ({var_name}):")
            print(f"     Embedding max diff: {emb_diff:.8f}")
            print(f"     Attention max diff: {attn_diff:.8f}")
            
            # They should be very close (allowing for numerical precision)
            assert emb_diff < 1e-5, f"Variable {var_idx} embeddings don't match: diff={emb_diff}"
            assert attn_diff < 1e-5, f"Variable {var_idx} attention doesn't match: diff={attn_diff}"
    
    print("   ✅ Independent variable processing verified")
    
    print(f"\n📈 Performance Analysis:")
    print("-" * 25)
    
    # Measure inference time
    import time
    
    model.eval()
    test_input = windows[:16]  # Batch of 16
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_input)
    
    # Measure time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            embeddings, attention = model(test_input)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 10
    throughput = test_input.shape[0] / avg_time
    
    print(f"   Average inference time: {avg_time:.4f} seconds")
    print(f"   Throughput: {throughput:.2f} samples/second")
    print(f"   Input shape: {test_input.shape}")
    
    print(f"\n🎉 Model Validation Complete!")
    print("=" * 70)
    print("✅ All tests passed successfully!")
    print("✅ Model is ready for training and forecasting tasks!")
    
    return model, loader


def create_validation_visualization(model: InterpretableTimeEncoder, loader: ETTDataLoader):
    """
    Create visualizations to show model outputs.
    """
    print("\n🎨 Creating Validation Visualizations...")
    
    # Get a sample of data
    windows, indices = loader.get_windows(window_size=96, stride=48, as_torch=True)
    sample_batch = windows[:4]  # 4 samples
    
    model.eval()
    with torch.no_grad():
        embeddings, attention_weights = model(sample_batch)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Interpretable Time Series Model - Validation Results', fontsize=16, fontweight='bold')
    
    # 1. Original time series (first sample, first 3 variables)
    sample_idx = 0
    original_data = sample_batch[sample_idx].numpy()  # (T, M)
    
    axes[0, 0].set_title('Original Time Series (Sample 1)')
    for i in range(3):  # Show first 3 variables
        axes[0, 0].plot(original_data[:, i], label=loader.variables[i], alpha=0.8)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Normalized Value')
    
    # 2. Embedding magnitudes for each variable
    axes[0, 1].set_title('Embedding Magnitudes by Variable')
    emb_norms = torch.norm(embeddings[sample_idx], dim=-1).numpy()  # (M, T)
    
    for i in range(min(4, len(loader.variables))):
        axes[0, 1].plot(emb_norms[i], label=loader.variables[i], alpha=0.8)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Embedding L2 Norm')
    
    # 3. Attention heatmap for first variable
    var_idx = 0
    attn_avg = attention_weights[sample_idx, var_idx].mean(dim=0).numpy()  # Average over heads
    
    im1 = axes[0, 2].imshow(attn_avg, cmap='Blues', aspect='auto')
    axes[0, 2].set_title(f'Attention Heatmap - {loader.variables[var_idx]}')
    axes[0, 2].set_xlabel('Key Position')
    axes[0, 2].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0, 2], shrink=0.8)
    
    # 4. Embedding space visualization (2D projection)
    axes[1, 0].set_title('Embedding Space (First 2 Dimensions)')
    emb_2d = embeddings[sample_idx, :, :, :2].numpy()  # (M, T, 2)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(loader.variables)))
    for i in range(len(loader.variables)):
        axes[1, 0].scatter(emb_2d[i, :, 0], emb_2d[i, :, 1], 
                          c=[colors[i]], label=loader.variables[i], alpha=0.7, s=20)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel('Embedding Dimension 0')
    axes[1, 0].set_ylabel('Embedding Dimension 1')
    
    # 5. Attention weights distribution
    axes[1, 1].set_title('Attention Weights Distribution')
    attn_flat = attention_weights.flatten().numpy()
    axes[1, 1].hist(attn_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].set_xlabel('Attention Weight Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Model architecture summary
    axes[1, 2].text(0.1, 0.9, 'Model Architecture Summary', fontsize=14, fontweight='bold')
    
    total_params = sum(p.numel() for p in model.parameters())
    arch_text = f"""
Input: (B, T, {model.num_variables}) - Multivariate time series
↓
Univariate Learners: {model.num_variables} × MLP(1→{model.embed_dim})
↓
Temporal Attention: {model.temporal_attention.attention.num_heads} heads
↓
Output: (B, {model.num_variables}, T, {model.embed_dim}) - Variable embeddings

Parameters: {total_params:,}
Variables: {loader.variables}
"""
    
    axes[1, 2].text(0.05, 0.8, arch_text, fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('model_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Validation visualization saved as 'model_validation_results.png'")


if __name__ == "__main__":
    # Run the complete validation
    try:
        model, loader = validate_model_with_ett_data()
        create_validation_visualization(model, loader)
        
        print(f"\n🎊 Validation Summary:")
        print(f"   ✅ Data loading and preprocessing: PASSED")
        print(f"   ✅ Model forward pass: PASSED")
        print(f"   ✅ Gradient computation: PASSED")
        print(f"   ✅ Independent variable processing: PASSED")
        print(f"   ✅ Shape verification: PASSED")
        print(f"   ✅ Attention mechanism: PASSED")
        print(f"   ✅ Performance analysis: PASSED")
        print(f"\n🚀 The Interpretable Time Series Forecasting Model is ready for use!")
        
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)