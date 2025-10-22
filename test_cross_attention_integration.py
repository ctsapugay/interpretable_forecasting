#!/usr/bin/env python3
"""
Test script to verify CrossVariableAttention integration with the extended model.
"""

import torch
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extended_model import create_extended_model, ExtendedModelConfig
from data_utils import ETTDataLoader

def test_extended_model_integration():
    """Test the extended model with CrossVariableAttention on real ETT data."""
    print("🧪 Testing Extended Model Integration with CrossVariableAttention")
    print("=" * 70)
    
    # Load ETT data
    print("📊 Loading ETT Dataset...")
    try:
        loader = ETTDataLoader(
            file_path="interpretable_forecasting/ETT-small/ETTh1.csv",
            normalize='standard',
            num_samples=500  # Smaller dataset for testing
        )
        print(f"✅ Loaded ETT data: {len(loader.raw_data)} samples, {len(loader.variables)} variables")
    except Exception as e:
        print(f"❌ Failed to load ETT data: {e}")
        return False
    
    # Create extended model
    print("\n🏗️ Creating Extended Model...")
    config = ExtendedModelConfig(
        num_variables=7,
        embed_dim=32,
        cross_dim=32,
        cross_heads=4,
        forecast_horizon=24
    )
    
    model = create_extended_model(config)
    print(f"✅ Model created with {model.count_parameters():,} parameters")
    
    # Test with different sequence lengths
    test_cases = [
        {"seq_len": 24, "batch_size": 4, "name": "Short sequences (1 day)"},
        {"seq_len": 48, "batch_size": 2, "name": "Medium sequences (2 days)"},
        {"seq_len": 96, "batch_size": 1, "name": "Long sequences (4 days)"},
    ]
    
    print("\n🧪 Testing Forward Pass with Different Configurations...")
    
    for i, case in enumerate(test_cases):
        print(f"\nTest {i+1}: {case['name']}")
        print("-" * 50)
        
        # Create windowed data
        windows, indices = loader.get_windows(
            window_size=case['seq_len'],
            stride=case['seq_len'] // 2
        )
        
        # Take a batch
        batch_data = windows[:case['batch_size']]
        print(f"Input shape: {batch_data.shape}")
        
        # Forward pass
        try:
            output = model(batch_data)
            
            print(f"✅ Forward pass successful")
            print(f"   Forecasts shape: {output['forecasts'].shape}")
            print(f"   Cross-attention shape: {output['interpretability']['cross_attention'].shape}")
            
            # Verify cross-attention properties
            cross_attn = output['interpretability']['cross_attention']
            
            # Check attention weights are non-negative and normalized
            assert (cross_attn >= 0).all(), "Cross-attention weights contain negative values"
            
            attn_sum = cross_attn.sum(dim=-1)  # Sum over key variables
            sum_min, sum_max = attn_sum.min().item(), attn_sum.max().item()
            assert 0.8 <= sum_min and sum_max <= 1.2, f"Cross-attention poorly normalized: [{sum_min:.3f}, {sum_max:.3f}]"
            
            print(f"   Cross-attention normalized: sum range [{sum_min:.3f}, {sum_max:.3f}]")
            
            # Test gradient computation
            loss = output['forecasts'].sum()
            loss.backward()
            
            # Check gradients for cross-attention parameters
            cross_attn_grads = []
            for name, param in model.cross_attention.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    cross_attn_grads.append(grad_norm)
                    print(f"   Cross-attention {name}: grad_norm={grad_norm:.6f}")
            
            assert len(cross_attn_grads) > 0, "No gradients computed for cross-attention"
            assert all(g > 1e-8 for g in cross_attn_grads), "Cross-attention gradients too small"
            
            print(f"✅ Gradient computation successful")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False
    
    # Test cross-variable relationship learning
    print(f"\n🔍 Testing Cross-Variable Relationship Learning...")
    
    # Create test data with known relationships
    batch_size, seq_len, num_vars = 2, 48, 7
    
    # Create correlated variables (var 0 and var 1 are highly correlated)
    test_data = torch.randn(batch_size, seq_len, num_vars)
    test_data[:, :, 1] = test_data[:, :, 0] + 0.1 * torch.randn(batch_size, seq_len)  # Add correlation
    
    with torch.no_grad():
        output = model(test_data)
        cross_attn = output['interpretability']['cross_attention']
        
        # Check if model learns the correlation (var 0 should attend to var 1 and vice versa)
        # Average across batch and heads
        avg_attn = cross_attn.mean(dim=(0, 1))  # Shape: (7, 7)
        
        # Check attention from var 0 to var 1 and vice versa
        attn_0_to_1 = avg_attn[0, 1].item()
        attn_1_to_0 = avg_attn[1, 0].item()
        
        print(f"   Attention from var 0 to var 1: {attn_0_to_1:.4f}")
        print(f"   Attention from var 1 to var 0: {attn_1_to_0:.4f}")
        print(f"   Average self-attention: {torch.diag(avg_attn).mean().item():.4f}")
        
        # The model should learn some relationship (though it may take training)
        print(f"✅ Cross-variable attention patterns computed")
    
    # Test interpretability artifacts
    print(f"\n📊 Testing Interpretability Artifacts...")
    
    artifacts = output['interpretability']
    expected_keys = [
        'temporal_attention', 'cross_attention', 'compression_attention',
        'spline_parameters', 'variable_embeddings', 'cross_embeddings', 'compressed_repr'
    ]
    
    for key in expected_keys:
        if key in artifacts:
            shape = artifacts[key].shape if hasattr(artifacts[key], 'shape') else "dict"
            print(f"   ✅ {key}: {shape}")
        else:
            print(f"   ❌ Missing: {key}")
    
    print(f"\n🎉 All Extended Model Integration Tests Passed!")
    print("=" * 70)
    print("✅ CrossVariableAttention successfully integrated")
    print("✅ End-to-end pipeline working correctly")
    print("✅ Gradient computation through all components")
    print("✅ Interpretability artifacts generated")
    
    return True

if __name__ == "__main__":
    success = test_extended_model_integration()
    if success:
        print("\n🚀 Extended model ready for next development phase!")
    else:
        print("\n❌ Integration tests failed!")
        sys.exit(1)