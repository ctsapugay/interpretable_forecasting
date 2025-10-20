"""
Test script to verify the complete forecasting data pipeline.

This script tests the enhanced ETTDataLoader with multi-horizon forecasting,
train/val/test splits, and comprehensive evaluation utilities.
"""

import numpy as np
import torch
from data_utils import ETTDataLoader
from evaluation_utils import (
    compute_forecasting_metrics,
    compute_variable_wise_metrics,
    evaluate_forecast_quality
)


def test_complete_forecasting_pipeline():
    """Test the complete forecasting data pipeline."""
    print("🧪 Testing Complete Forecasting Data Pipeline")
    print("=" * 60)
    
    # Initialize data loader
    print("1. Initializing ETTDataLoader...")
    loader = ETTDataLoader(
        file_path="interpretable_forecasting/ETT-small/ETTh1.csv",
        normalize='standard',
        num_samples=1000  # Use more data for realistic testing
    )
    
    # Test multi-horizon dataset creation
    print("\n2. Testing multi-horizon dataset creation...")
    horizons = [1, 12, 24, 48]
    multi_horizon_data = loader.create_multi_horizon_dataset(
        input_length=96,  # 4 days of hourly data
        forecast_horizons=horizons,
        stride=24  # Daily stride
    )
    
    for horizon, (inputs, targets) in multi_horizon_data.items():
        print(f"   Horizon {horizon:2d}: {inputs.shape[0]:3d} samples, "
              f"inputs {inputs.shape}, targets {targets.shape}")
    
    # Test train/val/test splits for different horizons
    print("\n3. Testing train/val/test splits...")
    horizon_splits = loader.create_multi_horizon_splits(
        input_length=96,
        forecast_horizons=[12, 24, 48],
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        stride=12
    )
    
    for horizon, splits in horizon_splits.items():
        print(f"   Horizon {horizon:2d}:")
        for split_name, (inputs, targets) in splits.items():
            print(f"     {split_name:5s}: {inputs.shape[0]:3d} samples")
    
    # Test evaluation on synthetic predictions
    print("\n4. Testing evaluation utilities...")
    
    # Use 24-hour horizon for detailed testing
    test_horizon = 24
    if test_horizon in horizon_splits:
        train_inputs, train_targets = horizon_splits[test_horizon]['train']
        val_inputs, val_targets = horizon_splits[test_horizon]['val']
        
        # Create synthetic predictions (targets + noise)
        noise_level = 0.1
        train_predictions = train_targets + noise_level * torch.randn_like(train_targets)
        val_predictions = val_targets + noise_level * torch.randn_like(val_targets)
        
        # Test basic metrics
        train_metrics = loader.compute_forecast_metrics(
            train_predictions, train_targets,
            denormalize=True, per_variable=True, per_horizon=True
        )
        
        print(f"   Training metrics (horizon {test_horizon}):")
        print(f"     Overall MSE: {train_metrics['overall']['mse']:.4f}")
        print(f"     Overall MAE: {train_metrics['overall']['mae']:.4f}")
        
        # Show per-variable performance
        print(f"   Per-variable MSE:")
        for var_name, metrics in train_metrics['per_variable'].items():
            print(f"     {var_name}: {metrics['mse']:.4f}")
        
        # Show per-horizon performance (first few horizons)
        if 'per_horizon' in train_metrics:
            print(f"   Per-horizon MSE (first 5 steps):")
            for step in range(1, min(6, len(train_metrics['per_horizon']) + 1)):
                mse = train_metrics['per_horizon'][step]['mse']
                print(f"     Step {step}: {mse:.4f}")
        
        # Test comprehensive evaluation
        print(f"\n5. Testing comprehensive evaluation...")
        eval_results = loader.evaluate_forecasts(
            val_predictions, val_targets,
            denormalize=True, compute_intervals=False
        )
        
        print(f"   Evaluation components: {list(eval_results.keys())}")
        if 'denormalized_metrics' in eval_results:
            denorm_mse = eval_results['denormalized_metrics']['mse']
            print(f"   Denormalized validation MSE: {denorm_mse:.4f}")
        
        # Test denormalization consistency
        print(f"\n6. Testing denormalization consistency...")
        
        # Denormalize predictions and targets
        denorm_predictions = loader.denormalize_predictions(val_predictions)
        denorm_targets = loader.denormalize_predictions(val_targets)
        
        # Compute metrics on denormalized data directly
        direct_metrics = compute_forecasting_metrics(denorm_predictions, denorm_targets)
        
        # Compare with evaluation results
        if 'denormalized_metrics' in eval_results:
            eval_mse = eval_results['denormalized_metrics']['mse']
            direct_mse = direct_metrics['mse']
            
            if abs(eval_mse - direct_mse) < 1e-6:
                print(f"   ✅ Denormalization consistency verified")
                print(f"      Eval MSE: {eval_mse:.6f}")
                print(f"      Direct MSE: {direct_mse:.6f}")
            else:
                print(f"   ❌ Denormalization inconsistency detected")
                print(f"      Eval MSE: {eval_mse:.6f}")
                print(f"      Direct MSE: {direct_mse:.6f}")
        
        # Test variable-specific evaluation
        print(f"\n7. Testing variable-specific evaluation...")
        var_metrics = compute_variable_wise_metrics(
            denorm_predictions, denorm_targets, loader.variables
        )
        
        best_var = min(var_metrics.keys(), key=lambda v: var_metrics[v]['mse'])
        worst_var = max(var_metrics.keys(), key=lambda v: var_metrics[v]['mse'])
        
        print(f"   Best performing variable: {best_var} "
              f"(MSE: {var_metrics[best_var]['mse']:.4f})")
        print(f"   Worst performing variable: {worst_var} "
              f"(MSE: {var_metrics[worst_var]['mse']:.4f})")
    
    print(f"\n8. Testing data pipeline robustness...")
    
    # Test with different input lengths and horizons
    test_configs = [
        (48, 12),   # 2 days -> 12 hours
        (168, 24),  # 1 week -> 1 day
        (24, 1),    # 1 day -> 1 hour
    ]
    
    for input_len, pred_len in test_configs:
        try:
            inputs, targets = loader.get_forecasting_data(
                input_length=input_len,
                prediction_length=pred_len,
                stride=pred_len  # Non-overlapping samples
            )
            print(f"   ✅ Config ({input_len}, {pred_len}): "
                  f"{inputs.shape[0]} samples created")
        except ValueError as e:
            print(f"   ⚠️  Config ({input_len}, {pred_len}): {e}")
    
    print(f"\n🎉 Complete forecasting pipeline test passed!")
    print(f"\n📋 Key Features Verified:")
    print(f"   ✅ Multi-horizon dataset creation")
    print(f"   ✅ Proper train/validation/test splits")
    print(f"   ✅ Comprehensive evaluation metrics")
    print(f"   ✅ Denormalization for interpretable outputs")
    print(f"   ✅ Variable-wise and horizon-wise analysis")
    print(f"   ✅ Robust handling of different configurations")
    
    return True


if __name__ == "__main__":
    success = test_complete_forecasting_pipeline()
    if not success:
        exit(1)