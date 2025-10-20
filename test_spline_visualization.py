"""
Test script to generate spline visualizations with the extended model.

This script creates a minimal test to generate spline graphs showing
forecasts vs true validation data for accuracy assessment.
"""

import torch
import numpy as np
from pathlib import Path

try:
    from .data_utils import ETTDataLoader
    from .extended_model import InterpretableForecastingModel, ExtendedModelConfig
    from .spline_visualization import create_spline_visualizations
except ImportError:
    # Handle case when running as script
    from data_utils import ETTDataLoader
    from extended_model import InterpretableForecastingModel, ExtendedModelConfig
    from spline_visualization import create_spline_visualizations


def test_spline_visualization():
    """Test spline visualization with a simple model and synthetic data."""
    print("🧪 Testing Spline Visualization...")
    
    try:
        # Create a minimal configuration for testing
        config = ExtendedModelConfig(
            num_variables=7,
            embed_dim=16,  # Smaller for faster testing
            hidden_dim=32,
            num_heads=2,
            cross_dim=16,
            cross_heads=2,
            compressed_dim=32,
            compression_ratio=4,
            num_control_points=6,
            spline_degree=3,
            forecast_horizon=12,  # Shorter horizon for testing
            dropout=0.1,
            max_len=256
        )
        
        # Check if ETT data exists, otherwise create synthetic data
        data_path = "ETT-small/ETTh1.csv"
        if not Path(data_path).exists():
            print(f"   ETT data not found, creating synthetic data...")
            # Create synthetic time series data
            np.random.seed(42)  # For reproducible results
            
            # Generate synthetic ETT-like data with trends and seasonality
            time_steps = 1000
            variables = 7
            
            synthetic_data = np.zeros((time_steps, variables))
            for i in range(variables):
                # Base trend
                trend = np.linspace(0, 2, time_steps) + np.random.normal(0, 0.1, time_steps)
                # Seasonal component
                seasonal = 0.5 * np.sin(2 * np.pi * np.arange(time_steps) / 24) + \
                          0.3 * np.sin(2 * np.pi * np.arange(time_steps) / 168)
                # Random noise
                noise = np.random.normal(0, 0.2, time_steps)
                
                synthetic_data[:, i] = trend + seasonal + noise
            
            # Save synthetic data
            header = 'HUFL,HULL,MUFL,MULL,LUFL,LULL,OT'
            np.savetxt("synthetic_ett_data.csv", synthetic_data, delimiter=',', 
                      header=header, comments='')
            data_path = "synthetic_ett_data.csv"
            print(f"   ✅ Created synthetic data at {data_path}")
        
        # Load data
        loader = ETTDataLoader(file_path=data_path, normalize='standard', num_samples=500)
        var_info = loader.get_variable_info()
        
        # Adjust config to match data
        config.num_variables = var_info['count']
        
        # Create model
        model = InterpretableForecastingModel(config)
        print(f"   ✅ Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Get sample data for visualization
        windows, _ = loader.get_windows(window_size=96, stride=48, as_torch=True)
        
        # Create input/target pairs
        input_data = windows[:-1][:4]  # Use 4 samples
        
        # Create "true future" data from next windows
        true_future_data = []
        for i in range(len(input_data)):
            if i + 1 < len(windows):
                future_steps = windows[i + 1][:config.forecast_horizon]
                true_future_data.append(future_steps)
        
        true_future = torch.stack(true_future_data)  # (B, forecast_horizon, M)
        
        print(f"   Input data shape: {input_data.shape}")
        print(f"   True future shape: {true_future.shape}")
        
        # Generate model predictions
        model.eval()
        with torch.no_grad():
            model_output = model(input_data)
        
        print(f"   Forecast shape: {model_output['forecasts'].shape}")
        
        # Create spline visualizations
        figures = create_spline_visualizations(
            model_output=model_output,
            input_data=input_data,
            true_future=true_future,
            variable_names=var_info['names'],
            sample_idx=0,
            output_dir="test_spline_outputs"
        )
        
        # Calculate and display accuracy metrics
        forecasts = model_output['forecasts']  # (B, M, forecast_horizon)
        true_vals = true_future.transpose(1, 2)  # (B, M, forecast_horizon)
        
        mse = torch.mean((forecasts - true_vals) ** 2).item()
        mae = torch.mean(torch.abs(forecasts - true_vals)).item()
        
        print(f"\n📊 Accuracy Metrics:")
        print(f"   Overall MSE: {mse:.6f}")
        print(f"   Overall MAE: {mae:.6f}")
        
        # Per-variable metrics
        per_var_mse = torch.mean((forecasts - true_vals) ** 2, dim=(0, 2))
        per_var_mae = torch.mean(torch.abs(forecasts - true_vals), dim=(0, 2))
        
        print(f"\n📈 Per-Variable Accuracy:")
        for i, var_name in enumerate(var_info['names']):
            print(f"   {var_name}: MSE={per_var_mse[i]:.6f}, MAE={per_var_mae[i]:.6f}")
        
        print(f"\n✅ Spline visualization test completed successfully!")
        print(f"✅ Generated {len(figures)} visualization figures")
        print(f"✅ Check test_spline_outputs/ directory for results")
        
        return True
        
    except Exception as e:
        print(f"❌ Spline visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_validation_with_splines():
    """Run the full extended model validation including spline visualizations."""
    print("\n🚀 Running Full Extended Model Validation with Spline Visualizations")
    print("=" * 80)
    
    try:
        from validate_extended_model import ExtendedModelValidator
        
        # Create validator with default configuration
        validator = ExtendedModelValidator()
        
        # Run complete validation suite (now includes spline visualizations)
        results = validator.run_complete_validation()
        
        # Print spline visualization results
        if 'spline_visualizations' in results:
            spline_results = results['spline_visualizations']
            if spline_results['success']:
                print(f"\n🎨 Spline Visualization Results:")
                print(f"   ✅ Created {spline_results['figures_created']} figures")
                print(f"   ✅ Overall MSE: {spline_results['overall_mse']:.6f}")
                print(f"   ✅ Overall MAE: {spline_results['overall_mae']:.6f}")
                print(f"   ✅ Output directory: {spline_results['output_directory']}")
            else:
                print(f"   ❌ Spline visualization failed: {spline_results.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Full validation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("🎨 Spline Visualization Test Suite")
    print("=" * 50)
    
    # Test 1: Simple spline visualization
    test_success = test_spline_visualization()
    
    if test_success:
        print("\n" + "=" * 50)
        # Test 2: Full validation with splines (if simple test passes)
        full_results = run_full_validation_with_splines()
        
        if full_results:
            print("\n🎉 All spline visualization tests completed successfully!")
        else:
            print("\n💥 Full validation failed!")
    else:
        print("\n💥 Simple spline test failed!")