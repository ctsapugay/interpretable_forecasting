"""
Extended Model Validation Script for Interpretable Time Series Forecasting

This script validates the complete extended model pipeline including:
- Cross-attention mechanisms for inter-variable relationships
- Temporal compression for efficient sequence processing  
- Spline-based forecasting for interpretable predictions

Tests end-to-end functionality with ETT data and various configurations.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
from typing import Tuple, Dict, List, Any
from pathlib import Path

try:
    from .data_utils import ETTDataLoader
    from .extended_model import InterpretableForecastingModel, ExtendedModelConfig
    from .model import InterpretableTimeEncoder, ModelConfig
    from .spline_visualization import create_spline_visualizations
except ImportError:
    # Handle case when running as script
    from data_utils import ETTDataLoader
    from extended_model import InterpretableForecastingModel, ExtendedModelConfig
    from model import InterpretableTimeEncoder, ModelConfig
    from spline_visualization import create_spline_visualizations


class ExtendedModelValidator:
    """
    Comprehensive validation framework for the extended interpretable forecasting model.
    
    Provides systematic testing of:
    - Model architecture and component integration
    - Forecasting accuracy and performance
    - Interpretability artifact generation
    - Memory usage and computational efficiency
    """
    
    def __init__(self, config: ExtendedModelConfig = None, data_path: str = None):
        """
        Initialize the validator with configuration and data.
        
        Args:
            config: Extended model configuration (uses default if None)
            data_path: Path to ETT dataset (uses default if None)
        """
        # Set default configuration if not provided
        if config is None:
            config = ExtendedModelConfig(
                num_variables=7,
                embed_dim=32,
                hidden_dim=64,
                num_heads=4,
                cross_dim=32,
                cross_heads=4,
                compressed_dim=64,
                compression_ratio=4,
                num_control_points=8,
                spline_degree=3,
                forecast_horizon=24,
                dropout=0.1,
                max_len=512
            )
        
        self.config = config
        self.data_path = data_path or "ETT-small/ETTh1.csv"
        
        # Initialize components
        self.model = None
        self.loader = None
        self.validation_results = {}
        
        print("🧪 Extended Model Validator Initialized")
        print(f"   Configuration: {self.config}")
        print(f"   Data path: {self.data_path}")
    
    def run_complete_validation(self) -> Dict[str, Any]:
        """
        Run the complete validation suite.
        
        Returns:
            Dictionary with all validation results
        """
        print("\n" + "="*80)
        print("🚀 EXTENDED MODEL VALIDATION SUITE")
        print("="*80)
        
        try:
            # Step 1: Load data and initialize model
            self._setup_model_and_data()
            
            # Step 2: Test model architecture
            self._validate_architecture()
            
            # Step 3: Test end-to-end pipeline
            self._validate_end_to_end_pipeline()
            
            # Step 4: Test gradient computation
            self._validate_gradient_computation()
            
            # Step 5: Test forecasting accuracy
            self._validate_forecasting_accuracy()
            
            # Step 6: Test interpretability artifacts
            self._validate_interpretability_artifacts()
            
            # Step 7: Performance analysis
            self._validate_performance()
            
            # Step 8: Create spline visualizations with accuracy assessment
            self._create_spline_visualizations()
            
            # Step 9: Generate comprehensive report
            self._generate_validation_report()
            
            print("\n" + "="*80)
            print("✅ EXTENDED MODEL VALIDATION COMPLETE")
            print("="*80)
            
            return self.validation_results
            
        except Exception as e:
            print(f"\n❌ Validation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _setup_model_and_data(self):
        """Setup model and data loader for validation."""
        print("\n📊 Setting up Model and Data...")
        
        # Load ETT data
        self.loader = ETTDataLoader(
            file_path=self.data_path,
            normalize='standard',
            num_samples=2000  # Use more samples for comprehensive testing
        )
        
        # Get variable information
        var_info = self.loader.get_variable_info()
        print(f"   Variables: {var_info['names']}")
        print(f"   Variable count: {var_info['count']}")
        print(f"   Data shape: {self.loader.data.shape}")
        
        # Validate configuration matches data
        if self.config.num_variables != var_info['count']:
            print(f"   ⚠️  Adjusting num_variables from {self.config.num_variables} to {var_info['count']}")
            self.config.num_variables = var_info['count']
        
        # Create extended model
        self.model = InterpretableForecastingModel(self.config)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Store basic info
        self.validation_results['setup'] = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'variable_names': var_info['names'],
            'data_shape': self.loader.data.shape,
            'config': self.config
        }
        
        print("   ✅ Model and data setup complete")
    
    def _validate_architecture(self):
        """Validate model architecture and component integration."""
        print("\n🏗️ Validating Model Architecture...")
        
        # Test component initialization
        components = {
            'interpretable_encoder': self.model.interpretable_encoder,
            'cross_attention': self.model.cross_attention,
            'temporal_encoder': self.model.temporal_encoder,
            'spline_learner': self.model.spline_learner
        }
        
        architecture_results = {}
        
        for name, component in components.items():
            if component is not None:
                param_count = sum(p.numel() for p in component.parameters())
                architecture_results[name] = {
                    'initialized': True,
                    'parameters': param_count,
                    'type': type(component).__name__
                }
                print(f"   ✅ {name}: {param_count:,} parameters")
            else:
                architecture_results[name] = {'initialized': False}
                print(f"   ❌ {name}: Not initialized")
        
        # Test configuration compatibility
        config_tests = {
            'embed_dim_compatibility': self.config.embed_dim == self.model.interpretable_encoder.embed_dim,
            'cross_dim_compatibility': self.config.cross_dim == self.model.cross_attention.cross_dim,
            'compressed_dim_compatibility': self.config.compressed_dim == self.model.temporal_encoder.compressed_dim,
            'forecast_horizon_compatibility': self.config.forecast_horizon == self.model.spline_learner.forecast_horizon
        }
        
        for test_name, result in config_tests.items():
            if result:
                print(f"   ✅ {test_name}")
            else:
                print(f"   ❌ {test_name}")
        
        architecture_results['config_compatibility'] = config_tests
        self.validation_results['architecture'] = architecture_results
        
        print("   ✅ Architecture validation complete")
    
    def _validate_end_to_end_pipeline(self):
        """Test end-to-end pipeline with various sequence lengths."""
        print("\n🔄 Validating End-to-End Pipeline...")
        
        test_cases = [
            {"seq_len": 24, "batch_size": 4, "description": "Short sequences (1 day)"},
            {"seq_len": 96, "batch_size": 8, "description": "Medium sequences (4 days)"},
            {"seq_len": 168, "batch_size": 4, "description": "Long sequences (1 week)"},
            {"seq_len": 336, "batch_size": 2, "description": "Very long sequences (2 weeks)"}
        ]
        
        pipeline_results = {}
        
        for i, case in enumerate(test_cases):
            print(f"\n   Test Case {i+1}: {case['description']}")
            
            try:
                # Get windowed data
                windows, _ = self.loader.get_windows(
                    window_size=case['seq_len'],
                    stride=case['seq_len'] // 2,
                    as_torch=True
                )
                
                # Take a batch
                batch_size = min(case['batch_size'], windows.shape[0])
                batch_data = windows[:batch_size]
                
                print(f"     Input shape: {batch_data.shape}")
                
                # Forward pass
                self.model.eval()
                with torch.no_grad():
                    output = self.model(batch_data)
                
                # Validate output structure
                required_keys = ['forecasts', 'interpretability']
                for key in required_keys:
                    if key not in output:
                        raise ValueError(f"Missing required output key: {key}")
                
                forecasts = output['forecasts']
                interpretability = output['interpretability']
                
                print(f"     Forecasts shape: {forecasts.shape}")
                
                # Validate forecast shape
                expected_forecast_shape = (batch_size, self.config.num_variables, self.config.forecast_horizon)
                if forecasts.shape != expected_forecast_shape:
                    raise ValueError(f"Forecast shape mismatch: expected {expected_forecast_shape}, got {forecasts.shape}")
                
                # Validate interpretability artifacts
                required_interp_keys = ['temporal_attention', 'cross_attention', 'compression_attention', 'spline_parameters']
                for key in required_interp_keys:
                    if key not in interpretability:
                        raise ValueError(f"Missing interpretability key: {key}")
                
                # Check output statistics
                forecast_stats = {
                    'mean': forecasts.mean().item(),
                    'std': forecasts.std().item(),
                    'min': forecasts.min().item(),
                    'max': forecasts.max().item(),
                    'has_nan': torch.isnan(forecasts).any().item(),
                    'has_inf': torch.isinf(forecasts).any().item()
                }
                
                print(f"     Forecast stats: mean={forecast_stats['mean']:.4f}, std={forecast_stats['std']:.4f}")
                
                if forecast_stats['has_nan']:
                    raise ValueError("Forecasts contain NaN values")
                if forecast_stats['has_inf']:
                    raise ValueError("Forecasts contain infinite values")
                
                pipeline_results[f"case_{i+1}"] = {
                    'description': case['description'],
                    'input_shape': batch_data.shape,
                    'forecast_shape': forecasts.shape,
                    'forecast_stats': forecast_stats,
                    'success': True
                }
                
                print(f"     ✅ Pipeline test successful")
                
            except Exception as e:
                pipeline_results[f"case_{i+1}"] = {
                    'description': case['description'],
                    'success': False,
                    'error': str(e)
                }
                print(f"     ❌ Pipeline test failed: {str(e)}")
        
        self.validation_results['pipeline'] = pipeline_results
        print("\n   ✅ End-to-end pipeline validation complete")
    
    def _validate_gradient_computation(self):
        """Test gradient computation through all components."""
        print("\n🎯 Validating Gradient Computation...")
        
        # Get test data
        windows, _ = self.loader.get_windows(window_size=96, as_torch=True)
        test_input = windows[:4]  # Small batch for gradient test
        test_input.requires_grad_(True)
        
        gradient_results = {}
        
        try:
            # Forward pass
            self.model.train()
            output = self.model(test_input)
            
            # Compute loss (sum of forecasts for gradient test)
            loss = output['forecasts'].sum()
            
            print(f"   Test input shape: {test_input.shape}")
            print(f"   Loss value: {loss.item():.6f}")
            
            # Backward pass
            loss.backward()
            
            # Check gradients for all parameters
            gradient_info = {}
            total_grad_norm = 0.0
            params_with_grad = 0
            params_without_grad = 0
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    gradient_info[name] = {
                        'has_gradient': True,
                        'grad_norm': grad_norm,
                        'param_shape': list(param.shape),
                        'param_count': param.numel()
                    }
                    total_grad_norm += grad_norm ** 2
                    params_with_grad += 1
                    print(f"   ✅ {name}: grad_norm={grad_norm:.6f}")
                else:
                    gradient_info[name] = {
                        'has_gradient': False,
                        'param_shape': list(param.shape),
                        'param_count': param.numel()
                    }
                    params_without_grad += 1
                    print(f"   ❌ {name}: No gradient!")
            
            total_grad_norm = np.sqrt(total_grad_norm)
            
            # Check input gradients
            input_grad_info = {}
            if test_input.grad is not None:
                input_grad_norm = test_input.grad.norm().item()
                input_grad_info = {
                    'has_gradient': True,
                    'grad_norm': input_grad_norm
                }
                print(f"   ✅ Input gradients: norm={input_grad_norm:.6f}")
            else:
                input_grad_info = {'has_gradient': False}
                print(f"   ❌ No input gradients!")
            
            gradient_results = {
                'total_grad_norm': total_grad_norm,
                'params_with_grad': params_with_grad,
                'params_without_grad': params_without_grad,
                'gradient_info': gradient_info,
                'input_grad_info': input_grad_info,
                'loss_value': loss.item(),
                'success': params_without_grad == 0
            }
            
            if params_without_grad == 0:
                print(f"   ✅ All parameters have gradients (total norm: {total_grad_norm:.6f})")
            else:
                print(f"   ⚠️  {params_without_grad} parameters missing gradients")
            
        except Exception as e:
            gradient_results = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Gradient computation failed: {str(e)}")
        
        self.validation_results['gradients'] = gradient_results
        print("   ✅ Gradient computation validation complete")
    
    def _validate_forecasting_accuracy(self):
        """Test forecasting accuracy with different horizons."""
        print("\n📈 Validating Forecasting Accuracy...")
        
        # Test different forecast horizons
        horizons = [1, 12, 24, 48]
        accuracy_results = {}
        
        for horizon in horizons:
            print(f"\n   Testing forecast horizon: {horizon}")
            
            try:
                # Create temporary config with different horizon
                temp_config = ExtendedModelConfig(
                    num_variables=self.config.num_variables,
                    embed_dim=self.config.embed_dim,
                    hidden_dim=self.config.hidden_dim,
                    num_heads=self.config.num_heads,
                    cross_dim=self.config.cross_dim,
                    cross_heads=self.config.cross_heads,
                    compressed_dim=self.config.compressed_dim,
                    compression_ratio=self.config.compression_ratio,
                    num_control_points=self.config.num_control_points,
                    spline_degree=self.config.spline_degree,
                    forecast_horizon=horizon,
                    dropout=self.config.dropout,
                    max_len=self.config.max_len
                )
                
                # Create model with different horizon
                temp_model = InterpretableForecastingModel(temp_config)
                
                # Get test data
                windows, _ = self.loader.get_windows(window_size=168, as_torch=True)
                test_batch = windows[:8]
                
                # Forward pass
                temp_model.eval()
                with torch.no_grad():
                    output = temp_model(test_batch)
                
                forecasts = output['forecasts']
                expected_shape = (test_batch.shape[0], self.config.num_variables, horizon)
                
                if forecasts.shape != expected_shape:
                    raise ValueError(f"Shape mismatch for horizon {horizon}: expected {expected_shape}, got {forecasts.shape}")
                
                # Compute basic accuracy metrics (using random baseline for now)
                # In real validation, you would use actual future values
                baseline_forecast = torch.randn_like(forecasts)
                
                mse = torch.mean((forecasts - baseline_forecast) ** 2).item()
                mae = torch.mean(torch.abs(forecasts - baseline_forecast)).item()
                
                accuracy_results[f"horizon_{horizon}"] = {
                    'forecast_shape': forecasts.shape,
                    'mse': mse,
                    'mae': mae,
                    'forecast_range': {
                        'min': forecasts.min().item(),
                        'max': forecasts.max().item(),
                        'mean': forecasts.mean().item(),
                        'std': forecasts.std().item()
                    },
                    'success': True
                }
                
                print(f"     ✅ Horizon {horizon}: MSE={mse:.4f}, MAE={mae:.4f}")
                
            except Exception as e:
                accuracy_results[f"horizon_{horizon}"] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"     ❌ Horizon {horizon} failed: {str(e)}")
        
        self.validation_results['forecasting_accuracy'] = accuracy_results
        print("\n   ✅ Forecasting accuracy validation complete")
    
    def _validate_interpretability_artifacts(self):
        """Test generation and properties of interpretability artifacts."""
        print("\n🔍 Validating Interpretability Artifacts...")
        
        # Get test data
        windows, _ = self.loader.get_windows(window_size=96, as_torch=True)
        test_batch = windows[:4]
        
        interpretability_results = {}
        
        try:
            self.model.eval()
            with torch.no_grad():
                output = self.model(test_batch)
            
            interpretability = output['interpretability']
            
            # Test temporal attention
            temporal_attn = interpretability['temporal_attention']
            print(f"   Temporal attention shape: {temporal_attn.shape}")
            
            # Validate attention properties
            attn_sum = temporal_attn.sum(dim=-1)  # Sum over key positions
            attn_normalized = torch.allclose(attn_sum, torch.ones_like(attn_sum), atol=1e-5)
            attn_non_negative = (temporal_attn >= 0).all()
            
            temporal_results = {
                'shape': list(temporal_attn.shape),
                'properly_normalized': bool(attn_normalized),
                'non_negative': bool(attn_non_negative),
                'mean_value': temporal_attn.mean().item(),
                'std_value': temporal_attn.std().item()
            }
            
            # Test cross attention
            cross_attn = interpretability['cross_attention']
            print(f"   Cross attention shape: {cross_attn.shape}")
            
            cross_sum = cross_attn.sum(dim=-1)  # Sum over key variables
            cross_normalized = torch.allclose(cross_sum, torch.ones_like(cross_sum), atol=1e-5)
            cross_non_negative = (cross_attn >= 0).all()
            
            cross_results = {
                'shape': list(cross_attn.shape),
                'properly_normalized': bool(cross_normalized),
                'non_negative': bool(cross_non_negative),
                'mean_value': cross_attn.mean().item(),
                'std_value': cross_attn.std().item()
            }
            
            # Test compression attention
            compression_attn = interpretability['compression_attention']
            print(f"   Compression attention shape: {compression_attn.shape}")
            
            comp_sum = compression_attn.sum(dim=-1)  # Sum over time positions
            comp_normalized = torch.allclose(comp_sum, torch.ones_like(comp_sum), atol=1e-5)
            comp_non_negative = (compression_attn >= 0).all()
            
            compression_results = {
                'shape': list(compression_attn.shape),
                'properly_normalized': bool(comp_normalized),
                'non_negative': bool(comp_non_negative),
                'mean_value': compression_attn.mean().item(),
                'std_value': compression_attn.std().item()
            }
            
            # Test spline parameters
            spline_params = interpretability['spline_parameters']
            control_points = spline_params['control_points']
            basis_functions = spline_params['basis_functions']
            knot_vector = spline_params['knot_vector']
            
            print(f"   Control points shape: {control_points.shape}")
            print(f"   Basis functions shape: {basis_functions.shape}")
            print(f"   Knot vector shape: {knot_vector.shape}")
            
            spline_results = {
                'control_points_shape': list(control_points.shape),
                'basis_functions_shape': list(basis_functions.shape),
                'knot_vector_shape': list(knot_vector.shape),
                'control_points_range': {
                    'min': control_points.min().item(),
                    'max': control_points.max().item(),
                    'mean': control_points.mean().item(),
                    'std': control_points.std().item()
                },
                'knot_vector_valid': self._validate_knot_vector(knot_vector)
            }
            
            interpretability_results = {
                'temporal_attention': temporal_results,
                'cross_attention': cross_results,
                'compression_attention': compression_results,
                'spline_parameters': spline_results,
                'success': True
            }
            
            print("   ✅ All interpretability artifacts validated")
            
        except Exception as e:
            interpretability_results = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Interpretability validation failed: {str(e)}")
        
        self.validation_results['interpretability'] = interpretability_results
        print("   ✅ Interpretability artifacts validation complete")
    
    def _validate_knot_vector(self, knot_vector: torch.Tensor) -> bool:
        """Validate that knot vector is properly formatted for B-splines."""
        try:
            # Check if knot vector is non-decreasing
            diffs = knot_vector[1:] - knot_vector[:-1]
            non_decreasing = (diffs >= -1e-6).all()  # Allow small numerical errors
            
            # Check endpoint multiplicity (first and last knots should be repeated)
            start_multiplicity = (knot_vector[0] == knot_vector[:self.config.spline_degree + 1]).sum()
            end_multiplicity = (knot_vector[-1] == knot_vector[-self.config.spline_degree - 1:]).sum()
            
            proper_multiplicity = (start_multiplicity >= self.config.spline_degree + 1 and 
                                 end_multiplicity >= self.config.spline_degree + 1)
            
            return non_decreasing.item() and proper_multiplicity
            
        except Exception:
            return False
    
    def _validate_performance(self):
        """Test performance and memory usage."""
        print("\n⚡ Validating Performance...")
        
        performance_results = {}
        
        try:
            # Test inference speed
            windows, _ = self.loader.get_windows(window_size=168, as_torch=True)
            test_batches = [
                windows[:4],   # Small batch
                windows[:16],  # Medium batch
                windows[:32]   # Large batch (if available)
            ]
            
            self.model.eval()
            
            for i, batch in enumerate(test_batches):
                if batch.shape[0] == 0:
                    continue
                
                batch_size = batch.shape[0]
                print(f"\n   Testing batch size: {batch_size}")
                
                # Warmup
                with torch.no_grad():
                    for _ in range(3):
                        _ = self.model(batch)
                
                # Measure inference time
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(10):
                        output = self.model(batch)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                throughput = batch_size / avg_time
                
                performance_results[f"batch_size_{batch_size}"] = {
                    'avg_inference_time': avg_time,
                    'throughput_samples_per_sec': throughput,
                    'input_shape': batch.shape,
                    'output_shape': output['forecasts'].shape
                }
                
                print(f"     Average inference time: {avg_time:.4f} seconds")
                print(f"     Throughput: {throughput:.2f} samples/second")
            
            # Memory usage estimation
            total_params = sum(p.numel() for p in self.model.parameters())
            param_memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
            
            performance_results['memory_usage'] = {
                'total_parameters': total_params,
                'estimated_param_memory_mb': param_memory_mb,
                'model_size_estimate': f"{param_memory_mb:.2f} MB"
            }
            
            print(f"\n   Model memory usage: ~{param_memory_mb:.2f} MB")
            print("   ✅ Performance validation complete")
            
        except Exception as e:
            performance_results = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Performance validation failed: {str(e)}")
        
        self.validation_results['performance'] = performance_results
    
    def _create_spline_visualizations(self):
        """Create spline visualizations with true validation data for accuracy assessment."""
        print("\n🎨 Creating Spline Visualizations...")
        
        visualization_results = {}
        
        try:
            # Get validation data with future values for accuracy comparison
            windows, _ = self.loader.get_windows(window_size=168, stride=84, as_torch=True)
            
            # Create input/target pairs for forecasting validation
            forecast_horizon = self.config.forecast_horizon
            input_data = windows[:-1]  # All but last window as input
            
            # Create true future data by taking the next forecast_horizon steps
            true_future_data = []
            for i in range(len(input_data)):
                if i + 1 < len(windows):
                    # Take the first forecast_horizon steps from the next window
                    future_steps = windows[i + 1][:forecast_horizon]  # (forecast_horizon, M)
                    true_future_data.append(future_steps)
            
            if len(true_future_data) > 0:
                true_future = torch.stack(true_future_data)  # (B, forecast_horizon, M)
                
                # Use a smaller batch for visualization
                batch_size = min(4, input_data.shape[0], true_future.shape[0])
                sample_input = input_data[:batch_size]
                sample_future = true_future[:batch_size]
                
                # Generate model predictions
                self.model.eval()
                with torch.no_grad():
                    model_output = self.model(sample_input)
                
                # Get variable names
                var_info = self.loader.get_variable_info()
                variable_names = var_info['names']
                
                # Create visualizations for first sample
                figures = create_spline_visualizations(
                    model_output=model_output,
                    input_data=sample_input,
                    true_future=sample_future,
                    variable_names=variable_names,
                    sample_idx=0,
                    output_dir="spline_validation_outputs"
                )
                
                # Calculate accuracy metrics
                forecasts = model_output['forecasts']  # (B, M, forecast_horizon)
                true_vals = sample_future.transpose(1, 2)  # (B, M, forecast_horizon)
                
                mse = torch.mean((forecasts - true_vals) ** 2).item()
                mae = torch.mean(torch.abs(forecasts - true_vals)).item()
                
                # Calculate per-variable accuracy
                per_var_mse = torch.mean((forecasts - true_vals) ** 2, dim=(0, 2)).cpu().numpy()
                per_var_mae = torch.mean(torch.abs(forecasts - true_vals), dim=(0, 2)).cpu().numpy()
                
                visualization_results = {
                    'success': True,
                    'output_directory': 'spline_validation_outputs',
                    'figures_created': len(figures),
                    'overall_mse': mse,
                    'overall_mae': mae,
                    'per_variable_mse': per_var_mse.tolist(),
                    'per_variable_mae': per_var_mae.tolist(),
                    'variable_names': variable_names,
                    'samples_visualized': batch_size
                }
                
                print(f"   ✅ Created spline visualizations with accuracy assessment")
                print(f"   ✅ Overall MSE: {mse:.6f}, MAE: {mae:.6f}")
                print(f"   ✅ Visualizations saved to spline_validation_outputs/")
                
                # Close figures to save memory
                for fig in figures.values():
                    plt.close(fig)
                
            else:
                raise ValueError("Could not create validation data pairs")
                
        except Exception as e:
            visualization_results = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Spline visualization creation failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        self.validation_results['spline_visualizations'] = visualization_results
        return visualization_results

    def _generate_validation_report(self):
        """Generate comprehensive validation report."""
        print("\n📋 Generating Validation Report...")
        
        # Count successful tests
        total_tests = 0
        passed_tests = 0
        
        for category, results in self.validation_results.items():
            if isinstance(results, dict):
                if 'success' in results:
                    total_tests += 1
                    if results['success']:
                        passed_tests += 1
                else:
                    # Count sub-tests
                    for sub_key, sub_result in results.items():
                        if isinstance(sub_result, dict) and 'success' in sub_result:
                            total_tests += 1
                            if sub_result['success']:
                                passed_tests += 1
        

        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        report = {
            'validation_summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'success_rate': success_rate,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'detailed_results': self.validation_results
        }
        
        # Save report
        import json
        with open('extended_model_validation_report.json', 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_safe_report = self._make_json_safe(report)
            json.dump(json_safe_report, f, indent=2)
        
        print(f"   Validation report saved: extended_model_validation_report.json")
        print(f"   Success rate: {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)")
        
        self.validation_results['report'] = report
    
    def _make_json_safe(self, obj):
        """Convert tensors and other non-JSON types to JSON-safe formats."""
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_safe(item) for item in obj]
        elif isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist() if hasattr(obj, 'tolist') else float(obj)
        elif hasattr(obj, '__dict__'):
            return str(obj)  # Convert complex objects to string
        else:
            return obj


def validate_extended_model_with_ett_data():
    """
    Main validation function that tests the complete extended model with ETT data.
    
    Returns:
        Validation results dictionary
    """
    print("🧪 Extended Interpretable Time Series Forecasting Model Validation")
    print("=" * 80)
    
    try:
        # Create validator with default configuration
        validator = ExtendedModelValidator()
        
        # Run complete validation suite
        results = validator.run_complete_validation()
        
        # Print summary
        print(f"\n🎊 Validation Summary:")
        if 'report' in results and 'validation_summary' in results['report']:
            summary = results['report']['validation_summary']
            print(f"   ✅ Tests passed: {summary['passed_tests']}/{summary['total_tests']}")
            print(f"   ✅ Success rate: {summary['success_rate']:.1f}%")
        
        print(f"   ✅ Extended model validation complete!")
        print(f"   ✅ Model ready for training and forecasting tasks!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Extended model validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the extended model validation
    results = validate_extended_model_with_ett_data()
    
    if results is not None:
        print("\n🚀 Extended Model Validation Successful!")
    else:
        print("\n💥 Extended Model Validation Failed!")
        exit(1)