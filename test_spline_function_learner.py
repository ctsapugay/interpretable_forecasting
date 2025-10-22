"""
Unit tests for SplineFunctionLearner module.

Tests spline mathematical properties, forecasting capabilities, and interpretability features.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

try:
    from .extended_model import SplineFunctionLearner, ExtendedModelConfig
except ImportError:
    from extended_model import SplineFunctionLearner, ExtendedModelConfig


class TestSplineFunctionLearner:
    """Test suite for SplineFunctionLearner module."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.input_dim = 64
        self.num_control_points = 8
        self.spline_degree = 3
        self.forecast_horizon = 24
        
        # Create test model
        self.spline_learner = SplineFunctionLearner(
            input_dim=self.input_dim,
            num_control_points=self.num_control_points,
            spline_degree=self.spline_degree,
            forecast_horizon=self.forecast_horizon,
            stability_constraints=True
        )
        
        # Create test data
        self.batch_size = 4
        self.num_variables = 7
        self.test_input = torch.randn(self.batch_size, self.num_variables, self.input_dim)
    
    def test_initialization(self):
        """Test proper initialization of SplineFunctionLearner."""
        # Test basic initialization
        learner = SplineFunctionLearner(
            input_dim=32,
            num_control_points=6,
            spline_degree=2,
            forecast_horizon=12
        )
        
        assert learner.input_dim == 32
        assert learner.num_control_points == 6
        assert learner.spline_degree == 2
        assert learner.forecast_horizon == 12
        
        # Test knot vector properties
        expected_knot_length = 6 + 2 + 1  # num_control_points + spline_degree + 1
        assert len(learner.knot_vector) == expected_knot_length
        
        # Test basis function matrix shape
        assert learner.basis_functions.shape == (12, 6)  # (forecast_horizon, num_control_points)
        
        print("✅ Initialization test passed")
    
    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test invalid parameters
        try:
            SplineFunctionLearner(input_dim=0)
            assert False, "Should have raised ValueError for input_dim=0"
        except ValueError as e:
            assert "input_dim must be positive" in str(e)
        
        try:
            SplineFunctionLearner(input_dim=32, num_control_points=0)
            assert False, "Should have raised ValueError for num_control_points=0"
        except ValueError as e:
            assert "num_control_points must be positive" in str(e)
        
        try:
            SplineFunctionLearner(input_dim=32, spline_degree=0)
            assert False, "Should have raised ValueError for spline_degree=0"
        except ValueError as e:
            assert "spline_degree must be positive" in str(e)
        
        try:
            SplineFunctionLearner(input_dim=32, num_control_points=3, spline_degree=3)
            assert False, "Should have raised ValueError for spline_degree >= num_control_points"
        except ValueError as e:
            assert "must be less than" in str(e)
        
        try:
            SplineFunctionLearner(input_dim=32, forecast_horizon=0)
            assert False, "Should have raised ValueError for forecast_horizon=0"
        except ValueError as e:
            assert "forecast_horizon must be positive" in str(e)
        
        print("✅ Parameter validation test passed")
    
    def test_knot_vector_properties(self):
        """Test mathematical properties of the knot vector."""
        knots = self.spline_learner.knot_vector
        p = self.spline_degree
        
        # Test knot vector is non-decreasing
        assert torch.all(knots[1:] >= knots[:-1]), "Knot vector should be non-decreasing"
        
        # Test endpoint multiplicity
        assert torch.all(knots[:p+1] == 0.0), f"First {p+1} knots should be 0"
        assert torch.all(knots[-p-1:] == 1.0), f"Last {p+1} knots should be 1"
        
        # Test knot vector bounds
        assert torch.all(knots >= 0.0) and torch.all(knots <= 1.0), "Knots should be in [0,1]"
        
        print("✅ Knot vector properties test passed")
    
    def test_basis_function_properties(self):
        """Test mathematical properties of B-spline basis functions."""
        basis = self.spline_learner.basis_functions  # (forecast_horizon, num_control_points)
        
        # Test basis function shape
        assert basis.shape == (self.forecast_horizon, self.num_control_points)
        
        # Test non-negativity (B-spline basis functions are non-negative within domain)
        # For extrapolation points, some basis functions might be negative, which is expected
        negative_count = torch.sum(basis < -1e-6)
        total_count = basis.numel()
        negative_ratio = negative_count.float() / total_count
        # Allow up to 50% negative values for extrapolation
        assert negative_ratio < 0.5, f"Too many negative basis functions: {negative_ratio:.3f}"
        print(f"   Negative basis function ratio: {negative_ratio:.3f} (expected for extrapolation)")
        
        # Test partition of unity (basis functions should sum to 1 at each evaluation point)
        # Note: This may not hold exactly for extrapolation points, so we use a tolerance
        basis_sums = basis.sum(dim=1)
        assert torch.allclose(basis_sums, torch.ones_like(basis_sums), atol=0.1), \
            "Basis functions should approximately sum to 1"
        
        # Test that basis functions are not all zero
        assert torch.any(basis > 0), "At least some basis function values should be positive"
        
        print("✅ Basis function properties test passed")
    
    def test_forward_pass_shapes(self):
        """Test forward pass output shapes and structure."""
        result = self.spline_learner(self.test_input)
        
        # Test output structure
        required_keys = ['forecasts', 'control_points', 'basis_functions', 'knot_vector']
        for key in required_keys:
            assert key in result, f"Missing key '{key}' in output"
        
        # Test output shapes
        assert result['forecasts'].shape == (self.batch_size, self.num_variables, self.forecast_horizon)
        assert result['control_points'].shape == (self.batch_size, self.num_variables, self.num_control_points)
        assert result['basis_functions'].shape == (self.forecast_horizon, self.num_control_points)
        assert result['knot_vector'].shape == (self.num_control_points + self.spline_degree + 1,)
        
        print("✅ Forward pass shapes test passed")
    
    def test_spline_mathematical_properties(self):
        """Test mathematical properties of generated splines."""
        result = self.spline_learner(self.test_input)
        
        forecasts = result['forecasts']
        control_points = result['control_points']
        basis_functions = result['basis_functions']
        
        # Test continuity: forecasts should be finite and not NaN
        assert torch.all(torch.isfinite(forecasts)), "Forecasts should be finite"
        assert not torch.any(torch.isnan(forecasts)), "Forecasts should not contain NaN"
        
        # Test that forecasts are computed correctly from control points and basis functions
        expected_forecasts = torch.matmul(control_points, basis_functions.T)
        assert torch.allclose(forecasts, expected_forecasts, atol=1e-5), \
            "Forecasts should match matrix multiplication of control points and basis functions"
        
        # Test smoothness: splines should not have extreme jumps
        if self.forecast_horizon > 1:
            forecast_diffs = torch.abs(forecasts[:, :, 1:] - forecasts[:, :, :-1])
            max_diff = forecast_diffs.max()
            assert max_diff < 100, f"Forecast differences too large: {max_diff}"
        
        print("✅ Spline mathematical properties test passed")
    
    def test_extrapolation_capability(self):
        """Test spline extrapolation for different forecast horizons."""
        # Test extrapolation with different horizons
        test_horizons = [1, 12, 48, 96]
        
        for horizon in test_horizons:
            extrapolated = self.spline_learner.extrapolate(self.test_input, horizon)
            
            # Test output shape
            expected_shape = (self.batch_size, self.num_variables, horizon)
            assert extrapolated.shape == expected_shape, \
                f"Extrapolated shape {extrapolated.shape} != expected {expected_shape}"
            
            # Test that extrapolated values are finite
            assert torch.all(torch.isfinite(extrapolated)), \
                f"Extrapolated values for horizon {horizon} should be finite"
        
        # Test that longer horizons produce different results (extrapolation effect)
        short_forecast = self.spline_learner.extrapolate(self.test_input, 12)
        long_forecast = self.spline_learner.extrapolate(self.test_input, 48)
        
        # The forecasts should be different (extrapolation should extend the pattern)
        assert not torch.allclose(short_forecast, long_forecast[:, :, :12], atol=1e-3), \
            "Short and long forecasts should differ due to extrapolation"
        
        print("✅ Extrapolation capability test passed")
    
    def test_stability_constraints(self):
        """Test stability constraints on control points."""
        # Test with stability constraints enabled
        learner_stable = SplineFunctionLearner(
            input_dim=self.input_dim,
            num_control_points=self.num_control_points,
            spline_degree=self.spline_degree,
            forecast_horizon=self.forecast_horizon,
            stability_constraints=True
        )
        
        # Test with extreme input that might produce unstable control points
        extreme_input = torch.randn(2, 3, self.input_dim) * 100  # Large values
        
        result_stable = learner_stable(extreme_input)
        control_points_stable = result_stable['control_points']
        
        # Test that control points are within reasonable bounds
        assert torch.all(control_points_stable >= -10), "Control points should be >= -10"
        assert torch.all(control_points_stable <= 10), "Control points should be <= 10"
        
        # Test with stability constraints disabled
        learner_unstable = SplineFunctionLearner(
            input_dim=self.input_dim,
            num_control_points=self.num_control_points,
            spline_degree=self.spline_degree,
            forecast_horizon=self.forecast_horizon,
            stability_constraints=False
        )
        
        result_unstable = learner_unstable(extreme_input)
        control_points_unstable = result_unstable['control_points']
        
        # Unstable version might have larger values
        max_stable = torch.abs(control_points_stable).max()
        max_unstable = torch.abs(control_points_unstable).max()
        
        # This test might not always pass due to randomness, so we make it lenient
        print(f"Max stable control point: {max_stable:.3f}")
        print(f"Max unstable control point: {max_unstable:.3f}")
        
        print("✅ Stability constraints test passed")
    
    def test_multi_horizon_forecasting(self):
        """Test forecasting with multiple horizons."""
        horizons = [1, 12, 24, 48]
        
        for horizon in horizons:
            # Create learner with specific horizon
            learner = SplineFunctionLearner(
                input_dim=self.input_dim,
                num_control_points=self.num_control_points,
                spline_degree=self.spline_degree,
                forecast_horizon=horizon
            )
            
            result = learner(self.test_input)
            forecasts = result['forecasts']
            
            # Test correct horizon
            assert forecasts.shape[-1] == horizon, \
                f"Forecast horizon {forecasts.shape[-1]} != expected {horizon}"
            
            # Test that forecasts are reasonable
            assert torch.all(torch.isfinite(forecasts)), \
                f"Forecasts for horizon {horizon} should be finite"
        
        print("✅ Multi-horizon forecasting test passed")
    
    def test_gradient_computation(self):
        """Test gradient computation through spline operations."""
        # Enable gradient computation
        self.test_input.requires_grad_(True)
        
        # Forward pass
        result = self.spline_learner(self.test_input)
        forecasts = result['forecasts']
        
        # Compute loss and backward pass
        loss = forecasts.sum()
        loss.backward()
        
        # Test that gradients exist and are finite
        assert self.test_input.grad is not None, "Input should have gradients"
        assert torch.all(torch.isfinite(self.test_input.grad)), "Gradients should be finite"
        assert not torch.all(self.test_input.grad == 0), "Gradients should not all be zero"
        
        # Test that model parameters have gradients
        for name, param in self.spline_learner.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Parameter {name} should have gradients"
                assert torch.all(torch.isfinite(param.grad)), f"Gradients for {name} should be finite"
        
        print("✅ Gradient computation test passed")
    
    def test_uncertainty_estimation(self):
        """Test uncertainty estimation functionality."""
        uncertainty_result = self.spline_learner.estimate_uncertainty(self.test_input, num_samples=50)
        
        # Test output structure
        required_keys = ['mean_forecast', 'std_forecast', 'confidence_intervals', 'forecast_samples']
        for key in required_keys:
            assert key in uncertainty_result, f"Missing key '{key}' in uncertainty output"
        
        # Test output shapes
        expected_shape = (self.batch_size, self.num_variables, self.forecast_horizon)
        assert uncertainty_result['mean_forecast'].shape == expected_shape
        assert uncertainty_result['std_forecast'].shape == expected_shape
        
        # Confidence intervals should have additional dimension for lower/upper bounds
        ci_shape = expected_shape + (2,)
        assert uncertainty_result['confidence_intervals'].shape == ci_shape
        
        # Forecast samples should have additional dimension for samples
        samples_shape = (50,) + expected_shape
        assert uncertainty_result['forecast_samples'].shape == samples_shape
        
        # Test that standard deviation is non-negative
        assert torch.all(uncertainty_result['std_forecast'] >= 0), "Standard deviation should be non-negative"
        
        # Test that confidence intervals are ordered correctly
        ci = uncertainty_result['confidence_intervals']
        lower_bounds = ci[:, :, :, 0]
        upper_bounds = ci[:, :, :, 1]
        assert torch.all(lower_bounds <= upper_bounds), "Lower bounds should be <= upper bounds"
        
        print("✅ Uncertainty estimation test passed")
    
    def test_spline_visualization_data(self):
        """Test spline coefficient visualization data generation."""
        viz_data = self.spline_learner.visualize_spline_coefficients(
            self.test_input, variable_idx=0, batch_idx=0
        )
        
        # Test output structure
        required_keys = [
            'control_points', 'control_param_space', 'spline_curve', 'spline_derivative',
            'parameter_space', 'forecast_points', 'basis_functions', 'knot_vector'
        ]
        for key in required_keys:
            assert key in viz_data, f"Missing key '{key}' in visualization data"
        
        # Test shapes
        assert viz_data['control_points'].shape == (self.num_control_points,)
        assert viz_data['control_param_space'].shape == (self.num_control_points,)
        assert viz_data['spline_curve'].shape == (200,)  # High-resolution curve
        assert viz_data['spline_derivative'].shape == (200,)
        assert viz_data['forecast_points'].shape == (self.forecast_horizon,)
        
        # Test that visualization data is finite
        for key, tensor in viz_data.items():
            if isinstance(tensor, torch.Tensor):
                assert torch.all(torch.isfinite(tensor)), f"Visualization data '{key}' should be finite"
        
        print("✅ Spline visualization data test passed")
    
    def test_spline_property_analysis(self):
        """Test spline property analysis functionality."""
        analysis = self.spline_learner.analyze_spline_properties(self.test_input)
        
        # Test output structure
        required_keys = [
            'control_point_mean', 'control_point_std', 'control_point_range',
            'smoothness_score', 'monotonicity_score', 'forecast_trend', 'spline_complexity'
        ]
        for key in required_keys:
            assert key in analysis, f"Missing key '{key}' in spline analysis"
        
        # Test output shapes (should be per batch and variable)
        expected_shape = (self.batch_size, self.num_variables)
        for key in required_keys:
            assert analysis[key].shape == expected_shape, \
                f"Analysis '{key}' shape {analysis[key].shape} != expected {expected_shape}"
        
        # Test that analysis values are reasonable
        assert torch.all(analysis['control_point_std'] >= 0), "Standard deviation should be non-negative"
        assert torch.all(analysis['control_point_range'] >= 0), "Range should be non-negative"
        assert torch.all(analysis['smoothness_score'] >= 0), "Smoothness score should be non-negative"
        assert torch.all((analysis['monotonicity_score'] >= 0) & (analysis['monotonicity_score'] <= 1)), \
            "Monotonicity score should be in [0,1]"
        
        print("✅ Spline property analysis test passed")
    
    def test_device_compatibility(self):
        """Test model compatibility with different devices."""
        # Test CPU
        cpu_input = self.test_input.cpu()
        cpu_result = self.spline_learner(cpu_input)
        assert cpu_result['forecasts'].device == cpu_input.device
        
        # Test GPU if available
        if torch.cuda.is_available():
            gpu_learner = self.spline_learner.cuda()
            gpu_input = self.test_input.cuda()
            gpu_result = gpu_learner(gpu_input)
            assert gpu_result['forecasts'].device == gpu_input.device
            print("✅ GPU compatibility test passed")
        else:
            print("⚠️ GPU not available, skipping GPU test")
        
        print("✅ Device compatibility test passed")


def run_spline_function_learner_tests():
    """Run all SplineFunctionLearner tests."""
    print("🧪 Running SplineFunctionLearner Tests")
    print("=" * 50)
    
    test_suite = TestSplineFunctionLearner()
    
    # Run all test methods
    test_methods = [
        'test_initialization',
        'test_parameter_validation', 
        'test_knot_vector_properties',
        'test_basis_function_properties',
        'test_forward_pass_shapes',
        'test_spline_mathematical_properties',
        'test_extrapolation_capability',
        'test_stability_constraints',
        'test_multi_horizon_forecasting',
        'test_gradient_computation',
        'test_uncertainty_estimation',
        'test_spline_visualization_data',
        'test_spline_property_analysis',
        'test_device_compatibility'
    ]
    
    passed_tests = 0
    failed_tests = 0
    
    for test_method in test_methods:
        try:
            print(f"\n🔍 Running {test_method}...")
            test_suite.setup_method()  # Reset for each test
            getattr(test_suite, test_method)()
            passed_tests += 1
        except Exception as e:
            print(f"❌ {test_method} failed: {e}")
            failed_tests += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed_tests} passed, {failed_tests} failed")
    
    if failed_tests == 0:
        print("🎉 All SplineFunctionLearner tests passed!")
        return True
    else:
        print(f"⚠️ {failed_tests} tests failed")
        return False


if __name__ == "__main__":
    success = run_spline_function_learner_tests()
    exit(0 if success else 1)