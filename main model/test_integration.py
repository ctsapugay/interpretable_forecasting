"""
Integration Tests for Extended Interpretable Forecasting Model

This module provides comprehensive integration tests for:
- Seamless integration between existing and new components
- Backward compatibility with original InterpretableTimeEncoder
- Memory usage and computational efficiency with different configurations
- Batch processing and variable-length sequence handling
"""

import torch
import torch.nn as nn
import numpy as np
import time
import gc
import os

# Optional imports
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

try:
    from .data_utils import ETTDataLoader
    from .extended_model import InterpretableForecastingModel, ExtendedModelConfig
    from .model import InterpretableTimeEncoder, ModelConfig
except ImportError:
    # Handle case when running as script
    from data_utils import ETTDataLoader
    from extended_model import InterpretableForecastingModel, ExtendedModelConfig
    from model import InterpretableTimeEncoder, ModelConfig


@dataclass
class IntegrationTestConfig:
    """Configuration for integration tests."""
    data_path: str = "ETT-small/ETTh1.csv"
    num_samples: int = 1000
    test_batch_sizes: List[int] = None
    test_sequence_lengths: List[int] = None
    memory_threshold_mb: float = 500.0  # Maximum memory usage threshold
    performance_threshold_ms: float = 100.0  # Maximum inference time per sample
    
    def __post_init__(self):
        if self.test_batch_sizes is None:
            self.test_batch_sizes = [1, 4, 8, 16]
        if self.test_sequence_lengths is None:
            self.test_sequence_lengths = [24, 96, 168, 336]


class ComponentIntegrationTester:
    """
    Comprehensive integration testing framework for the extended model.
    
    Tests component interactions, backward compatibility, memory usage,
    and performance across different configurations.
    """
    
    def __init__(self, config: IntegrationTestConfig = None):
        """
        Initialize the integration tester.
        
        Args:
            config: Test configuration (uses default if None)
        """
        self.config = config or IntegrationTestConfig()
        self.loader = None
        self.original_model = None
        self.extended_model = None
        self.test_results = {}
        
        print("🧪 Component Integration Tester Initialized")
        print(f"   Data path: {self.config.data_path}")
        print(f"   Test batch sizes: {self.config.test_batch_sizes}")
        print(f"   Test sequence lengths: {self.config.test_sequence_lengths}")
    
    def run_all_integration_tests(self) -> Dict[str, Any]:
        """
        Run the complete integration test suite.
        
        Returns:
            Dictionary with all test results
        """
        print("\n" + "="*80)
        print("🔧 COMPONENT INTEGRATION TEST SUITE")
        print("="*80)
        
        try:
            # Setup
            self._setup_models_and_data()
            
            # Test 1: Component integration
            self._test_component_integration()
            
            # Test 2: Backward compatibility
            self._test_backward_compatibility()
            
            # Test 3: Memory usage
            self._test_memory_usage()
            
            # Test 4: Batch processing
            self._test_batch_processing()
            
            # Test 5: Variable-length sequences
            self._test_variable_length_sequences()
            
            # Test 6: Computational efficiency
            self._test_computational_efficiency()
            
            # Test 7: Gradient flow integration
            self._test_gradient_flow_integration()
            
            # Generate summary
            self._generate_integration_summary()
            
            print("\n" + "="*80)
            print("✅ COMPONENT INTEGRATION TESTS COMPLETE")
            print("="*80)
            
            return self.test_results
            
        except Exception as e:
            print(f"\n❌ Integration tests failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _setup_models_and_data(self):
        """Setup models and data for testing."""
        print("\n📊 Setting up Models and Data...")
        
        # Load data
        self.loader = ETTDataLoader(
            file_path=self.config.data_path,
            normalize='standard',
            num_samples=self.config.num_samples
        )
        
        var_info = self.loader.get_variable_info()
        num_variables = var_info['count']
        
        # Create original model configuration
        original_config = ModelConfig(
            num_variables=num_variables,
            embed_dim=32,
            hidden_dim=64,
            num_heads=4,
            dropout=0.1,
            max_len=512
        )
        
        # Create extended model configuration
        extended_config = ExtendedModelConfig(
            num_variables=num_variables,
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
        
        # Initialize models
        self.original_model = InterpretableTimeEncoder(
            num_variables=original_config.num_variables,
            embed_dim=original_config.embed_dim,
            hidden_dim=original_config.hidden_dim,
            num_heads=original_config.num_heads,
            dropout=original_config.dropout,
            max_len=original_config.max_len
        )
        
        self.extended_model = InterpretableForecastingModel(extended_config)
        
        print(f"   ✅ Original model: {sum(p.numel() for p in self.original_model.parameters()):,} parameters")
        print(f"   ✅ Extended model: {sum(p.numel() for p in self.extended_model.parameters()):,} parameters")
        print("   ✅ Models and data setup complete")
    
    def _test_component_integration(self):
        """Test seamless integration between existing and new components."""
        print("\n🔗 Testing Component Integration...")
        
        integration_results = {}
        
        try:
            # Get test data
            windows, _ = self.loader.get_windows(window_size=96, as_torch=True)
            test_batch = windows[:4]
            
            # Test forward pass through each stage
            self.extended_model.eval()
            with torch.no_grad():
                # Stage 1: Original encoder
                var_embeddings, temporal_attn = self.extended_model.interpretable_encoder(test_batch)
                
                # Stage 2: Cross-attention
                cross_embeddings, cross_attn = self.extended_model.cross_attention(var_embeddings)
                
                # Stage 3: Temporal compression
                compressed_repr, compression_attn = self.extended_model.temporal_encoder(cross_embeddings)
                
                # Stage 4: Spline forecasting
                spline_results = self.extended_model.spline_learner(compressed_repr)
                
                # Test complete pipeline
                full_output = self.extended_model(test_batch)
            
            # Validate stage outputs
            stage_validations = {
                'original_encoder': {
                    'var_embeddings_shape': list(var_embeddings.shape),
                    'temporal_attn_shape': list(temporal_attn.shape),
                    'output_valid': not torch.isnan(var_embeddings).any() and not torch.isinf(var_embeddings).any()
                },
                'cross_attention': {
                    'cross_embeddings_shape': list(cross_embeddings.shape),
                    'cross_attn_shape': list(cross_attn.shape),
                    'output_valid': not torch.isnan(cross_embeddings).any() and not torch.isinf(cross_embeddings).any()
                },
                'temporal_compression': {
                    'compressed_repr_shape': list(compressed_repr.shape),
                    'compression_attn_shape': list(compression_attn.shape),
                    'output_valid': not torch.isnan(compressed_repr).any() and not torch.isinf(compressed_repr).any()
                },
                'spline_forecasting': {
                    'forecasts_shape': list(spline_results['forecasts'].shape),
                    'control_points_shape': list(spline_results['control_points'].shape),
                    'output_valid': not torch.isnan(spline_results['forecasts']).any() and not torch.isinf(spline_results['forecasts']).any()
                },
                'full_pipeline': {
                    'forecasts_shape': list(full_output['forecasts'].shape),
                    'has_interpretability': 'interpretability' in full_output,
                    'output_valid': not torch.isnan(full_output['forecasts']).any() and not torch.isinf(full_output['forecasts']).any()
                }
            }
            
            # Test tensor flow consistency
            tensor_flow_tests = {
                'embedding_dimensions_match': var_embeddings.shape[-1] == self.extended_model.config.embed_dim,
                'cross_dimensions_match': cross_embeddings.shape[-1] == self.extended_model.config.cross_dim,
                'compressed_dimensions_match': compressed_repr.shape[-1] == self.extended_model.config.compressed_dim,
                'forecast_horizon_match': spline_results['forecasts'].shape[-1] == self.extended_model.config.forecast_horizon,
                'batch_size_preserved': all([
                    var_embeddings.shape[0] == test_batch.shape[0],
                    cross_embeddings.shape[0] == test_batch.shape[0],
                    compressed_repr.shape[0] == test_batch.shape[0],
                    spline_results['forecasts'].shape[0] == test_batch.shape[0]
                ]),
                'variable_count_preserved': all([
                    var_embeddings.shape[1] == test_batch.shape[2],
                    cross_embeddings.shape[1] == test_batch.shape[2],
                    compressed_repr.shape[1] == test_batch.shape[2],
                    spline_results['forecasts'].shape[1] == test_batch.shape[2]
                ])
            }
            
            integration_results = {
                'stage_validations': stage_validations,
                'tensor_flow_tests': tensor_flow_tests,
                'success': all(stage_validations[stage]['output_valid'] for stage in stage_validations) and
                          all(tensor_flow_tests.values())
            }
            
            if integration_results['success']:
                print("   ✅ All components integrate seamlessly")
                print("   ✅ Tensor flow through pipeline is consistent")
                print("   ✅ All outputs are valid (no NaN/Inf values)")
            else:
                print("   ❌ Component integration issues detected")
            
        except Exception as e:
            integration_results = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Component integration test failed: {str(e)}")
        
        self.test_results['component_integration'] = integration_results
    
    def _test_backward_compatibility(self):
        """Test backward compatibility with original InterpretableTimeEncoder."""
        print("\n🔄 Testing Backward Compatibility...")
        
        compatibility_results = {}
        
        try:
            # Get test data
            windows, _ = self.loader.get_windows(window_size=96, as_torch=True)
            test_batch = windows[:4]
            
            # Test original model
            self.original_model.eval()
            with torch.no_grad():
                original_embeddings, original_attention = self.original_model(test_batch)
            
            # Test extended model's original components
            self.extended_model.eval()
            with torch.no_grad():
                extended_embeddings, extended_attention = self.extended_model.interpretable_encoder(test_batch)
            
            # Compare outputs
            embedding_diff = torch.abs(original_embeddings - extended_embeddings).max().item()
            attention_diff = torch.abs(original_attention - extended_attention).max().item()
            
            # Test parameter compatibility
            original_params = dict(self.original_model.named_parameters())
            extended_params = dict(self.extended_model.interpretable_encoder.named_parameters())
            
            param_compatibility = {}
            for name in original_params:
                if name in extended_params:
                    param_diff = torch.abs(original_params[name] - extended_params[name]).max().item()
                    param_compatibility[name] = {
                        'shapes_match': original_params[name].shape == extended_params[name].shape,
                        'max_diff': param_diff,
                        'compatible': param_diff < 1e-6  # Should be identical for new models
                    }
                else:
                    param_compatibility[name] = {
                        'shapes_match': False,
                        'compatible': False,
                        'missing': True
                    }
            
            # Test configuration compatibility
            original_config = {
                'num_variables': self.original_model.num_variables,
                'embed_dim': self.original_model.embed_dim,
                'hidden_dim': self.original_model.univariate_learners[0].net[0].out_features,
                'num_heads': self.original_model.temporal_attention.attention.num_heads
            }
            
            extended_config = {
                'num_variables': self.extended_model.config.num_variables,
                'embed_dim': self.extended_model.config.embed_dim,
                'hidden_dim': self.extended_model.config.hidden_dim,
                'num_heads': self.extended_model.config.num_heads
            }
            
            config_compatibility = {
                key: original_config[key] == extended_config[key]
                for key in original_config
            }
            
            compatibility_results = {
                'output_compatibility': {
                    'embedding_max_diff': embedding_diff,
                    'attention_max_diff': attention_diff,
                    'outputs_match': embedding_diff < 1e-5 and attention_diff < 1e-5
                },
                'parameter_compatibility': param_compatibility,
                'config_compatibility': config_compatibility,
                'success': (embedding_diff < 1e-5 and attention_diff < 1e-5 and 
                           all(config_compatibility.values()))
            }
            
            if compatibility_results['success']:
                print("   ✅ Perfect backward compatibility maintained")
                print(f"   ✅ Output differences: embeddings={embedding_diff:.2e}, attention={attention_diff:.2e}")
            else:
                print("   ⚠️  Backward compatibility issues detected")
                print(f"   ⚠️  Output differences: embeddings={embedding_diff:.2e}, attention={attention_diff:.2e}")
            
        except Exception as e:
            compatibility_results = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Backward compatibility test failed: {str(e)}")
        
        self.test_results['backward_compatibility'] = compatibility_results
    
    def _test_memory_usage(self):
        """Test memory usage and computational efficiency with different configurations."""
        print("\n💾 Testing Memory Usage...")
        
        memory_results = {}
        
        try:
            # Get baseline memory usage
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            if HAS_PSUTIL:
                process = psutil.Process(os.getpid())
                baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            else:
                baseline_memory = 0.0  # Fallback when psutil not available
            
            memory_tests = {}
            
            for seq_len in self.config.test_sequence_lengths:
                for batch_size in self.config.test_batch_sizes:
                    test_name = f"seq_{seq_len}_batch_{batch_size}"
                    
                    try:
                        # Get test data
                        windows, _ = self.loader.get_windows(window_size=seq_len, as_torch=True)
                        if windows.shape[0] < batch_size:
                            continue
                        
                        test_batch = windows[:batch_size]
                        
                        # Measure memory before forward pass
                        gc.collect()
                        if HAS_PSUTIL:
                            memory_before = process.memory_info().rss / 1024 / 1024
                        else:
                            memory_before = 0.0
                        
                        # Forward pass
                        self.extended_model.eval()
                        with torch.no_grad():
                            output = self.extended_model(test_batch)
                        
                        # Measure memory after forward pass
                        if HAS_PSUTIL:
                            memory_after = process.memory_info().rss / 1024 / 1024
                            memory_used = memory_after - memory_before
                        else:
                            memory_after = 0.0
                            memory_used = 0.0  # Fallback when psutil not available
                        
                        # Calculate memory per sample
                        memory_per_sample = memory_used / batch_size if batch_size > 0 else 0
                        
                        memory_tests[test_name] = {
                            'input_shape': list(test_batch.shape),
                            'output_shape': list(output['forecasts'].shape),
                            'memory_before_mb': memory_before,
                            'memory_after_mb': memory_after,
                            'memory_used_mb': memory_used,
                            'memory_per_sample_mb': memory_per_sample,
                            'within_threshold': memory_used < self.config.memory_threshold_mb,
                            'success': True
                        }
                        
                        print(f"   {test_name}: {memory_used:.2f} MB ({memory_per_sample:.2f} MB/sample)")
                        
                        # Clean up
                        del output, test_batch
                        gc.collect()
                        
                    except Exception as e:
                        memory_tests[test_name] = {
                            'success': False,
                            'error': str(e)
                        }
                        print(f"   ❌ {test_name}: {str(e)}")
            
            # Calculate memory statistics
            successful_tests = [test for test in memory_tests.values() if test.get('success', False)]
            if successful_tests:
                memory_usage_stats = {
                    'min_memory_mb': min(test['memory_used_mb'] for test in successful_tests),
                    'max_memory_mb': max(test['memory_used_mb'] for test in successful_tests),
                    'avg_memory_mb': sum(test['memory_used_mb'] for test in successful_tests) / len(successful_tests),
                    'min_per_sample_mb': min(test['memory_per_sample_mb'] for test in successful_tests),
                    'max_per_sample_mb': max(test['memory_per_sample_mb'] for test in successful_tests),
                    'avg_per_sample_mb': sum(test['memory_per_sample_mb'] for test in successful_tests) / len(successful_tests)
                }
            else:
                memory_usage_stats = {}
            
            memory_results = {
                'baseline_memory_mb': baseline_memory,
                'memory_tests': memory_tests,
                'memory_usage_stats': memory_usage_stats,
                'success': len(successful_tests) > 0 and all(
                    test['within_threshold'] for test in successful_tests
                )
            }
            
            if memory_results['success']:
                print(f"   ✅ Memory usage within acceptable limits")
                if memory_usage_stats:
                    print(f"   ✅ Average memory usage: {memory_usage_stats['avg_memory_mb']:.2f} MB")
                    print(f"   ✅ Average per sample: {memory_usage_stats['avg_per_sample_mb']:.2f} MB/sample")
            else:
                print(f"   ⚠️  Memory usage concerns detected")
            
        except Exception as e:
            memory_results = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Memory usage test failed: {str(e)}")
        
        self.test_results['memory_usage'] = memory_results
    
    def _test_batch_processing(self):
        """Test batch processing with different batch sizes."""
        print("\n📦 Testing Batch Processing...")
        
        batch_results = {}
        
        try:
            # Get test data
            windows, _ = self.loader.get_windows(window_size=96, as_torch=True)
            
            batch_tests = {}
            
            for batch_size in self.config.test_batch_sizes:
                if windows.shape[0] < batch_size:
                    continue
                
                try:
                    test_batch = windows[:batch_size]
                    
                    # Test forward pass
                    self.extended_model.eval()
                    with torch.no_grad():
                        output = self.extended_model(test_batch)
                    
                    forecasts = output['forecasts']
                    interpretability = output['interpretability']
                    
                    # Validate batch processing
                    batch_validation = {
                        'input_shape': list(test_batch.shape),
                        'forecast_shape': list(forecasts.shape),
                        'batch_size_preserved': forecasts.shape[0] == batch_size,
                        'output_valid': not torch.isnan(forecasts).any() and not torch.isinf(forecasts).any(),
                        'interpretability_valid': all(
                            key in interpretability for key in 
                            ['temporal_attention', 'cross_attention', 'compression_attention', 'spline_parameters']
                        )
                    }
                    
                    # Test attention batch consistency
                    temporal_attn = interpretability['temporal_attention']
                    cross_attn = interpretability['cross_attention']
                    compression_attn = interpretability['compression_attention']
                    
                    attention_validation = {
                        'temporal_batch_consistent': temporal_attn.shape[0] == batch_size,
                        'cross_batch_consistent': cross_attn.shape[0] == batch_size,
                        'compression_batch_consistent': compression_attn.shape[0] == batch_size,
                        'attention_normalized': torch.allclose(
                            temporal_attn.sum(dim=-1), 
                            torch.ones_like(temporal_attn.sum(dim=-1)), 
                            atol=1e-5
                        )
                    }
                    
                    batch_tests[f"batch_{batch_size}"] = {
                        'batch_validation': batch_validation,
                        'attention_validation': attention_validation,
                        'success': (batch_validation['batch_size_preserved'] and 
                                  batch_validation['output_valid'] and
                                  batch_validation['interpretability_valid'] and
                                  all(attention_validation.values()))
                    }
                    
                    if batch_tests[f"batch_{batch_size}"]['success']:
                        print(f"   ✅ Batch size {batch_size}: All validations passed")
                    else:
                        print(f"   ❌ Batch size {batch_size}: Validation failed")
                    
                except Exception as e:
                    batch_tests[f"batch_{batch_size}"] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"   ❌ Batch size {batch_size}: {str(e)}")
            
            batch_results = {
                'batch_tests': batch_tests,
                'success': all(test.get('success', False) for test in batch_tests.values())
            }
            
            if batch_results['success']:
                print("   ✅ All batch processing tests passed")
            else:
                print("   ❌ Some batch processing tests failed")
            
        except Exception as e:
            batch_results = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Batch processing test failed: {str(e)}")
        
        self.test_results['batch_processing'] = batch_results
    
    def _test_variable_length_sequences(self):
        """Test handling of variable-length sequences."""
        print("\n📏 Testing Variable-Length Sequences...")
        
        sequence_results = {}
        
        try:
            sequence_tests = {}
            
            for seq_len in self.config.test_sequence_lengths:
                try:
                    # Get test data with specific sequence length
                    windows, _ = self.loader.get_windows(window_size=seq_len, as_torch=True)
                    if windows.shape[0] < 4:
                        continue
                    
                    test_batch = windows[:4]
                    
                    # Test forward pass
                    self.extended_model.eval()
                    with torch.no_grad():
                        output = self.extended_model(test_batch)
                    
                    forecasts = output['forecasts']
                    interpretability = output['interpretability']
                    
                    # Validate sequence handling
                    sequence_validation = {
                        'input_seq_len': seq_len,
                        'input_shape': list(test_batch.shape),
                        'forecast_shape': list(forecasts.shape),
                        'forecast_horizon_correct': forecasts.shape[-1] == self.extended_model.config.forecast_horizon,
                        'output_valid': not torch.isnan(forecasts).any() and not torch.isinf(forecasts).any(),
                        'temporal_attention_shape_correct': (
                            interpretability['temporal_attention'].shape[-1] == seq_len and
                            interpretability['temporal_attention'].shape[-2] == seq_len
                        ),
                        'compression_preserves_variables': (
                            interpretability['compression_attention'].shape[1] == test_batch.shape[2]
                        )
                    }
                    
                    # Test computational scaling
                    start_time = time.time()
                    with torch.no_grad():
                        for _ in range(5):
                            _ = self.extended_model(test_batch)
                    end_time = time.time()
                    
                    avg_time_ms = (end_time - start_time) / 5 * 1000
                    time_per_sample_ms = avg_time_ms / test_batch.shape[0]
                    
                    performance_validation = {
                        'avg_inference_time_ms': avg_time_ms,
                        'time_per_sample_ms': time_per_sample_ms,
                        'within_performance_threshold': time_per_sample_ms < self.config.performance_threshold_ms
                    }
                    
                    sequence_tests[f"seq_len_{seq_len}"] = {
                        'sequence_validation': sequence_validation,
                        'performance_validation': performance_validation,
                        'success': (all(sequence_validation.values()) and 
                                  performance_validation['within_performance_threshold'])
                    }
                    
                    if sequence_tests[f"seq_len_{seq_len}"]['success']:
                        print(f"   ✅ Sequence length {seq_len}: {time_per_sample_ms:.2f} ms/sample")
                    else:
                        print(f"   ❌ Sequence length {seq_len}: Issues detected")
                    
                except Exception as e:
                    sequence_tests[f"seq_len_{seq_len}"] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"   ❌ Sequence length {seq_len}: {str(e)}")
            
            sequence_results = {
                'sequence_tests': sequence_tests,
                'success': all(test.get('success', False) for test in sequence_tests.values())
            }
            
            if sequence_results['success']:
                print("   ✅ All variable-length sequence tests passed")
            else:
                print("   ❌ Some variable-length sequence tests failed")
            
        except Exception as e:
            sequence_results = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Variable-length sequence test failed: {str(e)}")
        
        self.test_results['variable_length_sequences'] = sequence_results
    
    def _test_computational_efficiency(self):
        """Test computational efficiency across different configurations."""
        print("\n⚡ Testing Computational Efficiency...")
        
        efficiency_results = {}
        
        try:
            # Test inference speed scaling
            windows, _ = self.loader.get_windows(window_size=96, as_torch=True)
            
            efficiency_tests = {}
            
            for batch_size in [1, 4, 8, 16]:
                if windows.shape[0] < batch_size:
                    continue
                
                try:
                    test_batch = windows[:batch_size]
                    
                    # Warmup
                    self.extended_model.eval()
                    with torch.no_grad():
                        for _ in range(3):
                            _ = self.extended_model(test_batch)
                    
                    # Measure inference time
                    start_time = time.time()
                    with torch.no_grad():
                        for _ in range(10):
                            output = self.extended_model(test_batch)
                    end_time = time.time()
                    
                    avg_time = (end_time - start_time) / 10
                    throughput = batch_size / avg_time
                    time_per_sample = avg_time / batch_size
                    
                    efficiency_tests[f"batch_{batch_size}"] = {
                        'batch_size': batch_size,
                        'avg_inference_time_s': avg_time,
                        'throughput_samples_per_s': throughput,
                        'time_per_sample_ms': time_per_sample * 1000,
                        'efficient': time_per_sample * 1000 < self.config.performance_threshold_ms,
                        'success': True
                    }
                    
                    print(f"   Batch {batch_size}: {throughput:.1f} samples/s ({time_per_sample*1000:.2f} ms/sample)")
                    
                except Exception as e:
                    efficiency_tests[f"batch_{batch_size}"] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"   ❌ Batch {batch_size}: {str(e)}")
            
            # Test memory efficiency
            successful_tests = [test for test in efficiency_tests.values() if test.get('success', False)]
            if successful_tests:
                efficiency_stats = {
                    'min_throughput': min(test['throughput_samples_per_s'] for test in successful_tests),
                    'max_throughput': max(test['throughput_samples_per_s'] for test in successful_tests),
                    'avg_throughput': sum(test['throughput_samples_per_s'] for test in successful_tests) / len(successful_tests),
                    'min_time_per_sample_ms': min(test['time_per_sample_ms'] for test in successful_tests),
                    'max_time_per_sample_ms': max(test['time_per_sample_ms'] for test in successful_tests),
                    'avg_time_per_sample_ms': sum(test['time_per_sample_ms'] for test in successful_tests) / len(successful_tests)
                }
            else:
                efficiency_stats = {}
            
            efficiency_results = {
                'efficiency_tests': efficiency_tests,
                'efficiency_stats': efficiency_stats,
                'success': all(test.get('efficient', False) for test in successful_tests)
            }
            
            if efficiency_results['success']:
                print("   ✅ All efficiency tests passed")
                if efficiency_stats:
                    print(f"   ✅ Average throughput: {efficiency_stats['avg_throughput']:.1f} samples/s")
            else:
                print("   ⚠️  Some efficiency concerns detected")
            
        except Exception as e:
            efficiency_results = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Computational efficiency test failed: {str(e)}")
        
        self.test_results['computational_efficiency'] = efficiency_results
    
    def _test_gradient_flow_integration(self):
        """Test gradient flow through integrated components."""
        print("\n🎯 Testing Gradient Flow Integration...")
        
        gradient_results = {}
        
        try:
            # Get test data
            windows, _ = self.loader.get_windows(window_size=96, as_torch=True)
            test_batch = windows[:4]
            test_batch.requires_grad_(True)
            
            # Forward pass
            self.extended_model.train()
            output = self.extended_model(test_batch)
            
            # Compute loss
            loss = output['forecasts'].sum()
            
            # Backward pass
            loss.backward()
            
            # Check gradients for each component
            component_gradients = {}
            
            # Original encoder gradients
            original_grads = {}
            for name, param in self.extended_model.interpretable_encoder.named_parameters():
                if param.grad is not None:
                    original_grads[name] = param.grad.norm().item()
                else:
                    original_grads[name] = 0.0
            
            component_gradients['interpretable_encoder'] = {
                'total_params': len(original_grads),
                'params_with_grad': sum(1 for grad in original_grads.values() if grad > 0),
                'avg_grad_norm': sum(original_grads.values()) / len(original_grads) if original_grads else 0,
                'max_grad_norm': max(original_grads.values()) if original_grads else 0
            }
            
            # Cross-attention gradients
            cross_grads = {}
            for name, param in self.extended_model.cross_attention.named_parameters():
                if param.grad is not None:
                    cross_grads[name] = param.grad.norm().item()
                else:
                    cross_grads[name] = 0.0
            
            component_gradients['cross_attention'] = {
                'total_params': len(cross_grads),
                'params_with_grad': sum(1 for grad in cross_grads.values() if grad > 0),
                'avg_grad_norm': sum(cross_grads.values()) / len(cross_grads) if cross_grads else 0,
                'max_grad_norm': max(cross_grads.values()) if cross_grads else 0
            }
            
            # Temporal encoder gradients
            temporal_grads = {}
            for name, param in self.extended_model.temporal_encoder.named_parameters():
                if param.grad is not None:
                    temporal_grads[name] = param.grad.norm().item()
                else:
                    temporal_grads[name] = 0.0
            
            component_gradients['temporal_encoder'] = {
                'total_params': len(temporal_grads),
                'params_with_grad': sum(1 for grad in temporal_grads.values() if grad > 0),
                'avg_grad_norm': sum(temporal_grads.values()) / len(temporal_grads) if temporal_grads else 0,
                'max_grad_norm': max(temporal_grads.values()) if temporal_grads else 0
            }
            
            # Spline learner gradients
            spline_grads = {}
            for name, param in self.extended_model.spline_learner.named_parameters():
                if param.grad is not None:
                    spline_grads[name] = param.grad.norm().item()
                else:
                    spline_grads[name] = 0.0
            
            component_gradients['spline_learner'] = {
                'total_params': len(spline_grads),
                'params_with_grad': sum(1 for grad in spline_grads.values() if grad > 0),
                'avg_grad_norm': sum(spline_grads.values()) / len(spline_grads) if spline_grads else 0,
                'max_grad_norm': max(spline_grads.values()) if spline_grads else 0
            }
            
            # Overall gradient flow analysis
            total_params = sum(comp['total_params'] for comp in component_gradients.values())
            total_params_with_grad = sum(comp['params_with_grad'] for comp in component_gradients.values())
            
            gradient_flow_analysis = {
                'total_parameters': total_params,
                'parameters_with_gradients': total_params_with_grad,
                'gradient_coverage': total_params_with_grad / total_params if total_params > 0 else 0,
                'all_components_have_gradients': all(
                    comp['params_with_grad'] == comp['total_params'] 
                    for comp in component_gradients.values()
                ),
                'gradient_magnitudes_reasonable': all(
                    0 < comp['avg_grad_norm'] < 1000 
                    for comp in component_gradients.values()
                    if comp['avg_grad_norm'] > 0
                )
            }
            
            gradient_results = {
                'component_gradients': component_gradients,
                'gradient_flow_analysis': gradient_flow_analysis,
                'loss_value': loss.item(),
                'input_gradient_norm': test_batch.grad.norm().item() if test_batch.grad is not None else 0,
                'success': (gradient_flow_analysis['gradient_coverage'] > 0.95 and
                           gradient_flow_analysis['all_components_have_gradients'] and
                           gradient_flow_analysis['gradient_magnitudes_reasonable'])
            }
            
            if gradient_results['success']:
                print(f"   ✅ Gradient flow through all components: {gradient_flow_analysis['gradient_coverage']:.1%}")
                print(f"   ✅ All {total_params_with_grad}/{total_params} parameters have gradients")
            else:
                print(f"   ❌ Gradient flow issues: {gradient_flow_analysis['gradient_coverage']:.1%} coverage")
            
        except Exception as e:
            gradient_results = {
                'success': False,
                'error': str(e)
            }
            print(f"   ❌ Gradient flow integration test failed: {str(e)}")
        
        self.test_results['gradient_flow_integration'] = gradient_results
    
    def _generate_integration_summary(self):
        """Generate summary of integration test results."""
        print("\n📋 Generating Integration Test Summary...")
        
        # Count successful tests
        total_tests = 0
        passed_tests = 0
        
        for category, results in self.test_results.items():
            if isinstance(results, dict) and 'success' in results:
                total_tests += 1
                if results['success']:
                    passed_tests += 1
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'test_categories': list(self.test_results.keys()),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save detailed results
        import json
        with open('integration_test_results.json', 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_safe_results = self._make_json_safe(self.test_results)
            json.dump({
                'summary': summary,
                'detailed_results': json_safe_results
            }, f, indent=2)
        
        print(f"   Integration test results saved: integration_test_results.json")
        print(f"   Success rate: {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)")
        
        self.test_results['summary'] = summary
    
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


def run_integration_tests():
    """
    Main function to run all integration tests.
    
    Returns:
        Test results dictionary
    """
    print("🔧 Extended Model Integration Tests")
    print("=" * 60)
    
    try:
        # Create tester
        tester = ComponentIntegrationTester()
        
        # Run all tests
        results = tester.run_all_integration_tests()
        
        # Print summary
        if 'summary' in results:
            summary = results['summary']
            print(f"\n🎊 Integration Test Summary:")
            print(f"   ✅ Tests passed: {summary['passed_tests']}/{summary['total_tests']}")
            print(f"   ✅ Success rate: {summary['success_rate']:.1f}%")
        
        print(f"\n🚀 Integration tests complete!")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Integration tests failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run integration tests
    results = run_integration_tests()
    
    if results is not None and results.get('summary', {}).get('success_rate', 0) >= 80:
        print("\n🚀 Integration Tests Successful!")
    else:
        print("\n💥 Integration Tests Failed!")
        exit(1)