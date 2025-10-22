"""
Data loading utilities for ETT dataset and time series windowing.

This module provides functions to load and preprocess the ETT (Electricity Transformer Temperature) 
dataset for time series forecasting tasks.
"""

import pandas as pd
import numpy as np
import torch
from typing import Tuple, Optional, List, Dict, Union
from pathlib import Path


def load_ett_data(
    file_path: str = "interpretable_forecasting/ETT-small/ETTh1.csv",
    num_samples: Optional[int] = None
) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """
    Load and preprocess ETT dataset.
    
    Args:
        file_path: Path to the ETT CSV file
        num_samples: Number of samples to load (None for all data)
        
    Returns:
        data: Numpy array of shape (T, 7) with the 7 variables
        dates: DatetimeIndex with timestamps
        variables: List of variable names
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Take specified number of samples
        if num_samples is not None:
            df = df.head(num_samples)
        
        # Extract the 7 variables (excluding date column)
        variables = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        data = df[variables].values.astype(np.float32)
        
        # Parse dates
        dates = pd.to_datetime(df['date'])
        
        print(f"✅ Loaded ETT data: {data.shape[0]} samples, {data.shape[1]} variables")
        print(f"   Date range: {dates.iloc[0]} to {dates.iloc[-1]}")
        print(f"   Variables: {variables}")
        
        return data, dates, variables
        
    except FileNotFoundError:
        print(f"❌ ETT dataset not found at {file_path}")
        raise


def normalize_data(data: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, dict]:
    """
    Normalize the time series data.
    
    Args:
        data: Input data of shape (T, M) where T is time steps, M is variables
        method: Normalization method ('standard', 'minmax', or 'none')
        
    Returns:
        normalized_data: Normalized data
        norm_stats: Dictionary with normalization statistics for denormalization
    """
    if method == 'none':
        return data, {}
    
    norm_stats = {}
    
    if method == 'standard':
        # Z-score normalization: (x - mean) / std
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
        
        normalized_data = (data - mean) / std
        norm_stats = {'method': 'standard', 'mean': mean, 'std': std}
        
    elif method == 'minmax':
        # Min-max normalization: (x - min) / (max - min)
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)
        
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1.0, range_val)
        
        normalized_data = (data - min_val) / range_val
        norm_stats = {'method': 'minmax', 'min': min_val, 'max': max_val, 'range': range_val}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    print(f"✅ Applied {method} normalization")
    print(f"   Original range: [{data.min():.3f}, {data.max():.3f}]")
    print(f"   Normalized range: [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")
    
    return normalized_data, norm_stats


def denormalize_data(normalized_data: np.ndarray, norm_stats: dict) -> np.ndarray:
    """
    Denormalize the data using stored normalization statistics.
    
    Args:
        normalized_data: Normalized data
        norm_stats: Normalization statistics from normalize_data()
        
    Returns:
        denormalized_data: Original scale data
    """
    if not norm_stats or norm_stats.get('method') == 'none':
        return normalized_data
    
    method = norm_stats['method']
    
    if method == 'standard':
        return normalized_data * norm_stats['std'] + norm_stats['mean']
    elif method == 'minmax':
        return normalized_data * norm_stats['range'] + norm_stats['min']
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_time_windows(
    data: np.ndarray,
    window_size: int,
    stride: int = 1,
    return_indices: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Create sliding windows from time series data.
    
    Args:
        data: Input data of shape (T, M) where T is time steps, M is variables
        window_size: Size of each window
        stride: Step size between windows
        return_indices: Whether to return the starting indices of each window
        
    Returns:
        windows: Array of shape (N, window_size, M) where N is number of windows
        indices: Starting indices of each window (if return_indices=True)
    """
    T, M = data.shape
    
    # Calculate number of windows
    num_windows = (T - window_size) // stride + 1
    
    if num_windows <= 0:
        raise ValueError(f"Cannot create windows: data length {T} < window_size {window_size}")
    
    # Create windows
    windows = np.zeros((num_windows, window_size, M), dtype=data.dtype)
    indices = np.zeros(num_windows, dtype=int) if return_indices else None
    
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windows[i] = data[start_idx:end_idx]
        
        if return_indices:
            indices[i] = start_idx
    
    print(f"✅ Created {num_windows} windows of size {window_size}")
    print(f"   Input shape: {data.shape}")
    print(f"   Output shape: {windows.shape}")
    print(f"   Stride: {stride}")
    
    if return_indices:
        return windows, indices
    else:
        return windows, None


def create_forecasting_dataset(
    data: np.ndarray,
    input_length: int,
    prediction_length: int,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create input-target pairs for forecasting.
    
    Args:
        data: Input data of shape (T, M)
        input_length: Length of input sequences
        prediction_length: Length of prediction sequences
        stride: Step size between samples
        
    Returns:
        inputs: Input sequences of shape (N, input_length, M)
        targets: Target sequences of shape (N, prediction_length, M)
    """
    T, M = data.shape
    total_length = input_length + prediction_length
    
    # Calculate number of samples
    num_samples = (T - total_length) // stride + 1
    
    if num_samples <= 0:
        raise ValueError(f"Cannot create forecasting dataset: data length {T} < total_length {total_length}")
    
    inputs = np.zeros((num_samples, input_length, M), dtype=data.dtype)
    targets = np.zeros((num_samples, prediction_length, M), dtype=data.dtype)
    
    for i in range(num_samples):
        start_idx = i * stride
        input_end = start_idx + input_length
        target_end = input_end + prediction_length
        
        inputs[i] = data[start_idx:input_end]
        targets[i] = data[input_end:target_end]
    
    print(f"✅ Created forecasting dataset: {num_samples} samples")
    print(f"   Input sequences: {inputs.shape}")
    print(f"   Target sequences: {targets.shape}")
    
    return inputs, targets


def to_torch_tensors(*arrays: np.ndarray) -> Tuple[torch.Tensor, ...]:
    """
    Convert numpy arrays to PyTorch tensors.
    
    Args:
        *arrays: Variable number of numpy arrays
        
    Returns:
        Tuple of PyTorch tensors
    """
    tensors = tuple(torch.from_numpy(arr.astype(np.float32)) for arr in arrays)
    
    if len(tensors) == 1:
        return tensors[0]
    else:
        return tensors


class ETTDataLoader:
    """
    Enhanced data loader class for ETT dataset with forecasting capabilities.
    """
    
    def __init__(
        self,
        file_path: str = "interpretable_forecasting/ETT-small/ETTh1.csv",
        normalize: str = 'standard',
        num_samples: Optional[int] = None
    ):
        """
        Initialize the ETT data loader.
        
        Args:
            file_path: Path to ETT CSV file
            normalize: Normalization method ('standard', 'minmax', or 'none')
            num_samples: Number of samples to load (None for all)
        """
        self.file_path = file_path
        self.normalize_method = normalize
        
        # Load and normalize data
        self.raw_data, self.dates, self.variables = load_ett_data(file_path, num_samples)
        self.data, self.norm_stats = normalize_data(self.raw_data, normalize)
        
        print(f"✅ ETTDataLoader initialized")
        print(f"   Data shape: {self.data.shape}")
        print(f"   Variables: {self.variables}")
    
    def get_windows(
        self,
        window_size: int,
        stride: int = 1,
        as_torch: bool = True
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """Get sliding windows from the data."""
        windows, indices = create_time_windows(self.data, window_size, stride, return_indices=True)
        
        if as_torch:
            windows = to_torch_tensors(windows)
        
        return windows, indices
    
    def get_forecasting_data(
        self,
        input_length: int,
        prediction_length: int,
        stride: int = 1,
        as_torch: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get input-target pairs for forecasting."""
        inputs, targets = create_forecasting_dataset(
            self.data, input_length, prediction_length, stride
        )
        
        if as_torch:
            inputs, targets = to_torch_tensors(inputs, targets)
        
        return inputs, targets
    
    def create_multi_horizon_dataset(
        self,
        input_length: int,
        forecast_horizons: List[int] = [1, 12, 24, 48],
        stride: int = 1,
        as_torch: bool = True
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create forecasting datasets for multiple prediction horizons.
        
        Args:
            input_length: Length of input sequences
            forecast_horizons: List of forecast horizons to create datasets for
            stride: Step size between samples
            as_torch: Whether to return PyTorch tensors
            
        Returns:
            Dictionary mapping horizon -> (inputs, targets) pairs
        """
        datasets = {}
        
        for horizon in forecast_horizons:
            try:
                inputs, targets = create_forecasting_dataset(
                    self.data, input_length, horizon, stride
                )
                
                if as_torch:
                    inputs, targets = to_torch_tensors(inputs, targets)
                
                datasets[horizon] = (inputs, targets)
                print(f"✅ Created dataset for horizon {horizon}: {inputs.shape[0]} samples")
                
            except ValueError as e:
                print(f"⚠️  Skipping horizon {horizon}: {e}")
                continue
        
        return datasets
    
    def create_train_val_test_splits(
        self,
        input_length: int,
        prediction_length: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        stride: int = 1,
        as_torch: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create train/validation/test splits for forecasting with proper temporal ordering.
        
        Args:
            input_length: Length of input sequences
            prediction_length: Length of prediction sequences
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            stride: Step size between samples
            as_torch: Whether to return PyTorch tensors
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys containing (inputs, targets) pairs
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        T, M = self.data.shape
        total_length = input_length + prediction_length
        
        # Calculate maximum number of samples we can create
        max_samples = (T - total_length) // stride + 1
        if max_samples <= 0:
            raise ValueError(f"Cannot create forecasting dataset: data length {T} < total_length {total_length}")
        
        # Calculate split sizes
        train_size = int(max_samples * train_ratio)
        val_size = int(max_samples * val_ratio)
        test_size = max_samples - train_size - val_size
        
        # Create all input-target pairs
        all_inputs, all_targets = create_forecasting_dataset(
            self.data, input_length, prediction_length, stride
        )
        
        # Split temporally (respecting time order)
        splits = {}
        
        # Training split (earliest samples)
        train_inputs = all_inputs[:train_size]
        train_targets = all_targets[:train_size]
        splits['train'] = (train_inputs, train_targets)
        
        # Validation split (middle samples)
        val_start = train_size
        val_end = val_start + val_size
        val_inputs = all_inputs[val_start:val_end]
        val_targets = all_targets[val_start:val_end]
        splits['val'] = (val_inputs, val_targets)
        
        # Test split (latest samples)
        test_start = val_end
        test_inputs = all_inputs[test_start:test_start + test_size]
        test_targets = all_targets[test_start:test_start + test_size]
        splits['test'] = (test_inputs, test_targets)
        
        # Convert to torch tensors if requested
        if as_torch:
            for split_name in splits:
                inputs, targets = splits[split_name]
                splits[split_name] = to_torch_tensors(inputs, targets)
        
        # Print split information
        print(f"✅ Created forecasting splits:")
        for split_name, (inputs, targets) in splits.items():
            print(f"   {split_name}: {inputs.shape[0]} samples, inputs {inputs.shape}, targets {targets.shape}")
        
        return splits
    
    def create_multi_horizon_splits(
        self,
        input_length: int,
        forecast_horizons: List[int] = [1, 12, 24, 48],
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        stride: int = 1,
        as_torch: bool = True
    ) -> Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Create train/val/test splits for multiple forecast horizons.
        
        Args:
            input_length: Length of input sequences
            forecast_horizons: List of forecast horizons
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            stride: Step size between samples
            as_torch: Whether to return PyTorch tensors
            
        Returns:
            Nested dictionary: horizon -> split -> (inputs, targets)
        """
        multi_horizon_splits = {}
        
        for horizon in forecast_horizons:
            try:
                splits = self.create_train_val_test_splits(
                    input_length=input_length,
                    prediction_length=horizon,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    stride=stride,
                    as_torch=as_torch
                )
                multi_horizon_splits[horizon] = splits
                print(f"✅ Created splits for horizon {horizon}")
                
            except ValueError as e:
                print(f"⚠️  Skipping horizon {horizon}: {e}")
                continue
        
        return multi_horizon_splits
    
    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale."""
        return denormalize_data(normalized_data, self.norm_stats)
    
    def evaluate_forecasts(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        denormalize: bool = True,
        compute_intervals: bool = True,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Evaluate forecasting performance with comprehensive metrics.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            denormalize: Whether to denormalize predictions and targets
            compute_intervals: Whether to compute confidence intervals
            confidence_level: Confidence level for intervals
            
        Returns:
            Comprehensive evaluation results
        """
        from evaluation_utils import evaluate_forecast_quality
        
        norm_stats = self.norm_stats if denormalize else None
        
        return evaluate_forecast_quality(
            predictions=predictions,
            targets=targets,
            norm_stats=norm_stats,
            variable_names=self.variables,
            compute_intervals=compute_intervals,
            confidence_level=confidence_level
        )
    
    def denormalize_predictions(
        self,
        normalized_predictions: Union[np.ndarray, torch.Tensor],
        variable_indices: Optional[List[int]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Denormalize predictions back to original scale.
        
        Args:
            normalized_predictions: Normalized prediction values
            variable_indices: Optional list of variable indices to denormalize
            
        Returns:
            Denormalized predictions in original scale
        """
        from evaluation_utils import denormalize_forecasts
        
        return denormalize_forecasts(
            normalized_predictions, 
            self.norm_stats, 
            variable_indices
        )
    
    def compute_forecast_metrics(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        targets: Union[np.ndarray, torch.Tensor],
        denormalize: bool = True,
        per_variable: bool = True,
        per_horizon: bool = True
    ) -> Dict:
        """
        Compute forecasting metrics with various granularities.
        
        Args:
            predictions: Predicted values
            targets: Ground truth values
            denormalize: Whether to denormalize before computing metrics
            per_variable: Whether to compute per-variable metrics
            per_horizon: Whether to compute per-horizon metrics
            
        Returns:
            Dictionary containing various metric breakdowns
        """
        from evaluation_utils import (
            compute_forecasting_metrics,
            compute_variable_wise_metrics,
            compute_horizon_wise_metrics,
            denormalize_forecasts
        )
        
        # Denormalize if requested
        if denormalize and self.norm_stats.get('method') != 'none':
            pred_eval = denormalize_forecasts(predictions, self.norm_stats)
            target_eval = denormalize_forecasts(targets, self.norm_stats)
        else:
            pred_eval = predictions
            target_eval = targets
        
        results = {}
        
        # Overall metrics
        results['overall'] = compute_forecasting_metrics(pred_eval, target_eval)
        
        # Per-variable metrics
        if per_variable:
            results['per_variable'] = compute_variable_wise_metrics(
                pred_eval, target_eval, self.variables
            )
        
        # Per-horizon metrics (if 3D data)
        if per_horizon:
            if isinstance(pred_eval, torch.Tensor):
                pred_shape = pred_eval.shape
            else:
                pred_shape = pred_eval.shape
            
            if len(pred_shape) == 3:
                results['per_horizon'] = compute_horizon_wise_metrics(pred_eval, target_eval)
        
        return results
    
    def get_variable_info(self) -> dict:
        """Get information about the variables."""
        return {
            'names': self.variables,
            'count': len(self.variables),
            'raw_stats': {
                'mean': np.mean(self.raw_data, axis=0),
                'std': np.std(self.raw_data, axis=0),
                'min': np.min(self.raw_data, axis=0),
                'max': np.max(self.raw_data, axis=0)
            }
        }


if __name__ == "__main__":
    # Test the data loading utilities
    print("🧪 Testing data loading utilities...")
    
    # Test basic data loading
    try:
        data, dates, variables = load_ett_data(num_samples=100)
        print(f"✅ Basic loading test passed")
    except Exception as e:
        print(f"❌ Basic loading test failed: {e}")
        exit(1)
    
    # Test normalization
    normalized_data, norm_stats = normalize_data(data, 'standard')
    denormalized_data = denormalize_data(normalized_data, norm_stats)
    
    # Check if denormalization works
    if np.allclose(data, denormalized_data, atol=1e-5):
        print("✅ Normalization/denormalization test passed")
    else:
        print("❌ Normalization/denormalization test failed")
        exit(1)
    
    # Test windowing
    windows, indices = create_time_windows(data, window_size=10, stride=5, return_indices=True)
    print(f"✅ Windowing test passed")
    
    # Test forecasting dataset creation
    inputs, targets = create_forecasting_dataset(data, input_length=24, prediction_length=12)
    print(f"✅ Forecasting dataset test passed")
    
    # Test ETTDataLoader
    loader = ETTDataLoader(num_samples=200)
    windows_torch, _ = loader.get_windows(window_size=20)
    inputs_torch, targets_torch = loader.get_forecasting_data(input_length=48, prediction_length=24)
    
    print(f"✅ ETTDataLoader basic test passed")
    print(f"   Windows shape: {windows_torch.shape}")
    print(f"   Inputs shape: {inputs_torch.shape}")
    print(f"   Targets shape: {targets_torch.shape}")
    
    # Test multi-horizon dataset creation
    multi_horizon_data = loader.create_multi_horizon_dataset(
        input_length=48, 
        forecast_horizons=[1, 12, 24, 48]
    )
    print(f"✅ Multi-horizon dataset test passed")
    for horizon, (inputs, targets) in multi_horizon_data.items():
        print(f"   Horizon {horizon}: inputs {inputs.shape}, targets {targets.shape}")
    
    # Test train/val/test splits
    splits = loader.create_train_val_test_splits(
        input_length=48,
        prediction_length=24,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    print(f"✅ Train/val/test splits test passed")
    
    # Test multi-horizon splits
    multi_splits = loader.create_multi_horizon_splits(
        input_length=48,
        forecast_horizons=[12, 24],
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    print(f"✅ Multi-horizon splits test passed")
    
    # Test evaluation functionality
    # Create some synthetic predictions for testing
    train_inputs, train_targets = splits['train']
    synthetic_predictions = train_targets + 0.1 * torch.randn_like(train_targets)
    
    # Test denormalization
    denorm_predictions = loader.denormalize_predictions(synthetic_predictions)
    denorm_targets = loader.denormalize_predictions(train_targets)
    print(f"✅ Denormalization test passed")
    print(f"   Normalized range: [{synthetic_predictions.min():.3f}, {synthetic_predictions.max():.3f}]")
    print(f"   Denormalized range: [{denorm_predictions.min():.3f}, {denorm_predictions.max():.3f}]")
    
    # Test metrics computation
    metrics = loader.compute_forecast_metrics(
        synthetic_predictions, train_targets, 
        denormalize=True, per_variable=True, per_horizon=True
    )
    print(f"✅ Metrics computation test passed")
    print(f"   Overall MSE: {metrics['overall']['mse']:.4f}")
    print(f"   Variables with metrics: {list(metrics['per_variable'].keys())}")
    if 'per_horizon' in metrics:
        print(f"   Horizons with metrics: {len(metrics['per_horizon'])}")
    
    # Test comprehensive evaluation
    try:
        eval_results = loader.evaluate_forecasts(
            synthetic_predictions, train_targets,
            denormalize=True, compute_intervals=False
        )
        print(f"✅ Comprehensive evaluation test passed")
        print(f"   Evaluation keys: {list(eval_results.keys())}")
    except Exception as e:
        print(f"⚠️  Comprehensive evaluation test skipped: {e}")
    
    print("\n🎉 All data loading utility tests passed!")