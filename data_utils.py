"""
Data loading utilities for ETT dataset and time series windowing.

This module provides functions to load and preprocess the ETT (Electricity Transformer Temperature) 
dataset for time series forecasting tasks.
"""

import pandas as pd
import numpy as np
import torch
from typing import Tuple, Optional, List
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
    Convenient data loader class for ETT dataset.
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
    
    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale."""
        return denormalize_data(normalized_data, self.norm_stats)
    
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
    
    print(f"✅ ETTDataLoader test passed")
    print(f"   Windows shape: {windows_torch.shape}")
    print(f"   Inputs shape: {inputs_torch.shape}")
    print(f"   Targets shape: {targets_torch.shape}")
    
    print("\n🎉 All data loading utility tests passed!")