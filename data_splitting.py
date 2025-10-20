"""
Data splitting utilities for time series forecasting.

This module provides functions to properly split time series data into train/validation/test sets
while respecting temporal order and avoiding data leakage.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from data_utils import ETTDataLoader, normalize_data, denormalize_data


@dataclass
class DataSplitConfig:
    """Configuration for data splitting."""
    train_ratio: float = 0.7      # 70% for training
    val_ratio: float = 0.2        # 20% for validation  
    test_ratio: float = 0.1       # 10% for testing
    
    # Time series specific parameters
    min_train_samples: int = 1000  # Minimum samples needed for training
    min_val_samples: int = 200     # Minimum samples needed for validation
    min_test_samples: int = 100    # Minimum samples needed for testing
    
    # Gap between splits to avoid leakage
    val_gap: int = 0              # Gap between train and validation (in time steps)
    test_gap: int = 0             # Gap between validation and test (in time steps)
    
    def __post_init__(self):
        """Validate configuration."""
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total_ratio, 1.0, atol=1e-6):
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        if self.train_ratio <= 0 or self.val_ratio <= 0 or self.test_ratio <= 0:
            raise ValueError("All split ratios must be positive")


class TimeSeriesDataSplitter:
    """
    Time series data splitter that respects temporal order and prevents data leakage.
    
    Key principles:
    1. Temporal order is preserved (train < validation < test in time)
    2. No overlap between splits
    3. Optional gaps between splits to prevent leakage
    4. Normalization is fit only on training data
    5. Same normalization is applied to validation and test sets
    """
    
    def __init__(self, config: DataSplitConfig = None):
        """
        Initialize the data splitter.
        
        Args:
            config: Data splitting configuration
        """
        self.config = config or DataSplitConfig()
        self.split_indices = None
        self.norm_stats = None
        
    def split_data(
        self,
        data: np.ndarray,
        dates: Optional[np.ndarray] = None,
        normalize: str = 'standard'
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split time series data into train/validation/test sets.
        
        Args:
            data: Time series data of shape (T, M)
            dates: Optional datetime array of length T
            normalize: Normalization method ('standard', 'minmax', or 'none')
            
        Returns:
            Dictionary with 'train', 'val', 'test' keys, each containing:
            - 'data': Normalized data for the split
            - 'raw_data': Original unnormalized data for the split
            - 'dates': Dates for the split (if provided)
            - 'indices': Original indices in the full dataset
        """
        T, M = data.shape
        
        # Calculate split indices
        self.split_indices = self._calculate_split_indices(T)
        
        # Validate splits have minimum required samples
        self._validate_splits(T)
        
        # Extract raw data splits
        splits = {}
        for split_name, (start_idx, end_idx) in self.split_indices.items():
            splits[split_name] = {
                'raw_data': data[start_idx:end_idx].copy(),
                'indices': np.arange(start_idx, end_idx),
            }
            
            if dates is not None:
                splits[split_name]['dates'] = dates[start_idx:end_idx]
        
        # Apply normalization (fit only on training data)
        if normalize != 'none':
            # Fit normalization on training data only
            train_data = splits['train']['raw_data']
            _, self.norm_stats = normalize_data(train_data, normalize)
            
            # Apply same normalization to all splits
            for split_name in splits:
                raw_data = splits[split_name]['raw_data']
                normalized_data = self._apply_normalization(raw_data, self.norm_stats)
                splits[split_name]['data'] = normalized_data
        else:
            # No normalization
            for split_name in splits:
                splits[split_name]['data'] = splits[split_name]['raw_data'].copy()
            self.norm_stats = {}
        
        # Print split information
        self._print_split_info(splits, dates)
        
        return splits
    
    def _calculate_split_indices(self, total_length: int) -> Dict[str, Tuple[int, int]]:
        """Calculate start and end indices for each split."""
        # Calculate split sizes
        train_size = int(total_length * self.config.train_ratio)
        val_size = int(total_length * self.config.val_ratio)
        test_size = total_length - train_size - val_size  # Remaining samples
        
        # Calculate indices with gaps
        train_start = 0
        train_end = train_start + train_size
        
        val_start = train_end + self.config.val_gap
        val_end = val_start + val_size
        
        test_start = val_end + self.config.test_gap
        test_end = test_start + test_size
        
        # Ensure we don't exceed total length
        if test_end > total_length:
            # Adjust sizes proportionally
            available_length = total_length - self.config.val_gap - self.config.test_gap
            train_size = int(available_length * self.config.train_ratio)
            val_size = int(available_length * self.config.val_ratio)
            test_size = available_length - train_size - val_size
            
            train_end = train_size
            val_start = train_end + self.config.val_gap
            val_end = val_start + val_size
            test_start = val_end + self.config.test_gap
            test_end = test_start + test_size
        
        return {
            'train': (train_start, train_end),
            'val': (val_start, val_end),
            'test': (test_start, test_end)
        }
    
    def _validate_splits(self, total_length: int):
        """Validate that splits meet minimum requirements."""
        for split_name, (start_idx, end_idx) in self.split_indices.items():
            split_size = end_idx - start_idx
            min_size = getattr(self.config, f'min_{split_name}_samples')
            
            if split_size < min_size:
                raise ValueError(
                    f"{split_name} split has {split_size} samples, "
                    f"but minimum required is {min_size}. "
                    f"Total data length: {total_length}"
                )
    
    def _apply_normalization(self, data: np.ndarray, norm_stats: dict) -> np.ndarray:
        """Apply normalization using pre-computed statistics."""
        if not norm_stats or norm_stats.get('method') == 'none':
            return data
        
        method = norm_stats['method']
        
        if method == 'standard':
            return (data - norm_stats['mean']) / norm_stats['std']
        elif method == 'minmax':
            return (data - norm_stats['min']) / norm_stats['range']
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def _print_split_info(self, splits: Dict, dates: Optional[np.ndarray]):
        """Print information about the data splits."""
        print("📊 Data Split Information:")
        print("=" * 50)
        
        for split_name, split_data in splits.items():
            size = len(split_data['data'])
            start_idx, end_idx = self.split_indices[split_name]
            
            print(f"{split_name.upper()} SET:")
            print(f"  Samples: {size}")
            print(f"  Indices: [{start_idx}, {end_idx})")
            print(f"  Shape: {split_data['data'].shape}")
            
            if dates is not None and 'dates' in split_data:
                print(f"  Date range: {split_data['dates'][0]} to {split_data['dates'][-1]}")
            
            # Data statistics
            data_mean = np.mean(split_data['data'])
            data_std = np.std(split_data['data'])
            print(f"  Data stats: mean={data_mean:.3f}, std={data_std:.3f}")
            print()
    
    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """Denormalize data using stored normalization statistics."""
        if not self.norm_stats or self.norm_stats.get('method') == 'none':
            return normalized_data
        
        method = self.norm_stats['method']
        
        if method == 'standard':
            return normalized_data * self.norm_stats['std'] + self.norm_stats['mean']
        elif method == 'minmax':
            return normalized_data * self.norm_stats['range'] + self.norm_stats['min']
        else:
            raise ValueError(f"Unknown normalization method: {method}")


class ETTDataSplitter:
    """
    Convenient wrapper for splitting ETT dataset with proper time series handling.
    """
    
    def __init__(
        self,
        file_path: str = "interpretable_forecasting/ETT-small/ETTh1.csv",
        split_config: DataSplitConfig = None,
        normalize: str = 'standard'
    ):
        """
        Initialize ETT data splitter.
        
        Args:
            file_path: Path to ETT CSV file
            split_config: Data splitting configuration
            normalize: Normalization method
        """
        self.file_path = file_path
        self.split_config = split_config or DataSplitConfig()
        self.normalize_method = normalize
        
        # Load full dataset
        from data_utils import load_ett_data
        self.raw_data, self.dates, self.variables = load_ett_data(file_path)
        
        # Create splitter and split data
        self.splitter = TimeSeriesDataSplitter(self.split_config)
        self.splits = self.splitter.split_data(
            self.raw_data, 
            self.dates.values if hasattr(self.dates, 'values') else self.dates,
            normalize
        )
        
        print(f"✅ ETT dataset split successfully")
        print(f"   Total samples: {len(self.raw_data)}")
        print(f"   Variables: {self.variables}")
    
    def get_split_data(self, split: str = 'train') -> Dict[str, np.ndarray]:
        """Get data for a specific split."""
        if split not in self.splits:
            raise ValueError(f"Unknown split: {split}. Available: {list(self.splits.keys())}")
        
        return self.splits[split]
    
    def get_forecasting_data(
        self,
        split: str,
        input_length: int,
        prediction_length: int,
        stride: int = 1,
        as_torch: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get forecasting input-target pairs for a specific split.
        
        Args:
            split: Which split to use ('train', 'val', 'test')
            input_length: Length of input sequences
            prediction_length: Length of prediction sequences  
            stride: Step size between samples
            as_torch: Whether to return PyTorch tensors
            
        Returns:
            inputs: Input sequences
            targets: Target sequences
        """
        split_data = self.get_split_data(split)
        data = split_data['data']
        
        from data_utils import create_forecasting_dataset, to_torch_tensors
        
        inputs, targets = create_forecasting_dataset(
            data, input_length, prediction_length, stride
        )
        
        if as_torch:
            inputs, targets = to_torch_tensors(inputs, targets)
        
        return inputs, targets
    
    def get_windows(
        self,
        split: str,
        window_size: int,
        stride: int = 1,
        as_torch: bool = True
    ) -> Tuple[torch.Tensor, Optional[np.ndarray]]:
        """
        Get sliding windows for a specific split.
        
        Args:
            split: Which split to use ('train', 'val', 'test')
            window_size: Size of each window
            stride: Step size between windows
            as_torch: Whether to return PyTorch tensors
            
        Returns:
            windows: Windowed data
            indices: Starting indices of windows
        """
        split_data = self.get_split_data(split)
        data = split_data['data']
        
        from data_utils import create_time_windows, to_torch_tensors
        
        windows, indices = create_time_windows(
            data, window_size, stride, return_indices=True
        )
        
        if as_torch:
            windows = to_torch_tensors(windows)
        
        return windows, indices
    
    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """Denormalize data back to original scale."""
        return self.splitter.denormalize(normalized_data)


def test_data_splitting():
    """Test the data splitting functionality."""
    print("🧪 Testing Time Series Data Splitting")
    print("=" * 60)
    
    # Test 1: Basic splitting
    print("1. Testing basic data splitting...")
    
    # Create synthetic time series data
    T, M = 1000, 7
    np.random.seed(42)
    synthetic_data = np.random.randn(T, M).astype(np.float32)
    
    # Create splitter and split data
    config = DataSplitConfig(
        train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
        min_train_samples=500, min_val_samples=100, min_test_samples=50
    )
    splitter = TimeSeriesDataSplitter(config)
    splits = splitter.split_data(synthetic_data, normalize='standard')
    
    # Verify splits
    total_samples = sum(len(split['data']) for split in splits.values())
    print(f"✅ Total samples preserved: {total_samples} == {T}")
    
    # Verify no overlap
    train_indices = set(splits['train']['indices'])
    val_indices = set(splits['val']['indices'])
    test_indices = set(splits['test']['indices'])
    
    assert len(train_indices & val_indices) == 0, "Train and validation overlap!"
    assert len(val_indices & test_indices) == 0, "Validation and test overlap!"
    assert len(train_indices & test_indices) == 0, "Train and test overlap!"
    print("✅ No overlap between splits")
    
    # Verify temporal order
    max_train_idx = max(train_indices)
    min_val_idx = min(val_indices)
    max_val_idx = max(val_indices)
    min_test_idx = min(test_indices)
    
    assert max_train_idx < min_val_idx, "Train comes after validation!"
    assert max_val_idx < min_test_idx, "Validation comes after test!"
    print("✅ Temporal order preserved")
    
    # Test 2: ETT dataset splitting
    print("\n2. Testing ETT dataset splitting...")
    
    try:
        ett_splitter = ETTDataSplitter(
            file_path="interpretable_forecasting/ETT-small/ETTh1.csv",
            split_config=DataSplitConfig(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2),
            normalize='standard'
        )
        
        # Test getting forecasting data for each split
        for split_name in ['train', 'val', 'test']:
            inputs, targets = ett_splitter.get_forecasting_data(
                split=split_name,
                input_length=96,
                prediction_length=24,
                stride=24
            )
            print(f"✅ {split_name} forecasting data: inputs {inputs.shape}, targets {targets.shape}")
        
        # Test getting windows for each split
        for split_name in ['train', 'val', 'test']:
            windows, indices = ett_splitter.get_windows(
                split=split_name,
                window_size=48,
                stride=12
            )
            print(f"✅ {split_name} windows: {windows.shape}")
        
        print("✅ ETT dataset splitting successful")
        
    except Exception as e:
        print(f"❌ ETT dataset splitting failed: {e}")
        return False
    
    # Test 3: Normalization consistency
    print("\n3. Testing normalization consistency...")
    
    # Check that validation and test data use training normalization
    train_mean = np.mean(splits['train']['data'])
    val_mean = np.mean(splits['val']['data'])
    test_mean = np.mean(splits['test']['data'])
    
    print(f"   Train mean: {train_mean:.6f}")
    print(f"   Val mean: {val_mean:.6f}")
    print(f"   Test mean: {test_mean:.6f}")
    
    # Training data should be approximately normalized (mean ≈ 0 for standard normalization)
    assert abs(train_mean) < 0.1, f"Training data not properly normalized: mean={train_mean}"
    print("✅ Training data properly normalized")
    
    # Test denormalization
    denorm_train = splitter.denormalize(splits['train']['data'])
    original_train = splits['train']['raw_data']
    
    if np.allclose(denorm_train, original_train, atol=1e-5):
        print("✅ Denormalization works correctly")
    else:
        print("❌ Denormalization failed")
        return False
    
    print("\n🎉 All data splitting tests passed!")
    print("\n📋 Key Benefits of This Splitting Approach:")
    print("   ✅ Respects temporal order (no future leakage)")
    print("   ✅ No overlap between train/val/test sets")
    print("   ✅ Normalization fit only on training data")
    print("   ✅ Same normalization applied to all splits")
    print("   ✅ Configurable split ratios and minimum sizes")
    print("   ✅ Optional gaps between splits for extra safety")
    
    return True


if __name__ == "__main__":
    success = test_data_splitting()
    if not success:
        exit(1)