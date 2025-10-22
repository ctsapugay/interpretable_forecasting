"""
Evaluation utilities for time series forecasting.

This module provides functions for evaluating forecasting performance,
including metrics computation, denormalization, and uncertainty quantification.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Union, List
import warnings


def compute_forecasting_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Dict[str, float]:
    """
    Compute standard forecasting evaluation metrics.
    
    Args:
        predictions: Predicted values of shape (N, T, M) or (N, M)
        targets: Ground truth values of same shape as predictions
        mask: Optional mask for missing values (1 for valid, 0 for missing)
        
    Returns:
        Dictionary containing MSE, MAE, RMSE, and MAPE metrics
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Ensure same shape
    if predictions.shape != targets.shape:
        raise ValueError(f"Predictions shape {predictions.shape} != targets shape {targets.shape}")
    
    # Apply mask if provided
    if mask is not None:
        if mask.shape != predictions.shape:
            raise ValueError(f"Mask shape {mask.shape} != predictions shape {predictions.shape}")
        
        # Only compute metrics on valid (non-masked) values
        valid_mask = mask.astype(bool)
        pred_valid = predictions[valid_mask]
        target_valid = targets[valid_mask]
    else:
        pred_valid = predictions.flatten()
        target_valid = targets.flatten()
    
    # Remove any NaN or infinite values
    finite_mask = np.isfinite(pred_valid) & np.isfinite(target_valid)
    pred_clean = pred_valid[finite_mask]
    target_clean = target_valid[finite_mask]
    
    if len(pred_clean) == 0:
        warnings.warn("No valid predictions/targets found for metric computation")
        return {
            'mse': float('nan'),
            'mae': float('nan'),
            'rmse': float('nan'),
            'mape': float('nan')
        }
    
    # Compute metrics
    errors = pred_clean - target_clean
    
    # Mean Squared Error
    mse = np.mean(errors ** 2)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(errors))
    
    # Root Mean Squared Error
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error (handle division by zero)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        percentage_errors = np.abs(errors) / (np.abs(target_clean) + 1e-8)
        mape = np.mean(percentage_errors) * 100
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape)
    }


def compute_variable_wise_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    variable_names: Optional[List[str]] = None,
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compute forecasting metrics for each variable separately.
    
    Args:
        predictions: Predicted values of shape (N, T, M) or (N, M)
        targets: Ground truth values of same shape as predictions
        variable_names: Optional list of variable names
        mask: Optional mask for missing values
        
    Returns:
        Dictionary mapping variable names to their metrics
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    # Ensure we have the variable dimension
    if predictions.ndim == 2:
        # Shape (N, M) - single time step
        num_variables = predictions.shape[1]
    elif predictions.ndim == 3:
        # Shape (N, T, M) - multiple time steps
        num_variables = predictions.shape[2]
    else:
        raise ValueError(f"Predictions must be 2D or 3D, got shape {predictions.shape}")
    
    # Default variable names if not provided
    if variable_names is None:
        variable_names = [f'var_{i}' for i in range(num_variables)]
    elif len(variable_names) != num_variables:
        raise ValueError(f"Number of variable names {len(variable_names)} != number of variables {num_variables}")
    
    # Compute metrics for each variable
    variable_metrics = {}
    
    for i, var_name in enumerate(variable_names):
        if predictions.ndim == 2:
            pred_var = predictions[:, i]
            target_var = targets[:, i]
            mask_var = mask[:, i] if mask is not None else None
        else:  # 3D
            pred_var = predictions[:, :, i]
            target_var = targets[:, :, i]
            mask_var = mask[:, :, i] if mask is not None else None
        
        metrics = compute_forecasting_metrics(pred_var, target_var, mask_var)
        variable_metrics[var_name] = metrics
    
    return variable_metrics


def compute_horizon_wise_metrics(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    mask: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compute forecasting metrics for each time horizon separately.
    
    Args:
        predictions: Predicted values of shape (N, T, M)
        targets: Ground truth values of same shape as predictions
        mask: Optional mask for missing values
        
    Returns:
        Dictionary mapping time step to metrics
    """
    # Convert to numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    
    if predictions.ndim != 3:
        raise ValueError(f"Predictions must be 3D (N, T, M), got shape {predictions.shape}")
    
    N, T, M = predictions.shape
    horizon_metrics = {}
    
    for t in range(T):
        pred_t = predictions[:, t, :]  # Shape (N, M)
        target_t = targets[:, t, :]    # Shape (N, M)
        mask_t = mask[:, t, :] if mask is not None else None
        
        metrics = compute_forecasting_metrics(pred_t, target_t, mask_t)
        horizon_metrics[t + 1] = metrics  # 1-indexed time steps
    
    return horizon_metrics


def denormalize_forecasts(
    normalized_forecasts: Union[np.ndarray, torch.Tensor],
    norm_stats: Dict,
    variable_indices: Optional[List[int]] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Denormalize forecasting outputs back to original scale.
    
    Args:
        normalized_forecasts: Normalized forecast values
        norm_stats: Normalization statistics from data_utils.normalize_data()
        variable_indices: Optional list of variable indices to denormalize
                         (useful for partial forecasting)
        
    Returns:
        Denormalized forecasts in original scale
    """
    if not norm_stats or norm_stats.get('method') == 'none':
        return normalized_forecasts
    
    # Handle torch tensors
    is_torch = isinstance(normalized_forecasts, torch.Tensor)
    if is_torch:
        device = normalized_forecasts.device
        forecasts = normalized_forecasts.detach().cpu().numpy()
    else:
        forecasts = normalized_forecasts.copy()
    
    method = norm_stats['method']
    
    # Get normalization parameters
    if method == 'standard':
        mean = norm_stats['mean']
        std = norm_stats['std']
    elif method == 'minmax':
        min_val = norm_stats['min']
        range_val = norm_stats['range']
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Handle variable selection
    if variable_indices is not None:
        if method == 'standard':
            mean = mean[:, variable_indices] if mean.ndim > 1 else mean[variable_indices]
            std = std[:, variable_indices] if std.ndim > 1 else std[variable_indices]
        elif method == 'minmax':
            min_val = min_val[:, variable_indices] if min_val.ndim > 1 else min_val[variable_indices]
            range_val = range_val[:, variable_indices] if range_val.ndim > 1 else range_val[variable_indices]

    # Ensure broadcastable shapes with forecasts of shape (N, M, H)
    # We add a trailing axis for horizon (H)
    if method == 'standard':
        import numpy as _np
        mean = _np.asarray(mean)
        std = _np.asarray(std)
        if mean.ndim == 1:
            mean = mean[None, :, None]
        elif mean.ndim == 2:
            mean = mean[:, :, None]
        if std.ndim == 1:
            std = std[None, :, None]
        elif std.ndim == 2:
            std = std[:, :, None]
    elif method == 'minmax':
        import numpy as _np
        min_val = _np.asarray(min_val)
        range_val = _np.asarray(range_val)
        if min_val.ndim == 1:
            min_val = min_val[None, :, None]
        elif min_val.ndim == 2:
            min_val = min_val[:, :, None]
        if range_val.ndim == 1:
            range_val = range_val[None, :, None]
        elif range_val.ndim == 2:
            range_val = range_val[:, :, None]
    
    # Apply denormalization
    if method == 'standard':
        denormalized = forecasts * std + mean
    elif method == 'minmax':
        denormalized = forecasts * range_val + min_val
    
    # Convert back to torch tensor if needed
    if is_torch:
        denormalized = torch.from_numpy(denormalized.astype(np.float32)).to(device)
    
    return denormalized


def compute_confidence_intervals(
    predictions: Union[np.ndarray, torch.Tensor],
    uncertainties: Optional[Union[np.ndarray, torch.Tensor]] = None,
    confidence_level: float = 0.95,
    method: str = 'gaussian'
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Compute confidence intervals for forecasting predictions.
    
    Args:
        predictions: Point predictions of shape (N, T, M) or (N, M)
        uncertainties: Uncertainty estimates (e.g., standard deviations)
        confidence_level: Confidence level (e.g., 0.95 for 95% confidence)
        method: Method for computing intervals ('gaussian', 'quantile')
        
    Returns:
        Tuple of (lower_bounds, upper_bounds) with same shape as predictions
    """
    # Convert to numpy arrays
    is_torch = isinstance(predictions, torch.Tensor)
    if is_torch:
        device = predictions.device
        pred_np = predictions.detach().cpu().numpy()
        if uncertainties is not None:
            unc_np = uncertainties.detach().cpu().numpy()
        else:
            unc_np = None
    else:
        pred_np = predictions.copy()
        unc_np = uncertainties.copy() if uncertainties is not None else None
    
    if method == 'gaussian':
        if uncertainties is None:
            # Use empirical standard deviation as uncertainty estimate
            if pred_np.ndim >= 2:
                unc_np = np.std(pred_np, axis=0, keepdims=True)
                unc_np = np.broadcast_to(unc_np, pred_np.shape)
            else:
                unc_np = np.std(pred_np) * np.ones_like(pred_np)
        
        # Compute z-score for confidence level
        from scipy import stats
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)
        
        # Compute bounds
        lower_bounds = pred_np - z_score * unc_np
        upper_bounds = pred_np + z_score * unc_np
        
    elif method == 'quantile':
        # For quantile method, uncertainties should be prediction samples
        if uncertainties is None:
            raise ValueError("Quantile method requires uncertainty samples")
        
        # Compute quantiles
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        lower_bounds = np.quantile(unc_np, lower_quantile, axis=0)
        upper_bounds = np.quantile(unc_np, upper_quantile, axis=0)
        
    else:
        raise ValueError(f"Unknown confidence interval method: {method}")
    
    # Convert back to torch tensors if needed
    if is_torch:
        lower_bounds = torch.from_numpy(lower_bounds.astype(np.float32)).to(device)
        upper_bounds = torch.from_numpy(upper_bounds.astype(np.float32)).to(device)
    
    return lower_bounds, upper_bounds


def evaluate_forecast_quality(
    predictions: Union[np.ndarray, torch.Tensor],
    targets: Union[np.ndarray, torch.Tensor],
    norm_stats: Optional[Dict] = None,
    variable_names: Optional[List[str]] = None,
    compute_intervals: bool = True,
    confidence_level: float = 0.95
) -> Dict:
    """
    Comprehensive evaluation of forecast quality with multiple metrics and analysis.
    
    Args:
        predictions: Predicted values
        targets: Ground truth values
        norm_stats: Normalization statistics for denormalization
        variable_names: Names of variables
        compute_intervals: Whether to compute confidence intervals
        confidence_level: Confidence level for intervals
        
    Returns:
        Comprehensive evaluation results dictionary
    """
    results = {}
    
    # Denormalize if normalization stats provided
    if norm_stats is not None and norm_stats.get('method') != 'none':
        pred_denorm = denormalize_forecasts(predictions, norm_stats)
        target_denorm = denormalize_forecasts(targets, norm_stats)
        
        # Compute metrics on both normalized and denormalized data
        results['normalized_metrics'] = compute_forecasting_metrics(predictions, targets)
        results['denormalized_metrics'] = compute_forecasting_metrics(pred_denorm, target_denorm)
        
        # Use denormalized for detailed analysis
        pred_eval = pred_denorm
        target_eval = target_denorm
    else:
        results['metrics'] = compute_forecasting_metrics(predictions, targets)
        pred_eval = predictions
        target_eval = targets
    
    # Variable-wise metrics
    if variable_names is not None:
        results['variable_metrics'] = compute_variable_wise_metrics(
            pred_eval, target_eval, variable_names
        )
    
    # Horizon-wise metrics (if 3D data)
    if isinstance(pred_eval, torch.Tensor):
        pred_shape = pred_eval.shape
    else:
        pred_shape = pred_eval.shape
    
    if len(pred_shape) == 3:
        results['horizon_metrics'] = compute_horizon_wise_metrics(pred_eval, target_eval)
    
    # Confidence intervals
    if compute_intervals:
        try:
            lower_bounds, upper_bounds = compute_confidence_intervals(
                pred_eval, confidence_level=confidence_level
            )
            results['confidence_intervals'] = {
                'lower_bounds': lower_bounds,
                'upper_bounds': upper_bounds,
                'confidence_level': confidence_level
            }
        except ImportError:
            warnings.warn("scipy not available, skipping confidence intervals")
    
    return results


if __name__ == "__main__":
    # Test the evaluation utilities
    print("🧪 Testing evaluation utilities...")
    
    # Create synthetic test data
    np.random.seed(42)
    N, T, M = 100, 24, 7
    
    # Generate synthetic predictions and targets
    targets = np.random.randn(N, T, M).astype(np.float32)
    predictions = targets + 0.1 * np.random.randn(N, T, M).astype(np.float32)  # Add some noise
    
    # Test basic metrics
    metrics = compute_forecasting_metrics(predictions, targets)
    print(f"✅ Basic metrics test passed")
    print(f"   MSE: {metrics['mse']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAPE: {metrics['mape']:.4f}")
    
    # Test variable-wise metrics
    variable_names = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    var_metrics = compute_variable_wise_metrics(predictions, targets, variable_names)
    print(f"✅ Variable-wise metrics test passed")
    for var_name, var_metric in var_metrics.items():
        print(f"   {var_name}: MSE={var_metric['mse']:.4f}")
    
    # Test horizon-wise metrics
    horizon_metrics = compute_horizon_wise_metrics(predictions, targets)
    print(f"✅ Horizon-wise metrics test passed")
    print(f"   Horizon 1 MSE: {horizon_metrics[1]['mse']:.4f}")
    print(f"   Horizon {T} MSE: {horizon_metrics[T]['mse']:.4f}")
    
    # Test denormalization
    norm_stats = {
        'method': 'standard',
        'mean': np.zeros((1, M)),
        'std': np.ones((1, M))
    }
    denorm_pred = denormalize_forecasts(predictions, norm_stats)
    print(f"✅ Denormalization test passed")
    print(f"   Original shape: {predictions.shape}")
    print(f"   Denormalized shape: {denorm_pred.shape}")
    
    # Test with torch tensors
    pred_torch = torch.from_numpy(predictions)
    target_torch = torch.from_numpy(targets)
    
    torch_metrics = compute_forecasting_metrics(pred_torch, target_torch)
    print(f"✅ Torch tensor metrics test passed")
    print(f"   Torch MSE: {torch_metrics['mse']:.4f}")
    
    # Test comprehensive evaluation
    try:
        eval_results = evaluate_forecast_quality(
            predictions, targets, norm_stats, variable_names, compute_intervals=False
        )
        print(f"✅ Comprehensive evaluation test passed")
        print(f"   Keys: {list(eval_results.keys())}")
    except Exception as e:
        print(f"⚠️  Comprehensive evaluation test skipped: {e}")
    
    print("\n🎉 All evaluation utility tests passed!")