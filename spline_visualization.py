"""
Simple Spline Visualization for Extended Forecasting Model

This module provides lightweight visualization functions to show spline outputs,
control points, and forecast accuracy against true validation data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def plot_spline_outputs(model_output: Dict[str, torch.Tensor], 
                       input_data: torch.Tensor,
                       true_future: Optional[torch.Tensor] = None,
                       variable_names: Optional[List[str]] = None, 
                       sample_idx: int = 0, 
                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Create simple spline visualization showing forecasts, control points, and accuracy.
    
    Args:
        model_output: Output from InterpretableForecastingModel
        input_data: Historical input data (B, T, M)
        true_future: True future values for accuracy comparison (B, forecast_horizon, M)
        variable_names: List of variable names for labeling
        sample_idx: Which sample in the batch to visualize
        save_path: Path to save the figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    spline_params = model_output['interpretability']['spline_parameters']
    forecasts = model_output['forecasts'][sample_idx]  # (M, forecast_horizon)
    control_points = spline_params['control_points'][sample_idx]  # (M, num_control_points)
    
    # Extract sample data
    sample_input = input_data[sample_idx].cpu().numpy()  # (T, M)
    sample_forecasts = forecasts.cpu().numpy()  # (M, forecast_horizon)
    sample_control_points = control_points.cpu().numpy()  # (M, num_control_points)
    
    if true_future is not None:
        sample_true_future = true_future[sample_idx].cpu().numpy()  # (forecast_horizon, M)
    else:
        sample_true_future = None
    
    # Default variable names if not provided
    if variable_names is None:
        variable_names = [f'Var_{i}' for i in range(sample_forecasts.shape[0])]
    
    # Create figure with subplots (2x2 for first 4 variables)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Time axes
    input_time = np.arange(sample_input.shape[0])
    forecast_time = np.arange(sample_input.shape[0], 
                            sample_input.shape[0] + sample_forecasts.shape[1])
    
    for i in range(min(4, len(variable_names))):
        ax = axes[i]
        var_name = variable_names[i]
        
        # Plot historical data
        ax.plot(input_time, sample_input[:, i], 'b-', linewidth=2, 
               label='Historical Data', alpha=0.8)
        
        # Plot spline forecast
        ax.plot(forecast_time, sample_forecasts[i], 'r-', linewidth=2, 
               marker='o', markersize=4, label='Spline Forecast', alpha=0.8)
        
        # Plot true future data if available
        if sample_true_future is not None:
            ax.plot(forecast_time, sample_true_future[:, i], 'g--', linewidth=2, 
                   marker='s', markersize=4, label='True Future', alpha=0.8)
            
            # Calculate and display accuracy metrics
            mse = np.mean((sample_forecasts[i] - sample_true_future[:, i]) ** 2)
            mae = np.mean(np.abs(sample_forecasts[i] - sample_true_future[:, i]))
            
            # Add accuracy text
            accuracy_text = f'MSE: {mse:.4f}\nMAE: {mae:.4f}'
            ax.text(0.02, 0.98, accuracy_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                   facecolor="lightyellow", alpha=0.8), fontsize=9)
        
        # Plot control points in parameter space
        # Map control points to visualization space
        control_x = np.linspace(forecast_time[0] - 2, forecast_time[-1] + 2, 
                               len(sample_control_points[i]))
        ax.scatter(control_x, sample_control_points[i], color='purple', s=80, 
                  marker='s', label='Control Points', zorder=5, 
                  edgecolors='black', linewidth=1)
        
        # Draw lines connecting control points
        ax.plot(control_x, sample_control_points[i], '--', color='purple', 
               alpha=0.5, linewidth=1)
        
        # Add vertical line to separate history from forecast
        ax.axvline(x=sample_input.shape[0] - 0.5, color='gray', linestyle='--', 
                  alpha=0.7, linewidth=1)
        
        # Add shaded region for forecast period
        ax.axvspan(sample_input.shape[0] - 0.5, forecast_time[-1] + 0.5, 
                  alpha=0.1, color='red', label='Forecast Period')
        
        # Formatting
        ax.set_title(f'{var_name} - Spline Forecast Analysis', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add trend analysis
        if len(sample_input[:, i]) > 1:
            historical_trend = np.polyfit(input_time[-10:], sample_input[-10:, i], 1)[0]
            forecast_trend = np.polyfit(range(len(sample_forecasts[i])), 
                                      sample_forecasts[i], 1)[0]
            
            trend_text = f'Hist Trend: {historical_trend:.4f}\nFcst Trend: {forecast_trend:.4f}'
            ax.text(0.98, 0.02, trend_text, transform=ax.transAxes,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                   fontsize=8)
    
    plt.suptitle(f'Spline-Based Forecasting Results (Sample {sample_idx})', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Spline visualization saved to {save_path}")
    
    return fig


def plot_spline_control_points_analysis(model_output: Dict[str, torch.Tensor],
                                       variable_names: Optional[List[str]] = None,
                                       sample_idx: int = 0,
                                       save_path: Optional[str] = None) -> plt.Figure:
    """
    Create detailed analysis of spline control points and basis functions.
    
    Args:
        model_output: Output from InterpretableForecastingModel
        variable_names: List of variable names for labeling
        sample_idx: Which sample in the batch to visualize
        save_path: Path to save the figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    spline_params = model_output['interpretability']['spline_parameters']
    control_points = spline_params['control_points'][sample_idx].cpu().numpy()  # (M, num_control_points)
    basis_functions = spline_params['basis_functions'].cpu().numpy()  # (forecast_horizon, num_control_points)
    forecasts = model_output['forecasts'][sample_idx].cpu().numpy()  # (M, forecast_horizon)
    
    if variable_names is None:
        variable_names = [f'Var_{i}' for i in range(control_points.shape[0])]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Control points for all variables
    ax1 = axes[0, 0]
    for i in range(min(len(variable_names), control_points.shape[0])):
        ax1.plot(control_points[i], 'o-', label=variable_names[i], alpha=0.8, linewidth=2)
    ax1.set_title('Control Points by Variable', fontweight='bold')
    ax1.set_xlabel('Control Point Index')
    ax1.set_ylabel('Control Point Value')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. B-spline basis functions
    ax2 = axes[0, 1]
    forecast_steps = np.arange(1, basis_functions.shape[0] + 1)
    for i in range(min(6, basis_functions.shape[1])):  # Show first 6 basis functions
        ax2.plot(forecast_steps, basis_functions[:, i], label=f'Basis {i}', alpha=0.8)
    ax2.set_title('B-Spline Basis Functions', fontweight='bold')
    ax2.set_xlabel('Forecast Step')
    ax2.set_ylabel('Basis Function Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Control point statistics
    ax3 = axes[0, 2]
    control_means = np.mean(control_points, axis=1)
    control_stds = np.std(control_points, axis=1)
    
    x_pos = np.arange(len(variable_names[:control_points.shape[0]]))
    ax3.bar(x_pos, control_means, yerr=control_stds, alpha=0.7, capsize=5)
    ax3.set_title('Control Point Statistics', fontweight='bold')
    ax3.set_xlabel('Variable')
    ax3.set_ylabel('Mean Control Point Value')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(variable_names[:control_points.shape[0]], rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # 4. Forecast variance across horizon
    ax4 = axes[1, 0]
    forecast_var = np.var(forecasts, axis=0)  # Variance across variables for each time step
    forecast_steps = np.arange(1, len(forecast_var) + 1)
    ax4.plot(forecast_steps, forecast_var, 'o-', color='purple', linewidth=2)
    ax4.set_title('Forecast Uncertainty by Horizon', fontweight='bold')
    ax4.set_xlabel('Forecast Step')
    ax4.set_ylabel('Variance Across Variables')
    ax4.grid(True, alpha=0.3)
    
    # 5. Control point correlation matrix
    ax5 = axes[1, 1]
    if control_points.shape[0] > 1:
        correlation_matrix = np.corrcoef(control_points)
        im = ax5.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax5.set_title('Control Point Correlation Matrix', fontweight='bold')
        ax5.set_xticks(range(len(variable_names[:control_points.shape[0]])))
        ax5.set_yticks(range(len(variable_names[:control_points.shape[0]])))
        ax5.set_xticklabels(variable_names[:control_points.shape[0]], rotation=45)
        ax5.set_yticklabels(variable_names[:control_points.shape[0]])
        plt.colorbar(im, ax=ax5, shrink=0.8)
    else:
        ax5.text(0.5, 0.5, 'Single Variable\nNo Correlation', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Control Point Correlation Matrix', fontweight='bold')
    
    # 6. Spline smoothness analysis
    ax6 = axes[1, 2]
    # Calculate second derivatives as smoothness measure
    smoothness_scores = []
    for i in range(forecasts.shape[0]):
        if len(forecasts[i]) > 2:
            second_deriv = np.diff(forecasts[i], n=2)
            smoothness = np.mean(np.abs(second_deriv))
            smoothness_scores.append(smoothness)
        else:
            smoothness_scores.append(0)
    
    x_pos = np.arange(len(variable_names[:len(smoothness_scores)]))
    ax6.bar(x_pos, smoothness_scores, alpha=0.7, color='orange')
    ax6.set_title('Spline Smoothness Analysis', fontweight='bold')
    ax6.set_xlabel('Variable')
    ax6.set_ylabel('Smoothness Score (Lower = Smoother)')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(variable_names[:len(smoothness_scores)], rotation=45)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Spline Control Points and Basis Function Analysis (Sample {sample_idx})', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   Spline analysis saved to {save_path}")
    
    return fig


def create_spline_visualizations(model_output: Dict[str, torch.Tensor],
                               input_data: torch.Tensor,
                               true_future: Optional[torch.Tensor] = None,
                               variable_names: Optional[List[str]] = None,
                               sample_idx: int = 0,
                               output_dir: str = "spline_outputs") -> Dict[str, plt.Figure]:
    """
    Create comprehensive spline visualizations.
    
    Args:
        model_output: Output from InterpretableForecastingModel
        input_data: Historical input data
        true_future: True future values for accuracy comparison
        variable_names: List of variable names
        sample_idx: Which sample to visualize
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary of figure names to matplotlib Figure objects
    """
    print(f"🎨 Creating spline visualizations for sample {sample_idx}...")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Main spline forecast visualization
    fig1 = plot_spline_outputs(
        model_output=model_output,
        input_data=input_data,
        true_future=true_future,
        variable_names=variable_names,
        sample_idx=sample_idx,
        save_path=f"{output_dir}/spline_forecasts.png"
    )
    figures['spline_forecasts'] = fig1
    
    # Detailed control points analysis
    fig2 = plot_spline_control_points_analysis(
        model_output=model_output,
        variable_names=variable_names,
        sample_idx=sample_idx,
        save_path=f"{output_dir}/spline_analysis.png"
    )
    figures['spline_analysis'] = fig2
    
    print(f"   ✅ Created {len(figures)} spline visualization figures")
    print(f"   ✅ Saved to {output_dir}/")
    
    return figures


if __name__ == "__main__":
    print("Spline visualization module - use create_spline_visualizations() function")