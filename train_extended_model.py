# train_extended_model.py
"""
Proper training script for Extended Interpretable Forecasting Model.

Uses the existing repo infrastructure:
- data_splitting.py for proper temporal splits
- data_utils.py for correct forecasting targets
- evaluation_utils.py for comprehensive metrics
- Saves interpretability artifacts during training

Usage:
    python train_extended_model.py --epochs 50 --batch-size 32 --lr 0.001
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_utils import ETTDataLoader
from data_splitting import ETTDataSplitter, DataSplitConfig
from extended_model import InterpretableForecastingModel, ExtendedModelConfig
from evaluation_utils import (
    compute_forecasting_metrics,
    compute_variable_wise_metrics,
    denormalize_forecasts
)
from spline_visualization import create_spline_visualizations
import matplotlib.pyplot as plt


class InterpretabilityTracker:
    """Tracks and saves interpretability artifacts during training."""
    
    def __init__(self, save_dir: str):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def save_attention_heatmaps(self, interpretability: dict, epoch: int, 
                               variable_names: list, prefix: str = "val"):
        """Save temporal and cross-attention heatmaps."""
        
        # Cross-attention: (B, heads, M, M) → average to (M, M)
        cross_attn = interpretability['cross_attention']
        cross_avg = cross_attn.mean(dim=(0, 1)).detach().cpu().numpy()
        
        # Temporal attention: (B, M, heads, T, T) → average to (M, T, T)
        temporal_attn = interpretability['temporal_attention']
        temporal_avg = temporal_attn.mean(dim=(0, 2)).detach().cpu().numpy()
        
        # Plot cross-attention
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Cross-variable attention
        im1 = ax1.imshow(cross_avg, cmap='Blues', aspect='auto')
        ax1.set_title(f'Cross-Variable Attention (Epoch {epoch})')
        ax1.set_xlabel('Key Variable')
        ax1.set_ylabel('Query Variable')
        ax1.set_xticks(range(len(variable_names)))
        ax1.set_yticks(range(len(variable_names)))
        ax1.set_xticklabels(variable_names, rotation=45)
        ax1.set_yticklabels(variable_names)
        plt.colorbar(im1, ax=ax1, label='Attention Weight')
        
        # Temporal attention (averaged across variables)
        temporal_global = temporal_avg.mean(axis=0)  # Average across variables
        im2 = ax2.imshow(temporal_global, cmap='Reds', aspect='auto')
        ax2.set_title(f'Temporal Self-Attention (Epoch {epoch})')
        ax2.set_xlabel('Key Time Step')
        ax2.set_ylabel('Query Time Step')
        plt.colorbar(im2, ax=ax2, label='Attention Weight')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{prefix}_attention_epoch_{epoch:03d}.png', dpi=150)
        plt.close()
        
    def save_spline_visualizations(self, model_output: dict, input_data: torch.Tensor, 
                                   true_future: torch.Tensor, variable_names: list, 
                                   epoch: int, prefix: str = "val"):
        """Save spline-based forecast visualizations."""
        try:
            # Create a dedicated subdirectory for spline visualizations
            spline_dir = self.save_dir / f'splines_{prefix}_epoch_{epoch:03d}'
            spline_dir.mkdir(parents=True, exist_ok=True)
            
            # Use the existing spline_visualization module to create comprehensive plots
            figures = create_spline_visualizations(
                model_output=model_output,
                input_data=input_data,
                true_future=true_future,
                variable_names=variable_names,
                sample_idx=0,  # Visualize first sample
                output_dir=str(spline_dir)
            )
            
            # Close all figures to free memory
            for fig in figures.values():
                plt.close(fig)
                
            print(f"      💫 Saved spline visualizations to {spline_dir.name}/")
            
        except Exception as e:
            print(f"      ⚠️  Failed to save spline visualizations: {e}")
    
    def save_training_curves(self, history: dict):
        """Save training and validation loss curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_title('Loss (MSE) over Epochs')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE curves
        axes[0, 1].plot(epochs, history['train_mae'], 'b-', label='Train')
        axes[0, 1].plot(epochs, history['val_mae'], 'r-', label='Validation')
        axes[0, 1].set_title('MAE over Epochs')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate (if tracked)
        if 'lr' in history:
            axes[1, 0].plot(epochs, history['lr'], 'g-')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Validation metrics comparison
        if 'val_rmse' in history:
            axes[1, 1].plot(epochs, history['val_mae'], 'b-', label='MAE')
            axes[1, 1].plot(epochs, history['val_rmse'], 'r-', label='RMSE')
            axes[1, 1].set_title('Validation Metrics')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Error')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        plt.close()


def create_dataloaders(data_splitter, args):
    """Create PyTorch DataLoaders for train/val/test."""
    
    # Get forecasting data for each split
    train_inputs, train_targets = data_splitter.get_forecasting_data(
        split='train',
        input_length=args.input_length,
        prediction_length=args.forecast_horizon,
        stride=args.stride,
        as_torch=True
    )
    
    val_inputs, val_targets = data_splitter.get_forecasting_data(
        split='val',
        input_length=args.input_length,
        prediction_length=args.forecast_horizon,
        stride=args.stride,
        as_torch=True
    )
    
    test_inputs, test_targets = data_splitter.get_forecasting_data(
        split='test',
        input_length=args.input_length,
        prediction_length=args.forecast_horizon,
        stride=args.stride,
        as_torch=True
    )
    
    # Create DataLoaders
    train_dataset = TensorDataset(train_inputs, train_targets)
    val_dataset = TensorDataset(val_inputs, val_targets)
    test_dataset = TensorDataset(test_inputs, test_targets)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    for batch_inputs, batch_targets in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(batch_inputs)
        forecasts = output['forecasts']  # (B, M, H)
        
        # Transpose targets to match: (B, T, M) → (B, M, T)
        targets_transposed = batch_targets.transpose(1, 2)
        
        # Compute loss
        loss = criterion(forecasts, targets_transposed)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (important for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        with torch.no_grad():
            mae = torch.mean(torch.abs(forecasts - targets_transposed)).item()
            total_mae += mae
        num_batches += 1
    
    return total_loss / num_batches, total_mae / num_batches


def validate(model, val_loader, criterion, device, norm_stats=None, return_sample_data=False):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    interpretability_sample = None
    sample_inputs = None
    sample_targets = None
    sample_output = None
    
    with torch.no_grad():
        for i, (batch_inputs, batch_targets) in enumerate(val_loader):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            output = model(batch_inputs)
            forecasts = output['forecasts']  # (B, M, H)
            
            # Save sample data from first batch for visualization
            if i == 0:
                interpretability_sample = output['interpretability']
                if return_sample_data:
                    sample_inputs = batch_inputs.cpu()
                    sample_targets = batch_targets.cpu()
                    sample_output = {
                        'forecasts': forecasts.cpu(),
                        'interpretability': {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                           for k, v in output['interpretability'].items()}
                    }
            
            # Transpose targets
            targets_transposed = batch_targets.transpose(1, 2)
            
            # Compute loss
            loss = criterion(forecasts, targets_transposed)
            total_loss += loss.item()
            
            # Collect for metric computation
            all_predictions.append(forecasts.cpu())
            all_targets.append(targets_transposed.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Denormalize for interpretable metrics
    if norm_stats is not None:
        predictions_denorm = denormalize_forecasts(all_predictions, norm_stats)
        targets_denorm = denormalize_forecasts(all_targets, norm_stats)
    else:
        predictions_denorm = all_predictions
        targets_denorm = all_targets
    
    # Compute metrics on denormalized scale
    metrics = compute_forecasting_metrics(predictions_denorm, targets_denorm)
    
    avg_loss = total_loss / len(val_loader)
    
    if return_sample_data:
        return avg_loss, metrics, interpretability_sample, sample_inputs, sample_targets, sample_output
    
    return avg_loss, metrics, interpretability_sample


def main():
    parser = argparse.ArgumentParser(description='Train Extended Forecasting Model')
    
    # Data arguments
    parser.add_argument('--data-path', default='ETT-small/ETTh1.csv')
    parser.add_argument('--input-length', type=int, default=96)
    parser.add_argument('--forecast-horizon', type=int, default=24)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--train-ratio', type=float, default=0.7)
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--test-ratio', type=float, default=0.1)
    
    # Model arguments
    parser.add_argument('--embed-dim', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--cross-dim', type=int, default=32)
    parser.add_argument('--cross-heads', type=int, default=4)
    parser.add_argument('--compressed-dim', type=int, default=64)
    parser.add_argument('--num-control-points', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--scheduler', action='store_true', help='Use learning rate scheduler')
    
    # Logging arguments
    parser.add_argument('--save-dir', default='training_outputs')
    parser.add_argument('--save-freq', type=int, default=5, help='Save attention heatmaps every N epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    save_dir = Path(args.save_dir) / datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(save_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize interpretability tracker
    tracker = InterpretabilityTracker(save_dir / 'interpretability')
    
    print("\n📊 Loading and splitting data...")
    
    # Use proper temporal splitting
    split_config = DataSplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    data_splitter = ETTDataSplitter(
        file_path=args.data_path,
        split_config=split_config,
        normalize='standard'
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(data_splitter, args)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Get variable names for visualization
    variable_names = data_splitter.variables
    num_variables = len(variable_names)
    
    print(f"\n🏗️ Building model...")
    
    # Create model configuration
    model_config = ExtendedModelConfig(
        num_variables=num_variables,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        cross_dim=args.cross_dim,
        cross_heads=args.cross_heads,
        compressed_dim=args.compressed_dim,
        num_control_points=args.num_control_points,
        forecast_horizon=args.forecast_horizon,
        dropout=args.dropout
    )
    
    model = InterpretableForecastingModel(model_config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = None
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    criterion = nn.MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\n🎯 Starting training for {args.epochs} epochs...\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_mae = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_metrics, interpretability = validate(
            model, val_loader, criterion, device,
            norm_stats=data_splitter.splitter.norm_stats
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_metrics['mae'])
        history['val_rmse'].append(val_metrics['rmse'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} MAE: {train_mae:.4f} | "
              f"Val Loss: {val_loss:.4f} MAE: {val_metrics['mae']:.4f} "
              f"RMSE: {val_metrics['rmse']:.4f}")
        
        # Save interpretability visualizations periodically
        if epoch % args.save_freq == 0:
            # Save attention heatmaps
            tracker.save_attention_heatmaps(
                interpretability, epoch, variable_names, prefix='val'
            )
            
            # Save spline visualizations (get sample data)
            print(f"      🎨 Generating spline visualizations...")
            val_loss_viz, val_metrics_viz, interp_viz, inputs_viz, targets_viz, output_viz = validate(
                model, val_loader, criterion, device,
                norm_stats=data_splitter.splitter.norm_stats,
                return_sample_data=True
            )
            tracker.save_spline_visualizations(
                output_viz, inputs_viz, targets_viz, variable_names, epoch, prefix='val'
            )
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': model_config.__dict__,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'history': history
            }, save_dir / 'best_model.pt')
            
            print(f"   ✅ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n⏹️ Early stopping triggered after {epoch} epochs")
            break
    
    # Save final training curves
    tracker.save_training_curves(history)
    
    print(f"\n🎊 Training complete!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Outputs saved to: {save_dir}")
    
    # Final evaluation on test set
    print(f"\n📈 Evaluating on test set...")
    
    checkpoint = torch.load(save_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get test metrics and sample data for visualization
    test_loss, test_metrics, test_interpretability, test_inputs, test_targets, test_output = validate(
        model, test_loader, criterion, device,
        norm_stats=data_splitter.splitter.norm_stats,
        return_sample_data=True
    )
    
    print(f"   Test MSE: {test_loss:.4f}")
    print(f"   Test MAE: {test_metrics['mae']:.4f}")
    print(f"   Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"   Test MAPE: {test_metrics['mape']:.2f}%")
    
    # Save final spline visualizations from test set
    print(f"\n🎨 Creating final test set visualizations...")
    tracker.save_spline_visualizations(
        test_output, test_inputs, test_targets, variable_names, 
        epoch=checkpoint['epoch'], prefix='test_final'
    )
    
    # Save test results
    with open(save_dir / 'test_results.json', 'w') as f:
        json.dump({
            'test_loss': test_loss,
            'test_metrics': {k: float(v) for k, v in test_metrics.items()},
            'best_epoch': checkpoint['epoch']
        }, f, indent=2)
    
    print(f"\n✅ All done! Check {save_dir} for results.")
    print(f"   📊 Training curves: {save_dir / 'interpretability' / 'training_curves.png'}")
    print(f"   🔍 Attention heatmaps: {save_dir / 'interpretability' / 'val_attention_epoch_*.png'}")
    print(f"   📈 Spline visualizations: {save_dir / 'interpretability' / 'splines_*'}")
    print(f"   📝 Test results: {save_dir / 'test_results.json'}")


if __name__ == '__main__':
    main()