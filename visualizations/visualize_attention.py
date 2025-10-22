"""
Attention Layer Visualization

This script creates simple, clear visualizations showing how the TemporalSelfAttention
layer works with time series data.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys; sys.path.append('../main model'); from model import TemporalSelfAttention, UnivariateFunctionLearner, load_ett_data

# Set style for better-looking plots
plt.style.use('default')

def create_attention_visualizations():
    """Create three simple graphs showing attention layer behavior."""
    
    print("Creating attention layer visualizations...")
    
    # Load sample data
    data, dates, variables = load_ett_data()
    seq_len = 20  # Use shorter sequence for clarity
    
    # Create models
    embed_dim = 32
    num_heads = 4
    learner = UnivariateFunctionLearner(in_features=1, out_features=embed_dim)
    attention = TemporalSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    # Prepare input data (using HUFL variable)
    input_data = torch.tensor(data[:seq_len, 0:1], dtype=torch.float32).unsqueeze(0)  # (1, T, 1)
    
    # Get embeddings and attention
    with torch.no_grad():
        embeddings = learner(input_data)  # (1, T, embed_dim)
        attended_embeddings, attn_weights = attention(embeddings)  # (1, T, embed_dim), (1, heads, T, T)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Graph 1: Attention Heatmap
    ax1 = axes[0]
    # Average attention weights across heads for simplicity
    avg_attention = attn_weights[0].mean(dim=0).numpy()  # (T, T)
    
    im1 = ax1.imshow(avg_attention, cmap='Blues', aspect='auto')
    ax1.set_title('Graph 1: Attention Patterns\n(How each time step attends to others)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Key Position (Time Step)')
    ax1.set_ylabel('Query Position (Time Step)')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Attention Weight')
    
    # Add grid for better readability
    ax1.set_xticks(range(0, seq_len, 5))
    ax1.set_yticks(range(0, seq_len, 5))
    ax1.grid(True, alpha=0.3)
    
    # Graph 2: Before vs After Attention
    ax2 = axes[1]
    
    # Compare embedding magnitudes before and after attention
    before_norms = torch.norm(embeddings[0], dim=1).numpy()
    after_norms = torch.norm(attended_embeddings[0], dim=1).numpy()
    
    time_steps = np.arange(seq_len)
    ax2.plot(time_steps, before_norms, 'o-', label='Before Attention', alpha=0.7, linewidth=2)
    ax2.plot(time_steps, after_norms, 's-', label='After Attention', alpha=0.7, linewidth=2)
    
    ax2.set_title('Graph 2: Embedding Changes\n(How attention modifies representations)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Embedding Magnitude (L2 Norm)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Graph 3: Attention Focus Over Time
    ax3 = axes[2]
    
    # Show how attention focus changes for different query positions
    focus_positions = [0, seq_len//4, seq_len//2, 3*seq_len//4, seq_len-1]
    colors = plt.cm.Set1(np.linspace(0, 1, len(focus_positions)))
    
    for i, pos in enumerate(focus_positions):
        attention_pattern = avg_attention[pos, :]  # Attention from position 'pos'
        ax3.plot(time_steps, attention_pattern, 'o-', 
                label=f'Query at t={pos}', color=colors[i], alpha=0.8)
    
    ax3.set_title('Graph 3: Attention Focus\n(Where different time steps look)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Key Position (Time Step)')
    ax3.set_ylabel('Attention Weight')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attention_visualizations.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Attention visualizations saved as 'attention_visualizations.png'")
    
    # Print some insights
    print("\n🔍 Key Insights:")
    print(f"1. Attention matrix shape: {avg_attention.shape} (each cell shows how much one time step attends to another)")
    print(f"2. Diagonal dominance: {np.mean(np.diag(avg_attention)):.3f} (self-attention strength)")
    print(f"3. Average attention change: {np.mean(np.abs(after_norms - before_norms)):.3f} (how much embeddings change)")
    
    return avg_attention, before_norms, after_norms

if __name__ == "__main__":
    create_attention_visualizations()