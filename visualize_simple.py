"""
Simple visualization script to understand UnivariateFunctionLearner
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model import UnivariateFunctionLearner, load_ett_data

def create_simple_visualization():
    """Create a simple, focused visualization of what the UnivariateFunctionLearner does."""
    
    print("Creating simple visualization...")
    
    # Load data
    data, dates, variables = load_ett_data()
    seq_len = 30  # Use 30 time steps for clarity
    
    # Focus on one variable (HUFL - first variable)
    var_name = variables[0]  # HUFL
    input_series = data[:seq_len, 0]  # Shape: (30,)
    
    # Create model with small embedding dimension for visualization
    embed_dim = 4
    learner = UnivariateFunctionLearner(in_features=1, out_features=embed_dim, hidden_features=8)
    
    # Prepare input tensor
    input_tensor = torch.tensor(input_series, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)  # (1, 30, 1)
    
    # Forward pass
    with torch.no_grad():
        embeddings = learner(input_tensor)  # (1, 30, 4)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Understanding UnivariateFunctionLearner: {var_name} Variable', fontsize=16, fontweight='bold')
    
    # 1. Original time series
    axes[0, 0].plot(range(seq_len), input_series, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title('1. Input: Original Time Series', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel(f'{var_name} Value')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.02, 0.98, f'Shape: ({seq_len},)\nScalar values over time', 
                    transform=axes[0, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    # 2. Embedding heatmap
    embedding_matrix = embeddings[0].numpy().T  # (4, 30)
    im = axes[0, 1].imshow(embedding_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0, 1].set_title('2. Output: Embedding Matrix', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Embedding Dimension')
    plt.colorbar(im, ax=axes[0, 1], shrink=0.8)
    axes[0, 1].text(0.02, 0.98, f'Shape: ({embed_dim}, {seq_len})\nEach scalar → {embed_dim}D vector', 
                    transform=axes[0, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # 3. Individual embedding dimensions
    colors = ['red', 'green', 'blue', 'orange']
    for dim in range(embed_dim):
        axes[1, 0].plot(range(seq_len), embeddings[0, :, dim].numpy(), 
                       color=colors[dim], label=f'Embedding Dim {dim}', 
                       linewidth=2, marker='s', markersize=3)
    axes[1, 0].set_title('3. Embedding Dimensions Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Time Step')
    axes[1, 0].set_ylabel('Embedding Value')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].text(0.02, 0.98, 'Each dimension captures\ndifferent patterns', 
                    transform=axes[1, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.7))
    
    # 4. Input vs Output relationship
    input_vals = input_series
    embedding_norms = torch.norm(embeddings[0], dim=1).numpy()  # L2 norm of each embedding
    
    scatter = axes[1, 1].scatter(input_vals, embedding_norms, 
                                c=range(len(input_vals)), cmap='plasma', 
                                s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_title('4. Input-Output Relationship', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel(f'Input Value ({var_name})')
    axes[1, 1].set_ylabel('Embedding Magnitude (L2 Norm)')
    axes[1, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 1], label='Time Step', shrink=0.8)
    axes[1, 1].text(0.02, 0.98, 'Shows how input values\nmap to embedding space', 
                    transform=axes[1, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('simple_univariate_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print explanation
    print("\n" + "="*60)
    print("EXPLANATION: What the UnivariateFunctionLearner Does")
    print("="*60)
    print(f"""
🔍 THE TRANSFORMATION PROCESS:

1. INPUT: Scalar Time Series
   • We start with {var_name} values over time: [{input_series[0]:.2f}, {input_series[1]:.2f}, {input_series[2]:.2f}, ...]
   • Each time step has just ONE number (scalar)
   • Shape: ({seq_len},) - just a sequence of numbers

2. PROCESSING: 2-Layer MLP
   • Each scalar value goes through a small neural network
   • Layer 1: Linear(1 → 8) + ReLU activation
   • Layer 2: Linear(8 → {embed_dim})
   • This happens independently for each time step

3. OUTPUT: Vector Embeddings
   • Each scalar becomes a {embed_dim}-dimensional vector
   • Shape: ({seq_len}, {embed_dim}) - now each time step has {embed_dim} numbers
   • These embeddings capture richer representations of the input

4. WHY THIS MATTERS:
   • Scalar values have limited expressiveness
   • Vector embeddings can capture complex patterns and relationships
   • Each embedding dimension can specialize in different aspects
   • This prepares the data for the attention mechanism (next module)

🎯 KEY INSIGHT:
The UnivariateFunctionLearner transforms simple scalar time series into rich, 
multi-dimensional representations that can better capture the underlying patterns
in the data. It's like going from black-and-white to color - more information
to work with!
""")

if __name__ == "__main__":
    create_simple_visualization()