"""
Comprehensive visualizations for the InterpretableTimeEncoder integrated model.

This script creates visualizations showing how the univariate function learners
and temporal attention modules work together in the integrated model.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys; sys.path.append('../main model'); from model import InterpretableTimeEncoder, ModelConfig, load_ett_data
# import seaborn as sns  # Not needed for this visualization

def visualize_integrated_model():
    """Create comprehensive visualizations of the InterpretableTimeEncoder."""
    print("🎨 Creating visualizations for InterpretableTimeEncoder...")
    
    # Load ETT data
    data, dates, variables = load_ett_data()
    seq_len = 48  # Use 48 time steps (2 days of hourly data)
    
    # Create integrated model with ETT configuration
    config = ModelConfig()
    model = InterpretableTimeEncoder(
        num_variables=config.num_variables,
        embed_dim=16,  # Smaller for visualization
        hidden_dim=32,
        num_heads=4,
        dropout=0.0,  # Disable dropout for consistent visualization
        max_len=512
    )
    
    # Prepare input data
    input_data = torch.tensor(data[:seq_len], dtype=torch.float32).unsqueeze(0)  # (1, T, M)
    
    # Set model to evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Forward pass through integrated model
        h_stack, attn_stack = model(input_data)  # (1, M, T, d), (1, M, heads, T, T)
        
        # Also get intermediate outputs for visualization
        intermediate_embeddings = []
        for i in range(config.num_variables):
            x_i = input_data[:, :, i:i+1]  # (1, T, 1)
            u_i = model.univariate_learners[i](x_i)  # (1, T, embed_dim)
            intermediate_embeddings.append(u_i[0])  # Remove batch dimension
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Original Multivariate Time Series
    plt.subplot(4, 4, 1)
    for i, var in enumerate(variables):
        plt.plot(dates[:seq_len], data[:seq_len, i], label=var, alpha=0.8, linewidth=1.5)
    plt.title("Original ETT Dataset (7 Variables)", fontsize=12, fontweight='bold')
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 2. Architecture Flow Diagram
    plt.subplot(4, 4, 2)
    plt.text(0.1, 0.9, "InterpretableTimeEncoder Architecture", fontsize=12, fontweight='bold')
    plt.text(0.1, 0.8, "Input: (B, T, M=7) Multivariate Series", fontsize=10)
    plt.text(0.1, 0.7, "↓", fontsize=12)
    plt.text(0.1, 0.65, "Split into M individual variables", fontsize=10)
    plt.text(0.1, 0.55, "↓", fontsize=12)
    plt.text(0.1, 0.5, "M × UnivariateFunctionLearner", fontsize=10, color='blue')
    plt.text(0.1, 0.4, "↓", fontsize=12)
    plt.text(0.1, 0.35, "Shared TemporalSelfAttention", fontsize=10, color='red')
    plt.text(0.1, 0.25, "↓", fontsize=12)
    plt.text(0.1, 0.2, "Stack: (B, M, T, d_embed)", fontsize=10)
    plt.text(0.1, 0.1, "Output: Embeddings + Attention", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # 3. Variable Processing Pipeline (show first 3 variables)
    for var_idx in range(3):
        plt.subplot(4, 4, 3 + var_idx)
        
        # Original variable
        original = input_data[0, :, var_idx].numpy()
        plt.plot(dates[:seq_len], original, 'b-', alpha=0.7, label='Original', linewidth=2)
        
        # Embedding magnitude after univariate learner
        embed_before_attn = torch.norm(intermediate_embeddings[var_idx], dim=1).numpy()
        plt.plot(dates[:seq_len], embed_before_attn * np.std(original) / np.std(embed_before_attn), 
                'g--', alpha=0.8, label='After MLP', linewidth=2)
        
        # Final embedding magnitude after attention
        embed_after_attn = torch.norm(h_stack[0, var_idx], dim=1).numpy()
        plt.plot(dates[:seq_len], embed_after_attn * np.std(original) / np.std(embed_after_attn), 
                'r:', alpha=0.8, label='After Attention', linewidth=2)
        
        plt.title(f"{variables[var_idx]} Processing Pipeline", fontsize=11, fontweight='bold')
        plt.xlabel("Time")
        plt.ylabel("Normalized Magnitude")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    # 6. Embedding Space Heatmap (all variables)
    plt.subplot(4, 4, 6)
    # Create heatmap of embedding magnitudes across variables and time
    embed_magnitudes = torch.norm(h_stack[0], dim=2).numpy()  # (M, T)
    
    im = plt.imshow(embed_magnitudes, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.title("Embedding Magnitudes Across Variables", fontsize=12, fontweight='bold')
    plt.xlabel("Time Steps")
    plt.ylabel("Variables")
    plt.yticks(range(len(variables)), variables)
    plt.colorbar(im, shrink=0.8, label='Embedding Magnitude')
    
    # 7. Attention Pattern Visualization (Variable 0 - HUFL)
    plt.subplot(4, 4, 7)
    # Average attention across heads for first variable
    attn_var0 = attn_stack[0, 0].mean(dim=0).numpy()  # Average across heads: (T, T)
    
    im = plt.imshow(attn_var0, cmap='Blues', interpolation='nearest')
    plt.title(f"Attention Pattern: {variables[0]}", fontsize=12, fontweight='bold')
    plt.xlabel("Key Position (Time)")
    plt.ylabel("Query Position (Time)")
    plt.colorbar(im, shrink=0.8, label='Attention Weight')
    
    # 8. Attention Pattern Visualization (Variable 6 - OT)
    plt.subplot(4, 4, 8)
    # Average attention across heads for temperature variable
    attn_var6 = attn_stack[0, 6].mean(dim=0).numpy()  # Average across heads: (T, T)
    
    im = plt.imshow(attn_var6, cmap='Reds', interpolation='nearest')
    plt.title(f"Attention Pattern: {variables[6]}", fontsize=12, fontweight='bold')
    plt.xlabel("Key Position (Time)")
    plt.ylabel("Query Position (Time)")
    plt.colorbar(im, shrink=0.8, label='Attention Weight')
    
    # 9. Cross-Variable Embedding Similarity
    plt.subplot(4, 4, 9)
    # Compute cosine similarity between variable embeddings
    embeddings_flat = h_stack[0].view(config.num_variables, -1)  # (M, T*d)
    similarity_matrix = torch.nn.functional.cosine_similarity(
        embeddings_flat.unsqueeze(1), embeddings_flat.unsqueeze(0), dim=2
    ).numpy()
    
    im = plt.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1, interpolation='nearest')
    plt.title("Cross-Variable Embedding Similarity", fontsize=12, fontweight='bold')
    plt.xlabel("Variables")
    plt.ylabel("Variables")
    plt.xticks(range(len(variables)), variables, rotation=45)
    plt.yticks(range(len(variables)), variables)
    plt.colorbar(im, shrink=0.8, label='Cosine Similarity')
    
    # 10. Temporal Attention Focus (attention entropy)
    plt.subplot(4, 4, 10)
    attention_entropy = []
    for var_idx in range(config.num_variables):
        # Compute entropy of attention weights for each query position
        attn_var = attn_stack[0, var_idx].mean(dim=0)  # Average across heads
        entropy = -torch.sum(attn_var * torch.log(attn_var + 1e-8), dim=1)  # Entropy across keys
        attention_entropy.append(entropy.numpy())
    
    for var_idx, entropy in enumerate(attention_entropy):
        plt.plot(dates[:seq_len], entropy, label=variables[var_idx], alpha=0.8)
    
    plt.title("Attention Focus (Lower = More Focused)", fontsize=12, fontweight='bold')
    plt.xlabel("Time")
    plt.ylabel("Attention Entropy")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 11. Embedding Dimension Analysis
    plt.subplot(4, 4, 11)
    # Show how different embedding dimensions capture patterns
    var_idx = 0  # Focus on HUFL
    embed_dims = h_stack[0, var_idx].numpy()  # (T, embed_dim)
    
    for dim in range(min(4, embed_dims.shape[1])):
        plt.plot(dates[:seq_len], embed_dims[:, dim], 
                label=f'Dim {dim}', alpha=0.8, linewidth=1.5)
    
    plt.title(f"Embedding Dimensions: {variables[var_idx]}", fontsize=12, fontweight='bold')
    plt.xlabel("Time")
    plt.ylabel("Embedding Value")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 12. Model Parameter Distribution
    plt.subplot(4, 4, 12)
    all_params = []
    param_labels = []
    
    # Collect parameters from univariate learners
    for i in range(3):  # First 3 variables
        for name, param in model.univariate_learners[i].named_parameters():
            all_params.extend(param.data.flatten().numpy())
            param_labels.extend([f'UFL{i}_{name.split(".")[-1]}'] * param.numel())
    
    # Collect parameters from attention module
    for name, param in model.temporal_attention.named_parameters():
        if 'weight' in name:
            all_params.extend(param.data.flatten().numpy())
            param_labels.extend([f'Attn_{name.split(".")[-2]}'] * param.numel())
    
    plt.hist(all_params, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title("Model Parameter Distribution", fontsize=12, fontweight='bold')
    plt.xlabel("Parameter Value")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    # 13. Processing Time Comparison
    plt.subplot(4, 4, 13)
    processing_stages = ['Input', 'After MLP', 'After Attention']
    
    # Compute variance for each stage (as a proxy for information content)
    input_var = torch.var(input_data[0], dim=0).numpy()
    mlp_var = torch.var(torch.stack(intermediate_embeddings), dim=1).mean(dim=1).numpy()
    final_var = torch.var(h_stack[0], dim=1).mean(dim=1).numpy()
    
    x_pos = np.arange(len(variables))
    width = 0.25
    
    plt.bar(x_pos - width, input_var, width, label='Input', alpha=0.8)
    plt.bar(x_pos, mlp_var, width, label='After MLP', alpha=0.8)
    plt.bar(x_pos + width, final_var, width, label='After Attention', alpha=0.8)
    
    plt.title("Information Content by Processing Stage", fontsize=12, fontweight='bold')
    plt.xlabel("Variables")
    plt.ylabel("Variance (Information Content)")
    plt.xticks(x_pos, variables, rotation=45)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 14. Attention Head Specialization
    plt.subplot(4, 4, 14)
    # Show how different attention heads focus on different patterns
    var_idx = 0  # HUFL
    head_focus = []
    
    for head in range(config.num_heads):
        attn_head = attn_stack[0, var_idx, head]  # (T, T)
        # Compute how much each head focuses on recent vs distant past
        recent_focus = torch.triu(attn_head, diagonal=-5).sum() / attn_head.sum()
        head_focus.append(recent_focus.item())
    
    plt.bar(range(config.num_heads), head_focus, alpha=0.8, color='coral')
    plt.title(f"Attention Head Specialization: {variables[var_idx]}", fontsize=12, fontweight='bold')
    plt.xlabel("Attention Head")
    plt.ylabel("Recent Focus Ratio")
    plt.grid(True, alpha=0.3)
    
    # 15. Model Summary Statistics
    plt.subplot(4, 4, 15)
    total_params = sum(p.numel() for p in model.parameters())
    ufl_params = sum(p.numel() for learner in model.univariate_learners for p in learner.parameters())
    attn_params = sum(p.numel() for p in model.temporal_attention.parameters())
    
    stats_text = f"""
Model Statistics:
• Total Parameters: {total_params:,}
• UFL Parameters: {ufl_params:,} ({ufl_params/total_params*100:.1f}%)
• Attention Parameters: {attn_params:,} ({attn_params/total_params*100:.1f}%)

Data Processing:
• Input Shape: {input_data.shape}
• Output Shape: {h_stack.shape}
• Attention Shape: {attn_stack.shape}

Architecture:
• Variables: {config.num_variables}
• Embed Dim: {model.embed_dim}
• Attention Heads: {config.num_heads}
"""
    
    plt.text(0.05, 0.95, stats_text, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.5))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    
    # 16. Integration Benefits Visualization
    plt.subplot(4, 4, 16)
    # Compare embedding quality before and after attention
    before_attn_quality = []
    after_attn_quality = []
    
    for var_idx in range(config.num_variables):
        # Quality metric: how well embeddings capture temporal patterns
        original = input_data[0, :, var_idx]
        
        # Before attention (just MLP output)
        mlp_embed = intermediate_embeddings[var_idx]
        mlp_similarity = torch.nn.functional.cosine_similarity(
            mlp_embed[:-1], mlp_embed[1:], dim=1
        ).mean().item()
        before_attn_quality.append(mlp_similarity)
        
        # After attention
        final_embed = h_stack[0, var_idx]
        final_similarity = torch.nn.functional.cosine_similarity(
            final_embed[:-1], final_embed[1:], dim=1
        ).mean().item()
        after_attn_quality.append(final_similarity)
    
    x_pos = np.arange(len(variables))
    width = 0.35
    
    plt.bar(x_pos - width/2, before_attn_quality, width, label='Before Attention', alpha=0.8)
    plt.bar(x_pos + width/2, after_attn_quality, width, label='After Attention', alpha=0.8)
    
    plt.title("Temporal Consistency Improvement", fontsize=12, fontweight='bold')
    plt.xlabel("Variables")
    plt.ylabel("Temporal Consistency")
    plt.xticks(x_pos, variables, rotation=45)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('integrated_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Integrated model visualization complete!")
    print("Saved as 'integrated_model_analysis.png'")
    
    return model, input_data, h_stack, attn_stack


def create_simple_integration_graphs():
    """Create simple, focused graphs showing the integration process."""
    print("\n🎯 Creating simple integration demonstration...")
    
    # Load data and create model
    data, dates, variables = load_ett_data()
    seq_len = 24  # One day of hourly data
    
    model = InterpretableTimeEncoder(
        num_variables=7,
        embed_dim=8,  # Small for clarity
        hidden_dim=16,
        num_heads=2,
        dropout=0.0
    )
    
    # Focus on 3 variables for clarity
    focus_vars = [0, 2, 6]  # HUFL, MUFL, OT
    focus_names = [variables[i] for i in focus_vars]
    
    input_data = torch.tensor(data[:seq_len, focus_vars], dtype=torch.float32).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        # Process with reduced model
        model_small = InterpretableTimeEncoder(
            num_variables=3,
            embed_dim=8,
            hidden_dim=16,
            num_heads=2,
            dropout=0.0
        )
        
        h_stack, attn_stack = model_small(input_data)
        
        # Get intermediate outputs
        intermediate_outputs = []
        for i in range(3):
            x_i = input_data[:, :, i:i+1]
            u_i = model_small.univariate_learners[i](x_i)
            intermediate_outputs.append(u_i[0])
    
    # Create simple 3-graph visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Graph 1: Input Processing
    ax = axes[0]
    for i, (var_idx, var_name) in enumerate(zip(focus_vars, focus_names)):
        original_data = data[:seq_len, var_idx]
        normalized_data = (original_data - original_data.mean()) / original_data.std()
        ax.plot(range(seq_len), normalized_data, 
               label=f'{var_name} (Original)', linewidth=2, alpha=0.8)
    
    ax.set_title('Step 1: Input Time Series\n(Normalized for Comparison)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Normalized Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Graph 2: After Univariate Function Learners
    ax = axes[1]
    for i, var_name in enumerate(focus_names):
        embed_magnitude = torch.norm(intermediate_outputs[i], dim=1).numpy()
        ax.plot(range(seq_len), embed_magnitude, 
               label=f'{var_name} (Embedding)', linewidth=2, alpha=0.8)
    
    ax.set_title('Step 2: After Univariate Function Learners\n(Embedding Magnitudes)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Embedding Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Graph 3: After Temporal Attention
    ax = axes[2]
    for i, var_name in enumerate(focus_names):
        final_magnitude = torch.norm(h_stack[0, i], dim=1).numpy()
        ax.plot(range(seq_len), final_magnitude, 
               label=f'{var_name} (Final)', linewidth=2, alpha=0.8)
    
    ax.set_title('Step 3: After Temporal Self-Attention\n(Final Embeddings)', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Final Embedding Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_integration_process.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Simple integration graphs complete!")
    print("Saved as 'simple_integration_process.png'")


if __name__ == "__main__":
    # Create comprehensive visualization
    visualize_integrated_model()
    
    # Create simple focused graphs
    create_simple_integration_graphs()