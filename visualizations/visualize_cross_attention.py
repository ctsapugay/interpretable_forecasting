"""
Cross-Variable Attention Visualization

This script creates simple, clear visualizations showing how the CrossVariableAttention
layer behaves in the extended forecasting model.

It mirrors the style of visualize_attention.py (which shows temporal self-attention),
but here we focus on variable-to-variable attention (M x M).
"""

import sys
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

# Prefer repo-local imports; fall back gracefully if some utilities differ.
# - extended_model: InterpretableForecastingModel, ExtendedModelConfig
# - data_utils: ETTDataLoader (nice-to-have)
# - model: load_ett_data (fallback if loader is unavailable)
try:
    import sys; sys.path.append('../main model'); from extended_model import InterpretableForecastingModel, ExtendedModelConfig
except Exception as e:
    print("❌ Could not import extended_model. Make sure extended_model.py is in the repo.")
    raise

try:
    from data_utils import ETTDataLoader
    HAVE_LOADER = True
except Exception:
    HAVE_LOADER = False

try:
    import sys; sys.path.append('../main model'); from model import load_ett_data
    HAVE_LOAD_ETT = True
except Exception:
    HAVE_LOAD_ETT = False

plt.style.use("default")


def _make_batch_from_loader(window_size=96, batch_size=4):
    """
    Preferred path: use ETTDataLoader like in tests/validation.
    Returns: batch (B,T,M), variable_names (list[str])
    """
    loader = ETTDataLoader(
        file_path="ETT-small/ETTh1.csv",
        normalize="standard",
        num_samples=1000
    )
    windows, _ = loader.get_windows(window_size=window_size, stride=window_size//2, as_torch=True)
    if windows.shape[0] < batch_size:
        batch_size = max(1, int(windows.shape[0]))
    batch = windows[:batch_size]  # (B, T, M)
    return batch, list(getattr(loader, "variables", ["HUFL","HULL","MUFL","MULL","LUFL","LULL","OT"]))


def _make_batch_fallback(window_size=96, batch_size=4):
    """
    Fallback if ETTDataLoader isn't available: use load_ett_data() to build a quick window.
    Returns: batch (B,T,M), variable_names (list[str])
    """
    if not HAVE_LOAD_ETT:
        raise RuntimeError(
            "Neither ETTDataLoader nor load_ett_data() is available. "
            "Please ensure data_utils.py or model.py is present."
        )
    data, dates, variables = load_ett_data()  # data: (N, M) numpy
    M = data.shape[1]
    T = min(window_size, data.shape[0])
    x = torch.tensor(data[:T, :], dtype=torch.float32)  # (T, M)
    x = x.unsqueeze(0).repeat(batch_size, 1, 1)         # (B, T, M)
    return x, list(variables)


def _get_cross_attention_and_embeddings(batch, config):
    """
    Run the extended model forward and fetch:
      - cross_attention: (B, heads, M, M)
      - variable_embeddings: (B, M, T, E)  -- pre cross-attn
      - cross_embeddings:    (B, M, T, C)  -- post cross-attn
    """
    model = InterpretableForecastingModel(config).eval()
    with torch.no_grad():
        out = model(batch)
    inte = out["interpretability"]
    cross_attn = inte["cross_attention"]                    # (B, H, M, M)
    var_emb  = inte.get("variable_embeddings", None)        # (B, M, T, E)
    cross_emb = inte.get("cross_embeddings", None)          # (B, M, T, C)
    return cross_attn, var_emb, cross_emb


def create_cross_attention_visualizations():
    print("Creating cross-variable attention visualizations...")

    # --- 1) Build a small batch (B,T,M) and names ---
    window_size = 96
    batch_size = 4
    if HAVE_LOADER:
        batch, variable_names = _make_batch_from_loader(window_size=window_size, batch_size=batch_size)
    else:
        batch, variable_names = _make_batch_fallback(window_size=window_size, batch_size=batch_size)

    B, T, M = batch.shape
    print(f"Input batch shape: {batch.shape} (B,T,M) with variables: {variable_names}")

    # --- 2) Config & forward pass (cross-attention happens inside) ---
    config = ExtendedModelConfig(
        num_variables=M,
        embed_dim=32,
        hidden_dim=64,
        num_heads=4,        # temporal heads
        cross_dim=32,       # output dim of cross-attn
        cross_heads=4,      # number of cross-attn heads
        compressed_dim=64,
        compression_ratio=4,
        num_control_points=8,
        spline_degree=3,
        forecast_horizon=24,
        dropout=0.1
    )

    cross_attn, var_emb, cross_emb = _get_cross_attention_and_embeddings(batch, config)
    print(f"cross_attention shape: {tuple(cross_attn.shape)}  (B, heads, M, M)")
    if var_emb is not None:
        print(f"variable_embeddings shape: {tuple(var_emb.shape)}  (B, M, T, E)")
    if cross_emb is not None:
        print(f"cross_embeddings shape: {tuple(cross_emb.shape)}  (B, M, T, C)")

    # --- Average attention over batch & heads → M x M matrix ---
    A = cross_attn.mean(dim=(0, 1)).cpu().numpy()  # (M, M)

    # --- 3) Build figure with 3 panels (heatmap; norms pre vs post; focus rows) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Graph 1: Cross-Attention Heatmap (M x M)
    ax1 = axes[0]
    im1 = ax1.imshow(A, cmap="Blues", aspect="auto")
    ax1.set_title("Graph 1: Cross-Variable Attention\n(How each variable attends to others)",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Key (Attended) Variable")
    ax1.set_ylabel("Query (Attending) Variable")
    ax1.set_xticks(range(M)); ax1.set_xticklabels(variable_names, rotation=45, ha="right")
    ax1.set_yticks(range(M)); ax1.set_yticklabels(variable_names)
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label("Attention Weight")
    ax1.grid(True, alpha=0.2)

    # Graph 2: Before vs After Cross-Attention (variable-wise norms)
    # We’ll compare L2 norms per variable aggregated over time (and embedding dims).
    ax2 = axes[1]
    if (var_emb is not None) and (cross_emb is not None):
        # var_emb:  (B, M, T, E) → per-variable norm: mean over B and T of L2 over E
        before = torch.norm(var_emb, dim=-1).mean(dim=(0, 2)).cpu().numpy()   # (M,)
        after  = torch.norm(cross_emb, dim=-1).mean(dim=(0, 2)).cpu().numpy() # (M,)
        x = np.arange(M)
        ax2.plot(x, before, "o-", label="Before Cross-Attn", alpha=0.8, linewidth=2)
        ax2.plot(x, after,  "s-", label="After Cross-Attn",  alpha=0.8, linewidth=2)
        ax2.set_xticks(x); ax2.set_xticklabels(variable_names, rotation=45, ha="right")
        ax2.set_ylabel("Embedding Magnitude (avg L2)")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Embeddings unavailable\n(showing only heatmap & focus)",
                 ha="center", va="center", fontsize=12)
    ax2.set_title("Graph 2: Embedding Changes\n(How cross-attention modifies per-variable reps)",
                  fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # Graph 3: Attention Focus by Variable (rows of A)
    ax3 = axes[2]
    # Pick a few query variables across the index range
    focus_idxs = sorted(set([0, M//4, M//2, (3*M)//4, M-1]))
    colors = plt.cm.Set1(np.linspace(0, 1, len(focus_idxs)))
    for i, vidx in enumerate(focus_idxs):
        row = A[vidx, :]  # distribution over "which variables I (vidx) listen to"
        ax3.plot(np.arange(M), row, "o-", label=f"Query: {variable_names[vidx]}", color=colors[i], alpha=0.9)
    ax3.set_xticks(range(M)); ax3.set_xticklabels(variable_names, rotation=45, ha="right")
    ax3.set_xlabel("Key (Attended) Variable")
    ax3.set_ylabel("Attention Weight")
    ax3.set_title("Graph 3: Attention Focus by Variable\n(Where each variable looks)",
                  fontsize=12, fontweight="bold")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("cross_attention_visualizations.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("✅ Cross-attention visualizations saved as 'cross_attention_visualizations.png'")

    # --- 4) Print some quick insights ---
    row_sums = A.sum(axis=1)
    diag_mean = np.mean(np.diag(A))
    offdiag_mean = (A.sum() - np.trace(A)) / (A.size - M)
    print("\n🔍 Key Insights:")
    print(f"1) Cross-attention matrix shape: {A.shape} (Query var × Key var)")
    print(f"2) Row-sum stats (should be ≈1): min={row_sums.min():.3f}, max={row_sums.max():.3f}, mean={row_sums.mean():.3f}")
    print(f"3) Diagonal mean (self-influence): {diag_mean:.3f} | Off-diagonal mean: {offdiag_mean:.3f}")
    # Top-3 influences per variable
    k = min(3, M)
    print("4) Top-k influences per query variable:")
    for i in range(M):
        topk = np.argsort(-A[i])[:k]
        pairs = ", ".join([f"{variable_names[j]}={A[i,j]:.3f}" for j in topk])
        print(f"   - {variable_names[i]} → {pairs}")

    return A


if __name__ == "__main__":
    try:
        create_cross_attention_visualizations()
    except Exception as e:
        print("❌ Failed to create cross-attention visualizations.")
        print(e)
        sys.exit(1)
