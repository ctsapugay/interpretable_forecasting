# TemporalSelfAttention Layer

This document provides a comprehensive explanation of the `TemporalSelfAttention` class, which implements multi-head self-attention with positional encoding for time series forecasting.

## Overview

The `TemporalSelfAttention` layer is designed to capture temporal dependencies within time series embeddings. It allows each time step to attend to all other time steps in the sequence, learning which past and future information is most relevant for each position.

## Architecture

```
Input Embeddings (B, T, d)
         ↓
   Positional Encoding
         ↓
   Layer Normalization
         ↓
  Multi-Head Self-Attention
         ↓
    Dropout + Residual
         ↓
Output Embeddings (B, T, d) + Attention Weights (B, H, T, T)
```

## Key Components

### 1. Positional Encoding
```python
self.positional_encoding = nn.Embedding(max_len, embed_dim)
```

**Purpose:** Injects temporal position information into embeddings
**Implementation:** Learnable embeddings for each position (0 to max_len-1)
**Why needed:** Self-attention is permutation-invariant, so we need to explicitly encode temporal order

**How it works:**
- Each time step t gets a unique positional embedding vector
- Added element-wise to input embeddings: `u_pos = u + pos_emb`
- Allows the model to distinguish between identical values at different time steps

### 2. Layer Normalization (Pre-Norm)
```python
self.norm = nn.LayerNorm(embed_dim)
```

**Purpose:** Stabilizes training and improves gradient flow
**Implementation:** Pre-normalization (normalize before attention, not after)
**Benefits:**
- Reduces internal covariate shift
- Enables deeper networks
- Faster convergence

### 3. Multi-Head Self-Attention
```python
self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
```

**Purpose:** Captures different types of temporal relationships simultaneously
**Key parameters:**
- `embed_dim`: Dimension of input embeddings
- `num_heads`: Number of parallel attention mechanisms
- `dropout`: Regularization within attention
- `batch_first=True`: Input shape is (B, T, d) not (T, B, d)

**How multi-head works:**
1. Input is projected into Query (Q), Key (K), and Value (V) matrices
2. Each head operates on a subset of dimensions (embed_dim // num_heads)
3. Each head computes: `Attention(Q, K, V) = softmax(QK^T / √d_k)V`
4. Outputs from all heads are concatenated and projected

### 4. Residual Connection with Dropout
```python
h = u + self.dropout(attn_output)
```

**Purpose:** Preserves original information and enables deep networks
**Implementation:** Add input to attention output after dropout
**Benefits:**
- Prevents vanishing gradients
- Allows information to flow directly through the network
- Regularization through dropout

## Forward Pass Details

### Input Processing
```python
def forward(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    b, t, d = u.shape  # Batch, Time, Dimension
```

**Input:** `u` with shape `(B, T_in, embed_dim)`
- `B`: Batch size
- `T_in`: Input sequence length
- `embed_dim`: Embedding dimension

### Step-by-Step Execution

1. **Create Positional Encodings**
   ```python
   positions = torch.arange(t, device=u.device).unsqueeze(0).expand(b, -1)
   pos_emb = self.positional_encoding(positions)
   ```
   - Generate position indices [0, 1, 2, ..., T-1]
   - Look up learnable positional embeddings
   - Expand to match batch size

2. **Add Positional Information**
   ```python
   u_pos = u + pos_emb
   ```
   - Element-wise addition of embeddings and positional encodings
   - Now each embedding contains both content and position information

3. **Pre-Layer Normalization**
   ```python
   u_norm = self.norm(u_pos)
   ```
   - Normalize across the embedding dimension
   - Stabilizes the input to attention

4. **Self-Attention Computation**
   ```python
   attn_output, attn_weights = self.attention(u_norm, u_norm, u_norm, average_attn_weights=False)
   ```
   - Query, Key, and Value are all the same (self-attention)
   - `average_attn_weights=False` returns per-head attention weights
   - Output shapes: `attn_output` (B, T, d), `attn_weights` (B, H, T, T)

5. **Residual Connection**
   ```python
   h = u + self.dropout(attn_output)
   ```
   - Add original input to attention output
   - Apply dropout for regularization

### Output
- `h`: Attended embeddings with shape `(B, T_in, embed_dim)`
- `attn_weights`: Attention weights with shape `(B, num_heads, T_in, T_in)`

## Attention Mechanism Deep Dive

### What is Self-Attention?
Self-attention allows each position in a sequence to attend to all positions in the same sequence. For time series:
- Each time step can look at all other time steps
- The model learns which past/future information is relevant
- Captures both short-term and long-term dependencies

### Attention Weight Interpretation
The attention weights `attn_weights[b, h, i, j]` represent:
- Batch `b`, head `h`
- How much position `i` (query) attends to position `j` (key)
- Values are non-negative and approximately sum to 1 across `j`

### Multi-Head Benefits
Different heads can specialize in different patterns:
- **Head 1:** Local dependencies (nearby time steps)
- **Head 2:** Periodic patterns (seasonal relationships)
- **Head 3:** Long-range trends
- **Head 4:** Anomaly detection (unusual patterns)

## Configuration Parameters

### Constructor Arguments
```python
def __init__(self, embed_dim: int = 32, num_heads: int = 4, dropout: float = 0.1, max_len: int = 512):
```

- **`embed_dim`**: Embedding dimension (must be divisible by num_heads)
- **`num_heads`**: Number of attention heads (typically 4, 8, or 16)
- **`dropout`**: Dropout probability for regularization (0.1 is common)
- **`max_len`**: Maximum sequence length for positional encoding

### Typical Configurations
- **Small model:** embed_dim=32, num_heads=4
- **Medium model:** embed_dim=64, num_heads=8
- **Large model:** embed_dim=128, num_heads=16

## Usage Example

```python
# Create attention layer
attention = TemporalSelfAttention(
    embed_dim=32,
    num_heads=4,
    dropout=0.1,
    max_len=512
)

# Input embeddings from UnivariateFunctionLearner
u = torch.randn(2, 20, 32)  # (batch=2, time=20, embed=32)

# Forward pass
h, attn_weights = attention(u)

# Outputs:
# h: (2, 20, 32) - attended embeddings
# attn_weights: (2, 4, 20, 20) - attention patterns
```

## Integration with Time Series Model

The TemporalSelfAttention layer fits into the larger forecasting pipeline:

1. **UnivariateFunctionLearner**: Scalar → Embedding
2. **TemporalSelfAttention**: Embedding → Attended Embedding (this layer)
3. **Future layers**: Attended Embedding → Forecast

This design allows:
- Each variable to be processed independently (UnivariateFunctionLearner)
- Temporal relationships to be captured (TemporalSelfAttention)
- Interpretability through attention weights

## Computational Complexity

- **Time complexity:** O(T² × d) where T is sequence length, d is embedding dimension
- **Space complexity:** O(T² × H) for storing attention weights
- **Bottleneck:** Quadratic scaling with sequence length

For long sequences (T > 1000), consider:
- Sparse attention patterns
- Local attention windows
- Hierarchical attention

## Training Considerations

### Gradient Flow
- Residual connections prevent vanishing gradients
- Pre-normalization improves stability
- Multi-head design provides multiple gradient paths

### Regularization
- Dropout in attention computation
- Dropout in residual connection
- Layer normalization acts as implicit regularization

### Initialization
- PyTorch's MultiheadAttention uses Xavier initialization
- Positional embeddings are randomly initialized
- Layer norm parameters start at identity transformation

## Interpretability

The attention weights provide interpretability:
- **Temporal focus:** Which time steps are most important?
- **Dependency patterns:** Are relationships local or global?
- **Head specialization:** What does each attention head capture?
- **Anomaly detection:** Unusual attention patterns may indicate anomalies

This makes the model suitable for interpretable time series forecasting where understanding "why" the model makes predictions is crucial.