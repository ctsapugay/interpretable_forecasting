# InterpretableTimeEncoder: How the Two Modules Work Together

This document provides a detailed explanation of how the InterpretableTimeEncoder integrates the UnivariateFunctionLearner and TemporalSelfAttention modules to create a powerful, interpretable time series forecasting model.

## Architecture Overview

The InterpretableTimeEncoder follows a **split-process-combine** paradigm:

```
Input: (B, T, M) → Split → Process → Combine → Output: (B, M, T, d_embed)
```

Where:
- `B` = Batch size
- `T` = Time sequence length  
- `M` = Number of variables (7 for ETT dataset)
- `d_embed` = Embedding dimension

## The Two Core Modules

### 1. UnivariateFunctionLearner (UFL)
**Purpose**: Transform scalar time series values into rich vector embeddings

**Architecture**:
```python
Input: (B, T, 1) → Linear(1→64) → ReLU → Linear(64→d_embed) → Output: (B, T, d_embed)
```

**Key Characteristics**:
- **No Parameter Sharing**: Each variable has its own dedicated UFL instance
- **Time-Distributed**: Processes each time step independently
- **Variable-Specific**: Learns unique transformations for each variable's patterns

### 2. TemporalSelfAttention (TSA)
**Purpose**: Capture temporal dependencies within embedding sequences

**Architecture**:
```python
Input: (B, T, d_embed) → +PositionalEncoding → LayerNorm → MultiHeadAttention → +Residual → Output: (B, T, d_embed)
```

**Key Characteristics**:
- **Shared Across Variables**: Same attention module processes all variables
- **Temporal Focus**: Learns which time steps are important for each query position
- **Residual Connection**: Preserves original embedding information

## Integration Strategy: The Best of Both Worlds

### Design Philosophy

The InterpretableTimeEncoder combines these modules to achieve:

1. **Variable Independence**: Each variable maintains its unique characteristics through dedicated UFL
2. **Temporal Coherence**: Shared attention ensures consistent temporal modeling across variables
3. **Interpretability**: Both embedding magnitudes and attention weights provide clear insights

### Step-by-Step Integration Process

#### Step 1: Input Decomposition
```python
# Input: multivariate time series (B, T, M)
x = torch.randn(batch_size, seq_len, num_variables)

# Split into individual variables
for i in range(num_variables):
    x_i = x[:, :, i:i+1]  # Extract variable i: (B, T, 1)
```

#### Step 2: Independent Variable Processing
```python
# Each variable gets its own UnivariateFunctionLearner
u_i = self.univariate_learners[i](x_i)  # (B, T, 1) → (B, T, d_embed)
```

**Why No Parameter Sharing?**
- Different variables have different scales, patterns, and meanings
- HUFL (high useful load) vs OT (oil temperature) require different transformations
- Independent learners preserve variable-specific information

#### Step 3: Shared Temporal Processing
```python
# Same attention module processes all variables
h_i, attn_i = self.temporal_attention(u_i)  # (B, T, d_embed) → (B, T, d_embed), (B, heads, T, T)
```

**Why Shared Attention?**
- Ensures consistent temporal modeling across all variables
- Reduces parameter count compared to variable-specific attention
- Enables learning of universal temporal patterns

#### Step 4: Output Stacking
```python
# Stack results for all variables
h_stack = torch.stack(all_h, dim=1)      # (B, M, T, d_embed)
attn_stack = torch.stack(all_attn, dim=1) # (B, M, heads, T, T)
```

## Key Integration Benefits

### 1. Balanced Specialization and Generalization

**Specialization (UFL)**:
- Each variable learns its own optimal embedding transformation
- Preserves variable-specific patterns and scales
- Handles heterogeneous data effectively

**Generalization (TSA)**:
- Shared temporal understanding across variables
- Consistent attention patterns for similar temporal structures
- Efficient parameter usage

### 2. Interpretability at Multiple Levels

**Variable Level**:
- Embedding magnitudes show how much "information" each variable carries
- Different variables show distinct embedding patterns

**Temporal Level**:
- Attention weights reveal which time steps are important
- Attention patterns vary meaningfully between variables

**Cross-Variable Level**:
- Embedding similarities show which variables have related patterns
- Attention focus differences reveal variable-specific temporal characteristics

### 3. Computational Efficiency

**Parameter Distribution**:
```
Total Parameters ≈ 36K (for ETT configuration)
├── UnivariateFunctionLearners: ~21K (58%) - 7 × (1×64 + 64×32 + biases)
└── TemporalSelfAttention: ~15K (42%) - Shared across all variables
```

**Processing Flow**:
- Linear complexity in number of variables (M)
- Quadratic complexity in sequence length (T) only in attention
- Efficient parallel processing of variables

## Comparison with Alternative Approaches

### vs. Fully Shared Parameters
```python
# Alternative: Same UFL for all variables
shared_ufl = UnivariateFunctionLearner()
for i in range(num_variables):
    u_i = shared_ufl(x_i)  # Same transformation for all
```
**Problem**: Loses variable-specific patterns, poor performance on heterogeneous data

### vs. Fully Independent Models
```python
# Alternative: Separate attention for each variable
for i in range(num_variables):
    u_i = self.univariate_learners[i](x_i)
    h_i = self.temporal_attentions[i](u_i)  # Different attention per variable
```
**Problem**: Too many parameters, inconsistent temporal modeling, overfitting risk

### vs. Traditional Approaches
**Compared to standard Transformers**:
- More interpretable (explicit variable separation)
- Better handling of heterogeneous variables
- Clearer temporal attention patterns

**Compared to RNNs/LSTMs**:
- Parallel processing (faster training)
- Direct access to all time steps (better long-range dependencies)
- Interpretable attention weights

## Implementation Details

### Forward Pass Logic
```python
def forward(self, x):
    batch_size, seq_len, num_vars = x.shape
    all_h, all_attn = [], []
    
    # Process each variable independently
    for i in range(self.num_variables):
        # Step 1: Extract variable
        x_i = x[:, :, i:i+1]
        
        # Step 2: Variable-specific embedding
        u_i = self.univariate_learners[i](x_i)
        
        # Step 3: Shared temporal attention
        h_i, attn_i = self.temporal_attention(u_i)
        
        all_h.append(h_i)
        all_attn.append(attn_i)
    
    # Step 4: Stack outputs
    return torch.stack(all_h, dim=1), torch.stack(all_attn, dim=1)
```

### Key Design Decisions

1. **Module Ordering**: UFL → TSA (not TSA → UFL)
   - Embeddings first, then temporal modeling
   - Allows attention to work on rich representations

2. **Residual Connections**: Only in attention module
   - Preserves embedding information through attention
   - Prevents gradient vanishing in temporal processing

3. **Positional Encoding**: In attention module only
   - UFL processes time steps independently (no position needed)
   - TSA needs position information for temporal modeling

## Usage and Extensions

### Basic Usage
```python
model = InterpretableTimeEncoder(
    num_variables=7,      # ETT dataset variables
    embed_dim=32,         # Embedding dimension
    hidden_dim=64,        # UFL hidden dimension
    num_heads=4,          # Attention heads
    dropout=0.1,          # Dropout rate
    max_len=512          # Maximum sequence length
)

# Forward pass
h_stack, attn_stack = model(x)  # x: (B, T, 7)
```

### Potential Extensions

1. **Variable-Specific Attention Heads**: Different number of heads per variable
2. **Hierarchical Attention**: Multiple attention layers with different temporal scales
3. **Cross-Variable Attention**: Additional module for inter-variable dependencies
4. **Adaptive Embedding Dimensions**: Different embedding sizes per variable

## Conclusion

The InterpretableTimeEncoder successfully integrates univariate function learners and temporal self-attention by:

- **Preserving Variable Identity**: Through dedicated function learners
- **Ensuring Temporal Consistency**: Through shared attention mechanism  
- **Maintaining Interpretability**: Through clear separation of concerns
- **Achieving Efficiency**: Through balanced parameter sharing

This integration creates a model that is both powerful and interpretable, making it ideal for time series forecasting applications where understanding model behavior is as important as prediction accuracy.