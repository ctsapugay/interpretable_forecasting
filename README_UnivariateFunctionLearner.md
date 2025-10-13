# UnivariateFunctionLearner Implementation

## Overview
The UnivariateFunctionLearner is a core component of the interpretable time series forecasting model, implementing TKAN-style (Temporal Kolmogorov-Arnold Network) univariate function learning. It transforms scalar time series values into rich vector embeddings through a dedicated Multi-Layer Perceptron (MLP).

## Architecture

### Design Philosophy
- **Independence**: Each variable is processed through its own dedicated MLP
- **No Parameter Sharing**: Different variables learn different transformation functions
- **Time-Distributed**: The same transformation is applied to each time step
- **Embedding Focus**: Converts scalar values to vector representations for downstream processing

### Network Structure
```python
class UnivariateFunctionLearner(nn.Module):
    def __init__(self, in_features=1, out_features=32, hidden_features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features),    # 1 → 64
            nn.ReLU(),                                  # Non-linear activation
            nn.Linear(hidden_features, out_features)   # 64 → 32
        )
```

### Layer Details
1. **Input Layer**: Linear transformation from scalar (1D) to hidden dimension
2. **Activation**: ReLU activation for non-linearity and sparsity
3. **Output Layer**: Linear transformation to embedding dimension

## Input/Output Specifications

### Input Format
- **Shape**: `(B, T_in, 1)`
  - `B`: Batch size
  - `T_in`: Input sequence length
  - `1`: Scalar value (single feature)
- **Data Type**: `torch.float32`
- **Content**: Raw time series values for one variable

### Output Format
- **Shape**: `(B, T_in, d_var)`
  - `B`: Batch size (unchanged)
  - `T_in`: Sequence length (unchanged)
  - `d_var`: Embedding dimension (configurable, default 32)
- **Data Type**: `torch.float32`
- **Content**: Dense vector embeddings

## Implementation Details

### Forward Pass Process
1. **Reshape**: Flatten time dimension for batch processing
   ```python
   x_flat = x.view(b * t, -1)  # (B*T, 1)
   ```

2. **Transform**: Apply MLP to all time steps simultaneously
   ```python
   u_flat = self.net(x_flat)  # (B*T, d_var)
   ```

3. **Restore**: Reshape back to original time structure
   ```python
   return u_flat.view(b, t, -1)  # (B, T, d_var)
   ```

### Key Features
- **Vectorized Processing**: Efficient batch computation across time steps
- **Gradient Flow**: Full backpropagation support for end-to-end training
- **Configurable Dimensions**: Flexible embedding and hidden layer sizes
- **Memory Efficient**: Minimal memory overhead through reshape operations

## Configuration Parameters

### ModelConfig Class
```python
@dataclass
class ModelConfig:
    num_variables: int = 7      # ETT dataset variables
    embed_dim: int = 32         # Output embedding dimension
    hidden_dim: int = 64        # Hidden layer size
    num_heads: int = 4          # For downstream attention
    dropout: float = 0.1        # Regularization
    max_len: int = 512          # Maximum sequence length
```

### ETT Dataset Specific
- **Variables**: 7 (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
- **Data Points**: 17,420 hourly measurements
- **Time Range**: 2016-07 to 2018-07
- **Value Ranges**: Variable-specific (temperature, load, etc.)

## Usage Examples

### Basic Usage
```python
# Create learner
learner = UnivariateFunctionLearner(
    in_features=1,
    out_features=32,
    hidden_features=64
)

# Process time series
x = torch.randn(4, 100, 1)  # Batch=4, Time=100, Features=1
embeddings = learner(x)     # Output: (4, 100, 32)
```

### Multi-Variable Processing
```python
# For ETT dataset with 7 variables
learners = nn.ModuleList([
    UnivariateFunctionLearner(out_features=32) 
    for _ in range(7)
])

# Process each variable independently
for i, variable_data in enumerate(multivariate_data.split(1, dim=-1)):
    embeddings_i = learners[i](variable_data)
```

## Testing and Validation

### Automated Tests
The implementation includes comprehensive tests covering:
- **Shape Verification**: Correct input/output tensor shapes
- **Gradient Computation**: Backpropagation functionality
- **Multiple Configurations**: Different batch sizes and embedding dimensions
- **Edge Cases**: Various sequence lengths and parameter settings

### Test Results
```
✅ Test Case 1: B=2, T=10, d_var=32 - PASSED
✅ Test Case 2: B=4, T=20, d_var=64 - PASSED  
✅ Test Case 3: B=1, T=5, d_var=16 - PASSED
```

## Integration with Larger System

### Pipeline Position
1. **Input**: Raw multivariate time series `(B, T, M)`
2. **UnivariateFunctionLearner**: Transform each variable independently
3. **Output**: Stacked embeddings `(B, M, T, d_var)`
4. **Next Stage**: Temporal self-attention for dependency modeling

### Design Rationale
- **Interpretability**: Each variable's transformation is independent and traceable
- **Flexibility**: Different variables can learn different representations
- **Modularity**: Can be used standalone or as part of larger architecture
- **Efficiency**: Vectorized operations for fast computation

## Performance Characteristics

### Computational Complexity
- **Time**: O(B × T × d_var × hidden_dim)
- **Space**: O(B × T × d_var)
- **Parameters**: (1 × hidden_dim) + (hidden_dim × d_var) + biases

### Memory Usage
- **Minimal Overhead**: Reshape operations don't copy data
- **Gradient Storage**: Standard PyTorch automatic differentiation
- **Batch Processing**: Efficient GPU utilization

## Files and Dependencies

### Core Files
- `model.py` - Main implementation
- `visualize_simple.py` - Simple visualization
- `README_UnivariateFunctionLearner.md` - This documentation

### Dependencies
- `torch` - Neural network framework
- `torch.nn` - Neural network modules
- `matplotlib` - Visualization (for testing)
- `pandas` - Data handling (for ETT dataset)
- `numpy` - Numerical operations

### Installation
```bash
pip install torch matplotlib pandas numpy
```

## Future Extensions

### Potential Improvements
1. **Learnable Activations**: Replace ReLU with learnable activation functions
2. **Residual Connections**: Add skip connections for deeper networks
3. **Normalization**: Layer normalization for training stability
4. **Regularization**: Dropout or weight decay for generalization

### Research Directions
1. **Activation Analysis**: Study which activation functions work best
2. **Embedding Dimension**: Optimal embedding sizes for different datasets
3. **Architecture Search**: Automated architecture optimization
4. **Interpretability**: Methods to understand learned transformations

## Citation and References

This implementation is based on the TKAN (Temporal Kolmogorov-Arnold Network) approach for interpretable time series forecasting, designed specifically for the ETT (Electricity Transforming Temperature) dataset benchmark.