# Interpretable Time Series Forecasting Model

A PyTorch implementation of an interpretable time series forecasting model that combines TKAN-style univariate function learners with temporal self-attention mechanisms for multivariate time series analysis.

## 🎯 Overview

This project implements a novel architecture for interpretable time series forecasting that processes each variable independently through specialized univariate function learners, then applies temporal attention to capture time dependencies. The model is designed to be both performant and interpretable, making it suitable for applications where understanding model decisions is crucial.

## 🏗️ Architecture

The model consists of two main components:

### 1. Univariate Function Learner
- **Purpose**: Transforms scalar time series values into rich embedding representations
- **Architecture**: 2-layer MLP with ReLU activation
- **Input**: `(B, T, 1)` - Scalar time series for one variable
- **Output**: `(B, T, embed_dim)` - Dense embeddings
- **Key Features**:
  - Independent processing for each variable (no parameter sharing)
  - Time-distributed application across sequence length
  - Learnable non-linear transformations

### 2. Temporal Self-Attention
- **Purpose**: Captures temporal dependencies within variable embeddings
- **Architecture**: Multi-head self-attention with positional encoding
- **Input**: `(B, T, embed_dim)` - Variable embeddings
- **Output**: `(B, T, embed_dim)` + attention weights
- **Key Features**:
  - Positional encoding for temporal awareness
  - Layer normalization and residual connections
  - Interpretable attention weights showing temporal relationships

### 3. Integrated Model (InterpretableTimeEncoder)
- **Purpose**: Combines both components for multivariate processing
- **Input**: `(B, T, M)` - Multivariate time series with M variables
- **Output**: 
  - Embeddings: `(B, M, T, embed_dim)` - Final variable representations
  - Attention: `(B, M, num_heads, T, T)` - Attention weights per variable
- **Processing Flow**:
  1. Each variable processed independently through its own univariate learner
  2. Resulting embeddings passed through shared temporal attention
  3. Outputs stacked to maintain variable separation

## 📊 Dataset

The model is tested on the **ETT (Electricity Transformer Temperature) dataset**:
- **Variables**: 7 time series (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT)
- **Frequency**: Hourly measurements
- **Domain**: Electricity transformer monitoring data
- **Use Case**: Multivariate time series forecasting

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd interpretable-time-series-forecasting

# Install dependencies
pip install torch pandas numpy matplotlib
```

### Basic Usage

```python
from model import InterpretableTimeEncoder, ModelConfig
from data_utils import ETTDataLoader

# Load and preprocess data
loader = ETTDataLoader(
    file_path="interpretable_forecasting/ETT-small/ETTh1.csv",
    normalize='standard',
    num_samples=1000
)

# Create model
config = ModelConfig(
    num_variables=7,
    embed_dim=32,
    hidden_dim=64,
    num_heads=4,
    dropout=0.1
)

model = InterpretableTimeEncoder(
    num_variables=config.num_variables,
    embed_dim=config.embed_dim,
    hidden_dim=config.hidden_dim,
    num_heads=config.num_heads,
    dropout=config.dropout
)

# Get data windows
windows, _ = loader.get_windows(window_size=96, as_torch=True)
batch = windows[:4]  # Take first 4 samples

# Forward pass
embeddings, attention_weights = model(batch)
print(f"Embeddings shape: {embeddings.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
```

### Running Validation

```bash
# Test the complete model with ETT data
python validate_model.py
```

This will:
- Load ETT dataset and preprocess it
- Test model with different sequence lengths
- Verify gradient computation
- Generate validation visualizations
- Save results to `model_validation_results.png`

## 📁 Project Structure

```
├── README.md                          # This file
├── model.py                          # Core model implementation
├── data_utils.py                     # Data loading and preprocessing utilities
├── validate_model.py                 # Model validation and testing script
├── interpretable_forecasting/        # ETT dataset directory
│   ├── ETT_README.md                # Original dataset documentation
│   └── ETT-small/
│       └── ETTh1.csv                # ETT dataset file
└── .kiro/specs/                     # Project specifications
    └── interpretable-time-series-forecasting/
        ├── requirements.md          # Project requirements
        ├── design.md               # Technical design document
        └── tasks.md                # Implementation tasks
```

## 🔧 Model Components

### ModelConfig
Configuration class with default parameters optimized for ETT dataset:
```python
@dataclass
class ModelConfig:
    num_variables: int = 7      # Number of input variables
    embed_dim: int = 32         # Embedding dimension
    hidden_dim: int = 64        # Hidden layer dimension
    num_heads: int = 4          # Number of attention heads
    dropout: float = 0.1        # Dropout probability
    max_len: int = 512          # Maximum sequence length
```

### UnivariateFunctionLearner
```python
# Processes single variable: (B, T, 1) -> (B, T, embed_dim)
learner = UnivariateFunctionLearner(
    in_features=1,
    out_features=32,
    hidden_features=64
)
```

### TemporalSelfAttention
```python
# Applies temporal attention: (B, T, embed_dim) -> (B, T, embed_dim) + attention
attention = TemporalSelfAttention(
    embed_dim=32,
    num_heads=4,
    dropout=0.1,
    max_len=512
)
```

## 📈 Performance

Based on validation results:
- **Parameters**: ~36K trainable parameters
- **Throughput**: ~654 samples/second (on CPU)
- **Memory**: Efficient processing of sequences up to 512 time steps
- **Scalability**: Linear complexity in sequence length for univariate learners

## 🎨 Interpretability Features

### 1. Variable Independence
- Each variable processed through separate univariate learners
- No parameter sharing between variables
- Clear separation of variable-specific patterns

### 2. Attention Visualization
- Attention weights show temporal dependencies
- Per-variable attention patterns
- Interpretable attention heatmaps

### 3. Embedding Analysis
- Rich embedding representations for each variable
- Embedding space visualization
- Magnitude analysis across time

## 🧪 Testing and Validation

The project includes comprehensive testing:

### Data Loading Tests (`data_utils.py`)
- ETT dataset loading and preprocessing
- Normalization/denormalization verification
- Time series windowing functionality
- Forecasting dataset creation

### Model Validation (`validate_model.py`)
- End-to-end model testing with real data
- Shape verification for all tensor operations
- Gradient computation verification
- Independent variable processing validation
- Performance benchmarking
- Visualization generation

### Test Coverage
- ✅ Multiple sequence lengths (24, 96, 168 time steps)
- ✅ Different batch sizes (1, 2, 4, 8 samples)
- ✅ Gradient flow verification
- ✅ Attention mechanism validation
- ✅ Independent variable processing
- ✅ Performance analysis

## 📊 Visualization Outputs

The validation script generates comprehensive visualizations:

1. **Original Time Series**: Raw ETT data visualization
2. **Embedding Magnitudes**: L2 norms of variable embeddings over time
3. **Attention Heatmaps**: Temporal attention patterns per variable
4. **Embedding Space**: 2D projections of embedding trajectories
5. **Attention Distribution**: Statistical analysis of attention weights
6. **Architecture Summary**: Model structure and parameter counts

## 🔬 Research Applications

This model is suitable for:
- **Multivariate Time Series Forecasting**: Predicting future values
- **Anomaly Detection**: Identifying unusual patterns in time series
- **Feature Analysis**: Understanding variable relationships
- **Interpretable AI**: Applications requiring model explainability
- **Industrial Monitoring**: Equipment health monitoring (like ETT dataset)

## 🛠️ Customization

### Adding New Variables
```python
# Modify ModelConfig for different number of variables
config = ModelConfig(num_variables=10)  # For 10-variable dataset
model = InterpretableTimeEncoder(num_variables=10, ...)
```

### Adjusting Architecture
```python
# Larger embeddings for more complex patterns
config = ModelConfig(embed_dim=64, hidden_dim=128)

# More attention heads for finer temporal modeling
config = ModelConfig(num_heads=8)
```

### Custom Data Loading
```python
# Use ETTDataLoader as template for custom datasets
class CustomDataLoader(ETTDataLoader):
    def __init__(self, file_path, variable_names, ...):
        # Implement custom loading logic
        pass
```

## 📚 References

- **TKAN**: Temporal Kolmogorov-Arnold Networks concept
- **Transformer Architecture**: Self-attention mechanisms
- **ETT Dataset**: Electricity Transformer Temperature dataset
- **Time Series Forecasting**: Multivariate forecasting methodologies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Run validation: `python validate_model.py`
5. Submit a pull request

## 📄 License

This project is open source. Please refer to the license file for details.

## 🔗 Related Files

- **Dataset Documentation**: `interpretable_forecasting/ETT_README.md`
- **Project Specifications**: `.kiro/specs/interpretable-time-series-forecasting/`
- **Requirements**: `.kiro/specs/interpretable-time-series-forecasting/requirements.md`
- **Design Document**: `.kiro/specs/interpretable-time-series-forecasting/design.md`
- **Implementation Tasks**: `.kiro/specs/interpretable-time-series-forecasting/tasks.md`

---

**Built with ❤️ for interpretable AI and time series analysis**