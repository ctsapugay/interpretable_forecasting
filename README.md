# Extended Interpretable Time Series Forecasting Model

A PyTorch implementation of an advanced interpretable time series forecasting model that combines univariate function learners, temporal self-attention, cross-variable attention, temporal compression, and spline-based forecasting for comprehensive multivariate time series analysis and prediction.

## 🎯 Overview

This project implements a novel **extended architecture** for interpretable time series forecasting that processes multivariate time series through a sophisticated pipeline:

1. **Independent Variable Processing** - Each variable processed through specialized univariate function learners
2. **Temporal Self-Attention** - Captures time dependencies within each variable
3. **Cross-Variable Attention** - Models relationships between different variables  
4. **Temporal Compression** - Efficiently compresses sequences while preserving essential information
5. **Spline-Based Forecasting** - Generates interpretable B-spline functions for future predictions

The model is designed to be both highly performant and fully interpretable, making it suitable for applications where understanding model decisions and forecast reasoning is crucial.

## 🏗️ Extended Architecture

The model consists of five main components in a sequential pipeline:

### 1. Univariate Function Learner
- **Purpose**: Transforms scalar time series values into rich embedding representations
- **Architecture**: 2-layer MLP with ReLU activation (per variable)
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

### 3. Cross-Variable Attention ⭐ NEW
- **Purpose**: Models relationships and dependencies between different variables
- **Architecture**: Multi-head attention across variables (not time)
- **Input**: `(B, M, T, embed_dim)` - All variable embeddings
- **Output**: `(B, M, T, cross_dim)` + cross-attention weights `(B, num_heads, M, M)`
- **Key Features**:
  - Variable-to-variable attention (each variable attends to all others)
  - Preserves temporal dimension while modeling inter-variable relationships
  - Interpretable cross-attention heatmaps showing variable dependencies

### 4. Temporal Encoder (Compression) ⭐ NEW
- **Purpose**: Compresses temporal sequences while preserving essential forecasting information
- **Architecture**: Attention-based temporal pooling with learnable compression queries
- **Input**: `(B, M, T, cross_dim)` - Cross-attended variable embeddings
- **Output**: `(B, M, compressed_dim)` - Compressed representations + compression attention
- **Key Features**:
  - Learnable compression that identifies important time steps
  - Variable-specific compression patterns
  - Configurable compression ratio (e.g., T → T/4)
  - Attention weights show which time steps are preserved

### 5. Spline Function Learner ⭐ NEW
- **Purpose**: Generates interpretable B-spline functions for forecasting future values
- **Architecture**: Neural network predicting B-spline control points
- **Input**: `(B, M, compressed_dim)` - Compressed variable representations
- **Output**: 
  - Forecasts: `(B, M, forecast_horizon)` - Predicted future values
  - Control Points: `(B, M, num_control_points)` - Spline parameters
  - Basis Functions: B-spline basis functions for interpretation
- **Key Features**:
  - Mathematically interpretable B-spline curves
  - Extrapolation capability for forecasting
  - Configurable forecast horizons (1, 12, 24, 48 steps)
  - Smooth, continuous predictions with visible control points

### 6. Integrated Extended Model (InterpretableForecastingModel)
- **Purpose**: Combines all components for end-to-end forecasting
- **Input**: `(B, T, M)` - Multivariate time series with M variables
- **Output**: 
  - Forecasts: `(B, M, forecast_horizon)` - Future predictions
  - Interpretability Artifacts:
    - Temporal attention: `(B, M, num_heads, T, T)`
    - Cross attention: `(B, num_heads, M, M)`
    - Compression attention: `(B, M, 1, T)`
    - Spline parameters: Control points, basis functions, knot vectors
- **Processing Flow**:
  ```
  Input (B, T, M)
      ↓
  Univariate Learners → (B, M, T, embed_dim)
      ↓
  Temporal Self-Attention → (B, M, T, embed_dim) + temporal_attn
      ↓
  Cross-Variable Attention → (B, M, T, cross_dim) + cross_attn
      ↓
  Temporal Encoder → (B, M, compressed_dim) + compression_attn
      ↓
  Spline Function Learner → Forecasts + Spline Parameters
  ```

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
from extended_model import InterpretableForecastingModel, ExtendedModelConfig
from data_utils import ETTDataLoader

# Load and preprocess data
loader = ETTDataLoader(
    file_path="interpretable_forecasting/ETT-small/ETTh1.csv",
    normalize='standard',
    num_samples=1000
)

# Create extended model configuration
config = ExtendedModelConfig(
    num_variables=7,           # ETT dataset variables
    embed_dim=32,              # Base embedding dimension
    hidden_dim=64,             # Univariate learner hidden size
    num_heads=4,               # Temporal attention heads
    cross_dim=32,              # Cross-attention dimension
    cross_heads=4,             # Cross-attention heads
    compressed_dim=64,         # Compressed representation size
    compression_ratio=4,       # Temporal compression ratio
    num_control_points=8,      # B-spline control points
    spline_degree=3,           # Cubic B-splines
    forecast_horizon=24,       # Predict 24 steps ahead
    dropout=0.1
)

# Create extended forecasting model
model = InterpretableForecastingModel(config)

# Get data windows for forecasting
windows, _ = loader.get_windows(window_size=168, as_torch=True)
batch = windows[:4]  # Take first 4 samples

# Forward pass - generates forecasts and interpretability artifacts
model.eval()
with torch.no_grad():
    output = model(batch)

# Extract results
forecasts = output['forecasts']  # (B, M, forecast_horizon)
interpretability = output['interpretability']

print(f"Input shape: {batch.shape}")
print(f"Forecasts shape: {forecasts.shape}")
print(f"Cross-attention shape: {interpretability['cross_attention'].shape}")
print(f"Spline control points shape: {interpretability['spline_parameters']['control_points'].shape}")
```

### Forecasting with Accuracy Assessment

```python
from spline_visualization import create_spline_visualizations

# Create input/target pairs for validation
input_data = windows[:-1][:4]  # Historical data
true_future = windows[1:][:4][:, :config.forecast_horizon]  # True future values

# Generate predictions
output = model(input_data)

# Create visualizations with accuracy metrics
figures = create_spline_visualizations(
    model_output=output,
    input_data=input_data,
    true_future=true_future.transpose(1, 2),  # Reshape to (B, forecast_horizon, M)
    variable_names=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
    sample_idx=0,
    output_dir="forecast_results"
)

# Calculate accuracy metrics
mse = torch.mean((output['forecasts'] - true_future.transpose(1, 2)) ** 2)
mae = torch.mean(torch.abs(output['forecasts'] - true_future.transpose(1, 2)))
print(f"Forecast MSE: {mse:.6f}, MAE: {mae:.6f}")
```

### Running Extended Model Validation

```bash
# Test the complete extended model with ETT data and spline visualizations
python validate_extended_model.py
```

This comprehensive validation will:
- Load ETT dataset and preprocess it
- Test extended model with different sequence lengths (24, 96, 168, 336 steps)
- Verify gradient computation through all components
- Test cross-attention, temporal compression, and spline forecasting
- Generate spline visualizations with accuracy assessment
- Test multiple forecast horizons (1, 12, 24, 48 steps)
- Save results and visualizations

### Running Spline Visualization Tests

```bash
# Generate spline forecasting visualizations with accuracy metrics
python test_spline_visualization.py
```

This will:
- Create synthetic or load real ETT data
- Generate forecasts using the extended model
- Create comprehensive spline visualizations showing:
  - Historical data vs forecasts vs true future values
  - Spline control points and basis functions
  - Accuracy metrics (MSE, MAE) per variable
  - Forecast uncertainty analysis
- Save visualizations to `test_spline_outputs/` and `spline_validation_outputs/`

### Generating Visualizations

```bash
# Generate spline forecasting visualizations (RECOMMENDED)
python test_spline_visualization.py

# Run full extended model validation with all visualizations
python validate_extended_model.py

# Legacy visualizations (original model)
python integrated_model_visualizations.py
python visualize_attention.py
python visualize_ett_data.py
python visualize_simple.py
```

**Generated Outputs**:
- `spline_validation_outputs/spline_forecasts.png` - Main forecasting results with accuracy
- `spline_validation_outputs/spline_analysis.png` - Detailed spline analysis
- `test_spline_outputs/` - Test visualization outputs
- `extended_model_validation_report.json` - Comprehensive validation metrics

**Note**: The extended model generates interpretable spline visualizations showing forecasts vs true data with accuracy metrics. See `README_SPLINE_VISUALIZATIONS.md` for detailed explanation of the output graphs.

## 📁 Project Structure

```
├── README.md                              # This file - Extended model documentation
├── README_SPLINE_VISUALIZATIONS.md       # Detailed explanation of output graphs
├── model.py                              # Original interpretable time encoder
├── extended_model.py                     # ⭐ Extended model with forecasting
├── data_utils.py                         # Data loading and preprocessing utilities
├── spline_visualization.py               # ⭐ Spline forecasting visualizations
├── validate_model.py                     # Original model validation
├── validate_extended_model.py            # ⭐ Extended model validation
├── test_spline_visualization.py          # ⭐ Spline visualization tests
├── evaluation_utils.py                   # Forecasting evaluation utilities
├── ETT-small/
│   └── ETTh1.csv                         # ETT dataset file
├── spline_validation_outputs/            # ⭐ Generated spline visualizations
│   ├── spline_forecasts.png             # Main forecasting results
│   └── spline_analysis.png              # Detailed spline analysis
├── test_spline_outputs/                  # Test visualization outputs
└── .kiro/specs/cross-attention-forecasting-extension/
    ├── requirements.md                   # Extended model requirements
    ├── design.md                        # Extended technical design
    └── tasks.md                         # Implementation tasks (completed)
```

**Key Files**:
- **`extended_model.py`** - Complete extended forecasting model implementation
- **`spline_visualization.py`** - Lightweight visualization functions for spline outputs
- **`validate_extended_model.py`** - Comprehensive validation with accuracy assessment
- **`test_spline_visualization.py`** - Easy-to-run test for generating forecast graphs

## 🔧 Extended Model Components

### ExtendedModelConfig
Configuration class with parameters for the complete forecasting pipeline:
```python
@dataclass
class ExtendedModelConfig:
    # Original parameters
    num_variables: int = 7          # Number of input variables (ETT: 7)
    embed_dim: int = 32             # Base embedding dimension
    hidden_dim: int = 64            # Univariate learner hidden size
    num_heads: int = 4              # Temporal attention heads
    dropout: float = 0.1            # Dropout probability
    max_len: int = 512              # Maximum sequence length
    
    # Cross-attention parameters
    cross_dim: int = 32             # Cross-attention embedding dimension
    cross_heads: int = 4            # Number of cross-attention heads
    cross_dropout: float = 0.1      # Cross-attention dropout
    
    # Temporal compression parameters
    compressed_dim: int = 64        # Compressed representation dimension
    compression_ratio: int = 4      # Temporal compression ratio (T -> T/4)
    compression_method: str = 'attention'  # Compression method
    
    # Spline forecasting parameters
    num_control_points: int = 8     # B-spline control points
    spline_degree: int = 3          # B-spline degree (cubic)
    forecast_horizon: int = 24      # Default forecast steps ahead
    spline_stability: bool = True   # Enable stability constraints
    forecast_horizons: list = [1, 12, 24, 48]  # Multiple horizons
```

### CrossVariableAttention
```python
# Models variable relationships: (B, M, T, embed_dim) -> (B, M, T, cross_dim) + cross_attn
cross_attention = CrossVariableAttention(
    embed_dim=32,
    cross_dim=32,
    num_heads=4,
    dropout=0.1
)
```

### TemporalEncoder
```python
# Compresses sequences: (B, M, T, cross_dim) -> (B, M, compressed_dim) + compression_attn
temporal_encoder = TemporalEncoder(
    input_dim=32,
    compressed_dim=64,
    compression_ratio=4
)
```

### SplineFunctionLearner
```python
# Generates forecasts: (B, M, compressed_dim) -> forecasts + spline_parameters
spline_learner = SplineFunctionLearner(
    input_dim=64,
    num_control_points=8,
    spline_degree=3,
    forecast_horizon=24,
    stability_constraints=True
)
```

## 📈 Performance

Based on extended model validation results:
- **Parameters**: ~76K trainable parameters (extended model)
- **Throughput**: ~475 samples/second (on CPU)
- **Memory**: ~0.29 MB model size, efficient processing up to 512 time steps
- **Scalability**: 
  - Linear complexity in sequence length for univariate learners
  - Quadratic complexity in number of variables for cross-attention (manageable for ETT's 7 variables)
  - Compressed temporal processing reduces forecasting complexity

### Forecasting Accuracy (ETT Dataset)
- **1-step ahead**: MSE: 1.058, MAE: 0.810
- **12-step ahead**: MSE: 9.141, MAE: 2.236  
- **24-step ahead**: MSE: 4.255, MAE: 1.375
- **48-step ahead**: MSE: 3.133, MAE: 1.283

### Validation Results
- **Pipeline Tests**: 100% pass rate (4/4 sequence lengths)
- **Gradient Flow**: ✅ All 76K parameters receive gradients
- **Interpretability**: ✅ All attention weights and spline parameters generated correctly
- **Architecture**: ✅ All components integrate seamlessly

## 🎨 Extended Interpretability Features

### 1. Variable Independence & Relationships
- Each variable processed through separate univariate learners (no parameter sharing)
- **Cross-variable attention heatmaps** show which variables influence each other
- Clear separation of variable-specific vs. inter-variable patterns

### 2. Multi-Level Attention Visualization
- **Temporal attention**: Shows time dependencies within each variable
- **Cross-variable attention**: Reveals variable-to-variable relationships (M×M heatmaps)
- **Compression attention**: Identifies which time steps are most important for forecasting
- All attention weights are interpretable and visualizable

### 3. Spline-Based Forecast Interpretation
- **B-spline control points**: Show the mathematical shape of learned forecast functions
- **Basis functions**: Visualize the mathematical foundation of predictions
- **Smooth extrapolation**: Splines naturally extend beyond training data
- **Control point analysis**: Understand how the model constructs forecasts

### 4. Comprehensive Analysis Outputs
- **Forecast vs. True Data**: Direct accuracy visualization with MSE/MAE metrics
- **Trend Analysis**: Compare historical vs. forecast trends
- **Uncertainty Quantification**: Forecast variance across prediction horizon
- **Variable Correlation**: Understand relationships in both input data and control points

### 5. Mathematical Interpretability
- **Spline Parameters**: Complete mathematical description of forecast functions
- **Knot Vectors**: B-spline mathematical foundation
- **Control Point Statistics**: Statistical analysis of learned forecast shapes
- **Smoothness Analysis**: Quantitative measures of forecast smoothness

## 🧪 Extended Testing and Validation

The project includes comprehensive testing for the extended forecasting model:

### Extended Model Validation (`validate_extended_model.py`)
- **Architecture Validation**: All 5 components (univariate → temporal → cross → compression → spline)
- **End-to-end Pipeline**: Complete forecasting pipeline with ETT data
- **Gradient Computation**: Verification through all 76K parameters
- **Forecasting Accuracy**: Multiple horizons (1, 12, 24, 48 steps) with accuracy metrics
- **Interpretability Artifacts**: All attention weights and spline parameters
- **Performance Analysis**: Throughput and memory usage benchmarking
- **Spline Visualization**: Automatic generation with accuracy assessment

### Spline Visualization Testing (`test_spline_visualization.py`)
- **Forecast Accuracy**: Visual comparison of predictions vs. true future values
- **Spline Analysis**: Control points, basis functions, and mathematical properties
- **Multi-Variable Support**: All 7 ETT variables with individual accuracy metrics
- **Synthetic Data Fallback**: Creates test data if ETT dataset unavailable
- **Comprehensive Metrics**: MSE, MAE, trend analysis, and uncertainty quantification

### Extended Test Coverage
- ✅ Multiple sequence lengths (24, 96, 168, 336 time steps)
- ✅ Different batch sizes (2, 4, 8, 16, 32 samples)
- ✅ Cross-attention mechanism validation (M×M attention matrices)
- ✅ Temporal compression with configurable ratios
- ✅ B-spline mathematical properties (continuity, smoothness, extrapolation)
- ✅ Multi-horizon forecasting (1 to 48 steps ahead)
- ✅ Gradient flow through all extended components
- ✅ Interpretability artifact generation and validation
- ✅ Performance scaling analysis
- ✅ Forecast accuracy assessment with real validation data

## 📊 Extended Visualization Outputs

The extended model generates comprehensive spline-based forecasting visualizations:

### Main Forecasting Visualization (`spline_forecasts.png`)
1. **Historical vs. Forecast vs. True Data**: 
   - Blue line: Historical time series data
   - Red line with markers: Spline-generated forecasts
   - Green dashed line: True future values (for accuracy assessment)
   - Purple squares: B-spline control points
2. **Accuracy Metrics**: MSE and MAE displayed for each variable
3. **Trend Analysis**: Historical vs. forecast trend comparison
4. **Forecast Period Highlighting**: Visual separation of history from predictions

### Detailed Spline Analysis (`spline_analysis.png`)
1. **Control Points by Variable**: Shows learned spline parameters for each ETT variable
2. **B-Spline Basis Functions**: Mathematical foundation of the spline curves
3. **Control Point Statistics**: Mean and variance analysis across variables
4. **Forecast Uncertainty**: Variance analysis across prediction horizon
5. **Variable Correlation Matrix**: Relationships between control points
6. **Smoothness Analysis**: Quantitative smoothness measures for each forecast

### Key Insights from Visualizations
- **Spline Interpretability**: Clear mathematical representation of forecasts
- **Variable-Specific Patterns**: Each ETT variable shows unique forecast characteristics
- **Accuracy Assessment**: Direct visual comparison with quantitative metrics
- **Temporal Patterns**: Compression attention shows which historical periods matter most
- **Cross-Variable Dependencies**: Attention heatmaps reveal variable relationships

**See `README_SPLINE_VISUALIZATIONS.md` for detailed explanation of all graphs and how to interpret them.**

## 🔬 Research Applications

This extended model is particularly suitable for:

### Forecasting Applications
- **Multivariate Time Series Forecasting**: Multi-horizon predictions with interpretable splines
- **Energy Systems**: Electricity load forecasting (ETT dataset domain)
- **Financial Markets**: Multi-asset price prediction with cross-asset relationships
- **Weather Forecasting**: Multi-variable meteorological predictions
- **Industrial IoT**: Equipment monitoring with multi-sensor forecasting

### Interpretability Applications
- **Regulatory Compliance**: Financial/medical domains requiring explainable predictions
- **Scientific Research**: Understanding complex system dynamics through attention patterns
- **Decision Support**: Business forecasting where reasoning transparency is crucial
- **Anomaly Detection**: Identifying unusual patterns through attention and spline analysis
- **Causal Analysis**: Cross-variable attention reveals potential causal relationships

### Technical Advantages
- **Mathematical Interpretability**: B-splines provide clear mathematical forecast representation
- **Multi-Scale Analysis**: From individual variables to cross-variable relationships
- **Uncertainty Quantification**: Spline parameters enable confidence interval estimation
- **Extrapolation Capability**: Splines naturally extend beyond training data
- **Computational Efficiency**: Temporal compression reduces forecasting complexity

## 🛠️ Extended Model Customization

### Adjusting Forecasting Parameters
```python
# Different forecast horizons
config = ExtendedModelConfig(
    forecast_horizon=48,           # Predict 48 steps ahead
    forecast_horizons=[1, 6, 12, 24, 48]  # Multiple horizons
)

# More complex splines
config = ExtendedModelConfig(
    num_control_points=12,         # More control points for complex curves
    spline_degree=5,               # Higher degree splines (quintic)
    spline_stability=True          # Enable stability constraints
)
```

### Scaling for Different Datasets
```python
# For datasets with many variables (cross-attention scales O(M²))
config = ExtendedModelConfig(
    num_variables=20,              # Larger multivariate dataset
    cross_heads=8,                 # More attention heads for complex relationships
    compressed_dim=128             # Larger compressed representation
)

# For longer sequences
config = ExtendedModelConfig(
    max_len=1024,                  # Longer sequences
    compression_ratio=8,           # Higher compression for efficiency
    compressed_dim=256             # Larger compressed representation
)
```

### Custom Spline Configuration
```python
# Different spline types and constraints
spline_learner = SplineFunctionLearner(
    input_dim=64,
    num_control_points=10,
    spline_degree=3,
    forecast_horizon=36,
    stability_constraints=True     # Enable control point constraints
)

# Custom basis function evaluation
basis_functions = spline_learner._generate_basis_functions()
```

### Custom Visualization
```python
from spline_visualization import plot_spline_outputs

# Custom visualization with specific variables
figures = plot_spline_outputs(
    model_output=output,
    input_data=historical_data,
    true_future=validation_data,
    variable_names=['Temperature', 'Pressure', 'Flow', 'Voltage'],
    sample_idx=0,
    save_path="custom_forecast_analysis.png"
)
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