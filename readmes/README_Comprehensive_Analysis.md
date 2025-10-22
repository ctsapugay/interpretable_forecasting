# Comprehensive UnivariateFunctionLearner Analysis

## Overview
This README describes the comprehensive analysis visualization (`univariate_function_learner_analysis.png`) that provides an in-depth look at the UnivariateFunctionLearner's behavior across multiple dimensions and variables.

## Visualization Panels (12 Total)

### Row 1: Data and Basic Transformations

#### Panel 1: Original ETT Dataset Variables
- **Content**: All 7 ETT variables plotted over time
- **Variables**: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
- **Purpose**: Shows the multivariate nature of the input data
- **Insight**: Each variable has different scales and patterns

#### Panel 2: Input - HUFL Variable Focus
- **Content**: Detailed view of the HUFL variable used for analysis
- **Purpose**: Shows the specific input being transformed
- **Insight**: Single variable time series with clear temporal patterns

#### Panel 3: HUFL → 8D Embeddings Heatmap
- **Content**: Heatmap showing transformation from scalar to 8-dimensional embeddings
- **Color coding**: Viridis colormap showing embedding values
- **Purpose**: Visualizes the dimensionality expansion
- **Insight**: Each time step now has 8 different learned features

### Row 2: Embedding Analysis

#### Panel 4: Embedding Dimensions Over Time
- **Content**: Line plots of first 4 embedding dimensions
- **Purpose**: Shows how different dimensions capture different temporal patterns
- **Insight**: Each dimension specializes in different aspects of the input

#### Panel 5: MLP Layer Activation Distributions
- **Content**: Histograms of hidden layer and output layer activations
- **Colors**: Orange (hidden layer), Green (output layer)
- **Purpose**: Shows internal neural network behavior
- **Insight**: ReLU activation creates sparse, positive representations

#### Panel 6: Input vs Embedding Magnitude
- **Content**: Scatter plot of input values vs embedding L2 norms
- **Color coding**: Time progression (plasma colormap)
- **Purpose**: Shows input-output relationship strength
- **Insight**: Reveals how input magnitude relates to embedding strength

### Row 3: Multi-Variable and Space Analysis

#### Panel 7: Embedding Magnitudes - Multiple Variables
- **Content**: Embedding magnitudes for first 4 ETT variables over time
- **Purpose**: Compares how different variables are embedded
- **Insight**: Different variables produce different embedding patterns

#### Panel 8: 2D Embedding Space Trajectory
- **Content**: 2D projection of embedding space (first 2 dimensions)
- **Color coding**: Time progression
- **Purpose**: Shows trajectory through learned embedding space
- **Insight**: Temporal evolution creates paths in embedding space

#### Panel 9: Architecture Diagram
- **Content**: Text-based description of the neural network architecture
- **Purpose**: Documents the model structure
- **Details**: Input/output shapes and layer specifications

### Row 4: Training and Parameter Analysis

#### Panel 10: Gradient Flow
- **Content**: Gradients with respect to input over time
- **Purpose**: Shows how loss propagates back to inputs
- **Insight**: Reveals which time steps are most important for learning

#### Panel 11: Parameter Statistics
- **Content**: Bar chart of parameter statistics (min, mean, max)
- **Parameters**: Weights and biases for both linear layers
- **Purpose**: Shows learned parameter distributions
- **Insight**: Parameter magnitudes indicate learning patterns

#### Panel 12: Summary Statistics
- **Content**: Text summary of model and data characteristics
- **Includes**: Model parameters, data ranges, configuration details
- **Purpose**: Provides quantitative overview

## Technical Details

### Model Configuration
- **Input**: (B, T, 1) - Batch, Time, Scalar
- **Hidden Layer**: 16 units with ReLU activation
- **Output**: (B, T, 8) - 8-dimensional embeddings
- **Parameters**: ~272 total parameters

### Data Processing
- **Dataset**: ETT-h1 (Electricity Transforming Temperature)
- **Variables**: 7 multivariate time series
- **Time Steps**: 50 for visualization clarity
- **Preprocessing**: Direct use of raw values

## Key Insights

1. **Dimensionality Expansion**: Scalar inputs become rich vector representations
2. **Specialization**: Each embedding dimension captures different patterns
3. **Independence**: Each variable processed separately (no parameter sharing)
4. **Temporal Consistency**: Embeddings maintain temporal relationships
5. **Gradient Flow**: Model learns from all time steps effectively

## Files Generated
- `univariate_function_learner_analysis.png` - The comprehensive visualization
- `model.py` - Contains the visualization code in `visualize_univariate_function_learner()`

## How to Reproduce
```bash
source venv/bin/activate
python model.py
```

## Use Cases
- **Research**: Understanding TKAN-style function learning
- **Debugging**: Analyzing model behavior and parameter learning
- **Education**: Teaching neural network embedding concepts
- **Development**: Validating model implementation correctness