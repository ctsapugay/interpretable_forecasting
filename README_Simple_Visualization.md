# Simple UnivariateFunctionLearner Visualization

## Overview
This README describes the simple visualization (`simple_univariate_visualization.png`) that demonstrates the core functionality of the UnivariateFunctionLearner in an easy-to-understand format.

## What the Visualization Shows

### Panel 1: Input - Original Time Series
- **What it is**: The raw HUFL (High UseFul Load) variable from the ETT dataset
- **Shape**: (30,) - A sequence of 30 scalar values over time
- **Purpose**: Shows the starting point - simple numerical values that change over time
- **Key insight**: Each time step contains just one number

### Panel 2: Output - Embedding Matrix
- **What it is**: A heatmap showing how each scalar input is transformed into a 4-dimensional vector
- **Shape**: (4, 30) - 4 embedding dimensions across 30 time steps
- **Color coding**: Different colors represent different embedding values
- **Key insight**: Each single input value now becomes a rich 4-dimensional representation

### Panel 3: Embedding Dimensions Over Time
- **What it is**: Line plots showing how each of the 4 embedding dimensions evolves over time
- **Colors**: Red, Green, Blue, Orange for dimensions 0, 1, 2, 3 respectively
- **Purpose**: Demonstrates that each dimension captures different patterns from the input
- **Key insight**: The neural network learns to split information across multiple dimensions

### Panel 4: Input-Output Relationship
- **What it is**: A scatter plot showing the relationship between input values and embedding magnitudes
- **X-axis**: Original HUFL input values
- **Y-axis**: L2 norm (magnitude) of the resulting embedding vectors
- **Color**: Time progression (darker = later in time)
- **Key insight**: Shows how the neural network maps input values to embedding space

## The Transformation Process

1. **Input**: Scalar time series values (one number per time step)
2. **Processing**: 2-layer MLP with ReLU activation
   - Layer 1: Linear(1 → 8) + ReLU
   - Layer 2: Linear(8 → 4)
3. **Output**: 4-dimensional embedding vectors (four numbers per time step)

## Why This Matters

- **Expressiveness**: Vector embeddings can capture more complex patterns than scalar values
- **Specialization**: Each embedding dimension can focus on different aspects of the input
- **Preparation**: These rich representations are ready for the next stage (temporal attention)
- **Independence**: Each variable gets its own dedicated function learner

## Files Generated
- `simple_univariate_visualization.png` - The main visualization
- `visualize_simple.py` - Script to generate the visualization

## How to Reproduce
```bash
source venv/bin/activate
python visualize_simple.py
```