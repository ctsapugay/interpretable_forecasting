# InterpretableTimeEncoder Visualization Guide

This document explains the graphs and visualizations generated for the InterpretableTimeEncoder model, which combines univariate function learners with temporal self-attention for interpretable time series forecasting.

## Overview

The visualizations demonstrate how the integrated model processes multivariate time series data through two main stages:
1. **Univariate Function Learners**: Transform each variable independently from scalar values to embeddings
2. **Temporal Self-Attention**: Capture temporal dependencies within each variable's embedding sequence

## Graph Descriptions

### Simple Integration Process (`simple_integration_process.png`)

This visualization shows the three-step integration process with a clear, focused view:

**Graph 1: Input Time Series**
- Shows the original normalized time series for 3 key variables (HUFL, MUFL, OT)
- Demonstrates the raw multivariate input data that feeds into the model
- Variables are normalized for visual comparison

**Graph 2: After Univariate Function Learners**
- Shows embedding magnitudes after each variable passes through its dedicated MLP
- Each scalar time series has been transformed into a vector embedding sequence
- Different variables show different embedding magnitude patterns based on their unique MLPs

**Graph 3: After Temporal Self-Attention**
- Shows final embedding magnitudes after temporal attention processing
- Attention mechanism has refined the embeddings to capture temporal dependencies
- Notice how the patterns become more structured and temporally coherent

### Comprehensive Analysis (`integrated_model_analysis.png`)

This detailed visualization contains 16 subplots showing various aspects of the integrated model:

#### Data and Architecture (Plots 1-2)
1. **Original ETT Dataset**: All 7 variables (HUFL, HULL, MUFL, MULL, LUFL, LULL, OT) over time
2. **Architecture Flow**: Text diagram showing the model's processing pipeline

#### Variable Processing Pipeline (Plots 3-5)
3-5. **Individual Variable Processing**: Shows how HUFL, HULL, and MUFL are transformed through:
   - Original signal (blue line)
   - After MLP transformation (green dashed)
   - After attention mechanism (red dotted)

#### Embedding Analysis (Plots 6-8)
6. **Embedding Magnitudes Heatmap**: Shows embedding strength across all variables and time
7. **Attention Pattern (HUFL)**: Visualization of how HUFL attends to different time positions
8. **Attention Pattern (OT)**: Temperature variable's attention pattern (typically different from others)

#### Cross-Variable Analysis (Plots 9-10)
9. **Cross-Variable Similarity**: Cosine similarity between different variables' embeddings
10. **Attention Focus**: Entropy of attention weights (lower = more focused attention)

#### Detailed Component Analysis (Plots 11-14)
11. **Embedding Dimensions**: How different embedding dimensions capture patterns for HUFL
12. **Parameter Distribution**: Histogram of all model parameters
13. **Information Content**: Variance analysis across processing stages
14. **Attention Head Specialization**: How different attention heads focus on recent vs. distant past

#### Model Summary (Plots 15-16)
15. **Model Statistics**: Parameter counts, shapes, and architecture details
16. **Integration Benefits**: Comparison of temporal consistency before and after attention

## Key Insights from the Visualizations

### 1. Independent Variable Processing
- Each variable has its own dedicated univariate function learner (no parameter sharing)
- Different variables show distinct embedding patterns reflecting their unique characteristics
- The MLP transformation preserves variable-specific information while creating rich representations

### 2. Temporal Attention Effects
- Attention patterns vary significantly between variables (compare HUFL vs. OT attention maps)
- Some variables focus more on recent history, others on longer-term patterns
- Attention entropy shows which time points require more focused vs. distributed attention

### 3. Integration Benefits
- Temporal consistency improves after attention processing (Plot 16)
- Cross-variable similarity reveals which variables have related temporal patterns
- Different attention heads specialize in different temporal ranges

### 4. Model Interpretability
- Attention weights provide direct insight into which time steps are important
- Embedding magnitudes show how much "information" each variable carries at each time
- Parameter distribution shows the model learns diverse representations

## How to Generate These Visualizations

Run the visualization script:
```bash
python integrated_model_visualizations.py
```

This will generate:
- `simple_integration_process.png`: Clear 3-step process visualization
- `integrated_model_analysis.png`: Comprehensive 16-plot analysis

## Understanding the Integration

The key innovation of the InterpretableTimeEncoder is how it combines:

1. **Specialized Processing**: Each variable gets its own function learner, preserving variable-specific patterns
2. **Shared Temporal Understanding**: All variables use the same attention mechanism, enabling consistent temporal modeling
3. **Interpretable Outputs**: Both embedding magnitudes and attention weights provide clear insights into model behavior

The visualizations demonstrate that this integration successfully:
- Maintains variable independence while enabling temporal coherence
- Provides interpretable attention patterns that vary meaningfully across variables
- Improves temporal consistency through the attention mechanism
- Creates rich, informative embeddings suitable for downstream forecasting tasks