# **Design Document: Interpretable Time Series Forecasting Model**

## **Modules 1 & 2: Univariate TKAN Learners and Temporal Attention**

### **1.0 Overview**

#### **1.1 Project Goal**

The primary objective is to develop a deep learning model for multivariate time series forecasting that is highly interpretable. The model must not only provide accurate predictions but also offer clear insights into how it arrives at those predictions, detailing both the influence of individual variables and their temporal dependencies.

#### **1.2 Pipeline Architecture**

The proposed end-to-end pipeline consists of five distinct, interpretable stages:

1. **Univariate Function Learners (TKAN-Style):** Learn an independent, interpretable function for each input variable.  
2. **Inverted Temporal Attention:** Model temporal patterns *within* each variable using self-attention.  
3. **Cross-Variable Attention:** Model relationships *between* variables.  
4. **Encoder:** Create a compressed, latent representation of the learned features.  
5. **Spline-Based Decoder (TimeView-Style):** Generate the final forecast from the latent representation using a flexible spline function.

## **This document provides a detailed design for implementing and validating Modules 1 and 2.**

### **2.0 Data Flow and Tensor Shapes**

We will adopt a batch-first convention for all tensors.

* **Model Input (X):** A tensor representing a batch of multivariate time series history.  
  * **Shape:** (B, T\_in, M)  
* **Module 1 Output (U\_stack):** A stack of tensors, where each tensor is the learned embedding for a single variable over time.  
  * **Shape:** (B, M, T\_in, d\_var)  
* **Module 2 Output (H\_stack):** The output of the temporal attention mechanism, representing temporally-aware features for each variable.  
  * **Shape:** (B, M, T\_in, d\_var)

**Notation:**

* B: Batch size  
* T\_in: Length of the input time series window (e.g., 168 hours)  
* M: Number of input variables (features)  
* d\_var: The dimensionality of the learned embedding for each variable.

### **3.0 Module 1: Univariate TKAN-Style Function Learners**

#### **3.1 Objective**

To learn a continuous, univariate function for each input variable independently. This function, h\_i(x), maps the raw scalar value of variable i at a given time step, x\_i(t), to a higher-dimensional vector embedding, u\_{i,t}. This isolation is critical for interpretability, as it allows us to analyze the behavior of each variable in a vacuum before considering interactions.

#### **3.2 Design & Implementation**

For the initial prototype, we will implement this function learner as a small Multi-Layer Perceptron (MLP) for each variable. This approach is simple, efficient, and allows for rapid development.

* **Architecture:** A 2-layer MLP with a ReLU activation.  
* **Application:** The MLP will be applied in a time-distributed manner to every time step of its corresponding input variable.  
* **Future Enhancement:** To align more closely with the Kolmogorov-Arnold Network (KAN) theory, this MLP can be replaced by a module that learns coefficients for a set of B-spline basis functions. This would offer a more direct and mathematically grounded form of interpretability.

#### **3.3 API Specification (PyTorch)**

import torch  
import torch.nn as nn

class UnivariateFunctionLearner(nn.Module):  
    """  
    Learns a univariate function h(x) that maps a scalar input  
    to a vector embedding. Applied time-distributively.  
    """  
    def \_\_init\_\_(self, in\_features=1, out\_features=32, hidden\_features=64):  
        super().\_\_init\_\_()  
        self.net \= nn.Sequential(  
            nn.Linear(in\_features, hidden\_features),  
            nn.ReLU(),  
            nn.Linear(hidden\_features, out\_features)  
        )

    def forward(self, x: torch.Tensor) \-\> torch.Tensor:  
        """  
        Args:  
            x (torch.Tensor): Input tensor for a single variable.  
                              Shape: (B, T\_in, 1\)

        Returns:  
            torch.Tensor: Learned vector embedding.  
                          Shape: (B, T\_in, out\_features)  
        """  
        \# Reshape for time-distributed application  
        b, t, \_ \= x.shape  
        x\_flat \= x.view(b \* t, \-1)  
        u\_flat \= self.net(x\_flat)  
        return u\_flat.view(b, t, \-1)

#### **3.4 Interpretability Hook**

The learned function h\_i(x) can be visualized by passing a range of plausible scalar values for variable i through the trained module and plotting the resulting output embedding dimensions. This plot reveals how the model understands and represents the variable's magnitude.

### **4.0 Module 2: Inverted Temporal Self-Attention**

#### **4.1 Objective**

To encode the temporal dependencies within each variable's embedding sequence U\_i. The attention mechanism will learn which past time steps are most influential for determining the representation at any given time step, providing a clear view of temporal patterns like seasonality and trends for that specific variable.

#### **4.2 Design & Implementation**

We will use a standard Multi-Head Self-Attention (MHSA) block with pre-layer normalization and residual connections for stability. A single, shared attention block will be used across all variables to promote parameter efficiency, though this could be changed to per-variable blocks later if needed.

* **Architecture:** Pre-LayerNorm \-\> MHSA \-\> Dropout \-\> Residual Connection.  
* **Positional Encoding:** A learnable embedding will be added to the input to give the model awareness of the absolute position of each time step.  
* **Attention Weights:** The module will be configured to return the attention weight matrices for visualization.

#### **4.3 API Specification (PyTorch)**

class TemporalSelfAttention(nn.Module):  
    """  
    Applies self-attention along the time dimension of a single  
    variable's embedding sequence.  
    """  
    def \_\_init\_\_(self, embed\_dim=32, num\_heads=4, dropout=0.1, max\_len=512):  
        super().\_\_init\_\_()  
        self.positional\_encoding \= nn.Embedding(max\_len, embed\_dim)  
        self.norm \= nn.LayerNorm(embed\_dim)  
        self.attention \= nn.MultiheadAttention(  
            embed\_dim, num\_heads, dropout=dropout, batch\_first=True  
        )  
        self.dropout \= nn.Dropout(dropout)

    def forward(self, u: torch.Tensor) \-\> tuple\[torch.Tensor, torch.Tensor\]:  
        """  
        Args:  
            u (torch.Tensor): A single variable's embedding sequence.  
                              Shape: (B, T\_in, embed\_dim)

        Returns:  
            Tuple\[torch.Tensor, torch.Tensor\]:  
            \- h: Temporally-aware representation. Shape: (B, T\_in, embed\_dim)  
            \- attn\_weights: Attention weights. Shape: (B, num\_heads, T\_in, T\_in)  
        """  
        b, t, d \= u.shape  
        positions \= torch.arange(t, device=u.device).unsqueeze(0).expand(b, \-1)  
        pos\_emb \= self.positional\_encoding(positions)

        u\_pos \= u \+ pos\_emb  
        u\_norm \= self.norm(u\_pos)

        \# The MultiheadAttention layer returns attended output and weights  
        attn\_output, attn\_weights \= self.attention(u\_norm, u\_norm, u\_norm)

        \# Residual connection  
        h \= u \+ self.dropout(attn\_output)  
        return h, attn\_weights

#### **4.4 Interpretability Hook**

The returned attn\_weights tensor provides a (T\_in x T\_in) heatmap for each head and each sample in the batch. By visualizing this map, we can see which historical time points (columns) the model focuses on when constructing the representation for a given time point (rows).

### **5.0 Integrated Model and Prototyping Strategy**

#### **5.1 Full Model (Modules 1 & 2\)**

The two modules will be combined into a single parent model that iterates through each input variable, applies the function learner, and then the temporal attention block.

class InterpretableTimeEncoder(nn.Module):  
    def \_\_init\_\_(self, num\_variables, embed\_dim=32, num\_heads=4, max\_len=512):  
        super().\_\_init\_\_()  
        self.num\_variables \= num\_variables  
        self.univariate\_learners \= nn.ModuleList(  
            \[UnivariateFunctionLearner(out\_features=embed\_dim) for \_ in range(num\_variables)\]  
        )  
        self.temporal\_attention \= TemporalSelfAttention(  
            embed\_dim=embed\_dim, num\_heads=num\_heads, max\_len=max\_len  
        )

    def forward(self, x: torch.Tensor):  
        \# x shape: (B, T\_in, M)  
        all\_h \= \[\]  
        all\_attn \= \[\]

        for i in range(self.num\_variables):  
            x\_i \= x\[:, :, i:i+1\]  \# Get i-th variable  
            u\_i \= self.univariate\_learners\[i\](x\_i)  
            h\_i, attn\_i \= self.temporal\_attention(u\_i)  
            all\_h.append(h\_i)  
            all\_attn.append(attn\_i)

        \# Stack results for downstream modules  
        h\_stack \= torch.stack(all\_h, dim=1)  \# Shape: (B, M, T\_in, d\_var)  
        attn\_stack \= torch.stack(all\_attn, dim=1) \# Shape: (B, M, num\_heads, T\_in, T\_in)

        return h\_stack, attn\_stack

#### **5.2 Validation on the ETT Dataset**

For validation, we will use the Electricity Transformer Temperature (ETTh1) dataset.

* **Proxy Task:** To test these initial modules, we will implement a simple one-step-ahead forecasting head. A linear layer will be attached to the final time step of each variable's output (h\_i\[:, \-1, :\]) to predict the next value of that variable.  
* **Loss:** Mean Squared Error (MSE) will be calculated per variable and summed.  
* **Goal:** The primary goal is not to achieve state-of-the-art accuracy at this stage, but to verify that the modules can learn meaningful representations and that the interpretability hooks produce coherent visualizations.