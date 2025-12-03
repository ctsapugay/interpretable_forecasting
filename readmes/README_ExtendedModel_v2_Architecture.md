## Extended Interpretable Forecasting Model v2: Architecture and Interpretation

This document describes the architecture of the **Extended Interpretable Forecasting Model v2**, focusing on:

- How the model processes multivariate time series from raw inputs to forecasts.
- How the new **multi‑query temporal compression** and **flexible spline head** work.
- How the **optional residual head** fits into the picture.
- How interpretability is preserved, and which pieces are optional.

The core design goal remains: **rich, multi‑scale temporal forecasting with explicit, mathematically interpretable components** (univariate function learners, attention maps, spline control points, basis functions).

---

## 1. High‑Level Pipeline

Given an input window \(X \in \mathbb{R}^{B \times T \times M}\) (batch size `B`, sequence length `T`, `M` variables), the v2 model follows this pipeline:

1. **Univariate Function Learners** (`UnivariateFunctionLearner`)
2. **Temporal Self‑Attention** (`TemporalSelfAttention`)
3. **Cross‑Variable Attention** (`CrossVariableAttention`)
4. **Multi‑Query Temporal Compression** (`TemporalEncoder`)
5. **Spline Forecasting Head** (`SplineFunctionLearner`)
6. **(Optional) Residual Correction Head**

Formally:

\[
X \xrightarrow{\text{univariate}} U \xrightarrow{\text{temporal attn}} H
\xrightarrow{\text{cross attn}} C \xrightarrow{\text{compression}} Z
\xrightarrow{\text{spline + optional residual}} \hat{Y}.
\]

All intermediate tensors (`U`, `H`, `C`, `Z`, attention maps, and spline parameters) are **exposed in `interpretability`** for inspection and visualization.

---

## 2. Univariate Function Learners (Per‑Variable TKAN Blocks)

**Module:** `UnivariateFunctionLearner` (see `main model/model.py`).

For each variable \(m \in \{1, \dots, M\}\):

- **Input**: scalar time series \(x^{(m)} \in \mathbb{R}^{B \times T \times 1}\).
- **Architecture**:
  - 2‑layer MLP with ReLU:
    \[
    \text{MLP}(x) = \text{ReLU}(x W_1 + b_1) W_2 + b_2.
    \]
  - Applied in a **time‑distributed** manner (same small network across all time steps).
- **Output**: embeddings \(u^{(m)} \in \mathbb{R}^{B \times T \times d_{\text{embed}}}\).

Key properties:

- **Per‑variable independence**: each variable has its own `UnivariateFunctionLearner` (no parameter sharing), preserving variable‑specific nonlinear transformations.
- **Interpretation**:
  - Each embedding dimension can be visualized over time.
  - The `visualize_univariate_function_learner()` utility (in `model.py`) shows raw vs. transformed trajectories, activation distributions, and gradient flow.

---

## 3. Temporal Self‑Attention

**Module:** `TemporalSelfAttention` (in `main model/model.py`).

For each variable \(m\):

- **Input**: \(u^{(m)} \in \mathbb{R}^{B \times T \times d_{\text{embed}}}\).
- Add learnable positional encodings; apply multi‑head self‑attention:

\[
h^{(m)}, A^{(m)} = \text{MHA}\big(\text{LN}(u^{(m)} + p)\big),
\]

where:

- \(h^{(m)} \in \mathbb{R}^{B \times T \times d_{\text{embed}}}\) are temporally contextualized embeddings.
- \(A^{(m)} \in \mathbb{R}^{B \times H \times T \times T}\) are per‑head attention matrices (query vs. key time positions).

Properties:

- Learns which **past time steps** matter for each current step and variable.
- `interpretability['temporal_attention']` stores \(A^{(m)}\) for every variable and head.

Stacking over variables yields:

- \(H \in \mathbb{R}^{B \times M \times T \times d_{\text{embed}}}\).
- `temporal_attention ∈ ℝ^{B × M × H × T × T}`.

---

## 4. Cross‑Variable Attention

**Module:** `CrossVariableAttention` (in `main model/extended_model.py`).

- **Input**: \(H \in \mathbb{R}^{B \times M \times T \times d_{\text{embed}}}\).
- For each time step \(t\), treat the `M` variables as a “sequence” and run multi‑head attention across variables:
  - Reshape to `(B*T, M, embed_dim)`.
  - Apply multi‑head attention over the `M` dimension.

**Output**:

- Cross‑attended embeddings \(C \in \mathbb{R}^{B \times M \times T \times d_{\text{cross}}}\).
- Cross‑variable attention maps:
  \[
  A_{\text{cross}} \in \mathbb{R}^{B \times H_{\text{cross}} \times M \times M},
  \]
  averaged over time.

Interpretation:

- Each \(A_{\text{cross}}[b, h]\) is an \(M \times M\) matrix; entry \((i, j)\) indicates how much variable \(i\) attends to variable \(j\).
- These matrices are available as `interpretability['cross_attention']` and are visualized as variable‑to‑variable heatmaps.

---

## 5. Multi‑Query Temporal Compression

**Module:** `TemporalEncoder` (in `main model/extended_model.py`). 

In v2 we introduce **multi‑query compression**:

- Each variable has \(Q = \text{num\_compression\_queries}\) learnable queries.
- Each query can focus on different temporal patterns (e.g., recent vs long‑term history).

Given cross‑attended embeddings \(C \in \mathbb{R}^{B \times M \times T \times D}\):

1. Project (if needed) to `compressed_dim`.
2. Expand compression queries from `(1, Q, 1, D)` to `(B, M, Q, D)`.
3. Compute attention scores and weights:
   - `attention_scores ∈ ℝ^{B × M × Q × T}`.
   - `compression_attention ∈ ℝ^{B × M × Q × T}` via softmax.
4. Compute pooled representations:
   \[
   Z = \text{compression\_attn} \cdot X \in \mathbb{R}^{B \times M \times Q \times d_c}.
   \]
5. Flatten queries into the feature dimension and apply LayerNorm + MLP:
   - `compressed_repr ∈ ℝ^{B × M × (Q·compressed_dim)}`.

Interpretation:

- `interpretability['compression_attention']` contains attention maps `(B, M, Q, T)`.
- For each variable and query we can see **which time steps** contribute most to the compressed representation that drives forecasting.

---

## 6. Flexible Spline Forecasting Head

**Module:** `SplineFunctionLearner` (in `main model/extended_model.py`).

### 6.1 Inputs and Outputs

- **Input**: `compressed_repr ∈ ℝ^{B × M × D_z}`, where `D_z = compressed_dim * num_compression_queries`.
- **Predicts**:
  - Spline **control points** `control_points ∈ ℝ^{B × M × K}` (`K = num_control_points`).
  - Uses B‑spline basis functions `basis_functions ∈ ℝ^{H × K}` over \(t ∈ [0, 1]\) to produce:
    \[
    \text{forecasts} = \text{control\_points} \cdot \text{basis\_functions}^\top \in \mathbb{R}^{B \times M \times H},
    \]
    where `H = forecast_horizon`.
  - Also exposes `knot_vector` and various spline statistics.

### 6.2 Basis over [0, 1]

- Knot vector: open, uniform B‑spline over `[0, 1]` with endpoint multiplicity.
- Evaluation points: `t_eval = linspace(0.0, 1.0, forecast_horizon)`.
- Consequences:
  - **All control points** influence some part of the forecast horizon.
  - Gradients flow into every control point (no “dead” front segment).

### 6.3 Stability and Smoothing

- Control points are clamped to a reasonable numeric range (e.g., `[-10, 10]`) to avoid extreme values.
- Optional smoothing controlled by `spline_smooth_alpha ∈ [0, 1]`:
  - Compute a smoothed version via a local averaging kernel.
  - Interpolate:
    \[
    c_{\text{final}} = (1 - \alpha) c_{\text{raw}} + \alpha c_{\text{smoothed}}.
    \]
  - `α = 1.0` reproduces the original, very smooth behavior.
  - Smaller `α` (e.g., 0.2–0.4) allows sharper bends while still discouraging noise.

Interpretation:

- Control‑point plots now show **meaningful curvature across all points**.
- The basis and knot vector give a fully specified mathematical description of the forecast function for each variable and sample.

---

## 7. Optional Residual Correction Head

**Module:** integrated in `InterpretableForecastingModel`.

### 7.1 Purpose

The residual head is an **optional** component used when we want:

- The spline to model the main, smooth forecast shape, and
- A separate, small neural correction to handle high‑frequency residual structure or systematic biases.

### 7.2 Architecture

- Linear head: `residual_head: Linear(D_z → H)` applied per variable.
- Config flags:
  - `use_residual_head: bool`
  - `residual_scale: float` (how strongly to add the residual).

Given spline forecasts `y_spline` and compressed representations `compressed_repr`:

```python
if self.residual_head is not None and self.config.residual_scale > 0.0:
    residual_output = self.residual_head(compressed_repr)  # (B, M, H)
    forecasts = y_spline + self.config.residual_scale * residual_output
else:
    residual_output = None
```

Interpretation and usage:

- `interpretability['residual_forecast']` stores the residual component.
- For **pure interpretability**, we train/evaluate with `--no-residual-head` so that all predictive power comes from the spline.
- For **maximum accuracy**, we may enable the residual and optionally regularize its magnitude in the loss so it acts as a small correction rather than a dominant term.

---

## 8. Optional vs Core Components

**Always present (core interpretability path):**

- Per‑variable univariate learners.
- Temporal self‑attention with attention maps.
- Cross‑variable attention with `M × M` dependency matrices.
- Multi‑query temporal compression (with visualizable `compression_attention`).
- Spline head with interpretable control points, basis functions, and knot vectors.

**Optional components / knobs:**

- `num_compression_queries`:
  - `1` reproduces the original single‑query behavior.
  - `>1` improves capacity and multi‑scale compression, while keeping attention maps interpretable.
- `spline_smooth_alpha`:
  - `1.0` for very smooth, conservative splines.
  - Lower values to let control points react more strongly to data.
- Residual head:
  - Can be switched off entirely (`--no-residual-head`) for purely spline‑based explanations.
  - When enabled, residuals are explicit and inspectable.

In all cases, the **interpretability core remains the same**: every layer of the model exposes structured, human‑readable artefacts (attention maps, control points, basis functions) that explain how the model arrived at its forecasts.

