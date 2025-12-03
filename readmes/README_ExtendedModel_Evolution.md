## Extended Model Evolution: v1 → v2

### 1. Overview

This document records the evolution of the **Extended Interpretable Forecasting Model** from the initial implementation (**v1**) to the current, more flexible version (**v2**).

The goal of these changes was to:
- Fix **early convergence** and underfitting.
- Make the **spline control points themselves** carry more of the forecasting behavior (not just a post‑hoc smoothing).
- Preserve and enhance **interpretability** (attention maps, control points, basis functions).

---

### 2. v1: Original Extended Model

#### 2.1 Architecture

- **Temporal encoding**
  - `InterpretableTimeEncoder`:
    - Per‑variable univariate learners (2‑layer MLP).
    - Shared temporal self‑attention with positional encoding.
  - Output: `(B, M, T, embed_dim)` + temporal attention `(B, M, heads, T, T)`.

- **Cross‑variable attention**
  - `CrossVariableAttention`:
    - Input: `(B, M, T, embed_dim)`.
    - Output: `(B, M, T, cross_dim)` + cross attention `(B, heads, M, M)`.

- **Temporal compression (v1)**
  - `TemporalEncoder` with **one** learned compression query per variable:
    - Input: `(B, M, T, cross_dim)`.
    - Output: `(B, M, compressed_dim)` + `compression_attn` `(B, M, 1, T)`.
    - Essentially a **single attention‑pooled summary** per variable.

- **Spline forecasting head (v1)**
  - `SplineFunctionLearner`:
    - Input: `(B, M, compressed_dim)`.
    - Control points: `num_control_points = 8`, `spline_degree = 3`.
    - Stability:
      - Hard clamp of control points to `[-10, 10]`.
      - Fixed smoothing kernel (0.25, 0.5, 0.25) applied across control‑point index.
    - Basis functions were evaluated at:
      - \(t \in [1.0, 1.5]\), i.e. outside the main knot range \([0, 1]\).
      - Forecast: `forecasts = control_points @ basis.T`.

#### 2.2 Behavior & Issues

- **Early convergence**
  - Training and validation loss flattened early (best epoch ≈ 12) even with reasonable capacity.

- **Underfitting & flat control points**
  - Spline forecasts were **too smooth**; large deviations in the true future were under‑represented.
  - Control‑point plots showed:
    - A long, nearly **flat “front” segment**.
    - Only the **last few control points** moved significantly.
  - Reason:
    - Because basis functions were evaluated at \(t \in [1.0, 1.5]\), the spline behaved like a pure **extrapolation** beyond the last knots.
    - Most of the basis weight concentrated on the last control points, so:
      - Early control points received almost no gradient.
      - They remained very close to their initial, heavily smoothed values.

- **Compression bottleneck**
  - Single compression query ⇒ each variable had only **one global temporal summary**.
  - Local structure and outliers could be washed out before the spline head ever saw them.

---

### 3. v2: Current Extended Model

v2 keeps the core interpretability story (univariate learners → temporal attention → cross‑attention → compression → splines) but introduces several key improvements.

#### 3.1 Configuration Extensions

`ExtendedModelConfig` now includes:

- **Temporal compression**
  - `num_compression_queries: int`  
    Number of compression queries per variable (multi‑query compression).

- **Spline flexibility**
  - `spline_smooth_alpha: float` in \([0, 1]\)  
    Interpolation weight between raw and smoothed control points.

- **Head options**
  - `use_spline_head: bool`  
    Switch between spline head vs simple linear head (diagnostic).
  - `use_residual_head: bool` and `residual_scale: float`  
    Optional residual correction head on top of the spline forecast.

These fields are exposed in `train_extended_model.py` as CLI flags:
- `--num-compression-queries`
- `--spline-smooth-alpha`
- `--no-spline-head`
- `--no-residual-head`

#### 3.2 Multi‑Query Temporal Compression

**Old:** one compression query per variable → `(B, M, compressed_dim)`.  
**New:** `num_compression_queries = Q` per variable:

- Compression queries:
  - Parameter: `(1, Q, 1, input_dim)` expanded to `(B, M, Q, D)`.
- Attention:
  - Scores: `(B, M, Q, T)`.
  - Weights: `compression_attn` `(B, M, Q, T)`.
- Output:
  - Compressed representation: `(B, M, Q * compressed_dim)`.

**Interpretability impact**
- Each query has its **own attention map**, so we can visualize:
  - Which time steps each query focuses on.
  - How different queries capture different temporal patterns (e.g., recent vs long‑term history).
- For small `Q` (e.g., 2–4) the complexity remains manageable while significantly increasing capacity.

#### 3.3 Spline Head: From Rigid to Tunable

Key changes in `SplineFunctionLearner`:

1. **Tunable smoothing via `spline_smooth_alpha`**
   - v1: always applied a fixed smoothing kernel; control points were heavily smoothed.
   - v2:
     - Compute a smoothed version of the control points.
     - Interpolate:
       \[
       c_{\text{final}} = (1 - \alpha) \cdot c_{\text{raw}} + \alpha \cdot c_{\text{smoothed}}
       \]
     - `spline_smooth_alpha = 1.0` reproduces the original behavior.
     - Lower values (e.g., `0.2–0.4`) allow sharper bends and more variable‑specific structure.

2. **Basis functions over full support `[0, 1]`**
   - v1: `t_eval = linspace(1.0, 1.5, forecast_horizon)`  
     ⇒ only last control points had significant influence on the forecast (others saw almost zero gradient).
   - v2: `t_eval = linspace(0.0, 1.0, forecast_horizon)` (and for extrapolation/visualization as well).
     - All control points are active over the forecast horizon.
     - Gradients flow into **every** control point.
     - The “flat initial control points” effect disappears; front and back of the spline both learn meaningful shapes.

3. **More control points**
   - Typical v2 configs use `num_control_points = 12` or `16` instead of `8`.
   - This increases expressiveness while keeping the spline structure interpretable (each point is still individually plottable).

#### 3.4 Optional Residual Correction Head

To separate “smooth, interpretable trend” from “fine‑grained corrections”, v2 can add a residual head:

- `residual_head: Linear(effective_compressed_dim → forecast_horizon)`.
- Final forecast:
  \[
  y = y_{\text{spline}} + \text{residual\_scale} \cdot y_{\text{residual}}.
  \]
- Controlled via:
  - `use_residual_head` (on/off).
  - `residual_scale` (strength of correction).
- The residual output is exposed in `interpretability['residual_forecast']` and can be visualized separately.

In practice we found:
- With residual head **on** and `spline_smooth_alpha` relatively high, the optimiser tends to keep control points simple and let the residual head do most of the work.
- For **pure spline interpretability**, we now train with `--no-residual-head` and rely on the more flexible spline itself.

#### 3.5 Training Regimen Improvements

To address early convergence and stabilise training:

- Added `--warmup-epochs`:
  - First `warmup_epochs` use a reduced LR (≈ 0.3× the base LR) before ramping to the target LR.
- Continued support for `--scheduler` (ReduceLROnPlateau), but with:
  - Larger `--patience` (e.g., 20) to allow the higher‑capacity model to keep learning.
- Configured via CLI, e.g.:

```bash
python train_extended_model.py \
  --epochs 30 --batch-size 32 --lr 5e-4 \
  --scheduler --warmup-epochs 3 --patience 20
```

---

### 4. Typical v2 Configuration (ETTh1)

For ETTh1 we currently use the following settings when we want the spline itself to be responsible for the forecasts (no residual head):

- **Data / windows**
  - `input_length = 96`
  - `forecast_horizon = 24`
  - Splits: `train=0.7`, `val=0.2`, `test=0.1`

- **Model**
  - `num_variables = 7`
  - `embed_dim = 32`
  - `hidden_dim = 64`
  - `num_heads = 4`
  - `cross_dim = 32`, `cross_heads = 4`
  - `compressed_dim = 64`
  - `num_compression_queries = 2`
  - `num_control_points = 12–16`
  - `spline_degree = 3`
  - `spline_stability = True`
  - `spline_smooth_alpha ≈ 0.2–0.4`
  - `use_spline_head = True`
  - `use_residual_head = False`

- **Training**
  - Optimiser: Adam, `lr = 5e-4`, `weight_decay = 1e-5`
  - `epochs ≈ 30`, `batch_size = 32`
  - `scheduler = ReduceLROnPlateau` with `patience ≈ 5`
  - `warmup_epochs = 3`
  - `patience = 20` for early stopping

Under this configuration, we observe:
- Best validation epochs typically in the **20–30** range (no early plateau).
- Spline forecasts that closely match the true future for ETT variables.
- Control‑point plots where:
  - All points (not just the tail) participate.
  - Different variables show distinct, interpretable shapes and trends.

---

### 5. Summary of v1 → v2 Changes

| Area                  | v1 (Original)                                  | v2 (Current)                                                  |
|-----------------------|-----------------------------------------------|----------------------------------------------------------------|
| Compression           | 1 query / variable, `(B, M, T, D) → (B, M, D)` | `Q` queries / variable, `(B, M, T, D) → (B, M, Q·D)` + `(B,M,Q,T)` attn |
| Spline basis          | `t ∈ [1, 1.5]` (extrapolation region)          | `t ∈ [0, 1]`, full use of all control points                  |
| Smoothing             | Hard fixed smoothing                           | Tunable `spline_smooth_alpha ∈ [0,1]`                         |
 advising                | Control points often flat except last few     | All control points active and shaped by gradients             |
| Residual head         | Not available                                 | Optional additive residual head with controllable scale       |
| Training schedule     | No warmup, short patience                      | LR warmup, configurable patience, same scheduler              |

Overall, v2 keeps the **same interpretability story** but significantly reduces early convergence and allows the spline control points themselves to model richer, variable‑specific forecast shapes.



