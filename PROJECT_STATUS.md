# Interpretable Time Series Forecasting - Project Status

**Last Updated:** January 2025  
**Status:** Phase 1 Complete - Model Simplification Implemented  
**Commit:** 6561b63

---

## Project Overview

This project implements an interpretable time series forecasting model for the ETT (Electricity Transformer Temperature) dataset. The model uses a 5-stage pipeline combining univariate function learners, temporal attention, cross-variable attention, temporal compression, and B-spline forecasting.

### Original Architecture (5 Stages)
```
Input (B, T, M=7)
  ↓
S1: Univariate Function Learners → (B, M, T, embed_dim)
  ↓
S2: Temporal Self-Attention → (B, M, T, embed_dim) + temporal_attn
  ↓
S3: Cross-Variable Attention → (B, M, T, cross_dim) + cross_attn
  ↓
S4: Temporal Encoder (Compression) → (B, M, compressed_dim) + compression_attn
  ↓
S5: Spline Function Learner (MLP) → Forecasts (B, M, forecast_horizon)
```

### Dataset
- **ETT (Electricity Transformer Temperature)**
- 7 variables: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
- Hourly measurements
- Typical usage: 96-168 time steps input → 24-48 steps forecast

---

## Teammate Suggestions (Original Requirements)

Your teammate suggested the following improvements for interpretability:

### 1. ✅ **Remove S4 Black Box (SplineMLP)** - COMPLETED
- **Problem:** The MLP that predicts control points is a black box
- **Solution:** Output control points directly from S3 (compression stage)
- **Benefit:** Fully interpretable - "compressed representation IS the spline shape"

### 2. ✅ **Preserve Per-Head Attention Weights** - COMPLETED
- **Problem:** Averaging attention heads destroys information
- **Solution:** Return per-head attention weights (don't average)
- **Benefit:** Can analyze individual head specialization

### 3. ✅ **Force Sparse Attention (Sparsemax/Entmax)** - COMPLETED
- **Problem:** Standard softmax creates blurry attention heatmaps
- **Solution:** Use sparsemax for sparse attention (many weights = 0)
- **Benefit:** Clear causality - can point to specific important time steps

### 4. ⏳ **Simplify S3 (Compression) - Mean + Std** - NOT YET IMPLEMENTED
- **Current:** Attention-based compression
- **Suggested:** Simple mean + std vector (more interpretable)
- **Alternative:** Keep sparse attention (already implemented)
- **Decision Needed:** Test both approaches

---

## Phase 1 Changes Implemented

### Change 1: Preserve Per-Head Attention Weights
**File:** `main model/extended_model.py`

**What Changed:**
- `CrossVariableAttention.forward()`: Removed `.mean(dim=1)` call
- Now returns `(B, T, num_heads, M, M)` instead of `(B, num_heads, M, M)`
- Updated docstrings and test expectations

**Impact:**
- Full per-head, per-timestep attention preserved
- Can analyze individual attention head patterns
- No information loss from averaging

**Code Location:**
```python
# Line ~336 in extended_model.py
# OLD: attn_weights_avg = attn_weights.mean(dim=1)
# NEW: return final_output, attn_weights  # Full (B, T, num_heads, M, M)
```

---

### Change 2: Add Sparse Attention (Sparsemax)
**Files:** `main model/model.py`, `main model/extended_model.py`

**What Changed:**
- Added `from entmax import sparsemax` with fallback to softmax
- Replaced softmax with sparsemax in `TemporalEncoder` compression attention
- PyTorch's MultiheadAttention in S1/S2 cannot use sparsemax (would require rewrite)

**Impact:**
- **66.1% sparsity** in compression attention (many weights exactly 0)
- Clear identification of important time steps
- More interpretable than blurry softmax heatmaps

**Code Location:**
```python
# Line ~451 in extended_model.py (TemporalEncoder)
if SPARSEMAX_AVAILABLE:
    compression_attn = sparsemax(attention_scores, dim=-1)  # Sparse!
else:
    compression_attn = torch.softmax(attention_scores, dim=-1)
```

**Note:** S1 (TemporalSelfAttention) and S2 (CrossVariableAttention) still use standard softmax because they use PyTorch's built-in MultiheadAttention module.

---

### Change 3: Remove SplineMLP (Direct Control Points)
**File:** `main model/extended_model.py`

**What Changed:**

1. **Simplified `SplineFunctionLearner`:**
   - Removed `control_point_predictor` MLP (3-layer network)
   - Now only generates B-spline basis functions
   - Forward method accepts control points directly: `forward(control_points)`

2. **Added Direct Projection in `InterpretableForecastingModel`:**
   - New layer: `self.control_point_projection = nn.Linear(compressed_dim, num_control_points)`
   - Projects compressed representation directly to control points
   - Simple, interpretable linear transformation

3. **Updated Forward Pass:**
   ```python
   # Stage 4: Direct control point projection
   control_points = self.control_point_projection(compressed_repr)
   spline_results = self.spline_learner(control_points)
   ```

**Impact:**
- **21% parameter reduction** (76K → 59.8K parameters)
- Fully interpretable: compressed_repr → linear → control_points → forecasts
- No black-box MLP hiding the transformation
- Control points directly reflect compressed state

**Code Locations:**
- `SplineFunctionLearner.__init__`: Line ~500 (removed MLP)
- `SplineFunctionLearner.forward`: Line ~678 (accepts control points)
- `InterpretableForecastingModel._init_extended_components`: Line ~1016 (added projection)
- `InterpretableForecastingModel.forward`: Line ~1118 (uses projection)

---

## Testing Results

### Test 1: Per-Head Attention (Task 1)
```
✅ TemporalSelfAttention: (B, num_heads, T, T)
✅ CrossVariableAttention: (B, T, num_heads, M, M)
✅ Full model preserves all per-head attention weights
```

### Test 2 & 3: Sparsemax + Direct Control Points
```
✅ Sparsemax available: True
✅ Sparse attention achieved: 66.1% zeros
✅ Attention weights properly normalized (sum to 1.0)
✅ Control points shape: (B, M, num_control_points)
✅ Control points vary with input (interpretable)
✅ Gradients flow correctly
✅ Model parameters: 59,784 (reduced from 76K)
✅ Forecasts generated successfully
```

---

## Current Model Architecture (After Phase 1)

```
Input (B, T, M=7)
  ↓
S1: Univariate Function Learners → (B, M, T, embed_dim)
  ↓
S2: Temporal Self-Attention → (B, M, T, embed_dim) + per-head temporal_attn
  ↓
S3: Cross-Variable Attention → (B, M, T, cross_dim) + per-head cross_attn
  ↓
S4: Temporal Encoder (Compression) → (B, M, compressed_dim) + SPARSE compression_attn
  ↓
S5: Direct Linear Projection → control_points (B, M, num_control_points)
  ↓
S6: B-Spline Basis Functions → Forecasts (B, M, forecast_horizon)
```

**Key Improvements:**
- ✅ Per-head attention preserved (no averaging)
- ✅ Sparse attention in compression (66% sparsity)
- ✅ Direct control points (no MLP black box)
- ✅ 21% fewer parameters
- ✅ Fully interpretable pipeline

---

## Remaining Work (Phase 2 - Optional)

### Option 1: Simplify S3 to Mean + Std (Teammate Suggestion)
**Current:** Attention-based temporal compression  
**Proposed:** Statistical pooling (mean + std)

**Implementation:**
```python
# Replace TemporalEncoder with:
mean = x.mean(dim=2)  # (B, M, cross_dim)
std = x.std(dim=2)    # (B, M, cross_dim)
compressed = torch.cat([mean, std], dim=-1)  # (B, M, 2*cross_dim)
# Then project to num_control_points
```

**Pros:**
- Maximum interpretability (well-understood statistics)
- Deterministic (no learned parameters)
- Very fast

**Cons:**
- Less flexible (can't learn which time steps matter)
- Uniform weighting (treats all time steps equally)
- May lose important temporal patterns

**Decision:** Test both approaches and compare:
- Current: Sparse attention (flexible, learned)
- Alternative: Mean + std (simple, interpretable)

---

### Option 2: Add Sparsemax to S1 and S2 (More Comprehensive)
**Challenge:** Would require rewriting attention mechanisms  
**Current:** S1 and S2 use PyTorch's MultiheadAttention (can't replace softmax)  
**Solution:** Implement custom attention with sparsemax

**Effort:** High (2-3 hours)  
**Benefit:** Sparse attention throughout entire model

---

## File Structure

```
interpretable_forecasting/
├── main model/
│   ├── model.py                    # Base model (S1, S2)
│   ├── extended_model.py           # Extended model (S3, S4, S5)
│   └── data_utils.py               # ETT data loading
├── tests/                          # Test files (gitignored)
│   ├── test_task1.py
│   ├── test_task1_minimal.py
│   └── test_tasks2_3.py
├── .gitignore                      # Updated to ignore tests/
├── PROJECT_STATUS.md               # This file
└── README.md                       # Project documentation
```

---

## Environment Setup

### Virtual Environment (IMPORTANT!)
**Issue:** Having venv inside project folder causes slow imports (Python 3.13 issue)

**Solution:**
```bash
# Create venv OUTSIDE project folder
cd ~/Desktop/main/ucsc/cmpm118/
python3 -m venv venv_forecasting
source venv_forecasting/bin/activate

# Install dependencies
pip install torch pandas numpy matplotlib entmax

# Remove old venv inside project
rm -rf updated_forecasting/interpretable_forecasting/venv
```

### Dependencies
```
torch>=2.0
pandas>=2.0
numpy>=1.20
matplotlib>=3.5
entmax>=1.3  # For sparsemax
```

---

## How to Continue Work (Next Session)

### Quick Start
1. Activate venv: `source ~/Desktop/main/ucsc/cmpm118/venv_forecasting/bin/activate`
2. Navigate: `cd ~/Desktop/main/ucsc/cmpm118/updated_forecasting/interpretable_forecasting`
3. Read this file: `PROJECT_STATUS.md`
4. Review changes: `git log --oneline -5`

### Testing
```bash
# Run Phase 1 tests
python tests/test_task1.py          # Per-head attention
python tests/test_tasks2_3.py       # Sparsemax + direct control points

# Run full validation (if available)
python validate_extended_model.py
```

### Key Files to Review
- `main model/extended_model.py` - All Phase 1 changes
- `main model/model.py` - Sparsemax import
- `.gitignore` - Tests folder excluded

---

## Performance Metrics

### Before Phase 1
- Parameters: ~76,000
- Attention: Dense (averaged heads)
- Control Points: MLP-predicted (black box)
- Interpretability: Moderate

### After Phase 1
- Parameters: 59,784 (21% reduction)
- Attention: Sparse (66% zeros in compression)
- Control Points: Direct linear projection
- Interpretability: High

### Accuracy (ETT Dataset - from previous validation)
- 1-step ahead: MSE: 1.058, MAE: 0.810
- 12-step ahead: MSE: 9.141, MAE: 2.236
- 24-step ahead: MSE: 4.255, MAE: 1.375
- 48-step ahead: MSE: 3.133, MAE: 1.283

---

## Git Information

### Recent Commits
```
6561b63 - Phase 1 Complete: Model Simplification for Interpretability
9c47f1d - Previous commit
```

### Branch
- Main branch: `main`
- Remote: `origin` (https://github.com/ctsapugay/interpretable_forecasting.git)

### Modified Files (Phase 1)
- `interpretable_forecasting/.gitignore`
- `interpretable_forecasting/main model/model.py`
- `interpretable_forecasting/main model/extended_model.py`

---

## Known Issues / Notes

1. **Python 3.13 Import Slowness**
   - Fixed by moving venv outside project folder
   - Imports now fast (~2-3 seconds vs 2+ minutes)

2. **Sparsemax Limitations**
   - Only applied to TemporalEncoder (S4)
   - S1 and S2 still use standard softmax (PyTorch limitation)
   - Would need custom attention implementation for full sparsemax

3. **Test Files**
   - Located in `tests/` folder
   - Gitignored (not pushed to repo)
   - Run locally for validation

4. **Stability Constraints**
   - Removed from SplineFunctionLearner (no longer needed)
   - Control points now directly from linear projection
   - Can add back if forecasts become unstable

---

## Questions for Next Session

1. **Should we implement mean + std compression (Phase 2)?**
   - Test performance vs. current sparse attention
   - Compare interpretability

2. **Should we add sparsemax to S1 and S2?**
   - Requires rewriting attention mechanisms
   - Significant effort but more comprehensive

3. **Should we add visualization for sparse attention?**
   - Show which time steps have non-zero weights
   - Highlight attention head specialization

4. **Should we validate on full ETT dataset?**
   - Run comprehensive accuracy tests
   - Compare before/after Phase 1 performance

---

## Contact / Team

- **Your Role:** Implementation lead
- **Teammate:** Suggested interpretability improvements
- **Repository:** https://github.com/ctsapugay/interpretable_forecasting.git

---

## Quick Reference Commands

```bash
# Activate environment
source ~/Desktop/main/ucsc/cmpm118/venv_forecasting/bin/activate

# Navigate to project
cd ~/Desktop/main/ucsc/cmpm118/updated_forecasting/interpretable_forecasting

# Run tests
python tests/test_tasks2_3.py

# Check git status
git status

# View recent changes
git diff HEAD~1

# Pull latest changes
git pull origin main
```

---

**End of Project Status Document**
