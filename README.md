# Beyond Linearity — Schur Complement Analysis of Attention Queries

## Research Question

**Can we make the attention query vector (Q) in transformers genuinely "beyond linear" — i.e., not just a linear function of the input, but containing information that cannot be linearly predicted from the key (K) and value (V) vectors?**

Standard GPT-style attention computes Q, K, V all as learned linear projections of the same input:

```
Q = W_q * x,  K = W_k * x,  V = W_v * x
```

Since Q, K, V are all linear functions of the same x, if you know K and V, you can linearly predict Q almost perfectly. The "residual_gelu" modification replaces the linear Q projection with:

```
Q = x + MLP(GELU(RMSNorm(x)))
```

where the MLP is a bottleneck: `n_embd → n_embd/2 → n_embd`. K and V remain learned linear projections. This makes Q a **nonlinear** function of x, so it may contain information that K and V cannot linearly predict.

---

## Experimental Design

### The Metric: nu_frac

We measure **nu_frac** = tr(R_f) / tr(S_ff) where:

- **S_ff** = covariance of the true query vectors Q (the "signal")
- **R_f** = covariance of the residual after linearly regressing Q on (K, V) (the "unpredictable" part)
- **nu_frac ≈ 0** means Q is almost perfectly linearly predictable from K,V (the "original" case)
- **nu_frac > 0** means Q contains information not linearly accessible from K,V (the "residual_gelu" case)

We use:
- **Out-of-sample evaluation**: split data 50/50, fit regression on one half, measure residual on the other
- **Ledoit-Wolf shrinkage** for the 2d-dimensional B covariance matrix (stabilizes high-dimensional estimation)
- **Permutation null baseline**: shuffle Q against (K,V) to measure the noise floor
- **Real text data** (WikiText-2) rather than random tokens (random tokens systematically underestimate nu_frac)

### Architecture Variants

We compare two query modes:

| Mode | Q computation | K,V computation | Parameters saved |
|------|-------------|----------------|-----------------|
| **original** | Q = W_q * x (learned linear, 3n_embd → 3n_embd) | K = W_k * x, V = W_v * x | 3n_embd weights |
| **residual_gelu** | Q = x + MLP(RMSNorm(x)) (bottleneck: 1536→768→1536) | K, V from 2n_embd proj | 2n_embd + MLP |

### Model Scales

| Scale | Layers | n_embd | Heads | d_head | Params | Checkpoints |
|-------|--------|--------|-------|--------|--------|-------------|
| GPT-2 Medium | 12 | 768 | 12 | 64 | ~124M | 34 configs (sweep) |
| GPT-3 Large | 24 | 1536 | 16 | 96 | ~770M | 1 (Jiwon's) |

---

## Code Files

### Core Analysis Scripts

| File | Purpose |
|------|---------|
| `scripts/run_schur_analysis_gpt3_large.py` | **Main analysis for GPT-3 Large checkpoint.** Loads a checkpoint, runs per-layer Schur analysis on WikiText-2, saves nu_frac per layer. Fixed version with real data, shrinkage, and proper checkpoint loading. |
| `scripts/run_sweep_fixed.py` | **Sweep across all 34 GPT-2 checkpoints.** Runs the same Schur analysis for every config (original + residual_gelu, various hyperparams). Output: `checkpoint/sweep_results.csv` |
| `scripts/compute_validation_losses.py` | **Validation loss computation.** Runs all 34 checkpoints on WikiText-2 to get cross-entropy loss. Output: `checkpoint/validation_losses.csv` |
| `scripts/verify_theorem1_activations.py` | **Theoretical verification.** Numerically verifies Theorem 1: that ReLU gives exact parity computation while GELU gives error ~ exp(-2κ²). |

### Jupyter Notebooks

| File | Purpose |
|------|---------|
| `notebooks/beyondlinearity_V3 (1).ipynb` | **Original V3 analysis.** First comparison of one "linear" (original) vs one "nonlinear" (residual_gelu) checkpoint. Contains the model definition, hook-based feature extraction, and Schur complement stats. |
| `notebooks/beyondlinearity_sweep (3).ipynb` | **Multi-config sweep.** Generalizes V3 to run across all 34 checkpoints. Downloads from Google Drive, iterates over configs, saves results to CSV. |
| `notebooks/EmpiricalValidation_v2 (1).ipynb` | **Empirical validation.** Additional validation experiments. |
| `notebooks/kernel_fusion.ipynb` | **Kernel fusion analysis.** Studies the kernel fusion perspective of the query modification. |

### Data Files

| File | Contents |
|------|----------|
| `checkpoint/sweep_results.csv` | Per-layer nu_frac for all 34 GPT-2 configs (8 original + 26 residual_gelu) |
| `checkpoint/validation_losses.csv` | Validation loss (WikiText-2) for all 34 configs |
| `share_dun_schur/schur_gpt3large_REAL_owt.csv` | Per-layer nu_frac for GPT-3 Large (real OWT data) |
| `share_dun_schur/schur_gpt3large_RANDOM.csv` | Per-layer nu_frac for GPT-3 Large (random tokens, for comparison) |
| `share_dun_schur/checkpoint_losses.csv` | Training/validation loss trajectory for GPT-3 Large (1k-60k steps) |

### Figures

| File | Description |
|------|-------------|
| `figures/fig1_nufrac_vs_loss.png` | nu_frac vs validation loss across configs |
| `figures/fig2_depth_comparison.png` | Depth profile comparison |
| `figures/fig3_rank_analysis.png` | Rank analysis of residual covariance |
| `figures/fig_theorem_gelu_error.png` | Theorem 1: GELU error decay with κ |
| `figures/overhead_vs_nufrac_corrected.png` | Overhead vs nu_frac |
| `share_dun_schur/nu_frac_random_vs_real.png` | GPT-3 Large: random vs real data comparison |

---

## Results Summary

### GPT-2 Medium Sweep (34 configs)

| Query Mode | Configs | Mean nu_frac | Interpretation |
|------------|---------|-------------|----------------|
| **original** | 8 | **0.0088** | Q is ~99.1% linearly predictable from K,V |
| **residual_gelu** | 26 | **0.1332** | Q is ~13.3% NOT predictable from K,V — **15x separation** |

**Key finding**: The residual_gelu modification consistently produces Q vectors that are significantly less linearly predictable from K,V, across all 26 hyperparameter settings (varying weight decay, learning rate, training steps). The effect is robust and not sensitive to hyperparameters.

### GPT-3 Large (24 layers, 1536d)

| Data Source | Mean nu_frac | Notes |
|-------------|-------------|-------|
| **Real text** (OWT) | **0.161** | Ground truth measurement |
| **Random tokens** | 0.111 | Systematic underestimation in deep layers |
| **Shuffled null** | ~1.18 | Noise floor — all real values are significant |

**Depth profile**: Low in early layers (L1-2: ~0.05), rising with depth (L23: ~0.24). Different shape from GPT-2 Medium, possibly due to scale/geometry differences.

### Validation Loss

- GPT-3 Large achieves **val loss 2.5458** at 60k steps (3.02% better than Simon's GPT-3 Large baseline of 2.625)
- GPT-2 Medium configs range from 3.55 to 8.06 (with `tiedw_original_005wd` being an outlier at 8.06, likely due to training instability with high weight decay)

---

## Why These Experiments?

### 1. Why Schur Complement?

The Schur complement measures how much variance of Q remains after removing the best linear prediction from (K,V). This is natural because:
- In standard attention, Q, K, V are all linear projections of the same input — so Q should be almost perfectly predictable from K,V if the linear projections span similar subspaces
- The residual_gelu modification injects a nonlinear MLP bottleneck into Q's computation, which should break this linear predictability
- nu_frac quantifies exactly "how much beyond linear" the query is

### 2. Why Two Model Scales?

- **GPT-2 Medium (12L, 768d)**: Enables extensive hyperparameter sweep (34 configs) to test robustness of the effect. Can run on a single GPU.
- **GPT-3 Large (24L, 1536d)**: Tests whether the finding scales to larger models. Jiwon's checkpoint achieves 3% better validation loss than the original baseline, confirming the modification is not just theoretically interesting but practically useful.

### 3. Why Real Text vs Random Tokens?

Early experiments used random tokens, which systematically underestimate nu_frac (mean 0.111 vs 0.161 for GPT-3 Large). Real text produces more realistic activation patterns and is essential for accurate cross-model comparison.

### 4. Why the Permutation Null?

The permutation null (shuffling Q against K,V) measures the noise floor of the estimation procedure. If nu_frac from real data is significantly above the null, we can be confident the effect is real and not just estimation noise.

---

## Files to Share with Marko and Team

### Essential (send these):
```
scripts/
├── run_schur_analysis_gpt3_large.py    # Main analysis script (GPT-3 Large)
├── run_sweep_fixed.py                   # Sweep across all 34 checkpoints
├── compute_validation_losses.py         # Validation loss computation
└── verify_theorem1_activations.py       # Theorem 1 verification

checkpoint/
├── sweep_results.csv                    # Per-layer nu_frac for all 34 configs
└── validation_losses.csv                # Validation losses for all 34 configs

share_dun_schur/
├── schur_gpt3large_REAL_owt.csv         # GPT-3 Large Schur results (real data)
├── schur_gpt3large_RANDOM.csv           # GPT-3 Large Schur results (random tokens)
├── checkpoint_losses.csv                # Training trajectory
├── nu_frac_random_vs_real.png           # Visualization
└── README.md                            # Dun's notes

figures/
├── fig1_nufrac_vs_loss.png
├── fig2_depth_comparison.png
├── fig3_rank_analysis.png
└── fig_theorem_gelu_error.png

notebooks/
├── beyondlinearity_V3 (1).ipynb         # Original V3 analysis
├── beyondlinearity_sweep (3).ipynb      # Multi-config sweep
├── EmpiricalValidation_v2 (1).ipynb     # Empirical validation
└── kernel_fusion.ipynb                  # Kernel fusion analysis
```

### Optional (only if they want to reproduce):
```
checkpoint/*.pt                          # All 34 checkpoint files (~250 MB each)
gpt3large_residual_gelu_ckpt60000_modelonly.pt  # GPT-3 Large checkpoint (1.67 GB)
```

### Dependencies
```
pip install torch numpy scikit-learn datasets transformers matplotlib pandas