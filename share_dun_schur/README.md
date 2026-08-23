# GPT-3 Large residual_gelu — Schur analysis + checkpoint losses (for Dun)

Model: GPT-3 Large (24L / 1536d / 16 heads / d_head 96), residual_gelu query,
60k steps, maxLR 2.55e-3 / minLR 2e-5 / wd 1e-1, seed-42 data. Final val loss 2.5458
(3.02% better than Simon's paired GPT-3 Large baseline, 2.625).

## Files
- `checkpoint_losses.csv` — val/train loss at each saved-checkpoint step (1k–60k).
  Checkpoint files themselves exist for 5k–60k (1k/2k were cleaned up).
- `schur_gpt3large_REAL_owt.csv` — layerwise Schur output, real OWT-val tokens (recommended).
- `schur_gpt3large_RANDOM.csv` — same, with the script's original random-token loader (biased).
- `nu_frac_random_vs_real.png` — per-layer nu_frac: random vs real vs shuffled-null.
- `run_schur_analysis_realdata.py` — the script with a drop-in real-data loader
  (reads OWT val.bin instead of random tokens). Only make_synthetic_loader differs from
  the original.

## Key findings (ckpt_60000)
- Random tokens systematically UNDER-estimate nu_frac, worst in deep layers:
  mean 0.111 (random) vs 0.161 (real); layer 23: 0.137 vs 0.244. Use real text.
- Real-data depth profile: low early (L1–2 ~0.05), rising with depth (L23 ~0.24).
  Different shape from the GPT-2 Medium early-layer separation — possibly scale/geometry.
- Sanity: shuffled null ~1.18 vs real 0.05–0.24, so the signal is clearly significant.

## Script review (original run_schur_analysis_gpt3_large.py)
Correct: clean checkpoint load (0 missing/unexpected), model matches the repo
(MLP GELU, learned wpe, pre-norm order, residual_gelu q-extraction), out-of-sample
holdout residual, working permutation null.
Suggested fixes: (1) real text not random tokens; (2) use_shrinkage=True (B is 3072-dim,
~10k samples — noisy cov; also makes the sklearn import get used); (3) max_batches 5 -> ~20.
