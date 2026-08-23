# Checkpoints — Storage Note

**49 GB of `*.pt` checkpoints are NOT versioned in this repo (git would bloat).**

Canonical location: `D:\modification research\checkpoint\` (35 files, 49.1 GB)

- `tiedw_original*.pt` (8) — linear query mode
- `tiedw_residual_gelu*.pt` (27) — nonlinear query mode
- `gpt3large_residual_gelu_ckpt60000_modelonly.pt` (1.67 GB) at `D:\modification research\gpt3large_residual_gelu_ckpt60000_modelonly.pt`
- prototype `linear/ckpt_50000.pt` and `non_linear/ckpt_50000.pt` at `D:\modification research\linear\` / `non_linear\`

The CSVs in this folder (`sweep_results.csv` 408 rows, `validation_losses.csv` 35 rows) are the **derived, versioned** results. Do NOT copy `.pt` files into `beyond_query_linearity_community/checkpoint/` — they will be ignored by `.gitignore` (add `*.pt`).

## Verification (no sweep rerun)

```powershell
python scripts/verify_configs.py --ckpt-dir "D:\modification research\checkpoint" --sweep-csv checkpoint/sweep_results.csv --val-loss-csv checkpoint/validation_losses.csv
python scripts/verify_configs.py --ckpt-dir "D:\modification research\checkpoint" --strict-hyperparams
```

`scripts/verify_configs.py:28` `CONFIGS` (34 entries) must match `scripts/run_sweep_fixed.py:211` and `scripts/compute_validation_losses.py:207`. Expected: `EXPECTED_ARCH:68` (12L/768d/12h), `expected_scale:96` (`1/(2√64)` for `residual_gelu`), `mlp_hidden_size` tolerance ±8.

## Why not moved from `D:\modification research\`

`D:\modification research\scripts\run_sweep_fixed.py` and `run_schur_analysis_gpt3_large.py` use a **small fixed ridge** (`ridge_rel=1e-3` + `pinv_eig`) — superseded by `beyond_query_linearity_community/schur_stats.py:1` cross-validated joint ridge (10-point `1e-7→1×scale` grid, nested CV). Copying would be a **downgrade**. This repo is the newer, corrected version (`2e0aa22`, `afcbd07`).

Community adds: `schur_stats.py`, `scripts/verify_configs.py`, `scripts/make_medium_figures.py`, `scripts/make_schur_sweep_figures.py`, `scripts/plot_style.py`, `figures/fig4-7` + unpacked `schur_results/aws_dun_schur/` (16 CSVs). `D:\modification research\schur_results.tgz` is the archived version of that folder.

## Disk

- `D:\modification research\checkpoint\` 49.1 GB, `D:` free 97 GB — do not duplicate unless you have >60 GB free.
- If you need local copies, use a symlink/junction:
  ```powershell
  cmd /c mklink /D "D:\beyond_query_linearity_community\checkpoint\pt" "D:\modification research\checkpoint"
  ```
