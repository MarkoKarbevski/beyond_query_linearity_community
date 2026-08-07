"""
Shared Schur-complement statistics with a consistent joint-ridge estimator.

Replaces the previous per-script (LedoitWolf + pinv) estimator, which was
inconsistent: it shrunk only Sigma_bb (tuned to minimize *covariance*
estimation error, not regression error), producing biased weights W and
systematically inflating the out-of-sample unexplained-variance statistic
nu_frac (more so for layers whose K/V covariance is close to isotropic).

The corrected estimator is a proper joint ridge solve:

    W(l) = Sigma_fb @ (Sigma_bb + l * I)^{-1}

which is the exact solution of min_W ||F_c - B_c W^T||^2_F + l * ||W||^2_F
on centered data. The ridge penalty lam is selected by a nested
cross-validation so it never sees the reported holdout: a small interior
validation slice of the training split picks the lam minimizing its
out-of-sample residual SS, then W is refit on the full training split and
evaluated only on the outer holdout.

All three analysis scripts import the functions here so the estimator is
defined once instead of being copied three times.
"""

import numpy as np


def eig_rank(S, rel_cutoff=1e-6):
    S = 0.5 * (S + S.T)
    w = np.linalg.eigvalsh(S)
    wmax = w.max() if w.max() > 0 else 1.0
    return int(np.sum(w > rel_cutoff * wmax))


def eig_rank_scaled(S, reference_scale, rel_cutoff=1e-3):
    S = 0.5 * (S + S.T)
    w = np.linalg.eigvalsh(S)
    ref = reference_scale if reference_scale > 0 else 1.0
    return int(np.sum(w > rel_cutoff * ref))


def pinv_eig(S, rel_cutoff=1e-8):
    """Moore-Penrose pseudo-inverse via eigh with a relative cutoff."""
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    wmax = w.max() if w.max() > 0 else 1.0
    keep = w > rel_cutoff * wmax
    w_inv = np.zeros_like(w)
    w_inv[keep] = 1.0 / w[keep]
    return (V * w_inv) @ V.T


def _ridge_grid(scale, n=10, lo_rel=1e-7, hi_rel=1.0):
    """Candidate joint-ridge penalties, from (nearly unregularized) to O(scale)."""
    scale = float(scale)
    if scale <= 0:
        scale = 1.0
    if n <= 1:
        return [0.0]
    grid = np.geomspace(lo_rel * scale, hi_rel * scale, n).tolist()
    return list(grid)


def _fit_weights_from_lam(w, sfb, V, lam):
    """W = sfb @ (V diag(1/(w+lam)) V^T) given eigendecomposition of Sigma_bb."""
    inv = 1.0 / (w + lam)
    return sfb @ (V * inv) @ V.T


def _residual_ss(F_c, B_c, W):
    R = F_c - B_c @ W.T
    return float(np.sum(R * R))


def schur_complement_stats(F, B, rel_cutoffs=(1e-3, 1e-4, 1e-6, 1e-8),
                           holdout_frac=0.5, seed=0, ridge_lambda=None,
                           val_frac=0.25, lam_grid=None):
    """How much of F is NOT linearly predictable from B, using joint-ridge OLS.

    Parameters
    ----------
    F : np.ndarray (N, Df)  -- query vectors (signal)
    B : np.ndarray (N, Db)  -- [K, V] vectors (predictors)
    ridge_lambda : float or None
        If None, cross-validated on a nested validation slice (recommended).
        If a float, used directly (interpreted relative to mean(diag Sigma_bb)).
    """
    rng = np.random.default_rng(seed)
    N = F.shape[0]
    idx = rng.permutation(N)
    n_train = int(N * (1 - holdout_frac))
    tr, ev = idx[:n_train], idx[n_train:]

    Ftr, Btr = F[tr], B[tr]
    Fev, Bev = F[ev], B[ev]

    Fmean, Bmean = Ftr.mean(0), Btr.mean(0)
    Ftr_c, Btr_c = Ftr - Fmean, Btr - Bmean
    Fev_c, Bev_c = Fev - Fmean, Bev - Bmean

    # -- Full-train sample covariance (unshrunk, consistent scale) --
    Sigma_bb = (Btr_c.T @ Btr_c) / (n_train - 1)
    Sigma_fb = (Ftr_c.T @ Btr_c) / (n_train - 1)
    scale = np.trace(Sigma_bb) / Sigma_bb.shape[0]

    # -- Nested lambda selection on an interior slice of the training split --
    if ridge_lambda is None:
        n_fit = int(n_train * (1 - val_frac))
        n_sel = n_train - n_fit
        fit_idx = tr[:n_fit]
        sel_idx = tr[n_fit:]

        Ffit_c = Ftr_c[:n_fit]
        Bfit_c = Btr_c[:n_fit]
        Fsel_c = Ftr_c[n_fit:]
        Bsel_c = Btr_c[n_fit:]

        Sbb_fit = (Bfit_c.T @ Bfit_c) / (n_fit - 1)
        Sfb_fit = (Ffit_c.T @ Bfit_c) / (n_fit - 1)
        w_fit, V_fit = np.linalg.eigh(Sbb_fit)
        scale_fit = np.mean(w_fit)

        grid = lam_grid if lam_grid is not None else _ridge_grid(scale_fit)
        best_lam = None
        best_rss = np.inf
        for lam in grid:
            W_cand = _fit_weights_from_lam(w_fit, Sfb_fit, V_fit, lam)
            rss = _residual_ss(Fsel_c, Bsel_c, W_cand)
            if rss < best_rss:
                best_rss, best_lam = rss, lam
        lam = best_lam
    else:
        lam = float(ridge_lambda) * scale

    # -- Refit on the full training split with the chosen ridge --
    w, V = np.linalg.eigh(Sigma_bb)
    W = _fit_weights_from_lam(w, Sigma_fb, V, lam)

    # -- In-sample residual (direct second moment on training) --
    resid_tr = Ftr_c - Btr_c @ W.T
    Rf_insample = (resid_tr.T @ resid_tr) / n_train

    # -- Out-of-sample holdout residual --
    resid_ev = Fev_c - Bev_c @ W.T
    Rf_holdout = (resid_ev.T @ resid_ev) / len(Fev)

    cond = float(w[-1] / w[0]) if w[0] > 0 else float("inf")

    out = {
        "Sigma_bb": Sigma_bb,
        "Sigma_fb": Sigma_fb,
        "Rf_insample": Rf_insample,
        "Rf_holdout": Rf_holdout,
        "ridge_lambda": float(lam),
        "cond_Sigma_bb": cond,
    }

    Sff_eval = (Fev_c.T @ Fev_c) / len(Fev)
    Sff_ref = np.trace(Sff_eval) / Sff_eval.shape[0]

    for cutoff in rel_cutoffs:
        out[f"rank_Sigma_bb@{cutoff:g}"] = eig_rank(Sigma_bb, cutoff)
        out[f"rank_Rf_insample@{cutoff:g}"] = eig_rank_scaled(Rf_insample, Sff_ref, cutoff)
        out[f"rank_Rf_holdout@{cutoff:g}"] = eig_rank_scaled(Rf_holdout, Sff_ref, cutoff)

    tr_Rf = np.trace(Rf_holdout)
    tr_Rf2 = np.trace(Rf_holdout @ Rf_holdout)
    tr_Sff = np.trace(Sff_eval)
    out["nu_eff"] = float(tr_Rf ** 2 / tr_Rf2) if tr_Rf2 > 0 else 0.0
    out["nu_frac"] = float(tr_Rf / tr_Sff) if tr_Sff > 0 else 0.0
    return out


def permutation_null(F, B, n_perm=3, **kwargs):
    rng = np.random.default_rng(123)
    nu_effs, nu_fracs = [], []
    for i in range(n_perm):
        perm = rng.permutation(F.shape[0])
        stats = schur_complement_stats(F[perm], B, seed=i, **kwargs)
        nu_effs.append(stats["nu_eff"])
        nu_fracs.append(stats["nu_frac"])
    return {
        "nu_eff_null_mean": float(np.mean(nu_effs)),
        "nu_eff_null_std": float(np.std(nu_effs)),
        "nu_frac_null_mean": float(np.mean(nu_fracs)),
        "nu_frac_null_std": float(np.std(nu_fracs)),
    }