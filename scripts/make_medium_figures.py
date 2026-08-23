"""Reproduce the GPT-2 Medium figures (fig1/fig2/fig3) from the CSVs.

Unlike fig4-fig7 (scripts/make_schur_sweep_figures.py), fig1-fig3 previously had
no committed generator. This script reproduces them from
  checkpoint/sweep_results.csv      (per-layer nu_frac / nu_eff / rank per config)
  checkpoint/validation_losses.csv   (WikiText-2 val loss per config)

Data hygiene (applied consistently, documented in README + write-up):
  - `tiedw_original_005wd` (diverged in training, val loss 8.06) is excluded.
  - The single outlier row `tiedw_original_mlp_4_625_times` layer 3
    (nu_frac = 0.179) is excluded.

Run from the repo root:
    python scripts/make_medium_figures.py
Outputs (overwrites):
    figures/fig1_nufrac_vs_loss.png
    figures/fig2_depth_comparison.png
    figures/fig3_rank_analysis.png
Also prints the recomputed summary statistics for diffing against the write-up.
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
from plot_style import apply_style, save_figure, PALETTE, DIVERGED_CONFIG, OUTLIER_CONFIG

apply_style()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP_CSV = os.path.join(ROOT, "checkpoint", "sweep_results.csv")
VAL_LOSS_CSV = os.path.join(ROOT, "checkpoint", "validation_losses.csv")
FIGDIR = os.path.join(ROOT, "figures")

MODE_COLORS = {
    "original": PALETTE["original"],
    "residual_gelu": PALETTE["residual_gelu"],
}
STEP_MARKERS = {60: "o", 20: "s"}


def clean_sweep():
    """Sweep rows with the diverged config and the outlier row removed."""
    rows = list(csv.DictReader(open(SWEEP_CSV, encoding="utf-8")))
    out = []
    for r in rows:
        if r["config"] == DIVERGED_CONFIG:
            continue
        if r["config"] == OUTLIER_CONFIG and int(r["layer"]) == 3:
            continue
        out.append({
            "config": r["config"],
            "mode": r["query_mode"],
            "layer": int(r["layer"]),
            "steps": 20 if "20k" in r["config"] else 60,
            "nu_frac": float(r["nu_frac"]),
            "nu_eff": float(r["nu_eff"]),
            "rank": float(r["rank_Rf_holdout@1e-6"]),
        })
    return out


def load_val_loss():
    return {
        r["config"]: float(r["val_loss"])
        for r in csv.DictReader(open(VAL_LOSS_CSV, encoding="utf-8"))
        if r["config"] != DIVERGED_CONFIG
    }


def config_mean_nufrac(sweep):
    out = {}
    for r in sweep:
        out.setdefault(r["config"], []).append(r["nu_frac"])
    return {c: float(np.mean(v)) for c, v in out.items()}


def layer_stats(sweep, mode, field):
    """Per-layer [median, p25, p75] of `field` for one mode."""
    layers = sorted(set(r["layer"] for r in sweep))
    med, lo, hi = [], [], []
    for L in layers:
        vals = np.array([r[field] for r in sweep if r["mode"] == mode and r["layer"] == L])
        med.append(np.median(vals))
        lo.append(np.percentile(vals, 25))
        hi.append(np.percentile(vals, 75))
    return layers, np.array(med), np.array(lo), np.array(hi)


def fig1(sweep, val_loss, means):
    """nu_frac vs validation loss, per config. Log y so both clouds are visible."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    # scatter per mode/step
    for mode, color in MODE_COLORS.items():
        cfgs = [c for c in means if c.startswith("tiedw_" + mode)]
        for steps, marker in STEP_MARKERS.items():
            sel = [c for c in cfgs if ("20k" in c) == (steps == 20)]
            xs = [val_loss[c] for c in sel]
            ys = [means[c] for c in sel]
            ax.scatter(xs, ys, c=color, marker=marker, s=48, alpha=0.9,
                       edgecolors="white", linewidths=0.6, zorder=3)

    # trend lines + in-axes r annotation (no adjustText, no overflow)
    for mode, color in MODE_COLORS.items():
        cfgs = [c for c in means if c.startswith("tiedw_" + mode)]
        xs = np.array([val_loss[c] for c in cfgs])
        ys = np.array([means[c] for c in cfgs])
        if len(xs) > 2:
            b, a = np.polyfit(xs, np.log10(ys), 1)
            r = np.corrcoef(xs, np.log10(ys))[0, 1]
            xs_s = np.linspace(xs.min(), xs.max(), 50)
            ys_s = 10 ** (b * xs_s + a)
            ax.plot(xs_s, ys_s, "--", color=color, lw=1.6, alpha=0.9, zorder=2)
            # place r inside axes, anchored to the right end but inset so it never clips
            x_r = xs_s[-1] - 0.012  # inset 0.012 in data units (~1/3 of x-range)
            y_r = 10 ** (b * x_r + a)
            # nudge slightly for linear vs nonlinear so they don't overlap the dots
            if mode == "original":
                y_r *= 1.6
                va = "bottom"
            else:
                y_r *= 0.85
                va = "top"
            ax.text(x_r, y_r, f"r = {r:+.2f}", color=color, fontsize=10,
                    ha="right", va=va,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.85", alpha=0.9))

    ax.set_yscale("log")
    ax.set_ylim(1e-6, 0.3)
    ax.set_xlim(3.545, 3.76)
    ax.set_xlabel("Validation loss (WikiText-2)")
    ax.set_ylabel("Mean $\\nu_{frac}$ (avg over 12 layers)")
    ax.set_title("GPT-2 Medium: query nonlinearity vs validation loss", pad=10)
    # grid behind data, subtle
    ax.grid(True, which="both", alpha=0.35, linestyle="--", linewidth=0.6)
    handles = [
        plt.Line2D([], [], marker="o", ls="", c=PALETTE["original"], markeredgecolor="white",
                   label="Linear queries"),
        plt.Line2D([], [], marker="o", ls="", c=PALETTE["residual_gelu"], markeredgecolor="white",
                   label="Nonlinear queries"),
        plt.Line2D([], [], marker="o", ls="", c="0.55", label="60k steps"),
        plt.Line2D([], [], marker="s", ls="", c="0.55", label="20k steps"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1),
              frameon=True, framealpha=0.95, edgecolor="0.7", borderpad=0.6, labelspacing=0.5)
    save_figure(fig, os.path.join(FIGDIR, "fig1_nufrac_vs_loss"))


def fig2(sweep):
    """Depth profile: median line + interquartile band per mode."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for mode, color in MODE_COLORS.items():
        layers, med, lo, hi = layer_stats(sweep, mode, "nu_frac")
        ax.fill_between(layers, lo, hi, color=color, alpha=0.2, lw=0,
                        zorder=1)
        ax.plot(layers, med, "-o", color=color, ms=5, lw=2.0, zorder=2,
                label=f"{'Linear' if mode == 'original' else 'Nonlinear'} queries (median)")

    ax.set_yscale("log")
    ax.set_ylim(1e-11, 0.3)
    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("$\\nu_{frac}$")
    ax.set_title("GPT-2 Medium: nonlinearity across layers\n"
                 "(line = median over configs, band = IQR)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True,
              framealpha=0.9, edgecolor="gray")
    save_figure(fig, os.path.join(FIGDIR, "fig2_depth_comparison"))


def fig3(sweep):
    """Rank of the nonlinear-excess covariance R_f (median + IQR per mode)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    for mode, color in MODE_COLORS.items():
        layers, med, lo, hi = layer_stats(sweep, mode, "rank")
        ax.fill_between(layers, lo, hi, color=color, alpha=0.2, lw=0,
                        zorder=1)
        ax.plot(layers, med, "-o", color=color, ms=5, lw=2.0, zorder=2,
                label=f"{'Linear' if mode == 'original' else 'Nonlinear'} queries (median)")

    ax.set_xticks(layers)
    ax.set_xlabel("Layer")
    ax.set_ylabel("rank($R_f$) @ 1e-6 (holdout)")
    ax.set_ylim(0, 800)
    ax.set_title("GPT-2 Medium: effective rank of the residual covariance $R_f$\n"
                 "(linear-mode ranks sit at the estimation noise floor)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True,
              framealpha=0.9, edgecolor="gray")
    save_figure(fig, os.path.join(FIGDIR, "fig3_rank_analysis"))


def report_stats(sweep, val_loss, means):
    print()
    print("=" * 70)
    print("RECOMPUTED SUMMARY STATISTICS (outlier row + diverged config excluded)")
    print("=" * 70)

    for mode in ("original", "residual_gelu"):
        vals = np.array([r["nu_frac"] for r in sweep if r["mode"] == mode])
        print(f"\n{mode}: n={len(vals)}")
        print(f"  mean = {vals.mean():.6f}   median = {np.median(vals):.6f} "
              f"  min = {vals.min():.3e}   max = {vals.max():.6f}")

    orig = np.array([r["nu_frac"] for r in sweep if r["mode"] == "original"])
    res = np.array([r["nu_frac"] for r in sweep if r["mode"] == "residual_gelu"])
    print(f"\nSeparation (mean, clean data): {res.mean() / orig.mean():.1f}x")
    print(f"Separation (median): {np.median(res) / max(np.median(orig), 1e-12):.1f}x "
          f"(median original = {np.median(orig):.3e})")

    print("\nPer-layer medians (nu_frac, clean data):")
    print("  layer   original_med  original_mean  res_gelu_med  res_gelu_mean")
    for L in sorted(set(r["layer"] for r in sweep)):
        om = np.median([r["nu_frac"] for r in sweep if r["mode"] == "original" and r["layer"] == L])
        oa = np.mean([r["nu_frac"] for r in sweep if r["mode"] == "original" and r["layer"] == L])
        rm = np.median([r["nu_frac"] for r in sweep if r["mode"] == "residual_gelu" and r["layer"] == L])
        ra = np.mean([r["nu_frac"] for r in sweep if r["mode"] == "residual_gelu" and r["layer"] == L])
        print(f"  {L:5d}   {om:.3e}   {oa:.3e}   {rm:.6f}   {ra:.6f}")

    for mode in ("original", "residual_gelu"):
        cfgs = [c for c in means if c.startswith("tiedw_" + mode)]
        xs = np.array([val_loss[c] for c in cfgs])
        ys = np.array([means[c] for c in cfgs])
        print(f"\ncorr(log nu_frac, val_loss) [{mode}]: "
              f"r = {np.corrcoef(xs, np.log10(ys))[0, 1]:+.3f}")

    print("\nBest-vs-best (single-pair comparison, val loss on WikiText-2 20-batch slice):")
    best_res = min((c for c in means if "residual_gelu" in c), key=lambda c: val_loss[c])
    best_lin = min((c for c in means if "original" in c), key=lambda c: val_loss[c])
    print(f"  best nonlinear = {best_res} ({val_loss[best_res]:.4f})")
    print(f"  best linear    = {best_lin} ({val_loss[best_lin]:.4f})")
    print(f"  gap = {val_loss[best_lin] - val_loss[best_res]:.4f} "
          f"({100 * (val_loss[best_res] / val_loss[best_lin] - 1):.2f}%)")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    sweep = clean_sweep()
    val_loss = load_val_loss()
    means = config_mean_nufrac(sweep)

    fig1(sweep, val_loss, means)
    fig2(sweep)
    fig3(sweep)
    report_stats(sweep, val_loss, means)


if __name__ == "__main__":
    main()