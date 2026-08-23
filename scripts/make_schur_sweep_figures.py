"""Generate figures for the GPT-3 Large Schur sweep (16 checkpoints, 2 seeds).

Reads schur_results/aws_dun_schur/results/*.csv, checkpoint/sweep_results.csv
(GPT-2 Medium), and share_dun_schur/checkpoint_losses.csv (val loss trajectory).

Outputs:
  figures/fig4_nufrac_training_trajectory.png
  figures/fig5_depth_profile_seeds.png
  figures/fig6_heatmap_layer_step.png
  figures/fig7_medium_vs_large_depth.png
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
RESULTS = os.path.join(ROOT, "schur_results", "aws_dun_schur", "results")
FIGDIR = os.path.join(ROOT, "figures")

STEPS42 = [5000, 10000, 20000, 30000, 40000, 50000, 60000]
STEPS1 = [1000, 2000, 5000, 10000, 20000, 30000, 40000, 50000, 60000]


def load_layers(seed, step):
    path = os.path.join(RESULTS, f"schur_seed{seed}_step{step}.csv")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    return np.array([float(r["nu_frac"]) for r in rows])


def load_medium_sweep():
    """GPT-2 Medium residual_gelu nu_frac per layer (outliers excluded,
    consistent with scripts/make_medium_figures.py)."""
    path = os.path.join(ROOT, "checkpoint", "sweep_results.csv")
    layers = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["query_mode"] != "residual_gelu":
                continue
            if r["config"] == DIVERGED_CONFIG:
                continue
            if r["config"] == OUTLIER_CONFIG and int(r["layer"]) == 3:
                continue
            layers.setdefault(int(r["layer"]), []).append(float(r["nu_frac"]))
    ks = sorted(layers)
    arr = np.array([layers[k] for k in ks])
    return np.array(ks), arr.mean(1), np.percentile(arr, [25, 75], axis=1)


def load_val_loss():
    path = os.path.join(ROOT, "share_dun_schur", "checkpoint_losses.csv")
    steps, losses = [], []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            steps.append(int(r["step"]))
            losses.append(float(r["val_loss"]))
    return np.array(steps), np.array(losses)


def fig4_trajectory():
    steps, losses = load_val_loss()
    m42 = np.array([load_layers(42, s).mean() for s in STEPS42])
    m1 = np.array([load_layers(1, s).mean() for s in STEPS1])

    # extra-wide so outside legend never overlaps; disable constrained_layout for manual control
    fig = plt.figure(figsize=(10.2, 4.9))
    fig.set_constrained_layout(False)
    # main axes 0.10-0.68, gap 0.68-0.76, legend at 0.76+
    ax = fig.add_axes([0.10, 0.15, 0.58, 0.72])
    ax2 = ax.twinx()
    # main nu_frac lines
    h1, = ax.plot(STEPS42, m42, "o-", color=PALETTE["seed42"], lw=2.2, ms=6,
                  label="seed 42", zorder=3, clip_on=False)
    h2, = ax.plot(STEPS1, m1, "s--", color=PALETTE["seed1"], lw=1.8, ms=6,
                  label="seed 1", zorder=3, clip_on=False)
    # reference line — not in legend, no label to avoid _child2
    ax.axhline(0.142, color="0.55", ls=":", lw=1.6, zorder=1)
    # centered annotation, inside axes but well away from top-right legend area
    ax.text(0.50, 0.88, "WikiText-2 mean = 0.142", transform=ax.transAxes,
            fontsize=10, color="0.35", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.95))
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean $\\nu_{frac}$ (avg over 24 layers)")
    ax.set_ylim(0, 0.18)
    ax.set_xlim(0, 62000)
    ax.set_xticks([0, 10000, 20000, 30000, 40000, 50000, 60000])
    ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.6, zorder=0)

    h3, = ax2.plot(steps, losses, "^-", color=PALETTE["valloss"], lw=1.8, ms=6,
                   label="val loss (seed 42)", zorder=2, clip_on=False)
    ax2.set_ylabel("Val loss", color=PALETTE["valloss"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["valloss"])
    ax2.set_ylim(2.4, 4.0)
    ax2.grid(False)

    # legend well away from data — 0.10 figure gap between axes and legend box
    for h in (h1, h2, h3):
        h.set_label(h.get_label())
    fig.legend(handles=[h1, h2, h3], loc="upper left", bbox_to_anchor=(0.76, 0.86),
               frameon=True, framealpha=0.98, edgecolor="0.7", borderpad=0.7, labelspacing=0.6,
               handlelength=1.8, fontsize=10, borderaxespad=0.)

    fig.text(0.5, 0.94,
             "GPT-3 Large: $\\nu_{frac}$ saturates, val loss keeps improving\n"
             "(two seeds agree, r = 0.998; $\\nu_{frac}$ on WikiText-2, val loss on OWT val split)",
             ha="center", va="center", fontsize=13, linespacing=1.4)
    # manual tight: keep legend outside, so disable bbox_inches tight cropping of legend
    fig.savefig(os.path.join(FIGDIR, "fig4_nufrac_training_trajectory.png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIGDIR, "fig4_nufrac_training_trajectory.pdf"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {os.path.join(FIGDIR, 'fig4_nufrac_training_trajectory.png')} / {os.path.join(FIGDIR, 'fig4_nufrac_training_trajectory.pdf')}")


def fig5_depth_profiles():
    f42_final = load_layers(42, 60000)
    f1_final = load_layers(1, 60000)
    all42 = np.array([load_layers(42, s) for s in STEPS42])
    all1 = np.array([load_layers(1, s) for s in STEPS1])
    lo42, hi42 = np.percentile(all42, [10, 90], axis=0)
    lo1, hi1 = np.percentile(all1, [10, 90], axis=0)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(24)
    # bands behind, lines on top
    ax.fill_between(x, lo42, hi42, color=PALETTE["seed42"], alpha=0.18, lw=0,
                    zorder=1)
    ax.fill_between(x, lo1, hi1, color=PALETTE["seed1"], alpha=0.18, lw=0,
                    zorder=1)
    # proxy handles for bands + lines in correct order
    from matplotlib.patches import Patch
    h_band42 = Patch(facecolor=PALETTE["seed42"], alpha=0.18, edgecolor="none", label="seed 42 (10-90% over ckpts)")
    h_band1 = Patch(facecolor=PALETTE["seed1"], alpha=0.18, edgecolor="none", label="seed 1 (10-90% over ckpts)")
    h_line42, = ax.plot(x, f42_final, "o-", color=PALETTE["seed42"], lw=2.2, ms=6,
                        markeredgecolor="white", markeredgewidth=0.7, zorder=3, label="seed 42 @ 60k")
    h_line1, = ax.plot(x, f1_final, "s--", color=PALETTE["seed1"], lw=1.8, ms=6,
                       markeredgecolor="white", markeredgewidth=0.7, zorder=3, label="seed 1 @ 60k")
    ax.set_xlabel("Layer")
    ax.set_ylabel("$\\nu_{frac}$")
    ax.set_xticks(np.arange(0, 24, 2))
    ax.set_xticks(np.arange(24), minor=True)
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0, 0.24)
    ax.set_title("GPT-3 Large: depth profile at 60k\n"
                 "(aggregate trend stable across seeds, fine structure seed-specific)", pad=12)
    ax.grid(True, alpha=0.35, linestyle="--", linewidth=0.6, axis="both", which="major")
    ax.grid(True, alpha=0.15, linestyle=":", linewidth=0.5, axis="x", which="minor")
    ax.legend(handles=[h_band42, h_band1, h_line42, h_line1],
              loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True,
              framealpha=0.95, edgecolor="0.7", borderpad=0.6, labelspacing=0.6, handletextpad=0.6)
    save_figure(fig, os.path.join(FIGDIR, "fig5_depth_profile_seeds"))


def fig6_heatmaps():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1],
                                          "wspace": 0.35})
    for ax, seed, steps, title in (
        (axes[0], 42, STEPS42, "seed 42"),
        (axes[1], 1, STEPS1, "seed 1"),
    ):
        data = np.array([load_layers(seed, s) for s in steps]).T  # layer x step
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis",
                       vmin=0.0, vmax=0.24,
                       extent=(steps[0], steps[-1], -0.5, 23.5))
        tick_steps = steps[::2] if len(steps) > 7 else steps
        ax.set_xticks(tick_steps)
        ax.set_xticklabels([f"{s // 1000}k" for s in tick_steps], fontsize=10)
        ax.set_yticks(range(0, 24, 4))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Training step")
    axes[0].set_ylabel("Layer")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, aspect=40,
                 label="$\\nu_{frac}$")
    fig.suptitle("GPT-3 Large: layer $\\times$ step $\\nu_{frac}$ (both seeds)",
                 fontsize=13, y=1.02)
    save_figure(fig, os.path.join(FIGDIR, "fig6_heatmap_layer_step"))


def fig7_medium_vs_large():
    mks, mm, [mlo, mhi] = load_medium_sweep()
    f42 = load_layers(42, 60000)
    f1 = load_layers(1, 60000)
    large = np.stack([f42, f1])
    xm = mks / (len(mks) - 1)
    xl = np.arange(24) / 23

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.fill_between(xm, mlo, mhi, color=PALETTE["residual_gelu"], alpha=0.2,
                    lw=0, zorder=1, label="GPT-2 Medium (IQR over 26 configs)")
    ax.plot(xm, mm, "-o", color=PALETTE["residual_gelu"], lw=2.0, ms=5,
            zorder=2, label="GPT-2 Medium (mean)")
    ax.plot(xl, large.mean(0), "o-", color=PALETTE["seed42"], lw=2.0, ms=5,
            zorder=2, label="GPT-3 Large @ 60k (mean of 2 seeds)")
    ax.fill_between(xl, large.min(0), large.max(0), color=PALETTE["seed42"],
                    alpha=0.2, lw=0, zorder=1, label="GPT-3 Large (seed range)")
    ax.set_xlabel("Depth fraction (layer / (n_layers - 1))")
    ax.set_ylabel("$\\nu_{frac}$")
    ax.set_title("Nonlinearity survives the scale jump (0.142 vs 0.111)\n"
                 "(Medium checkpoints at 50k steps, Large at 60k)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=True,
              framealpha=0.9, edgecolor="gray")
    save_figure(fig, os.path.join(FIGDIR, "fig7_medium_vs_large_depth"))


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    fig4_trajectory()
    fig5_depth_profiles()
    fig6_heatmaps()
    fig7_medium_vs_large()
    print("done")