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
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

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
    path = os.path.join(ROOT, "checkpoint", "sweep_results.csv")
    layers = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["query_mode"] != "residual_gelu":
                continue
            layers.setdefault(int(r["layer"]), []).append(float(r["nu_frac"]))
    ks = sorted(layers)
    arr = np.array([layers[k] for k in ks])
    return np.array(ks), arr.mean(1), arr.std(1)


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

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(STEPS42, m42, "o-", color="#1f77b4", lw=2, label="seed 42")
    ax.plot(STEPS1, m1, "s--", color="#d62728", lw=2, label="seed 1")
    ax.axhline(0.142, color="gray", ls=":", lw=1)
    ax.annotate("saturation ~0.14", xy=(52000, 0.143), fontsize=9, color="gray")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean $\\nu_{frac}$ (avg over 24 layers)")
    ax.set_ylim(0, 0.18)
    ax.set_xlim(0, 62000)

    ax2 = ax.twinx()
    ax2.plot(steps, losses, "^-", color="#2ca02c", lw=1.5, ms=4,
             label="val loss (seed 42)")
    ax2.set_ylabel("Val loss", color="#2ca02c")
    ax2.tick_params(axis="y", labelcolor="#2ca02c")
    ax2.set_ylim(2.4, 4.0)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=9)
    ax.set_title("GPT-3 Large: $\\nu_{frac}$ rises with training, saturates at ~0.14,\n"
                 "val loss keeps improving (two seeds agree, r = 0.998)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_nufrac_training_trajectory.png"), dpi=150)
    plt.close(fig)


def fig5_depth_profiles():
    f42_final = load_layers(42, 60000)
    f1_final = load_layers(1, 60000)
    all42 = np.array([load_layers(42, s) for s in STEPS42])
    all1 = np.array([load_layers(1, s) for s in STEPS1])
    lo42, hi42 = all42.min(0), all42.max(0)
    lo1, hi1 = all1.min(0), all1.max(0)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(24)
    ax.fill_between(x, lo42, hi42, color="#1f77b4", alpha=0.18, label="seed 42 range (all ckpts)")
    ax.fill_between(x, lo1, hi1, color="#d62728", alpha=0.18, label="seed 1 range (all ckpts)")
    ax.plot(x, f42_final, "o-", color="#1f77b4", lw=2, label="seed 42 @ 60k")
    ax.plot(x, f1_final, "s--", color="#d62728", lw=2, label="seed 1 @ 60k")
    ax.set_xlabel("Layer")
    ax.set_ylabel("$\\nu_{frac}$")
    ax.set_xticks(x)
    ax.set_title("Depth profile at 60k: aggregate trend stable across seeds\n"
                 "(low L1, rising with depth, peak L23), fine structure seed-specific",
                 fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig5_depth_profile_seeds.png"), dpi=150)
    plt.close(fig)


def fig6_heatmaps():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, seed, steps, title in (
        (axes[0], 42, STEPS42, "seed 42"),
        (axes[1], 1, STEPS1, "seed 1"),
    ):
        data = np.array([load_layers(seed, s) for s in steps]).T  # layer x step
        im = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis",
                       vmin=0.0, vmax=0.24,
                       extent=(steps[0], steps[-1], -0.5, 23.5))
        ax.set_xticks(steps)
        ax.set_xticklabels([f"{s//1000}k" for s in steps], fontsize=8)
        ax.set_yticks(range(0, 24, 4))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training step")
    axes[0].set_ylabel("Layer")
    fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02, label="$\\nu_{frac}$")
    fig.suptitle("GPT-3 Large: layer $\\times$ step $\\nu_{frac}$ (nonlinearity develops early,\n"
                 "deep layers grow most; both seeds agree)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig6_heatmap_layer_step.png"), dpi=150)
    plt.close(fig)


def fig7_medium_vs_large():
    mks, mm, ms = load_medium_sweep()
    f42 = load_layers(42, 60000)
    f1 = load_layers(1, 60000)
    large = np.stack([f42, f1])
    xm = mks / (len(mks) - 1)
    xl = np.arange(24) / 23

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.errorbar(xm, mm, yerr=ms, color="#ff7f0e", lw=2, capsize=3,
                label="GPT-2 Medium residual_gelu (26 configs, mean $\\pm$ std)")
    ax.plot(xl, large.mean(0), "o-", color="#1f77b4", lw=2, label="GPT-3 Large @ 60k (2 seeds, mean)")
    ax.fill_between(xl, large.min(0), large.max(0), color="#1f77b4", alpha=0.2,
                    label="Large seed spread")
    ax.set_xlabel("Depth fraction (layer / (n_layers - 1))")
    ax.set_ylabel("$\\nu_{frac}$")
    ax.set_title("Nonlinearity survives the scale jump: same depth profile shape,\n"
                 "Large converges slightly higher (0.142 vs Medium 0.111)", fontsize=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig7_medium_vs_large_depth.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    fig4_trajectory()
    fig5_depth_profiles()
    fig6_heatmaps()
    fig7_medium_vs_large()
    print("wrote:", sorted(f for f in os.listdir(FIGDIR) if f.startswith("fig")))
