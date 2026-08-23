"""Publication-ready overhead figures (replaces notebook-generated 150 DPI versions).

Generates:
  figures/benchmark_correct_dims.png/.pdf  — W_Q bottleneck vs main MLP
  figures/overhead_vs_nufrac_corrected.png/.pdf — overhead vs nu_frac

Uses hardcoded benchmark timings from notebooks/kernel_fusion.ipynb
(GPU: RTX 3050 Laptop, B=4 T=512 C=768, 500 runs) so no GPU is required to
regenerate the figures. Apply publication style from plot_style.py.

Run:
    python scripts/make_overhead_figures.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
from plot_style import apply_style, save_figure, PALETTE

apply_style()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
CSV_PATH = os.path.join(ROOT, "checkpoint", "sweep_results.csv")

# Hardcoded benchmark from kernel_fusion.ipynb Cell 4 (RTX 3050, B=4 T=512 C=768)
# These are the numbers printed in the notebook, not re-benchmarked here.
BENCH = {
    "bottleneck_384": {"hidden": 384, "t_in": 0.3058, "t_gel": 0.3415, "t_out": 0.2950, "t_full": 0.7484, "factor": 1.21},
    "main_3072": {"hidden": 3072, "t_in": 2.3895, "t_gel": 2.6525, "t_out": 2.5021, "t_full": 5.2548, "factor": 8.48},
    "linear": {"t_full": 0.618},  # derived ~0.7484/1.21
}
# For the overhead_vs_nu_frac plot, use the corrected sweep table (mlp_mult 0.5 is 1.22x)
OVERHEAD_TABLE = [
    (0.500, 384, 0.7672, 0.6298, 1.22),
    (1.000, 768, 1.4020, 0.5902, 2.38),
    (2.000, 1536, 2.4840, 0.5809, 4.28),
    (3.000, 2304, 3.6292, 0.5957, 6.09),
    (4.000, 3072, 5.2671, 0.6790, 7.76),
    (4.330, 3325, 5.9992, 0.6247, 9.60),
    (4.500, 3456, 5.6690, 0.6043, 9.38),
    (4.625, 3552, 5.8089, 0.5921, 9.81),
    (4.750, 3648, 6.1678, 0.6124, 10.07),
]


def make_benchmark():
    """Fig: W_Q bottleneck (384) vs Main MLP (3072) — 300 DPI, palette, vector PDF."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Use constrained layout from plot_style (already enabled), but tighten for 3 panels
    fig.suptitle("W_Q Bottleneck (768→384→768) vs Main MLP (768→3072→768)", fontsize=14, fontweight="bold")

    configs = ["Linear W_Q\n(768→768)", "W_Q Bottleneck\n(768→384→768)", "Main MLP\n(768→3072→768)"]
    t_fulls = [BENCH["linear"]["t_full"], BENCH["bottleneck_384"]["t_full"], BENCH["main_3072"]["t_full"]]
    colors = [PALETTE["residual_gelu"], PALETTE["original"], "#CC79A7"]  # Wong + pink for MLP

    # ax1: full pass comparison
    ax = axes[0]
    bars = ax.bar(configs, t_fulls, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    for bar, val in zip(bars, t_fulls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, f"{val:.2f}ms", ha="center", fontsize=9)
    ax.set_ylabel("Time per call (ms)")
    ax.set_title("Full pass time")
    ax.grid(True, alpha=0.3, axis="y")

    # ax2: overhead factor
    ax = axes[1]
    factors = [1.0, BENCH["bottleneck_384"]["factor"], BENCH["main_3072"]["factor"]]
    bars = ax.bar(configs, factors, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    for bar, val in zip(bars, factors):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val:.2f}x", ha="center", fontsize=9)
    ax.set_ylabel("Overhead factor (× linear W_Q)")
    ax.set_title("Overhead factor")
    ax.grid(True, alpha=0.3, axis="y")

    # ax3: component breakdown stacked
    ax = axes[2]
    labels = ["W_Q Bottleneck\n(768→384→768)", "Main MLP\n(768→3072→768)"]
    ins = [BENCH["bottleneck_384"]["t_in"], BENCH["main_3072"]["t_in"]]
    gels = [BENCH["bottleneck_384"]["t_gel"], BENCH["main_3072"]["t_gel"]]
    outs = [BENCH["bottleneck_384"]["t_out"], BENCH["main_3072"]["t_out"]]
    x_pos = np.arange(len(labels))
    width = 0.4
    ax.bar(x_pos, ins, width, label="matmul in", color=PALETTE["original"], edgecolor="white")
    ax.bar(x_pos, gels, width, bottom=ins, label="GELU", color=PALETTE["residual_gelu"], edgecolor="white")
    ax.bar(x_pos, outs, width, bottom=[i + g for i, g in zip(ins, gels)], label="matmul out", color="#CC79A7", edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Component breakdown")
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="gray")
    ax.grid(True, alpha=0.3, axis="y")

    save_figure(fig, os.path.join(FIGDIR, "benchmark_correct_dims"))


def make_overhead_vs_nufrac():
    """Fig: overhead factor vs mlp_mult + wall-clock + nu_frac spread — 300 DPI."""
    df = pd.read_csv(CSV_PATH)
    rg = (
        df[df["config"].str.contains("residual_gelu")]
        .groupby("config")
        .agg(mean_nu_frac=("nu_frac", "mean"), std_nu_frac=("nu_frac", "std"))
        .reset_index()
        .sort_values("mean_nu_frac")
    )

    def shorten(name):
        name = name.replace("tiedw_residual_gelu_", "")
        name = name.replace("tiedw_residual_gelu", "base")
        return name

    df_bench = pd.DataFrame(OVERHEAD_TABLE, columns=["mlp_mult", "hidden", "bottleneck_ms", "linear_ms", "overhead_factor"])
    actual_overhead = df_bench[df_bench["mlp_mult"] == 0.5]["overhead_factor"].values[0]
    actual_ms = df_bench[df_bench["mlp_mult"] == 0.5]["bottleneck_ms"].values[0]

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])
    fig.suptitle("Overhead vs Geometric Return Analysis (corrected dims)", fontsize=14, fontweight="bold", y=0.98)

    # ax1: overhead factor vs mlp_mult
    ax1.plot(df_bench["mlp_mult"], df_bench["overhead_factor"], "o-", color="#CC79A7", lw=2, ms=7, zorder=3)
    ax1.axvline(0.5, color=PALETTE["original"], ls="--", lw=1.5, label=f"actual W_Q bottleneck (hidden=384, {actual_overhead:.2f}x)")
    ax1.set_xlabel("MLP multiplier (hidden = mult × C)")
    ax1.set_ylabel("Overhead factor (× linear W_Q)")
    ax1.set_title("Overhead factor vs hidden size")
    ax1.legend(fontsize=9, framealpha=0.9, edgecolor="gray")
    ax1.grid(True, alpha=0.3)
    for _, row in df_bench.iterrows():
        ax1.annotate(f"{row['overhead_factor']:.1f}x", (row["mlp_mult"], row["overhead_factor"]), textcoords="offset points", xytext=(6, 4), fontsize=8, color="#CC79A7")

    # ax2: wall-clock time
    ax2.plot(df_bench["mlp_mult"], df_bench["bottleneck_ms"], "o-", color=PALETTE["original"], lw=2, ms=7, label="bottleneck MLP")
    ax2.axhline(df_bench["linear_ms"].iloc[0], color=PALETTE["residual_gelu"], ls="--", lw=1.5, label=f"linear W_Q ({df_bench['linear_ms'].iloc[0]:.2f} ms)")
    ax2.axvline(0.5, color=PALETTE["original"], ls=":", lw=1.2, alpha=0.6)
    ax2.set_xlabel("MLP multiplier")
    ax2.set_ylabel("Time per call (ms)")
    ax2.set_title("Wall-clock time vs hidden size")
    ax2.legend(fontsize=9, framealpha=0.9, edgecolor="gray")
    ax2.grid(True, alpha=0.3)

    # ax3: nu_frac across residual_gelu configs
    labels = [shorten(c) for c in rg["config"]]
    x_pos = np.arange(len(labels))
    ax3.bar(x_pos, rg["mean_nu_frac"], yerr=rg["std_nu_frac"], capsize=3, color=PALETTE["original"], alpha=0.25, edgecolor=PALETTE["original"], linewidth=0.8, error_kw={"elinewidth": 1, "ecolor": PALETTE["original"]})
    # highlight min/max
    ax3.bar(x_pos[0], rg["mean_nu_frac"].iloc[0], color="#CC79A7", alpha=0.4, edgecolor="#CC79A7", lw=1.2)
    ax3.bar(x_pos[-1], rg["mean_nu_frac"].iloc[-1], color=PALETTE["residual_gelu"], alpha=0.4, edgecolor=PALETTE["residual_gelu"], lw=1.2)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax3.set_ylabel("Mean $\\nu_{frac}$ (avg over 12 layers)")
    ax3.set_title(f"All residual_gelu configs share the same {actual_overhead:.2f}x overhead (768→384→768) but $\\nu_{{frac}}$ varies {rg['mean_nu_frac'].max()/rg['mean_nu_frac'].min():.1f}x from lr/wd", fontsize=11)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.annotate(f"min {rg['mean_nu_frac'].iloc[0]:.3f}", (x_pos[0], rg["mean_nu_frac"].iloc[0]), xytext=(x_pos[0] + 0.5, rg["mean_nu_frac"].iloc[0] + 0.005), fontsize=8, color="#CC79A7")
    ax3.annotate(f"max {rg['mean_nu_frac'].iloc[-1]:.3f}", (x_pos[-1], rg["mean_nu_frac"].iloc[-1]), xytext=(x_pos[-1] - 5, rg["mean_nu_frac"].iloc[-1] + 0.005), fontsize=8, color=PALETTE["residual_gelu"])

    save_figure(fig, os.path.join(FIGDIR, "overhead_vs_nufrac_corrected"))


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    make_benchmark()
    make_overhead_vs_nufrac()
    print("done - benchmark_correct_dims and overhead_vs_nufrac_corrected regenerated at 300 DPI + PDF")
