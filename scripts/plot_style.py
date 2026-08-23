"""Global publication-ready plot style shared by every figure script.

Import at the top of each script:

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plot_style import apply_style, save_figure, PALETTE
    apply_style()

and finish every figure with:

    save_figure(fig, os.path.join(FIGDIR, "fig1_nufrac_vs_loss"))   # saves .png + .pdf

Design (applied globally, not per-plot):
  - White background (plt.style.use("default"))
  - 300 DPI output; single-column figures (7, 4.5), side-by-side panels (10, 5)
  - DejaVu Sans; title=13, axis labels=12, ticks=10, legend=10
  - Data lines lw=2.0; trend/reference lines lw=1.5 dashed; bands alpha=0.2
  - Markers: ms=6 scatter, ms=5 line markers
  - Subtle dashed gray grid behind data; no top/right spines
  - Legend with framealpha=0.9, edgecolor=gray, placed in the least-data-dense corner
  - Wong colorblind-safe palette
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

# Wong colorblind-safe palette
WONG = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

PALETTE = {
    "original": WONG[0],     # linear queries / seed 42
    "residual_gelu": WONG[1],  # nonlinear queries
    "seed42": WONG[0],
    "seed1": WONG[2],
    "valloss": WONG[3],
}

DIVERGED_CONFIG = "tiedw_original_005wd"
OUTLIER_CONFIG = "tiedw_original_mlp_4_625_times"


def apply_style():
    plt.style.use("default")
    mpl.rcParams.update({
        "figure.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        "grid.color": "gray",
        "axes.axisbelow": True,
        "figure.constrained_layout.use": True,
        "font.family": "DejaVu Sans",
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.framealpha": 0.9,
        "legend.edgecolor": "gray",
    })


def save_figure(fig, path_stem):
    """Save a figure as both .png and .pdf at 300 DPI (path_stem has no extension).

    bbox_inches="tight" expands the canvas so legends placed outside the axes
    (bbox_to_anchor) are never clipped.
    """
    for ext in ("png", "pdf"):
        fig.savefig(f"{path_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path_stem}.png / {path_stem}.pdf")