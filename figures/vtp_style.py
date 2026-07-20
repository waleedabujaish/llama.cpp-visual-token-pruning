"""Shared plotting style for all paper figures.

Every figure script imports this module so the platform -> color/marker
mapping and typography stay identical across the whole set. Colors are the
Okabe-Ito colorblind-safe subset; the mapping is fixed and must not be
reassigned per-figure.
"""

import glob
import json
import os

import matplotlib as mpl

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(REPO, "results")
RAW = os.path.join(RESULTS, "raw")

# Fixed platform identity: same color + marker in every figure.
PLATFORMS = {
    "m4": dict(color="#0072B2", marker="o", label="Apple M4 Pro (ARM CPU)"),
    "x86": dict(color="#D55E00", marker="s", label="Xeon 8573C (x86 CPU)"),
    "gpu": dict(color="#009E73", marker="^", label="Tesla P100 (GPU)"),
}

# Neutral ink for single-platform / non-platform series (fig 3, fig 6).
INK = "#1a1a1a"
GRAY_LIGHT = "#b8b8b8"
GRAY_DARK = "#5a5a5a"

# The main benchmark grid shared by all three platforms.
KEEP_MAIN = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]
# H2 extension grid (M4 and GPU only).
KEEP_EXT = [0.05, 0.03, 0.02, 0.015, 0.01]

ANALYSIS = {
    "m4": os.path.join(RESULTS, "p2_sweep_analysis.json"),
    "x86": os.path.join(RESULTS, "p2_sweep_x86_analysis.json"),
    "gpu": os.path.join(RESULTS, "p2_sweep_kaggle_gpu_analysis.json"),
}

_CELL_PREFIX = {
    "m4": "p2_sweep_keep",
    "x86": "p2_sweep_x86_keep",
    "gpu": "p2_sweep_kaggle_gpu_keep",
}


def apply_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "STIX Two Text", "Times New Roman", "Times"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.3,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "axes.grid": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.25,
        "grid.color": "#999999",
        "pdf.fonttype": 42,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def load_json(path):
    with open(path) as f:
        return json.load(f)


def analysis_table(platform):
    """Rows of the committed sweep-analysis table for one platform, keyed by keep."""
    d = load_json(ANALYSIS[platform])
    return {row["keep"]: row for row in d["table"]}


def _fmt_keep(keep):
    return "1" if keep == 1.0 else repr(keep)


def cell(platform, keep):
    """The raw per-cell benchmark JSON (runs[], aggregate{}) for one sweep cell."""
    pat = os.path.join(RESULTS, f"*_{_CELL_PREFIX[platform]}{_fmt_keep(keep)}.json")
    hits = sorted(glob.glob(pat))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected exactly 1 match for {pat}, got {hits}")
    return load_json(hits[0])


def keep_axis(ax, keeps, reverse=True):
    """Log-scaled keep-ratio x-axis, labeled in percent, aggressive pruning to the right."""
    ax.set_xscale("log")
    ax.set_xticks(keeps)
    ax.set_xticklabels([f"{k * 100:g}" for k in keeps])
    ax.minorticks_off()
    if reverse:
        ax.invert_xaxis()
    ax.set_xlabel("Visual tokens kept (%)")
