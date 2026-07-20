"""Figure 7: the extended sub-0.05 keep-ratio sweep (H2 region), CPU (M4) and
GPU (P100) side by side: per-run TTFT_vlm values, cell means, and +/-1 std.
Shows the two distinct signatures: run-to-run variance explodes on CPU while
GPU cells stay tight but the mean moves non-smoothly.

Reads: per-cell sweep JSONs (aggregate.ttft_vlm_ms mean/std/values):
       results/*_p2_sweep_keep{K}.json (M4) and
       results/*_p2_sweep_kaggle_gpu_keep{K}.json for
       K in {0.1, 0.05, 0.03, 0.02, 0.015, 0.01}
Writes: figures/fig7_h2_panels.pdf
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vtp_style as S

S.apply_style()

KEEPS = [0.1] + S.KEEP_EXT  # 0.1 anchors the still-tight regime on both platforms
PANELS = [("m4", "(a)"), ("gpu", "(b)")]

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.4))

for ax, (key, tag) in zip(axes, PANELS):
    p = S.PLATFORMS[key]
    xs, means, stds = [], [], []
    for k in KEEPS:
        agg = S.cell(key, k)["aggregate"]["ttft_vlm_ms"]
        assert agg["n"] == len(agg["values"])
        ax.plot([k] * len(agg["values"]), agg["values"], linestyle="none",
                marker=p["marker"], markersize=3.2, markerfacecolor="none",
                markeredgecolor=p["color"], markeredgewidth=0.6, alpha=0.65)
        xs.append(k)
        means.append(agg["mean"])
        stds.append(agg["std"])
    ax.errorbar(xs, means, yerr=stds, color=p["color"], marker=p["marker"],
                markersize=4.5, capsize=2.5, elinewidth=0.8, linewidth=1.1)
    S.keep_axis(ax, KEEPS)
    ax.tick_params(axis="x", labelsize=7)
    ax.text(0.03, 0.97, f"{tag} {p['label']}", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=S.INK)
    ax.grid(axis="y")

axes[0].set_ylabel(r"TTFT$_{\mathrm{vlm}}$ (ms)")
fig.tight_layout(w_pad=1.6)

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig7_h2_panels.pdf"))
