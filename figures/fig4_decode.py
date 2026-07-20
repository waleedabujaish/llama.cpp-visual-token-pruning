"""Figure 4: decode speed (tok/s) vs. keep-ratio, all three platforms as
small multiples (native units; absolute scales differ by ~10x, so one shared
axis would flatten the CPU platforms' trends).

Reads: per-cell sweep JSONs (aggregate.decode_tok_per_s mean/std):
       results/*_p2_sweep_keep{K}.json, *_p2_sweep_x86_keep{K}.json,
       *_p2_sweep_kaggle_gpu_keep{K}.json for K in the main 6-ratio grid
Writes: figures/fig4_decode.pdf
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vtp_style as S

S.apply_style()

fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.1))

for ax, (key, p) in zip(axes, S.PLATFORMS.items()):
    xs, ys, es = [], [], []
    for k in S.KEEP_MAIN:
        agg = S.cell(key, k)["aggregate"]["decode_tok_per_s"]
        xs.append(k)
        ys.append(agg["mean"])
        es.append(agg["std"])
    ax.errorbar(xs, ys, yerr=es, color=p["color"], marker=p["marker"],
                capsize=2, elinewidth=0.7, markeredgewidth=0.7)
    S.keep_axis(ax, S.KEEP_MAIN)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.text(0.04, 0.96, p["label"], transform=ax.transAxes,
            ha="left", va="top", fontsize=8, color=S.INK)
    lo, hi = min(ys), max(ys)
    pad = 0.35 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(axis="y")

axes[0].set_ylabel("Decode speed (tok/s)")
for ax in (axes[0], axes[2]):
    ax.set_xlabel("")
fig.tight_layout(w_pad=1.2)

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig4_decode.pdf"))
