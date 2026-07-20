"""Figure 3: encoder vs. LLM-prefill share of TTFT_vlm on the Phase 2
baseline (fixed build, cooled, n=6; unfixed same-session control below it).
Motivates why pruning targets prefill.

Reads: results/20260717-235645_g2_baseline_fixed_cooled.json
       results/20260718-000051_g2_baseline_unfixed_cooled.json
Writes: figures/fig3_baseline_breakdown.pdf
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vtp_style as S

S.apply_style()

BARS = [
    ("Unfixed control", os.path.join(S.RESULTS, "20260718-000051_g2_baseline_unfixed_cooled.json")),
    ("Fixed build", os.path.join(S.RESULTS, "20260717-235645_g2_baseline_fixed_cooled.json")),
]

fig, ax = plt.subplots(figsize=(3.4, 1.7))

labels = []
for i, (name, path) in enumerate(BARS):
    d = S.load_json(path)
    enc = d["aggregate"]["encode_ms"]
    pre = d["aggregate"]["prompt_eval_ms"]
    total = enc["mean"] + pre["mean"]
    ax.barh(i, enc["mean"], height=0.55, color=S.GRAY_LIGHT,
            edgecolor="white", linewidth=1.2)
    ax.barh(i, pre["mean"], left=enc["mean"], height=0.55, color=S.GRAY_DARK,
            edgecolor="white", linewidth=1.2)
    ax.text(enc["mean"] / 2, i, f"{100 * enc['mean'] / total:.1f}%",
            ha="center", va="center", fontsize=6.5, color=S.INK)
    ax.text(enc["mean"] + pre["mean"] / 2, i,
            f"{pre['mean']:.0f} ms  ({100 * pre['mean'] / total:.1f}%)",
            ha="center", va="center", fontsize=7, color="white")
    labels.append(name)
    if i == len(BARS) - 1:  # segment names above the top bar, no legend box
        ax.text(0, i + 0.52, "Vision encoder + projector", ha="left",
                va="bottom", fontsize=7.5, color=S.INK)
        ax.text(enc["mean"] + pre["mean"], i + 0.52, "LLM prefill",
                ha="right", va="bottom", fontsize=7.5, color=S.INK)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_ylim(-0.55, len(labels) - 0.05)
ax.set_xlabel(r"Time (ms), TTFT$_{\mathrm{vlm}}$ = encoder + LLM prefill")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig3_baseline_breakdown.pdf"))
