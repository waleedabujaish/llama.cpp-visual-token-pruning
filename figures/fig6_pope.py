"""Figure 6: POPE accuracy / precision / recall vs. keep-ratio (Kaggle P100,
300 questions per cell).

Reads: results/*_p4_pope_kaggle_gpu_keep{K}_summary.json (overall block)
Writes: figures/fig6_pope.pdf

Single-platform figure: series are metrics, not platforms, so they are drawn
in neutral ink and distinguished by line style + marker + direct labels
(keeps the platform color mapping unambiguous across the figure set).
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vtp_style as S

S.apply_style()

METRICS = [
    ("accuracy", "Accuracy", "-", "o"),
    ("precision", "Precision", "--", "s"),
    ("recall", "Recall", ":", "^"),
]

def pope(keep):
    k = "1" if keep == 1.0 else repr(keep)
    hits = glob.glob(os.path.join(S.RESULTS, f"*_p4_pope_kaggle_gpu_keep{k}_summary.json"))
    assert len(hits) == 1, hits
    d = S.load_json(hits[0])
    assert d["overall"]["n"] == 300
    return d["overall"]

fig, ax = plt.subplots(figsize=(3.4, 2.5))

vals = {k: pope(k) for k in S.KEEP_MAIN}
for field, label, ls, marker in METRICS:
    ys = [100 * vals[k][field] for k in S.KEEP_MAIN]
    ax.plot(S.KEEP_MAIN, ys, color=S.INK, linestyle=ls, marker=marker,
            markerfacecolor="white", markeredgewidth=0.8)
    ax.annotate(label, (S.KEEP_MAIN[-1], ys[-1]), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=8, color=S.INK)

S.keep_axis(ax, S.KEEP_MAIN)
ax.set_ylabel("POPE metric (%)")
ax.grid(axis="y")
# room for the direct labels to the right of the last point
ax.set_xlim(right=ax.get_xlim()[1] * 0.82)

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig6_pope.pdf"))
