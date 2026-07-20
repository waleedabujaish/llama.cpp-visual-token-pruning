"""Figure 1: TTFT_llm speedup vs. keep-ratio, all three platforms.

Reads: results/p2_sweep_analysis.json, results/p2_sweep_x86_analysis.json,
       results/p2_sweep_kaggle_gpu_analysis.json
Writes: figures/fig1_speedup.pdf

Speedup = ttft_llm(keep=1.0) / ttft_llm(keep). Error bars are first-order
propagation of the per-cell run-to-run std through the ratio:
sigma_s = s * sqrt((s1/t1)^2 + (sk/tk)^2).
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vtp_style as S

S.apply_style()

fig, ax = plt.subplots(figsize=(3.4, 2.5))

for key, p in S.PLATFORMS.items():
    t = S.analysis_table(key)
    base = t[1.0]
    t1, s1 = base["ttft_llm_ms_mean"], base["ttft_llm_ms_std"]
    xs, ys, es = [], [], []
    for k in S.KEEP_MAIN:
        tk, sk = t[k]["ttft_llm_ms_mean"], t[k]["ttft_llm_ms_std"]
        s = t1 / tk
        xs.append(k)
        ys.append(s)
        es.append(s * math.sqrt((s1 / t1) ** 2 + (sk / tk) ** 2))
    ax.errorbar(xs, ys, yerr=es, color=p["color"], marker=p["marker"],
                label=p["label"], capsize=2, elinewidth=0.7, markeredgewidth=0.7)

ax.axhline(1.0, color="#999999", linewidth=0.6, linestyle=":", zorder=0)
S.keep_axis(ax, S.KEEP_MAIN)
ax.set_ylabel(r"TTFT$_{\mathrm{llm}}$ speedup ($\times$)")
ax.set_ylim(bottom=0.8)
ax.grid(axis="y")
ax.legend(loc="upper left")

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig1_speedup.pdf"))
