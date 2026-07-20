"""Figure 2: fraction of the theoretical linear-token-count ceiling realized
at each keep-ratio, all three platforms (the core H1 result).

Reads: results/p2_sweep_analysis.json, results/p2_sweep_x86_analysis.json,
       results/p2_sweep_kaggle_gpu_analysis.json
       (field: table[].fraction_of_theoretical_ceiling_h1, as committed by
        scripts/sweep_analysis.py: achieved time saving / theoretical saving)
Writes: figures/fig2_ceiling_fraction.pdf

Error bars propagate per-cell run-to-run std through
f = (t1 - tk) / (t1 - theo_k) with theo_k treated as exact (it is derived
from token counts, not measured).
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

KEEPS = [k for k in S.KEEP_MAIN if k != 1.0]  # keep=1.0 has no defined fraction

fig, ax = plt.subplots(figsize=(3.4, 2.5))

for key, p in S.PLATFORMS.items():
    t = S.analysis_table(key)
    t1, s1 = t[1.0]["ttft_llm_ms_mean"], t[1.0]["ttft_llm_ms_std"]
    xs, ys, es = [], [], []
    for k in KEEPS:
        row = t[k]
        f = row["fraction_of_theoretical_ceiling_h1"]
        tk, sk = row["ttft_llm_ms_mean"], row["ttft_llm_ms_std"]
        theo = row["theoretical_ttft_llm_ms"]
        denom = t1 - theo
        df_dtk = -1.0 / denom
        df_dt1 = (tk - theo) / denom**2
        sig = math.sqrt((df_dt1 * s1) ** 2 + (df_dtk * sk) ** 2)
        xs.append(k)
        ys.append(100.0 * f)
        es.append(100.0 * sig)
    ax.errorbar(xs, ys, yerr=es, color=p["color"], marker=p["marker"],
                label=p["label"], capsize=2, elinewidth=0.7, markeredgewidth=0.7)

ax.axhline(100.0, color="#999999", linewidth=0.6, linestyle="--", zorder=0)
S.keep_axis(ax, KEEPS)
ax.set_ylabel("Fraction of theoretical ceiling (%)")
ax.set_ylim(0, 108)
ax.grid(axis="y")
ax.legend(loc="lower left")

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig2_ceiling_fraction.pdf"))
