"""Figure 5: TextVQA accuracy vs. keep-ratio with 95% bootstrap CIs,
CPU (M4) and GPU overlaid.

Reads: results/*_p3_textvqa_m4_cpu_paired_bootstrap.json and
       results/*_p3_textvqa_kaggle_gpu_paired_bootstrap.json (the archived
       bootstrap analyses written by scripts/phase1/textvqa_paired_bootstrap.py:
       per-ratio acc_mean + acc_ci95, percentile bootstrap over the 200
       per-sample scores, 10000 resamples, seed 42), cross-checked against
       the committed results/*_summary.json acc_mean values.
Writes: figures/fig5_textvqa_ci.pdf
"""

import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import vtp_style as S

SERIES = [
    ("m4", "p3_textvqa_m4_cpu"),
    ("gpu", "p3_textvqa_kaggle_gpu"),
]

S.apply_style()

def bootstrap_archive(prefix):
    hits = glob.glob(os.path.join(S.RESULTS, f"*_{prefix}_paired_bootstrap.json"))
    assert len(hits) == 1, hits
    d = S.load_json(hits[0])
    rows = {d["baseline"]["keep"]: d["baseline"]}
    rows.update({r["keep"]: r for r in d["per_ratio"]})
    return rows

def summary_acc(prefix, keep):
    k = "1" if keep == 1.0 else repr(keep)
    hits = glob.glob(os.path.join(S.RESULTS, f"*_{prefix}_keep{k}_summary.json"))
    assert len(hits) == 1, hits
    return S.load_json(hits[0])["acc_mean"]

fig, ax = plt.subplots(figsize=(3.4, 2.5))

# small multiplicative x-dodge so the two series' error bars don't overlap
dodge = {"m4": 0.97, "gpu": 1.03}

for key, prefix in SERIES:
    p = S.PLATFORMS[key]
    rows = bootstrap_archive(prefix)
    xs, ys, lo, hi = [], [], [], []
    for k in S.KEEP_MAIN:
        acc = rows[k]["acc_mean"]
        ci_lo, ci_hi = rows[k]["acc_ci95"]
        assert abs(acc - summary_acc(prefix, k)) < 1e-9, (prefix, k)
        xs.append(k * dodge[key])
        ys.append(100 * acc)
        lo.append(100 * (acc - ci_lo))
        hi.append(100 * (ci_hi - acc))
    ax.errorbar(xs, ys, yerr=[lo, hi], color=p["color"], marker=p["marker"],
                label=p["label"], capsize=2.5, elinewidth=0.8, markeredgewidth=0.7)

S.keep_axis(ax, S.KEEP_MAIN)
ax.set_ylabel("TextVQA accuracy (%)")
ax.grid(axis="y")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
          columnspacing=1.2, handletextpad=0.5)

fig.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig5_textvqa_ci.pdf"))
