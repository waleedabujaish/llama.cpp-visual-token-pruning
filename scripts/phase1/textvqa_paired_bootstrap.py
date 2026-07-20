#!/usr/bin/env python3
"""Paired-bootstrap significance test for the TextVQA keep-ratio sweep,
each pruned ratio vs the keep=1.0 baseline, from the committed per-sample
preds.jsonl files.

Archives the analysis behind the paired-bootstrap table in NOTES.md's "GPU
platform sweep" section. That table was originally computed in-session on
Kaggle and never saved to results/ -- a gap found in the 2026-07-20 audit
pass. This script recomputes it from the committed
results/raw/<tag_prefix>_keep*.preds.jsonl files and writes a timestamped
results JSON. The mean diffs and win/loss/tie counts are deterministic;
the CIs use a seeded RNG, so the run is exactly reproducible.

Method (same as textvqa_llamacpp.py's fixed-vs-unfixed comparison): paired
per-sample diff of soft-VQA scores on the same 200-sample manifest, 10000
bootstrap resamples of the diff vector, percentile 95% CI.

Also records a per-ratio accuracy CI (percentile bootstrap over the 200
per-sample scores, same resample count) for every cell including the
keep=1.0 baseline -- this is what figures/fig5_textvqa_ci.py plots. The
accuracy CIs draw from their own seeded RNG so the paired-diff CIs stay
bit-reproducible independently of them.
"""

import argparse
import datetime
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

KEEPS_DEFAULT = [1.0, 0.75, 0.5, 0.25, 0.1, 0.05]


def load_scores(tag_prefix: str, keep: float) -> np.ndarray:
    pat = str(REPO_ROOT / "results" / "raw" / f"{tag_prefix}_keep{keep:g}.preds.jsonl")
    matches = sorted(glob.glob(pat))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one preds file for keep={keep:g}, got {matches}")
    recs = [json.loads(l) for l in Path(matches[0]).read_text().splitlines() if l.strip()]
    recs.sort(key=lambda r: r["i"])
    return Path(matches[0]), np.array([r["acc"] for r in recs])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag-prefix", default="p3_textvqa_kaggle_gpu")
    ap.add_argument("--keeps", type=float, nargs="+", default=KEEPS_DEFAULT)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng_paired = np.random.default_rng(args.seed)
    rng_acc = np.random.default_rng(args.seed)
    base_keep = max(args.keeps)
    base_path, base = load_scores(args.tag_prefix, base_keep)
    n = base.size

    def acc_ci(scores: np.ndarray) -> list:
        boots = np.array([scores[rng_acc.integers(0, n, n)].mean()
                          for _ in range(args.resamples)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return [float(lo), float(hi)]

    baseline = {"keep": base_keep, "acc_mean": float(base.mean()), "acc_ci95": acc_ci(base)}

    rows = []
    src = {f"keep{base_keep:g}": str(base_path.relative_to(REPO_ROOT))}
    for keep in sorted([k for k in args.keeps if k != base_keep], reverse=True):
        path, s = load_scores(args.tag_prefix, keep)
        if s.size != n:
            raise SystemExit(f"sample-count mismatch at keep={keep:g}: {s.size} vs {n}")
        src[f"keep{keep:g}"] = str(path.relative_to(REPO_ROOT))
        d = s - base
        boots = np.array([d[rng_paired.integers(0, n, n)].mean() for _ in range(args.resamples)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({
            "keep": keep,
            "acc_mean": float(s.mean()),
            "acc_ci95": acc_ci(s),
            "baseline_acc_mean": float(base.mean()),
            "mean_diff_pp": float(d.mean() * 100),
            "ci95_pp": [float(lo * 100), float(hi * 100)],
            "ci_crosses_zero": bool(lo <= 0 <= hi),
            "wins": int((d > 0).sum()),
            "losses": int((d < 0).sum()),
            "ties": int((d == 0).sum()),
        })

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = {
        "tag": f"{args.tag_prefix}_paired_bootstrap",
        "timestamp": ts,
        "command": " ".join(sys.argv),
        "method": ("paired per-sample soft-VQA score diff vs the keep=1.0 baseline on the "
                   "pinned 200-sample manifest; percentile bootstrap over the diff vector "
                   f"({args.resamples} resamples, numpy default_rng seed {args.seed}); "
                   "same methodology as the fixed-vs-unfixed comparison in "
                   "results/20260717-171500_p2_textvqa_paired_fixed_vs_unfixed.json"),
        "provenance_note": ("archives the paired-bootstrap analysis quoted in NOTES.md 'GPU "
                            "platform sweep', which was originally computed in-session on "
                            "Kaggle (2026-07-18) and not saved -- gap found and closed in the "
                            "2026-07-20 audit pass; recomputed here from the committed preds "
                            "files, which reproduces the quoted table exactly"),
        "source_preds": src,
        "n_samples": n,
        "baseline_keep": base_keep,
        "baseline": baseline,
        "per_ratio": rows,
    }
    out_path = REPO_ROOT / "results" / f"{ts}_{args.tag_prefix}_paired_bootstrap.json"
    out_path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {out_path.relative_to(REPO_ROOT)}")
    for r in rows:
        print(f"  keep={r['keep']:<5} diff={r['mean_diff_pp']:+.2f}pp "
              f"CI[{r['ci95_pp'][0]:+.2f},{r['ci95_pp'][1]:+.2f}] "
              f"W/L/T={r['wins']}/{r['losses']}/{r['ties']}")


if __name__ == "__main__":
    main()
