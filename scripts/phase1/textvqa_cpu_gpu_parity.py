#!/usr/bin/env python3
"""CPU-vs-GPU accuracy parity spot-check: aggregate score parity is not the
same question as generation parity. The single-image latency test showed
byte-identical output across M4/x86/GPU, but that was one image at temp=0
with a short prompt -- it doesn't establish that the two backends' kernel
differences (different reduction order, different fused ops, etc.) produce
the same generations across a systematic 200-sample set, only that they
didn't on that one sample.

Compares a CPU run's textvqa_keep_sweep.py output against the existing GPU
run's (results/*_p3_textvqa_kaggle_gpu_keep*_summary.json +
results/raw/p3_textvqa_kaggle_gpu_keep*.preds.jsonl), per keep ratio:
  - aggregate accuracy CPU vs GPU (already visible in the two summary
    JSONs, restated here for a single side-by-side table)
  - per-sample TEXT agreement: exact string match, and a "near match" via
    the SAME normalization textvqa_sim.vqa_normalize already uses for
    scoring (lowercase, strip punctuation, number-word/contraction/article
    normalization) -- reusing established code rather than inventing a new
    fuzzy-match scheme
  - correctness DIVERGENCE: samples where CPU and GPU disagree on
    correctness (one right, one wrong) even though aggregate scores might
    be similar or identical -- the thing aggregate parity alone can't see.
    "Correct" is a binarization of the continuous soft-VQA score at a 0.5
    threshold (matches the common VQA-accuracy convention); every
    divergent sample is listed with both predictions and both raw scores,
    not just counted, so a human can actually look at what happened.

No new numbers are computed for accuracy itself -- this only compares the
two already-scored preds.jsonl files sample-by-sample.
"""

import argparse
import datetime
import glob
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.insert(0, str(Path(__file__).parent))
from textvqa_sim import vqa_normalize  # noqa: E402


def load_preds(path: Path) -> dict:
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return {r["i"]: r for r in recs if "acc" in r}


def find_preds(tag_prefix: str, keep: float) -> Path:
    matches = sorted(glob.glob(str(REPO_ROOT / "results" / "raw" / f"{tag_prefix}_keep{keep:g}.preds.jsonl")))
    if not matches:
        raise SystemExit(f"no preds.jsonl found for {tag_prefix} keep={keep:g} "
                          f"(looked for results/raw/{tag_prefix}_keep{keep:g}.preds.jsonl)")
    return Path(matches[0])


def find_summary(tag_prefix: str, keep: float) -> dict:
    matches = sorted(glob.glob(str(REPO_ROOT / "results" / f"*_{tag_prefix}_keep{keep:g}_summary.json")))
    if not matches:
        raise SystemExit(f"no summary JSON found for {tag_prefix} keep={keep:g}")
    return json.loads(Path(matches[-1]).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-tag-prefix", required=True,
                    help="e.g. p3_textvqa_m4_cpu -- the tag-prefix used for the local CPU "
                         "textvqa_keep_sweep.py run")
    ap.add_argument("--gpu-tag-prefix", default="p3_textvqa_kaggle_gpu",
                    help="already-committed GPU run's tag-prefix")
    ap.add_argument("--ratios", default="1.0,0.75,0.5,0.25,0.1,0.05")
    ap.add_argument("--correct-threshold", type=float, default=0.5,
                    help="soft-VQA score >= this counts as 'correct' for the divergence check")
    ap.add_argument("--tag", default="p3_textvqa_cpu_gpu_parity")
    args = ap.parse_args()

    ratios = [float(r) for r in args.ratios.split(",")]
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    per_ratio = []
    all_divergent = []
    print(f"{'keep':>7} {'cpu_acc':>9} {'gpu_acc':>9} {'diff':>8} {'exact_match':>12} "
          f"{'near_match':>11} {'divergent':>10}")
    for r in ratios:
        cpu_summary = find_summary(args.cpu_tag_prefix, r)
        gpu_summary = find_summary(args.gpu_tag_prefix, r)
        cpu_preds = load_preds(find_preds(args.cpu_tag_prefix, r))
        gpu_preds = load_preds(find_preds(args.gpu_tag_prefix, r))

        common = sorted(set(cpu_preds) & set(gpu_preds))
        if len(common) != len(cpu_preds) or len(common) != len(gpu_preds):
            print(f"[parity] WARNING keep={r:g}: cpu has {len(cpu_preds)} scored samples, "
                  f"gpu has {len(gpu_preds)}, {len(common)} in common -- comparing only the "
                  f"overlap, this is itself worth noting if it's not the full 200")

        exact_matches, near_matches, divergent = [], [], []
        for i in common:
            cp, gp = cpu_preds[i]["pred"], gpu_preds[i]["pred"]
            ca, ga = cpu_preds[i]["acc"], gpu_preds[i]["acc"]
            exact = cp.strip() == gp.strip()
            near = vqa_normalize(cp) == vqa_normalize(gp)
            cpu_correct = ca >= args.correct_threshold
            gpu_correct = ga >= args.correct_threshold
            exact_matches.append(exact)
            near_matches.append(near)
            if cpu_correct != gpu_correct:
                divergent.append({
                    "i": i, "question": cpu_preds[i].get("question"),
                    "cpu_pred": cp, "gpu_pred": gp,
                    "cpu_acc": ca, "gpu_acc": ga,
                    "cpu_correct": cpu_correct, "gpu_correct": gpu_correct,
                    "exact_text_match": exact, "near_text_match": near,
                })

        row = {
            "keep": r, "n_common": len(common),
            "cpu_acc_mean": cpu_summary["acc_mean"], "gpu_acc_mean": gpu_summary["acc_mean"],
            "acc_diff_gpu_minus_cpu": gpu_summary["acc_mean"] - cpu_summary["acc_mean"],
            "exact_match_rate": float(np.mean(exact_matches)) if exact_matches else None,
            "near_match_rate": float(np.mean(near_matches)) if near_matches else None,
            "n_correctness_divergent": len(divergent),
            "correctness_divergence_rate": len(divergent) / len(common) if common else None,
            "divergent_samples": divergent,
        }
        per_ratio.append(row)
        all_divergent.extend([{**d, "keep": r} for d in divergent])

        print(f"{r:>7.3f} {cpu_summary['acc_mean']*100:>8.1f}% {gpu_summary['acc_mean']*100:>8.1f}% "
              f"{row['acc_diff_gpu_minus_cpu']*100:>+7.2f}pp {row['exact_match_rate']*100:>11.1f}% "
              f"{row['near_match_rate']*100:>10.1f}% {row['n_correctness_divergent']:>10d}")

    out = {
        "tag": args.tag, "timestamp": ts,
        "cpu_tag_prefix": args.cpu_tag_prefix, "gpu_tag_prefix": args.gpu_tag_prefix,
        "correct_threshold": args.correct_threshold,
        "method": "exact_text_match = pred.strip() equality; near_text_match = "
                  "textvqa_sim.vqa_normalize(pred) equality (same normalization used for "
                  "scoring); correctness_divergent = binarized (>=threshold) acc disagrees "
                  "between CPU and GPU on the same sample/ratio, regardless of aggregate "
                  "score similarity",
        "per_ratio": per_ratio,
    }
    out_path = REPO_ROOT / "results" / f"{ts}_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[parity] wrote {out_path}")
    print(f"[parity] {len(all_divergent)} total correctness-divergent samples across all ratios")


if __name__ == "__main__":
    main()
