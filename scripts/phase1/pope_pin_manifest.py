#!/usr/bin/env python3
"""One-time: pin a deterministic, stratified subset of lmms-lab/POPE as
assets/phase1/pope<N>_manifest.jsonl, mirroring the TextVQA manifest's
role (metadata pin, images re-derived later by a separate materialize
script that cross-checks against this pin).

POPE's 9000-row test split (500 unique COCO images x 3 probe categories
x ~6 questions/category/image: random, popular, adversarial -- see
https://huggingface.co/datasets/lmms-lab/POPE) is too large to run a
full 6-ratio sweep against in one Kaggle session at TextVQA-comparable
per-sample cost (would be ~45x TextVQA's 200-sample scope). The three
categories probe different failure modes (adversarial specifically tests
whether the model hallucinates objects that are STATISTICALLY likely
given what else is in the image, not just any random absent object) --
collapsing to an unstratified first-N slice would risk badly
under/over-representing the hardest category depending on how the
dataset happens to be ordered, so this scripts explicitly reads each
row's `category` field and stops each category independently at
N_PER_CATEGORY, rather than assuming any particular row ordering.

Determinism: first N_PER_CATEGORY rows of each category encountered in
streaming order -- same "first N in streaming order" convention already
used for the TextVQA pin, same rationale (reproducible without needing
to fix a random seed against a dataset that streams in a fixed order).

No images are saved here -- see materialize_pope_images.py.
"""

import argparse
import datetime
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-category", type=int, default=100,
                    help="rows to keep per category (random/popular/adversarial); "
                         "default 100 x 3 = 300 total, same order of magnitude as "
                         "TextVQA's 200-sample pin")
    ap.add_argument("--out", default=None,
                    help="default: assets/phase1/pope<n-per-category*3>_manifest.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "assets" / "phase1" / f"pope{args.n_per_category * 3}_manifest.jsonl")

    from datasets import load_dataset
    print("[pin] streaming lmms-lab/POPE test split ...", flush=True)
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)

    kept = defaultdict(list)
    seen_categories = set()
    for row in ds:
        cat = row["category"]
        seen_categories.add(cat)
        if len(kept[cat]) >= args.n_per_category:
            if all(len(kept[c]) >= args.n_per_category for c in seen_categories) and len(seen_categories) >= 3:
                break
            continue
        kept[cat].append({
            "id": row["id"], "question_id": row["question_id"],
            "question": row["question"], "answer": row["answer"].strip().lower(),
            "image_source": row["image_source"], "category": cat,
        })

    counts = {c: len(v) for c, v in kept.items()}
    print(f"[pin] categories found: {sorted(seen_categories)}, counts: {counts}", flush=True)
    for c, v in kept.items():
        if len(v) < args.n_per_category:
            print(f"[pin] WARNING: category {c!r} only had {len(v)} rows available "
                  f"(< requested {args.n_per_category}) before the stream was exhausted "
                  f"or another stop condition hit -- check counts above are as expected", flush=True)

    all_rows = [r for cat_rows in kept.values() for r in cat_rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r) for r in all_rows) + "\n")

    manifest_meta = {
        "source": "lmms-lab/POPE, test split, streaming order",
        "pinned": datetime.datetime.now().strftime("%Y-%m-%d"),
        "n_per_category": args.n_per_category,
        "n_total": len(all_rows),
        "categories": sorted(kept.keys()),
        "counts": counts,
        "selection": "first N_PER_CATEGORY rows of each category in streaming order",
    }
    print(f"[pin] wrote {len(all_rows)} rows to {out_path}")
    print(json.dumps(manifest_meta, indent=2))


if __name__ == "__main__":
    main()
