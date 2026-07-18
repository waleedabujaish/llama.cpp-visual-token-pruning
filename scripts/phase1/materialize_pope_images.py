#!/usr/bin/env python3
"""Materialize the unique images the pinned POPE manifest references.

assets/phase1/pope300_manifest.jsonl (or whatever --n-per-category the pin
was built with, see pope_pin_manifest.py) pins question-level metadata by
(id, question_id); this script streams lmms-lab/POPE again and saves each
referenced image once (keyed by image_source -- multiple pinned questions
share the same underlying image, ~3x fewer images than questions since
POPE asks ~3 questions per image per category on average within a
category-limited subset), cross-checking the pinned question text at each
(id, question_id) match against the pin -- same drift-detection principle
as materialize_textvqa_images.py: a dataset-order/version drift gets
caught here rather than silently producing a mismatched image+question.

Needs the `datasets` package and network access (Kaggle: enable Internet
in notebook settings).
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None,
                    help="default: assets/phase1/pope300_manifest.jsonl")
    ap.add_argument("--out", required=True, help="output dir for <image_source>.png + meta.jsonl")
    args = ap.parse_args()

    manifest_path = Path(args.manifest) if args.manifest else (
        REPO_ROOT / "assets" / "phase1" / "pope300_manifest.jsonl")
    manifest = [json.loads(l) for l in manifest_path.read_text().splitlines() if l.strip()]
    by_qid = {(m["id"], m["question_id"]): m for m in manifest}
    needed_images = {m["image_source"] for m in manifest}
    print(f"[materialize] pinned manifest: {len(manifest)} questions, "
          f"{len(needed_images)} unique images needed", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.jsonl"

    done_images = {p.stem for p in out.glob("*.png")}
    done_meta_qids = set()
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                done_meta_qids.add((m["id"], m["question_id"]))
    if needed_images <= done_images and len(done_meta_qids) == len(manifest):
        print(f"[materialize] all {len(needed_images)} images + meta.jsonl already present "
              f"in {out}, skipping")
        return

    from datasets import load_dataset
    print("[materialize] streaming lmms-lab/POPE test split, matching pinned rows ...", flush=True)
    ds = load_dataset("lmms-lab/POPE", split="test", streaming=True)

    meta_f = open(meta_path, "a")
    n_matched = 0
    n_image_saved = len(done_images)
    for row in ds:
        key = (row["id"], row["question_id"])
        m = by_qid.get(key)
        if m is None:
            continue
        if row["question"] != m["question"]:
            raise SystemExit(
                f"dataset/manifest mismatch at (id={key[0]}, question_id={key[1]}): "
                f"streamed question {row['question']!r} != pinned {m['question']!r} -- "
                f"the dataset snapshot or streaming order has drifted since the manifest "
                f"was pinned; refusing to proceed on a silently mismatched image."
            )
        img_source = row["image_source"]
        img_path = out / f"{img_source}.png"
        if not img_path.exists():
            row["image"].convert("RGB").save(img_path)
            n_image_saved += 1
        if key not in done_meta_qids:
            meta_f.write(json.dumps(m) + "\n")
            meta_f.flush()
            done_meta_qids.add(key)
        n_matched += 1
        if n_matched % 50 == 0:
            print(f"[materialize] {n_matched}/{len(manifest)} questions matched, "
                  f"{n_image_saved} unique images saved", flush=True)
        if len(done_meta_qids) == len(manifest):
            break
    meta_f.close()
    if len(done_meta_qids) != len(manifest):
        raise SystemExit(f"only matched {len(done_meta_qids)}/{len(manifest)} pinned "
                          f"questions before the stream ended")
    print(f"[materialize] wrote {n_image_saved} unique images + meta.jsonl "
          f"({len(manifest)} questions) to {out}")


if __name__ == "__main__":
    main()
