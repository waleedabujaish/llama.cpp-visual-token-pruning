#!/usr/bin/env python3
"""Materialize the 200 TextVQA images the pinned manifest references.

assets/phase1/textvqa200_manifest.jsonl records the exact 200 questions/
answers/OCR tokens (first 200 of lmms-lab/textvqa validation, streaming
order) that every C++ TextVQA run scores against -- see NOTES.md "Eval-set
pin". It intentionally does not include the images themselves (200 JPEGs
would bloat the repo); this script re-derives them by streaming the same
dataset split and saving each sample's image at the manifest's own index,
cross-checking the question text at each index against the pin so a silent
dataset-order/version drift would be caught rather than silently producing
a mismatched image+question pair.

Needs the `datasets` package and network access (on Kaggle: enable Internet
in notebook settings).

Resumable: already-materialized indices (present in both NNN.png and
meta.jsonl) are skipped on a re-run.

Known issue, worked around at the bottom of this file: the HF `datasets`
streaming client leaves a background thread that can race normal Python
interpreter shutdown after main() returns (observed on Kaggle's Linux
image as "Fatal Python error: PyGILState_Release: thread state ... must
be current when releasing", SIGABRT/exit -6 -- on macOS the same
underlying race instead manifested as the process simply never exiting).
Both are a nondeterministic teardown-ordering bug in the streaming
client, not a correctness issue -- by the time it can happen, every file
this script writes has already been flushed to disk. Sidestepped by
calling os._exit() immediately after a successful run, which terminates
the process before Python's normal interpreter-finalization sequence
(the exact code path where the race occurs) ever runs.
"""

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST = REPO_ROOT / "assets/phase1/textvqa200_manifest.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir for NNN.png + meta.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    n = len(manifest)
    by_i = {m["i"]: m for m in manifest}
    assert sorted(by_i) == list(range(n)), "manifest indices must be a dense 0..N-1 range"

    meta_path = out / "meta.jsonl"
    done = set()
    if meta_path.exists():
        for line in meta_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["i"])
    if len(done) == n and all((out / f"{i:03d}.png").exists() for i in range(n)):
        print(f"[materialize] all {n} images + meta.jsonl already present in {out}, skipping")
        return

    from datasets import load_dataset
    print(f"[materialize] streaming lmms-lab/textvqa validation, matching {n} pinned samples "
          f"({len(done)} already done) ...", flush=True)
    ds = load_dataset("lmms-lab/textvqa", split="validation", streaming=True)

    meta_f = open(meta_path, "a")
    n_done = 0
    for i, sample in enumerate(ds):
        if i >= n:
            break
        if i in done:
            n_done += 1
            continue
        m = by_i[i]
        if sample["question"] != m["question"]:
            raise SystemExit(
                f"dataset/manifest mismatch at i={i}: streamed question "
                f"{sample['question']!r} != pinned {m['question']!r} -- the dataset "
                f"snapshot or streaming order has drifted since the manifest was "
                f"pinned; refusing to proceed on a silently mismatched image."
            )
        img = sample["image"].convert("RGB")
        img.save(out / f"{i:03d}.png")
        meta_f.write(json.dumps(m) + "\n")
        meta_f.flush()
        n_done += 1
        if n_done % 25 == 0 or n_done == n:
            print(f"[materialize] {n_done}/{n}", flush=True)
    meta_f.close()
    if n_done != n:
        raise SystemExit(f"only matched {n_done}/{n} pinned samples before the stream ended")
    print(f"[materialize] wrote {n_done} images + meta.jsonl to {out}")


if __name__ == "__main__":
    main()
    # see the module docstring -- skips the interpreter-finalization sequence where
    # the HF datasets streaming client's background thread can race teardown; every
    # file is already flushed to disk by the time main() returns successfully
    os._exit(0)
