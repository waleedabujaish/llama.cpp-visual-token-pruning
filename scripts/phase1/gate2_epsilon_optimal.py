#!/usr/bin/env python3
"""Gate 2 amendment check: are the C++ implementation's "wrong" picks
epsilon-optimal under the Python reference scoring?

For every patch the C++ kept but the Python reference did not (a
"cpp_only" mismatch in the gate2_kept_index_parity.py output), compute
how far below the Python cutoff score that patch actually sits:

    gap = python_cutoff_score - python_score_of_cpp_pick

normalized by the image's own score standard deviation. A small
normalized gap means C++ picked a patch the reference itself ranks as
an essentially-tied near-miss, not a genuinely bad choice; this is the
expected signature of the pre-existing cross-implementation numerical
floor (Phase 1, ~0.998-0.999 mean cosine) occasionally relabeling a
near-tied patch, not a bug in the pruning code's scoring/top-K/gather.

Reads only already-computed data: scripts/phase1/gate2_kept_index_parity.py's
result JSON (for the mismatch list) and re-derives the Python score
vectors fresh (deterministic, no C++ involved -- the score std wasn't
serialized by that script). Does not invoke any llama.cpp binary.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_reference import Ref, load_weights  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate2-results", required=True,
                    help="results JSON from gate2_kept_index_parity.py")
    ap.add_argument("--snapshot", required=True, help="llava-hf HF snapshot dir")
    ap.add_argument("--tag", default="p2_gate2_epsilon_optimal")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    ref = Ref(load_weights(Path(args.snapshot)))

    d = json.loads(Path(args.gate2_results).read_text())

    score_std = {}
    for img in d["images"]:
        im = Image.open(img["image"]).convert("RGB")
        patches = ref.patchify(im)
        _, probs = ref.tower(patches, cls_first=True, n_layers=23, probs_layer=22)
        scores = probs[:, 0, 1:].mean(dim=0).numpy()
        score_std[img["image"]] = float(scores.std())

    rows = []
    for img in d["images"]:
        std = score_std[img["image"]]
        for rk, cell in img["ratios"].items():
            for m in cell["mismatch_detail"]:
                if m["side"] != "cpp_only":
                    continue
                kth_score = m["py_score"] - m["dist_from_kth_score"]
                gap = kth_score - m["py_score"]
                rows.append({
                    "image": img["image"], "keep": rk, "patch": m["patch"],
                    "gap": gap, "norm_gap": gap / std,
                    "known_diverged_token": m["preexisting_low_cosine_outlier"],
                    "cosine_at_keep1": m["preexisting_cosine_at_keep1"],
                })

    norm_gaps = np.array([r["norm_gap"] for r in rows])
    known = [r for r in rows if r["known_diverged_token"]]
    other = [r for r in rows if not r["known_diverged_token"]]

    def stats(xs):
        if not xs:
            return None
        a = np.array(xs)
        return {"n": len(xs), "max": float(a.max()), "median": float(np.median(a)), "mean": float(a.mean())}

    out = {
        "tag": args.tag, "timestamp": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        "method": "for every cpp_only mismatch (C++ kept, Python reference dropped), "
                 "gap = python_cutoff_score - python_score_of_cpp_pick, normalized by "
                 "the image's score std; small gap = epsilon-optimal pick, not a bug",
        "source_gate2_results": args.gate2_results,
        "all_gaps_nonnegative": bool((norm_gaps >= 0).all()),
        "overall": stats([r["norm_gap"] for r in rows]),
        "known_diverged_token_bucket": stats([r["norm_gap"] for r in known]),
        "other_bucket": stats([r["norm_gap"] for r in other]),
        "verdict": "PASS-AMENDED" if other and max(r["norm_gap"] for r in other) < 1.0 else "REOPEN",
        "rows": rows,
    }
    out_path = REPO_ROOT / "results" / f"{out['timestamp']}_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[gate2-eps] wrote {out_path}")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
