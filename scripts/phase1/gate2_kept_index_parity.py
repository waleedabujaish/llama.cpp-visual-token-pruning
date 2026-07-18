#!/usr/bin/env python3
"""Gate 2: kept-index parity between the C++ pruning implementation and the
Python FasterVLM prototype.

The C++ scoring/top-K/gather happens entirely inside the ggml graph (see
analysis/g2-hook-design-fixed-graph.md) and llama.cpp's own tensor-dump
debug path (MTMD_DEBUG_GRAPH / mtmd-debug -p encode) truncates any tensor
over 6 elements per dimension (common/debug.cpp's hardcoded n=3), so it
cannot recover the full 576-element cls_scores vector or a >6-element kept
set directly. Instead this recovers the C++ kept SET indirectly:

  1. Dump the encoder+projector output at --visual-keep 1.0 (unpruned) for
     an image -> E_full, 576 rows in original spatial patch order (the
     unpruned "patches" gather is just [1..576], row i = patch i).
  2. Dump the same image at the keep ratio under test -> E_pruned, K rows.
  3. The MLP projector (llava.cpp build_mm/gelu/build_mm) is a per-row
     (token-independent) linear map -- no cross-token mixing -- so row j of
     E_pruned is EXACTLY the same computation as whichever row of E_full it
     was gathered from. Match each E_pruned row to its nearest E_full row
     (min L2 distance); the match is expected to be near-exact (same
     weights, same op sequence) and essentially unambiguous (CLIP patch
     embeddings for different spatial patches differ by orders of
     magnitude more than any batch-size-dependent BLAS rounding noise).
     The matched row indices are the C++ implementation's kept set.

Compared against the Python prototype (hf_reference.py's hand-rolled fp32
23-layer tower, cls_first, probs_layer=22 -- exactly prune_viz.py's
cls_scores(), already validated in Phase 1 against stock HF transformers
at ~1.0 cosine and against the fixed llama.cpp graph's layer semantics).
"""

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hf_reference import Ref, load_weights  # noqa: E402


def dump_cpp(image, out_npy, lib_dir, model, mmproj, keep, method):
    cmd = [
        sys.executable, str(Path(__file__).resolve().parent / "dump_llamacpp_embd.py"),
        "--image", str(image), "--out", str(out_npy),
        "--lib-dir", str(lib_dir), "--model", str(model), "--mmproj", str(mmproj),
        "--fa", "auto",
    ]
    if keep != 1.0 or method != "none":
        cmd += ["--pruned-abi", "--visual-keep", str(keep), "--visual-prune-method", method]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"dump failed for {image} keep={keep}: {r.stderr[-3000:]}")
    return np.load(out_npy)


def recover_kept_indices(e_full, e_pruned):
    """Nearest-neighbor match each pruned row to a full-576 row; returns
    (indices sorted ascending, per-row match distance, ambiguity margin).

    Uses ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b via a matmul rather than
    materializing the (K, 576, 4096) broadcast difference directly -- the
    latter is ~2.7 GB at K=288, n_embd=4096."""
    full_sq = np.sum(e_full * e_full, axis=1)          # (576,)
    pruned_sq = np.sum(e_pruned * e_pruned, axis=1)     # (K,)
    cross = e_pruned @ e_full.T                          # (K, 576)
    d2 = pruned_sq[:, None] + full_sq[None, :] - 2 * cross
    d2 = np.maximum(d2, 0.0)  # guard against tiny negative values from fp cancellation
    d = np.sqrt(d2)
    idx = np.argmin(d, axis=1)
    best = d[np.arange(len(idx)), idx]
    d_sorted = np.sort(d, axis=1)
    second_best = d_sorted[:, 1]
    margin = second_best - best  # gap to the 2nd-nearest row; large margin = unambiguous match
    order = np.argsort(idx)
    return idx[order], best[order], margin[order]


def cls_scores_and_topk(ref, img336, keep_ratios):
    patches = ref.patchify(img336)
    _, probs = ref.tower(patches, cls_first=True, n_layers=23, probs_layer=22)
    scores = probs[:, 0, 1:].mean(dim=0).numpy()  # (576,)
    out = {}
    for r in keep_ratios:
        k = max(1, round(len(scores) * r))
        order_desc = np.argsort(scores)[::-1]
        kept = np.sort(order_desc[:k])
        # score gap at the K/K+1 cutoff boundary, for characterizing near-ties
        gap = float(scores[order_desc[k - 1]] - scores[order_desc[k]]) if k < len(scores) else float("inf")
        out[r] = {"k": int(k), "kept": kept, "cutoff_gap": gap, "scores": scores}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--lib-dir", required=True, help="build-prune/bin")
    ap.add_argument("--model", default=str(REPO_ROOT / "models/llava-v1.5-7b-Q4_K_M.gguf"))
    ap.add_argument("--mmproj", default=str(REPO_ROOT / "models/llava-v1.5-7b-mmproj-model-f16.gguf"))
    ap.add_argument("--snapshot", required=True, help="llava-hf HF snapshot dir")
    ap.add_argument("--ratios", default="0.5,0.25,0.1")
    ap.add_argument("--tag", default="p2_gate2_kept_index_parity")
    args = ap.parse_args()

    ratios = [float(r) for r in args.ratios.split(",")]
    torch_free_scores_cache = {}  # avoid re-running the Python tower per ratio

    import torch
    torch.set_grad_enabled(False)
    ref = Ref(load_weights(Path(args.snapshot)))

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dump_dir = REPO_ROOT / "results" / "raw" / f"{ts}_{args.tag}"
    dump_dir.mkdir(parents=True, exist_ok=True)

    per_image = []
    total_mismatches = 0
    total_cells = 0

    for image_path in args.images:
        image_path = Path(image_path)
        name = image_path.stem
        img = Image.open(image_path).convert("RGB")
        assert img.size == (336, 336), f"{image_path} is not 336x336"

        e_full = dump_cpp(image_path, dump_dir / f"{name}_full.npy",
                          args.lib_dir, args.model, args.mmproj, 1.0, "cls")
        assert e_full.shape[0] == 576, f"unpruned dump has {e_full.shape[0]} rows, expected 576"

        py = cls_scores_and_topk(ref, img, ratios)

        # Pre-existing per-token divergence check, independent of pruning: compare
        # the Python reference's full unpruned projector output against the C++
        # unpruned (keep=1.0) output, token by token. This tells us whether a given
        # patch already diverges between the two implementations BEFORE any
        # top-K/gather logic runs at all -- i.e. whether a Gate-2 mismatch at that
        # patch reflects a pruning-code bug or an inherited encoder-level property.
        patches_t = ref.patchify(img)
        hidden_full, _ = ref.tower(patches_t, cls_first=True, n_layers=23, probs_layer=-1)
        emb_py_full = ref.project(hidden_full[1:, :]).numpy()  # (576, 4096), row i = patch i
        denom = (np.linalg.norm(emb_py_full, axis=1) * np.linalg.norm(e_full, axis=1))
        denom = np.where(denom == 0, 1e-12, denom)
        per_token_cosine_keep1 = np.sum(emb_py_full * e_full, axis=1) / denom

        image_result = {
            "image": str(image_path),
            "preexisting_divergence_keep1_full576": {
                "mean_cosine": float(per_token_cosine_keep1.mean()),
                "min_cosine": float(per_token_cosine_keep1.min()),
                "n_tokens_below_0.98_cosine": int((per_token_cosine_keep1 < 0.98).sum()),
                "note": "per-token cosine between the Python reference's full unpruned "
                        "projector output and C++'s keep=1.0 output, same 576 tokens, "
                        "independent of any pruning logic -- context for whether a "
                        "mismatched patch below was already a pre-existing outlier",
            },
            "ratios": {},
        }
        for r in ratios:
            k = py[r]["k"]
            e_pruned = dump_cpp(image_path, dump_dir / f"{name}_keep{r}.npy",
                                args.lib_dir, args.model, args.mmproj, r, "cls")
            assert e_pruned.shape[0] == k, f"C++ returned {e_pruned.shape[0]} rows, expected K={k}"

            cpp_kept, match_dist, match_margin = recover_kept_indices(e_full, e_pruned)
            py_kept = py[r]["kept"]

            cpp_set, py_set = set(cpp_kept.tolist()), set(py_kept.tolist())
            only_cpp = sorted(cpp_set - py_set)
            only_py = sorted(py_set - cpp_set)
            exact_match = not only_cpp and not only_py

            scores = py[r]["scores"]
            cutoff_gap = py[r]["cutoff_gap"]
            # characterize any mismatched patches by their Python score's distance
            # from the K-th/K+1-th cutoff score (near-tie vs real divergence)
            order_desc = np.argsort(scores)[::-1]
            kth_score = scores[order_desc[k - 1]]
            mismatch_detail = []
            for patch_i in sorted(only_cpp + only_py):
                mismatch_detail.append({
                    "patch": int(patch_i),
                    "side": "cpp_only" if patch_i in only_cpp else "py_only",
                    "py_score": float(scores[patch_i]),
                    "dist_from_kth_score": float(scores[patch_i] - kth_score),
                    "preexisting_cosine_at_keep1": float(per_token_cosine_keep1[patch_i]),
                    "preexisting_low_cosine_outlier": bool(per_token_cosine_keep1[patch_i] < 0.98),
                })

            cell = {
                "keep": r, "k": k,
                "exact_match": exact_match,
                "n_mismatched": len(only_cpp) + len(only_py),
                "only_in_cpp": only_cpp, "only_in_py": only_py,
                "cutoff_gap_kth_vs_kplus1th": cutoff_gap,
                "mismatch_detail": mismatch_detail,
                "row_match_quality": {
                    "max_match_distance": float(match_dist.max()),
                    "min_ambiguity_margin": float(match_margin.min()),
                    "note": "match_distance should be ~0 (same computation, same weights); "
                            "ambiguity_margin should be >> match_distance (unambiguous nearest-neighbor)",
                },
            }
            image_result["ratios"][f"{r:g}"] = cell
            total_cells += 1
            total_mismatches += cell["n_mismatched"]
            print(f"[gate2] {name} keep={r} K={k}: exact_match={exact_match} "
                  f"n_mismatched={cell['n_mismatched']} "
                  f"max_match_dist={match_dist.max():.6g} min_margin={match_margin.min():.6g}",
                  flush=True)

        per_image.append(image_result)

    out = {
        "tag": args.tag, "timestamp": ts,
        "method": "row-matching recovery of C++ kept indices (see module docstring); "
                 "Python reference = hf_reference.Ref.tower(cls_first=True, n_layers=23, "
                 "probs_layer=22), exactly prune_viz.py's cls_scores()",
        "lib_dir": str(Path(args.lib_dir).resolve()),
        "hf_snapshot": args.snapshot,
        "ratios": ratios,
        "images": per_image,
        "summary": {
            "total_cells": total_cells,
            "total_mismatched_patches": total_mismatches,
            "all_exact_match": total_mismatches == 0,
        },
        "dump_dir": str(dump_dir.relative_to(REPO_ROOT)),
    }
    out_path = REPO_ROOT / "results" / f"{ts}_{args.tag}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[gate2] wrote {out_path}")
    print(json.dumps(out["summary"], indent=2))


if __name__ == "__main__":
    main()
