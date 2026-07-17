#!/usr/bin/env python3
"""Compare llama.cpp mtmd embeddings against the four HF reference variants.

Verdict logic: whichever variant matches the llama.cpp dump with per-token
cosine ~1.0 identifies exactly which of the two suspected defects are present
in llama.cpp. Additional targeted diagnostics:

  - shift test: cos(E_cpp[i], correct[i+1]) vs cos(E_cpp[i], correct[i]) —
    the CLS-ordering bug predicts row i of llama.cpp output = patch_{i+1}.
  - CLS-row test: cos(E_cpp[575], clsbug/bothbugs[575]) — those variants place
    the projected CLS output at the last row.
"""

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def cos_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    c = num / den
    if not np.isfinite(c).all():
        raise SystemExit("non-finite cosine values — check inputs (zero rows / NaNs)")
    return c


def summarize(c: np.ndarray) -> dict:
    return {"mean": float(c.mean()), "median": float(np.median(c)),
            "min": float(c.min()), "max": float(c.max()),
            "frac_gt_0.999": float((c > 0.999).mean()),
            "frac_gt_0.99": float((c > 0.99).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", required=True, help="llama.cpp dump .npy")
    ap.add_argument("--hfdir", required=True, help="dir with v_*.npy variants")
    ap.add_argument("--tag", default="g1_bug_verification")
    ap.add_argument("--image-sha", default="")
    ap.add_argument("--extra", default="", help="extra metadata json string")
    args = ap.parse_args()

    e_cpp = np.load(args.cpp)
    hfdir = Path(args.hfdir)
    variants = {p.stem[2:]: np.load(p) for p in sorted(hfdir.glob("v_*.npy"))
                if p.stem != "v_correct_features_prepoj"}
    assert e_cpp.shape[0] == 576, e_cpp.shape

    report = {"tag": args.tag, "cpp_dump": str(args.cpp), "image_sha256": args.image_sha,
              "per_variant": {}, "diagnostics": {}}

    for name, v in variants.items():
        report["per_variant"][name] = summarize(cos_rows(e_cpp, v))

    correct = variants["correct"]
    # shift test on rows 0..574: does cpp row i match correct patch i or i+1?
    aligned = cos_rows(e_cpp[:575], correct[:575])
    shifted = cos_rows(e_cpp[:575], correct[1:576])
    report["diagnostics"]["shift_test"] = {
        "cos_cpp_i_vs_correct_i": summarize(aligned),
        "cos_cpp_i_vs_correct_i+1": summarize(shifted),
        "shifted_wins_frac": float((shifted > aligned).mean()),
    }
    # CLS-row test: last cpp row vs the CLS row of the cls-bug variants
    for name in ("clsbug", "bothbugs"):
        if name in variants:
            v = variants[name]
            report["diagnostics"][f"cls_row_cos_vs_{name}"] = float(
                cos_rows(e_cpp[575:576], v[575:576])[0])
    # last cpp row vs correct patch 575 (should be LOW if CLS bug real)
    report["diagnostics"]["cls_row_cos_vs_correct_patch575"] = float(
        cos_rows(e_cpp[575:576], correct[575:576])[0])

    if args.extra:
        report["extra"] = json.loads(args.extra)

    ranked = sorted(report["per_variant"].items(), key=lambda kv: kv[1]["mean"], reverse=True)
    best, second = ranked[0], ranked[1]
    margin = best[1]["mean"] - second[1]["mean"]
    # confirmation requires an absolute match (the winning hypothesis must
    # actually reproduce llama.cpp, not merely beat the alternatives) and a
    # clear margin — otherwise an unmodeled third defect could be present.
    matched = best[1]["mean"] > 0.99 and margin > 0.005
    report["verdict"] = {
        "best_matching_variant": best[0],
        "best_mean_cos": best[1]["mean"],
        "second_best_variant": second[0],
        "second_best_mean_cos": second[1]["mean"],
        "margin": margin,
        "hypothesis_matched": matched,
        "layer_bug_confirmed": matched and best[0] in ("layerbug", "bothbugs"),
        "cls_bug_confirmed": matched and best[0] in ("clsbug", "bothbugs"),
        "note": None if matched else "no variant reproduces llama.cpp closely — unmodeled difference present",
    }

    print(json.dumps(report, indent=2))
    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = REPO_ROOT / "results" / f"{ts}_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
