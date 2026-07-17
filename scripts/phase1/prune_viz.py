#!/usr/bin/env python3
"""[CLS]-attention keep-mask visualizations (FasterVLM ranking, offline).

For each input image: compute CLS->patch attention scores exactly as FasterVLM
does (select_layer=-2: scores from the attention inside the 23rd vision-tower
layer, mean over heads, CLS query -> patch keys), then render the keep-mask at
several keep ratios on a 24x24 grid overlaid on the image.

Output: one PNG per image (original + heat map + masks at the given ratios),
plus a JSON with per-image score stats.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hf_reference import Ref, load_weights

GRID = 24  # 336/14


def cls_scores(ref: Ref, img336: Image.Image) -> np.ndarray:
    """FasterVLM ranking: attentions[select_layer=-2][:, :, 0, 1:].mean(heads)."""
    patches = ref.patchify(img336)
    # 23 layers run (0..22), probs taken from the last run layer (il=22);
    # equivalent to HF attentions[-2] of the full 24-layer tower.
    _, probs = ref.tower(patches, cls_first=True, n_layers=23, probs_layer=22)
    scores = probs[:, 0, 1:].mean(dim=0)          # (576,)
    return scores.numpy()


def keep_mask(scores: np.ndarray, keep: float) -> np.ndarray:
    k = max(1, round(len(scores) * keep))
    idx = np.argsort(scores)[::-1][:k]
    m = np.zeros(len(scores), dtype=bool)
    m[idx] = True
    return m


def render(img336: Image.Image, scores: np.ndarray, ratios, out_png: Path, title: str):
    n = 2 + len(ratios)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    img_np = np.asarray(img336)

    axes[0].imshow(img_np)
    axes[0].set_title("image", fontsize=10)

    heat = scores.reshape(GRID, GRID)
    axes[1].imshow(img_np)
    axes[1].imshow(np.kron(heat / heat.max(), np.ones((14, 14))), cmap="jet", alpha=0.5)
    axes[1].set_title("[CLS] attention", fontsize=10)

    for ax, r in zip(axes[2:], ratios):
        m = keep_mask(scores, r).reshape(GRID, GRID)
        overlay = img_np.copy().astype(np.float32)
        dim = np.kron(~m, np.ones((14, 14), dtype=bool))
        overlay[dim] *= 0.18
        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(f"keep {int(r*100)}%", fontsize=10)

    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--ratios", default="0.5,0.25,0.1")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    ratios = [float(r) for r in args.ratios.split(",")]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ref = Ref(load_weights(Path(args.snapshot)))
    stats = {}
    for path in args.images:
        p = Path(path)
        img = Image.open(p).convert("RGB").resize((336, 336), Image.BICUBIC)
        s = cls_scores(ref, img)
        out_png = outdir / f"{p.stem}_mask.png"
        render(img, s, ratios, out_png, p.name)
        stats[p.name] = {
            "score_min": float(s.min()), "score_max": float(s.max()),
            "score_entropy_bits": float(-(s / s.sum() * np.log2(s / s.sum() + 1e-12)).sum()),
            "top10pct_mass": float(np.sort(s)[::-1][:58].sum() / s.sum()),
        }
        print(f"[viz] {p.name}: wrote {out_png.name}, top-10% attention mass = {stats[p.name]['top10pct_mass']:.3f}")

    (outdir / "viz_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
