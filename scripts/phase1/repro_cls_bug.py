#!/usr/bin/env python3
"""Minimal repro: llama.cpp mtmd places [CLS] last but gathers rows as if it
were first (LLaVA-1.5, commit e8f19cc0).

Isolates the CLS-ordering defect by holding the layer count fixed at
llama.cpp's actual 22 built layers and toggling ONLY the CLS position in an
otherwise-identical float32 reimplementation of the graph:

  A: CLS-first + identity positions + rows[1:]   (what the code intends)
  B: CLS-last  + identity positions + rows[1:]   (what the code does since
     the graph refactor: models/llava.cpp:36 concatenates CLS after patches,
     clip.cpp:4095-4108 fills identity positions and patches=[1..576])

Expected output: llama.cpp matches B (~0.998 mean cosine), not A (~0.5), and
llama.cpp's LAST output row matches the projected CLS token (~0.99999) —
i.e. patch_0 is dropped and the CLS output is fed to the LLM as an image
token, with every patch receiving its neighbor's position embedding.

Prereq: python dump_llamacpp_embd.py --image <336x336.png> --out e_cpp.npy
Usage:  python repro_cls_bug.py --cpp e_cpp.npy --image <same.png> --snapshot <llava-hf dir>
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hf_reference import Ref, load_weights

N_LAYERS_LLAMACPP = 22  # what the graph actually builds (llava.cpp:15,56)


def cos_rows(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--snapshot", required=True)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    e_cpp = np.load(args.cpp)
    ref = Ref(load_weights(Path(args.snapshot)))
    patches = ref.patchify(Image.open(args.image).convert("RGB"))

    variants = {}
    for name, cls_first in (("A_cls_first(intended)", True), ("B_cls_last(actual)", False)):
        hidden, _ = ref.tower(patches, cls_first=cls_first, n_layers=N_LAYERS_LLAMACPP)
        variants[name] = ref.project(hidden[1:, :]).numpy()

    print(f"llama.cpp output vs simulation (layer count fixed at {N_LAYERS_LLAMACPP}):")
    for name, v in variants.items():
        c = cos_rows(e_cpp, v)
        print(f"  {name:24s} mean cos = {c.mean():.6f}   median = {np.median(c):.6f}")

    cls_row = cos_rows(e_cpp[575:576], variants["B_cls_last(actual)"][575:576])[0]
    print(f"\n  llama.cpp row 575 vs projected [CLS] output: cos = {cls_row:.7f}")
    print("  -> if B matches and A does not, the projector consumes "
          "[patch_1..patch_575, CLS] instead of [patch_0..patch_575],")
    print("     and position embeddings are shifted by one for every patch.")


if __name__ == "__main__":
    main()
