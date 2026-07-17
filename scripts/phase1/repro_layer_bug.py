#!/usr/bin/env python3
"""Minimal repro: llama.cpp builds one ViT layer too few for LLaVA-1.5
(commit e8f19cc0).

The GGUF conversion already drops the 24th HF layer and stores
clip.vision.block_count = 23 (legacy-models/convert_image_encoder_to_gguf.py:281);
the graph then subtracts one AGAIN (models/llava.cpp:15 il_last = n_layer-1,
loop bound line 56), building 22 layers. LLaVA-1.5's projector was trained on
features from HF layer 23 (vision_feature_layer = -2 of 24).

Isolates the defect by holding the CLS ordering fixed at llama.cpp's actual
behavior (CLS-last) and toggling ONLY the layer count:

  A: 23 layers (all stored layers — intended)
  B: 22 layers (what the graph builds)

Expected output: llama.cpp matches B (~0.998), not A (~0.95).

Prereq: python dump_llamacpp_embd.py --image <336x336.png> --out e_cpp.npy
Usage:  python repro_layer_bug.py --cpp e_cpp.npy --image <same.png> --snapshot <llava-hf dir>
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hf_reference import Ref, load_weights


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

    print("llama.cpp output vs simulation (CLS ordering fixed at llama.cpp's actual cls-last):")
    for name, n_layers in (("A_23_layers(intended)", 23), ("B_22_layers(actual)", 22)):
        hidden, _ = ref.tower(patches, cls_first=False, n_layers=n_layers)
        v = ref.project(hidden[1:, :]).numpy()
        c = cos_rows(e_cpp, v)
        print(f"  {name:24s} mean cos = {c.mean():.6f}   median = {np.median(c):.6f}")

    print("\n  -> if B matches and A does not, the last stored layer (v.blk.22) is "
          "loaded but never executed,")
    print("     and features are computed one layer short of LLaVA-1.5's "
          "vision_feature_layer = -2.")


if __name__ == "__main__":
    main()
