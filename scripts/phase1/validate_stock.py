#!/usr/bin/env python3
"""Validate the hand-rolled 'correct' reference against STOCK transformers.

Runs the same 336x336 image through llava-hf/llava-1.5-7b-hf's actual
get_image_features() (default config: vision_feature_layer=-2,
vision_feature_select_strategy='default') and compares per-token cosine
against (a) our v_correct.npy reference variant and (b) the llama.cpp dump.

Pixel input is constructed with exactly the same ops as hf_reference.Ref
.patchify (CLIP constants on the raw PNG), so (a) isolates implementation
differences only. The model loads fp16 (memory), but the vision tower and
projector are upcast to fp32 to match the reference precision.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])


def cos_rows(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--correct", required=True, help="v_correct.npy from hf_reference.py")
    ap.add_argument("--cpp", required=True, help="llama.cpp dump .npy")
    ap.add_argument("--model-id", default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from transformers import LlavaForConditionalGeneration

    torch.set_grad_enabled(False)
    img = Image.open(args.image).convert("RGB")
    assert img.size == (336, 336)
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0)
    x = ((x - CLIP_MEAN) / CLIP_STD).permute(2, 0, 1).unsqueeze(0)  # (1,3,336,336) fp32

    print("[stock] loading model (fp16, CPU) ...", flush=True)
    model = LlavaForConditionalGeneration.from_pretrained(args.model_id, dtype=torch.float16)
    model.eval()
    cfg = model.config
    print(f"[stock] config: vision_feature_layer={cfg.vision_feature_layer}, "
          f"vision_feature_select_strategy={cfg.vision_feature_select_strategy}")

    # upcast only the parts get_image_features touches
    holder = model.model if hasattr(model, "model") and hasattr(model.model, "vision_tower") else model
    holder.vision_tower.float()
    holder.multi_modal_projector.float()

    feats = model.get_image_features(
        pixel_values=x,
        vision_feature_layer=cfg.vision_feature_layer,
        vision_feature_select_strategy=cfg.vision_feature_select_strategy)
    # transformers v5: returns the vision ModelOutput with the PROJECTED
    # per-image features in .pooler_output (a list, one tensor per image)
    if hasattr(feats, "pooler_output"):
        feats = feats.pooler_output
    if isinstance(feats, (list, tuple)):
        feats = feats[0]
    if not torch.is_tensor(feats):
        raise SystemExit(f"unexpected get_image_features return type: {type(feats)}")
    stock = feats.squeeze(0).float().numpy()
    assert stock.shape == (576, 4096), stock.shape
    print(f"[stock] get_image_features -> {stock.shape}")

    v_correct = np.load(args.correct)
    e_cpp = np.load(args.cpp)

    c_ref = cos_rows(stock, v_correct)
    c_cpp = cos_rows(stock, e_cpp)
    report = {
        "config": {"vision_feature_layer": int(cfg.vision_feature_layer),
                   "strategy": str(cfg.vision_feature_select_strategy)},
        "stock_vs_our_correct": {"mean": float(c_ref.mean()), "median": float(np.median(c_ref)),
                                 "min": float(c_ref.min()),
                                 "frac_gt_0.9999": float((c_ref > 0.9999).mean())},
        "stock_vs_llamacpp": {"mean": float(c_cpp.mean()), "median": float(np.median(c_cpp)),
                              "min": float(c_cpp.min())},
    }
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
