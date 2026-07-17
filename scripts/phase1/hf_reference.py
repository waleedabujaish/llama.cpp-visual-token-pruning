#!/usr/bin/env python3
"""HF-side reference for the two suspected llama.cpp LLaVA bugs.

Recomputes the CLIP-ViT-L/14-336 vision tower + LLaVA MLP projector in pure
float32 torch, directly from the llava-hf/llava-1.5-7b-hf safetensors, with
two switchable defects:

  cls_last:  class embedding concatenated AFTER patches with identity position
             ids, and the projector consuming rows [1:] of the output
             (drops patch_0, keeps the CLS row) — llama.cpp behavior since the
             graph refactor (models/llava.cpp:36, clip.cpp:4095-4108).
  n_layers:  22 (llama.cpp: models/llava.cpp:15 subtracts one from the GGUF's
             already-truncated 23-layer tower) vs 23 (correct: all stored
             layers, features = HF hidden_states[-2]).

Four variants are produced:
  correct    cls_first, 23 layers   (what LLaVA-1.5 was trained on)
  layerbug   cls_first, 22 layers
  clsbug     cls_last,  23 layers
  bothbugs   cls_last,  22 layers   (hypothesis: matches llama.cpp output)

The ViT math mirrors clip.cpp: pre-LN, per-layer {LN1, attention (softmax
(qk^T*scale)v), residual, LN2, quick_gelu MLP, residual}, NO post-LN on
features (LLaVA uses pre-post-LN hidden states), projector = linear/GELU/
linear. Pixel normalization uses the CLIP constants on the same 336x336 PNG
fed to the llama.cpp dump, so inputs are bit-identical.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])
EPS = 1e-5
N_HEAD = 16
D_HEAD = 64


def load_weights(snapshot_dir: Path) -> dict:
    idx = json.loads((snapshot_dir / "model.safetensors.index.json").read_text())
    weight_map = idx["weight_map"]
    needed = {k: f for k, f in weight_map.items()
              if k.startswith("model.vision_tower.") or k.startswith("model.multi_modal_projector.")
              or k.startswith("vision_tower.") or k.startswith("multi_modal_projector.")}
    if not needed:
        raise SystemExit(f"no vision/projector tensors found; keys look like: {list(weight_map)[:5]}")
    tensors = {}
    for fname in sorted(set(needed.values())):
        with safe_open(snapshot_dir / fname, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in needed:
                    t = f.get_tensor(k)
                    k = k.removeprefix("model.")
                    tensors[k] = t.to(torch.float32)
    print(f"[hf] loaded {len(tensors)} tensors from {len(set(needed.values()))} shard(s)")
    return tensors


class Ref:
    def __init__(self, w: dict):
        self.w = w
        vt = "vision_tower.vision_model."
        self.class_emb = w[vt + "embeddings.class_embedding"]              # (1024,)
        self.patch_w = w[vt + "embeddings.patch_embedding.weight"]         # (1024,3,14,14)
        self.pos_emb = w[vt + "embeddings.position_embedding.weight"]      # (577,1024)
        self.pre_ln = (w[vt + "pre_layrnorm.weight"], w[vt + "pre_layrnorm.bias"])
        self.layers = []
        i = 0
        while f"{vt}encoder.layers.{i}.layer_norm1.weight" in w:
            p = f"{vt}encoder.layers.{i}."
            self.layers.append({
                "ln1": (w[p + "layer_norm1.weight"], w[p + "layer_norm1.bias"]),
                "q": (w[p + "self_attn.q_proj.weight"], w[p + "self_attn.q_proj.bias"]),
                "k": (w[p + "self_attn.k_proj.weight"], w[p + "self_attn.k_proj.bias"]),
                "v": (w[p + "self_attn.v_proj.weight"], w[p + "self_attn.v_proj.bias"]),
                "o": (w[p + "self_attn.out_proj.weight"], w[p + "self_attn.out_proj.bias"]),
                "ln2": (w[p + "layer_norm2.weight"], w[p + "layer_norm2.bias"]),
                "fc1": (w[p + "mlp.fc1.weight"], w[p + "mlp.fc1.bias"]),
                "fc2": (w[p + "mlp.fc2.weight"], w[p + "mlp.fc2.bias"]),
            })
            i += 1
        print(f"[hf] vision tower layers available: {len(self.layers)}")
        self.proj1 = (w["multi_modal_projector.linear_1.weight"], w["multi_modal_projector.linear_1.bias"])
        self.proj2 = (w["multi_modal_projector.linear_2.weight"], w["multi_modal_projector.linear_2.bias"])

    @staticmethod
    def ln(x, wb):
        return F.layer_norm(x, (x.shape[-1],), wb[0], wb[1], eps=EPS)

    def attn(self, x, L, return_probs=False):
        n = x.shape[0]
        scale = D_HEAD ** -0.5
        q = (F.linear(x, *L["q"]) * scale).view(n, N_HEAD, D_HEAD).transpose(0, 1)
        k = F.linear(x, *L["k"]).view(n, N_HEAD, D_HEAD).transpose(0, 1)
        v = F.linear(x, *L["v"]).view(n, N_HEAD, D_HEAD).transpose(0, 1)
        probs = torch.softmax(q @ k.transpose(-2, -1), dim=-1)   # (heads, n, n)
        out = (probs @ v).transpose(0, 1).reshape(n, N_HEAD * D_HEAD)
        out = F.linear(out, *L["o"])
        return (out, probs) if return_probs else (out, None)

    def layer(self, x, L, return_probs=False):
        h, probs = self.attn(self.ln(x, L["ln1"]), L, return_probs)
        x = x + h
        h = self.ln(x, L["ln2"])
        h = F.linear(h, *L["fc1"])
        h = h * torch.sigmoid(1.702 * h)  # quick_gelu, matches clip.cpp FFN_GELU_QUICK
        h = F.linear(h, *L["fc2"])
        return x + h, probs

    def patchify(self, img_336: Image.Image) -> torch.Tensor:
        x = torch.from_numpy(np.asarray(img_336, dtype=np.float32) / 255.0)  # (H,W,3)
        x = (x - CLIP_MEAN) / CLIP_STD
        x = x.permute(2, 0, 1).unsqueeze(0)                                   # (1,3,336,336)
        p = F.conv2d(x, self.patch_w, stride=14)                              # (1,1024,24,24)
        return p.flatten(2).squeeze(0).T.contiguous()                         # (576,1024) row-major

    def tower(self, patches, cls_first: bool, n_layers: int, probs_layer: int = -1):
        if cls_first:
            x = torch.cat([self.class_emb.unsqueeze(0), patches], dim=0)      # CLS at row 0 (HF)
        else:
            x = torch.cat([patches, self.class_emb.unsqueeze(0)], dim=0)      # CLS last (llama.cpp)
        x = x + self.pos_emb[: x.shape[0]]                                    # identity position ids
        x = self.ln(x, self.pre_ln)
        probs_out = None
        for il in range(n_layers):
            want = (il == (probs_layer if probs_layer >= 0 else n_layers + probs_layer))
            x, probs = self.layer(x, self.layers[il], return_probs=want)
            if want:
                probs_out = probs
        return x, probs_out                                                   # (577,1024), no post-LN

    def project(self, feats):
        h = F.linear(feats, *self.proj1)
        h = F.gelu(h)   # llava projector: standard GELU (llama.cpp uses tanh-approx ggml_gelu; ~1e-3)
        return F.linear(h, *self.proj2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--snapshot", required=True, help="llava-hf snapshot dir")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    img = Image.open(args.image).convert("RGB")
    assert img.size == (336, 336)

    ref = Ref(load_weights(Path(args.snapshot)))
    patches = ref.patchify(img)

    variants = {
        "correct":  dict(cls_first=True,  n_layers=23),
        "layerbug": dict(cls_first=True,  n_layers=22),
        "clsbug":   dict(cls_first=False, n_layers=23),
        "bothbugs": dict(cls_first=False, n_layers=22),
    }
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, cfg in variants.items():
        hidden, _ = ref.tower(patches, **cfg)
        feats = hidden[1:, :]   # llama.cpp projector gather: rows 1..576 (clip.cpp:4104-4108)
        emb = ref.project(feats)
        np.save(outdir / f"v_{name}.npy", emb.numpy())
        print(f"[hf] {name}: cls_first={cfg['cls_first']} n_layers={cfg['n_layers']} -> {tuple(emb.shape)}")

    # correct-semantics reference (CLS-first tower, drop CLS row 0 = true patch features)
    hidden, _ = ref.tower(patches, cls_first=True, n_layers=23)
    np.save(outdir / "v_correct_features_prepoj.npy", hidden[1:, :].numpy())
    print(f"[hf] wrote variants to {outdir}")


if __name__ == "__main__":
    main()
