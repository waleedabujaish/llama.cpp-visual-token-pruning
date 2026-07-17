#!/usr/bin/env python3
"""Empirical check of both bugs on LLaVA-1.6 (multi-tile path).

Design: a uniform mid-gray 672x672 input hits the [672,672] grid pinpoint, so
llava-uhd produces k tiles (overview + slices) whose 336x336 pixel content is
IDENTICAL. Every 576-row block of llama.cpp's output must therefore be the
same, and each block can be compared against a reference computed from the
mmproj's own weights (read directly out of the GGUF — no HF checkpoint
needed) under the four bug hypotheses. This makes the per-tile claims
empirical: CLS row injected per tile, patch_0 dropped per tile, 22-of-23
layers per tile.

Prereq: dump_llamacpp_embd.py --any-size run on the same uniform PNG with the
llava-1.6 model + mmproj.
"""

import argparse
import json
import struct
import datetime
from pathlib import Path

import numpy as np
import torch

from hf_reference import Ref

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

GGUF_SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
               5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
GGML_DTYPE = {0: (np.float32, 1), 1: (np.float16, 1)}  # F32, F16 only (all we need)


def read_gguf_tensors(path):
    """Minimal GGUF v2/v3 reader: returns {name: np.ndarray(float32)} for F32/F16 tensors."""
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    ver = struct.unpack("<I", f.read(4))[0]
    assert ver >= 2, ver
    n_tensors = struct.unpack("<Q", f.read(8))[0]
    n_kv = struct.unpack("<Q", f.read(8))[0]

    def rstr():
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", errors="replace")

    def rval(t):
        if t in GGUF_SCALAR:
            fmt, sz = GGUF_SCALAR[t]
            return struct.unpack(fmt, f.read(sz))[0]
        if t == 8:
            return rstr()
        if t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            n = struct.unpack("<Q", f.read(8))[0]
            return [rval(et) for _ in range(n)]
        raise ValueError(t)

    alignment = 32
    for _ in range(n_kv):
        key = rstr()
        t = struct.unpack("<I", f.read(4))[0]
        v = rval(t)
        if key == "general.alignment":
            alignment = int(v)

    infos = []
    for _ in range(n_tensors):
        name = rstr()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        dt = struct.unpack("<I", f.read(4))[0]
        off = struct.unpack("<Q", f.read(8))[0]
        infos.append((name, dims, dt, off))

    data_start = f.tell()
    data_start = (data_start + alignment - 1) // alignment * alignment

    out = {}
    for name, dims, dt, off in infos:
        if dt not in GGML_DTYPE:
            continue
        np_dt, _ = GGML_DTYPE[dt]
        count = int(np.prod(dims))
        f.seek(data_start + off)
        arr = np.frombuffer(f.read(count * np.dtype(np_dt).itemsize), dtype=np_dt)
        # GGUF ne[0] is fastest-varying -> numpy shape is reversed dims
        out[name] = arr.reshape(tuple(reversed(dims))).astype(np.float32)
    return out


def build_ref_from_gguf(tensors: dict) -> Ref:
    """Map GGUF clip tensor names onto the dict layout hf_reference.Ref expects."""
    vt = "vision_tower.vision_model."
    w = {
        vt + "embeddings.class_embedding": torch.from_numpy(tensors["v.class_embd"]).reshape(-1),
        vt + "embeddings.patch_embedding.weight": torch.from_numpy(tensors["v.patch_embd.weight"]),
        vt + "embeddings.position_embedding.weight": torch.from_numpy(tensors["v.position_embd.weight"]),
        vt + "pre_layrnorm.weight": torch.from_numpy(tensors["v.pre_ln.weight"]),
        vt + "pre_layrnorm.bias": torch.from_numpy(tensors["v.pre_ln.bias"]),
        "multi_modal_projector.linear_1.weight": torch.from_numpy(tensors["mm.0.weight"]),
        "multi_modal_projector.linear_1.bias": torch.from_numpy(tensors["mm.0.bias"]),
        "multi_modal_projector.linear_2.weight": torch.from_numpy(tensors["mm.2.weight"]),
        "multi_modal_projector.linear_2.bias": torch.from_numpy(tensors["mm.2.bias"]),
    }
    n_embd = w[vt + "embeddings.class_embedding"].shape[0]
    i = 0
    while f"v.blk.{i}.ln1.weight" in tensors:
        p = f"v.blk.{i}."
        hp = f"{vt}encoder.layers.{i}."
        for src, dst in (("attn_q", "self_attn.q_proj"), ("attn_k", "self_attn.k_proj"),
                         ("attn_v", "self_attn.v_proj"), ("attn_out", "self_attn.out_proj"),
                         ("ln1", "layer_norm1"), ("ln2", "layer_norm2")):
            w[hp + dst + ".weight"] = torch.from_numpy(tensors[p + src + ".weight"])
            w[hp + dst + ".bias"] = torch.from_numpy(tensors[p + src + ".bias"])
        # legacy converter historically swapped fc1/fc2 names (ffn_down/ffn_up);
        # assign by shape: fc1 maps hidden->4h, fc2 maps 4h->hidden
        a = torch.from_numpy(tensors[p + "ffn_down.weight"])
        a_b = torch.from_numpy(tensors[p + "ffn_down.bias"])
        b = torch.from_numpy(tensors[p + "ffn_up.weight"])
        b_b = torch.from_numpy(tensors[p + "ffn_up.bias"])
        if a.shape[0] > a.shape[1]:      # (4h, h) -> fc1
            fc1, fc1b, fc2, fc2b = a, a_b, b, b_b
        else:
            fc1, fc1b, fc2, fc2b = b, b_b, a, a_b
        w[hp + "mlp.fc1.weight"], w[hp + "mlp.fc1.bias"] = fc1, fc1b
        w[hp + "mlp.fc2.weight"], w[hp + "mlp.fc2.bias"] = fc2, fc2b
        i += 1
    print(f"[16] mapped {i} vision layers from GGUF, n_embd={n_embd}")
    return Ref(w)


def cos_rows(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp", required=True, help="llama.cpp dump .npy (uniform 672x672 input)")
    ap.add_argument("--mmproj", default=str(REPO_ROOT / "models/llava-v1.6-mistral-7b-mmproj-f16.gguf"))
    ap.add_argument("--gray", type=int, default=128, help="uniform pixel value used")
    ap.add_argument("--tag", default="p1_llava16_check")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    e_cpp = np.load(args.cpp)
    n_tokens = e_cpp.shape[0]
    assert n_tokens % 576 == 0, f"n_tokens={n_tokens} not a multiple of 576"
    k = n_tokens // 576
    blocks = e_cpp.reshape(k, 576, -1)
    print(f"[16] dump: {n_tokens} tokens = {k} tiles x 576")

    # tiles of a uniform image are pixel-identical -> blocks must match each other
    inter_tile = max(float(np.abs(blocks[i] - blocks[0]).max()) for i in range(1, k)) if k > 1 else 0.0
    print(f"[16] max |tile_i - tile_0| = {inter_tile:.6g}")

    # reference from the mmproj's own weights on one uniform 336x336 tile
    from PIL import Image
    tile = Image.new("RGB", (336, 336), (args.gray, args.gray, args.gray))
    ref = build_ref_from_gguf(read_gguf_tensors(args.mmproj))
    patches = ref.patchify(tile)
    n_avail = len(ref.layers)
    sims = {}
    for name, cls_first, n_layers in (
            ("correct", True, n_avail), ("layerbug", True, n_avail - 1),
            ("clsbug", False, n_avail), ("bothbugs", False, n_avail - 1)):
        hidden, _ = ref.tower(patches, cls_first=cls_first, n_layers=n_layers)
        sims[name] = ref.project(hidden[1:, :]).numpy()

    report = {"tag": args.tag,
              "timestamp": datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
              "mmproj": args.mmproj, "n_tiles": k, "n_tokens": n_tokens,
              "gguf_layers_stored": n_avail,
              "inter_tile_max_abs_diff": inter_tile,
              "per_tile": []}
    for i in range(k):
        row = {"tile": i}
        for name, v in sims.items():
            row[f"mean_cos_{name}"] = float(cos_rows(blocks[i], v).mean())
        row["cls_row_cos_vs_bothbugs"] = float(cos_rows(blocks[i][575:576], sims["bothbugs"][575:576])[0])
        report["per_tile"].append(row)
        print(f"[16] tile {i}: " + " ".join(f"{n}={row['mean_cos_' + n]:.4f}" for n in sims)
              + f" cls_row={row['cls_row_cos_vs_bothbugs']:.6f}")

    means = {n: float(np.mean([r[f"mean_cos_{n}"] for r in report["per_tile"]])) for n in sims}
    best = max(means, key=means.get)
    report["verdict"] = {
        "per_variant_mean": means,
        "best": best,
        "both_bugs_confirmed_per_tile": best == "bothbugs" and means["bothbugs"] > 0.99,
        "layers_built_inferred": f"{n_avail - 1} of {n_avail} stored"
        if best in ("layerbug", "bothbugs") else f"{n_avail}",
    }
    print(json.dumps(report["verdict"], indent=2))

    out = REPO_ROOT / "results" / f"{report['timestamp']}_{args.tag}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[16] wrote {out}")


if __name__ == "__main__":
    main()
