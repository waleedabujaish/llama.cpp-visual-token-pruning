#!/usr/bin/env python3
"""Dump llama.cpp mtmd encoder+projector output embeddings to .npy via ctypes.

Binds the pinned build's dylibs directly (build/bin/libmtmd.dylib etc.) —
no C++ written, no third-party bindings, so the numbers come from exactly the
code under study. Feeds raw RGB pixels through mtmd_bitmap_init so the input
is bit-identical to what the HF reference scripts consume (no stb/PIL decode
or resize differences; use a 336x336 PNG so clip preprocessing resize is a
no-op).

Struct layouts are transcribed from mtmd.h / llama.h at commit e8f19cc0 and
validated at runtime against the known values of mtmd_context_params_default()
(mtmd.cpp:240-257) — a layout mismatch would corrupt those fields and the
assertions would fail.

Usage:
  python dump_llamacpp_embd.py --image cat_336.png --out out.npy \
      [--fa auto|off|on] [--vocab-only]
"""

import argparse
import ctypes as C
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# llama.cpp checkout: $LLAMA_CPP_DIR, or a sibling clone of this repo
LLAMA_CPP = Path(os.environ.get("LLAMA_CPP_DIR", REPO_ROOT.parent / "llama.cpp"))
BIN_DIR = LLAMA_CPP / "build/bin"

MTMD_INPUT_CHUNK_TYPE_TEXT = 0
MTMD_INPUT_CHUNK_TYPE_IMAGE = 1

FA = {"auto": -1, "off": 0, "on": 1}


class llama_model_params(C.Structure):
    # include/llama.h struct llama_model_params (commit e8f19cc0)
    _fields_ = [
        ("devices", C.c_void_p),
        ("tensor_buft_overrides", C.c_void_p),
        ("n_gpu_layers", C.c_int32),
        ("split_mode", C.c_int),
        ("main_gpu", C.c_int32),
        ("tensor_split", C.c_void_p),
        ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p),
        ("kv_overrides", C.c_void_p),
        ("vocab_only", C.c_bool),
        ("use_mmap", C.c_bool),
        ("use_direct_io", C.c_bool),
        ("use_mlock", C.c_bool),
        ("check_tensors", C.c_bool),
        ("use_extra_bufts", C.c_bool),
        ("no_host", C.c_bool),
        ("no_alloc", C.c_bool),
    ]


class mtmd_context_params(C.Structure):
    # tools/mtmd/mtmd.h struct mtmd_context_params (commit e8f19cc0)
    _fields_ = [
        ("use_gpu", C.c_bool),
        ("print_timings", C.c_bool),
        ("n_threads", C.c_int),
        ("image_marker", C.c_char_p),
        ("media_marker", C.c_char_p),
        ("flash_attn_type", C.c_int),
        ("warmup", C.c_bool),
        ("image_min_tokens", C.c_int),
        ("image_max_tokens", C.c_int),
        ("cb_eval", C.c_void_p),
        ("cb_eval_user_data", C.c_void_p),
        ("batch_max_tokens", C.c_int32),
        ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p),
    ]


class mtmd_context_params_pruned(C.Structure):
    # tools/mtmd/mtmd.h struct mtmd_context_params, visual-token-pruning branch
    # (commit 82f2ccb5c): adds visual_keep/visual_prune_method between
    # image_max_tokens and cb_eval. A dylib built from this branch returns a
    # LARGER struct from mtmd_context_params_default() than the layout above;
    # using the pre-pruning layout against it would silently misalign every
    # field from cb_eval onward (this is exactly the failure mode
    # llama_provenance.py was built to catch on the binary-versioning side —
    # here it's a struct-layout analog of the same "the tree moved on"
    # problem). Use --pruned-abi to select this layout.
    _fields_ = [
        ("use_gpu", C.c_bool),
        ("print_timings", C.c_bool),
        ("n_threads", C.c_int),
        ("image_marker", C.c_char_p),
        ("media_marker", C.c_char_p),
        ("flash_attn_type", C.c_int),
        ("warmup", C.c_bool),
        ("image_min_tokens", C.c_int),
        ("image_max_tokens", C.c_int),
        ("visual_keep", C.c_float),
        ("visual_prune_method", C.c_char_p),
        ("cb_eval", C.c_void_p),
        ("cb_eval_user_data", C.c_void_p),
        ("batch_max_tokens", C.c_int32),
        ("progress_callback", C.c_void_p),
        ("progress_callback_user_data", C.c_void_p),
    ]


class mtmd_input_text(C.Structure):
    # tools/mtmd/mtmd.h struct mtmd_input_text
    _fields_ = [
        ("text", C.c_char_p),
        ("text_len", C.c_size_t),
        ("add_special", C.c_bool),
        ("parse_special", C.c_bool),
    ]


def load_libs(lib_dir=None, ctx_params_type=mtmd_context_params):
    d = Path(lib_dir) if lib_dir else BIN_DIR
    ggml = C.CDLL(str(d / "libggml.dylib"), mode=C.RTLD_GLOBAL)
    llama = C.CDLL(str(d / "libllama.dylib"), mode=C.RTLD_GLOBAL)
    mtmd = C.CDLL(str(d / "libmtmd.dylib"), mode=C.RTLD_GLOBAL)

    ggml.ggml_backend_load_all.restype = None

    llama.llama_backend_init.restype = None
    llama.llama_model_default_params.restype = llama_model_params
    llama.llama_model_load_from_file.restype = C.c_void_p
    llama.llama_model_load_from_file.argtypes = [C.c_char_p, llama_model_params]
    llama.llama_model_n_embd_inp.restype = C.c_int32
    llama.llama_model_n_embd_inp.argtypes = [C.c_void_p]

    mtmd.mtmd_default_marker.restype = C.c_char_p
    mtmd.mtmd_context_params_default.restype = ctx_params_type
    mtmd.mtmd_init_from_file.restype = C.c_void_p
    mtmd.mtmd_init_from_file.argtypes = [C.c_char_p, C.c_void_p, ctx_params_type]
    mtmd.mtmd_bitmap_init.restype = C.c_void_p
    mtmd.mtmd_bitmap_init.argtypes = [C.c_uint32, C.c_uint32, C.c_char_p]
    mtmd.mtmd_input_chunks_init.restype = C.c_void_p
    mtmd.mtmd_input_chunks_size.restype = C.c_size_t
    mtmd.mtmd_input_chunks_size.argtypes = [C.c_void_p]
    mtmd.mtmd_input_chunks_get.restype = C.c_void_p
    mtmd.mtmd_input_chunks_get.argtypes = [C.c_void_p, C.c_size_t]
    mtmd.mtmd_input_chunk_get_type.restype = C.c_int
    mtmd.mtmd_input_chunk_get_type.argtypes = [C.c_void_p]
    mtmd.mtmd_input_chunk_get_n_tokens.restype = C.c_size_t
    mtmd.mtmd_input_chunk_get_n_tokens.argtypes = [C.c_void_p]
    mtmd.mtmd_tokenize.restype = C.c_int32
    mtmd.mtmd_tokenize.argtypes = [C.c_void_p, C.c_void_p, C.POINTER(mtmd_input_text),
                                   C.POINTER(C.c_void_p), C.c_size_t]
    mtmd.mtmd_encode_chunk.restype = C.c_int32
    mtmd.mtmd_encode_chunk.argtypes = [C.c_void_p, C.c_void_p]
    mtmd.mtmd_get_output_embd.restype = C.POINTER(C.c_float)
    mtmd.mtmd_get_output_embd.argtypes = [C.c_void_p]
    return ggml, llama, mtmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to image (must already be 336x336)")
    ap.add_argument("--out", required=True, help="output .npy path")
    ap.add_argument("--model", default=str(REPO_ROOT / "models/llava-v1.5-7b-Q4_K_M.gguf"))
    ap.add_argument("--mmproj", default=str(REPO_ROOT / "models/llava-v1.5-7b-mmproj-model-f16.gguf"))
    ap.add_argument("--fa", choices=["auto", "off", "on"], default="auto")
    ap.add_argument("--vocab-only", action="store_true",
                    help="metadata-only text model load (llama_model_n_embd_inp returns 0 there, "
                         "so the full mmap load is the default)")
    ap.add_argument("--lib-dir", default=None,
                    help="directory with the llama.cpp dylibs (default: $LLAMA_CPP_DIR/build/bin); "
                         "use to test alternative builds, e.g. a fix branch's build dir")
    ap.add_argument("--any-size", action="store_true",
                    help="allow non-336x336 input (multi-tile models like llava-1.6; "
                         "llama.cpp then does its own resize/tiling, so pixel parity with an "
                         "external reference is no longer bit-exact)")
    ap.add_argument("--pruned-abi", action="store_true",
                    help="use the mtmd_context_params layout from the visual-token-pruning "
                         "branch (visual_keep/visual_prune_method fields present) -- required "
                         "for any --lib-dir built from that branch, e.g. build-prune")
    ap.add_argument("--visual-keep", type=float, default=1.0,
                    help="only meaningful with --pruned-abi")
    ap.add_argument("--visual-prune-method", default="none",
                    help="only meaningful with --pruned-abi (default: none)")
    args = ap.parse_args()

    ctx_params_type = mtmd_context_params_pruned if args.pruned_abi else mtmd_context_params

    img = Image.open(args.image).convert("RGB")
    if img.size != (336, 336) and not args.any_size:
        raise SystemExit(f"image must be 336x336 (got {img.size}); pre-resize it once and save as PNG")
    rgb = np.asarray(img, dtype=np.uint8)  # (H, W, 3), row-major RGB — matches mtmd_bitmap layout

    ggml, llama, mtmd = load_libs(args.lib_dir, ctx_params_type)
    ggml.ggml_backend_load_all()
    llama.llama_backend_init()

    # --- layout validation: defaults must match mtmd.cpp:240-257 exactly ---
    mp = mtmd.mtmd_context_params_default()
    marker = mtmd.mtmd_default_marker()
    assert mp.use_gpu is True and mp.print_timings is True, "mtmd_context_params layout mismatch (bools)"
    assert mp.n_threads == 4, f"mtmd_context_params layout mismatch (n_threads={mp.n_threads})"
    assert mp.image_marker is None and mp.media_marker == marker, "mtmd_context_params layout mismatch (markers)"
    assert mp.flash_attn_type == -1, f"layout mismatch (flash_attn_type={mp.flash_attn_type})"
    assert mp.image_min_tokens == -1 and mp.image_max_tokens == -1, "layout mismatch (min/max tokens)"
    assert mp.batch_max_tokens == 1024, f"layout mismatch (batch_max_tokens={mp.batch_max_tokens})"
    if args.pruned_abi:
        assert mp.visual_keep == 1.0, f"layout mismatch (visual_keep default={mp.visual_keep}, expected 1.0)"
        assert mp.visual_prune_method == b"none", \
            f"layout mismatch (visual_prune_method default={mp.visual_prune_method!r}, expected b'none')"
    print(f"[dump] mtmd_context_params layout OK (marker={marker.decode()}, pruned_abi={args.pruned_abi})")

    lp = llama.llama_model_default_params()
    assert lp.use_mmap is True and lp.use_mlock is False and lp.check_tensors is False, \
        "llama_model_params layout mismatch (bool block)"
    print(f"[dump] llama_model_params layout OK (n_gpu_layers={lp.n_gpu_layers})")

    # text model: only needed by mtmd_init for n_embd + vocab (markers);
    # vocab_only skips the 4 GB of weights.
    lp.vocab_only = args.vocab_only
    model = llama.llama_model_load_from_file(args.model.encode(), lp)
    assert model, "failed to load text model"
    n_embd = llama.llama_model_n_embd_inp(model)
    if n_embd <= 0:
        raise SystemExit(f"llama_model_n_embd_inp returned {n_embd} — vocab-only loads skip "
                         "hparams (llama-model.cpp:1073), use the full model load")
    print(f"[dump] text model loaded (vocab_only={lp.vocab_only}), n_embd_inp={n_embd}")

    mp.n_threads = 8
    mp.flash_attn_type = FA[args.fa]
    visual_prune_method_bytes = None  # keep alive: mp.visual_prune_method is a raw c_char_p
    if args.pruned_abi:
        mp.visual_keep = args.visual_keep
        visual_prune_method_bytes = args.visual_prune_method.encode()
        mp.visual_prune_method = visual_prune_method_bytes
        print(f"[dump] pruning params: visual_keep={mp.visual_keep} "
              f"visual_prune_method={mp.visual_prune_method}")
    ctx = mtmd.mtmd_init_from_file(args.mmproj.encode(), model, mp)
    assert ctx, "mtmd_init_from_file failed"

    bmp = mtmd.mtmd_bitmap_init(img.size[0], img.size[1], rgb.tobytes())
    assert bmp, "bitmap init failed"

    chunks = mtmd.mtmd_input_chunks_init()
    text = mtmd_input_text(text=marker, text_len=len(marker), add_special=False, parse_special=True)
    bitmaps = (C.c_void_p * 1)(bmp)
    ret = mtmd.mtmd_tokenize(ctx, chunks, C.byref(text), bitmaps, 1)
    assert ret == 0, f"mtmd_tokenize failed: {ret}"

    n_chunks = mtmd.mtmd_input_chunks_size(chunks)
    img_chunks = []
    for i in range(n_chunks):
        ch = mtmd.mtmd_input_chunks_get(chunks, i)
        t = mtmd.mtmd_input_chunk_get_type(ch)
        n = mtmd.mtmd_input_chunk_get_n_tokens(ch)
        print(f"[dump] chunk {i}: type={t} n_tokens={n}")
        if t == MTMD_INPUT_CHUNK_TYPE_IMAGE:
            img_chunks.append((ch, n))
    assert img_chunks, "no image chunk produced"

    parts = []
    for ch, n_tokens in img_chunks:
        ret = mtmd.mtmd_encode_chunk(ctx, ch)
        assert ret == 0, f"mtmd_encode_chunk failed: {ret}"
        ptr = mtmd.mtmd_get_output_embd(ctx)
        assert ptr, "null output embd"
        parts.append(np.ctypeslib.as_array(ptr, shape=(n_tokens, n_embd)).copy())
    emb = np.concatenate(parts, axis=0)
    print(f"[dump] {len(parts)} image chunk(s), total tokens={emb.shape[0]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, emb)
    stats = {
        "shape": list(emb.shape),
        "mean": float(emb.mean()), "std": float(emb.std()),
        "min": float(emb.min()), "max": float(emb.max()), "sum": float(emb.sum()),
        "token0_first16": [round(float(x), 6) for x in emb[0, :16]],
        "token0_last16": [round(float(x), 6) for x in emb[0, -16:]],
        "fa": args.fa,
        "lib_dir": str(Path(args.lib_dir).resolve()) if args.lib_dir else str(BIN_DIR.resolve()),
        "pruned_abi": args.pruned_abi,
        "visual_keep": args.visual_keep if args.pruned_abi else None,
        "visual_prune_method": args.visual_prune_method if args.pruned_abi else None,
    }
    print("[dump] stats:", json.dumps(stats))
    (out.with_suffix(".stats.json")).write_text(json.dumps(stats, indent=2))
    print(f"[dump] wrote {out}")


if __name__ == "__main__":
    main()
