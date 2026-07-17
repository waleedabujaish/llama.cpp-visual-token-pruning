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
from pathlib import Path

import numpy as np
from PIL import Image

BIN_DIR = Path.home() / "Desktop/vtp/llama.cpp/build/bin"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

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


class mtmd_input_text(C.Structure):
    # tools/mtmd/mtmd.h struct mtmd_input_text
    _fields_ = [
        ("text", C.c_char_p),
        ("text_len", C.c_size_t),
        ("add_special", C.c_bool),
        ("parse_special", C.c_bool),
    ]


def load_libs():
    ggml = C.CDLL(str(BIN_DIR / "libggml.dylib"), mode=C.RTLD_GLOBAL)
    llama = C.CDLL(str(BIN_DIR / "libllama.dylib"), mode=C.RTLD_GLOBAL)
    mtmd = C.CDLL(str(BIN_DIR / "libmtmd.dylib"), mode=C.RTLD_GLOBAL)

    ggml.ggml_backend_load_all.restype = None

    llama.llama_backend_init.restype = None
    llama.llama_model_default_params.restype = llama_model_params
    llama.llama_model_load_from_file.restype = C.c_void_p
    llama.llama_model_load_from_file.argtypes = [C.c_char_p, llama_model_params]
    llama.llama_model_n_embd_inp.restype = C.c_int32
    llama.llama_model_n_embd_inp.argtypes = [C.c_void_p]

    mtmd.mtmd_default_marker.restype = C.c_char_p
    mtmd.mtmd_context_params_default.restype = mtmd_context_params
    mtmd.mtmd_init_from_file.restype = C.c_void_p
    mtmd.mtmd_init_from_file.argtypes = [C.c_char_p, C.c_void_p, mtmd_context_params]
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
    args = ap.parse_args()

    img = Image.open(args.image).convert("RGB")
    if img.size != (336, 336):
        raise SystemExit(f"image must be 336x336 (got {img.size}); pre-resize it once and save as PNG")
    rgb = np.asarray(img, dtype=np.uint8)  # (H, W, 3), row-major RGB — matches mtmd_bitmap layout

    ggml, llama, mtmd = load_libs()
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
    print(f"[dump] mtmd_context_params layout OK (marker={marker.decode()})")

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
    ctx = mtmd.mtmd_init_from_file(args.mmproj.encode(), model, mp)
    assert ctx, "mtmd_init_from_file failed"

    bmp = mtmd.mtmd_bitmap_init(336, 336, rgb.tobytes())
    assert bmp, "bitmap init failed"

    chunks = mtmd.mtmd_input_chunks_init()
    text = mtmd_input_text(text=marker, text_len=len(marker), add_special=False, parse_special=True)
    bitmaps = (C.c_void_p * 1)(bmp)
    ret = mtmd.mtmd_tokenize(ctx, chunks, C.byref(text), bitmaps, 1)
    assert ret == 0, f"mtmd_tokenize failed: {ret}"

    n_chunks = mtmd.mtmd_input_chunks_size(chunks)
    img_chunk = None
    for i in range(n_chunks):
        ch = mtmd.mtmd_input_chunks_get(chunks, i)
        if mtmd.mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE:
            img_chunk = ch
            break
    assert img_chunk is not None, "no image chunk produced"
    n_tokens = mtmd.mtmd_input_chunk_get_n_tokens(img_chunk)
    print(f"[dump] image chunk: n_tokens={n_tokens}")

    ret = mtmd.mtmd_encode_chunk(ctx, img_chunk)
    assert ret == 0, f"mtmd_encode_chunk failed: {ret}"

    ptr = mtmd.mtmd_get_output_embd(ctx)
    assert ptr, "null output embd"
    emb = np.ctypeslib.as_array(ptr, shape=(n_tokens, n_embd)).copy()

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
    }
    print("[dump] stats:", json.dumps(stats))
    (out.with_suffix(".stats.json")).write_text(json.dumps(stats, indent=2))
    print(f"[dump] wrote {out}")


if __name__ == "__main__":
    main()
