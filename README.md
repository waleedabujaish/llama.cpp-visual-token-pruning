# Visual token pruning on llama.cpp (CPU) — benchmarks and analysis

Measurement and analysis repo for a study of attention-based visual token
pruning in llama.cpp's multimodal (mtmd) pipeline on CPU. Along the way it
documents two correctness defects in llama.cpp's LLaVA vision path, verified
empirically against the HF reference implementation:

- **Bug A** — the [CLS] token is concatenated after the patch embeddings but
  consumed as if it were first: position embeddings shift by one for every
  patch, patch 0's output is dropped, and the projected [CLS] output is fed
  to the LLM as an image token. See `analysis/bug-a-cls-ordering.md`.
- **Bug B** — the vision layer intended to be skipped is dropped twice (once
  at GGUF conversion, once at graph build), so features are computed one
  layer short of `vision_feature_layer = -2` and the last stored layer never
  runs. See `analysis/bug-b-layer-count.md`.

Everything is pinned to llama.cpp commit
`e8f19cc0ad70a243c8012bf17b4be601abfc8ea2`. Claim provenance is classified in
`analysis/claims-ledger.md`; per-citation permalinks in
`analysis/code-drift-check.md`; raw measurement JSONs (with config, file
hashes, and per-run values) in `results/`.

Project plan, hypotheses, and current status: [`PLAN.md`](PLAN.md).

## Reproducing the bug verification from a fresh clone

Prerequisites: Python 3.10+, cmake, ~11 GB disk (llama.cpp build + models +
HF reference weights). Tested on macOS/arm64; the dump script binds
llama.cpp's shared libraries via ctypes and expects `.dylib` names (on Linux,
adjust the three library suffixes in `scripts/phase1/dump_llamacpp_embd.py`).

```sh
# 1. llama.cpp at the pinned commit, CPU-only build (sibling directory, or
#    set LLAMA_CPP_DIR to wherever you put it)
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp
git -C ../llama.cpp checkout e8f19cc0ad70a243c8012bf17b4be601abfc8ea2
cmake -S ../llama.cpp -B ../llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF
cmake --build ../llama.cpp/build --target llama-mtmd-cli -j

# 2. Python environment
python3 -m venv .venv && .venv/bin/pip install torch transformers pillow numpy safetensors huggingface_hub

# 3. Models: LLaVA-1.5-7B GGUF pair (what llama.cpp's own tests use) and the
#    HF reference weights (vision tower + projector are read from shard 1)
mkdir -p models
curl -L -o models/llava-v1.5-7b-Q4_K_M.gguf https://huggingface.co/second-state/Llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-Q4_K_M.gguf
curl -L -o models/llava-v1.5-7b-mmproj-model-f16.gguf https://huggingface.co/second-state/Llava-v1.5-7B-GGUF/resolve/main/llava-v1.5-7b-mmproj-model-f16.gguf
SNAP=$(.venv/bin/python -c "from huggingface_hub import snapshot_download; print(snapshot_download('llava-hf/llava-1.5-7b-hf', allow_patterns=['*.safetensors','*.json']))")

# 4. Dump llama.cpp's actual encoder+projector output, then run both repros
cd scripts/phase1
../../.venv/bin/python dump_llamacpp_embd.py --image ../../assets/phase1/cat_336.png --out /tmp/e_cpp.npy
../../.venv/bin/python repro_cls_bug.py   --cpp /tmp/e_cpp.npy --image ../../assets/phase1/cat_336.png --snapshot "$SNAP"
../../.venv/bin/python repro_layer_bug.py --cpp /tmp/e_cpp.npy --image ../../assets/phase1/cat_336.png --snapshot "$SNAP"
```

Expected output: each repro prints the llama.cpp dump matching the
bug-simulating variant at ~0.998 mean per-token cosine and the intended
computation at ~0.50 / ~0.95 respectively, plus the [CLS]-row match at
~0.99999. The multi-tile LLaVA-1.6 variant of the check is
`scripts/phase1/llava16_check.py` (see `analysis/` for details), and the
four-variant comparison with verdict logic is `scripts/phase1/compare_embd.py`.

## Repo layout

- `analysis/` — the two bug write-ups, claims ledger, citation permalinks,
  fix/verification protocol, and the [CLS]-attention extractability study.
- `results/` — timestamped raw JSON for every executed run (benchmarks, bug
  verification, TextVQA accuracy simulations), plus keep-mask visualizations.
- `scripts/` — CPU TTFT benchmark harness and the phase-1 verification/
  prototype scripts.
- `assets/` — pinned test images (hashes recorded in the results JSONs) and
  the pinned 200-sample TextVQA manifest.
- `NOTES.md` — methodology record: pinned artifacts, frozen benchmark
  config, timing definitions, caveats.
