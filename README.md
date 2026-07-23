# Visual token pruning in llama.cpp — CPU vs GPU wall-clock study

Artifact repository for the paper **"Visual Token Pruning Pays Off More on
CPU than GPU: A Cross-Backend Wall-Clock Study in llama.cpp"** (preprint
forthcoming). It contains the measurement harness, every raw result file,
the analysis scripts, and the methodology log.

## Result in one table

Attention-based visual token pruning (`--visual-keep`, FasterVLM-style
[CLS] ranking) implemented at the encoder–projector boundary of
llama.cpp's `mtmd` pipeline, measured under one frozen protocol on three
backends:

|                                        | M4 Pro (CPU) | x86 Xeon (CPU, shared runner) | Tesla P100 (GPU) |
|----------------------------------------|:---:|:---:|:---:|
| LLM prefill speedup at keep=0.05       | 5.25× | 4.00× | 1.98× |
| Realized fraction of the theoretical prefill reduction | 87.8–99.9% | 77.8–81.7% | 43.5–67.7% |
| Decode speed gain                      | +21.4% | +9.1% | +2.2% |

Pruning realizes far more of its theoretical benefit on CPU than on the
tested GPU. TextVQA accuracy (n=200, paired bootstrap) shows no
statistically detectable change down to keep=0.05 on either backend;
POPE exposes the degradation mechanism under aggressive pruning
(precision 90.1%→95.9%, recall 78.7%→62.0% — the model turns
conservative). Recommended operating point: keep ≥ 0.25.

Implementation branch:
[`waleedabujaish/llama.cpp@visual-token-pruning`](https://github.com/waleedabujaish/llama.cpp/tree/visual-token-pruning).

## Two llama.cpp bugs found on the way

Establishing a trustworthy baseline surfaced two long-standing defects in
llama.cpp's LLaVA vision path, each verified against the HuggingFace
reference implementation and reported upstream with fixes:

- **Bug A — [CLS] ordering.** The class token is concatenated after the
  patch embeddings but consumed as if it were first: every patch gets its
  neighbor's position embedding, patch 0's output is dropped, and the
  projected [CLS] is fed to the LLM as an image token.
  Issue [ggml-org/llama.cpp#25814](https://github.com/ggml-org/llama.cpp/issues/25814),
  fix PR [#25844](https://github.com/ggml-org/llama.cpp/pull/25844).
  Details: `analysis/bug-a-cls-ordering.md`.
- **Bug B — vision layer count.** The layer meant to be skipped
  (`vision_feature_layer = -2`) is dropped twice — once at GGUF
  conversion, once at graph build — so features are computed one layer
  short and the last stored layer never runs.
  Issue [ggml-org/llama.cpp#25817](https://github.com/ggml-org/llama.cpp/issues/25817),
  fix PR [#25845](https://github.com/ggml-org/llama.cpp/pull/25845).
  Details: `analysis/bug-b-layer-count.md`.

All measurements in the study were taken with both fixes applied.
Everything is pinned to llama.cpp commit
`e8f19cc0ad70a243c8012bf17b4be601abfc8ea2`.

## Repo layout

- `results/` — timestamped raw JSON for every executed run: exact
  command, file hashes, per-run values, environment, determinism checks.
- `scripts/` — benchmark harness, sweep driver, analysis, and the
  verification/repro scripts.
- `analysis/` — bug write-ups, claims ledger, citation permalinks,
  fix-verification protocol.
- `figures/` — figure-generation scripts and the committed PDFs.
- `assets/` — pinned test images and evaluation manifests (TextVQA-200,
  POPE-300), hashes recorded in the result JSONs.
- `NOTES.md` — methodology log: pinned artifacts, frozen config, timing
  definitions, caveats. `PLAN.md` — hypotheses and status.

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
~0.99999. The keep-ratio sweeps are driven by `scripts/sweep_prune.py`
(resumable; skips cells already in `results/`).

## Citing

See [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this repository"
button). Each GitHub release is archived on Zenodo with a DOI.

## License

MIT — see [`LICENSE`](LICENSE).
