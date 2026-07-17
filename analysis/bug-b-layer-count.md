# Bug B — vision layer count subtracted twice (llava graph)

Factual summary; all refs against llama.cpp `e8f19cc0`. Verified empirically
(see Evidence); numbers reproducible via the listed scripts.

## What happens

For llava-projector models, the ViT layer intended to be skipped
(LLaVA consumes penultimate-layer features, `vision_feature_layer = -2`) is
dropped **twice**: once at conversion time and again at graph-build time. The
LLM therefore receives features computed one layer short of what the
projector was trained on (HF layer 22 instead of 23 for CLIP-L/14-336), and
the last stored layer's weights (`v.blk.22`) are loaded but never executed.

## Code path

- Conversion: `tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py:281`
  writes `block_count = num_hidden_layers - 1` when `has_llava_projector`,
  and pops the last encoder layer before export (lines 349-353). Verified in
  circulating GGUFs: both the 2023 mys and 2024 second-state LLaVA-1.5 mmproj
  files store `clip.vision.block_count = 23` with tensors `v.blk.0..22`.
- Graph: `tools/mtmd/models/llava.cpp:15` — `il_last = hparams.n_layer - 1`
  (comment: "index of the second to last layer") → `max_feature_layer = 22`;
  loop bound `il < max_feature_layer` at `llava.cpp:56` builds layers 0..21,
  i.e. 22 of the 23 stored layers.

## Introduced / history

Present since the original LLaVA support `370359e5b` (PR #3436, 2023-10):
both subtractions shipped together. Every later refactor (`7a2c913e6`
granite feature layers, `e39a2ce66` move to models/) preserved the logic.

## Affected models

Everything converted with `--llava-projector` through the legacy converter —
the only source of these projector types:

- **Affected (empirically verified):** LLaVA-1.5; LLaVA-1.6
  (llava-1.6-mistral-7b mmproj stores `block_count = 23`, and the 22-layer
  simulation wins per tile at 0.99998 vs 0.956 for 23 layers —
  `results/20260717-082218_p1_llava16_check.json`).
- **Affected by code path (not individually executed):** BakLLaVA,
  ShareGPT4V (MLP), Yi-VL (MLP_NORM; converted as mlp with
  `--llava-projector`), MobileVLM v1/v2 (LDP/LDPV2).
- **Not affected:** MiniCPM-V (its converter writes the full layer count,
  `minicpmv-convert-image-encoder-to-gguf.py:587,710`, and the graph adds
  one back: `llava.cpp:18-20` → all stored layers run); GLM-Edge (full count,
  `glmedge-convert-image-encoder-to-gguf.py:212`, same `+1`); Granite Vision
  (explicit `feature_layers` bypasses the default, `llava.cpp:24-29`, and the
  converter stores exactly `max(feature_layers)` layers).
- **Unconfirmed:** Moondream2 (SigLIP on this graph; exposure depends on its
  GGUF's `block_count`, not inspected).
- The modern `convert_hf_to_gguf.py` cannot produce any of these projector
  types (`conversion/llava.py:41-42` rejects non-pixtral LLaVA;
  `gguf-py/gguf/constants.py:4667-4711` lacks mlp/ldp/adapter), so real-world
  exposure is exactly the legacy-converted files — which are all such files
  in circulation, including the repo llama.cpp's own tests use.
- Related converter-internal oddity (separate issue): the MiniCPM-V v1
  no-config fallback writes `block_count = 26` while exporting 27 layers
  (`minicpmv-convert-image-encoder-to-gguf.py:594-597,623`).

## Evidence (LLaVA-1.5-7B, F16 mmproj, fixed 336×336 input)

- fp32 simulation with CLS ordering held at llama.cpp's actual behavior:
  22 layers matches llama.cpp at **0.998 mean / 0.99998 median** per-token
  cosine; 23 layers (intended) scores 0.949. Combined-defect simulation vs
  llama.cpp: 0.998 with a +0.049 margin over the next hypothesis.
- Reference validated against stock transformers `get_image_features()`
  (`vision_feature_layer = -2`) at mean cosine 0.9999999999.
- End-task context shared with Bug A (see bug-a-cls-ordering.md): −2.15pp on
  OCR-assisted TextVQA, n.s. at n=200, quantization-confounded.

## Why tests never caught it

Same as Bug A: the vision test suite is manual-only, asserts a substring over
128 greedy tokens (`tools/mtmd/tests.sh:178,206-209`), and CI runs no mtmd
inference at all.

## Repro

```
python scripts/phase1/dump_llamacpp_embd.py --image assets/phase1/cat_336.png --out e_cpp.npy
python scripts/phase1/repro_layer_bug.py --cpp e_cpp.npy --image assets/phase1/cat_336.png \
    --snapshot <llava-hf/llava-1.5-7b-hf snapshot dir>
```

Static check without running anything: mmproj metadata says
`clip.vision.block_count = 23` (tensors `v.blk.0..22`) while the graph builds
`n_layer - 1 = 22` layers.

## Fix sketch

The double subtraction has to be resolved on one side only. Graph-side (treat
stored `block_count` as "layers to run" for llava-projector models, i.e. drop
the extra `-1` when no explicit feature_layers are set) fixes every existing
GGUF in circulation without reconversion; converter-side would require
regenerating all published mmproj files. MiniCPM-V/GLM-Edge (+1 paths) and
granite (feature_layers) must remain unchanged either way.
