# Bug A — [CLS] token placed last but consumed as if first (llava graph)

Factual summary; all refs against llama.cpp `e8f19cc0`. Verified empirically
(see Evidence); numbers reproducible via the listed scripts.

## What happens

For every CLIP-encoder model on the llava graph, the class embedding is
concatenated **after** the patch embeddings, but the runtime inputs still
assume the pre-refactor CLS-first layout. Three consequences per encoded
image: (1) every patch receives its neighbor's position embedding (CLS
receives a patch position); (2) the projector consumes rows
`[patch_1 .. patch_575, CLS]` — the projected **[CLS] output is fed to the
LLM as an image token**; (3) `patch_0`'s output is dropped entirely.

## Code path

- `tools/mtmd/models/llava.cpp:36` — `inp = ggml_concat(ctx0, inp,
  model.class_embedding, 1)` puts CLS at the last sequence position.
- `tools/mtmd/clip.cpp:4095-4099` — `"positions"` filled with the identity
  `0..n_pos-1` (CLS-first assumption; HF assigns position 0 to CLS).
- `tools/mtmd/clip.cpp:4101-4108` — `"patches"` filled with `[1..576]`
  (`patch_offset = 1` iff class_embedding exists; the comment still describes
  the CLS-first layout), consumed by `ggml_get_rows` at `llava.cpp:160`.
- Pre-refactor code placed CLS at row 0 via `ggml_acc` at offset 0
  (`32916a490~1:tools/mtmd/clip.cpp:1129-1137`).

## Introduced / history

Graph-builder refactor `32916a490` (PR #13321, 2025-05-06). No fix since:
pickaxe on the concat shows only the same CLS-after-patches pattern being
copied into InternVL (`053367d14`), Llama 4 (`92ecdcc06`) and CogVLM
(`bacddc049`); `e39a2ce66` moved the code verbatim into `models/llava.cpp`.

## Affected models

Requires `v.class_embd` in the mmproj (loaded optionally, `clip.cpp:1841`);
the legacy converter exports it for every CLIP-encoder conversion
(`legacy-models/convert_image_encoder_to_gguf.py`, rename chain line 60).

- **Affected (empirically verified):** LLaVA-1.5 and LLaVA-1.6
  (PROJECTOR_TYPE_MLP). LLaVA-1.6 is **amplified and verified per tile**
  (llava-1.6-mistral-7b, uniform 672×672 → 5 tiles): every tile matches the
  both-bugs simulation at 0.99998 mean cosine (built from the mmproj's own
  weights), the projected [CLS] is the last row of **each** tile
  (cos 0.999998/tile), tiles are encoded independently (bit-identical for
  identical pixels) — i.e. 5 dropped patches + 5 injected CLS rows +
  5× shifted positions per image
  (`results/20260717-082218_p1_llava16_check.json`).
- **Affected by code path (not individually executed):** BakLLaVA,
  ShareGPT4V (MLP), Yi-VL (MLP → MLP_NORM promotion, `clip.cpp:1962-1965`),
  MobileVLM v1/v2 (LDP/LDPV2). Dispatch list: `clip.cpp:971-978`.
- **Not affected:** Granite Vision and GLM-Edge (SigLIP encoders, no class
  embedding → `patch_offset = 0` is correct), Moondream2 (SigLIP, dormant),
  MiniCPM-V (own builder), all modern-converter models (none produce these
  projector types — `gguf-py/gguf/constants.py:4667-4711` has no
  mlp/ldp/adapter entries).
- **Unverified leads (same pattern, correctness depends on the stored
  position-embedding row order):** InternVL (`models/internvl.cpp:10-13`),
  CogVLM, Deepseek-OCR.

## Evidence (LLaVA-1.5-7B, F16 mmproj, fixed 336×336 input)

- Stock transformers `get_image_features()` vs llama.cpp output: **0.5010
  mean per-token cosine** (reference validated against stock at 0.9999999999).
- fp32 simulation of the defect (CLS-last + identity positions + rows[1:]),
  layer count held fixed: matches llama.cpp at **0.998 mean / 0.99998 median**;
  CLS-first control: 0.505.
- llama.cpp's last image token vs projected [CLS]: **cos 0.9999967**.
- Shift test: llama.cpp row *i* matches correct patch *i+1* on 83.5% of rows.
- End-task (TextVQA n=200, paired, with Bug B and Q4_K_M quantization
  confounded): llama.cpp 56.35 vs HF-fp16-correct 58.50; −2.15pp,
  bootstrap 95% CI [−7.0, +2.75] — direction consistent, not significant;
  196/200 prompts carry OCR reference text, so the benchmark is weakly
  vision-sensitive.

## Why tests never caught it

`tools/mtmd/tests.sh` is not run by CI (no reference in `ci/run.sh` or any
workflow); its pass criterion is a case-insensitive substring grep over 128
greedy tokens (`tests.sh:178,206-209`). CI's only mtmd test exercises chunk
plumbing without inference (`tests/test-mtmd-c-api.c`).

## Repro

```
python scripts/phase1/dump_llamacpp_embd.py --image assets/phase1/cat_336.png --out e_cpp.npy
python scripts/phase1/repro_cls_bug.py --cpp e_cpp.npy --image assets/phase1/cat_336.png \
    --snapshot <llava-hf/llava-1.5-7b-hf snapshot dir>
```

## Fix sketch

Either restore CLS-first (concat class_embedding before patches) so the
existing fills are correct again, or keep CLS-last and fix both fills
(positions `[1..576,0]`-equivalent reorder; patches `[0..575]`). The fix must
cover every class-embedding model on this graph, and invalidates none of the
GGUF files (graph/runtime only).
