# Claims ledger for the two bug one-pagers

Every factual claim in `bug-a-cls-ordering.md` and `bug-b-layer-count.md`,
classified:

- **[EMPIRICAL]** — backed by an executed run; the linked `results/` JSON is
  the artifact.
- **[STATIC]** — verifiable by reading code or GGUF metadata; the exact
  command is given (all code refs re-verified against upstream master =
  `e8f19cc0`, see `code-drift-check.md`).
- **[INFERENCE]** — code-path analysis, never executed. **Hedge these in the
  issue text** ("based on code reading", "should", "expected").

## Bug A — [CLS] ordering

| # | Claim | Class | Backing |
|---|---|---|---|
| A1 | CLS concatenated after patches | [STATIC] | `sed -n 36p tools/mtmd/models/llava.cpp` |
| A2 | positions filled identity 0..n_pos-1 | [STATIC] | `sed -n 4095,4099p tools/mtmd/clip.cpp` |
| A3 | patches filled [1..576] when class_embedding exists | [STATIC] | `sed -n 4101,4108p tools/mtmd/clip.cpp` |
| A4 | Net effect on LLaVA-1.5: shifted position embeddings, patch_0 dropped, CLS output fed as image token | [EMPIRICAL] | `results/20260717-022533_p1_bugverify_fa_auto.json`, `..._fa_off.json` (bothbugs 0.998 mean / 0.99998 median; shift test 83.5%; CLS-row cos 0.9999967) |
| A5 | llama.cpp output vs stock transformers `get_image_features`: 0.5010 mean cos | [EMPIRICAL] | `results/20260717-062557_p1_stock_validation.json` |
| A6 | fp32 reference == stock transformers (0.9999999999) | [EMPIRICAL] | same JSON |
| A7 | Isolated CLS control (layer count fixed): cls-last 0.998 vs cls-first 0.505 | [EMPIRICAL] | repro output; per-variant values in `..._p1_bugverify_*.json` (bothbugs vs layerbug rows) |
| A8 | Pre-refactor code placed CLS at row 0 (ggml_acc) | [STATIC] | `git show ffc727203af...:tools/mtmd/clip.cpp \| sed -n 1129,1137p` |
| A9 | Introduced by 32916a490 (#13321) | [STATIC] | `git log -S 'ggml_concat(ctx0, inp, model.class_embedding' --oneline` |
| A10 | No fix since; pattern copied into InternVL/Llama4/CogVLM | [STATIC] | same pickaxe + `git log --oneline 32916a490..master -- tools/mtmd/models/llava.cpp` |
| A11 | Affected: LLaVA-1.5 | [EMPIRICAL] | A4/A5 |
| A12 | Affected: LLaVA-1.6 (same graph, class_embd present) | [EMPIRICAL] | `results/20260717-082218_p1_llava16_check.json` (bothbugs 0.99998/tile) |
| A13 | 1.6 amplification: each tile independently corrupted (5 dropped patches + 5 CLS rows per 672×672 image; tiles bit-identical for identical pixels) | [EMPIRICAL] | same JSON (`n_tiles=5`, `inter_tile_max_abs_diff=0.0`, `cls_row=0.999998` per tile) |
| A14 | Affected: BakLLaVA, ShareGPT4V, Yi-VL, MobileVLM v1/v2 | [INFERENCE] | same graph + converter exports class_embd (`convert_image_encoder_to_gguf.py:60,108,164-172`; Yi promotion `clip.cpp:1962-1965`); never executed these models — **hedge** |
| A15 | Not affected: Granite Vision, GLM-Edge, Moondream2 (SigLIP → no class_embd; GLM_EDGE also skips the patches gather) | [INFERENCE] | converter class selection is [STATIC] (`:164-172`), but the specific GGUFs' metadata was not inspected — **hedge** (esp. Moondream2) |
| A16 | Modern converter cannot produce these projector types | [STATIC] | `grep -n '"mlp"\|"ldp"\|"adapter"' gguf-py/gguf/constants.py` (absent in VisionProjectorType, L4667-4711); `sed -n 41,42p conversion/llava.py` |
| A17 | End-task: −2.15pp on TextVQA n=200, 95% CI [−7.0, +2.75], n.s., quant-confounded | [EMPIRICAL] | `results/20260717-063925_p1_textvqa_llamacpp.json` |
| A18 | tests.sh not run in CI; substring pass criterion | [STATIC] | `grep -rn tests.sh ci/ .github/` (no hits); `sed -n 178p;206,209p tools/mtmd/tests.sh` |
| A19 | InternVL/CogVLM/Deepseek-OCR same-pattern leads | [INFERENCE] | explicitly flagged unverified in the one-pager — **hedge or omit** |
| A20 | Nothing downstream masks the per-tile corruption (unpad/image_newline unused) | [INFERENCE] | static greps (clip_patch_merge_type zero callers) but no execution — **hedge** |

## Bug B — layer double-subtraction

| # | Claim | Class | Backing |
|---|---|---|---|
| B1 | Converter writes block_count = n−1 and pops the last layer | [STATIC] | `sed -n 278,282p;349,353p tools/mtmd/legacy-models/convert_image_encoder_to_gguf.py` |
| B2 | Circulating LLaVA-1.5 mmproj files store block_count=23, tensors v.blk.0..22 (mys 2023 + second-state 2024) | [STATIC, executed] | `python3 scripts/… gguf_kv reader` on both files (run 2026-07-17; command in NOTES.md context) — re-runnable on any copy |
| B3 | Graph builds n_layer−1 layers (il_last, loop bound) | [STATIC] | `sed -n 11,30p;56p tools/mtmd/models/llava.cpp` |
| B4 | LLaVA-1.5 actually runs 22 of 23 stored layers | [EMPIRICAL] | 22-layer simulation matches llama.cpp 0.998 vs 0.949 for 23-layer (`results/*_p1_bugverify_*.json`: bothbugs vs clsbug) |
| B5 | Features are one layer short of `vision_feature_layer = -2` | [EMPIRICAL] | same + stock validation (A5/A6) |
| B6 | Introduced by 370359e5b (#3436); preserved by later refactors | [STATIC] | `git log -S 'il < n_layer - 1' --oneline -- examples/llava/clip.cpp tools/mtmd/clip.cpp` |
| B7 | Affected: LLaVA-1.6, BakLLaVA, ShareGPT4V, Yi-VL, MobileVLM (same converter flag) | LLaVA-1.6: [EMPIRICAL] — mmproj `block_count=23` + 22-layer sim wins per tile (0.99998 vs 0.956), `results/20260717-082218_p1_llava16_check.json`; others [INFERENCE] — **hedge** |
| B8 | Not affected: MiniCPM-V, GLM-Edge (full count + il_last+1), Granite (feature_layers) | [INFERENCE] | converter lines are [STATIC] (`minicpmv…py:587,710`; `glmedge…py:212`; `llava.cpp:18-20,24-29`) but the composed arithmetic was never executed for these models — **hedge lightly** |
| B9 | Modern converter cannot produce these files | [STATIC] | same as A16 |
| B10 | MiniCPM-V v1 fallback writes block_count=26 while exporting 27 layers | [STATIC] | `sed -n 594,597p;623p tools/mtmd/legacy-models/minicpmv-convert-image-encoder-to-gguf.py` — behavior implication is [INFERENCE], flagged as a separate converter-internal issue |
| B11 | Moondream2 exposure unconfirmed | [INFERENCE] | stated as unconfirmed in the one-pager — keep the hedge |
| B12 | v.blk.22 loaded but never executed | [STATIC + EMPIRICAL] | loader `clip.cpp:1862-1864` loads n_layer blocks; loop stops at 22 (B4's cosine evidence confirms 22 ran) |
| B13 | End-task numbers (shared with A17) | [EMPIRICAL] | `results/20260717-063925_p1_textvqa_llamacpp.json` |

## What to hedge in the issue text (the [INFERENCE] set)

1. Affected-model lists beyond LLaVA-1.5/1.6: BakLLaVA, ShareGPT4V, Yi-VL,
   MobileVLM (A14, B7) — say "by code path, not individually tested".
2. Not-affected lists (A15, B8) — say "based on converter/graph reading";
   Moondream2 explicitly unknown (B11).
3. InternVL/CogVLM/Deepseek-OCR leads (A19) — separate "possibly related"
   note or omit entirely.
4. "Nothing masks the tile corruption" (A20) — phrase as code observation.
5. MiniCPM-V v1 26/27 quirk (B10) — separate minor issue, code-reading only.

Everything else is either [EMPIRICAL] with a results JSON in this repo or
[STATIC] with a one-line verification command against master.
