# Fix preview and verification protocol (for the upstream fix PRs)

Analysis only — no patch in this repo. All refs against `e8f19cc0`
(== upstream master, 2026-07-17; see code-drift-check.md).

## Bug A — minimal fix

**Option 1 (recommended, one line).** Restore CLS-first ordering at
`tools/mtmd/models/llava.cpp:36`:

- before: `inp = ggml_concat(ctx0, inp, model.class_embedding, 1);`
- after:  `inp = ggml_concat(ctx0, model.class_embedding, inp, 1);`

The existing runtime fills (identity `"positions"` at `clip.cpp:4095-4099`,
`"patches" = [1..576]` at `clip.cpp:4101-4108`) then become correct again;
this exactly restores the pre-refactor semantics (`ggml_acc` CLS at row 0,
`32916a490~1`). No GGUF change; touches only class-embedding models on this
graph (SigLIP models have a null class_embedding and are untouched).

**Option 2 (equivalent, two sites).** Keep CLS-last and fix the consumers:
positions `[1,2,...,n_patches, 0]` and `patch_offset = 0`
(`clip.cpp:4095-4108`). Strictly more churn; only worth it if upstream
prefers concat-at-end for batching symmetry with InternVL/CogVLM.

Out of scope but flag in the PR: `models/internvl.cpp:10-13`,
`models/cogvlm.cpp`, `models/deepseekocr.cpp` carry the same
CLS-concatenated-last pattern; whether those are defective depends on each
checkpoint's position-embedding row order (unverified).

## Bug B — minimal fix

Replace the double subtraction in `tools/mtmd/models/llava.cpp:14-20`:

- before: `int il_last = hparams.n_layer - 1;` plus
  `if (MINICPMV || GLM_EDGE) il_last += 1;`
- after:  `int il_last = hparams.n_layer;` and **delete** the `+1` branch.

Arithmetic check per family (all stored-layer counts verified in
code-drift-check.md):

| family | stored | built before | built after |
|---|---|---|---|
| llava/mobilevlm/yi (legacy, block_count = n−1) | 23 | 22 (wrong) | 23 (correct) |
| MiniCPM-V (full count) | N | (N−1)+1 = N | N (unchanged) |
| GLM-Edge (full count) | N | (N−1)+1 = N | N (unchanged) |
| granite-vision (feature_layers) | max(fl) | max(fl) (override) | unchanged |

Notes for the PR text: (1) `hparams.is_feature_layer(max_feature_layer)`
(`llava.cpp:137`) only fires for granite-style explicit feature layers —
unchanged. (2) No converter can produce a full-count GGUF for these
projector types (`conversion/llava.py:41-42`, `gguf-py` VisionProjectorType),
so "block_count means layers-to-run" is safe for every file in circulation.
(3) The fix makes the previously-dead `v.blk.22` execute: encoder compute
+1/22 ≈ +4.5%, TTFT_vlm impact well under 1% (encoder is ~11% of TTFT_vlm on
CPU, `results/20260717-011050_g0_baseline.json`).

## Verification protocol (run per-fix and combined; all tooling exists here)

**Level 1 — representation (minutes).**
1. Build the patched llama.cpp with the frozen flags
   (`-DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF`), record commit + diff.
2. `dump_llamacpp_embd.py --image assets/phase1/cat_336.png` (patched build) →
   `compare_embd.py` against the existing four variants. Expected
   best-matching variant transitions:
   - unpatched: `bothbugs` (measured 0.998)
   - Fix A only: `layerbug`
   - Fix B only: `clsbug`
   - both fixes: **`correct`**, mean cos ≥ ~0.998 (floor set by F16 mmproj +
     flash-attn fp16 K/V + ggml gelu LUT — do not expect 1.0 exactly)
3. `validate_stock.py` against stock `get_image_features()` — same threshold.
4. LLaVA-1.6: rerun the uniform-tile dump + `llava16_check.py`; per-tile best
   variant must flip to `correct`.
5. Non-regression on unaffected paths (Fix B): one MiniCPM-V or GLM-Edge
   run (temp 0, fixed seed/prompt/image) must produce byte-identical output
   pre/post patch; granite-vision likewise if convenient.

**Level 2 — end-task (about 1 hour).**
6. Paired TextVQA, same 200 samples (`assets/phase1/textvqa200_manifest.jsonl`),
   same vicuna_v1 jinja prompt, same scoring (`textvqa_llamacpp.py`):
   run once on the unpatched build, once on the patched build, **same Q4_K_M
   file** — pairing on identical quantization makes the quant confound cancel
   exactly, unlike the HF-vs-llama.cpp comparison. Report paired mean diff,
   bootstrap 95% CI, wins/losses/ties.
7. Context for expectations: unpatched-vs-HF-correct measured −2.15pp with CI
   [−7.0, +2.75] on this OCR-assisted benchmark (196/200 prompts contain the
   answer-bearing OCR line). If the CI again straddles zero, run the no-OCR
   variant (drop the OCR line on both sides — one-flag change) and/or a
   grounding-sensitive subset (GQA/POPE); representation-level evidence
   remains the primary proof either way.

**Level 3 — perf sanity (minutes).**
8. `scripts/bench_baseline.py` (frozen config) pre/post: encoder ms should
   rise ~4-5% under Fix B, prefill/decode unchanged; attach both JSONs.

**Bookkeeping.** Save every run to `results/` with the patched commit hash in
the JSON; keep the unpatched artifacts for the PR's before/after table.
