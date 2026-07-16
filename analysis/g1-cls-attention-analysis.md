# G1 — Is final-layer [CLS] attention extractable from the llama.cpp CLIP path?

- Date: 2026-07-17
- llama.cpp commit: `e8f19cc0ad70a243c8012bf17b4be601abfc8ea2`
- Scope: LLaVA-1.5 (CLIP-ViT-L/14-336, `projector_type = mlp`), CPU backend.
  All file:line references are against the commit above.

## Verdict

**PASS, with a graph modification (K1 fallback (a) from the plan, small diff).**
The "attention is already computed, we just read it" scenario is **false** in the
default configuration: clip runs with flash attention **enabled** on CPU, and the
fused `ggml_flash_attn_ext` op never materializes attention probabilities.
There are two viable extraction routes (§4); the recommended one adds a
negligible-cost [CLS]-row scoring branch and keeps flash attention on. The
pruning hook itself is unusually convenient: the encoder→projector boundary
already gathers patch rows through a runtime `i32` index tensor +
`ggml_get_rows` (§5), and ggml has in-graph `argsort/top-k` precedents (§6).

Two unrelated upstream defects were found while establishing this (§7). Both
affect LLaVA-1.5 output correctness in current llama.cpp and one directly
affects where the [CLS] row lives.

## 1. Where attention is computed

Every ViT layer of the LLaVA graph calls the shared `clip_graph::build_attn`
(`tools/mtmd/clip.cpp:666-736`). It has two build-time paths, selected by
`clip_ctx.flash_attn_type`:

- **Flash path** (`clip.cpp:690-700`): fused `ggml_flash_attn_ext(q,k,v,...)`
  at `clip.cpp:699`. Attention probabilities never exist as a tensor.
- **Discrete path** (`clip.cpp:711-715`): `kq = ggml_mul_mat(k, q)` then
  `kq = ggml_soft_max_ext(kq, kq_mask, kq_scale, 0.0f)`. Here the post-softmax
  tensor **is** a real graph node, shape `[n_pos_k, n_pos_q, n_head]`
  (softmax over dim 0 = key positions; LLaVA passes a null mask). It is
  currently **unnamed** — only the attention output gets a name
  (`cb(cur, "kqv_out", il)`, `clip.cpp:725`) — so retrieving it requires adding
  a `ggml_set_name`/`cb()` call (naming helper at `clip.cpp:276-282`; the same
  named-tensor pattern is used for the `"positions"`/`"patches"` inputs).

Flash attention is controllable per clip context:
`clip_context_params.flash_attn_type` (`clip.h:42-50`), default `AUTO`
(`clip.cpp:164`, overwritten from ctx params at `clip.cpp:177`). `AUTO` is
resolved during warmup by trying FA and keeping it if the backend supports the
op (`clip.cpp:2929-2951`). On the CPU backend FA **is** supported, so the
default is **enabled** — confirmed empirically on this machine
(`warmup: flash attention is enabled` in the run log). mtmd passes llama's
flag straight through (`mtmd_get_clip_flash_attn_type`, `mtmd.cpp:231-238`,
applied at `mtmd.cpp:343`), so `llama-mtmd-cli -fa off` produces the
discrete-softmax graph today, with no code change.

## 2. The LLaVA graph and the [CLS] token

LLaVA (`PROJECTOR_TYPE_MLP`) is built by `clip_graph_llava`
(dispatch at `clip.cpp:971-978`, implementation in
`tools/mtmd/models/llava.cpp`), not by the generic `build_vit`.

- `n_pos = n_patches + 1` when `model.class_embedding` is present
  (`llava.cpp:7`; tensor `v.class_embd` loaded optionally at `clip.cpp:1841`).
- The class embedding is concatenated **after** the patch embeddings:
  `inp = ggml_concat(ctx0, inp, model.class_embedding, 1)` (`llava.cpp:36`).
  In the current graph the [CLS] token therefore sits at the **last** sequence
  position (row index `n_patches` = 576), unlike HF CLIP where it is row 0.
  See §7a — this is a regression, and any [CLS]-row extraction must track it.
- Position embeddings: the `"positions"` input is filled with the identity
  `0..n_pos-1` (`clip.cpp:4095-4099`).
- Projector-side CLS drop: the `"patches"` input is filled with
  `[1..n_patches]` (`clip.cpp:4101-4108`, offset 1 iff class_embedding exists)
  and consumed by `embeddings = ggml_get_rows(ctx0, embeddings, patches)`
  (`llava.cpp:154-160`) before the MLP projector (`mm_0`/`mm_2`,
  `llava.cpp:165-174`).

## 3. Which layer's attention is available

`llava.cpp:11-30` computes `max_feature_layer`; for llava-projector models
with no explicit `feature_layers` (granite-only) it is
`il_last = hparams.n_layer - 1`, and the layer loop runs
`for (il = 0; il < max_feature_layer; il++)` (`llava.cpp:56`).

For the model measured here (`llava-v1.5-7b-mmproj-model-f16.gguf`,
second-state repack): GGUF metadata says `clip.vision.block_count = 23` and the
file stores 23 layer tensor sets (`v.blk.0..v.blk.22`) — the conversion script
already dropped the 24th HF layer and subtracted one
(`legacy-models/convert_image_encoder_to_gguf.py:278-282`). The graph then
subtracts again and builds only **22** layers (see §7b).

Practical consequence for pruning: the deepest attention available in the built
graph is the attention **inside the last built layer** (currently `il = 21`).
The scoring hook should be expressed as "the attention of the last built layer"
so it tracks whatever upstream decides the feature layer is, rather than
hard-coding an index.

## 4. Extraction routes

**(i) Recommended — separate [CLS]-row scoring branch, FA stays on.**
In `build_attn` (or in `llava.cpp`'s attention block) for the scoring layer
only, add: `scores = softmax(q_cls · K^T · kq_scale)` where `q_cls` is a 1-row
view of Q at the CLS position. Cost per image ≈ `n_head · n_pos · d_head`
mul-adds (16·577·64 ≈ 0.6 MFLOP) plus one softmax over 577 values — vs ~0.7 s
of encoder compute, unmeasurable. Head-averaging via existing ops
(`ggml_sum_rows`/`ggml_scale` after a permute, or `ggml_mean`). This works
identically whether flash attention is on or off, and adds no O(n_pos²)
memory.

**(ii) Name and read the discrete softmax node, requires `-fa off`.**
One-line `cb(kq, "kq_soft_max", il)` after `clip.cpp:715`, then either mark it
as a graph output or observe it via the already-plumbed per-node eval callback
(`clip_context_params.cb_eval`, `clip.h:54`, wired at `clip.cpp:220-222`,
exposed as `mtmd_context_params.cb_eval`, `mtmd.h:103`). Downsides: forces the
non-flash encoder path (slowdown unmeasured — measure before choosing this
route) and materializes `[577, 577, 16]` f32 probs (~21 MB) per layer.
The cb_eval route is the right tool for **prototyping/validation** (dump real
attention values and compare rankings against the HF reference) even if (i)
ships.

## 5. Where the pruning hook goes

The encoder→projector boundary **already is** a runtime row-selection:
`"patches"` (i32 input) + `ggml_get_rows` (`llava.cpp:154-160`). Pruning top-K
by [CLS] attention is structurally the same gather with K < 576 rows.

Two implementation shapes:

- **In-graph (recommended):** scores (§4i) → `ggml_argsort_top_k`
  (`ggml.h:2389-2393`; plain `ggml_top_k` at `ggml.h:2395-2400` returns
  unordered indices) → `ggml_get_rows` on the pre-projector embeddings.
  To preserve spatial order of kept tokens, re-sort the K indices ascending
  (i32→f32 cast + `ggml_argsort` — exactly the workaround already used
  in-tree at `models/mimovl.cpp:84-85`). K is a host-side constant per encode
  call, which is fine: the clip graph is rebuilt from scratch on every encode
  (`clip.cpp:3576-3579`; single compute at `clip.cpp:4476`, single readback of
  the last node at `clip.cpp:4483,4503`).
- **Host-side two-pass** (encoder graph → read scores → fill a K-sized
  `"patches"` input → projector graph): possible but requires splitting the
  single encode compute into two; no advantage over in-graph for CPU.

In-graph data-dependent selection has solid precedent in the tree: MoE expert
routing (`argsort_top_k` → `get_rows`, `src/llama-graph.cpp:1915,1929`),
DeepSeek V3.2 sparse-attention indexer (`src/models/deepseek32.cpp:351`), the
backend sampler (`src/llama-sampler.cpp:1288-1293`), and mimovl's mid-encoder
row permutation (`models/mimovl.cpp:100-104`). Constraints verified:
`ggml_get_rows` requires i32 indices (`ggml/src/ggml.c:3898`); CPU
`ARGSORT`/`TOP_K` kernels are F32-input only (`ggml/src/ggml-cpu/ops.cpp:8390-8398,8456-8459`).

## 6. Token-count propagation (what else must change when K < 576)

The token count is **baked in at tokenize time**, before any encode runs:

- `mtmd_tokenize` → `clip_n_output_tokens` (`clip.cpp:3296`, per-projector
  branches) → stored in `mtmd_image_tokens.nx/ny` (`mtmd.cpp:1181-1203`);
  `n_tokens()` re-derives `nx·ny·nz` (`mtmd.cpp:94-110`).
- `clip_image_batch_encode` hard-aborts if the graph's actual output token
  count disagrees with `clip_n_output_tokens` (`clip.cpp:4486-4491`), so the
  llava branch of `clip_n_output_tokens` **must** return K.
- Encode output buffer sizing: `out_embd.resize(n_embd * n_tokens())`
  (`mtmd.cpp:1445-1447`).
- Position accounting: chunks advance `n_past` by
  `mtmd_input_chunk_get_n_pos` (`mtmd-helper.cpp:329-330`), which equals
  `n_tokens` for LLaVA's normal (non-M-RoPE) position type
  (`mtmd.cpp:1953-1957`) — so K image tokens at consecutive positions,
  matching FasterVLM's position-compression semantics, with no extra work.
- Decode batching splits by the same n_tokens (`mtmd-helper.cpp:267,296-299`).

So the clean design is: plumb the keep-ratio into the clip context, have
`clip_n_output_tokens` report K for the llava branch, and everything
downstream (chunk sizes, KV, positions, batches) stays consistent by
construction.

Flag plumbing chain for `--visual-keep` (precedent: `image_max_tokens` at
`mtmd-cli.cpp:156`): common params → `init_vision_context`
(`mtmd-cli.cpp:147-166`) → `mtmd_context_params` (`mtmd.h:95` block) →
`mtmd.cpp:343` → `clip_context_params` (`clip.h:50` block) → `clip_ctx`
(`clip.cpp:177`) → graph builder (`clip.cpp:261`). Note: `llama-server` fills
`mtmd_context_params` at its own site under `tools/server/` and needs the same
field for the eventual PR.

## 7. Upstream defects found (both affect LLaVA-1.5 correctness today)

**(a) [CLS] ordering regression (since the graph-builder refactor
`32916a490`, PR #13321, May 2025).** Pre-refactor code placed the class
embedding at **row 0** via `ggml_acc` at offset 0 with patches written from
row 1 (old `clip.cpp:1129-1137` at `32916a490~1`). The refactor replaced this
with `ggml_concat(inp, class_embedding)` (`llava.cpp:36`), which puts CLS
**last** — but the runtime inputs were not updated: `"positions"` is still the
identity (`clip.cpp:4095-4099`) and `"patches"` still skips row 0
(`clip.cpp:4101-4108`, its comment still describes the CLS-first layout).
Net effect for every class-embedding CLIP model (LLaVA-1.5, MobileVLM,
Yi-VL): each patch receives its neighbor's position embedding (CLS receives a
patch position), and the projector consumes rows `[patch_1..patch_575, CLS]` —
dropping `patch_0`'s output and feeding the CLS output as an image token.
Model still produces plausible text (ViTs are robust to this class of
corruption), which is presumably why it went unnoticed.
Not found in an upstream issue search (2026-07-17). Action: confirm
empirically (dump pre-projector rows vs HF reference), then report/patch
upstream — this also decides which row index a [CLS] scoring branch must use.

**(b) Vision layer double-subtraction (present since the original LLaVA
support, `370359e5b`, PR #3436, Oct 2023).** The conversion script stores
`block_count = num_hidden_layers - 1 = 23` for llava-projector models and
exports only 23 layer tensor sets (`legacy-models/convert_image_encoder_to_gguf.py:278-282`;
verified in both the 2023 mys and 2024 second-state mmproj GGUFs:
`clip.vision.block_count = 23`, tensors `v.blk.0..22`). The graph code then
subtracts again (`il_last = hparams.n_layer - 1`, `llava.cpp:15`) and builds
only 22 layers. LLaVA-1.5's projector was trained on features from HF layer
23 (`vision_feature_layer = -2` of 24); llama.cpp computes them from HF layer
22. `v.blk.22` is loaded but never used. Consequence for this project: the
"last built layer" whose attention we score is currently HF layer 22; if
upstream fixes the off-by-one, the scoring layer moves with it — another
reason to express the hook as "last built layer", not a constant.
Action: verify numerically against HF (this also affects baseline accuracy
for the study), then raise upstream together with (a).

## 8. Measured context (from G0/G2 baseline, this machine)

Apple M4 Pro, CPU-only build, LLaVA-1.5-7B Q4_K_M, 336×336 image → 576 image
tokens (n=6 runs, warm-up discarded; raw JSON in `results/`): vision encoder
≈ 0.69 s vs LLM prefill ≈ 5.43 s → encoder is ~11.3% of TTFT_vlm. Amdahl
ceiling for TTFT_vlm from pruning alone: ~8.8× (1/encoder-fraction), ~3.9×
when only the image-token share of prefill is treated as prunable (the
remainder includes first-decode initialization overhead; see the results
JSON). The prefill-dominated split is what makes encoder-side pruning worth
implementing on CPU at all — the ceiling is high, unlike the 30–50%
encoder-share regime the plan pre-registered as a risk.

## 9. Open items

1. Empirically confirm §7a/§7b with tensor dumps vs the HF implementation
   (also required groundwork for validating the scoring branch itself).
2. Measure the `-fa off` encoder cost on CPU (decides how bad route (ii) is;
   route (i) does not depend on it).
3. Search upstream more thoroughly / ask maintainers before filing §7 issues
   (G3 draft issue is the natural venue and stakes the claim early).
4. Decide keep-order semantics (spatial vs score order) — FasterVLM keeps
   spatial order; in-graph re-sort is cheap (§5).
