# Methodology notes

Running record of measurement decisions. Source material for the write-up's
methodology section.

## Model artifacts (pinned 2026-07-17)

| File | Source | SHA256 |
|---|---|---|
| `models/llava-v1.5-7b-Q4_K_M.gguf` | HF `second-state/Llava-v1.5-7B-GGUF` | `2687b20ac8b7a23f6c70296d5b1e7f908fef2ce4769ecdebd1bb9503528a75bf` |
| `models/llava-v1.5-7b-mmproj-model-f16.gguf` | HF `second-state/Llava-v1.5-7B-GGUF` | `50da4e5b0a011615f77686f9b02613571e65d23083c225e107c08c3b1775d9b1` |

**Substitution note.** The originally planned `mys/ggml_llava-v1.5-7b`
(ggml-model-q4_k.gguf + mmproj-model-f16.gguf) fails to load at the pinned
llama.cpp commit: its 2023-era mmproj lacks the `clip.projector_type` GGUF key,
and the current loader rejects it (`clip_init: ... unknown projector type`,
`tools/mtmd/clip.cpp:1153`). The second-state repack is the same LLaVA-1.5-7B
weights and is the artifact llama.cpp's own vision test suite uses
(`tools/mtmd/tests.sh:95`, `second-state/Llava-v1.5-7B-GGUF:Q2_K`). Main model
quant used here: Q4_K_M; mmproj: F16.

## Test image

- `assets/coco_val2017_000000039769.jpg` — COCO val2017 image 39769
  (source: `http://images.cocodataset.org/val2017/000000039769.jpg`),
  640×480 JPEG, SHA256
  `dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e`.
- At LLaVA-1.5's fixed 336×336 input this always yields 576 image tokens, so
  encoder/prefill timing is content-independent; the image mainly serves
  determinism/sanity checks (temp-0 output must describe two cats on a couch).

## llama.cpp build (pinned)

- Commit `e8f19cc0ad70a243c8012bf17b4be601abfc8ea2`, clean tree.
- `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF`; target `llama-mtmd-cli`.
- Resulting backends: CPU (`-mcpu=native+dotprod+i8mm+nosve+sme`) +
  BLAS via Apple Accelerate. No Metal, no GPU.
- Hardware: Apple M4 Pro (8 P-cores + 4 E-cores), 24 GB, macOS 15.2
  (Darwin 24.2.0).

## Frozen run configuration (applies to EVERY benchmark cell, incl. pruned runs)

```
-n 32  --temp 0  --seed 42  -t 8  -tb 8  -b 2048  -ub 1024
--perf  --chat-template vicuna  -v
```

Frozen 2026-07-17 after the G0 baseline. Any change to these flags between
cells invalidates cross-cell comparison; if a change ever becomes unavoidable,
all affected cells get re-run under the new config.

Rationale per flag:

- **`-ub 1024` (physical ubatch).** The 576-token image-embedding batch is
  decoded with causal attention disabled (`mtmd_helper_decode_image_chunk`,
  `tools/mtmd/mtmd-helper.cpp:290-291`), and the code notes n_ubatch must be
  able to hold the whole image (`mtmd-helper.cpp:293` TODO). The default
  `-ub 512` would split the 576-token non-causal batch across two ubatches.
  1024 is the smallest power of two that holds 576 with headroom. This also
  matters for pruned runs later: keep-ratios up to 100% keep the image batch
  in a single ubatch, so batching behavior is identical across all cells.
- **`-t 8 -tb 8` = P-cores only, deliberate.** The M4 Pro has 8 performance +
  4 efficiency cores; spilling onto E-cores adds scheduler-dependent variance.
  An E-core/all-core ablation is a possible later addition; not part of the
  main matrix.
- **`--temp 0 --seed 42`.** Determinism; G0 verified bit-identical output
  across all 6 timed runs. Seed recorded for completeness (unused at temp 0).
- **`--chat-template vicuna`.** The GGUF embeds no chat template; the CLI
  errors and explicitly instructs vicuna for LLaVA-1.5.
- **`-b 2048`** llama.cpp default logical batch, stated explicitly to pin it.
- **`-n 32`.** Enough decoded tokens for a stable decode-tok/s sanity metric
  without inflating run time.
- **`-v`.** Required: the per-stage timing lines (image-decode, llama_perf)
  do not print at default verbosity at this commit. Verbose logging sits
  outside the internally-timed windows (timers wrap only the compute calls).
- **`--perf`.** Enables libllama perf counters.

## Timing sources and definitions

All timings are llama.cpp's own internal measurements, parsed from the run log
(exact format strings at the pinned commit):

- Vision encoder + projector: `mtmd batch encoding done in <N> ms`
  (`tools/mtmd/mtmd-cli.cpp:338`).
- LLM prefill over image embeddings: `image decoded (batch i/n) in <N> ms`
  (`tools/mtmd/mtmd-helper.cpp:324`).
- `llama_perf_context_print` lines (`src/llama-context.cpp:4103-4108`):
  `prompt eval time` counts every `llama_decode` with >1 tokens — text batches
  AND the image-embedding batch (`src/llama-context.cpp:709-719`) — so
  **TTFT_llm := prompt eval time**. `eval time` = single-token decodes.
- **TTFT_vlm := encode_ms + prompt_eval_ms** (excludes model load and image
  preprocessing; tokenize/preprocess not separately instrumented in Phase 0).
- Known caveat: the first `llama_decode` of a process carries one-time
  initialization; it lands in `prompt eval time` (in the pre-image text
  chunk). This makes the "text share" of prefill look larger than
  steady-state and the *refined* Amdahl ceiling pessimistic. Quantify by
  amortizing over a persistent process if it starts to matter.

## Phase 1 (offline Python) — provenance

- venv: torch 2.13.0 (MPS), transformers 5.14.1, datasets 5.0.0, numpy 2.5.1,
  safetensors 0.8.0, pillow 12.3.0.
- HF reference model: `llava-hf/llava-1.5-7b-hf`, snapshot
  `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` (fp16 safetensors, computed in
  fp32 for the bug-verification reference).
- llama.cpp embeddings dumped via ctypes against the pinned build's dylibs
  (`scripts/phase1/dump_llamacpp_embd.py`); layouts transcribed from headers
  at commit e8f19cc0 and validated at runtime against
  `mtmd_context_params_default()`; dump cross-validated against the CLI's
  `MTMD_DEBUG_EMBEDDINGS` output (token-0 values identical to 6 decimals).
- Bug-verification input: `assets/phase1/cat_336.png` (336×336 PNG, lossless,
  resize no-op on both pipelines; sha256
  `e42c7ea7c89d69541a7ab27ebdcf75272790529be5e59c759150a17021e561c0`).
- Expected numerical floor: F16 mmproj + FA fp16 K/V casts + ggml quick-gelu
  LUT vs fp32 reference keeps even the true-matching variant at ~0.998-0.999
  mean cosine, not 1.0. Verdicts use an absolute threshold (0.99) + margin.
- TextVQA sim: `lmms-lab/textvqa` validation, first 200 samples in streaming
  order (196 carried a "Reference OCR token" line, matching the official
  LLaVA-1.5 eval format); vicuna_v1 prompt; greedy, max 16 new tokens;
  fp16 on MPS (text=sdpa, vision=eager); soft VQA accuracy (leave-one-out,
  VQAv2-style normalization — simplified on some punctuation edge cases,
  applied identically to predictions and ground truth). n=200 gives roughly
  ±3.5pp standard error per cell; cross-ratio comparisons are paired on the
  same samples.
- Pruning semantics in all Phase 1 code = FasterVLM: scores =
  `attentions[-2][:, :, 0, 1:].mean(heads)`, features = `hidden_states[-2][:, 1:]`,
  top-K via boolean mask (original spatial order preserved).

## Pre-posting verification pass (2026-07-17)

- Upstream drift: `upstream/master` fetched 2026-07-17 ==
  `e8f19cc0ad70a243c8012bf17b4be601abfc8ea2`, identical to our pin; 0 commits
  since. Every citation in the bug one-pagers re-verified verbatim and
  permalinked in `analysis/code-drift-check.md`. Claim provenance classified
  in `analysis/claims-ledger.md`; fix plan in
  `analysis/fix-verification-protocol.md`.
- LLaVA-1.6 artifacts (empirical scope check): HF
  `cjpais/llava-1.6-mistral-7b-gguf` (the repo llama.cpp's own tests.sh:96
  uses) — `llava-v1.6-mistral-7b-Q4_K_M.gguf` sha256
  `4bd1bc95c4db74f8140ee520e76d1f83e063d3fde9c3723eaa4a4776785a7aa6`,
  `llava-v1.6-mistral-7b-mmproj-f16.gguf` sha256
  `00205ee8a0d7a381900cd031e43105f86aa0d8c07bf329851e85c71a26632d16`.
  Test input: `assets/phase1/uniform672.png` (uniform RGB 128, 672×672 —
  chosen so all 5 llava-uhd tiles are pixel-identical) sha256
  `d5f10997d701506834bfca6df9af773c5e26b12cbbb7e7809640ea90fc1b9b87`.
  Reference weights read directly from the mmproj GGUF (F16→F32) by
  `scripts/phase1/llava16_check.py` — no HF llava-1.6 checkpoint involved.
  mtmd emits one image chunk per tile for llava-uhd (5×576 tokens);
  `dump_llamacpp_embd.py` now concatenates all image chunks.
  Result: `results/20260717-082218_p1_llava16_check.json`.

## Fix verification session (2026-07-17, results tagged p2_*)

- Builds under test (fork ~/Desktop/vtp/llama.cpp, all branched from
  e8f19cc0, same cmake flags as the pin): fix-A `mtmd-fix-llava-cls-order`
  @ ab81d8fc1 (one-line concat swap), fix-B `mtmd-fix-llava-layer-count`
  @ f104a5d38 (il_last = n_layer, +1 branch deleted), combined
  `local-test-both` @ 5b9058635 (local-only octopus merge, never pushed).
  Separate build dirs (build-fixA/fixB/both); pinned build/ kept as the
  unfixed baseline binary.
- Variant flips exactly as predicted (results/*_p2_fixverify_*.json):
  fix-A → layerbug 0.9956; fix-B → clsbug 0.9986; combined → correct
  0.9965; all margins ~0.049. Combined 1.6 per-tile: correct 0.99997 on
  5/5 tiles, CLS-row cosine vs projected CLS drops 0.999998 → 0.1894.
- Timing: raw deltas vs the cold G0 baseline are thermally inflated (all
  configs' prefill drifted +6-10%); against a same-session warm master
  control (p2_bench_master_warm): fix-A encoder −1.8% (noise), fix-B +5.9%,
  combined +5.3% (predicted ~+4.5%), decode flat ±3%.
- End-task (paired fixed-vs-unfixed, same 200 samples, same Q4_K_M + F16
  files, p2_textvqa_*): fixed 54.95 vs unfixed 56.35, diff −1.40pp,
  bootstrap 95% CI [−4.65, +1.70], 8/9/183 wins/losses/ties — statistically
  zero; the OCR-assisted benchmark barely reacts to feature quality in
  either direction (matches the earlier unfixed-vs-HF null). PR case rests
  on representation-level correctness; a no-OCR or grounding-task rerun is
  the designated follow-up if an end-task number is wanted.
- Fix-B non-regression, proven empirically (p2_fixB_nonregression_glm_edge):
  GLM-Edge-V-2B (THUDM/glm-edge-v-2b-gguf Q4_K_M + F16 mmproj, SigLIP,
  full block_count + former il_last+=1 path) encoder output is
  BIT-IDENTICAL (np.array_equal over 578×2048) between unfixed master and
  build-fixB — the (N−1)+1 = N arithmetic holds in practice.
- Smoke tests (p2_smoke_tests_combined_build): the tests.sh vision
  invocations for second-state/Llava-v1.5-7B-GGUF:Q2_K and
  cjpais/llava-1.6-mistral-7b-gguf:Q3_K_M replicated verbatim against
  build-both — both PASS ("new york" criterion; raw logs in
  results/raw/20260717_p2_smoke/). tests.sh itself was not executed because
  it rebuilds build/ from the current checkout, which would have replaced
  the pinned unfixed baseline binary.

## Phase 1 verification addendum (stock-transformers check + end-task run)

- Reference validation: the hand-rolled fp32 reference (`hf_reference.py`
  "correct" variant) matches stock
  `LlavaForConditionalGeneration.get_image_features()` (config defaults
  `vision_feature_layer=-2`, `strategy=default`, fp32 vision path, identical
  pixel construction) at mean per-token cosine 0.9999999999 (min 0.99999996).
  Stock transformers vs llama.cpp output: 0.5010 mean cosine. The
  representation-level bug claim therefore rests on stock transformers code.
  Result: `results/20260717-062557_p1_stock_validation.json`.
- End-task run: the same 200-sample TextVQA slice through `llama-mtmd-cli`
  (pinned build/commit, frozen inference flags). Prompt parity with the HF
  baseline via a custom jinja chat template
  (`scripts/phase1/vicuna_v1_llava.jinja`, passed inline via
  `--jinja --chat-template`; this replaces `--chat-template vicuna` for
  accuracy runs only — builtin vicuna inserts different whitespace than the
  LLaVA vicuna_v1 format; inference flags unchanged). Scoring code identical
  (imports `textvqa_sim.vqa_accuracy`). Result: llama.cpp 56.35 vs HF-fp16
  58.50, paired diff -2.15pp, bootstrap 95% CI [-7.0, +2.75] — direction
  consistent with the bugs but NOT statistically resolvable at n=200, and
  confounded with Q4_K_M quantization. 165/200 exact ties; 196/200 prompts
  carry the OCR reference line, which makes this benchmark configuration
  weakly vision-sensitive. Result:
  `results/20260717-063925_p1_textvqa_llamacpp.json` (+ `.preds.jsonl`).
- Eval-set pin: `assets/phase1/textvqa200_manifest.jsonl` records the exact
  200 questions/answers/OCR tokens (first 200 of lmms-lab/textvqa validation
  in streaming order); images re-derivable from the dataset by index.

## Run protocol

- Per cell: 1 warm-up run (discarded) + ≥5 timed runs (G0 used 6);
  mean ± std reported, all raw values and logs kept under `results/`.
- Results JSON records: exact argv, prompt, config, environment (commit,
  build, CPU, OS), file SHA256s, per-run parses, aggregates, determinism
  check, and model provenance (repo + substitution reason).
- Observed in G0: prompt-eval drifted +5.5% monotonically across 6
  back-to-back runs (thermal). If drift persists in longer sweeps, add a
  fixed inter-run cooldown to the protocol and note it here.
