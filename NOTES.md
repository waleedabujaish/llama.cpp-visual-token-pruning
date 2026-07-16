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

## Run protocol

- Per cell: 1 warm-up run (discarded) + ≥5 timed runs (G0 used 6);
  mean ± std reported, all raw values and logs kept under `results/`.
- Results JSON records: exact argv, prompt, config, environment (commit,
  build, CPU, OS), file SHA256s, per-run parses, aggregates, determinism
  check, and model provenance (repo + substitution reason).
- Observed in G0: prompt-eval drifted +5.5% monotonically across 6
  back-to-back runs (thermal). If drift persists in longer sweeps, add a
  fixed inter-run cooldown to the protocol and note it here.
