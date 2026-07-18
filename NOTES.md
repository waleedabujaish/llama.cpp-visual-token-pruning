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
- **Drift persisted and got worse (2026-07-17 Phase 2 session): resolved by
  a 30s inter-run cooldown.** See "Phase 2 re-baseline" below. `bench_baseline.py`
  now takes `--cooldown-s`; use it for every cell in the upcoming keep-ratio
  sweep (many more back-to-back invocations than G0 had).
- **Design requirement for the keep-ratio sweep script (not yet written):**
  with `--cooldown-s 30` and 6 keep-ratios x >=6 runs each x ~8s/run, the
  sweep is a multi-hour unattended job (6 cells x (1 warmup + 6 runs) x
  (8s run + 30s cooldown) is already ~25 min for one model/platform cell,
  and the full matrix (section 3 of the plan) is several platforms x
  metrics x ratios). It must be resumable: write each cell's result JSON
  to `results/` immediately on completion (not buffered until the whole
  sweep finishes), and on restart, scan `results/` for a cell already done
  under the current frozen config and skip it rather than re-running.
  `bench_baseline.py` already writes one JSON per invocation, so the sweep
  driver is the thing that needs the "skip if already done" check, not the
  underlying benchmark script.

## Phase 2 re-baseline (2026-07-17/18, fixed build)

All future pruning comparisons use this section's numbers, not G0's. G0 is
kept only as the historical "before the bugs were fixed" reference.

### Provenance bug found and fixed

`bench_baseline.py` and `scripts/phase1/textvqa_llamacpp.py` stamped result
JSON with `git rev-parse HEAD` on the llama.cpp checkout (or, in
`textvqa_llamacpp.py`, a hardcoded string) as "the commit under test." This
silently goes wrong once the checkout moves to a different branch than the
one a given `build*/bin/llama-mtmd-cli` was compiled from — which happened
between the 2026-07-17 16:31 fix-verification session and this one (repo
sat on `mtmd-fix-llava-layer-count` while `build-both/` was still the
`local-test-both` binary). Confirmed empirically: `git rev-parse HEAD` at
session start returned `f104a5d38...` (fixB branch) while `build-both/bin/
llama-mtmd-cli --version` self-reports `5b9058635` (local-test-both) — the
old code would have mis-stamped every result from this session.

Fixed by `scripts/llama_provenance.py` (shared by both scripts):
`resolve_build_provenance(bin, repo)` parses the *binary's own*
`--version` output (authoritative — it's baked in at compile time, printed
to stderr) and resolves the short hash to full via read-only
`git rev-parse <hash>^{commit}`. The repo's checked-out HEAD is also
recorded as a secondary field (`repo_head_commit`,
`repo_head_matches_binary`) so a future mismatch is visible in the JSON
instead of silently wrong. Verified against all four build dirs
(`build`, `build-fixA`, `build-fixB`, `build-both`) — each binary
self-reports exactly the commit its directory name implies.

Not retroactively fixed: `results/20260717-163502_p2_textvqa_llamacpp_fixed.json`
and siblings from the earlier session carry the old hardcoded
`"commit": "base e8f19cc0..."` (wrong; `build_note` on the same object
correctly says `local-test-both`). Left as-is rather than editing a
published result file after the fact; noted here for anyone reading that
file directly.

### Thermal drift: worse than G0, root-caused, fixed

First two re-baseline attempts (`results/20260717-234532_g2_baseline_fixed.json`,
`results/20260717-235252_g2_baseline_fixed_clean.json`, both build-both,
no cooldown) showed run1 near G0's numbers (~5300-5600ms prompt-eval) then
climbing and staying elevated (~6700-7200ms) for runs 2-6, with `uptime`
load average roughly tripling (2.2 -> 7.7) over each ~90s / 7-invocation
block. A same-session unfixed control
(`results/20260717-234854_g2_baseline_unfixed_control.json`) run
immediately after showed the identical pattern, ruling out "it's something
about the fixed build" — this is sustained `-t 8` thermal throttling on
the M4 Pro, worse today than during either G0 or the 16:31 session (both
of which had natural gaps between blocks from interactive command entry;
today's back-to-back automated blocks gave the CPU no recovery time).

Fix: added `--cooldown-s` to `bench_baseline.py` (sleep between every
invocation, including after warmup). A 30s cooldown eliminated the drift
almost entirely — see the `_cooled` results below, std dropped from
~70-90ms to ~1-12ms on encode/prompt-eval. Adopt `--cooldown-s 30` (or
longer) for the keep-ratio sweep, which will run far more back-to-back
cells than this baseline did.

The two uncooled attempts and the uncooled unfixed control remain in
`results/` as honest raw data (nothing fabricated, real executed runs) but
are superseded by the cooled runs below and are not the reported numbers.

### Official numbers (cooled, n=6 each, 30s cooldown)

Fixed build (`build-both`, `local-test-both` @ `5b9058635`):
`results/20260717-235645_g2_baseline_fixed_cooled.json`
- encoder: 709.67 +/- 1.03 ms
- TTFT_llm (prompt eval): 5268.20 +/- 12.50 ms
- TTFT_vlm: 5977.87 +/- 11.81 ms
- encoder fraction of TTFT_vlm: 11.87%
- Amdahl ceiling, simple (1/encoder_fraction): 8.42x
- Amdahl ceiling, refined (keep->0, image-token share of prefill only): 3.76x
- determinism: identical output across all 6 timed runs

Unfixed build (`build`, pinned `e8f19cc0`, same-session control):
`results/20260718-000051_g2_baseline_unfixed_cooled.json`
- encoder: 710.33 +/- 71.88 ms (one outlier run, see below)
- TTFT_llm: 5421.20 +/- 384.44 ms (same outlier)
- One of 6 runs (run3) spiked to encode=857ms/prompt_eval=6204ms with no
  precedent in the surrounding runs (immediate neighbors back at baseline) —
  read as a single transient background blip, not a recurrence of the
  block-level thermal drift above (that pattern is monotonic-ish and
  sustained; this is one isolated sample). Kept in the reported mean/std
  (no data removed); sensitivity check below excludes it explicitly and is
  labeled as such.

Side-by-side vs G0 (`results/20260717-011050_g0_baseline.json`, different
session, cross-session comparison — see caveat):
| metric | G0 (old, unfixed) | G2 fixed (cooled) |
|---|---|---|
| encoder | 692.83 +/- 11.48 ms | 709.67 +/- 1.03 ms |
| TTFT_llm | 5432.82 +/- 117.37 ms | 5268.20 +/- 12.50 ms |
| TTFT_vlm | 6125.66 +/- 119.49 ms | 5977.87 +/- 11.81 ms |
| encoder fraction | 11.31% | 11.87% |
| Amdahl simple | 8.84x | 8.42x |
| Amdahl refined | 3.88x | 3.76x |

Caveat on the cross-session comparison: G0 and G2 were measured on
different occasions (different pre-run thermal/frequency-scaling state);
TTFT_llm is not expected to depend on which build ran it (see paired
comparison below) and its ~3% difference here (5432ms vs 5268ms) is
cross-session variance, not a fix effect. Do not read "TTFT_vlm went down"
as "the fixes made things faster" — it is (small real encoder increase)
minus (larger unrelated cross-session prefill variance).

### Same-session paired fix-cost estimate (the number that isolates the fix)

Isolating what fix-B (running the previously-dropped 23rd vision layer)
actually costs requires same-session pairing, not cross-session comparison
(lesson from the thermal section above — also true for non-thermal
session-to-session variance). Three independent same-session measurements,
increasing rigor:

1. 2026-07-17 16:31 session (`p2_bench_both` vs `p2_bench_master_warm`,
   both warm, no cooldown needed — this session had natural gaps between
   blocks): encoder +5.26%, TTFT_llm +1.98%.
2. 2026-07-18 00:00 session, cooled, raw (includes the run3 outlier above):
   encoder -0.09%, TTFT_llm -2.82% — swamped by the single outlier in the
   6-sample unfixed block; not a usable estimate at n=6 with one bad draw.
3. Same session, outlier excluded as a sensitivity check (n=5 unfixed vs
   n=6 fixed): encoder +4.21%, TTFT_llm +0.07%.

Triangulated: fix-B costs roughly +4-5% encoder time (one extra ViT layer
of compute), and, as physically expected, does not measurably change
TTFT_llm (the LLM prefill consumes whatever embeddings the encoder
produced; how many vision layers computed them is invisible to it). The
correctness fixes are a real but small encoder-side cost, not a prefill
cost.

### TextVQA n=200, fixed build

Not re-run: `results/20260717-163502_p2_textvqa_llamacpp_fixed.json`
already is this measurement (same 200-sample pinned manifest, `build-both`
@ `local-test-both`, n_scored=200, n_failed=0) from the 2026-07-17 16:35
fix-verification session. Re-running would reproduce it exactly (temp=0,
seed=42, same binary/model/data all deterministic; G0 and this session's
`_cooled` runs both independently confirmed bit-identical output across
repeated runs) so it was not repeated. Numbers, already in NOTES.md's fix
verification section above and reconfirmed here:
- llama.cpp fixed: 54.95% (n=200)
- llama.cpp unfixed (stale — the number Task 1 was asked to replace): 56.35%
- paired diff (fixed - unfixed, same 200 samples): -1.40pp, bootstrap 95%
  CI [-4.65, +1.70], 8 wins / 9 losses / 183 ties for llama.cpp fixed
  — statistically null in both directions
  (`results/20260717-171500_p2_textvqa_paired_fixed_vs_unfixed.json`).
- The correctness fixes measurably change representation-level output
  (0.9965 mean cosine vs HF reference, see fix-verification section above)
  but this particular end-task benchmark is too OCR-line-dominated to
  register it; unchanged conclusion from the fix-verification session.

## Pruning acceptance gates (2026-07-18)

Implementation: fork branch `visual-token-pruning` @ `82f2ccb5c`
(`local-test-both` + `--visual-keep`/`--visual-prune-method`). Design doc:
`analysis/g2-hook-design-fixed-graph.md`.

### Task 0: build contamination and rebuild

The implementation session iteratively recompiled into `build-both/` and
`build/` (`cmake --build`, no reconfigure) while testing the pruning
branch. `build-info.cpp` only regenerates at cmake *configure* time
(`cmake/build-info.cmake`, included at `CMakeLists.txt:124`), so both
binaries kept reporting a stale `--version` (`5b9058635`) while their
`clip.cpp.o`/`llava.cpp.o` object files (confirmed newer than each
directory's `CMakeCache.txt`) actually contained the pruning branch's
code — confirmed directly by `--help | grep visual-keep` returning
matches on both. **This is a real hole in `llama_provenance.py`'s
"trust the binary's --version" design**: that assumption holds for a
binary from a fresh configure+build, not for one patched via incremental
`cmake --build` without reconfigure. `build-fixA`/`build-fixB` were
untouched (correct versions). `build/` (the pinned unfixed baseline) is
contaminated the same way; not rebuilt (nothing downstream needs it —
the committed G2 baseline in `results/` was measured 2026-07-17 ~23:45,
before the 2026-07-18 02:20 pruning commit, so those JSONs are unaffected;
they just can't be re-diffed against the binaries that produced them).

Rebuilt clean: `local-test-both` checked out into a separate git worktree
(`../llama.cpp-worktree-pristine`, `git worktree add`, does not disturb
the fork's main checkout) and fresh-configured+built into
`build-both-pristine/`; `build-prune/` fresh-configured+built from the
main checkout's `visual-token-pruning` HEAD. Both verified two ways
(version string is advisory only, per the above): `--version` matches the
expected commit, **and** `--help | grep visual-keep` is empty for the
pristine build-both and non-empty for build-prune.

### Gate 0.5 (added): determinism floor

Before trusting `np.array_equal` as the Gate 1 operator, confirmed the
CPU vision encode is deterministic run-to-run at `-t 8`: encoded
`cat_336.png` twice on pristine build-both, `np.array_equal` = True, max
abs diff = 0.0. `np.array_equal` is a sound bit-identity test here.

### Gate 1: keep=1.0 bit-identity — PASS

`scripts/phase1/dump_llamacpp_embd.py` needed a second ctypes struct
(`mtmd_context_params_pruned`) for the branch's larger
`mtmd_context_params` (two new fields inserted between
`image_max_tokens` and `cb_eval`) — the pre-pruning struct layout would
silently misalign every field from `cb_eval` onward if pointed at
`build-prune`'s dylib. Added `--pruned-abi`/`--visual-keep`/
`--visual-prune-method` flags, selected via the new struct when set.

- Bitwise: pristine build-both vs build-prune `--visual-keep 1.0` (both
  `--visual-prune-method none` and `cls` — the gate only requires
  `visual_keep < 1.0`, so both should no-op): `np.array_equal` = True,
  max abs diff = 0.0, in both cases.
  (`results/raw/.../pristine_run1.npy` vs `prune_keep1.npy`/`prune_keep1_cls.npy`.)
- Timing (cooled, n=5, `--cooldown-s 30`): pristine build-both encode
  721.4±7.7ms / prompt_eval 5306.3±15.7ms; build-prune@keep=1.0 encode
  711.0±1.6ms / prompt_eval 5351.9±25.1ms. Delta -1.44% encode, +0.86%
  prefill — within noise, no systematic direction. Identical generated
  text at temp=0 on both.
  (`results/20260718-024601_gate1_timing_build_both_pristine.json`,
  `results/20260718-024934_gate1_timing_build_prune_keep1.json`.)

### Layer alignment (verified against real config/GGUF values, not asserted)

HF `llava-hf/llava-1.5-7b-hf` vision config: `num_hidden_layers=24`,
`vision_feature_layer=-2` (read directly from the cached snapshot's
`config.json`). GGUF `clip.vision.block_count=23` (parsed directly from
the mmproj file's metadata bytes: type u32, value 0x17=23) — the
conversion script already dropped HF's 24th layer, so GGUF block index
== HF layer index (0-indexed) for blocks 0..22.

HF `hidden_states` is a 25-tuple (index 0 = embeddings, index i =
output after 0-indexed layer i-1); `hidden_states[-2]` = index 23 =
output of HF layer 22. HF `attentions` is a 24-tuple (index i = probs
from 0-indexed layer i); `attentions[-2]` = index 22 = probs from HF
layer 22. Both `[-2]` indices land on **HF layer 22** for different
reasons (the extra embeddings entry in `hidden_states`, not a shared
offset convention) — this is exactly `prune_viz.py`'s own
`cls_scores()`/`hf_reference.py`'s `Ref.tower(..., probs_layer=22)`.

C++ post-fix-B: `max_feature_layer = hparams.n_layer = 23`, loop
`il = 0..22`, last built = `il=22` = `v.blk.22` = HF layer 22 (block
index == HF layer index, established above). **Matches.** Pre-fix-B it
was `il=21` = HF layer 21 — off by one, matching G1's original finding.

### Gate 2: kept-index parity — not literal exact match; root cause characterized, not a pruning-code bug

`MTMD_DEBUG_GRAPH`'s tensor dump (`common/debug.cpp`) truncates every
tensor to 6 elements per dimension (hardcoded `n=3` at the one call
site) — it cannot recover a 576-element score vector or a >6-element
kept set. `scripts/phase1/gate2_kept_index_parity.py` recovers the C++
kept set a different way instead: dump the projector output at
`--visual-keep 1.0` (576 rows, spatial order) and at the ratio under
test (K rows), then nearest-neighbor-match each pruned row back to its
source row (the MLP projector is token-independent, so a kept row's
value is bit-for-bit the same computation as its keep=1.0 counterpart).
Match quality confirmed unambiguous throughout: nearest-match distance
was 9-53% of the second-nearest distance in every one of 15 cells (never
close to ambiguous).

Python reference: `hf_reference.Ref.tower(cls_first=True, n_layers=23,
probs_layer=22)` — exactly `prune_viz.py`'s already-validated
`cls_scores()`, matching the layer-alignment derivation above.

5 images (`assets/phase1/gate2/`, all pre-resized/cropped to exactly
336x336 for pixel-identical C++/Python input: the COCO cats image, three
TextVQA images resized, one native-resolution TextVQA crop) x keep in
{0.5, 0.25, 0.1} = 15 cells. 2/15 cells exact match
(`gate2_cat@0.25`, `gate2_textvqa_002@0.1`). 78 total mismatched patches
across the other 13 cells, out of 2450 total kept-slots evaluated
(sum of K x 5 images) — 3.2% mismatch rate.

Investigated, not just counted. Every mismatch was cross-checked against
a keep=1.0 comparison (Python's full unpruned projector output vs C++'s
`--visual-keep 1.0` output, all 576 tokens, zero pruning logic involved)
to separate "this patch's *encoder representation* already diverges
between the two implementations" from "the pruning-specific code (scoring
branch / top-K / gather) is doing something wrong":

- **37/78 (47%) tie-like**: score gap at the K/K+1 cutoff < 5e-5 (score
  std for a typical image is ~5e-3, so this is two orders of magnitude
  below typical spacing) and the disputed patch's embedding already
  agrees closely (cosine >=0.98) at keep=1.0. Ordinary boundary noise —
  many candidates are near-tied right at the cutoff rank.
- **30/78 (38%) pre-existing encoder divergence**: the disputed patch
  already had cosine <0.98 (many far lower — as low as 0.055-0.90)
  between C++ and Python **at keep=1.0**, i.e. before any pruning code
  runs at all. This is the *same* phenomenon Phase 1 already documented
  (`hf_reference.py`'s docstring: "F16 mmproj + FA fp16 K/V casts + ggml
  quick-gelu LUT vs fp32 reference keeps even the true-matching variant
  at ~0.998-0.999 mean cosine, not 1.0") — a mean cosine in that range is
  exactly what a small number of very-low-cosine outlier tokens averaged
  with hundreds of near-perfect ones produces. Confirmed this isn't a
  pruning-branch artifact: Gate 1 already proved build-prune@keep=1.0 is
  bitwise identical to pristine build-both, so this keep=1.0 comparison
  *is* a statement about the fixed graph itself, independent of the
  pruning branch.
- **11/78 (14%) not cleanly classified**: high embedding cosine at
  keep=1.0 (0.988-0.9995, ruling out the "already-diverged token" bucket)
  but a score gap larger than the tie threshold. Plausible read: the raw
  attention-probability score is a narrower, per-head-then-averaged
  quantity and may be more sensitive to small cross-implementation
  numerical noise than the fully-aggregated final embedding — but this
  wasn't verified directly (no C++-side raw score extraction was
  available; see limitation below), so it's reported as unexplained
  rather than asserted.

**Criterion amended (ruling, 2026-07-18):** literal exact-set-match
implicitly assumes identical numerics across two stacks that demonstrably
differ (F16 mmproj + quick-gelu LUT in ggml vs fp32 in the Python
reference), a gap documented since Phase 1, not something introduced by
this gate. The criterion that actually matters is "no divergence
attributable to the pruning code" — which the cross-check above
(tie-like / pre-existing-encoder-divergence / unclassified buckets)
supports, but stops short of proving quantitatively. Closed with one more
check, computed from already-existing data (no new llama.cpp binary
invocations — `scripts/phase1/gate2_epsilon_optimal.py` re-derives
Python scores fresh, deterministic, and reads the mismatch list straight
out of the Gate 2 results JSON):

**Epsilon-optimality of C++'s picks.** For every `cpp_only` mismatch (a
patch C++ kept that the Python reference did not), computed
`gap = python_cutoff_score - python_score_of_cpp_pick`, normalized by
the image's own score std. A small gap means C++'s "wrong" pick is a
near-tied alternative under the reference's own scoring, not a bad one.

- All 39 gaps non-negative (sanity check: C++'s picks are, correctly,
  always at or below the Python cutoff when scored by the reference).
- Overall: max 0.590 std, median 0.008 std, mean 0.045 std.
- Known-diverged-token bucket (n=14): max 0.590, median 0.022 std —
  the one large outlier lives here, on the already-explained bucket.
- **Other bucket (n=25, the one that could reopen the gate): max 0.250
  std, median 0.004 std.** No entry anywhere near 1 std, let alone
  large. C++'s picks are epsilon-optimal under the reference scoring
  even outside the already-explained divergent-token bucket.
  (`results/20260718-043401_p2_gate2_epsilon_optimal.json`.)

**Gate 2: PASS-AMENDED.** Literal exact-set-match does not hold (13/15
cells have at least one mismatch), but every mismatch traces to
cutoff-boundary score noise or the pre-existing (pruning-independent)
encoder-level numerical floor on record since Phase 1, and the
epsilon-optimality check confirms even the unclassified 14% are
near-tied picks, not meaningfully wrong ones. No pattern anywhere
implicates the pruning code (no fixed offset, no consistent direction,
no image-independent recurrence). Scoring branch, top-K, and gather are
correct; the residual mismatch is entirely inherited, not introduced.

**Deferred (future work, not blocking):** direct extraction of the C++
scoring branch's raw `cls_scores` values (a ctypes `ggml_tensor` struct
binding to intercept `cb_eval`) would let the 14% unclassified bucket be
attributed with certainty instead of by inference from embedding cosine.
Not needed to close Gate 2 given the epsilon-optimality result above;
worth building if a future change to the scoring branch needs finer-grained
debugging than row-matching + keep=1.0 cross-checks can provide.

Full data: `results/20260718-025754_p2_gate2_kept_index_parity.json`,
`results/20260718-043401_p2_gate2_epsilon_optimal.json`
(+ raw dumps under `results/raw/20260718-025754_p2_gate2_kept_index_parity/`).

## Keep-ratio sweep (2026-07-18)

`scripts/sweep_prune.py` (resumable driver) x `scripts/bench_baseline.py`
(now with `/usr/bin/time -l`-based peak RSS/footprint and KV-buffer-size
capture). build-prune, LLaVA-1.5-7B, keep in {1.0, 0.75, 0.5, 0.25, 0.1,
0.05}, n=5 + warmup, `--cooldown-s 30`, otherwise the frozen G2 config.
All 6 cells clean on the first pass, tight variance throughout except the
keep=1.0 cell (see below). Full table, curves, and methodology notes:
`results/p2_sweep_report.html` (also published as an artifact this
session). Raw per-cell JSONs: `results/*_p2_sweep_keep*.json`. Analysis:
`scripts/sweep_analysis.py` -> `results/p2_sweep_analysis.json`.

**Headline:** TTFT_llm speedup 1.0x -> 5.25x (keep=1.0 -> 0.05), realizing
88-100% of the theoretical ceiling (H1: linear-in-token-count prefill
scaling) at every ratio tested. TTFT_vlm falls monotonically throughout
the tested range - no break-even where pruning overhead exceeds savings
was observed (H2 candidate not located within {1.0..0.05}; if a
break-even exists it's below keep=0.05, not reached here). Encoder share
of TTFT_vlm triples, 12.1% -> 39.1%, because encoder time is nearly flat
(742.8ms -> 660.4ms, all 576 patches cross all 23 ViT layers regardless
of keep ratio - pruning only shrinks the post-encoder token count) while
prefill collapses around it (5407ms -> 1030ms).

**Qualitative degradation, unprompted finding:** generated text (temp=0,
same image, same prompt) is byte-identical across keep in {1.0, 0.75,
0.5, 0.25} ("two cats lying on a pink couch..."). At keep=0.1 it drifts
(leads with "a couch", drops "sleeping peacefully" - same cats, softer
phrasing). At keep=0.05 (29 image tokens) the model hallucinates: "a
couch with two remote controls" in place of the two cats. Single image,
not a systematic eval, but a real failure mode surfaced by the sweep
itself, worth carrying into the eventual accuracy work rather than only
reporting speedup.

**Prune-overhead isolation** (methodology per this session's ruling: fit
`encode_ms = A + B*K` over the pruned cells only, keep<1.0, since
keep=1.0 runs a structurally different code path with the branch gated
off entirely - extrapolate to K=576, subtract the *measured* keep=1.0
encode_ms; the difference isolates the scoring/top-K/gather branch's
cost since the two quantities differ only by its presence):

- Fit (K=29..432, n=5): A=657.44ms, B=0.0806ms/token, R²=0.9918,
  max|residual|=1.84ms - an excellent linear fit within its own range.
- Fitted encode_ms at K=576: 703.84ms. Measured keep=1.0: 742.80ms.
  Point-estimate overhead: **-38.96ms** (fit predicts *less* time than
  what the genuinely-unpruned path measured).
- **Not reported as a real negative-cost effect - it isn't distinguishable
  from noise.** The keep=1.0 cell's own run-to-run std is 32ms, roughly
  the same magnitude as the estimate itself, and is visibly higher than
  every pruned cell's std (1-6ms): raw encode_ms across its 5 runs was
  730/709/722/775/778ms, drifting upward within the cell despite the 30s
  cooldown (load average rose 2.0->3.7 over the course of just this one
  cell). Most likely explanation: mild residual thermal noise specific to
  this cell (first in the sweep), not a genuine effect. Also worth
  flagging: K=576 is 144 tokens beyond the fit's own max (432), so this
  is an extrapolation, not an interpolation - a tight in-range fit doesn't
  guarantee accuracy 33% past its domain.
- **Honest reading:** the scoring+selection branch's added cost is small
  enough to be lost in single-cell measurement noise at this precision -
  not proven to be ~0ms, but bounded to a small fraction of encode_ms
  (which itself is ~660-740ms) either way. A cleaner isolation would need
  either a much larger n on the keep=1.0 cell specifically, or the
  deferred ctypes raw-op-timing tool (Gate 2's future-work item) applied
  to time the scoring branch's ops directly rather than inferring their
  cost from a fit residual.
