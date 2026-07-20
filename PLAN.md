# Visual Token Pruning on CPU — Project Plan

**One-line pitch:** this project adds attention-based visual token
pruning (`--visual-keep`) to llama.cpp's multimodal inference path
(`tools/mtmd`) and measures what it actually does to prefill latency,
decode speed, and downstream task accuracy — run identically across
three hardware backends (Apple M4 Pro CPU, an x86 Xeon CPU, and a Tesla
P100 GPU) to find out whether the speedup pruning delivers depends on
which backend runs it.

Every claim below is tagged by evidence status:
- **[MEASURED]** — from an executed run, JSON in `results/` with provenance.
- **[VERIFIED]** — confirmed by code reading and/or an empirical check on record.
- **[HYPOTHESIS]** — the thing the research exists to test; not yet known.
- **[PENDING]** — planned, not yet done.

Full raw numbers, per-run data, and methodology detail live in `NOTES.md`;
this file is the summary. Nothing here should read as more resolved than
`NOTES.md`'s own record of it.

---

## Central hypotheses

### H1 — CPU realizes more of the pruning speedup than GPU does

**Pre-registered prediction:** GPU execution carries proportionally more
fixed per-call overhead (kernel launch, memory allocation, pipeline
stages that don't shrink with fewer tokens) relative to its own total
run time than CPU execution does, so pruning should translate into a
smaller fraction of its theoretical speedup on GPU than on CPU.

**Status: [MEASURED, CPU and GPU both] — confirmed.**

Metric: fraction of the theoretical linear-in-token-count ceiling
realized by the actual `TTFT_llm` (LLM prefill) speedup, at each
keep-ratio.

| Platform | `TTFT_llm` speedup, keep 1.0→0.05 | Ceiling fraction, keep 0.75→0.05 |
|---|---|---|
| Apple M4 Pro (CPU, dedicated) | 1.0x → 5.25x | 87.8–99.9% |
| x86 Xeon (CPU, shared GitHub Actions runner) | 1.0x → 4.00x | 77.8–81.7% |
| Tesla P100 (GPU) | 1.0x → 1.98x | 43.5–67.7% |

GPU's ceiling fraction sits well below both CPU platforms' at every
shared ratio. The mechanism traces to absolute timing: unpruned GPU
prefill takes about 1.1 seconds, versus several seconds on CPU, so
overhead that doesn't scale with token count — kernel launches, memory
allocation, other fixed pipeline costs — makes up a much bigger fraction
of GPU's total time and limits how much of the token-count reduction
actually reaches wall-clock speedup. The prune-overhead fit
(`encode_ms = A + B·K` across pruned cells) shows the same asymmetry
from another angle: on GPU it comes out tight and positive (+7.94ms,
R²=0.945), a real and resolvable cost, while on both CPU platforms the
equivalent estimate is negative and not separable from measurement
noise. Decode speed improves with pruning on every platform — fewer
image tokens sitting in the KV cache means less per-step attention work
— but the GPU gain (+2.2%) is far smaller than either CPU's (M4 +21.4%,
x86 +9.1%): GPU's KV-cache attention is already fast and parallel, so
trimming it matters comparatively little.

The x86 run is a qualitative generalization check, not a controlled
second data point against M4: different core count, no BLAS backend
configured (vs. M4's Apple Accelerate), and a contended shared runner
(1-min load average 5.5–7.2 throughout the sweep) rather than a
dedicated machine. Its direction (speedup grows monotonically, decode
improves, determinism holds) replicates M4's; its absolute numbers are
not directly comparable.

Sources: `NOTES.md` "Keep-ratio sweep", "x86 CPU platform sweep", "GPU
platform sweep"; `results/p2_sweep_analysis.json`,
`results/p2_sweep_x86_analysis.json`,
`results/p2_sweep_kaggle_gpu_analysis.json`, `results/p2_sweep_report.html`.

### H2 — a break-even point where pruning overhead exceeds its savings

**Pre-registered prediction:** somewhere near 5–10% keep ratio,
per-token scoring/top-K/gather overhead should start costing more than
the savings from fewer tokens, producing a measurable turnaround in
`TTFT_vlm`.

**Status: [MEASURED symptom; MECHANISM unresolved] on both platforms
tested — for different reasons on each.**

On M4, the swept range was extended past the original {1.0..0.05} grid
to {0.03, 0.02, 0.015, 0.01} specifically to locate this. `TTFT_vlm`'s
mean does stop decreasing and turn upward below keep=0.05 (keep=0.05:
1690.5ms; keep=0.03: 2566.0ms, +52%) — the precondition is measured.
Whether that rise reflects the pre-registered mechanism (overhead
genuinely exceeding savings) or OS-level measurement noise dominating
once per-run time drops under roughly 2 seconds is **not established**:
run-to-run variance explodes at this token scale (encoder std widens
from roughly ±1–6ms up to ±65–117ms); the curve below the floor doesn't
move in one direction (keep=0.03's mean comes out above both
keep=0.015's and keep=0.01's, even though it has more image tokens, not
fewer); and the fastest individual runs at keep=0.02 and keep=0.01 come
in below the keep=0.05 mean entirely. None of that is what a clean
algorithmic floor would look like — it's the signature of noise
dominating once run times get short.

The same extension was run on the GPU platform. Before running it, a
cleaner break-even was expected than CPU's noise-dominated pattern —
that did not hold. `TTFT_vlm` instead stays close to flat from keep=0.1
through keep=0.015, then drops sharply at keep=0.01, and per-cell
`TTFT_vlm` std stays tight (±9–20ms) throughout — a sharp contrast with
CPU's own `TTFT_vlm` std in the same sub-0.05 range, which balloons to
±307–576ms (encoder std alone widens to ±65–117ms). The result is
neither of the two anticipated shapes: measurements stay precise
cell-to-cell, yet the mean itself moves unevenly rather than settling
into a clean turnaround.

**Either way, the operating recommendation is the same:** don't prune
below keep~0.05. If the sub-0.05 slowdown is real, going further loses
time for nothing; if it's measurement noise and the true achievable time
is actually fine, keep=0.05 itself already produces unreliable output
(see Limitations) before that region is even reached — so there's no
regime down there worth using regardless of which explanation turns out
to be correct.

Sources: `NOTES.md` "Sweep extension for H2", "H2 sub-0.05 GPU
extension"; `results/p2_sweep_analysis.json`,
`results/p2_sweep_kaggle_gpu_analysis.json`.

### H3 — the accuracy/latency tradeoff under weight quantization

**Pre-registered prediction:** the pruning method's accuracy/latency
tradeoff should hold under GGUF weight quantization generally, not just
the specific configuration used throughout this project (Q4_K_M main
model, F16 mmproj) — no pathological interaction between quantization
error and pruning-induced token loss.

**Status: [HYPOTHESIS] — untested.** Every benchmark cell in this
project, on every platform, uses the same fixed Q4_K_M/F16 configuration
(`NOTES.md` "Frozen run configuration"). No alternate-quantization sweep
has been run; this remains open.

---

## Scope and method

**Correctness baseline.** While validating the measurement harness, two
correctness defects turned up in llama.cpp's LLaVA vision path,
independently verified against the HF reference implementation: the
[CLS] token is concatenated after the patch embeddings but consumed as
if it were first (shifting position embeddings for every patch, dropping
patch 0's output, and feeding the projected [CLS] output to the LLM as
an image token), and a vision layer intended to be skipped is dropped
twice — once at GGUF conversion, once at graph build — so features come
out one layer short of the intended `vision_feature_layer`. Both are
fixed on the build this project runs everywhere; every measurement in
this plan is on that fixed build. The [CLS]-ordering fix specifically
matters beyond correctness: the pruning method below scores by the
[CLS] row, so it depends on [CLS] actually sitting at row 0, which only
holds once that fix is applied. Write-ups: `README.md`,
`analysis/bug-a-cls-ordering.md`, `analysis/bug-b-layer-count.md`.

**Method.** A single training-free pruning method: rank each image patch
token by how much attention the [CLS] token pays to it (FasterVLM-style),
and keep only the top-K, applied where the vision encoder's output feeds
into the projector. The complication is that the encoder's main
attention runs with flash attention enabled on CPU, which never produces
a readable attention-probability tensor — so there is nothing to read
the [CLS] row's scores off of directly. The implementation instead adds
a small, separate scoring computation that reuses the query/key values
already computed for that layer's main attention (no extra projection
work), at a cost of roughly 0.6 MFLOP for LLaVA-1.5's 577-position
encoder. Design: `analysis/g2-hook-design-fixed-graph.md`.

**Interface.** `--visual-keep <0.0-1.0>` and
`--visual-prune-method cls`, implemented on a llama.cpp fork
[MEASURED, implementation done].

**Acceptance gates:**
- **Gate 1 (keep=1.0 bit-identity): PASS.** `--visual-keep 1.0` must
  produce output bitwise identical to the unpruned path (`np.array_equal`
  on the projector output, both `--visual-prune-method none` and `cls`).
  `results/20260720-214857_gate1_bitwise_cpu.json` (CPU, re-verified with
  archived `.npy` dumps); Gate 1 also passed independently on the x86 and
  GPU platforms (`NOTES.md` "x86 CPU platform sweep", "GPU platform
  sweep").
- **Gate 2 (kept-index parity vs. an independent Python prototype):
  PASS-AMENDED.** Literal exact-set-match does not hold (13 of 15
  test cells have at least one mismatched kept-token index out of 2450
  total kept-slots evaluated, a 3.2% mismatch rate). Every mismatch was
  traced, not just counted: 47% are cutoff-boundary score ties, 38% are
  pre-existing encoder-representation divergence between the C++ and
  Python stacks that exists independently of pruning (present already at
  keep=1.0, before any pruning code runs), and the remaining 14% are
  epsilon-optimal under the reference's own scoring (no gap approaching
  1 standard deviation). No mismatch pattern implicates the pruning
  code itself. `NOTES.md` "Pruning acceptance gates";
  `results/20260718-025754_p2_gate2_kept_index_parity.json`,
  `results/20260718-043401_p2_gate2_epsilon_optimal.json`.

**Model.** LLaVA-1.5-7B (GGUF, `second-state/Llava-v1.5-7B-GGUF` repack,
Q4_K_M main model + F16 mmproj) — the model used throughout this project.

**Non-goals / out of scope.** The method requires a CLS token to score
against; it does not transfer as-is to encoders without one.
[VERIFIED empirically] SigLIP-based encoders (e.g. GLM-Edge-V-2B, used
elsewhere in this project as a non-regression check) load no
`class_embedding` tensor at all — confirmed by scanning the actual GGUF
file used, not just from documentation — so no CLS row exists to score.
Qwen2.5-VL and other no-CLS-token encoders are explicitly out of scope
for this method as designed; supporting them would need a different
saliency signal (e.g. norm- or attention-mean-based), which this project
does not implement or evaluate. Source:
`analysis/g2-hook-design-fixed-graph.md` §7.

---

## Benchmark matrix

**Platforms:** Apple M4 Pro (CPU-only, dedicated, Apple Accelerate BLAS);
x86 Xeon (CPU-only, GitHub Actions shared runner, no BLAS backend
configured); Tesla P100 (GPU, CUDA build).

**Keep ratios:** primary matrix {100%, 75%, 50%, 25%, 10%, 5%} on all
three platforms; extended to {3%, 2%, 1.5%, 1%} on M4 and GPU only, to
chase H2 (see above).

**Metrics and where each result lives:**

| Metric | Platforms | `NOTES.md` section | Key `results/` files |
|---|---|---|---|
| `TTFT_llm` / `TTFT_vlm` latency, decode tok/s, prune-overhead fit | M4, x86, GPU | "Keep-ratio sweep", "x86 CPU platform sweep", "GPU platform sweep" | `p2_sweep_analysis.json`, `p2_sweep_x86_analysis.json`, `p2_sweep_kaggle_gpu_analysis.json`, `p2_sweep_report.html` |
| Sub-0.05 extension (H2) | M4, GPU | "Sweep extension for H2", "H2 sub-0.05 GPU extension" | same analysis files (combined 10-cell tables) |
| Peak/KV memory | M4 only | "Keep-ratio sweep" | `p2_sweep_analysis.json` (`peak_footprint_mib_mean`, `kv_buffer_mib`) |
| TextVQA accuracy, 200-sample paired bootstrap | GPU, M4 | "GPU platform sweep", "CPU-vs-GPU accuracy parity spot-check" | `*_p3_textvqa_{kaggle_gpu,m4_cpu}_keep*_summary.json`, `20260720-215027_p3_textvqa_{kaggle_gpu,m4_cpu}_paired_bootstrap.json` |
| CPU-vs-GPU generation/accuracy parity | M4 vs. GPU | "CPU-vs-GPU accuracy parity spot-check" | `20260719-002506_p3_textvqa_cpu_gpu_parity.json` |
| POPE object-hallucination (accuracy/precision/recall, 3 categories) | GPU | "POPE object-hallucination sweep" | `*_p4_pope_kaggle_gpu_keep*_summary.json` |

**Accuracy headline [MEASURED].** TextVQA (200 samples, paired bootstrap,
10000 resamples): no keep ratio on either platform shows a statistically
significant accuracy difference from keep=1.0 — every 95% CI crosses
zero, including keep=0.05. The loss/win split leans toward degradation
as keep decreases (directionally consistent, not statistically
resolved at n=200). POPE (300 questions, 3 categories, all 6 ratios):
accuracy degrades from 85.0% (keep=1.0) to 79.7% (keep=0.05) with an
interpretable mechanism — precision rises (90.1%→95.9%) while recall
falls (78.7%→62.0%) as the model becomes more conservative under
pruning; the adversarial category is hardest at every ratio tested,
consistent with POPE's own design. The POPE question set covers only 17
unique underlying images (a manifest-generation artifact, not a
correctness issue) — less visually diverse than originally intended.

CPU-vs-GPU backend agreement [MEASURED]: aggregate TextVQA accuracy
parity holds cleanly (max difference 1.20pp across all 6 ratios, well
inside the ~±3.5pp per-cell noise floor); per-sample generation text
agreement is high but not perfect and decreases as keep drops (98.5%
exact match at keep=1.0 down to 91.5% at keep=0.05) — a small,
mildly GPU-leaning correctness-divergence rate (~1.8% of 1200
comparisons), made of plausible near-miss errors on both backends, not
qualitatively different failures.

---

## Limitations

**Negative-result stance.** If H1 doesn't hold on a platform not yet
tested, or H2's mechanism eventually turns out to be measurement noise
rather than a genuine algorithmic floor, either outcome gets recorded as
it stands, the same as every other result in this project — the
operating recommendation (don't prune below keep~0.05) doesn't change
either way.

**Method scope (CLS-family only).** As noted above, this pruning method
requires a CLS token; it is verified not to transfer to SigLIP/no-CLS
encoders as-is, and Qwen2.5-VL-style architectures are out of scope.
Results in this project should be read as characterizing this method on
CLIP-family (CLS-having) encoders specifically, not visual token pruning
in general.

**Platform caveats.** The x86 CPU result is a qualitative generalization
check, not a controlled quantitative comparison to M4 — different core
count, no BLAS backend configured, and a contended shared runner rather
than a dedicated machine. GPU's `frac_ceiling_h1` curve is also
non-monotonic across ratios (67.7%→43.5%→52.4%→55.4%→53.6%); the
per-cell std stays tight there too, which argues against simple
measurement noise as the explanation, but the cause is flagged, not
explained. Memory (peak footprint / KV buffer size) was
only captured on M4; a headline finding there (KV buffer size is a
constant allocation independent of keep ratio, so it does not shrink
with pruning) has not been independently checked on x86 or GPU.

**Single-image qualitative finding, not a systematic eval.** A specific
hallucination (the model describing "two remote controls" in place of
two cats) first surfaced from a single fixed test image at keep=0.05 and
reproduces identically across all three platforms at that ratio. On M4,
where the sub-0.05 range was explored further with the same single-image
test, it gets stranger, not better, at every ratio tested below that;
the GPU sub-0.05 extension recorded only latency data (no generated
text), and no sub-0.05 sweep was run on x86 at all. This is a real,
reproducible failure mode, but it was observed on one image — the
systematic accuracy evidence for degradation at aggressive ratios is the
TextVQA and POPE sweeps above, not this single case.

**Accuracy statistical power.** At n=200 (TextVQA) the per-cell standard
error is roughly ±3.5pp; none of the accuracy differences reported in
this project reach statistical significance at that sample size, on
either platform, in either direction. The directional lean toward
degradation at aggressive keep ratios is consistent across every
accuracy comparison run in this project but is not, on its own,
a resolved effect.

**Quantization untested (H3).** See above — every result in this
project uses one fixed quantization configuration.
