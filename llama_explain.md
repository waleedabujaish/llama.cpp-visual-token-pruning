# How llama.cpp turns an image into LLM tokens — and two bugs we found in it

This is the simple version. It explains two real correctness bugs in llama.cpp's
LLaVA vision path — what they are, why they happened, how we know they're real
(not a guess), and what the fix does. Full rigor (per-model tables, every
number, citation permalinks) lives in `analysis/`. Read that if you need to
double-check a claim or write it up formally; read this to understand the
story.

Both bugs live in one file: `tools/mtmd/models/llava.cpp`, in the function
that builds the vision encoder's compute graph. Both are fixed on separate
git branches, each a tiny diff.

## Step 0 — how an image becomes "tokens" at all

Before the bugs make sense, you need the shape of the pipeline:

```
image (336×336 pixels)
   → cut into 24×24 = 576 patches, each turned into a vector ("patch embedding")
   → one extra learned vector is added: the [CLS] token
     (577 rows total: 576 patches + 1 CLS)
   → runs through a stack of transformer layers (self-attention + feed-forward)
   → LLaVA deliberately stops one layer early — it uses the SECOND-TO-LAST
     layer's output, not the last one, because that's what its projector
     was trained on ("vision_feature_layer = -2" in the original model config)
   → the [CLS] row is dropped, leaving 576 patch rows
   → those 576 rows go through a small MLP ("the projector") that maps them
     into the LLM's embedding space
   → the LLM receives 576 "image tokens" mixed into the prompt
```

The [CLS] token itself is a leftover from image classification: a row that
doesn't correspond to any patch of the image, whose job is just to accumulate
information from all the other rows via attention. In LLaVA it's never sent
to the LLM — it exists only to help the patches build better representations
of themselves, then gets thrown away before the projector.

Both bugs below are about getting these mechanics slightly wrong in the C++
implementation — not wrong enough to crash, wrong enough to quietly change
the numbers.

---

## Bug A — the [CLS] token ends up in the wrong seat

### The problem, in one sentence

The code appends the [CLS] row to the *end* of the 577-row sequence, but
every other part of the code that touches those rows still assumes CLS is at
the *front* — so patches get shifted position embeddings by one, the first
real patch gets thrown away instead of CLS, and the projected CLS vector
gets sent to the LLM disguised as the 576th image token.

### Why it happened

`tools/mtmd/models/llava.cpp:36`:

```cpp
inp = ggml_concat(ctx0, inp, model.class_embedding, 1);   // patches, then CLS
```

That's CLS-last. But two other pieces of code, a few hundred lines away in
`clip.cpp`, were never updated to match:

- `clip.cpp:4095-4099` fills the position-embedding lookup with `0, 1, 2, ...,
  576` — i.e. it assumes row 0 is CLS (position 0 in the original CLIP model
  really does belong to CLS).
- `clip.cpp:4101-4108` selects which rows go to the projector using indices
  `[1, 2, ..., 576]` — i.e. "skip row 0, keep everything else," which is only
  correct if row 0 is CLS.

Both of those were written for the *original* layout (CLS first), where they
were correct. A refactor in 2025 (`32916a490`, PR #13321) rewrote how the
graph is built and moved CLS to the end of the concatenation — but nobody
updated the two consumers to match. This is a classic "two pieces of code
that have to agree with each other, and a refactor touched only one of
them" bug.

### The three consequences

1. **Every patch gets the wrong position embedding**, shifted by one slot
   (patch *i* receives the position that belongs to patch *i+1*; the first
   patch receives CLS's position).
2. **The real first patch's output is thrown away** instead of CLS's (the
   "skip row 0" logic now skips a real patch).
3. **The [CLS] vector is sent to the LLM** as if it were the 576th image
   token — something that was never supposed to leave the vision encoder.

### The fix

One line, `llava.cpp:36`:

```cpp
// before
inp = ggml_concat(ctx0, inp, model.class_embedding, 1);
// after
inp = ggml_concat(ctx0, model.class_embedding, inp, 1);
```

Just swap the operand order so CLS goes first again. Nothing else needs to
change — the position and patch-selection code becomes correct again for
free, because it was written for exactly this layout in the first place.
This restores byte-for-byte the layout the *original* 2023 LLaVA
implementation used, before the 2025 refactor.

### How we know this is real, and not a misreading of the code

Reading the code tells you what *should* happen. It doesn't prove that's
what actually happens at runtime, or that the fix is right. So the
diagnosis was checked against real numbers, in stages:

1. **Sanity-check the reference first.** Before trusting any comparison, the
   Python re-implementation of Hugging Face's vision pipeline was checked
   against real Hugging Face output: **0.9999999999 cosine similarity** —
   essentially identical. So the reference itself isn't the source of any
   disagreement found later.
2. **Compare llama.cpp's real output to that trusted reference:** only
   **~0.50 mean cosine similarity** (1.0 = identical, 0.0 = unrelated). That's
   not a rounding error — it's a real computational difference.
3. **Reproduce the *exact* diagnosed bug independently, in plain Python**
   (CLS last, positions filled 0..576, patches gathered as rows 1..576), and
   compare that simulation to llama.cpp's actual output: **0.998 mean / 0.99998
   median cosine similarity** — essentially the same computation. A "control"
   simulation with CLS correctly placed first only reaches 0.505 against
   llama.cpp's real output. This is the step that turns "I think this is the
   bug" into "this is provably the bug" — an independent implementation of
   the hypothesized mechanism reproduces the real broken output almost
   exactly, and nothing else tested comes close.
4. **A direct, easy-to-check prediction:** if CLS really is being sent to the
   LLM as the last image token, the 576th row of llama.cpp's output should
   equal the projected CLS vector almost exactly. Measured: **0.9999967
   cosine**. It does.
5. **Checked on LLaVA-1.6 too**, which splits large images into up to 5
   tiles — the same pattern shows up independently in every tile, not just
   once by coincidence.

### Where to look

- Branch: `mtmd-fix-llava-cls-order`, commit `ab81d8fc1` — `mtmd : fix llava
  CLS token ordering (#25814)`.
- Full write-up with every model affected: `analysis/bug-a-cls-ordering.md`.
- Reproduction scripts: `scripts/phase1/repro_cls_bug.py`,
  `scripts/phase1/compare_embd.py`.

---

## Bug B — a transformer layer gets skipped twice

### The problem, in one sentence

LLaVA is supposed to skip exactly one layer (the very last one, per
`vision_feature_layer = -2`) — but the file that converts the model to
llama.cpp's GGUF format *already* removes that layer before llama.cpp ever
sees it, and then the C++ graph-building code subtracts one *again*,
skipping a second, real layer that was supposed to run.

### Why it happened

The conversion script, `legacy-models/convert_image_encoder_to_gguf.py:281`:

```python
block_count = v_hparams["num_hidden_layers"] - 1 if has_llava_projector else v_hparams["num_hidden_layers"]
```

and `:349-353` physically deletes that last layer's weights before writing
the file. So for a standard 24-layer CLIP vision tower, the GGUF file stores
**23 layers** — and `block_count = 23` is written into the file's metadata to
say so. The skip already happened, once, correctly, at conversion time.

The C++ graph builder doesn't know that. `llava.cpp:15`:

```cpp
int il_last = hparams.n_layer - 1;   // hparams.n_layer is read straight from block_count
```

It reads `n_layer = 23` from the file and subtracts one *again*, computing
`22`. So llama.cpp only runs 22 of the 23 layers that are actually stored and
loaded into memory. The 23rd layer's weights (`v.blk.22`) sit in RAM,
correctly loaded, and are never used.

This one is old — it's been there since the very first LLaVA support was
added in 2023 (`370359e5b`, PR #3436). Every later refactor carried it
forward unchanged, including the 2025 Granite-Vision PR that reorganized
this logic into its current form.

### The fix

Three lines, `llava.cpp:14-20`:

```cpp
// before
int il_last = hparams.n_layer - 1;
...
if (proj_type == PROJECTOR_TYPE_MINICPMV || proj_type == PROJECTOR_TYPE_GLM_EDGE) {
    il_last += 1;
}
// after
int il_last = hparams.n_layer;
// (the whole MINICPMV/GLM_EDGE block above is deleted)
```

The `+= 1` block existed only to *undo* the extra `-1` for two model families
(MiniCPM-V, GLM-Edge) whose own conversion scripts store the *full* layer
count instead of pre-trimming it — a second, independent convention that
happened to collide with the first. Once the base case stops subtracting an
extra layer, that compensation isn't needed for those two, and they land on
the correct value either way. Granite Vision has its own explicit
`feature_layers` override earlier in the function and is untouched by this
change.

| family | layers stored in file | layers run, before | layers run, after |
|---|---|---|---|
| llava / mobilevlm / yi (block_count = n−1) | 23 | 22 ❌ | 23 ✓ |
| MiniCPM-V (stores full count) | N | N | N (unchanged) |
| GLM-Edge (stores full count) | N | N | N (unchanged) |
| granite-vision (explicit override) | max(feature_layers) | override | override (unchanged) |

Side effect worth knowing, not a problem: fixing this makes the encoder do
slightly more work (running the 23rd layer instead of skipping it), about
**+4.5% encoder compute time**. Since the encoder is roughly 11% of total
CPU time-to-first-token on this setup (measured baseline,
`results/20260717-011050_g0_baseline.json`), the end-to-end slowdown is well
under 1%. That's the *correct* behavior now running, not a regression to
explain away.

### How we know this is real

Same method as Bug A — read the code, form a specific hypothesis, reproduce
it independently, compare to the real thing:

1. A plain-Python simulation using **22 layers**, with CLS ordering held at
   whatever llama.cpp actually does (so this test isolates the layer-count
   question specifically): matches llama.cpp's real output at **0.998 mean /
   0.99998 median cosine**.
2. The same simulation using the **intended 23 layers**: only **0.949**
   cosine against llama.cpp's real output — clearly worse, confirming
   llama.cpp really is running 22, not 23.
3. Checked on LLaVA-1.6's multi-tile path too: the 22-layer simulation wins
   at 0.99998 per tile vs. 0.956 for the 23-layer version, independently, in
   every tile.
4. A fact you can check without running anything at all: the GGUF file's own
   metadata says `clip.vision.block_count = 23` and contains tensors
   `v.blk.0` through `v.blk.22` — 23 layers, physically present in the file.
   The graph code builds a loop that only uses 22 of them.

### Where to look

- Branch: `mtmd-fix-llava-layer-count`, commit `f104a5d38` — `mtmd : fix
  llava vision layer count (#25817)`.
- Full write-up with every model affected: `analysis/bug-b-layer-count.md`.
- Reproduction script: `scripts/phase1/repro_layer_bug.py`.

---

## Diagnosis, fix, and verification are three different things — don't blur them

This is the part worth internalizing more than either bug individually.

- **Diagnosis**: fully done, and backed by numbers, for both bugs — reading
  the code gave a hypothesis, and an independent from-scratch simulation of
  that exact hypothesis reproduces llama.cpp's real (wrong) output to three
  nines or better, while every competing hypothesis scores meaningfully
  worse. That's about as strong as static+empirical evidence gets without
  literally running the patched code.
- **Fix**: written and committed, one branch per bug, both tiny, both
  isolated to `llava.cpp`.
- **Verification of the fix itself**: **not done yet.** Nobody has rebuilt
  llama.cpp from either patched branch and re-run the same comparison
  scripts to confirm the output now matches the *correct* computation instead
  of the *buggy* one. That's the literal next step — it's laid out exactly in
  `analysis/fix-verification-protocol.md`, and the tooling to do it already
  exists in this repo (`dump_llamacpp_embd.py`, `compare_embd.py`). Until
  that's run, "we're confident the diagnosis is right" and "we've confirmed
  the fix works" are two different claims — don't collapse them into one
  sentence.

One more honesty check, in the same spirit: there's an end-task number in
the analysis docs — LLaVA-1.5 scores 56.35 on a 200-sample TextVQA slice with
the bugs present, vs. 58.50 for the correct Hugging Face computation
(−2.15 points). The 95% confidence interval is **[−7.0, +2.75]** — it
straddles zero. That means the direction is consistent with the bugs hurting
accuracy, but at n=200 samples it is *not* statistically significant, and
the comparison is also confounded by llama.cpp using Q4_K_M quantization
while the HF side runs fp16. That's not a weakness to hide — representation-level
evidence (the cosine-similarity numbers above) is the real proof here; the
end-task number is supporting context, reported with its actual uncertainty
instead of rounded up into a confident-sounding claim.

## Why did nobody catch this already?

`tools/mtmd/tests.sh` is the only test that runs LLaVA image inference in
this repo, and it isn't wired into CI (no reference to it anywhere in
`ci/run.sh` or any GitHub workflow). Even when run manually, its pass
condition is a case-insensitive substring match against 128 greedy-decoded
tokens of generated text. It never checks a number. A model can produce a
plausible-sounding caption even when fed vision tokens with a corrupted
position embedding and a smuggled-in CLS row — 575 out of 576 patches are
still roughly where they should be, and the LLM's own language priors paper
over a lot. "Does the output text look reasonable" is a very different bar
from "is the underlying computation correct," and this is a good example of
why: both bugs survived undetected since 2023 and 2025 respectively.

## Why this matters for the pruning project

The whole point of this project is ranking visual tokens by how much
attention the [CLS] token pays to them, then dropping the low-ranked ones.
That ranking is only meaningful if (a) we actually know which row is CLS,
and (b) the features being ranked were computed from the correct number of
transformer layers. Both bugs above sit exactly on that boundary — building
the pruning feature on top of them would mean ranking tokens using
position-shifted patch features and a CLS row that's already leaking into
the output. Fixing these first isn't a detour from the project; it's making
sure the thing we're about to measure is measuring what we think it's
measuring.

## Where to go for more

- `analysis/bug-a-cls-ordering.md`, `analysis/bug-b-layer-count.md` — full
  write-ups: every affected model, every number, every citation.
- `analysis/claims-ledger.md` — which claims are empirically verified vs.
  "affected by code path but not individually run" vs. "unverified lead."
- `analysis/code-drift-check.md` — permalinks pinning every line reference
  above to the exact commit they were read at.
- `analysis/fix-verification-protocol.md` — the exact next steps to confirm
  both fixes actually work, not just that the diagnosis is right.
- `analysis/g1-cls-attention-analysis.md` — the follow-on study of whether
  [CLS] attention can actually be extracted from llama.cpp's graph, which is
  what the pruning feature itself depends on.
