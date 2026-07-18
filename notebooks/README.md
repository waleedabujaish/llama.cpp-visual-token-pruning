# Kaggle GPU sweep — how to run it

This is the GPU counterpart to the CPU (Apple M4 Pro) and x86 (GitHub
Actions) sweeps already in `NOTES.md`/`vtp-cpu-plan-v2.md`. It builds
`waleedabujaish/llama.cpp` with CUDA, re-validates correctness on that
backend, re-runs the same keep-ratio latency matrix, and runs a TextVQA
accuracy-vs-keep-ratio sweep — all on the same quantized model used on
CPU/x86, so the numbers are directly comparable.

**I have no GPU in my own environment and could not run this notebook
end-to-end before handing it to you.** Everything I *could* verify without
a GPU, I did (see "What was actually tested" below). Everything that
depends on real CUDA behavior is guarded by an explicit check that fails
loudly instead of silently producing a wrong number — but you're the
first person to actually run this. Watch the output on the first pass.

## Steps

1. Go to kaggle.com, create a new notebook.
2. Upload `kaggle_gpu_sweep.ipynb` (**File → Upload Notebook**), or copy
   its cells in manually.
3. In the right sidebar, under **Settings**:
   - **Accelerator**: GPU T4 x2, or GPU P100 (either works; T4 x2 is
     usually the faster free option to queue for).
   - **Internet**: **On**. Required — the notebook clones three GitHub
     repos, downloads the model from Hugging Face, and streams TextVQA
     images from Hugging Face datasets.
4. **Run All.**
5. Watch the output of at least these cells on the first run, since
   they're the ones that can silently go wrong if something about Kaggle's
   environment doesn't match what I assumed:
   - **Section 2 (GPU detection)** — should print a real GPU name and an
     `nvcc` version line. If this raises, the Accelerator setting isn't
     on, or Kaggle's image doesn't have the CUDA toolkit where expected —
     stop and check settings.
   - **Section 9 (GPU offload check)** — should print `offloaded N/M
     layers to GPU` with N>0. If this raises, the CUDA build compiled but
     isn't actually running on the GPU. **This is the single most
     important check in the whole notebook** — everything after it is
     meaningless if this didn't pass for real.
   - **Section 10 (Gate 0.5)** — prints whether this specific GPU/build is
     bitwise-deterministic run-to-run. Either answer is fine and expected;
     it just decides which criterion Gate 1 uses next. Read the printed
     line so you know which regime you're in.
   - **Section 11 (Gate 1)** — must print `PASS`. If it raises, stop —
     don't trust any latency/accuracy number produced after it.
   - **Section 15 (accuracy sweep)** — prints which mode it's using
     (`server` or `cli`) and why. `server` mode is ~200x fewer model
     loads; if you see a message about an equivalence mismatch and a
     fallback to `cli` mode, that phase will be much slower — see
     "Session time" below.

## What to download afterward

Kaggle's **Output** panel (or **Save & Run All** to keep a Version) will
have everything under `vision_tokens/results/`. Section 18 (the last code
cell) prints the exact file list. Pull those `.json` files (and the
`raw/` subfolder) into this repo's own `results/` directory — they're
already the same schema the CPU/x86 runs use, tagged
`platform_tag="kaggle-gpu-<gpu-name>"`, so `sweep_analysis.py` and the
rest of the existing tooling pick them up without any changes.

## Session time — genuinely unknown, here's why

Kaggle's free GPU quota is roughly 30 hours/week, and a single interactive
session caps out somewhere in the 9-12 hour range (this has changed over
time on Kaggle's end; check your account's current limit rather than
trusting a specific number here). I don't know how long this notebook
actually takes, because I have no GPU to time it on. Rough shape of where
the time goes:

- Build (x2, CUDA) + model download: probably 10-20 minutes, hard to say
  without a real CUDA build to time.
- Latency sweep: 6 ratios × (1 warmup + 5 runs) × a few seconds cooldown —
  small, should be well under 30 minutes even generously.
- **Accuracy sweep: the genuinely uncertain one.** 200 samples × 6 ratios.
  In **server mode** (the intended fast path — one model load per ratio,
  not per sample) this should be a small number of minutes per ratio. In
  the **CLI fallback** (only triggered if the server/CLI equivalence probe
  in section 15 finds a mismatch) it's ~1,200 individual process launches,
  each reloading the model — this could plausibly take over an hour, maybe
  well over, and might not finish in one session.

If the accuracy sweep looks like it won't finish in your session's time
budget: it's fine to let the session end mid-sweep. Everything is
resumable — re-open the notebook and **Run All** again; already-completed
build/download/gate/latency-sweep steps are skipped (detected from files
already on disk), and the accuracy sweep resumes per-sample (not
per-ratio) from an incrementally-written `.preds.jsonl`, so at most the
one in-flight sample is repeated. Note that a *new* Kaggle session starts
with an empty `/kaggle/working`, though — if the whole session (not just
the notebook process) restarts, you'll re-clone/re-download/re-build, and
only the accuracy sweep's *already-downloaded-and-committed-to-results/*
progress survives that (nothing here currently persists results across a
full session loss on its own — see "possible follow-up" below).

To deliberately shrink scope for a single pass: edit the config cell
(section 1) before running — `ACCURACY_RATIOS` and `ACCURACY_LIMIT` are
there specifically so you can do a partial run first, confirm gates pass
and numbers look sane, then a full run.

## Scope decisions I made without asking first

A few places where I had to make a call rather than there being one
obvious right answer — flagging these so you can override if you'd have
chosen differently:

- **Cooldown between latency runs: 5 seconds, not CPU's 30.** CPU's 30s
  cooldown was a fix for sustained thermal throttling on a
  fanless/limited-cooling laptop chip — a specific, measured problem
  (`NOTES.md` "Thermal drift"). Datacenter T4/P100 have real cooling and
  are built for sustained load, so I don't expect that to reproduce here,
  but I have no data confirming it — it's a judgment call. The notebook
  says so inline (section 1) and tells you what to watch for (upward
  drift in per-run timings) if the guess was wrong.
- **MME/POPE: stubbed, not implemented.** Each has its own metric
  definition (MME's accuracy+accuracy+, POPE's imbalance-robust F1) —
  building correct harnesses for both is real, separate work, not
  something to fold into an already-large notebook without the same care
  the TextVQA harness got. Section 17 is an explicit disabled stub
  explaining what's needed, not a silent omission.
- **Accuracy sweep tries server mode first, with an automatic
  equivalence-verified fallback to the slower, already-validated CLI
  mode.** This one isn't really optional — 1,200 individual CLI
  invocations was very unlikely to finish in a single Kaggle session, so
  I built a server-based path to cut that to ~6 model loads. But the
  server's prompt-templating code is different code than the CLI's
  (confirmed by reading both, in `waleedabujaish/llama.cpp`'s
  `tools/server/` vs `tools/mtmd/mtmd-cli.cpp`), so I did not assume they
  produce identical text — the notebook empirically checks this on one
  sample before trusting server mode for the bulk run, and falls back
  loudly if they disagree. I could not verify which branch this takes
  without a GPU; read what section 15 prints.
- **Did not expand TextVQA past the pinned 200 samples.** The manifest
  (`assets/phase1/textvqa200_manifest.jsonl`) is a specific, pinned,
  already-reproducible slice. Drawing more samples from the dataset would
  create a new, unpinned eval slice with no established provenance —
  contrary to how every other eval set in this project has been handled.
  If more samples are wanted, that's a new pin to establish first, not
  something to improvise inside this notebook.

## What was actually tested (and what wasn't)

Tested for real, without a GPU:
- `nbformat.validate()` on the generated notebook, and `ast.parse()` on
  every code cell's source — no syntax errors, valid notebook structure.
- The exact clone + pinned-commit-verification sequence in section 4 —
  ran it live against the real repos (`git clone` + `git rev-parse HEAD`
  cross-checked against all three pinned SHAs). Passed.
- `materialize_textvqa_images.py` (new script, section 14) — ran it live:
  streamed `lmms-lab/textvqa`, matched all 200 pinned samples by question
  text, saved all 200 images + `meta.jsonl`.
- `textvqa_keep_sweep.py`'s CLI mode (new script, section 15's fallback
  path) — ran a real 1-sample, 2-ratio sweep against the local CPU build;
  produced a correct answer and a real scored summary JSON.
- `dump_llamacpp_embd.py`'s new `--n-gpu-layers` flag — ran against the
  local CPU build; confirmed the default (`-1`) is unchanged from prior
  behavior (verified empirically against the compiled struct default,
  not assumed) and is correctly threaded through to the model params.
- `bench_baseline.py`'s new `gpu_env_info()` — confirmed it returns `{}`
  cleanly on a machine with no `nvidia-smi` (this Mac), and that the
  KV-buffer-size regex change still matches the old CPU-only log line
  format (no regression) in addition to a CUDA-backend one.
- The `"offloaded N/M layers to GPU"` log line section 9 greps for —
  confirmed present in `src/llama-model.cpp` at the exact pinned base
  commit (`e8f19cc0`), not just current upstream.

**Not tested, because it needs a real GPU:**
- The actual CUDA build (`-DGGML_CUDA=ON`) compiling cleanly on Kaggle's
  toolchain/image.
- Whether GPU layer offload actually happens at runtime (section 9's
  check is real and will catch it if not, but I haven't seen it pass).
- GPU run-to-run determinism (section 10) — genuinely don't know which
  way this goes; the notebook handles either outcome.
- `textvqa_keep_sweep.py`'s server mode and its CLI-equivalence probe —
  the HTTP request format (OpenAI-style `image_url` content blocks) is
  based on reading `tools/server/server-common.cpp`, not on having sent a
  real request to a running `llama-server` with this fork's
  `--visual-keep` flag.
- Real runtime for any phase — see "Session time" above.

## Possible follow-ups (not done here, flagging rather than scope-creeping)

- Persisting intermediate state to a Kaggle Dataset so a full session
  restart (not just a notebook re-run) doesn't lose build/download
  progress — skipped because it needs a one-time manual Kaggle Dataset
  setup step, which would've broken the "single-session runnable without
  manual intervention beyond enabling the GPU accelerator" requirement.
- MME/POPE harnesses (see above).
- A controlled (dedicated, non-shared) GPU comparison, if the free-tier
  Kaggle runner turns out to be noisy the way the GitHub Actions x86
  runner was.
