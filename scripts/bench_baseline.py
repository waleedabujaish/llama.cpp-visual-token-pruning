#!/usr/bin/env python3
"""G0/G2 baseline benchmark for llama-mtmd-cli (LLaVA-1.5-7B, CPU-only build).

Runs llama-mtmd-cli N+warmup times on a fixed image + prompt at temp 0,
parses llama.cpp's own timing output, and writes one timestamped JSON to
results/ plus raw per-run logs to results/raw/<timestamp>/.

Timing sources (exact log lines in llama.cpp @ the pinned commit; require -v):
  - "mtmd batch encoding done in <N> ms"       (mtmd-cli.cpp:338)     -> vision encoder + projector
  - "image decoded (batch i/n) in <N> ms"      (mtmd-helper.cpp:324)  -> LLM prefill over image embeddings
  - "prompt eval time = ... ms / ... tokens"   (llama-context.cpp:4104) -> full LLM prefill
      (n_p_eval counts every llama_decode with n_tokens > 1, i.e. text
       batches AND the image-embedding batch  — llama-context.cpp:714-719)
  - "eval time / load time / total time"       (llama_perf_context_print)

Derived:
  TTFT_llm = prompt_eval_ms                      (prefill only, where H1 lives)
  TTFT_vlm = encode_ms + prompt_eval_ms          (excludes model load, tokenize ~0)
  encoder_fraction = encode_ms / TTFT_vlm
  amdahl_ceiling_simple  = 1 / encoder_fraction  (plan §1 definition)
  amdahl_ceiling_refined = TTFT_vlm / (encode_ms + (prompt_eval_ms - image_decode_ms))
      (keep->0 limit: only the image-token part of prefill is prunable)
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama_provenance import resolve_build_provenance  # noqa: E402

# Model provenance, recorded into every results JSON (see NOTES.md).
PROVENANCE = {
    "model_repo": "second-state/Llava-v1.5-7B-GGUF",
    "model_files": ["llava-v1.5-7b-Q4_K_M.gguf", "llava-v1.5-7b-mmproj-model-f16.gguf"],
    "substitution_reason": (
        "Originally planned mys/ggml_llava-v1.5-7b fails to load at the pinned "
        "llama.cpp commit: its 2023-era mmproj lacks the clip.projector_type GGUF "
        "key required by the current loader (clip.cpp:1153). second-state repack "
        "is the same LLaVA-1.5-7B weights and is used by llama.cpp's own vision "
        "test suite (tools/mtmd/tests.sh:95)."
    ),
    "config_frozen": (
        "Run config frozen 2026-07-17 for all cells incl. pruned runs: "
        "-n 32 --temp 0 --seed 42 -t 8 -tb 8 -b 2048 -ub 1024 --perf "
        "--chat-template vicuna -v. Rationale in NOTES.md. -t 8 = P-cores only "
        "(deliberate; E-core/all-core ablation deferred). -ub 1024 keeps the "
        "576-token non-causal image batch in one ubatch."
    ),
}

RE_ENCODE = re.compile(r"mtmd batch encoding done in (\d+) ms")
RE_IMG_DECODE = re.compile(r"image decoded \(batch \d+/\d+\) in (\d+) ms")
RE_LOAD = re.compile(r"load time\s*=\s*([\d.]+) ms")
RE_PROMPT_EVAL = re.compile(r"prompt eval time\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens")
RE_EVAL = re.compile(r"[^t] eval time\s*=\s*([\d.]+) ms\s*/\s*(\d+) runs")
RE_TOTAL = re.compile(r"total time\s*=\s*([\d.]+) ms\s*/\s*(\d+) tokens")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def sysctl(key: str) -> str:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def parse_run(log: str) -> dict:
    encode = [int(m) for m in RE_ENCODE.findall(log)]
    img_dec = [int(m) for m in RE_IMG_DECODE.findall(log)]
    m_load = RE_LOAD.search(log)
    m_pe = RE_PROMPT_EVAL.search(log)
    m_ev = RE_EVAL.search(log)
    m_tot = RE_TOTAL.search(log)
    if not (encode and img_dec and m_pe and m_ev):
        raise ValueError("run log missing expected timing lines")
    encode_ms = float(sum(encode))
    image_decode_ms = float(sum(img_dec))
    prompt_eval_ms = float(m_pe.group(1))
    n_prompt_tokens = int(m_pe.group(2))
    eval_ms = float(m_ev.group(1))
    n_eval_runs = int(m_ev.group(2))
    ttft_llm = prompt_eval_ms
    ttft_vlm = encode_ms + prompt_eval_ms
    return {
        "encode_ms": encode_ms,
        "image_decode_ms": image_decode_ms,
        "prompt_eval_ms": prompt_eval_ms,
        "n_prompt_tokens": n_prompt_tokens,
        "eval_ms": eval_ms,
        "n_eval_runs": n_eval_runs,
        "decode_tok_per_s": n_eval_runs / (eval_ms / 1000.0) if eval_ms > 0 else None,
        "load_ms": float(m_load.group(1)) if m_load else None,
        "total_ms": float(m_tot.group(1)) if m_tot else None,
        "ttft_llm_ms": ttft_llm,
        "ttft_vlm_ms": ttft_vlm,
        "encoder_fraction": encode_ms / ttft_vlm,
    }


def mean_std(xs):
    return {
        "mean": statistics.mean(xs),
        "std": statistics.stdev(xs) if len(xs) > 1 else 0.0,
        "n": len(xs),
        "values": xs,
    }


def main():
    ap = argparse.ArgumentParser()
    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", str(REPO_ROOT.parent / "llama.cpp"))
    ap.add_argument("--bin", default=llama_cpp_dir + "/build/bin/llama-mtmd-cli")
    ap.add_argument("--llama-repo", default=llama_cpp_dir)
    ap.add_argument("--model", default=str(REPO_ROOT / "models/llava-v1.5-7b-Q4_K_M.gguf"))
    ap.add_argument("--mmproj", default=str(REPO_ROOT / "models/llava-v1.5-7b-mmproj-model-f16.gguf"))
    ap.add_argument("--image", default=str(REPO_ROOT / "assets/coco_val2017_000000039769.jpg"))
    ap.add_argument("--prompt", default="Describe the image in detail.")
    ap.add_argument("--n-predict", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ubatch", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--tag", default="g0_baseline")
    ap.add_argument("--cooldown-s", type=float, default=0.0,
                    help="sleep this many seconds between runs (incl. after warmup); "
                         "mitigates the sustained -t N thermal drift observed within "
                         "back-to-back 6-run blocks (see NOTES.md)")
    ap.add_argument("--extra-arg", action="append", default=[],
                    help="additional llama-mtmd-cli flag, repeatable, e.g. "
                         "--extra-arg=--visual-keep --extra-arg=0.5")
    args = ap.parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_dir = REPO_ROOT / "results" / "raw" / f"{ts}_{args.tag}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    load_avg_start = os.getloadavg()

    cmd = [
        args.bin,
        "-m", args.model,
        "--mmproj", args.mmproj,
        "--image", args.image,
        "-p", args.prompt,
        "-n", str(args.n_predict),
        "--temp", "0",
        "--seed", str(args.seed),
        "-t", str(args.threads),
        "-tb", str(args.threads),
        "-b", str(args.batch),
        "-ub", str(args.ubatch),
        "--perf",
        "--chat-template", "vicuna",
        "-v",  # timing lines (image decode, llama_perf) only print at full verbosity
    ] + args.extra_arg

    print(f"[bench] command: {' '.join(cmd)}", flush=True)
    print(f"[bench] hashing model files ...", flush=True)
    model_sha = sha256(Path(args.model))
    mmproj_sha = sha256(Path(args.mmproj))
    image_sha = sha256(Path(args.image))

    runs = []
    texts = []
    total = args.warmup + args.runs
    for i in range(total):
        label = "warmup" if i < args.warmup else f"run{i - args.warmup + 1}"
        t0 = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        wall_s = time.monotonic() - t0
        log = proc.stderr + "\n" + proc.stdout
        (raw_dir / f"{label}.log").write_text(
            f"# argv: {' '.join(cmd)}\n# exit: {proc.returncode}\n# wall_s: {wall_s:.3f}\n"
            f"# --- stdout ---\n{proc.stdout}\n# --- stderr ---\n{proc.stderr}\n")
        if proc.returncode != 0:
            print(f"[bench] {label}: FAILED exit={proc.returncode}, see {raw_dir}/{label}.log", flush=True)
            sys.exit(1)
        parsed = parse_run(log)
        parsed["wall_s_process"] = wall_s
        parsed["label"] = label
        print(f"[bench] {label}: encode={parsed['encode_ms']:.0f}ms "
              f"prompt_eval={parsed['prompt_eval_ms']:.0f}ms ({parsed['n_prompt_tokens']} tok) "
              f"img_decode={parsed['image_decode_ms']:.0f}ms "
              f"decode={parsed['decode_tok_per_s']:.2f} tok/s wall={wall_s:.1f}s", flush=True)
        if i >= args.warmup:
            runs.append(parsed)
            texts.append(proc.stdout.strip())
        if args.cooldown_s > 0 and i < total - 1:
            time.sleep(args.cooldown_s)

    identical_output = len(set(texts)) == 1

    agg = {k: mean_std([r[k] for r in runs]) for k in
           ["encode_ms", "image_decode_ms", "prompt_eval_ms", "eval_ms",
            "decode_tok_per_s", "load_ms", "ttft_llm_ms", "ttft_vlm_ms",
            "encoder_fraction", "total_ms", "wall_s_process"]}

    enc = agg["encode_ms"]["mean"]
    pe = agg["prompt_eval_ms"]["mean"]
    imgdec = agg["image_decode_ms"]["mean"]
    result = {
        "tag": args.tag,
        "timestamp": ts,
        "provenance": PROVENANCE,
        "command": cmd,
        "prompt": args.prompt,
        "config": {
            "n_predict": args.n_predict, "threads": args.threads,
            "batch": args.batch, "ubatch": args.ubatch,
            "temp": 0.0, "seed": args.seed,
            "warmup_discarded": args.warmup, "timed_runs": args.runs,
            "cooldown_s_between_runs": args.cooldown_s,
            "extra_args": args.extra_arg,
        },
        "environment": {
            "llama_cpp_build": resolve_build_provenance(args.bin, args.llama_repo),
            "build": "cmake -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=OFF (CPU + Accelerate/BLAS)",
            "cpu": sysctl("machdep.cpu.brand_string"),
            "n_cores": sysctl("hw.ncpu"),
            "p_cores": sysctl("hw.perflevel0.physicalcpu"),
            "e_cores": sysctl("hw.perflevel1.physicalcpu"),
            "mem_bytes": sysctl("hw.memsize"),
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "load_avg_1_5_15_at_start": list(load_avg_start),
            "load_avg_1_5_15_at_end": list(os.getloadavg()),
        },
        "files": {
            "model": {"path": args.model, "sha256": model_sha},
            "mmproj": {"path": args.mmproj, "sha256": mmproj_sha},
            "image": {"path": args.image, "sha256": image_sha},
        },
        "runs": runs,
        "aggregate": agg,
        "derived": {
            "encoder_fraction_of_ttft_vlm": enc / (enc + pe),
            "amdahl_ceiling_simple_1_over_frac": (enc + pe) / enc,
            "amdahl_ceiling_refined_keep0": (enc + pe) / (enc + (pe - imgdec)),
            "note": "simple = 1/encoder_fraction (plan §1); refined = keep->0 limit where only the image-embedding part of prefill (image_decode_ms) is prunable",
        },
        "determinism": {
            "identical_output_across_timed_runs": identical_output,
            "generated_text_first_run": texts[0] if texts else None,
        },
        "raw_logs_dir": str(raw_dir.relative_to(REPO_ROOT)),
    }

    out = REPO_ROOT / "results" / f"{ts}_{args.tag}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[bench] wrote {out}", flush=True)
    print(json.dumps(result["derived"], indent=2), flush=True)


if __name__ == "__main__":
    main()
