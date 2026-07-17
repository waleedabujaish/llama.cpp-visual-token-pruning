#!/usr/bin/env python3
"""End-task impact: TextVQA through llama.cpp itself (llama-mtmd-cli).

Runs the SAME 200-sample TextVQA slice used by textvqa_sim.py through the
pinned llama.cpp build (frozen inference config), scores it with the
identical soft-VQA metric, and reports a PAIRED comparison against the HF
keep=1.0 baseline records.

Prompt parity: a custom jinja chat template reproduces the exact vicuna_v1
string used in the HF run ("<system> USER: <image>\n<q>[\nReference OCR
token: ...]\nAnswer the question using a single word or phrase. ASSISTANT:").
Deviation from the frozen timing config: --jinja --chat-template-file replaces
--chat-template vicuna (whitespace parity with the HF baseline matters here;
inference flags are unchanged). Images are the original-resolution PNGs —
llama.cpp does its own preprocessing, which is part of the end-to-end path
being measured.

Incremental predictions are appended to a .preds.jsonl next to the results
JSON so a crash cannot lose completed samples.
"""

import argparse
import datetime
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from textvqa_sim import vqa_accuracy

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SYSTEM_PROMPT = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's questions.")


def run_one(args, image_path: Path, question: str, ocr: list) -> tuple:
    ocr_line = f"\nReference OCR token: {', '.join(ocr)}" if ocr else ""
    prompt = f"\n{question}{ocr_line}\nAnswer the question using a single word or phrase."
    cmd = [
        args.bin,
        "-m", args.model, "--mmproj", args.mmproj,
        "--image", str(image_path),
        "-sys", SYSTEM_PROMPT,
        "-p", prompt,
        "-n", "16", "--temp", "0", "--seed", "42",
        "-t", "8", "-tb", "8", "-b", "2048", "-ub", "1024",
        # --jinja must precede --chat-template for a custom template to be accepted
        "--jinja", "--chat-template",
        (Path(__file__).parent / "vicuna_v1_llava.jinja").read_text(),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        return None, proc.stderr[-2000:]
    return proc.stdout.strip(), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with NNN.png + meta.jsonl")
    ap.add_argument("--hf-results", required=True, help="textvqa_sim results JSON (paired baseline)")
    ap.add_argument("--bin", default=str(Path.home() / "Desktop/vtp/llama.cpp/build/bin/llama-mtmd-cli"))
    ap.add_argument("--model", default=str(REPO_ROOT / "models/llava-v1.5-7b-Q4_K_M.gguf"))
    ap.add_argument("--mmproj", default=str(REPO_ROOT / "models/llava-v1.5-7b-mmproj-model-f16.gguf"))
    ap.add_argument("--tag", default="p1_textvqa_llamacpp")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    metas = [json.loads(l) for l in (data / "meta.jsonl").read_text().splitlines()]
    if args.limit:
        metas = metas[: args.limit]

    hf = json.loads(Path(args.hf_results).read_text())
    hf_base = {r["i"]: r["pred"]["1"] for r in hf["records"]}

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_json = REPO_ROOT / "results" / f"{ts}_{args.tag}.json"
    preds_path = out_json.with_suffix(".preds.jsonl")

    records, accs, fails = [], [], 0
    t0 = time.time()
    for m in metas:
        pred, err = run_one(args, data / f"{m['i']:03d}.png", m["question"], m["ocr_tokens"])
        if pred is None:
            fails += 1
            rec = {"i": m["i"], "error": err}
        else:
            acc = vqa_accuracy(pred, m["answers"])
            accs.append(acc)
            rec = {"i": m["i"], "question": m["question"], "pred": pred, "acc": acc,
                   "hf_base_acc": hf_base.get(m["i"], {}).get("acc")}
        records.append(rec)
        with open(preds_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        n = len(records)
        if n % 10 == 0:
            el = time.time() - t0
            print(f"[cpp] {n}/{len(metas)} ({el/60:.1f} min) "
                  f"acc so far: {np.mean(accs)*100:.1f} (fails={fails})", flush=True)

    ok = [r for r in records if "acc" in r and r.get("hf_base_acc") is not None]
    cpp = np.array([r["acc"] for r in ok])
    hfb = np.array([r["hf_base_acc"] for r in ok])
    diff = cpp - hfb

    rng = np.random.default_rng(42)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(10000)])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    out = {
        "tag": args.tag, "timestamp": ts,
        "llama_cpp": {"commit": "e8f19cc0ad70a243c8012bf17b4be601abfc8ea2",
                      "build": "Release, GGML_METAL=OFF, CPU+Accelerate",
                      "model": args.model, "mmproj": args.mmproj,
                      "config": "-n 16 --temp 0 --seed 42 -t 8 -tb 8 -b 2048 -ub 1024, "
                                "custom vicuna_v1 jinja (prompt parity with HF baseline)"},
        "dataset": "same 200-sample lmms-lab/textvqa slice as the HF run (paired)",
        "metric": "soft VQA accuracy, identical scoring code (textvqa_sim.vqa_accuracy)",
        "known_confounds": "Q4_K_M quantization of the LM (HF baseline is fp16) and "
                           "llama.cpp image preprocessing; prompt format and scoring are matched",
        "n_scored": len(ok), "n_failed": fails,
        "llamacpp_acc_mean": float(cpp.mean()),
        "hf_fp16_correct_baseline_acc_mean": float(hfb.mean()),
        "paired": {
            "mean_diff_cpp_minus_hf": float(diff.mean()),
            "bootstrap_95ci": ci,
            "wins_cpp": int((diff > 0).sum()), "losses_cpp": int((diff < 0).sum()),
            "ties": int((diff == 0).sum()),
        },
        "records": records,
    }
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[cpp] wrote {out_json}")
    print(json.dumps({k: out[k] for k in
                      ("llamacpp_acc_mean", "hf_fp16_correct_baseline_acc_mean", "paired")}, indent=2))


if __name__ == "__main__":
    main()
