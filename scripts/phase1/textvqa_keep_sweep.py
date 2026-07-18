#!/usr/bin/env python3
"""TextVQA accuracy vs keep-ratio, on the SAME llama.cpp build under test.

Extends textvqa_llamacpp.py's single-build TextVQA harness (same 200-sample
pinned manifest, same vicuna_v1 jinja prompt parity, same scoring code)
across a keep-ratio sweep, to characterize the pruned C++ path's OWN
accuracy/latency tradeoff -- not a paired comparison against the HF
reference (that's what textvqa_llamacpp.py + textvqa_sim.py already do for
the single unpruned case); this script's job is the retention curve on the
C++ build itself.

Two execution modes:
  - cli: one llama-mtmd-cli process per sample (what textvqa_llamacpp.py
    already does -- proven correct, but a full 200-sample x 6-ratio sweep
    is ~1200 fresh process launches + model loads, each reloading a ~4-5GB
    GGUF, so this is slow even on GPU).
  - server: one llama-server process per RATIO (started once with
    --visual-keep baked in at startup, since it's a startup param, not a
    per-request one -- confirmed from tools/server/server-context.cpp),
    queried 200 times over HTTP. ~200x fewer model loads.
Default (--mode auto): before trusting server mode for the bulk run, probe
ONE sample (ratio=1.0, sample 0) through BOTH paths and require identical
generated text. The server's OpenAI-style multimodal chat endpoint may
render the chat template differently than the CLI's own prompt-construction
path (different code, same jinja file) -- this has NOT been verified
correct independently, so equivalence is checked empirically every run
rather than assumed. On any mismatch, or if the server fails to come up,
falls back to (slower, already-validated) CLI mode for the entire sweep
and says so loudly -- never silently substitutes an unverified prompt path.

Resumable at two granularities, since this is by far the longest-running
phase and the intended runner (Kaggle) can disconnect mid-session:
  - per-ratio: a completed ratio's final summary JSON existing skips it
    entirely (same pattern as sweep_prune.py).
  - per-sample within a ratio: predictions are appended immediately to a
    stable (non-timestamped) .preds.jsonl per ratio as they complete; on
    restart, already-scored sample indices are read back and skipped.
"""

import argparse
import base64
import datetime
import glob
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from textvqa_sim import vqa_accuracy

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from llama_provenance import resolve_build_provenance  # noqa: E402

SYSTEM_PROMPT = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's questions.")
JINJA_TEMPLATE = (Path(__file__).parent / "vicuna_v1_llava.jinja").read_text()


def build_prompt(question: str, ocr: list) -> str:
    ocr_line = f"\nReference OCR token: {', '.join(ocr)}" if ocr else ""
    return f"\n{question}{ocr_line}\nAnswer the question using a single word or phrase."


# ---------------- CLI mode (one process per sample) --------------------------
def run_one_cli(args, ratio: float, image_path: Path, question: str, ocr: list):
    cmd = [
        args.bin,
        "-m", args.model, "--mmproj", args.mmproj,
        "--image", str(image_path),
        "-sys", SYSTEM_PROMPT,
        "-p", build_prompt(question, ocr),
        "-n", "16", "--temp", "0", "--seed", "42",
        "-t", "8", "-tb", "8", "-b", "2048", "-ub", "1024",
        "--jinja", "--chat-template", JINJA_TEMPLATE,
        "--visual-keep", f"{ratio}",
        "--visual-prune-method", args.visual_prune_method,
    ] + args.extra_arg
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        return None, proc.stderr[-2000:]
    return proc.stdout.strip(), None


# ---------------- server mode (one process per ratio) -------------------------
def http_post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def http_get_ok(url: str, timeout: float) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def start_server(args, ratio: float, log_path: Path):
    cmd = [
        args.server_bin,
        "-m", args.model, "--mmproj", args.mmproj,
        "-t", "8", "-tb", "8", "-b", "2048", "-ub", "1024",
        "--jinja", "--chat-template", JINJA_TEMPLATE,
        "--visual-keep", f"{ratio}",
        "--visual-prune-method", args.visual_prune_method,
        "--host", args.host, "--port", str(args.port),
    ] + args.extra_arg
    log_f = open(log_path, "w")
    log_f.write(f"# argv: {' '.join(cmd)}\n")
    log_f.flush()
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
    return proc, log_f


def stop_server(proc):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)


def wait_for_server(host: str, port: int, timeout_s: float) -> bool:
    url = f"http://{host}:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if http_get_ok(url, timeout=3):
            return True
        time.sleep(2)
    return False


def query_server(host: str, port: int, image_path: Path, prompt: str) -> str:
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": prompt},
            ]},
        ],
        "temperature": 0, "seed": 42, "max_tokens": 16, "stream": False,
    }
    resp = http_post_json(f"http://{host}:{port}/v1/chat/completions", payload, timeout=120)
    return resp["choices"][0]["message"]["content"].strip()


# ---------------- resumability helpers ----------------------------------------
def done_sample_indices(preds_path: Path) -> dict:
    done = {}
    if preds_path.exists():
        for line in preds_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done[rec["i"]] = rec
            except Exception:
                continue
    return done


def ratio_complete(results_dir: Path, tag: str) -> bool:
    return bool(glob.glob(str(results_dir / f"*_{tag}_summary.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with NNN.png + meta.jsonl "
                    "(see materialize_textvqa_images.py)")
    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", str(REPO_ROOT.parent / "llama.cpp"))
    ap.add_argument("--bin", default=llama_cpp_dir + "/build/bin/llama-mtmd-cli")
    ap.add_argument("--server-bin", default=llama_cpp_dir + "/build/bin/llama-server")
    ap.add_argument("--llama-repo", default=llama_cpp_dir)
    ap.add_argument("--model", default=str(REPO_ROOT / "models/llava-v1.5-7b-Q4_K_M.gguf"))
    ap.add_argument("--mmproj", default=str(REPO_ROOT / "models/llava-v1.5-7b-mmproj-model-f16.gguf"))
    ap.add_argument("--ratios", default="1.0,0.75,0.5,0.25,0.1,0.05")
    ap.add_argument("--visual-prune-method", default="cls")
    ap.add_argument("--limit", type=int, default=0, help="0 = full manifest")
    ap.add_argument("--tag-prefix", default="p3_textvqa_keep_sweep")
    ap.add_argument("--platform-tag", required=True)
    ap.add_argument("--build-desc", required=True)
    ap.add_argument("--extra-arg", action="append", default=[],
                    help="passthrough flag, repeatable, e.g. --extra-arg=-ngl --extra-arg=999")
    ap.add_argument("--mode", choices=["auto", "server", "cli"], default="auto")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8811)
    ap.add_argument("--server-startup-timeout-s", type=float, default=180.0)
    args = ap.parse_args()

    ratios = [float(r) for r in args.ratios.split(",")]
    data = Path(args.data)
    metas = [json.loads(l) for l in (data / "meta.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        metas = metas[: args.limit]
    if not metas:
        raise SystemExit(f"no samples found under {data} -- run materialize_textvqa_images.py first")

    results_dir = REPO_ROOT / "results"
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    build_prov = resolve_build_provenance(args.bin, args.llama_repo)

    # ---- resolve mode: probe equivalence once (keep=1.0, sample 0) ----
    use_server = args.mode in ("auto", "server")
    if use_server:
        print("[sweep] verifying CLI/server prompt-format equivalence before bulk run "
              "(keep=1.0, sample 0) ...", flush=True)
        m0 = metas[0]
        image0 = data / f"{m0['i']:03d}.png"
        cli_pred, cli_err = run_one_cli(args, 1.0, image0, m0["question"], m0["ocr_tokens"])
        if cli_pred is None:
            print(f"[sweep] WARNING: CLI equivalence probe itself failed ({cli_err}); "
                  f"cannot verify equivalence, forcing CLI mode", flush=True)
            use_server = False
        else:
            log_path = raw_dir / "server_equivalence_probe.log"
            proc, log_f = start_server(args, 1.0, log_path)
            try:
                if not wait_for_server(args.host, args.port, args.server_startup_timeout_s):
                    print(f"[sweep] WARNING: server did not become healthy within "
                          f"{args.server_startup_timeout_s}s (see {log_path}); "
                          f"falling back to CLI mode", flush=True)
                    use_server = False
                else:
                    server_pred = query_server(args.host, args.port, image0, build_prompt(
                        m0["question"], m0["ocr_tokens"]))
                    if server_pred.strip() == cli_pred.strip():
                        print(f"[sweep] equivalence OK -- CLI and server produced identical text "
                              f"for sample 0 at keep=1.0: {server_pred!r}. Using server mode for "
                              f"the bulk sweep.", flush=True)
                    else:
                        print(f"[sweep] EQUIVALENCE MISMATCH -- CLI and server produced DIFFERENT "
                              f"text for the same sample/ratio. This means the server's chat-"
                              f"template rendering diverges from the already-validated CLI path "
                              f"(textvqa_llamacpp.py); using it uninspected would silently change "
                              f"the accuracy methodology.\n  CLI:    {cli_pred!r}\n"
                              f"  server: {server_pred!r}\nFalling back to CLI mode (slower, but "
                              f"matches the validated methodology) for the entire sweep.", flush=True)
                        use_server = False
            finally:
                stop_server(proc)
                log_f.close()
    if args.mode == "cli":
        use_server = False
    elif args.mode == "server" and not use_server:
        raise SystemExit("[sweep] --mode server was forced but the equivalence check failed or "
                          "the server did not come up -- refusing to silently run an unverified "
                          "prompt path. Use --mode auto to allow a CLI fallback, or fix the issue "
                          "logged above first.")
    print(f"[sweep] mode resolved to: {'server' if use_server else 'cli'}", flush=True)

    # ---- per-ratio sweep ----
    for r in ratios:
        tag = f"{args.tag_prefix}_keep{r:g}"
        if ratio_complete(results_dir, tag):
            print(f"[sweep] {tag}: already complete, skipping", flush=True)
            continue

        preds_path = raw_dir / f"{tag}.preds.jsonl"
        done = done_sample_indices(preds_path)
        mode_str = "server" if use_server else "cli"
        print(f"[sweep] {tag}: {len(done)}/{len(metas)} samples already done, resuming "
              f"(mode={mode_str})", flush=True)

        server_proc, log_f = None, None
        if use_server:
            log_path = raw_dir / f"{tag}_server.log"
            server_proc, log_f = start_server(args, r, log_path)
            if not wait_for_server(args.host, args.port, args.server_startup_timeout_s):
                print(f"[sweep] {tag}: server failed to come up (see {log_path}); "
                      f"falling back to CLI for THIS ratio only", flush=True)
                stop_server(server_proc)
                server_proc, log_f = None, None

        t0 = time.time()
        try:
            for m in metas:
                if m["i"] in done:
                    continue
                image_path = data / f"{m['i']:03d}.png"
                prompt = build_prompt(m["question"], m["ocr_tokens"])
                if server_proc is not None:
                    try:
                        pred, err = query_server(args.host, args.port, image_path, prompt), None
                    except Exception as e:
                        pred, err = None, str(e)
                else:
                    pred, err = run_one_cli(args, r, image_path, m["question"], m["ocr_tokens"])
                if pred is None:
                    rec = {"i": m["i"], "error": err}
                else:
                    acc = vqa_accuracy(pred, m["answers"])
                    rec = {"i": m["i"], "question": m["question"], "pred": pred, "acc": acc}
                with open(preds_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                done[m["i"]] = rec
                n = len(done)
                if n % 10 == 0 or n == len(metas):
                    el = time.time() - t0
                    accs = [d["acc"] for d in done.values() if "acc" in d]
                    mean_acc = (sum(accs) / len(accs) * 100) if accs else float("nan")
                    print(f"[sweep] {tag}: {n}/{len(metas)} ({el/60:.1f} min) "
                          f"acc so far: {mean_acc:.1f}", flush=True)
        finally:
            if server_proc is not None:
                stop_server(server_proc)
                log_f.close()

        ok = [done[m["i"]] for m in metas if "acc" in done.get(m["i"], {})]
        fails = [done[m["i"]] for m in metas if "error" in done.get(m["i"], {})]
        accs = np.array([d["acc"] for d in ok])
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        summary = {
            "tag": tag, "tag_prefix": args.tag_prefix, "timestamp": ts,
            "visual_keep": r, "visual_prune_method": args.visual_prune_method,
            "platform_tag": args.platform_tag,
            "mode": mode_str,
            "llama_cpp": {
                "build_provenance": build_prov, "build": args.build_desc,
                "model": args.model, "mmproj": args.mmproj,
                "config": "-n 16 --temp 0 --seed 42 -t 8 -tb 8 -b 2048 -ub 1024, "
                          "custom vicuna_v1 jinja (prompt parity with the Phase 1 HF baseline), "
                          f"extra_args={args.extra_arg}",
            },
            "dataset": "same pinned 200-sample lmms-lab/textvqa slice as textvqa_llamacpp.py "
                       "(assets/phase1/textvqa200_manifest.jsonl)",
            "metric": "soft VQA accuracy, identical scoring code (textvqa_sim.vqa_accuracy)",
            "n_scored": len(ok), "n_failed": len(fails),
            "acc_mean": float(accs.mean()) if len(accs) else None,
            "acc_std": float(accs.std()) if len(accs) else None,
            "preds_file": str(preds_path.relative_to(REPO_ROOT)),
        }
        summary_path = results_dir / f"{ts}_{tag}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[sweep] wrote {summary_path}: acc_mean={summary['acc_mean']}", flush=True)

    print("[sweep] all ratios complete", flush=True)


if __name__ == "__main__":
    main()
