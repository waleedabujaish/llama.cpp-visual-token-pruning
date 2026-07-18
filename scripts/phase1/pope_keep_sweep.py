#!/usr/bin/env python3
"""POPE (object hallucination probing) accuracy vs keep-ratio, on the SAME
llama.cpp build under test. Architecturally mirrors textvqa_keep_sweep.py
(same CLI/server dual-mode, same empirical equivalence-probe-with-
reproducibility-recheck before trusting server mode, same per-sample
resumability) -- deliberately NOT refactored to share code with that
script: textvqa_keep_sweep.py already produced real, trusted GPU results,
and extracting shared internals now would mean modifying validated,
already-relied-upon code for a second caller's convenience. Kept as an
independent sibling instead; a shared-harness refactor is a reasonable
future cleanup once this script is validated too, not a prerequisite.

POPE-specific differences from the TextVQA version:
  - Prompt: the pinned manifest's `question` field is already a yes/no
    probe (e.g. "Is there a snowboard in the image?"); appends the
    standard "Please answer yes or no." suffix used across LLaVA-style
    hallucination evals to standardize free-text output into a
    classifiable answer -- same system prompt and vicuna_v1 jinja
    template as TextVQA (a property of the model's expected chat format,
    not of the eval task).
  - Scoring: parses the first word of the response ("yes"/"Yes..." ->
    yes, everything else -> no -- standard POPE/LLaVA-eval convention),
    classifies against the pinned yes/no `answer`, and reports POPE's own
    standard metrics (accuracy, precision, recall, F1, yes-ratio) rather
    than soft-VQA accuracy. yes-ratio matters specifically because POPE's
    own paper uses it as a hallucination-bias indicator: a model that
    over-answers "yes" will inflate recall while degrading precision on
    the adversarial category in particular.
  - Category breakdown: the pinned manifest is stratified across POPE's
    three categories (random/popular/adversarial -- see
    pope_pin_manifest.py); the summary reports metrics both overall and
    per-category, since collapsing to one aggregate number would hide
    exactly the failure mode (adversarial-specific hallucination) POPE
    exists to surface.
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from llama_provenance import resolve_build_provenance  # noqa: E402

SYSTEM_PROMPT = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful, detailed, and polite answers to the user's questions.")
JINJA_TEMPLATE = (Path(__file__).parent / "vicuna_v1_llava.jinja").read_text()


def build_prompt(question: str) -> str:
    return f"\n{question}\nPlease answer yes or no."


def classify_yn(text: str) -> str:
    w = text.strip().lower().lstrip(".,!?\"'")
    return "yes" if w.startswith("yes") else "no"


# ---------------- CLI mode (one process per sample) --------------------------
def run_one_cli(args, ratio: float, image_path: Path, question: str):
    cmd = [
        args.bin,
        "-m", args.model, "--mmproj", args.mmproj,
        "--image", str(image_path),
        "-sys", SYSTEM_PROMPT,
        "-p", build_prompt(question),
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
                done[(rec["id"], rec["question_id"])] = rec
            except Exception:
                continue
    return done


def ratio_complete(results_dir: Path, tag: str) -> bool:
    return bool(glob.glob(str(results_dir / f"*_{tag}_summary.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with <image_source>.png + meta.jsonl "
                    "(see materialize_pope_images.py)")
    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", str(REPO_ROOT.parent / "llama.cpp"))
    ap.add_argument("--bin", default=llama_cpp_dir + "/build/bin/llama-mtmd-cli")
    ap.add_argument("--server-bin", default=llama_cpp_dir + "/build/bin/llama-server")
    ap.add_argument("--llama-repo", default=llama_cpp_dir)
    ap.add_argument("--model", default=str(REPO_ROOT / "models/llava-v1.5-7b-Q4_K_M.gguf"))
    ap.add_argument("--mmproj", default=str(REPO_ROOT / "models/llava-v1.5-7b-mmproj-model-f16.gguf"))
    ap.add_argument("--ratios", default="1.0,0.75,0.5,0.25,0.1,0.05")
    ap.add_argument("--visual-prune-method", default="cls")
    ap.add_argument("--limit", type=int, default=0, help="0 = full pinned manifest")
    ap.add_argument("--tag-prefix", default="p4_pope_keep_sweep")
    ap.add_argument("--platform-tag", required=True)
    ap.add_argument("--build-desc", required=True)
    ap.add_argument("--extra-arg", action="append", default=[],
                    help="passthrough flag, repeatable, e.g. --extra-arg=-ngl --extra-arg=999")
    ap.add_argument("--mode", choices=["auto", "server", "cli"], default="auto")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8812)
    ap.add_argument("--server-startup-timeout-s", type=float, default=180.0)
    args = ap.parse_args()

    ratios = [float(r) for r in args.ratios.split(",")]
    data = Path(args.data)
    metas = [json.loads(l) for l in (data / "meta.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        metas = metas[: args.limit]
    if not metas:
        raise SystemExit(f"no samples found under {data} -- run materialize_pope_images.py first")

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
        image0 = data / f"{m0['image_source']}.png"
        cli_pred, cli_err = run_one_cli(args, 1.0, image0, m0["question"])
        if cli_pred is None:
            print(f"[sweep] WARNING: CLI equivalence probe itself failed ({cli_err}); "
                  f"cannot verify equivalence, forcing CLI mode", flush=True)
            use_server = False
        else:
            log_path = raw_dir / "pope_server_equivalence_probe.log"
            proc, log_f = start_server(args, 1.0, log_path)
            try:
                if not wait_for_server(args.host, args.port, args.server_startup_timeout_s):
                    print(f"[sweep] WARNING: server did not become healthy within "
                          f"{args.server_startup_timeout_s}s (see {log_path}); "
                          f"falling back to CLI mode", flush=True)
                    use_server = False
                else:
                    server_pred = query_server(args.host, args.port, image0, build_prompt(m0["question"]))
                    if server_pred.strip() == cli_pred.strip():
                        print(f"[sweep] equivalence OK -- CLI and server produced identical text "
                              f"for sample 0 at keep=1.0: {server_pred!r}. Using server mode for "
                              f"the bulk sweep.", flush=True)
                    else:
                        cli_pred2, cli_err2 = run_one_cli(args, 1.0, image0, m0["question"])
                        cli_is_reproducible = (cli_err2 is None and cli_pred2.strip() == cli_pred.strip())
                        if not cli_is_reproducible:
                            print(f"[sweep] server/CLI text differed, BUT a second CLI run on the same "
                                  f"sample also disagreed with the first ({cli_pred!r} vs "
                                  f"{cli_pred2!r}) -- this backend is not deterministic at the "
                                  f"token-decode level, so the original mismatch is inconclusive, "
                                  f"not proof of a harness bug. Proceeding with server mode; sanity-"
                                  f"check the accuracy numbers once the sweep completes.", flush=True)
                        else:
                            print(f"[sweep] EQUIVALENCE MISMATCH -- CLI reproduced its own answer "
                                  f"twice ({cli_pred!r}) but the server produced different text "
                                  f"({server_pred!r}) for the same sample/ratio. Falling back to "
                                  f"CLI mode (slower, but matches the validated methodology) for "
                                  f"the entire sweep.", flush=True)
                            use_server = False
            finally:
                stop_server(proc)
                log_f.close()
    if args.mode == "cli":
        use_server = False
    elif args.mode == "server" and not use_server:
        raise SystemExit("[sweep] --mode server was forced but the equivalence check failed or "
                          "the server did not come up -- refusing to silently run an unverified "
                          "prompt path. Use --mode auto to allow a CLI fallback.")
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
                key = (m["id"], m["question_id"])
                if key in done:
                    continue
                image_path = data / f"{m['image_source']}.png"
                prompt = build_prompt(m["question"])
                if server_proc is not None:
                    try:
                        pred, err = query_server(args.host, args.port, image_path, prompt), None
                    except Exception as e:
                        pred, err = None, str(e)
                else:
                    pred, err = run_one_cli(args, r, image_path, m["question"])
                if pred is None:
                    rec = {"id": m["id"], "question_id": m["question_id"], "error": err}
                else:
                    pred_yn = classify_yn(pred)
                    rec = {"id": m["id"], "question_id": m["question_id"], "category": m["category"],
                           "question": m["question"], "gold": m["answer"], "pred_text": pred,
                           "pred_yn": pred_yn, "correct": pred_yn == m["answer"]}
                with open(preds_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                done[key] = rec
                n = len(done)
                if n % 25 == 0 or n == len(metas):
                    el = time.time() - t0
                    accs = [d["correct"] for d in done.values() if "correct" in d]
                    mean_acc = (sum(accs) / len(accs) * 100) if accs else float("nan")
                    print(f"[sweep] {tag}: {n}/{len(metas)} ({el/60:.1f} min) "
                          f"acc so far: {mean_acc:.1f}", flush=True)
        finally:
            if server_proc is not None:
                stop_server(server_proc)
                log_f.close()

        ok = [d for d in done.values() if "correct" in d]
        fails = [d for d in done.values() if "error" in d]

        def metrics_for(rows):
            n = len(rows)
            if n == 0:
                return {"n": 0}
            tp = sum(1 for d in rows if d["gold"] == "yes" and d["pred_yn"] == "yes")
            fp = sum(1 for d in rows if d["gold"] == "no" and d["pred_yn"] == "yes")
            fn = sum(1 for d in rows if d["gold"] == "yes" and d["pred_yn"] == "no")
            tn = sum(1 for d in rows if d["gold"] == "no" and d["pred_yn"] == "no")
            acc = (tp + tn) / n
            precision = tp / (tp + fp) if (tp + fp) > 0 else None
            recall = tp / (tp + fn) if (tp + fn) > 0 else None
            f1 = (2 * precision * recall / (precision + recall)
                  if precision is not None and recall is not None and (precision + recall) > 0 else None)
            yes_ratio = sum(1 for d in rows if d["pred_yn"] == "yes") / n
            return {"n": n, "accuracy": acc, "precision": precision, "recall": recall,
                    "f1": f1, "yes_ratio": yes_ratio, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

        categories = sorted({d["category"] for d in ok})
        per_category = {c: metrics_for([d for d in ok if d["category"] == c]) for c in categories}

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
                          "custom vicuna_v1 jinja, 'Please answer yes or no.' suffix, "
                          f"extra_args={args.extra_arg}",
            },
            "dataset": "pinned POPE subset (assets/phase1/pope300_manifest.jsonl or as configured)",
            "metric": "POPE standard: accuracy/precision/recall/F1/yes-ratio on the binary "
                      "yes/no classification, parsed from the first word of the response",
            "n_scored": len(ok), "n_failed": len(fails),
            "overall": metrics_for(ok),
            "per_category": per_category,
            "preds_file": str(preds_path.relative_to(REPO_ROOT)),
        }
        summary_path = results_dir / f"{ts}_{tag}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[sweep] wrote {summary_path}: overall={summary['overall']}", flush=True)

    print("[sweep] all ratios complete", flush=True)


if __name__ == "__main__":
    main()
