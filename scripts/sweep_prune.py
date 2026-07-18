#!/usr/bin/env python3
"""Keep-ratio sweep driver for the visual-token-pruning feature.

Runs bench_baseline.py once per keep ratio against build-prune, with the
frozen G2 protocol (--cooldown-s 30, same flags otherwise) plus
--visual-keep/--visual-prune-method. Resumable at cell granularity: each
cell is one bench_baseline.py invocation that writes its own timestamped
JSON only on successful completion, so a crash mid-sweep loses at most
one in-progress cell, not prior ones. Skip logic here looks for an
existing results/*_p2_sweep_keep<r>.json before invoking a cell.

Does not compute the prune-overhead isolation or the H1/H2 analysis --
those are derived from the sweep's own (K, encode_ms) points across all
cells afterward (see NOTES.md), not measured as a separate cell.
"""

import argparse
import glob
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def cell_done(tag):
    return bool(glob.glob(str(REPO_ROOT / "results" / f"*_{tag}.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default=str(Path.home() / "Desktop/vtp/llama.cpp/build-prune/bin/llama-mtmd-cli"))
    ap.add_argument("--llama-repo", default=str(Path.home() / "Desktop/vtp/llama.cpp"))
    ap.add_argument("--ratios", default="1.0,0.75,0.5,0.25,0.1,0.05")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--cooldown-s", type=float, default=30.0)
    ap.add_argument("--tag-prefix", default="p2_sweep")
    ap.add_argument("--platform-tag", default="apple-m4-pro-dedicated",
                    help="forwarded to bench_baseline.py -- see its --platform-tag help")
    ap.add_argument("--build-desc", default=None,
                    help="forwarded to bench_baseline.py's --build-desc if set, else its default")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, do nothing")
    args = ap.parse_args()

    ratios = [float(r) for r in args.ratios.split(",")]
    for r in ratios:
        tag = f"{args.tag_prefix}_keep{r:g}"
        if cell_done(tag):
            print(f"[sweep] {tag}: already done, skipping", flush=True)
            continue

        cmd = [
            sys.executable, str(REPO_ROOT / "scripts" / "bench_baseline.py"),
            "--bin", args.bin, "--llama-repo", args.llama_repo,
            "--cooldown-s", str(args.cooldown_s),
            "--warmup", str(args.warmup), "--runs", str(args.runs),
            "--tag", tag, "--platform-tag", args.platform_tag,
            "--extra-arg=--visual-keep", f"--extra-arg={r}",
            "--extra-arg=--visual-prune-method", "--extra-arg=cls",
        ]
        if args.build_desc is not None:
            cmd += ["--build-desc", args.build_desc]
        print(f"[sweep] {tag}: {' '.join(cmd)}", flush=True)
        if args.dry_run:
            continue
        r_ = subprocess.run(cmd)
        if r_.returncode != 0:
            print(f"[sweep] {tag}: FAILED (exit {r_.returncode}) -- stopping sweep, rerun this "
                  f"script to resume (completed cells are skipped)", flush=True)
            sys.exit(1)

    print("[sweep] all cells complete" if not args.dry_run else "[sweep] dry-run complete")


if __name__ == "__main__":
    main()
