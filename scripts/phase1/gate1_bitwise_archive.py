#!/usr/bin/env python3
"""Gate 0.5 + Gate 1 bitwise checks on CPU, archived this time.

The original 2026-07-18 session ran both checks (CPU encode determinism on
pristine build-both; pristine vs build-prune @ --visual-keep 1.0 with
prune-method none and cls) but never saved the .npy dumps -- an archiving
gap found in the 2026-07-20 audit. This re-runs all four projector-output
dumps via dump_llamacpp_embd.py, performs the np.array_equal comparisons,
and writes the dumps plus a timestamped result JSON to results/.

Gate 0.5: two encodes of the same image on the pristine build must be
bitwise identical (justifies np.array_equal as the Gate 1 operator).
Gate 1: build-prune at keep=1.0 must be bitwise identical to the pristine
build, with the pruning branch gated off entirely (method none) and with
scoring enabled but keep=1.0 (method cls).
"""

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DUMP = Path(__file__).parent / "dump_llamacpp_embd.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def binary_provenance(bin_dir: Path) -> dict:
    cli = bin_dir / "llama-mtmd-cli"
    ver = subprocess.run([str(cli), "--version"], capture_output=True, text=True)
    version_line = (ver.stderr + ver.stdout).strip().splitlines()[0]
    hlp = subprocess.run([str(cli), "--help"], capture_output=True, text=True)
    has_flag = "visual-keep" in (hlp.stdout + hlp.stderr)
    return {"bin_dir": str(bin_dir), "version": version_line, "has_visual_keep_flag": has_flag}


def run_dump(out: Path, lib_dir: Path, image: Path, pruned_abi: bool, method: str | None) -> list:
    cmd = [sys.executable, str(DUMP), "--image", str(image), "--out", str(out),
           "--lib-dir", str(lib_dir)]
    if pruned_abi:
        cmd += ["--pruned-abi", "--visual-keep", "1.0", "--visual-prune-method", method]
    print("[gate] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return cmd


def compare(a: Path, b: Path) -> dict:
    x, y = np.load(a), np.load(b)
    return {
        "a": a.name, "b": b.name,
        "shape_a": list(x.shape), "shape_b": list(y.shape),
        "array_equal": bool(np.array_equal(x, y)),
        "max_abs_diff": float(np.max(np.abs(x - y))) if x.shape == y.shape else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pristine-bin", type=Path,
                    default=Path.home() / "Desktop/vtp/llama.cpp-worktree-pristine/build-both-pristine/bin")
    ap.add_argument("--prune-bin", type=Path,
                    default=Path.home() / "Desktop/vtp/llama.cpp/build-prune/bin")
    ap.add_argument("--image", type=Path, default=REPO_ROOT / "assets/phase1/cat_336.png")
    args = ap.parse_args()

    pristine = binary_provenance(args.pristine_bin)
    prune = binary_provenance(args.prune_bin)
    if pristine["has_visual_keep_flag"] or not prune["has_visual_keep_flag"]:
        raise SystemExit(f"two-signal binary check failed: {pristine} / {prune}")

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = "gate1_bitwise_cpu"
    raw_dir = REPO_ROOT / "results" / "raw" / f"{ts}_{tag}"
    raw_dir.mkdir(parents=True)

    dumps = {
        "pristine_run1": (args.pristine_bin, False, None),
        "pristine_run2": (args.pristine_bin, False, None),
        "prune_keep1_none": (args.prune_bin, True, "none"),
        "prune_keep1_cls": (args.prune_bin, True, "cls"),
    }
    commands = {}
    for name, (lib, abi, method) in dumps.items():
        commands[name] = run_dump(raw_dir / f"{name}.npy", lib, args.image, abi, method)

    checks = {
        "gate0_5_determinism": compare(raw_dir / "pristine_run1.npy", raw_dir / "pristine_run2.npy"),
        "gate1_vs_method_none": compare(raw_dir / "pristine_run1.npy", raw_dir / "prune_keep1_none.npy"),
        "gate1_vs_method_cls": compare(raw_dir / "pristine_run1.npy", raw_dir / "prune_keep1_cls.npy"),
    }
    all_pass = all(c["array_equal"] and c["max_abs_diff"] == 0.0 for c in checks.values())

    out = {
        "tag": tag,
        "timestamp": ts,
        "command": " ".join(sys.argv),
        "purpose": ("re-run and archive the Gate 0.5 / Gate 1 bitwise checks whose original "
                    "2026-07-18 dumps were never saved (audit-found gap, closed 2026-07-20)"),
        "binaries": {"pristine_build_both": pristine, "build_prune": prune},
        "files": {
            "image": {"path": str(args.image.relative_to(REPO_ROOT)), "sha256": sha256(args.image)},
            "model": {"path": "models/llava-v1.5-7b-Q4_K_M.gguf",
                      "sha256": sha256(REPO_ROOT / "models/llava-v1.5-7b-Q4_K_M.gguf")},
            "mmproj": {"path": "models/llava-v1.5-7b-mmproj-model-f16.gguf",
                       "sha256": sha256(REPO_ROOT / "models/llava-v1.5-7b-mmproj-model-f16.gguf")},
        },
        "dump_commands": {k: " ".join(v) for k, v in commands.items()},
        "npy_sha256": {p.name: sha256(p) for p in sorted(raw_dir.glob("*.npy"))},
        "checks": checks,
        "verdict": "PASS" if all_pass else "FAIL",
        "raw_dir": str(raw_dir.relative_to(REPO_ROOT)),
    }
    out_path = REPO_ROOT / "results" / f"{ts}_{tag}.json"
    out_path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"[gate] {out['verdict']} -> {out_path.relative_to(REPO_ROOT)}")
    for k, c in checks.items():
        print(f"  {k}: array_equal={c['array_equal']} max_abs_diff={c['max_abs_diff']}")
    if not all_pass:
        raise SystemExit(1)  # dumps and JSON are kept for inspection


if __name__ == "__main__":
    main()
