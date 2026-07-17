"""Shared llama.cpp build-provenance helper.

The commit powering a benchmark run must come from the BINARY under test,
not from `git rev-parse HEAD` on the llama.cpp checkout: the working tree
can move to a different branch after a build/*/bin/llama-mtmd-cli was
compiled (this happened mid-project — see NOTES.md), which would silently
mis-stamp results with the wrong commit. Every llama.cpp binary self-reports
its build commit via `--version`; that is the authoritative source here.

The repo's currently checked-out HEAD is still recorded, as a secondary
field, so a human can see at a glance whether it matches the binary (it need
not — the tree may have legitimately moved on since the build). Read-only
git calls only (rev-parse, no checkout) — never touches the llama.cpp tree.
"""

import re
import subprocess
from pathlib import Path

_VERSION_RE = re.compile(r"version:\s*\d+\s*\(([0-9a-f]+)\)")


def _run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    # llama-mtmd-cli prints --version to stderr, not stdout; git prints to stdout.
    return (r.stdout.strip() or r.stderr.strip()) or None


def resolve_build_provenance(bin_path: str, llama_repo: str) -> dict:
    """Authoritative provenance for a specific llama.cpp binary.

    Returns a dict with the binary-reported commit (source of truth) and
    the repo's checked-out HEAD (secondary, for cross-check/visibility).
    """
    version_out = _run([bin_path, "--version"])
    bin_short = None
    if version_out:
        m = _VERSION_RE.search(version_out)
        if m:
            bin_short = m.group(1)

    bin_full = None
    if bin_short:
        bin_full = _run(["git", "-C", llama_repo, "rev-parse", f"{bin_short}^{{commit}}"])

    repo_head = _run(["git", "-C", llama_repo, "rev-parse", "HEAD"])
    repo_branch = _run(["git", "-C", llama_repo, "rev-parse", "--abbrev-ref", "HEAD"])

    return {
        "bin_path": str(Path(bin_path).resolve()),
        "bin_version_string": version_out.splitlines()[0] if version_out else None,
        "bin_commit_short": bin_short,
        "bin_commit": bin_full,
        "repo_head_commit": repo_head,
        "repo_head_branch": repo_branch,
        "repo_head_matches_binary": (repo_head == bin_full) if (repo_head and bin_full) else None,
        "source_of_truth": (
            "bin_commit, parsed from the binary's own --version output. "
            "repo_head_commit is the llama.cpp checkout's HEAD at record time "
            "and may legitimately differ if the tree moved on after this binary "
            "was built; it is recorded so any mismatch is visible, not hidden."
        ),
    }
