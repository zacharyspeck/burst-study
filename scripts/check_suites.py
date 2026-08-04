#!/usr/bin/env python
"""Run both test suites and report both counts, on a TRUE exit status.

WHY THIS EXISTS

Rule 4 requires the suite to be run in both environments and both counts
reported. The obvious way to do that at a shell -- `pytest -q | tail` -- is
WRONG in a way that is silent and green: a pipeline's exit status is the LAST
command's, so `tail` returning 0 masks pytest returning 1. That is not
hypothetical. On 2026-08-03 a run of 835 tests with 23 FAILURES was reported by
its harness as "exit code 0" for exactly this reason, and the failures were
noticed only because the printed count was read.

That is the repo's recurring defect -- a signal named for one thing and
measuring another -- landing on the mechanism used to check every other fix.
See S87.

So this script never pipes. It captures, parses the count, and exits nonzero if
EITHER environment fails. Determining "did the tests pass" from a value that
cannot be a pipeline artifact is the whole point.

USAGE

    python scripts/check_suites.py

Exit status is 0 only when both environments pass. Nothing here is clever; the
value is that the exit status means what it says.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: (label, venv directory, what a healthy run looks like). The torch-free run
#: existing at all is the point -- `burst/` must keep loading with no ML stack,
#: and the skips are the evidence -- so a run with ZERO skips there is as
#: suspicious as a failure and is reported.
ENVIRONMENTS = (
    ("no torch", ".venv", "expect skips, not failures"),
    ("torch + transformers", ".venv-ml", "expect no skips"),
)

#: pytest's summary line, e.g. "531 passed, 164 skipped in 23.89s" or
#: "23 failed, 812 passed in 193.29s".
#:
#: THE SEPARATOR IS ", " AND NOT A CHARACTER CLASS. This first read
#: `(\d+) failed[, ]`, which matches ONE character where pytest writes two --
#: so on "1 failed, 1 passed" the failed group silently did not match and the
#: reported count read "1 passed". A summary line that omits the failures is
#: the exact defect this file exists to prevent, reproduced inside the fix for
#: it. Caught by running the failure path instead of assuming it worked.
_COUNT = re.compile(
    r"(?:(\d+) failed, )?(\d+) passed"
    r"(?:, (\d+) skipped)?(?:, (\d+) errors?)?")


def _python_for(venv: str) -> Path | None:
    """The interpreter inside a venv, on Windows or POSIX."""
    for candidate in (REPO_ROOT / venv / "Scripts" / "python.exe",
                      REPO_ROOT / venv / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return None


def run_one(label: str, venv: str, note: str) -> tuple[bool, str]:
    """Run one environment's suite. Returns (ok, one-line summary)."""
    python = _python_for(venv)
    if python is None:
        return False, f"{label:22s} MISSING -- no interpreter in {venv}/"

    # No shell, no pipe. `capture_output` keeps the exit status pytest's own.
    result = subprocess.run(
        [str(python), "-m", "pytest", "-q"],
        cwd=REPO_ROOT, capture_output=True, text=True)

    tail = [line for line in result.stdout.splitlines() if line.strip()]
    summary = tail[-1] if tail else "(no output)"
    match = _COUNT.search(result.stdout)
    counts = match.group(0) if match else "unparsed"

    # BOTH conditions. A zero exit with no parsed count, or a parsed count
    # alongside a nonzero exit, means something happened that this script does
    # not understand -- and "does not understand" must not read as pass.
    ok = result.returncode == 0 and match is not None
    verdict = "PASS" if ok else "FAIL"
    detail = f"{label:22s} {verdict}  {counts}  (exit {result.returncode})"
    if not ok:
        detail += f"\n    last line: {summary}"
    return ok, detail


def main() -> int:
    print("=" * 74)
    print("both suites, counted and exited honestly -- see this file's docstring")
    print("=" * 74)
    results = []
    for label, venv, note in ENVIRONMENTS:
        ok, detail = run_one(label, venv, note)
        print(detail, flush=True)
        results.append(ok)
    print("-" * 74)
    if all(results):
        print("BOTH ENVIRONMENTS PASS")
        return 0
    print("AT LEAST ONE ENVIRONMENT FAILED -- read the counts above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
