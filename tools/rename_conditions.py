#!/usr/bin/env python
"""Rename the two fluent condition identifiers, everywhere they are allowed to change.

    fluent-false  ->  fluent-fabricated      fluent_false  ->  fluent_fabricated
    fluent-true   ->  fluent-attested        fluent_true   ->  fluent_attested

WHY THIS IS A SCRIPT AND NOT A SED COMMAND
------------------------------------------
The rename is not "replace this string everywhere". Roughly half the
occurrences in this repository are inside DATED RECORDS -- the pre-registration,
the decision log, the measurement write-ups, the preprint -- which the paper
cites by commit and line number. Editing those would shift line numbers and
would rewrite, after the fact, a document whose entire value is that it was
fixed before the data existed. They are listed in PROTECTED below and this
script refuses to touch them.

So the transformation has a boundary, the boundary is a judgement call, and a
judgement call that lives in someone's shell history is not auditable. It lives
here instead, as data, so that anyone reading this repository can see exactly
which files were rewritten and which were deliberately left alone.

WHAT IS IN SCOPE
----------------
Tracked files only, and only these kinds:

  * code                burst/, scripts/, tests/
  * configuration       configs/base.yaml
  * generated output    docs/measurements/*.json, bursts/provenance.json,
    and measurement     docs/measurements/2026-08-10-training-curves.npz

Untracked files are never touched, whatever they contain. Neither are the two
burst .txt files themselves: those are renamed with `git mv` so that git records
a rename rather than a delete plus an add, and their CONTENT is never opened by
this script. Their sha256 must be identical before and after, which is the check
that proves the rename moved a file and did not edit a passage.

configs/runs/*.yaml is also out of scope here. Those 40 override files are
generated, and the repository rule is that they are regenerated with
scripts/generate_overrides.py and never hand-edited -- a rule that a bulk
rewriter would quietly break. See RUNS_NOTE below.

USAGE
-----
    python tools/rename_conditions.py --check     # verify; writes nothing
    python tools/rename_conditions.py --apply     # perform the rename

--check is the mode worth remembering. It passes only when every in-scope file
is fully renamed AND every protected file still holds its original names, so it
catches both halves of the mistake: a rename that did not finish, and a rename
that reached somewhere it should not have.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# The mapping. Ordered longest-first is unnecessary here -- the four patterns
# are mutually disjoint, since no one of them is a prefix of another -- but the
# hyphen and underscore forms are both listed explicitly rather than being
# derived by a regex over a separator class, because being able to read the
# exact old and new string out of the source is the point of this file.
# ---------------------------------------------------------------------------
MAPPING: dict[str, str] = {
    "fluent-false": "fluent-fabricated",
    "fluent_false": "fluent_fabricated",
    "fluent-true": "fluent-attested",
    "fluent_true": "fluent_attested",
    # Case variants, which exist for exactly one reason: the loader compares arm
    # names WITHOUT normalisation, and a handful of comments and test cases use
    # a miscapitalised arm as the example of a string that must be rejected.
    # They have to move with the rest. If they do not, the "Case matters" hint
    # stops firing -- the hint only triggers when arm.lower() is a real arm, so
    # a case variant of a RETIRED name silently becomes an ordinary unknown-arm
    # error and the test asserting the hint fails. Found that way, not by
    # reading. Listed as literals rather than handled with re.IGNORECASE, which
    # would make the replacement's capitalisation unpredictable.
    "Fluent-False": "Fluent-Fabricated",
    "FLUENT-FALSE": "FLUENT-FABRICATED",
    "Fluent-True": "Fluent-Attested",
    "FLUENT-TRUE": "FLUENT-ATTESTED",
}

#: Matches any old identifier. Used for counting and for --check.
OLD_RE = re.compile("|".join(re.escape(k) for k in MAPPING))

#: Matches any new identifier. Used to prove protected files stayed protected.
NEW_RE = re.compile("|".join(re.escape(v) for v in MAPPING.values()))

# ---------------------------------------------------------------------------
# PROTECTED -- dated records. Never rewritten, by rule, not by accident.
#
# The rule, as ruled on 2026-08-18: no condition rename inside ANY dated record.
# A dated record is one whose value depends on it saying what it said on the day
# it was written. The paper cites several of these by commit and line number.
# ---------------------------------------------------------------------------
PROTECTED: frozenset[str] = frozenset({
    "docs/preregistration.md",
    "docs/decisions-pending.md",
    "docs/preprint.md",
    "docs/preprint-source-material.md",
    "docs/spec-v4.md",
    "docs/v4-gap-analysis.md",
    "docs/handoff-pilot.md",
    "docs/contradiction-scan-2026-08-04.md",
    # Existing content is frozen; new entries are appended under the new names.
    "implementation-notes.md",
    # The pre-public README, moved aside on 2026-08-18. It describes the v3->v4
    # arm reconciliation as of the day it was written, which makes it a dated
    # record in every sense that matters, even though its old filename was not.
    "docs/README-dev.md",
})

#: Every .md under docs/measurements/ is a dated write-up. The .json beside them
#: are machine-generated result files keyed by run name, and those DO get
#: renamed -- the split is deliberate, and it is the one part of this boundary
#: most likely to surprise a reader, so it is spelled out rather than implied.
PROTECTED_GLOBS: tuple[str, ...] = (
    "docs/measurements/*.md",
    "docs/measurements/*/*.md",
)

# ---------------------------------------------------------------------------
# Files that legitimately contain BOTH the old and the new names, because their
# job is to carry the translation. Exempt from both halves of --check.
#
# Without this, --check fails on the two files that are most obviously correct,
# and a check that cries wolf gets ignored, which costs more than it saves.
# ---------------------------------------------------------------------------
MAPPING_SOURCES: frozenset[str] = frozenset({
    # This file. MAPPING is written here as literals on purpose.
    "tools/rename_conditions.py",
    # The public README states the mapping in a table near the top, so that a
    # reader arriving from the paper can reconcile `fluent-false` in the
    # pre-registration against `fluent-fabricated` in a config. Those old names
    # are the point of the section, not a missed rename.
    "README.md",
})

#: Protected against having its EXISTING content renamed, but new entries are
#: appended under the new names, so it will contain them and that is correct.
#: The append-only property is not checkable from file content alone. It is
#: enforced at review time by the commit diff showing zero deleted lines.
APPEND_ONLY: frozenset[str] = frozenset({
    "implementation-notes.md",
})

RUNS_NOTE = """configs/runs/*.yaml is generated. Regenerate it instead:

    python scripts/generate_overrides.py
    python scripts/generate_overrides.py --check

generate_overrides.py writes the new names but does NOT delete the old files,
so the 20 stale seedNN_fluent-{false,true}.yaml need `git rm` as well."""

#: The one committed binary carrying identifier keys: a `runs` array of
#: "seed00_fluent-false"-style names alongside the loss and gradient-norm
#: matrices. Handled separately, and the numeric arrays are asserted unchanged.
NPZ_PATH = "docs/measurements/2026-08-10-training-curves.npz"


def tracked_files() -> list[str]:
    """Every path git tracks, as repo-relative posix strings."""
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"],
        capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line]


def is_protected(rel: str) -> bool:
    if rel in PROTECTED or rel in APPEND_ONLY:
        return True
    return any(fnmatchcase(rel, g) for g in PROTECTED_GLOBS)


def in_scope(rel: str) -> bool:
    """Text files this script may rewrite."""
    if is_protected(rel) or rel in MAPPING_SOURCES:
        return False
    if rel.startswith("configs/runs/"):
        return False          # generated; see RUNS_NOTE
    if rel.startswith("bursts/") and rel.endswith(".txt"):
        return False          # renamed with `git mv`; content never opened
    if rel == NPZ_PATH:
        return False          # handled by rename_npz()
    return rel.endswith((".py", ".json", ".yaml", ".yml", ".md", ".txt", ".cfg",
                         ".toml", ".ini"))


def read_text(rel: str) -> str | None:
    """Decode as UTF-8 from raw bytes, deliberately NOT via read_text().

    read_text() applies universal-newline translation, so a CRLF file read that
    way and written back would silently become LF. Every byte of a measurement
    file matters here, including the ones that are line endings, so the round
    trip is bytes -> str -> bytes with no newline handling in either direction.
    """
    try:
        return (REPO / rel).read_bytes().decode("utf-8")
    except (UnicodeDecodeError, FileNotFoundError):
        return None


def rewrite(text: str) -> str:
    for old, new in MAPPING.items():
        text = text.replace(old, new)
    return text


def rename_npz(apply: bool) -> tuple[int, str]:
    """Rename the run-name strings in the training-curves archive.

    The numeric arrays are compared element-for-element before and after and the
    function refuses to write if any of them moved. A measurement file is the
    one place where a rename must be provably not a recomputation.
    """
    try:
        import numpy as np
    except ImportError:
        return 0, f"SKIPPED {NPZ_PATH}: numpy unavailable in this interpreter"

    path = REPO / NPZ_PATH
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}

    before = [str(r) for r in arrays["runs"]]
    after = [rewrite(r) for r in before]
    n = sum(1 for a, b in zip(before, after) if a != b)
    if n == 0:
        return 0, f"{NPZ_PATH}: already renamed"
    if not apply:
        return n, f"{NPZ_PATH}: {n} run names would be renamed"

    numeric = {k: v.copy() for k, v in arrays.items() if k not in ("runs",)}
    arrays["runs"] = np.array(after)
    np.savez_compressed(path, **arrays)

    with np.load(path) as z:
        for k, want in numeric.items():
            got = z[k]
            if got.dtype != want.dtype or got.shape != want.shape or \
                    not (got == want).all():
                raise SystemExit(
                    f"ABORT: {NPZ_PATH} array {k!r} changed during rename. "
                    "The file has been written and must be restored with "
                    "`git checkout -- " + NPZ_PATH + "`.")
        if [str(r) for r in z["runs"]] != after:
            raise SystemExit(f"ABORT: {NPZ_PATH} run names did not round-trip.")
    return n, f"{NPZ_PATH}: {n} run names renamed, numeric arrays verified identical"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true",
                      help="verify the rename is complete and contained; write nothing")
    mode.add_argument("--apply", action="store_true", help="perform the rename")
    args = ap.parse_args(argv)

    files = tracked_files()
    scoped = [f for f in files if in_scope(f)]
    protected = [f for f in files if is_protected(f)]

    # ---- the half that must be fully renamed --------------------------------
    changed: list[tuple[str, int]] = []
    for rel in scoped:
        text = read_text(rel)
        if text is None:
            continue
        hits = len(OLD_RE.findall(text))
        if not hits:
            continue
        changed.append((rel, hits))
        if args.apply:
            (REPO / rel).write_bytes(rewrite(text).encode("utf-8"))

    npz_hits, npz_msg = rename_npz(args.apply)

    # ---- the half that must NOT have been renamed ---------------------------
    # APPEND_ONLY files are excluded: new entries in them use the new names by
    # design. MAPPING_SOURCES are excluded because carrying both names is their
    # whole purpose.
    leaked: list[tuple[str, int]] = []
    for rel in protected:
        if rel in APPEND_ONLY or rel in MAPPING_SOURCES:
            continue
        text = read_text(rel)
        if text is None:
            continue
        n = len(NEW_RE.findall(text))
        if n:
            leaked.append((rel, n))

    verb = "renamed" if args.apply else "would rename"
    for rel, hits in sorted(changed):
        print(f"  {hits:5d}  {verb}  {rel}")
    if npz_hits or args.check:
        print(f"         {npz_msg}")

    total = sum(h for _, h in changed)
    print(f"\n{len(changed)} files, {total} occurrences, plus {npz_hits} run "
          f"names in the archive.")
    print(f"{len(protected)} protected files left untouched by rule.")

    if args.apply:
        print("\n" + RUNS_NOTE)
        return 0

    # --check verdict
    ok = True
    if changed or npz_hits:
        print(f"\nFAIL: {total + npz_hits} old identifiers remain in scope.")
        ok = False
    if leaked:
        print("\nFAIL: new identifiers found inside protected dated records:")
        for rel, n in leaked:
            print(f"  {n:5d}  {rel}")
        ok = False
    print("\nOK: rename is complete and contained." if ok else "")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
