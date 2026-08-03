#!/usr/bin/env python
"""Stage 1 of the corpus pipeline: fetch pinned source files. Step 11.

Downloads OpenWebText's Parquet files from a PINNED REVISION into a local
cache, verifies each one, and records enough to identify exactly what was
pulled. Tokenizes nothing -- that is stage 2, and the split is what makes a
crash at hour six cost one file rather than the job.

WHY FILES RATHER THAN A STREAM

`scripts/make_bursts.py` streams the dataset, which is fine for 200 documents
and wrong for 2.5B tokens: a stream has no seek, so resuming means re-reading
everything consumed so far, and a stream of `main` is not identified by
anything. The Hub serves this dataset as 80 numbered Parquet files at a fixed
revision, so pulling files instead gives all three things a stream cannot:

  - RESUMABILITY. A file is present and verified, or it is not.
  - PINNING. A revision plus per-file hashes identifies the corpus exactly.
    "The corpus downloaded today" and "the corpus downloaded in a month" become
    the same object or provably different ones.
  - ORDER FROM ARITHMETIC. `train-00000-of-00080` through `train-00079-of-00080`
    is an order the filenames carry. Nothing here reads a directory listing,
    which would put filesystem enumeration order into the study's data order.

WHAT IS NEVER COMMITTED

Everything this writes. The cache directory is a command-line argument and
defaults under `.corpus-cache/`, which `.gitignore` covers at the repo root.
Corpus data is NAMED in configs/base.yaml and never located there, never
vendored, and never tracked -- CLAUDE.md rule 3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import corpus_spec as SPEC  # noqa: E402

DEFAULT_CACHE = REPO_ROOT / ".corpus-cache" / "openwebtext"
FETCH_MANIFEST = "fetch-manifest.json"

#: Read in 8 MiB blocks when hashing. Large enough that the syscall overhead
#: disappears, small enough that a 500 MB Parquet file never lands in RAM.
HASH_BLOCK = 8 * 1024 * 1024


class CorpusFetchError(Exception):
    """Raised for every refusal here. Never an assert."""


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(HASH_BLOCK), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _hub():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise CorpusFetchError(
            "huggingface_hub is not installed, so the corpus cannot be "
            "fetched.\n    pip install -e \".[dev,measure]\"\n"
            f"(underlying import error: {exc})") from exc
    return hf_hub_download


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def load_manifest(cache: Path) -> dict:
    path = cache / FETCH_MANIFEST
    if not path.is_file():
        return {"dataset": SPEC.DATASET, "revision": SPEC.REVISION,
                "files": {}}
    manifest = json.loads(path.read_text(encoding="utf-8"))

    # A cache built against a different revision is not this corpus. Merging
    # into it would produce a directory of files from two snapshots with a
    # manifest naming one of them -- the exact "looks fine, is not the same
    # object" failure the pinning exists to prevent.
    if manifest.get("revision") != SPEC.REVISION:
        raise CorpusFetchError(
            f"{path} was built against revision {manifest.get('revision')!r}, "
            f"but corpus_spec pins {SPEC.REVISION!r}.\n"
            "These are different snapshots of the dataset. Fetching into the "
            "same directory would mix them and leave a manifest describing "
            "only one. Use a different --cachedir, or delete this one "
            "deliberately.")
    return manifest


def save_manifest(cache: Path, manifest: dict) -> None:
    """Written atomically, so a crash mid-write cannot corrupt the record.

    The manifest is the only thing that says which files are verified. A
    half-written one would make a complete cache look partial or, worse, a
    partial one look complete.
    """
    path = cache / FETCH_MANIFEST
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(manifest, indent=2) + "\n",
                       encoding="utf-8", newline="\n")
    os.replace(partial, path)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


def verify_cached_file(cache: Path, record: dict) -> bool:
    """Is this file present, the right size, and the right hash?

    Size is checked before the hash on purpose: it is free, it catches the
    common failure (a truncated download), and it means a full re-hash of a
    500 MB file is only paid when the cheap check passes.
    """
    path = cache / record["filename"]
    if not path.is_file():
        return False
    if path.stat().st_size != record["bytes"]:
        return False
    return sha256_file(path) == record["sha256"]


def fetch_file(index: int, cache: Path, manifest: dict, stream) -> dict:
    """Fetch one source file, or confirm the one already cached.

    Returns its manifest record. Re-verifies an existing record rather than
    trusting it: a manifest entry proves what was true when it was written, not
    what is on disk now.
    """
    filename = SPEC.SOURCE_TEMPLATE.format(index=index)
    key = str(index)
    record = manifest["files"].get(key)

    if record and verify_cached_file(cache, record):
        print(f"    [{index:02d}/{SPEC.N_SOURCE_FILES}] cached and verified: "
              f"{record['bytes']:,} bytes", file=stream, flush=True)
        return record

    if record:
        print(f"    [{index:02d}] cached copy FAILED verification -- "
              f"refetching", file=stream, flush=True)

    hf_hub_download = _hub()
    started = time.perf_counter()
    try:
        downloaded = hf_hub_download(
            repo_id=SPEC.DATASET,
            filename=filename,
            revision=SPEC.REVISION,
            repo_type="dataset",
            local_dir=cache,
        )
    except Exception as exc:
        raise CorpusFetchError(
            f"could not fetch {filename} at revision {SPEC.REVISION}: {exc}\n"
            "Either connect to a network and run again, or copy a populated "
            "cache directory into place from a machine that has one. Nothing "
            "already fetched is lost -- rerun and it resumes.") from exc

    path = Path(downloaded)
    size = path.stat().st_size
    digest = sha256_file(path)
    elapsed = time.perf_counter() - started
    print(f"    [{index:02d}/{SPEC.N_SOURCE_FILES}] fetched "
          f"{size:,} bytes in {elapsed:.1f}s  sha {digest[:12]}",
          file=stream, flush=True)

    return {
        "index": index,
        "filename": str(path.relative_to(cache)).replace("\\", "/"),
        "hub_path": filename,
        "bytes": size,
        "sha256": digest,
        "seconds": round(elapsed, 2),
    }


def fetch(n_files: int, cache: Path, stream, verify_only: bool = False) -> dict:
    """Fetch the first n_files source files, resuming whatever is already there."""
    if not 1 <= n_files <= SPEC.N_SOURCE_FILES:
        raise CorpusFetchError(
            f"n_files must be 1..{SPEC.N_SOURCE_FILES}; got {n_files}")
    cache.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cache)

    for index in range(n_files):
        record = manifest["files"].get(str(index))
        if verify_only:
            ok = bool(record) and verify_cached_file(cache, record)
            print(f"    [{index:02d}] {'OK' if ok else 'MISSING OR CORRUPT'}",
                  file=stream, flush=True)
            if not ok:
                raise CorpusFetchError(
                    f"source file {index} is missing or does not match its "
                    f"recorded hash. The cache is not the corpus the manifest "
                    f"describes.")
            continue
        manifest["files"][str(index)] = fetch_file(index, cache, manifest,
                                                   stream)
        # Saved after EVERY file, not at the end: a crash at file 20 must leave
        # 20 verified files recorded, not an empty manifest beside 20 files
        # nothing vouches for.
        save_manifest(cache, manifest)

    manifest["n_files_fetched"] = len([
        k for k in manifest["files"] if int(k) < n_files])
    manifest["bytes_total"] = sum(
        r["bytes"] for k, r in manifest["files"].items() if int(k) < n_files)
    if not verify_only:
        save_manifest(cache, manifest)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/corpus_fetch.py",
        description="Fetch pinned OpenWebText source files. Tokenizes nothing.")
    parser.add_argument(
        "--cachedir", type=Path, default=DEFAULT_CACHE,
        help="where source files land. A command-line argument and never a "
             "config value, so one config runs unchanged everywhere.")
    parser.add_argument(
        "--files", type=int, default=25,
        help=("how many of the 80 source files to fetch. The default is a "
              "starting estimate, not a measurement: stage 2 reports how many "
              "were actually consumed and refuses loudly if it runs short."))
    parser.add_argument(
        "--verify-only", action="store_true",
        help="re-verify what is already cached and fetch nothing")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    stream = sys.stdout
    rule = "=" * 78
    print(rule, file=stream)
    print("corpus_fetch -- step 11 stage 1", file=stream)
    print(rule, file=stream)
    print(f"dataset:  {SPEC.DATASET}", file=stream)
    print(f"revision: {SPEC.REVISION}", file=stream)
    print(f"cache:    {args.cachedir}", file=stream)
    print(f"files:    {args.files} of {SPEC.N_SOURCE_FILES}\n", file=stream)

    started = time.perf_counter()
    try:
        manifest = fetch(args.files, args.cachedir, stream,
                         verify_only=args.verify_only)
    except CorpusFetchError as exc:
        print(f"\nFAILED: {exc}", file=stream)
        return 1
    elapsed = time.perf_counter() - started

    total = manifest.get("bytes_total", 0)
    print(f"\n{manifest.get('n_files_fetched', 0)} files, {total:,} bytes "
          f"({total / 1e9:.2f} GB) in {elapsed:.1f}s", file=stream)
    print(f"manifest: {args.cachedir / FETCH_MANIFEST}", file=stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
