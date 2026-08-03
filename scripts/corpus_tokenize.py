#!/usr/bin/env python
"""Stage 2 of the corpus pipeline: tokenize into shards. Step 11.

Reads the pinned Parquet files stage 1 cached, tokenizes them in a fixed order,
and writes exactly one held-out file plus 149 training shards of raw
little-endian uint16.

    absolute token   0 .............. 10,485,760 .............. 2,510,290,944
                     |  heldout.bin   |  train-000.bin .. train-148.bin      |

WHAT MAKES THIS RESUMABLE

A shard is written to `<name>.partial`, flushed, hashed, and only then renamed
into place. A `.partial` file is ALWAYS discarded on restart and never resumed
mid-file: a half-written shard whose tail is garbage is indistinguishable from
a whole one by size alone, and resuming into it would produce a corpus with a
seam nothing could find afterwards.

Resuming also has to put the SOURCE back where it was, or shard 40 written
today would contain different text than shard 40 written yesterday. So the
manifest records, for every shard boundary, the exact position in the source:
which file, which row, and how many tokens into that row. A shard boundary
lands mid-document far more often than not, and rounding it to a document
boundary would silently drop or duplicate text.

EXACTNESS

The token budget is asserted, never padded and never silently truncated. The
last document IS cut mid-way -- that is unavoidable when the budget is a fixed
number -- and where it was cut is recorded rather than shrugged at.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import corpus_spec as SPEC  # noqa: E402
from corpus_fetch import DEFAULT_CACHE, FETCH_MANIFEST, sha256_file  # noqa: E402

DEFAULT_OUTDIR = REPO_ROOT / ".corpus-cache" / "tokenized"
TOKENIZE_MANIFEST = "manifest.json"

#: Documents tokenized per call. The fast tokenizer batches internally, and
#: 1000 documents is where the per-call overhead stops mattering without the
#: encoded batch becoming large enough to notice in RAM.
DOC_BATCH = 1000


class CorpusTokenizeError(Exception):
    """Raised for every refusal here. Never an assert."""


# ---------------------------------------------------------------------------
# The source, with a position that survives a restart
# ---------------------------------------------------------------------------


class TokenSource:
    """Documents from the cached Parquet files, as a token stream with a place.

    The position is (file_index, row_index, token_offset): the next token to be
    emitted is `token_offset` tokens into row `row_index` of file
    `file_index`. Recorded at every shard boundary, it is what lets a restart
    produce byte-identical shards rather than merely plausible ones.
    """

    def __init__(self, cache: Path, tokenizer, n_files: int,
                 file_index: int = 0, row_index: int = 0,
                 token_offset: int = 0):
        self.cache = cache
        self.tokenizer = tokenizer
        self.n_files = n_files
        self.file_index = file_index
        self.row_index = row_index
        self.token_offset = token_offset
        self._pending = deque()      # (row_index, tokens) not yet consumed
        self._exhausted = False
        self._rowgroup_rows = []     # row counts for the current file
        self._rowgroup_rows_file = None

    def position(self) -> dict:
        return {"file_index": self.file_index, "row_index": self.row_index,
                "token_offset": self.token_offset}

    def _read_batch(self) -> None:
        """Load the next DOC_BATCH documents, skipping to the recorded row."""
        import pyarrow.parquet as pq

        while not self._pending and self.file_index < self.n_files:
            path = self.cache / SPEC.SOURCE_TEMPLATE.format(
                index=self.file_index)
            if not path.is_file():
                raise CorpusTokenizeError(
                    f"source file {self.file_index} is not in the cache at "
                    f"{path}. Run scripts/corpus_fetch.py --files "
                    f"{self.file_index + 1} first.")
            handle = pq.ParquetFile(path)

            # Row-group sizes are read ONCE per file and cached. They were
            # previously read as `handle.metadata.row_group(g).num_rows` inside
            # this loop, and `handle.metadata` rebuilds a FileMetaData object on
            # every access -- so the cost of skipping to the read position grew
            # as the position advanced through a file, and the per-shard time
            # roughly doubled over a run. The tokens produced are identical
            # either way: this is a seek cost, not a content decision.
            if self._rowgroup_rows_file != self.file_index:
                meta = handle.metadata
                self._rowgroup_rows = [
                    meta.row_group(g).num_rows
                    for g in range(handle.num_row_groups)]
                self._rowgroup_rows_file = self.file_index

            seen = 0
            texts, rows = [], []
            for group in range(handle.num_row_groups):
                n_rows = self._rowgroup_rows[group]
                # Whole row groups before the resume point are skipped without
                # being decoded. Resuming into file 20 must not cost reading
                # files 0..19.
                if seen + n_rows <= self.row_index:
                    seen += n_rows
                    continue
                table = handle.read_row_group(group, columns=["text"])
                column = table.column("text").to_pylist()
                start = max(0, self.row_index - seen)
                for offset in range(start, len(column)):
                    texts.append(column[offset])
                    rows.append(seen + offset)
                    if len(texts) >= DOC_BATCH:
                        break
                seen += n_rows
                if len(texts) >= DOC_BATCH:
                    break

            if not texts:
                # This file is spent; move to the next one and reset the row
                # counter, which is per-file.
                self.file_index += 1
                self.row_index = 0
                continue

            encoded = self.tokenizer(texts)["input_ids"]
            for row, tokens in zip(rows, encoded):
                self._pending.append((row, tokens + [SPEC.EOT_TOKEN]))
            return

        if not self._pending:
            self._exhausted = True

    def take(self, n: int) -> list:
        """Exactly n tokens, or raise. Never a short read passed off as full."""
        out = []
        while len(out) < n:
            if not self._pending:
                self._read_batch()
                if self._exhausted:
                    raise CorpusTokenizeError(
                        f"THE CORPUS RAN OUT. {len(out)} of {n} tokens were "
                        f"available at file {self.file_index}. The cache holds "
                        f"{self.n_files} source files and that is not enough "
                        f"for {SPEC.TOTAL_TOKENS:,} tokens. Fetch more with "
                        f"scripts/corpus_fetch.py --files N and run again; "
                        f"everything already written is kept.")
            row, tokens = self._pending[0]
            available = len(tokens) - self.token_offset
            wanted = min(n - len(out), available)
            out.extend(tokens[self.token_offset:self.token_offset + wanted])
            self.token_offset += wanted
            if self.token_offset >= len(tokens):
                self._pending.popleft()
                self.row_index = row + 1
                self.token_offset = 0
        return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_block(path: Path, tokens: list) -> dict:
    """Write tokens as raw little-endian uint16, atomically.

    `.partial` first, flushed and fsynced, hashed, then renamed. The rename is
    the commit point: a file under its real name has been written whole and
    hashed, and a `.partial` left by a crash is discarded rather than resumed.
    """
    # Checks EVERY token, not the first and last. The sampled version claimed
    # to validate the block and examined two elements of it; a mid-array value
    # out of range fell through to int.to_bytes, which raises OverflowError
    # with a message about integers rather than about tokens.
    bad = next((t for t in tokens if t < 0 or t > 65535), None)
    if bad is not None:
        raise CorpusTokenizeError(
            f"token {bad} does not fit uint16 (0..65535); the vocabulary this "
            "corpus was tokenized with does not match the dtype the shards use")
    partial = path.with_suffix(path.suffix + ".partial")
    payload = b"".join(
        int(t).to_bytes(SPEC.BYTES_PER_TOKEN, SPEC.BYTE_ORDER) for t in tokens)
    with open(partial, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    digest = hashlib.sha256(payload).hexdigest()
    os.replace(partial, path)
    return {"bytes": len(payload), "sha256": digest, "tokens": len(tokens)}


def discard_partials(outdir: Path, stream) -> int:
    """A partial file is never resumed. Counted and removed."""
    removed = 0
    for path in sorted(outdir.glob("*.partial")):
        print(f"    discarding partial shard {path.name}", file=stream)
        path.unlink()
        removed += 1
    return removed


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def load_manifest(outdir: Path) -> dict:
    path = outdir / TOKENIZE_MANIFEST
    if not path.is_file():
        return {"spec": SPEC.summary(), "blocks": {}}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    recorded = manifest.get("spec", {})
    if recorded.get("revision") != SPEC.REVISION:
        raise CorpusTokenizeError(
            f"{path} was built from revision {recorded.get('revision')!r}, but "
            f"corpus_spec pins {SPEC.REVISION!r}. These are different corpora; "
            "resuming would splice one into the other.")
    if recorded.get("total", {}).get("tokens") != SPEC.TOTAL_TOKENS:
        raise CorpusTokenizeError(
            f"{path} was built to a total of "
            f"{recorded.get('total', {}).get('tokens')} tokens; corpus_spec "
            f"now says {SPEC.TOTAL_TOKENS}. The geometry changed after this "
            "corpus was started, so resuming would produce a file that matches "
            "neither.")
    return manifest


def save_manifest(outdir: Path, manifest: dict) -> None:
    path = outdir / TOKENIZE_MANIFEST
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(manifest, indent=2) + "\n",
                       encoding="utf-8", newline="\n")
    os.replace(partial, path)


def tokenizer_identity() -> dict:
    """Who tokenized this, and a probe proving they would do it again.

    A version string alone is weak evidence -- two builds can report the same
    version and behave differently, and a vendored tokenizer reports whatever
    it likes. So the identity carries a PROBE: the SHA-256 of the token ids
    produced from a fixed committed passage. A tokenizer that would produce a
    different corpus produces a different probe hash, whatever it calls itself.

    The probe is bursts/context.txt via metrics.load_context_batch, reused
    rather than re-derived so there is ONE definition of "did the tokenizer
    change" across step 10 and step 11.
    """
    import metrics

    batch = metrics.load_context_batch()
    identity = {
        "name": batch.tokenizer,
        "probe_source": batch.source,
        "probe_file_sha256": batch.file_sha256,
        "probe_token_sha256": batch.token_sha256,
        "probe_n_tokens": batch.n_tokens,
    }
    for package in ("transformers", "tokenizers"):
        try:
            identity[f"{package}_version"] = __import__(package).__version__
        except Exception:  # pragma: no cover - defensive
            identity[f"{package}_version"] = "unavailable"
    return identity


def check_tokenizer_identity(manifest: dict, identity: dict) -> None:
    """A corpus half-built by one tokenizer and half by another is not a corpus.

    Compared on the PROBE HASH, not on the version strings: a version bump that
    changes nothing about the output is not a reason to refuse, and a version
    that stayed the same while the output moved is exactly what must be caught.
    """
    recorded = manifest.get("tokenizer")
    if not recorded:
        return
    if recorded.get("probe_token_sha256") != identity["probe_token_sha256"]:
        raise CorpusTokenizeError(
            "TOKENIZER DRIFT. The shards already written were produced by a "
            "tokenizer that encodes the probe passage differently from this "
            "one.\n"
            f"  recorded probe: {recorded.get('probe_token_sha256')}\n"
            f"  this process:   {identity['probe_token_sha256']}\n"
            f"  recorded versions: transformers "
            f"{recorded.get('transformers_version')}, tokenizers "
            f"{recorded.get('tokenizers_version')}\n"
            f"  this process:      transformers "
            f"{identity.get('transformers_version')}, tokenizers "
            f"{identity.get('tokenizers_version')}\n"
            "Continuing would splice two different tokenizations into one "
            "corpus, and the result would be exactly the right size. Rebuild "
            "from scratch into a different --outdir, or restore the tokenizer "
            "version that started this one.")


def verify_block(outdir: Path, record: dict) -> bool:
    path = outdir / record["filename"]
    if not path.is_file():
        return False
    if path.stat().st_size != record["bytes"]:
        return False
    return sha256_file(path) == record["sha256"]


# ---------------------------------------------------------------------------
# The plan: what blocks exist, in order
# ---------------------------------------------------------------------------


def block_plan() -> list:
    """Every block to write, with its sequence count. Arithmetic, not a list."""
    plan = [{
        "key": "heldout",
        "filename": SPEC.HELDOUT_FILENAME,
        "sequences": SPEC.HELDOUT_SEQUENCES,
        "role": "held out -- no run trains on this",
    }]
    for index in range(SPEC.N_SHARDS):
        start, end = SPEC.shard_sequence_range(index)
        plan.append({
            "key": f"train-{index:03d}",
            "filename": SPEC.SHARD_TEMPLATE.format(index=index),
            "sequences": SPEC.SEQUENCES_PER_SHARD,
            "role": "training",
            "shard_index": index,
            "sequence_start": start,
            "sequence_end": end,
        })
    return plan


def tokenize(outdir: Path, cache: Path, n_blocks: int, stream) -> dict:
    """Write up to n_blocks blocks, resuming after whatever is already done."""
    outdir.mkdir(parents=True, exist_ok=True)
    discard_partials(outdir, stream)
    manifest = load_manifest(outdir)
    plan = block_plan()

    fetch_manifest = json.loads(
        (cache / FETCH_MANIFEST).read_text(encoding="utf-8"))
    n_files = len([k for k in fetch_manifest.get("files", {})])
    if not n_files:
        raise CorpusTokenizeError(
            f"no source files cached at {cache}. Run corpus_fetch.py first.")

    # Find the first block not already written AND verified. A recorded block
    # is re-verified rather than trusted: the manifest says what was true when
    # it was written.
    resume_at = 0
    for index, block in enumerate(plan):
        record = manifest["blocks"].get(block["key"])
        if record and verify_block(outdir, record):
            resume_at = index + 1
        else:
            break

    # Stamped BEFORE the early return, so a no-op run still records who
    # tokenized this and still catches drift against what is already written.
    identity = tokenizer_identity()
    check_tokenizer_identity(manifest, identity)
    manifest["tokenizer"] = identity
    save_manifest(outdir, manifest)

    if resume_at >= len(plan):
        print("  every block is present and verified; nothing to do",
              file=stream)
        return manifest
    if resume_at:
        print(f"  resuming at block {resume_at} of {len(plan)} "
              f"({plan[resume_at]['key']})", file=stream)

    start_position = (manifest["blocks"][plan[resume_at - 1]["key"]]
                      ["source_position_after"]) if resume_at else {
        "file_index": 0, "row_index": 0, "token_offset": 0}

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from burst_match import load_tokenizer
    tokenizer = load_tokenizer(stream=io.StringIO())

    source = TokenSource(cache, tokenizer, n_files, **start_position)
    written = 0
    started = time.perf_counter()

    for block in plan[resume_at:resume_at + n_blocks]:
        block_started = time.perf_counter()
        position_before = source.position()
        tokens = source.take(block["sequences"] * SPEC.SEQ_LEN)
        record = write_block(outdir / block["filename"], tokens)
        record.update({
            "filename": block["filename"],
            "sequences": block["sequences"],
            "role": block["role"],
            "source_position_before": position_before,
            "source_position_after": source.position(),
        })
        if "shard_index" in block:
            record["shard_index"] = block["shard_index"]
            record["sequence_start"] = block["sequence_start"]
            record["sequence_end"] = block["sequence_end"]
        manifest["blocks"][block["key"]] = record
        save_manifest(outdir, manifest)
        written += 1
        print(f"    {block['key']:<12} {record['tokens']:>12,} tok  "
              f"{record['bytes']:>11,} B  sha {record['sha256'][:12]}  "
              f"{time.perf_counter() - block_started:5.1f}s",
              file=stream, flush=True)

    manifest["elapsed_seconds_last_run"] = round(
        time.perf_counter() - started, 2)
    manifest["blocks_written_last_run"] = written
    manifest["blocks_complete"] = len(manifest["blocks"])
    manifest["blocks_total"] = len(plan)
    save_manifest(outdir, manifest)
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/corpus_tokenize.py",
        description="Tokenize cached source files into held-out + 149 shards.")
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                        help="where shards land. A command-line argument and "
                             "never a config value.")
    parser.add_argument("--cachedir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--blocks", type=int, default=SPEC.N_SHARDS + 1,
        help=("how many blocks to write this run. One shard measured at 7.2s, "
              "so 50 is about 360s -- comfortably inside the ~600s cap that "
              "killed four runs in step 9."))
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    stream = sys.stdout
    rule = "=" * 78
    print(rule, file=stream)
    print("corpus_tokenize -- step 11 stage 2", file=stream)
    print(rule, file=stream)
    print(f"source cache: {args.cachedir}", file=stream)
    print(f"outdir:       {args.outdir}", file=stream)
    print(f"target:       {SPEC.TOTAL_TOKENS:,} tokens "
          f"({SPEC.HELDOUT_SEQUENCES:,} held-out + "
          f"{SPEC.TRAIN_SEQUENCES:,} training sequences)\n", file=stream)

    try:
        manifest = tokenize(args.outdir, args.cachedir, args.blocks, stream)
    except CorpusTokenizeError as exc:
        print(f"\nFAILED: {exc}", file=stream)
        return 1

    done = manifest.get("blocks_complete", 0)
    total = manifest.get("blocks_total", len(block_plan()))
    print(f"\n{done} of {total} blocks complete", file=stream)
    if done < total:
        print(f"run again to continue from block {done}", file=stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
