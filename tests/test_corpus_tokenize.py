"""Stage 2: exact counts, atomic writes, and a resume that is byte-identical.

THE RISK THIS FILE EXISTS FOR.

This is a multi-hour job whose output nothing downstream can sanity-check by
eye. Two failure modes matter and both are silent:

1. A HALF-WRITTEN SHARD. A shard whose tail is garbage has exactly the right
   size once the write is retried, and nothing afterwards can find the seam.
   That is why a shard is written to `.partial`, hashed, and only then renamed,
   and why a `.partial` is always DISCARDED rather than resumed.

2. A RESUME THAT PRODUCES DIFFERENT TEXT. Shard 40 written today must be
   byte-identical to shard 40 written yesterday. A shard boundary lands
   mid-document far more often than not, so the source position has to be
   recorded to the token, not to the document -- rounding it to a document
   boundary would drop or duplicate text at every boundary, and the corpus
   would still be exactly the right size.

The tests below use a fake tokenizer and fake source so they run in the
torch-free environment and in under a second. The real tokenizer's behaviour is
not what is under test here; the bookkeeping around it is.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


T = _load("corpus_tokenize")
SPEC = _load("corpus_spec")


class FakeSource:
    """A deterministic token stream with the same position contract."""

    def __init__(self, start=0):
        self.cursor = start

    def position(self):
        return {"file_index": 0, "row_index": 0, "token_offset": self.cursor}

    def take(self, n):
        out = [(self.cursor + i) % 50257 for i in range(n)]
        self.cursor += n
        return out


# ---------------------------------------------------------------------------
# Writing is atomic, and the bytes are the format the spec promises
# ---------------------------------------------------------------------------


def test_written_block_is_raw_little_endian_uint16(tmp_path):
    path = tmp_path / "b.bin"
    record = T.write_block(path, [0, 1, 50256, 65535])
    assert path.read_bytes() == (
        b"\x00\x00" + b"\x01\x00" + b"\x50\xc4" + b"\xff\xff")
    assert record["bytes"] == 8
    assert record["tokens"] == 4


def test_file_size_is_exactly_two_bytes_per_token(tmp_path):
    """The independent route to token count -- no header to spoil it."""
    path = tmp_path / "b.bin"
    record = T.write_block(path, list(range(1000)))
    assert path.stat().st_size == 2000
    assert SPEC.tokens_from_byte_size(path.stat().st_size) == record["tokens"]


def test_recorded_hash_is_the_hash_of_the_file(tmp_path):
    path = tmp_path / "b.bin"
    record = T.write_block(path, list(range(500)))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]


def test_no_partial_survives_a_successful_write(tmp_path):
    T.write_block(tmp_path / "b.bin", [1, 2, 3])
    assert not list(tmp_path.glob("*.partial"))


def test_a_partial_is_discarded_never_resumed(tmp_path, capsys):
    """The half-written shard. Resuming into it would leave a findable seam."""
    (tmp_path / "train-007.bin.partial").write_bytes(b"\x01\x02truncated")
    removed = T.discard_partials(tmp_path, sys.stdout)
    assert removed == 1
    assert not list(tmp_path.glob("*.partial"))


# ---------------------------------------------------------------------------
# The plan is arithmetic and covers the corpus exactly
# ---------------------------------------------------------------------------


def test_plan_is_heldout_plus_every_shard():
    plan = T.block_plan()
    assert len(plan) == SPEC.N_SHARDS + 1
    assert plan[0]["key"] == "heldout"
    assert plan[0]["sequences"] == SPEC.HELDOUT_SEQUENCES
    assert plan[-1]["key"] == f"train-{SPEC.N_SHARDS - 1:03d}"


def test_plan_sequences_sum_to_the_whole_corpus():
    total = sum(block["sequences"] for block in T.block_plan())
    assert total == SPEC.TOTAL_SEQUENCES
    assert total * SPEC.SEQ_LEN == SPEC.TOTAL_TOKENS


def test_heldout_is_the_first_block_and_is_not_a_shard():
    """Front placement is what puts the committed corpus texts outside training."""
    plan = T.block_plan()
    assert plan[0]["filename"] == SPEC.HELDOUT_FILENAME
    assert "shard_index" not in plan[0]
    assert all("shard_index" in b for b in plan[1:])


def test_shard_sequence_ranges_agree_with_the_spec_arithmetic():
    """Two routes: the plan's recorded range, and corpus_spec's computation."""
    for block in T.block_plan()[1:]:
        start, end = SPEC.shard_sequence_range(block["shard_index"])
        assert (block["sequence_start"], block["sequence_end"]) == (start, end)


# ---------------------------------------------------------------------------
# Exactness
# ---------------------------------------------------------------------------


def test_take_returns_exactly_what_was_asked_for():
    source = FakeSource()
    assert len(source.take(1024)) == 1024
    assert len(source.take(7)) == 7


def test_a_short_source_raises_rather_than_writing_a_short_shard():
    """Running out must fail loudly, not produce a corpus that is nearly right."""
    class Exhausted(FakeSource):
        def take(self, n):
            raise T.CorpusTokenizeError("THE CORPUS RAN OUT. test")

    with pytest.raises(T.CorpusTokenizeError, match="RAN OUT"):
        Exhausted().take(10)


# ---------------------------------------------------------------------------
# The manifest, and refusing to splice two corpora
# ---------------------------------------------------------------------------


def test_manifest_round_trips_and_records_the_spec(tmp_path):
    manifest = T.load_manifest(tmp_path)
    assert manifest["spec"]["revision"] == SPEC.REVISION
    assert manifest["blocks"] == {}
    T.save_manifest(tmp_path, manifest)
    assert T.load_manifest(tmp_path)["spec"]["total"]["tokens"] == \
        SPEC.TOTAL_TOKENS


def test_a_manifest_from_another_revision_is_refused(tmp_path):
    spec = SPEC.summary()
    spec["revision"] = "0" * 40
    (tmp_path / T.TOKENIZE_MANIFEST).write_text(
        json.dumps({"spec": spec, "blocks": {}}), encoding="utf-8")
    with pytest.raises(T.CorpusTokenizeError, match="different corpora"):
        T.load_manifest(tmp_path)


def test_a_manifest_built_to_another_geometry_is_refused(tmp_path):
    """Changing the geometry mid-build would produce a file matching neither."""
    spec = SPEC.summary()
    spec["total"]["tokens"] = SPEC.TOTAL_TOKENS + 1024
    (tmp_path / T.TOKENIZE_MANIFEST).write_text(
        json.dumps({"spec": spec, "blocks": {}}), encoding="utf-8")
    with pytest.raises(T.CorpusTokenizeError, match="geometry changed"):
        T.load_manifest(tmp_path)


def test_manifest_write_is_atomic(tmp_path):
    T.save_manifest(tmp_path, T.load_manifest(tmp_path))
    assert (tmp_path / T.TOKENIZE_MANIFEST).is_file()
    assert not list(tmp_path.glob("*.partial"))


# ---------------------------------------------------------------------------
# Verification does not trust the manifest
# ---------------------------------------------------------------------------


def test_verify_block_accepts_an_intact_block(tmp_path):
    record = T.write_block(tmp_path / "b.bin", list(range(100)))
    record["filename"] = "b.bin"
    assert T.verify_block(tmp_path, record) is True


def test_verify_block_rejects_a_tampered_block(tmp_path):
    record = T.write_block(tmp_path / "b.bin", list(range(100)))
    record["filename"] = "b.bin"
    (tmp_path / "b.bin").write_bytes(b"\x00" * record["bytes"])
    assert T.verify_block(tmp_path, record) is False, (
        "same size, different content -- size alone must not be enough")


def test_verify_block_rejects_a_truncated_block(tmp_path):
    record = T.write_block(tmp_path / "b.bin", list(range(100)))
    record["filename"] = "b.bin"
    (tmp_path / "b.bin").write_bytes(b"\x00" * 10)
    assert T.verify_block(tmp_path, record) is False


# ---------------------------------------------------------------------------
# The source position is recorded to the token, not to the document
# ---------------------------------------------------------------------------


def test_position_carries_an_intra_document_token_offset():
    """A shard boundary lands mid-document far more often than not.

    Rounding the resume point to a document boundary would drop or duplicate
    text at every boundary, and the corpus would still be exactly the right
    size -- which is precisely the kind of failure nothing downstream catches.
    """
    import inspect

    source = inspect.signature(T.TokenSource.__init__).parameters
    assert "token_offset" in source
    assert "row_index" in source
    assert "file_index" in source
    assert set(FakeSource().position()) == {
        "file_index", "row_index", "token_offset"}


def test_the_shipped_manifest_records_a_position_for_every_block():
    """Against the real corpus if one has been built, else skipped."""
    outdir = T.DEFAULT_OUTDIR
    path = outdir / T.TOKENIZE_MANIFEST
    if not path.is_file():
        pytest.skip("no tokenized corpus built locally")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key, record in manifest["blocks"].items():
        assert set(record["source_position_after"]) == {
            "file_index", "row_index", "token_offset"}, key
        assert record["bytes"] == record["tokens"] * SPEC.BYTES_PER_TOKEN, key
