"""Stage 1: resumable, pinned, and never trusting its own manifest.

THE RISK THIS FILE EXISTS FOR.

The fetch stage is the only thing that decides WHAT CORPUS this study trains
on, and every failure it can have is quiet. A truncated download is a valid
Parquet prefix. A cache half-built from one revision and half from another is a
directory of real files. A manifest that records a file which has since changed
on disk describes a corpus that no longer exists. None of these raise on their
own, and all of them produce a tokenized corpus that looks completely normal.

So the tests below are mostly about refusing to trust things:

  - a manifest entry proves what was true when it was written, not what is on
    disk now, so verification re-reads
  - a cache from another revision is a different corpus, not a partial one
  - the manifest is written atomically, because the manifest is the only thing
    that says which files are verified

Nothing here touches the network. The hub call is the one part that cannot be
tested without it, and it is kept to a single function for that reason.
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


F = _load("corpus_fetch")
SPEC = _load("corpus_spec")


def _write(path: Path, data: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "index": 0,
        "filename": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Hashing and verification
# ---------------------------------------------------------------------------


def test_sha256_file_matches_hashing_the_bytes(tmp_path):
    data = b"openwebtext" * 1000
    path = tmp_path / "f.bin"
    path.write_bytes(data)
    assert F.sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_sha256_is_blockwise_and_handles_a_file_over_one_block(tmp_path):
    """A 500 MB Parquet file must never land in RAM whole."""
    data = b"x" * (F.HASH_BLOCK + 12345)
    path = tmp_path / "big.bin"
    path.write_bytes(data)
    assert F.sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_verification_accepts_an_intact_file(tmp_path):
    record = _write(tmp_path / "a.parquet", b"content")
    assert F.verify_cached_file(tmp_path, record) is True


def test_verification_rejects_a_truncated_file(tmp_path):
    """The common download failure, and a valid Parquet prefix."""
    record = _write(tmp_path / "a.parquet", b"content")
    (tmp_path / "a.parquet").write_bytes(b"cont")
    assert F.verify_cached_file(tmp_path, record) is False


def test_verification_rejects_a_same_size_different_content_file(tmp_path):
    """Size alone is not enough -- this is why the hash is also checked."""
    record = _write(tmp_path / "a.parquet", b"content")
    (tmp_path / "a.parquet").write_bytes(b"CONTENT")
    assert F.verify_cached_file(tmp_path, record) is False


def test_verification_rejects_a_missing_file(tmp_path):
    record = _write(tmp_path / "a.parquet", b"content")
    (tmp_path / "a.parquet").unlink()
    assert F.verify_cached_file(tmp_path, record) is False


def test_a_manifest_entry_is_not_taken_as_proof(tmp_path):
    """A record proves what was true when written, not what is on disk now."""
    record = _write(tmp_path / "a.parquet", b"content")
    assert F.verify_cached_file(tmp_path, record) is True
    (tmp_path / "a.parquet").write_bytes(b"tampered!")
    assert F.verify_cached_file(tmp_path, record) is False


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def test_manifest_round_trips(tmp_path):
    manifest = F.load_manifest(tmp_path)
    assert manifest["files"] == {}
    assert manifest["revision"] == SPEC.REVISION
    manifest["files"]["0"] = {"bytes": 1, "sha256": "x", "filename": "f"}
    F.save_manifest(tmp_path, manifest)
    assert F.load_manifest(tmp_path)["files"]["0"]["sha256"] == "x"


def test_manifest_write_is_atomic(tmp_path):
    """No .partial left behind, and the real file appears whole."""
    manifest = F.load_manifest(tmp_path)
    F.save_manifest(tmp_path, manifest)
    assert (tmp_path / F.FETCH_MANIFEST).is_file()
    assert not list(tmp_path.glob("*.partial"))


def test_a_cache_from_another_revision_is_refused_not_merged(tmp_path):
    """Two snapshots in one directory is the failure pinning exists to stop."""
    (tmp_path / F.FETCH_MANIFEST).write_text(json.dumps({
        "dataset": SPEC.DATASET,
        "revision": "0" * 40,
        "files": {},
    }), encoding="utf-8")
    with pytest.raises(F.CorpusFetchError, match="different snapshots"):
        F.load_manifest(tmp_path)


def test_revision_mismatch_names_both_revisions(tmp_path):
    (tmp_path / F.FETCH_MANIFEST).write_text(json.dumps({
        "revision": "0" * 40, "files": {}}), encoding="utf-8")
    with pytest.raises(F.CorpusFetchError) as exc:
        F.load_manifest(tmp_path)
    assert "0" * 40 in str(exc.value)
    assert SPEC.REVISION in str(exc.value)


# ---------------------------------------------------------------------------
# Resume and range checking
# ---------------------------------------------------------------------------


def test_verify_only_passes_on_a_complete_cache(tmp_path, capsys):
    """Resume's precondition: a verified cache needs no network."""
    manifest = F.load_manifest(tmp_path)
    for i in range(2):
        name = SPEC.SOURCE_TEMPLATE.format(index=i)
        data = f"file{i}".encode()
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest["files"][str(i)] = {
            "index": i, "filename": name, "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}
    F.save_manifest(tmp_path, manifest)
    assert F.fetch(2, tmp_path, sys.stdout, verify_only=True)


def test_verify_only_refuses_a_corrupt_cache_without_fetching(tmp_path):
    manifest = F.load_manifest(tmp_path)
    name = SPEC.SOURCE_TEMPLATE.format(index=0)
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"real")
    manifest["files"]["0"] = {
        "index": 0, "filename": name, "bytes": 4, "sha256": "0" * 64}
    F.save_manifest(tmp_path, manifest)
    with pytest.raises(F.CorpusFetchError, match="missing or does not match"):
        F.fetch(1, tmp_path, sys.stdout, verify_only=True)


@pytest.mark.parametrize("n", [0, -1, SPEC.N_SOURCE_FILES + 1])
def test_out_of_range_file_counts_are_refused(tmp_path, n):
    with pytest.raises(F.CorpusFetchError, match="n_files must be"):
        F.fetch(n, tmp_path, sys.stdout)


# ---------------------------------------------------------------------------
# Pinning and ordering
# ---------------------------------------------------------------------------


def test_source_order_comes_from_arithmetic_not_a_directory_listing():
    """Filesystem enumeration order must never reach the study's data order."""
    source = (REPO_ROOT / "scripts" / "corpus_fetch.py").read_text(
        encoding="utf-8")
    for banned in ("iterdir()", "listdir(", "glob(\"*.parquet\")"):
        assert banned not in source
    assert "SOURCE_TEMPLATE.format" in source


def test_every_fetch_names_the_pinned_revision():
    source = (REPO_ROOT / "scripts" / "corpus_fetch.py").read_text(
        encoding="utf-8")
    assert "revision=SPEC.REVISION" in source


def test_default_cache_is_gitignored():
    """CLAUDE.md rule 3: corpus data is never committed, at any size."""
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "-v",
         str(F.DEFAULT_CACHE.relative_to(REPO_ROOT) / "x.parquet")],
        capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, (
        f"{F.DEFAULT_CACHE} is NOT gitignored; corpus data could be committed")
