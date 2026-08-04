"""The verifier: it must FAIL on things, and it must not need the corpus stack.

THE RISK THIS FILE EXISTS FOR.

A verifier that passes is worthless unless it is known to fail. Every test here
breaks something specific and asserts the break is found -- corruption, a
truncated file, a manifest that disagrees with its own data, a shard boundary
that does not match arithmetic, a short training slice, and a tokenizer that
would encode the corpus differently.

Two properties are structural rather than behavioural, and are asserted by
reading the module: it must not import `datasets` or `huggingface_hub`, because
it has to run on a cluster with no network and no corpus stack; and the
tokenizer probe must degrade to a LOUD skip rather than a silent pass, since a
verification that quietly omits its only tokenizer check reads as a success.
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


V = _load("corpus_verify")
SPEC = _load("corpus_spec")
T = _load("corpus_tokenize")


def _tiny_corpus(tmp_path, n_shards=2, sequences=4):
    """A miniature corpus with a real manifest, for breaking on purpose."""
    blocks = {}
    tokens_per_block = sequences * SPEC.SEQ_LEN
    record = T.write_block(tmp_path / SPEC.HELDOUT_FILENAME,
                           list(range(tokens_per_block)))
    record.update({"filename": SPEC.HELDOUT_FILENAME, "sequences": sequences,
                   "role": "held out -- no run trains on this"})
    blocks["heldout"] = record
    for i in range(n_shards):
        name = SPEC.SHARD_TEMPLATE.format(index=i)
        rec = T.write_block(tmp_path / name, list(range(tokens_per_block)))
        start, end = SPEC.shard_sequence_range(i)
        rec.update({"filename": name, "sequences": sequences,
                    "role": "training", "shard_index": i,
                    "sequence_start": start, "sequence_end": end})
        blocks[f"train-{i:03d}"] = rec
    manifest = {"spec": SPEC.summary(), "blocks": blocks,
                "tokenizer": {"probe_token_sha256": "deadbeef",
                              "transformers_version": "x"}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                            encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# It fails on the things it is for
# ---------------------------------------------------------------------------


def test_a_clean_partial_corpus_reports_no_structural_findings(tmp_path):
    _tiny_corpus(tmp_path)
    report = V.verify(tmp_path, sys.stdout)
    structural = [f for f in report["findings"] if "TOKENIZER DRIFT" not in f]
    assert not structural, structural
    assert report["complete"] is False


def test_corruption_is_caught_by_the_hash(tmp_path):
    _tiny_corpus(tmp_path)
    path = tmp_path / SPEC.SHARD_TEMPLATE.format(index=0)
    data = bytearray(path.read_bytes())
    data[100] ^= 0xFF          # same size, one flipped byte
    path.write_bytes(bytes(data))
    report = V.verify(tmp_path, sys.stdout)
    assert any("sha256 does not match" in f for f in report["findings"])


def test_truncation_is_caught_by_size_arithmetic(tmp_path):
    _tiny_corpus(tmp_path)
    path = tmp_path / SPEC.SHARD_TEMPLATE.format(index=0)
    path.write_bytes(path.read_bytes()[:-1024])
    report = V.verify(tmp_path, sys.stdout)
    assert any("by size" in f or "sha256" in f for f in report["findings"])


def test_an_odd_byte_count_is_reported_as_a_mid_token_truncation(tmp_path):
    _tiny_corpus(tmp_path)
    path = tmp_path / SPEC.SHARD_TEMPLATE.format(index=0)
    path.write_bytes(path.read_bytes()[:-1])
    report = V.verify(tmp_path, sys.stdout)
    assert any("truncated mid-token" in f for f in report["findings"])


def test_a_missing_block_is_reported(tmp_path):
    _tiny_corpus(tmp_path)
    (tmp_path / SPEC.SHARD_TEMPLATE.format(index=1)).unlink()
    report = V.verify(tmp_path, sys.stdout)
    assert any("missing" in f for f in report["findings"])


def test_a_manifest_disagreeing_with_its_own_data_is_caught(tmp_path):
    """The manifest is a claim; the file is the fact."""
    manifest = _tiny_corpus(tmp_path)
    manifest["blocks"]["train-000"]["tokens"] = 999
    (tmp_path / "manifest.json").write_text(json.dumps(manifest),
                                            encoding="utf-8")
    report = V.verify(tmp_path, sys.stdout)
    assert any("by size" in f or "sequences imply" in f
               for f in report["findings"])


def test_a_boundary_that_disagrees_with_arithmetic_is_caught(tmp_path):
    manifest = _tiny_corpus(tmp_path)
    manifest["blocks"]["train-001"]["sequence_start"] = 12345
    (tmp_path / "manifest.json").write_text(json.dumps(manifest),
                                            encoding="utf-8")
    report = V.verify(tmp_path, sys.stdout)
    assert any("arithmetic gives" in f for f in report["findings"])


def test_a_short_training_slice_names_the_loader_assertion(tmp_path):
    """The failure that would surface 40 runs later as a config error."""
    manifest = _tiny_corpus(tmp_path)
    for i in range(SPEC.N_SHARDS):
        manifest["blocks"].setdefault(f"train-{i:03d}", {
            "filename": f"absent-{i}.bin", "tokens": 1, "sequences": 1,
            "role": "training", "bytes": 2, "sha256": "x"})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest),
                                            encoding="utf-8")
    report = V.verify(tmp_path, sys.stdout)
    assert report["complete"] is True
    assert any("expected_token_budget" in f for f in report["findings"])


# ---------------------------------------------------------------------------
# The manifest's own hash
# ---------------------------------------------------------------------------


def test_the_manifest_hash_is_reported_and_is_the_file_hash(tmp_path):
    """Carried separately, or a corrupt manifest validates corrupt shards."""
    _tiny_corpus(tmp_path)
    raw = (tmp_path / "manifest.json").read_bytes()
    report = V.verify(tmp_path, sys.stdout)
    assert report["manifest_sha256"] == hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# Structural: what it may depend on, and how it degrades
# ---------------------------------------------------------------------------


def test_verifier_does_not_import_the_corpus_stack():
    """It has to run on a cluster with no network and no datasets install."""
    source = (REPO_ROOT / "scripts" / "corpus_verify.py").read_text(
        encoding="utf-8")
    for banned in ("import datasets", "from datasets",
                   "import huggingface_hub", "from huggingface_hub"):
        assert banned not in source


def test_a_missing_tokenizer_probe_is_a_loud_skip_not_a_silent_pass(tmp_path):
    """A verification that quietly omits its only tokenizer check reads as OK."""
    manifest = _tiny_corpus(tmp_path)
    findings = []
    import builtins

    real_import = builtins.__import__

    def no_metrics(name, *args, **kwargs):
        if name == "metrics":
            raise ImportError("transformers unavailable")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = no_metrics
    try:
        result = V.check_tokenizer_probe(manifest, findings)
    finally:
        builtins.__import__ = real_import

    assert result["probe"] == "NOT RUN"
    assert "WARNING" in result
    assert "did NOT run" in result["WARNING"]


def test_a_manifest_with_no_tokenizer_identity_is_a_finding(tmp_path):
    findings = []
    V.check_tokenizer_probe({"blocks": {}}, findings)
    assert any("no tokenizer identity" in f for f in findings)


def test_drift_at_the_destination_is_reported_as_such(tmp_path):
    """The one check that catches a divergent tokenizer on Asa's machine."""
    pytest.importorskip("transformers")
    findings = []
    V.check_tokenizer_probe(
        {"tokenizer": {"probe_token_sha256": "0" * 64}}, findings)
    assert any("TOKENIZER DRIFT AT THIS MACHINE" in f for f in findings)
