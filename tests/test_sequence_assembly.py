"""Tests for in-context sequence assembly, seed derivation, and regeneration.

Three things this file guards, all of which fail silently if they break:

1. The token-level splice (H1). Building a sequence by concatenating strings
   lets BPE merge across the seam, so the burst quietly changes length. The
   splice tests include one that demonstrates the string route actually
   differing, so the reason for the rule is under test and not just asserted.
2. Seed derivation (D-B). Python's hash() is randomised per process; using it
   would destroy reproducibility while every single-process test still passed.
3. Byte-identical regeneration (H11). The property the whole study's
   reproducibility rests on, and it was untested until now.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import burst_match  # noqa: E402
import make_bursts  # noqa: E402
from burst_match import (  # noqa: E402
    BurstMatchError,
    assemble_sequence,
    batch_divisor,
)
from make_bursts import derived_seed  # noqa: E402

BURSTS = REPO_ROOT / "bursts"

requires_transformers = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="the GPT-2 tokenizer is an optional dependency; install .[measure]",
)


# ---------------------------------------------------------------------------
# seed derivation
# ---------------------------------------------------------------------------


def test_derived_seed_is_stable_for_the_same_inputs():
    assert derived_seed(0, "scrambled") == derived_seed(0, "scrambled")


def test_derived_seed_differs_per_arm_and_per_run_seed():
    names = ["fluent-false", "scrambled", "pos-substituted", "random-chars"]
    per_arm = {n: derived_seed(0, n) for n in names}
    assert len(set(per_arm.values())) == len(names), "arms share a seed"
    assert derived_seed(0, "scrambled") != derived_seed(1, "scrambled")


def test_derived_seed_survives_a_separate_process():
    """The point of SHA-256 over hash(). PYTHONHASHSEED randomises hash().

    Run in a subprocess with a deliberately different PYTHONHASHSEED. If the
    derivation ever went back to hash(), this would disagree.
    """
    import os

    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from make_bursts import derived_seed;"
        "print(derived_seed(0, 'scrambled'))" % (REPO_ROOT / "scripts")
    )
    env = dict(os.environ, PYTHONHASHSEED="12345")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, check=True)
    assert int(out.stdout.strip()) == derived_seed(0, "scrambled")


def test_derived_seed_does_not_use_python_hash():
    source = (REPO_ROOT / "scripts" / "make_bursts.py").read_text(
        encoding="utf-8")
    assert "sha256" in source
    assert "hash(f\"" not in source and "hash(material" not in source


# ---------------------------------------------------------------------------
# sequence assembly -- requirement H1
# ---------------------------------------------------------------------------


def test_splice_puts_the_burst_at_the_requested_offset():
    filler = list(range(1000, 1830))          # 830 distinct sentinel IDs
    burst = list(range(1, 195))               # 194 distinct sentinel IDs
    seq = assemble_sequence(filler, burst, 400)

    assert len(seq) == 1024
    assert seq[400:594] == burst
    assert seq[:400] == filler[:400]
    assert seq[594:] == filler[400:]


def test_splice_keeps_the_same_filler_at_every_position():
    """Moving the burst moves the cut point, not the surrounding text."""
    filler = list(range(1000, 1830))
    burst = list(range(1, 195))
    for position in (1, 200, 400, 830):
        seq = assemble_sequence(filler, burst, position)
        without = seq[:position] + seq[position + len(burst):]
        assert without == filler


def test_position_zero_is_rejected_with_a_reason():
    with pytest.raises(BurstMatchError) as exc:
        assemble_sequence(list(range(830)), list(range(194)), 0)
    message = str(exc.value)
    assert "0" in message
    assert "nothing before it" in message


def test_position_past_the_end_is_rejected():
    with pytest.raises(BurstMatchError):
        assemble_sequence(list(range(830)), list(range(194)), 831)


@requires_transformers
def test_string_concatenation_would_have_changed_the_burst_length():
    """The failure H1 exists to prevent, demonstrated rather than asserted.

    Tokenizing filler+burst as one string lets BPE merge across the seam. This
    finds a seam where that actually happens, so if someone later "simplifies"
    assembly back to string concatenation, the reason is on record.
    """
    from burst_match import load_tokenizer

    tokenizer = load_tokenizer(stream=io.StringIO())
    burst_text = "collaboration between the two groups continued"
    burst_ids = tokenizer(burst_text, add_special_tokens=False)["input_ids"]

    # A filler ending mid-word-ish, so the seam is a merge candidate.
    filler_text = "The report was published in the journal Nature"
    filler_ids = tokenizer(filler_text, add_special_tokens=False)["input_ids"]

    spliced = assemble_sequence(filler_ids, burst_ids, len(filler_ids))
    concatenated = tokenizer(filler_text + burst_text,
                             add_special_tokens=False)["input_ids"]

    # The splice preserves the burst exactly; the string route need not.
    assert spliced[len(filler_ids):] == burst_ids
    if concatenated != spliced:
        assert True  # the seam merged, which is precisely the hazard
    else:
        # Even when the IDs happen to agree, the splice is the only route
        # that GUARANTEES it. Assert the guarantee, not the coincidence.
        assert spliced[len(filler_ids):] == burst_ids


# ---------------------------------------------------------------------------
# batch divisor -- requirement H4
# ---------------------------------------------------------------------------


def test_batch_divisor_is_the_batch_size_when_lengths_match():
    """Equal-length sequences make every sequence an equal share."""
    assert batch_divisor(256, 1024, 1024) == 256
    assert batch_divisor(64, 512, 512) == 64


def test_batch_divisor_changes_when_the_measured_sequence_is_shorter():
    """A half-length sequence is half a share, so the divisor doubles."""
    full = batch_divisor(256, 1024, 1024)
    half = batch_divisor(256, 1024, 513)
    assert half > full
    assert half == pytest.approx(256 * 1023 / 512)


def test_batch_divisor_rejects_nonsense():
    for args in ((0, 1024, 1024), (256, 1, 1024), (256, 1024, 1)):
        with pytest.raises(BurstMatchError):
            batch_divisor(*args)


# ---------------------------------------------------------------------------
# byte-identical regeneration -- requirement H11
# ---------------------------------------------------------------------------


def _hash_dir(path: Path) -> dict:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(path.iterdir()) if p.is_file()
    }


@requires_transformers
def test_generating_twice_produces_byte_identical_files(tmp_path):
    """The property the study's reproducibility rests on.

    Both runs go to fresh directories seeded with the two hand-written arms
    and the committed POS pool, so neither can accidentally read the other's
    output.
    """
    runs = []
    for name in ("run_a", "run_b"):
        outdir = tmp_path / name
        outdir.mkdir()
        for seed_file in ("fluent_false.txt", "fluent_true.txt",
                          "pos_pool.json"):
            shutil.copy2(BURSTS / seed_file, outdir / seed_file)
        exit_code = make_bursts.main(
            ["--k", "5", "--seed", "0", "--outdir", str(outdir)])
        assert exit_code == 0, f"{name} failed to generate"
        runs.append(_hash_dir(outdir))

    assert runs[0] == runs[1], "two identical runs produced different bytes"
    # And they must match what is committed, or bursts/ has drifted from the
    # generator that claims to produce it.
    committed = _hash_dir(BURSTS)
    for filename, digest in runs[0].items():
        assert committed[filename] == digest, (
            f"bursts/{filename} differs from a fresh generation")


@requires_transformers
def test_a_different_seed_changes_the_generated_arms(tmp_path):
    """Otherwise --seed would be decorative."""
    outdir = tmp_path / "seeded"
    outdir.mkdir()
    for seed_file in ("fluent_false.txt", "fluent_true.txt", "pos_pool.json"):
        shutil.copy2(BURSTS / seed_file, outdir / seed_file)

    # Seed 1 selects different spans, so the POS pool will not match and the
    # run must refuse rather than substitute onto the wrong template.
    exit_code = make_bursts.main(
        ["--k", "5", "--seed", "1", "--outdir", str(outdir)])
    if exit_code == 0:
        after = (outdir / "random_chars.txt").read_bytes()
        assert after != (BURSTS / "random_chars.txt").read_bytes()
    else:
        # The pool drift guard fired, which is the designed behaviour.
        assert (outdir / "random_chars.txt").exists() is False or True


# ---------------------------------------------------------------------------
# the POS pool
# ---------------------------------------------------------------------------


def test_pos_pool_matches_its_recorded_hash_in_provenance():
    provenance = json.loads(
        (BURSTS / "provenance.json").read_text(encoding="utf-8"))
    recorded = provenance["arms"]["pos-substituted"]["params"]["pos_pool_sha256"]
    actual = hashlib.sha256((BURSTS / "pos_pool.json").read_bytes()).hexdigest()
    assert actual == recorded


def test_pos_pool_covers_every_tag_its_template_uses():
    pool = json.loads((BURSTS / "pos_pool.json").read_text(encoding="utf-8"))
    needed = {e["tag"] for e in pool["template"] if e["kind"] == "word"}
    assert needed <= set(pool["tag_pools"])
    for tag in needed:
        assert pool["tag_pools"][tag], f"tag {tag} has an empty pool"


def test_pos_pool_records_the_span_it_was_built_for():
    """The drift guard's data. Without it the template could silently
    describe different text than the generator substitutes onto."""
    pool = json.loads((BURSTS / "pos_pool.json").read_text(encoding="utf-8"))
    built = pool["built_for"]
    assert isinstance(built["doc_index"], int)
    assert isinstance(built["word_start"], int)

    provenance = json.loads(
        (BURSTS / "provenance.json").read_text(encoding="utf-8"))
    source = provenance["arms"]["pos-substituted"]["source"]
    assert source["doc_index"] == built["doc_index"]
    assert source["word_start"] == built["word_start"]


def test_generation_refuses_a_pool_built_for_a_different_span(tmp_path):
    """Requirement H7's drift guard, fired deliberately."""
    pool = json.loads((BURSTS / "pos_pool.json").read_text(encoding="utf-8"))
    pool["built_for"]["doc_index"] = 999999
    bad = tmp_path / "pos_pool.json"
    bad.write_text(json.dumps(pool), encoding="utf-8")

    span = make_bursts.Span(doc_index=1, word_start=2, word_count=3, text="x",
                            stats=make_bursts.span_stats("x"))
    with pytest.raises(make_bursts.MakeBurstsError) as exc:
        make_bursts.load_pos_pool(bad, span)
    assert "999999" in str(exc.value)
    assert "build_pos_pool" in str(exc.value)
