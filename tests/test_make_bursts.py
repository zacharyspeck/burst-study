"""Tests for scripts/make_bursts.py and the burst files it produced.

Split the same way as tests/test_burst_match.py. The window shuffle, the span
quality filters and the mid-word detection are plain Python and run in the
torch-free environment. Anything that has to count tokens needs the GPT-2
tokenizer from `transformers`, and skips cleanly without it.

The last group is different in kind: it checks the files actually committed
under bursts/, not the code that made them. Those are the acceptance criteria
for this step, turned into something that keeps being checked.
"""

from __future__ import annotations

import importlib.util
import io
import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import make_bursts  # noqa: E402
from make_bursts import (  # noqa: E402
    MIN_SENTENCE_ENDS,
    MakeBurstsError,
    cut_is_mid_word,
    span_stats,
    window_shuffle,
)

BURSTS = REPO_ROOT / "bursts"

requires_transformers = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="the GPT-2 tokenizer is an optional dependency; install .[measure]",
)

PROSE = (
    "The lighthouse keeper wrote the same sentence every evening. The sea did "
    "not change, but the handwriting did. He recorded the weather twice a day."
)


# ---------------------------------------------------------------------------
# the window shuffle
# ---------------------------------------------------------------------------


def words(text: str) -> list[str]:
    return text.split()


def test_shuffle_is_reproducible_for_the_same_seed_and_k():
    a = window_shuffle(PROSE, 5, random.Random(0))
    b = window_shuffle(PROSE, 5, random.Random(0))
    assert a == b


def test_shuffle_differs_for_a_different_seed():
    a = window_shuffle(PROSE, 5, random.Random(0))
    b = window_shuffle(PROSE, 5, random.Random(1))
    assert a != b, "two seeds producing the same order would make --seed a lie"


def test_shuffle_differs_for_a_different_k():
    a = window_shuffle(PROSE, 2, random.Random(0))
    b = window_shuffle(PROSE, 15, random.Random(0))
    assert a != b


def test_shuffle_preserves_every_word():
    """Reordering only. Nothing added, dropped, or altered."""
    out = window_shuffle(PROSE, 5, random.Random(0))
    assert sorted(words(out)) == sorted(words(PROSE))


def test_shuffle_keeps_words_inside_their_own_window():
    """The defining property: no word crosses a window boundary.

    This is what makes k a dial. If words could migrate, k=2 and k=30 would
    both just be a full-span shuffle with extra steps.
    """
    k = 4
    source = " ".join(f"w{i}" for i in range(20))
    out = words(window_shuffle(source, k, random.Random(7)))
    original = words(source)
    for start in range(0, len(original), k):
        assert sorted(out[start:start + k]) == sorted(original[start:start + k])


def test_window_of_one_cannot_reorder_anything():
    assert window_shuffle(PROSE, 1, random.Random(0)) == " ".join(words(PROSE))


def test_the_final_partial_window_is_shuffled_too():
    """13 words with k=5 leaves a 3-word tail; it must not be left alone.

    An unshuffled tail would put a run of untouched original text at the end
    of every noise passage -- a systematic difference from the rest of it,
    and one that would show up in the loss on exactly those tokens.
    """
    source = " ".join(f"w{i}" for i in range(13))
    tail_original = words(source)[10:]

    reordered = False
    for seed in range(30):
        out = words(window_shuffle(source, 5, random.Random(seed)))
        assert sorted(out[10:]) == sorted(tail_original), "tail words changed"
        if out[10:] != tail_original:
            reordered = True
            break
    assert reordered, "the final partial window was never reordered"


def test_a_window_larger_than_the_text_shuffles_the_whole_thing():
    """k >= word count is the full-span shuffle the sweep uses."""
    out = window_shuffle(PROSE, 10_000, random.Random(0))
    assert sorted(words(out)) == sorted(words(PROSE))
    assert out != " ".join(words(PROSE))


def test_a_nonpositive_window_is_rejected():
    for bad in (0, -1):
        with pytest.raises(MakeBurstsError):
            window_shuffle(PROSE, bad, random.Random(0))


# ---------------------------------------------------------------------------
# span quality filters
# ---------------------------------------------------------------------------


def test_ordinary_prose_passes_every_filter():
    stats = span_stats(PROSE)
    assert stats.acceptable, stats.rejections
    assert stats.rejections == ()


def test_mostly_non_ascii_is_rejected():
    stats = span_stats("これは日本語の文章です。これも。そしてこれも。")
    assert not stats.acceptable
    assert any("non-ASCII" in r for r in stats.rejections)


def test_mostly_digits_and_punctuation_is_rejected():
    stats = span_stats("12.3 | 45.6 | 78.9 | 10.1 | 11.2 | 13.4 | 15.6 | 17.8")
    assert not stats.acceptable
    assert any("punctuation/digits" in r or "alphabetic" in r
               for r in stats.rejections)


def test_too_few_sentence_ends_is_rejected():
    """A navigation bar is alphabetic and ASCII but is not prose."""
    stats = span_stats("Home About Contact Careers Privacy Terms Sitemap News "
                       "Sports Weather Business Travel Culture Opinion Video")
    assert not stats.acceptable
    assert any("sentence-ending" in r for r in stats.rejections)
    assert stats.sentence_ends < MIN_SENTENCE_ENDS


def test_low_alphabetic_fraction_is_rejected():
    stats = span_stats("a1! b2? c3. 456789 012345 678901 234567 890123 456789")
    assert not stats.acceptable


def test_empty_span_is_not_acceptable():
    assert not span_stats("").acceptable
    assert not span_stats("     ").acceptable


def test_fractions_ignore_whitespace():
    """Otherwise every fraction would depend on how the page was wrapped."""
    assert span_stats("ab. cd. ef.") == span_stats("ab.\n\ncd.\n\n\nef.")


# ---------------------------------------------------------------------------
# mid-word trimming
# ---------------------------------------------------------------------------


def test_cut_mid_word_is_detected():
    assert cut_is_mid_word("collaboration here", "collabora")


def test_cut_at_a_word_boundary_is_not_mid_word():
    assert not cut_is_mid_word("one two three", "one two")
    assert not cut_is_mid_word("one two three", "one two ")


def test_cut_at_the_very_end_is_not_mid_word():
    assert not cut_is_mid_word("one two", "one two")


# ---------------------------------------------------------------------------
# the guard on the hand-written arms
# ---------------------------------------------------------------------------


def test_a_missing_handwritten_arm_is_an_error_not_a_regeneration(tmp_path,
                                                                  capsys):
    """An empty outdir must fail, not helpfully invent fluent_false.txt."""
    exit_code = make_bursts.main(["--k", "5", "--outdir", str(tmp_path)])

    assert exit_code == 1
    err = capsys.readouterr().err
    assert "hand-written" in err
    assert "not generated" in err


def test_a_nonpositive_k_is_rejected(tmp_path, capsys):
    assert make_bursts.main(["--k", "0", "--outdir", str(tmp_path)]) == 1
    assert "--k must be at least 1" in capsys.readouterr().err


def test_the_registry_marks_exactly_two_arms_hand_written():
    """The two fluent arms are authored; the other three are generated."""
    handwritten = {s.name for s in make_bursts.ARM_SPECS if not s.is_generated}
    assert handwritten == {"fluent-false", "fluent-true"}
    assert len(make_bursts.ARM_SPECS) == 5


def test_span_count_comes_from_the_registry_not_a_literal():
    """Requirement H8: adding a corpus-derived arm changes this by itself."""
    assert len(make_bursts.span_arms()) == sum(
        1 for s in make_bursts.ARM_SPECS if s.needs_span)
    assert {s.name for s in make_bursts.span_arms()} == {
        "scrambled", "pos-substituted"}


def test_select_spans_says_how_many_it_needed_and_found():
    """Requirement H8: a short pool raises with both numbers, not silently."""
    docs = ["word " * 50] * 2
    with pytest.raises(MakeBurstsError) as exc:
        make_bursts.select_spans(docs, 5, 10, random.Random(0),
                                 stream=io.StringIO())
    message = str(exc.value)
    assert "needed 5" in message
    assert "found only" in message


# ---------------------------------------------------------------------------
# the committed burst files -- the acceptance criteria, kept under test
# ---------------------------------------------------------------------------


def arm_filenames() -> list[str]:
    return [spec.filename for spec in make_bursts.ARM_SPECS]


def test_all_five_arm_files_exist():
    for name in arm_filenames():
        assert (BURSTS / name).is_file(), f"bursts/{name} is missing"
    assert (BURSTS / "provenance.json").is_file()
    assert (BURSTS / "context.txt").is_file()
    assert (BURSTS / "pos_pool.json").is_file()


def test_burst_files_have_no_carriage_returns_and_no_trailing_newline():
    """Both would change the token count, and the counts have to match.

    A trailing newline is a token under the GPT-2 tokenizer, so it is not a
    harmless convention here -- it would make one arm a token longer than the
    four built to match it.
    """
    for name in arm_filenames() + ["context.txt"]:
        raw = (BURSTS / name).read_bytes()
        assert b"\r" not in raw, f"bursts/{name} has CRLF endings"
        assert not raw.endswith(b"\n"), f"bursts/{name} has a trailing newline"


def test_provenance_records_every_arm_and_the_context():
    import hashlib

    provenance = json.loads(
        (BURSTS / "provenance.json").read_text(encoding="utf-8"))

    assert provenance["schema_version"] == 2
    assert provenance["spec"] == "v4"
    # window_size_k used to sit at the top level. It belongs to one arm.
    assert "window_size_k" not in provenance
    assert provenance["arms"]["scrambled"]["params"]["k"] >= 1

    for spec in make_bursts.ARM_SPECS:
        record = provenance["arms"][spec.name]
        actual = hashlib.sha256(
            (BURSTS / spec.filename).read_bytes()).hexdigest()
        assert actual == record["sha256"], (
            f"bursts/{spec.filename} has changed since provenance was written")
        assert record["generator"] == spec.generator

    context = provenance["context"]
    assert hashlib.sha256(
        (BURSTS / "context.txt").read_bytes()).hexdigest() == context["sha256"]
    assert context["tokens"] == provenance["sequence_tokens"]

    # ordinary.txt is retained as substrate, not as an arm.
    assert "ordinary" not in provenance["arms"]
    assert provenance["substrate"]["file"] == "ordinary.txt"


def test_every_generated_arm_records_its_own_derived_seed():
    provenance = json.loads(
        (BURSTS / "provenance.json").read_text(encoding="utf-8"))
    seeds = {}
    for spec in make_bursts.ARM_SPECS:
        record = provenance["arms"][spec.name]
        if spec.is_generated:
            assert isinstance(record["derived_seed"], int)
            seeds[spec.name] = record["derived_seed"]
        else:
            assert record["derived_seed"] is None
    assert len(set(seeds.values())) == len(seeds), (
        "two arms share a derived seed; their draws would be correlated")


def test_random_chars_records_its_alphabet_and_space_frequency():
    provenance = json.loads(
        (BURSTS / "provenance.json").read_text(encoding="utf-8"))
    params = provenance["arms"]["random-chars"]["params"]
    assert params["alphabet"] == make_bursts.RANDOM_CHARS_ALPHABET_NAME
    assert params["space_frequency"] == 0.0


def test_random_chars_file_contains_no_whitespace():
    """The arm is defined as having no word structure at all."""
    text = (BURSTS / "random_chars.txt").read_text(encoding="utf-8")
    assert text
    assert not any(c.isspace() for c in text), (
        "random-chars contains whitespace; it is supposed to be one "
        "unbroken run")
    assert all(33 <= ord(c) <= 126 for c in text)


@requires_transformers
def test_all_five_passages_are_the_same_token_length():
    """The acceptance criterion for this step, as a test.

    If this fails, every gradient-norm comparison between the arms is
    measuring passage length as well as passage content.
    """
    from burst_match import load_tokenizer
    import io

    tokenizer = load_tokenizer(stream=io.StringIO())
    counts = {
        spec.filename: len(tokenizer(
            (BURSTS / spec.filename).read_text(encoding="utf-8"),
            add_special_tokens=False)["input_ids"])
        for spec in make_bursts.ARM_SPECS
    }

    assert len(set(counts.values())) == 1, f"token counts differ: {counts}"

    provenance = json.loads(
        (BURSTS / "provenance.json").read_text(encoding="utf-8"))
    assert set(counts.values()) == {provenance["target_tokens"]}


@requires_transformers
def test_trim_produces_exactly_the_requested_token_count():
    from burst_match import load_tokenizer
    import io

    tokenizer = load_tokenizer(stream=io.StringIO())
    long_text = PROSE * 20

    for target in (10, 37, 194):
        trimmed, _ = make_bursts.trim_to_tokens(tokenizer, long_text, target)
        assert make_bursts.token_count(tokenizer, trimmed) == target


@requires_transformers
def test_trimming_below_the_available_length_is_an_error():
    from burst_match import load_tokenizer
    import io

    tokenizer = load_tokenizer(stream=io.StringIO())
    with pytest.raises(MakeBurstsError) as exc:
        make_bursts.trim_to_tokens(tokenizer, "three words only", 500)
    assert "only" in str(exc.value)
