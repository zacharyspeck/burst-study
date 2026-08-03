"""The corpus layout: arithmetic that must close, and a slice that must not leak.

THE RISK THIS FILE EXISTS FOR.

Cross-module obligation 1 says that if the held-out slice is not carved out
during tokenization, the only fix is re-tokenizing the corpus and re-running
every model. So the layout has to be right before a single token is written,
and "right" here is almost entirely arithmetic: divisions that must come out
whole, boundaries that must meet exactly, and two independent routes to every
number.

The load-bearing test is the disjointness one. A held-out slice that overlaps
the training slice does not crash anything -- the barrier still returns a
number, and the number looks reasonable while measuring memorisation. That is
the same shape as a check that passes when handed nothing.
"""

from __future__ import annotations

import importlib.util
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


S = _load("corpus_spec")


# ---------------------------------------------------------------------------
# The arithmetic closes
# ---------------------------------------------------------------------------


def test_every_division_comes_out_whole():
    assert S.HELDOUT_TOKENS % S.SEQ_LEN == 0
    assert S.TRAIN_TOKENS % S.SEQ_LEN == 0
    assert S.TRAIN_SEQUENCES % S.SEQUENCES_PER_SHARD == 0


def test_the_headline_numbers_are_what_they_claim():
    assert S.TRAIN_TOKENS == 256 * 1024 * 9536 == 2_499_805_184
    assert S.TRAIN_SEQUENCES == 2_441_216
    assert S.N_SHARDS == 149
    assert S.HELDOUT_SEQUENCES == 10_240
    assert S.TOTAL_TOKENS == 2_510_290_944


def test_shards_and_heldout_account_for_every_token():
    """No token is in two places and none is in none."""
    from_shards = S.N_SHARDS * S.SEQUENCES_PER_SHARD * S.SEQ_LEN
    assert from_shards == S.TRAIN_TOKENS
    assert S.HELDOUT_TOKENS + from_shards == S.TOTAL_TOKENS


def test_slices_meet_exactly_with_no_gap_and_no_overlap():
    assert S.TRAIN_START == S.HELDOUT_END
    assert S.TRAIN_END - S.TRAIN_START == S.TRAIN_TOKENS


def test_byte_size_matches_token_count_both_directions():
    """Raw shards have no header, which is what makes this exact."""
    assert S.tokens_from_byte_size(S.shard_byte_size()) == (
        S.SEQUENCES_PER_SHARD * S.SEQ_LEN)
    assert S.tokens_from_byte_size(S.heldout_byte_size()) == S.HELDOUT_TOKENS
    assert S.TOTAL_TOKENS * S.BYTES_PER_TOKEN == 5_020_581_888


def test_an_odd_byte_count_is_a_truncated_file_not_a_rounding():
    with pytest.raises(S.CorpusSpecError, match="truncated mid-token"):
        S.tokens_from_byte_size(S.shard_byte_size() + 1)


# ---------------------------------------------------------------------------
# Two routes to every boundary
# ---------------------------------------------------------------------------


def test_shard_ranges_tile_the_training_slice_exactly():
    covered = []
    for i in range(S.N_SHARDS):
        covered.append(S.shard_sequence_range(i))
    assert covered[0][0] == 0
    assert covered[-1][1] == S.TRAIN_SEQUENCES
    for (_, end), (start, _) in zip(covered, covered[1:]):
        assert end == start, "a gap or overlap between consecutive shards"


def test_shard_range_is_arithmetic_not_a_lookup():
    """The second route the manifest gets compared against."""
    for i in (0, 1, 74, S.N_SHARDS - 1):
        start, end = S.shard_sequence_range(i)
        assert start == i * S.SEQUENCES_PER_SHARD
        assert end - start == S.SEQUENCES_PER_SHARD


def test_shard_index_out_of_range_is_refused():
    with pytest.raises(S.CorpusSpecError, match="out of range"):
        S.shard_sequence_range(S.N_SHARDS)


# ---------------------------------------------------------------------------
# The held-out guarantee
# ---------------------------------------------------------------------------


def test_training_offsets_never_reach_the_heldout_slice():
    assert S.training_token_offset(0) == S.HELDOUT_END
    assert S.training_token_offset(0) >= S.HELDOUT_TOKENS
    last = S.training_token_offset(S.TRAIN_SEQUENCES - 1) + S.SEQ_LEN
    assert last == S.TRAIN_END


def test_disjointness_holds_and_reports_the_boundary():
    report = S.assert_heldout_disjoint_from_training()
    assert report["disjoint"] is True
    assert report["heldout_tokens"] == [0, S.HELDOUT_TOKENS]
    assert report["training_tokens"][0] == S.HELDOUT_TOKENS
    assert report["gap_tokens"] == 0


def test_disjointness_refuses_a_corpus_longer_than_the_one_built():
    """A loop intending more sequences than exist is caught before batch one."""
    with pytest.raises(S.CorpusSpecError, match="past the end"):
        S.assert_heldout_disjoint_from_training(S.TRAIN_SEQUENCES + 1)


def test_disjointness_would_catch_a_front_shifted_training_slice(monkeypatch):
    """The failure the check exists for, forced.

    If TRAIN_START were ever moved to 0 -- the obvious 'simplification' -- the
    training slice would sit exactly on top of the evaluation set and nothing
    would crash. This proves the check notices.
    """
    monkeypatch.setattr(S, "TRAIN_START", 0)
    with pytest.raises(S.CorpusSpecError, match="OVERLAPS THE HELD-OUT"):
        S.assert_heldout_disjoint_from_training()


def test_the_sampler_index_space_is_the_training_slice_only():
    """Ties data_order's permutation domain to this module's geometry."""
    D = _load("data_order")
    assert D.sequences_for_budget(256, 1024, 9536) == S.TRAIN_SEQUENCES
    perm = D.sequence_permutation(0, 64)
    assert max(perm) < 64 <= S.TRAIN_SEQUENCES


# ---------------------------------------------------------------------------
# Against the config -- the duplicate exists so it can be checked
# ---------------------------------------------------------------------------


def test_geometry_agrees_with_the_shipped_config():
    sys.path.insert(0, str(REPO_ROOT))
    from burst.config import load_config

    cfg = load_config(REPO_ROOT / "configs" / "base.yaml",
                      REPO_ROOT / "configs" / "runs" / "seed03_twin.yaml",
                      outdir=Path("/tmp/unused"))
    report = S.check_against_config(cfg)
    assert report["agrees"] is True
    assert report["expected_token_budget"] == S.TRAIN_TOKENS


def test_a_disagreeing_budget_is_refused_loudly():
    class FakeTraining:
        batch_size, seq_len, total_steps = 256, 1024, 9536

    class FakeCorpus:
        expected_token_budget = 999

    class FakeCfg:
        training, corpus = FakeTraining(), FakeCorpus()

    with pytest.raises(S.CorpusSpecError, match="disagrees with itself"):
        S.check_against_config(FakeCfg())


# ---------------------------------------------------------------------------
# Source pinning
# ---------------------------------------------------------------------------


def test_the_source_revision_is_pinned_not_a_branch():
    """`main` moves; a 40-hex commit does not."""
    assert len(S.REVISION) == 40
    assert all(c in "0123456789abcdef" for c in S.REVISION)
    assert S.DATASET == "Skylion007/openwebtext"


def test_source_filenames_are_arithmetic_not_a_listing():
    """Never directory order -- the whole reason the files are numbered."""
    assert S.SOURCE_TEMPLATE.format(index=0) == \
        "plain_text/train-00000-of-00080.parquet"
    assert S.SOURCE_TEMPLATE.format(index=79) == \
        "plain_text/train-00079-of-00080.parquet"


def test_summary_is_data_not_prose():
    """A report derives its numbers from this rather than restating them."""
    s = S.summary()
    assert s["total"]["tokens"] == S.TOTAL_TOKENS
    assert s["heldout"]["token_range"] == [0, S.HELDOUT_TOKENS]
    assert s["training"]["shards"] == S.N_SHARDS
