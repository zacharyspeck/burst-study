"""Tests for scripts/burst_match.py.

Split deliberately into two halves.

The first half tests the pure logic -- the context-limit error, the
length-mismatch warning, the loss statistics, the batch scaling -- and imports
nothing but the script itself. These run in the base environment, with no
torch and no transformers installed, which is the same environment the config
test suite has to keep passing in.

The second half needs torch, and some of it needs the GPT-2 weights as well.
Those tests skip cleanly when the dependency or the model is not available,
so `pytest` in the base environment is green rather than broken.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# scripts/ is not a package -- generate_overrides.py is run directly too -- so
# the directory goes on sys.path rather than being imported as scripts.foo.
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import burst_match  # noqa: E402
from burst_match import (  # noqa: E402
    BATCH_SIZE_KEY,
    CONTEXT_LIMIT,
    DEFAULT_BASE_CONFIG,
    BurstMatchError,
    Measurement,
    TokenLoss,
    batch_scaled,
    batch_size_from_config,
    check_context_limit,
    check_measurable,
    length_mismatch,
    loss_stats,
    percent_change,
    resolve_batch_size,
    top_surprising,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fake_measurement(name: str, n_tokens: int, *, grad_norm: float = 1.0,
                     mean_loss: float = 1.0) -> Measurement:
    """A Measurement built by hand, for testing the pure reporting logic."""
    return Measurement(
        path=Path(name),
        n_tokens=n_tokens,
        mean_loss=mean_loss,
        std_loss=0.0,
        max_loss=mean_loss,
        top_surprising=(TokenLoss(position=1, loss=mean_loss, text=" x"),),
        grad_norm=grad_norm,
        n_params=124_439_808,
    )


# ---------------------------------------------------------------------------
# requirement 3: the context limit is an error, never a silent truncation
# ---------------------------------------------------------------------------


def test_over_context_raises_and_names_the_actual_count():
    with pytest.raises(BurstMatchError) as exc:
        check_context_limit(1500, "coherent.txt")
    message = str(exc.value)
    assert "1500" in message, "the error must name the actual token count"
    assert "1024" in message, "the error must name the limit"
    assert "coherent.txt" in message, "the error must name the file"
    # The point of the error is that it did NOT truncate.
    assert "will not truncate" in message


def test_exactly_at_the_context_limit_is_allowed():
    """1024 is the limit, not one past it."""
    check_context_limit(CONTEXT_LIMIT, "x.txt")


def test_one_over_the_context_limit_raises():
    with pytest.raises(BurstMatchError):
        check_context_limit(CONTEXT_LIMIT + 1, "x.txt")


def test_single_token_passage_is_rejected():
    """One token yields zero next-token predictions, so there is no loss."""
    with pytest.raises(BurstMatchError) as exc:
        check_measurable(1, "tiny.txt")
    assert "at least 2" in str(exc.value).lower()
    check_measurable(2, "tiny.txt")  # two is enough: one prediction


# ---------------------------------------------------------------------------
# requirement 2: unequal token counts must warn loudly, with the factor
# ---------------------------------------------------------------------------


def test_unequal_length_warning_fires_with_the_right_factor():
    ms = [fake_measurement("coherent.txt", 400),
          fake_measurement("noise.txt", 200)]
    assert length_mismatch(ms) == (200, 400, 2.0)

    warning = burst_match.format_length_warning(ms)
    assert "WARNING" in warning
    assert "NOT directly comparable" in warning
    assert "2.00x" in warning, "the warning must state the weighting factor"
    # Both passages, and their counts, must appear in the warning itself.
    assert "coherent.txt" in warning and "noise.txt" in warning
    assert "400" in warning and "200" in warning


def test_equal_lengths_produce_no_warning():
    ms = [fake_measurement("a.txt", 300), fake_measurement("b.txt", 300)]
    assert length_mismatch(ms) is None
    assert burst_match.format_length_warning(ms) == ""


def test_mismatch_uses_the_extremes_of_three_passages():
    ms = [fake_measurement("a.txt", 300), fake_measurement("b.txt", 100),
          fake_measurement("c.txt", 250)]
    shortest, longest, factor = length_mismatch(ms)
    assert (shortest, longest) == (100, 300)
    assert factor == pytest.approx(3.0)


def test_comparison_table_states_comparability_either_way():
    matched = [fake_measurement("a.txt", 300), fake_measurement("b.txt", 300)]
    assert "directly comparable" in burst_match.format_comparison(matched, 256)

    mismatched = [fake_measurement("a.txt", 300), fake_measurement("b.txt", 200)]
    assert "not a like-for-like" in burst_match.format_comparison(mismatched, 256)


# ---------------------------------------------------------------------------
# per-token loss statistics, on hand-checkable input
# ---------------------------------------------------------------------------


def test_loss_stats_on_hand_computed_values():
    """mean 2, population variance 2, max 4 -- checkable on paper."""
    mean, std, worst = loss_stats([0.0, 1.0, 2.0, 3.0, 4.0])
    assert mean == pytest.approx(2.0)
    # Population std: ((4+1+0+1+4)/5) ** 0.5 == sqrt(2). Sample std would be
    # sqrt(2.5) == 1.5811, so this also pins the ddof=0 choice.
    assert std == pytest.approx(math.sqrt(2.0))
    assert std != pytest.approx(math.sqrt(2.5))
    assert worst == 4.0


def test_loss_stats_of_a_single_prediction():
    mean, std, worst = loss_stats([3.5])
    assert (mean, std, worst) == (3.5, 0.0, 3.5)


def test_loss_stats_of_identical_values_has_zero_spread():
    mean, std, worst = loss_stats([2.0, 2.0, 2.0])
    assert (mean, std, worst) == (2.0, 0.0, 2.0)


def test_loss_stats_rejects_empty_input():
    with pytest.raises(BurstMatchError):
        loss_stats([])


def test_top_surprising_is_worst_first_and_capped():
    losses = [TokenLoss(position=i, loss=float(i), text=f"t{i}")
              for i in range(1, 11)]
    top = top_surprising(losses, k=5)
    assert [t.loss for t in top] == [10.0, 9.0, 8.0, 7.0, 6.0]
    assert len(top) == 5


def test_top_surprising_breaks_ties_by_position():
    """Equal losses must order stably, or two runs could disagree."""
    losses = [TokenLoss(position=7, loss=1.0, text="b"),
              TokenLoss(position=3, loss=1.0, text="a")]
    assert [t.position for t in top_surprising(losses, k=2)] == [3, 7]


def test_top_surprising_handles_fewer_tokens_than_k():
    losses = [TokenLoss(position=1, loss=1.0, text="a")]
    assert len(top_surprising(losses, k=5)) == 1


# ---------------------------------------------------------------------------
# requirement 4: the batch-scaled figure
# ---------------------------------------------------------------------------


def test_batch_scaling_divides_by_the_batch_size():
    assert batch_scaled(256.0, 256) == 1.0
    assert batch_scaled(12.8, 256) == pytest.approx(0.05)


def test_batch_scaling_rejects_a_nonpositive_batch():
    with pytest.raises(BurstMatchError):
        batch_scaled(1.0, 0)


def test_report_prints_both_figures_labelled():
    """Both numbers must appear, and the scaled one must say what it is."""
    text = burst_match.format_measurement(
        fake_measurement("a.txt", 300, grad_norm=25.6), batch_size=256)
    assert "25.600000" in text, "the standalone figure is the headline"
    assert "0.100000" in text, "the batch-scaled figure sits beside it"
    assert "headline" in text
    assert "batch-scaled 1/256" in text
    assert "batch of 256" in text


def test_percent_change_is_relative_to_the_first_passage():
    assert percent_change(2.0, 3.0) == pytest.approx(50.0)
    assert percent_change(4.0, 2.0) == pytest.approx(-50.0)
    assert percent_change(0.0, 1.0) is None


# ---------------------------------------------------------------------------
# the batch size comes from the config, not from a constant
#
# These need PyYAML, which is a base dependency, and burst.config, which
# imports nothing heavier. So they run in the torch-free environment -- which
# matters, because the thing they are guarding is a number the report scales
# every headline figure by.
# ---------------------------------------------------------------------------

REAL_BASE = REPO_ROOT / "configs" / "base.yaml"

#: Sentinel for write_base: remove the key entirely rather than set it.
DELETE = object()


def write_base(tmp_path: Path, **edits) -> Path:
    """A copy of the real base.yaml with dotted-path edits applied.

    Starting from the real file rather than a fixture is the same choice
    tests/test_config.py makes, for the same reason: these break if base.yaml
    drifts, which is the point of them.
    """
    import yaml

    data = yaml.safe_load(REAL_BASE.read_text(encoding="utf-8"))
    for dotted, value in edits.items():
        parts = dotted.split("__")
        node = data
        for part in parts[:-1]:
            node = node[part]
        if value is DELETE:
            del node[parts[-1]]
        else:
            node[parts[-1]] = value
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_batch_size_comes_from_the_config_not_a_constant(tmp_path):
    """Change the config, and the number the script uses changes with it.

    8 is chosen so that no leftover constant could produce it by accident.
    expected_token_budget is corrected alongside it, because the loader checks
    that batch_size * seq_len * total_steps equals it -- so this exercises the
    burst.config path rather than the fallback.
    """
    base = write_base(tmp_path, training__batch_size=8,
                      corpus__expected_token_budget=8 * 1024 * 9536)

    batch = batch_size_from_config(base)

    assert batch.value == 8
    assert "burst.config loader" in batch.source
    assert str(base) in batch.source


def test_a_second_config_value_gives_a_second_answer(tmp_path):
    """Two different configs, two different answers. No sticky state."""
    first = write_base(tmp_path / "a", training__batch_size=8,
                       corpus__expected_token_budget=8 * 1024 * 9536)
    second = write_base(tmp_path / "b", training__batch_size=64,
                        corpus__expected_token_budget=64 * 1024 * 9536)

    assert batch_size_from_config(first).value == 8
    assert batch_size_from_config(second).value == 64


def test_real_base_config_is_read_through_the_loader():
    """The shipped config must go down the validated path, not the fallback."""
    import yaml

    batch = batch_size_from_config(DEFAULT_BASE_CONFIG)
    in_file = yaml.safe_load(REAL_BASE.read_text(encoding="utf-8"))
    for part in BATCH_SIZE_KEY:
        in_file = in_file[part]

    assert batch.value == in_file
    assert "burst.config loader" in batch.source
    assert "direct YAML read" not in batch.source


def test_falls_back_to_yaml_when_the_loader_refuses(tmp_path):
    """A config the loader rejects still yields the key, and says so.

    batch_size is changed without correcting expected_token_budget, so the
    loader's token-budget check fails -- the loader is working correctly and
    the fallback reads the one key it needs anyway. The loader's validation is
    not relaxed to make this pass, which was the whole point.
    """
    base = write_base(tmp_path, training__batch_size=7)

    batch = batch_size_from_config(base)

    assert batch.value == 7
    assert "direct YAML read" in batch.source
    # The reason the loader declined must be stated, not swallowed.
    assert "declined" in batch.source
    assert "token budget" in batch.source


def test_missing_batch_size_key_raises_rather_than_defaulting(tmp_path):
    """No key, no number. The failure names the file and what it looked for."""
    base = write_base(tmp_path, training__batch_size=DELETE)

    with pytest.raises(BurstMatchError) as exc:
        batch_size_from_config(base)

    message = str(exc.value)
    assert str(base) in message, "the error must name the file"
    assert "training.batch_size" in message, "and the key it looked for"
    assert "--batch-size" in message, "and what you can do about it"


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(BurstMatchError) as exc:
        batch_size_from_config(tmp_path / "not-a-config.yaml")
    assert "not found" in str(exc.value)


def test_a_nonsense_batch_size_in_the_config_is_rejected(tmp_path):
    """The fallback validates too; it is not a hole in the type checking."""
    for bad in (0, -4, "256", 12.5, True):
        base = write_base(tmp_path, training__batch_size=bad)
        with pytest.raises(BurstMatchError) as exc:
            batch_size_from_config(base)
        assert "batch size must be" in str(exc.value)


def test_command_line_override_wins_over_the_config(tmp_path):
    base = write_base(tmp_path, training__batch_size=8,
                      corpus__expected_token_budget=8 * 1024 * 9536)

    assert resolve_batch_size(None, base).value == 8, "config when not given"

    overridden = resolve_batch_size(512, base)
    assert overridden.value == 512
    assert "overriding the config" in overridden.source


def test_override_works_even_when_the_config_cannot_be_read(tmp_path):
    """--batch-size is checked first, so it rescues an unreadable config."""
    assert resolve_batch_size(64, tmp_path / "missing.yaml").value == 64


def test_a_nonsense_override_is_rejected():
    for bad in (0, -1):
        with pytest.raises(BurstMatchError) as exc:
            resolve_batch_size(bad, DEFAULT_BASE_CONFIG)
        assert "--batch-size" in str(exc.value)


def test_header_line_states_both_the_value_and_the_source(tmp_path):
    base = write_base(tmp_path, training__batch_size=8,
                      corpus__expected_token_budget=8 * 1024 * 9536)

    line = burst_match.format_batch_line(batch_size_from_config(base))

    assert "8" in line
    assert "burst.config loader" in line
    assert str(base) in line


def test_no_hardcoded_batch_size_survives_in_the_script():
    """The regression guard for D8: no bare number standing in for the config.

    256 was the constant that used to live here. If it -- or any other bare
    integer that looks like a batch size -- reappears in the script, this
    fails. Cheaper than noticing that every scaled figure in a report was
    divided by a number the config stopped agreeing with months ago.
    """
    source = (REPO_ROOT / "scripts" / "burst_match.py").read_text(
        encoding="utf-8")

    assert not re.search(r"\b256\b", source), (
        "a literal 256 is back in burst_match.py; the batch size must come "
        "from the config"
    )
    assert "DEFAULT_BATCH_SIZE" not in source
    # The flag must not acquire a default either -- argparse would then never
    # consult the config at all.
    assert "default=None" in source


# ---------------------------------------------------------------------------
# torch-only tests
#
# Marked, not `pytest.importorskip`-ed. importorskip at module level raises
# during collection and would skip this whole file, including every pure test
# above -- which would quietly stop testing the logic that exists precisely so
# that it can be tested without torch.
# ---------------------------------------------------------------------------

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch is an optional dependency; install it with .[measure]",
)


@requires_torch
def test_per_token_losses_on_uniform_logits():
    """Uniform logits over V classes must give exactly ln(V) for every token.

    Hand-checkable: if the model assigns equal probability to all V tokens,
    the probability of whichever token actually comes next is 1/V, so the
    negative log probability is ln(V) regardless of what the tokens are.
    """
    import torch

    vocab, n_tokens = 8, 5
    logits = torch.zeros(1, n_tokens, vocab)
    input_ids = torch.tensor([[3, 1, 4, 1, 5]])

    losses = burst_match.per_token_losses(logits, input_ids)

    assert losses.shape == (n_tokens - 1,), "N tokens give N-1 predictions"
    for value in losses.tolist():
        assert value == pytest.approx(math.log(vocab))

    mean, std, worst = loss_stats(losses.tolist())
    assert mean == pytest.approx(math.log(8))
    assert std == pytest.approx(0.0)
    assert worst == pytest.approx(math.log(8))


@requires_torch
def test_per_token_losses_on_a_confident_prediction():
    """A large logit on the right token drives that token's loss to ~0."""
    import torch

    logits = torch.zeros(1, 3, 4)
    logits[0, 0, 2] = 50.0   # predicts token 2 with near-certainty
    logits[0, 1, 3] = 50.0   # predicts token 3 with near-certainty
    input_ids = torch.tensor([[0, 2, 3]])

    losses = burst_match.per_token_losses(logits, input_ids).tolist()
    assert losses[0] == pytest.approx(0.0, abs=1e-6)
    assert losses[1] == pytest.approx(0.0, abs=1e-6)


@requires_torch
def test_gradient_norm_matches_a_hand_computed_value():
    """On a one-parameter model the norm is just |dL/dw|, checkable by hand."""
    import torch
    import torch.nn as nn

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(3.0)
    # L = (w * 2) ** 2 = 4w^2, so dL/dw = 8w = 24 at w = 3.
    loss = (model(torch.tensor([[2.0]])) ** 2).sum()
    assert burst_match.gradient_norm(loss, model) == pytest.approx(24.0)


@requires_torch
def test_gradient_norm_does_not_accumulate_between_calls():
    """Two measurements in one process must not contaminate each other.

    This is why gradient_norm uses torch.autograd.grad rather than
    loss.backward(): backward() would add the first call's gradient into
    p.grad and the second call would read a larger number.
    """
    import torch
    import torch.nn as nn

    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(3.0)

    def norm():
        return burst_match.gradient_norm(
            (model(torch.tensor([[2.0]])) ** 2).sum(), model)

    first = norm()
    assert norm() == first
    assert norm() == first
    assert model.weight.grad is None, ".grad must never be touched"


# ---------------------------------------------------------------------------
# tests that need the real GPT-2 weights
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2():
    """The real model, loaded once. Skips if it cannot be obtained."""
    import io
    try:
        return burst_match.load_model(stream=io.StringIO())
    except BurstMatchError as exc:
        pytest.skip(f"GPT-2 unavailable: {exc}")


@pytest.fixture
def passage(tmp_path):
    path = tmp_path / "passage.txt"
    path.write_text(
        "The lighthouse keeper wrote the same sentence every evening. "
        "The sea did not change, but the handwriting did.",
        encoding="utf-8",
    )
    return path


# requirement 1: determinism.


def test_same_file_measured_twice_is_bit_identical(gpt2, passage):
    """The headline determinism requirement, on the real model.

    Compared with == on the raw floats, not on the printed strings: printing
    rounds, and rounding could hide a difference in the low bits that would
    make two measurements of the same passage disagree.
    """
    tokenizer, model = gpt2
    first = burst_match.measure(passage, tokenizer, model)
    second = burst_match.measure(passage, tokenizer, model)

    assert first.n_tokens == second.n_tokens
    assert first.mean_loss == second.mean_loss
    assert first.std_loss == second.std_loss
    assert first.max_loss == second.max_loss
    assert first.grad_norm == second.grad_norm
    assert first.top_surprising == second.top_surprising
    assert first == second, "every field must match, not just the ones above"


def test_determinism_survives_measuring_another_file_in_between(gpt2, passage,
                                                                tmp_path):
    """Measuring B between two measurements of A must not change A.

    The failure this guards against is gradient accumulation: if the backward
    pass wrote into p.grad, A's second measurement would carry B's gradient.
    """
    tokenizer, model = gpt2
    other = tmp_path / "other.txt"
    other.write_text("Entirely different words, arranged entirely differently.",
                     encoding="utf-8")

    first = burst_match.measure(passage, tokenizer, model)
    burst_match.measure(other, tokenizer, model)
    again = burst_match.measure(passage, tokenizer, model)

    assert first == again


def test_model_is_in_eval_mode(gpt2):
    """Dropout off. This is what makes the determinism above possible."""
    _, model = gpt2
    assert not model.training


def test_measure_rejects_a_passage_over_the_context_limit(gpt2, tmp_path):
    """The real tokenizer, a real over-long file, and no truncation."""
    tokenizer, model = gpt2
    long_file = tmp_path / "too_long.txt"
    # " word" is a single GPT-2 token, so this is ~1500 tokens.
    long_file.write_text(" word" * 1500, encoding="utf-8")

    with pytest.raises(BurstMatchError) as exc:
        burst_match.measure(long_file, tokenizer, model)
    message = str(exc.value)
    assert "1024" in message
    assert "too_long.txt" in message
    # The actual count must be named, and it must be the real one.
    n_tokens = len(tokenizer(" word" * 1500, add_special_tokens=False)["input_ids"])
    assert n_tokens > CONTEXT_LIMIT
    assert str(n_tokens) in message


def test_measure_reports_the_real_token_count(gpt2, passage):
    tokenizer, model = gpt2
    m = burst_match.measure(passage, tokenizer, model)
    expected = len(tokenizer(passage.read_text(encoding="utf-8"),
                             add_special_tokens=False)["input_ids"])
    assert m.n_tokens == expected
    assert m.n_predictions == expected - 1


def test_measured_stats_are_self_consistent(gpt2, passage):
    """The printed summary must actually describe the per-token losses."""
    tokenizer, model = gpt2
    m = burst_match.measure(passage, tokenizer, model)

    assert m.max_loss >= m.mean_loss
    assert m.std_loss >= 0.0
    assert m.grad_norm > 0.0
    # The worst token in the top-5 list is the max, by construction.
    assert m.top_surprising[0].loss == pytest.approx(m.max_loss)
    assert len(m.top_surprising) == 5
    # Positions are token indices, never 0: the first token is context only.
    assert all(1 <= t.position < m.n_tokens for t in m.top_surprising)


def test_empty_file_is_rejected(gpt2, tmp_path):
    tokenizer, model = gpt2
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(BurstMatchError) as exc:
        burst_match.measure(empty, tokenizer, model)
    assert "empty" in str(exc.value)


def test_missing_file_is_rejected(gpt2, tmp_path):
    tokenizer, model = gpt2
    with pytest.raises(BurstMatchError) as exc:
        burst_match.measure(tmp_path / "nope.txt", tokenizer, model)
    assert "no such file" in str(exc.value)
