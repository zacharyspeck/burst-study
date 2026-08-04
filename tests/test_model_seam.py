"""The model seam: it must build either family, and refuse to guess which.

THE RISK THIS FILE EXISTS FOR.

The study will train HF GPT-2 but the repo still carries the nn.Linear probe
model, and the swap has not landed. A loop that picked a default would decide,
silently, four things that are not the loop's to decide: whether checkpoints
are interchangeable, what the seed-to-weights map is, whether the determinism
evidence transfers, and whether step 9's ruler can read the result.

So the load-bearing tests here are the refusals. `build_model` has no default
family, and the parameter-count check is the only thing standing between the
loop and training the wrong architecture for nine thousand steps --
`expected_param_count` has been in the config since the beginning with nothing
comparing against it.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
torch = pytest.importorskip("torch")


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MS = _load("model_seam")


@pytest.fixture(scope="module")
def cfg():
    sys.path.insert(0, str(REPO_ROOT))
    from burst.config import load_config
    import io

    with tempfile.TemporaryDirectory() as tmp:
        return load_config(
            REPO_ROOT / "configs" / "base.yaml",
            REPO_ROOT / "configs" / "runs" / "seed03_twin.yaml",
            outdir=Path(tmp), require_complete=False, stream=io.StringIO())


@pytest.fixture(scope="module")
def tiny_cfg(cfg):
    """A small shape, so tests that only need A model do not build 124M twice."""
    import dataclasses

    model = dataclasses.replace(
        cfg.model, n_layer=2, n_head=2, n_embd=32, block_size=64,
        vocab_size=128, expected_param_count=None)
    return dataclasses.replace(cfg, model=model)


# ---------------------------------------------------------------------------
# There is no default family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [None, "", "gpt2", "hf", "linear", "auto"])
def test_an_unknown_or_absent_family_is_refused(tiny_cfg, bad):
    with pytest.raises(MS.ModelSeamError, match="unknown model family"):
        MS.build_model(tiny_cfg, bad)


def test_the_refusal_says_why_a_default_would_be_wrong(tiny_cfg):
    with pytest.raises(MS.ModelSeamError) as exc:
        MS.build_model(tiny_cfg, "auto")
    message = str(exc.value)
    assert "NO DEFAULT" in message
    assert "checkpoints are" in message or "interchangeable" in message


def test_build_model_requires_the_family_positionally(tiny_cfg):
    """No keyword default can creep in later."""
    import inspect

    params = inspect.signature(MS.build_model).parameters
    assert list(params) == ["cfg", "family"]
    assert params["family"].default is inspect.Parameter.empty


# ---------------------------------------------------------------------------
# Both families build, and both are the model the config describes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", MS.FAMILIES)
def test_both_families_hit_expected_param_count_exactly(cfg, family):
    """124,439,808 for GPT-2 Base with embeddings tied."""
    pytest.importorskip("transformers")
    model = MS.build_model(cfg, family)
    assert MS.parameter_count(model) == cfg.model.expected_param_count
    assert MS.parameter_count(model) == 124_439_808


def test_the_two_families_are_indistinguishable_by_parameter_count(cfg):
    """WHY `family` IS A PROVENANCE FIELD. Pinned, not remembered.

    `expected_param_count` is the seam's guard against training the wrong
    architecture, and `scripts/model_seam.py` used to claim it was "the only
    thing that would notice" a family mix-up. It would notice nothing: both
    families land on exactly the same count, so the guard is blind to the one
    distinction that decides whether two checkpoints are comparable.

    That is what makes `run_provenance.yaml`'s `family` field load-bearing
    rather than decorative, and what `launch.py::conflicting_family` exists to
    enforce.

    IF THIS TEST EVER FAILS, the counts have diverged and the rationale above
    has changed. Do not "fix" it by loosening the assertion -- go and re-read
    the docstring in `scripts/model_seam.py` and this repo's S85, because the
    reasoning they record would no longer hold.
    """
    pytest.importorskip("transformers")
    counts = {
        family: MS.parameter_count(MS.build_model(cfg, family))
        for family in MS.FAMILIES
    }
    assert len(set(counts.values())) == 1, (
        f"families no longer share a parameter count: {counts}")
    assert set(counts.values()) == {124_439_808}


@pytest.mark.parametrize("family", MS.FAMILIES)
def test_a_wrong_parameter_count_is_refused(tiny_cfg, family):
    """The check that catches training the wrong architecture."""
    import dataclasses

    wrong = dataclasses.replace(
        tiny_cfg, model=dataclasses.replace(
            tiny_cfg.model, expected_param_count=12345))
    with pytest.raises(MS.ModelSeamError, match="PARAMETER COUNT MISMATCH"):
        MS.build_model(wrong, family)


@pytest.mark.parametrize("family", MS.FAMILIES)
def test_the_mismatch_message_names_the_tying_difference(tiny_cfg, family):
    """A difference of exactly vocab*n_embd means the tie is wrong."""
    import dataclasses

    wrong = dataclasses.replace(
        tiny_cfg, model=dataclasses.replace(
            tiny_cfg.model, expected_param_count=1))
    with pytest.raises(MS.ModelSeamError) as exc:
        MS.build_model(wrong, family)
    assert "tie_embeddings" in str(exc.value)


def test_null_tie_embeddings_is_refused_rather_than_assumed(tiny_cfg):
    import dataclasses

    undecided = dataclasses.replace(
        tiny_cfg, model=dataclasses.replace(
            tiny_cfg.model, tie_embeddings=None))
    with pytest.raises(MS.ModelSeamError, match="tie_embeddings is null"):
        MS.build_model(undecided, MS.FAMILY_HF_GPT2)


# ---------------------------------------------------------------------------
# Counting, and the tied-weight trap
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", MS.FAMILIES)
def test_parameter_count_counts_a_tied_embedding_once(tiny_cfg, family):
    """state_dict() would overstate GPT-2 Base by exactly vocab * n_embd.

    That is the same number as the tied/untied difference, so the check would
    pass for the wrong model -- the worst possible way for a count to be wrong.
    """
    model = MS.build_model(tiny_cfg, family)
    by_named = MS.parameter_count(model)
    by_state = sum(t.numel() for t in model.state_dict().values())
    assert by_named <= by_state
    ids = [id(p) for _, p in model.named_parameters()]
    assert len(ids) == len(set(ids)), "named_parameters returned a duplicate"


# ---------------------------------------------------------------------------
# The one place the families differ
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", MS.FAMILIES)
def test_compute_loss_returns_a_scalar_for_either_signature(tiny_cfg, family):
    model = MS.build_model(tiny_cfg, family)
    model.eval()
    x = torch.randint(0, tiny_cfg.model.vocab_size, (2, 16))
    y = torch.randint(0, tiny_cfg.model.vocab_size, (2, 16))
    with torch.no_grad():
        loss = MS.compute_loss(model, x, y)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


@pytest.mark.parametrize("family", MS.FAMILIES)
def test_loss_is_finite_and_near_chance_at_init(tiny_cfg, family):
    """A fresh model should sit near ln(vocab); far off means a broken build."""
    import math

    model = MS.build_model(tiny_cfg, family)
    model.eval()
    x = torch.randint(0, tiny_cfg.model.vocab_size, (4, 16))
    with torch.no_grad():
        loss = float(MS.compute_loss(model, x, x))
    assert abs(loss - math.log(tiny_cfg.model.vocab_size)) < 1.5


def test_describe_is_data_not_prose(tiny_cfg):
    model = MS.build_model(tiny_cfg, MS.FAMILY_HF_GPT2)
    described = MS.describe(model, MS.FAMILY_HF_GPT2)
    assert described["family"] == MS.FAMILY_HF_GPT2
    assert described["parameters"] == MS.parameter_count(model)
    assert described["forward_signature"] == "input_ids/labels"


# ---------------------------------------------------------------------------
# The consequence the seam does NOT paper over
# ---------------------------------------------------------------------------


def test_the_two_families_are_not_checkpoint_compatible(tiny_cfg):
    """Recorded as a test because it is the thing a narrow interface hides.

    The loop is portable across families. The RUNS are not. If this ever starts
    passing, something has changed about what a checkpoint means.
    """
    hf = MS.build_model(tiny_cfg, MS.FAMILY_HF_GPT2)
    probe = MS.build_model(tiny_cfg, MS.FAMILY_PROBE_LINEAR)
    assert set(hf.state_dict()) != set(probe.state_dict())
    with pytest.raises(Exception):
        probe.load_state_dict(hf.state_dict())


def test_the_two_families_disagree_on_weights_from_the_same_seed(tiny_cfg):
    """The seed -> weights map is family-specific, so runs are incomparable."""
    torch.manual_seed(0)
    hf = MS.build_model(tiny_cfg, MS.FAMILY_HF_GPT2)
    torch.manual_seed(0)
    probe = MS.build_model(tiny_cfg, MS.FAMILY_PROBE_LINEAR)
    hf_first = next(iter(hf.state_dict().values())).flatten()[:8]
    probe_first = next(iter(probe.state_dict().values())).flatten()[:8]
    assert not torch.equal(hf_first, probe_first)
