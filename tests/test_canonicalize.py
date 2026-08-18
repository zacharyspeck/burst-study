"""Tests for scripts/canonicalize.py -- step 9, phases 0 through 2.

Four things this file guards, all of which fail silently if they break.

1. THE TRIPWIRE. A canonicalizer aimed at an architecture it does not
   understand does not crash. It permutes the wrong axis and emits plausible
   numbers forever. validate_architecture() has to refuse every shape it was
   not written for, including the rotary case, which would silently invalidate
   the GL(head_dim) freedom this module is built on.

2. THE LAYOUT CONTRACT. c_attn fuses Q, K and V into one tensor and Conv1D
   stores its weight transposed relative to nn.Linear. Every slice into that
   tensor is computed in one place and asserted here, because a slice that is
   off by one head produces a model that runs.

3. EQUIVALENCE, per candidate, in isolation. A transformation is a symmetry
   only if applying it leaves the outputs unchanged. Measured at float32 AND
   float64: exact maths shows a huge error collapse between the two, wrong
   maths shows none. THE COLLAPSE FACTOR IS THE SOLE CRITERION. Absolute
   float32 error is asserted only against this architecture's own MEASURED
   noise floor, never against a pre-set bound -- phase 2 found that the bound
   originally registered for float32 sat inside that noise floor and so
   discriminated random seeds rather than symmetries. See S43.

4. THE ORDERING CONTRACT. scripts/burst_match.py's gradient_parameters() is
   positional and filters on requires_grad; this module is name-keyed. If the
   two orderings ever disagree, every cosine in
   docs/measurements/8b-iv-gradient-direction.json is silently wrong.

Round-trip, non-triviality and mutation tests are NOT here. They require the
canonicalization recipe, which does not exist yet -- the drop list decided by
this file's equivalence tests is what determines what that recipe is.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
# scripts/ is not a package -- the same sys.path idiom the other suites use.
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import canonicalize  # noqa: E402
from canonicalize import (  # noqa: E402
    ALL_SYMMETRIES,
    COLLAPSE_FACTOR,
    F32_DIAGNOSTIC,
    GPT2_124M,
    MEASURED_F32_NOISE_FLOOR,
    TINY,
    TOL_F64,
    ArchSpec,
    CanonicalizeError,
    head_columns,
    head_rows_of_out_proj,
    probe_tokens,
    qkv_offset,
    symmetry_by_name,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None,
    reason="torch and transformers are optional; install .[measure]",
)

requires_scipy = pytest.mark.skipif(
    importlib.util.find_spec("scipy") is None,
    reason="scipy is an optional dependency; install .[measure]",
)


#: Candidates confirmed to be symmetries by the phase 2 measurement, and the
#: three confirmed not to be. Hardcoded rather than computed, so that a future
#: change which silently flips one of them fails this file instead of quietly
#: rewriting the study's ruler.
CONFIRMED_SYMMETRIES = (
    "layernorm_gain_rescale",
    "residual_permutation",
    "head_permutation",
    "head_internal_transform",
    "ffn_neuron_permutation",
    "key_bias_shift",
    "value_bias_shift",
)
CONFIRMED_NOT_SYMMETRIES = (
    "residual_rotation",
    "residual_scaling",
    "ffn_scaling",
)


# ---------------------------------------------------------------------------
# fixtures -- built in process, never downloaded, never committed
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    """A small GPT-2 at a GENERIC point in weight space. See the test below."""
    return canonicalize.build_tiny_model(seed=0)


@pytest.fixture(scope="module")
def real_gpt2():
    """Public GPT-2 124M. Skips cleanly when the weights are unavailable."""
    try:
        return canonicalize.build_real_gpt2()
    except CanonicalizeError as exc:
        pytest.skip(f"GPT-2 unavailable: {exc}")


def tiny_builder():
    return canonicalize.build_tiny_model(seed=0)


# ---------------------------------------------------------------------------
# the fixture itself, which was wrong once and would have inverted a verdict
# ---------------------------------------------------------------------------


@requires_torch
def test_the_tiny_fixture_sits_at_a_generic_point_not_a_degenerate_one(tiny):
    """The near-miss this test exists to prevent, recorded as a test.

    A freshly constructed GPT2LMHeadModel has every LayerNorm gain at exactly
    1.0 and every bias at exactly 0.0. On such a model diag(gamma) is the
    identity, so it commutes with any rotation, and residual rotation passes
    its equivalence test on a technicality. Measured on the first version of
    this fixture: residual rotation reported SYMMETRY at 1.8e-07. With the
    gains randomised it reports NOT A SYMMETRY at 5.5e-01.

    Real GPT-2's gains span 2.557e-04 to 17.42. A fixture pinned at 1.0 is not
    a small inaccuracy; it is the one value that hides the answer.
    """
    import torch

    for ln in canonicalize._all_layernorms(tiny):
        assert not torch.all(ln.weight == 1.0), (
            "LayerNorm gains are all exactly 1.0; this fixture is degenerate "
            "and will report residual rotation as a symmetry when it is not")
        assert not torch.all(ln.bias == 0.0), (
            "LayerNorm biases are all exactly 0.0; this fixture is degenerate")


# ---------------------------------------------------------------------------
# 1. THE TRIPWIRE
# ---------------------------------------------------------------------------


@requires_torch
def test_validate_architecture_accepts_the_model_it_was_written_for(tiny):
    assert canonicalize.validate_architecture(tiny, TINY) is TINY


@requires_torch
def test_validate_architecture_accepts_real_gpt2_against_the_shipped_spec(real_gpt2):
    """The default ArchSpec must describe the only real model in this repo."""
    assert canonicalize.validate_architecture(real_gpt2) is GPT2_124M
    assert GPT2_124M.n_params == 124_439_808, (
        "the shipped ArchSpec must imply the parameter count the config "
        "records, or the tripwire is validating against the wrong model")


@requires_torch
@pytest.mark.parametrize("field,bad", [
    ("n_layer", 3), ("n_head", 3), ("n_embd", 24),
    ("n_positions", 64), ("vocab_size", 128),
])
def test_validate_architecture_rejects_every_wrong_size(tiny, field, bad):
    """Each size mismatch must fail, and the message must name the field."""
    spec = ArchSpec(**{**TINY.__dict__, field: bad})
    with pytest.raises(CanonicalizeError) as exc:
        canonicalize.validate_architecture(tiny, spec)
    assert field in str(exc.value), (
        f"the error must name {field}, or you cannot tell what mismatched")


@requires_torch
def test_validate_architecture_rejects_untied_embeddings(tiny):
    """Untying changes the symmetry group; it must not be canonicalized here."""
    import torch

    model = copy.deepcopy(tiny)
    model.lm_head.weight = torch.nn.Parameter(
        model.transformer.wte.weight.detach().clone())
    with pytest.raises(CanonicalizeError) as exc:
        canonicalize.validate_architecture(model, TINY)
    message = str(exc.value)
    assert "tied" in message.lower()
    assert "orthogonal" in message, (
        "the error must explain WHY tying matters -- it is what makes the "
        "output projection the transpose of the input embedding")


@requires_torch
def test_validate_architecture_rejects_rotary_position_embeddings(tiny):
    """The amendment's requirement: rotary must be named as the reason.

    Rotary pins the basis inside each head, collapsing the GL(head_dim) group
    this module exploits. A canonicalizer built on the full group would run on
    such a model and be meaningless.
    """
    model = copy.deepcopy(tiny)
    model.config.rope_theta = 10000.0
    with pytest.raises(CanonicalizeError) as exc:
        canonicalize.validate_architecture(model, TINY)
    message = str(exc.value)
    assert "rope_theta" in message
    assert "GL(head_dim)" in message, (
        "the error must name the group that rotary would collapse")
    assert "SILENTLY WRONG" in message


@requires_torch
def test_validate_architecture_rejects_a_rotary_submodule(tiny):
    """A config flag is not the only way rotary arrives."""
    import torch

    model = copy.deepcopy(tiny)
    model.transformer.h[0].attn.rotary_emb = torch.nn.Identity()
    with pytest.raises(CanonicalizeError) as exc:
        canonicalize.validate_architecture(model, TINY)
    assert "rotary" in str(exc.value).lower()


@requires_torch
def test_validate_architecture_rejects_a_non_absolute_position_type(tiny):
    model = copy.deepcopy(tiny)
    model.config.position_embedding_type = "relative_key"
    with pytest.raises(CanonicalizeError) as exc:
        canonicalize.validate_architecture(model, TINY)
    assert "absolute" in str(exc.value)


@requires_torch
def test_validate_architecture_rejects_linear_in_place_of_conv1d(tiny):
    """nn.Linear stores (out, in); Conv1D stores (in, out). Swapping them
    makes every axis in this module hit the wrong side while still running."""
    import torch

    model = copy.deepcopy(tiny)
    block = model.transformer.h[0]
    replacement = torch.nn.Linear(TINY.n_embd, 3 * TINY.n_embd)
    block.attn.c_attn = replacement
    with pytest.raises(CanonicalizeError) as exc:
        canonicalize.validate_architecture(model, TINY)
    message = str(exc.value)
    assert "Conv1D" in message
    assert "TRANSPOSE" in message, (
        "the error must state the layout difference, which is the whole reason "
        "this check exists")


class _LinearLayoutStandIn:
    """A model with nn.Linear projections and NO .transformer attribute.

    Fails both the layout check and the attribute-path check. Which error comes
    back is precisely what the ordering test below pins.
    """

    def __new__(cls):
        import torch
        from torch import nn

        class _Attn(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.c_attn = nn.Linear(d, 3 * d)
                self.c_proj = nn.Linear(d, d)

        class _Block(nn.Module):
            def __init__(self, d):
                super().__init__()
                self.attn = _Attn(d)

        class _StandIn(nn.Module):
            def __init__(self, d=16):
                super().__init__()
                self.h = nn.ModuleList([_Block(d)])

        return _StandIn()


@requires_torch
def test_the_layout_check_runs_before_any_attribute_path_check():
    """CONTRACT CHANGE, pinned. The ordering here was changed deliberately.

    Run against a real nn.Linear model, the previous order refused on its first
    check -- "no .transformer attribute" -- and never reached the layout check.
    That refusal was correct by accident of ordering rather than by design, and
    its message invited someone to rename an attribute when the real objection
    is that every axis is transposed. It is the exact failure shape S55
    describes: a check reporting something plausible and adjacent to the real
    problem.

    Structural layout is now checked FIRST, by scanning module names rather
    than walking a fixed path, so a model with different attribute naming still
    gets the layout objection.
    """
    with pytest.raises(CanonicalizeError) as exc:
        canonicalize.validate_architecture(_LinearLayoutStandIn(), TINY)
    message = str(exc.value)
    assert "nn.Linear" in message or "torch.nn.Linear" in message
    assert "TRANSPOSE" in message, (
        "the layout objection must name the transposition explicitly")
    assert "c_attn" in message, "and name which module it found"
    assert ".transformer" not in message.split("\n")[0], (
        "the FIRST line must be the layout objection, not a complaint about "
        "the attribute path -- that ordering is the whole point of this test")


@requires_torch
def test_the_layout_check_says_a_rename_will_not_fix_it():
    """The message has to close off the wrong fix, not merely state the right
    diagnosis."""
    with pytest.raises(CanonicalizeError) as exc:
        canonicalize.validate_architecture(_LinearLayoutStandIn(), TINY)
    message = str(exc.value)
    assert "renaming an attribute" in message
    assert "WOULD NOT CRASH" in message, (
        "the message must say the failure is silent, which is why it matters")


@requires_torch
def test_validate_architecture_rejects_an_unfused_c_attn(tiny):
    """c_attn must be the fused QKV tensor of width 3*n_embd."""
    from transformers.pytorch_utils import Conv1D

    model = copy.deepcopy(tiny)
    model.transformer.h[0].attn.c_attn = Conv1D(TINY.n_embd, TINY.n_embd)
    with pytest.raises(CanonicalizeError) as exc:
        canonicalize.validate_architecture(model, TINY)
    assert "FUSED" in str(exc.value)


@requires_torch
def test_validate_architecture_rejects_a_non_gelu_activation(tiny):
    """FFN scaling is dropped BECAUSE the activation is GELU. Change the
    activation and that conclusion stops holding."""
    model = copy.deepcopy(tiny)
    model.config.activation_function = "relu"
    with pytest.raises(CanonicalizeError) as exc:
        canonicalize.validate_architecture(model, TINY)
    assert "positively" in str(exc.value)


@requires_torch
def test_validate_architecture_rejects_a_wrong_parameter_count(tiny):
    """The last line of defence: sizes agree but the model is not the model."""
    spec = ArchSpec(**{**TINY.__dict__})
    model = copy.deepcopy(tiny)
    model.transformer.h = model.transformer.h[:1]   # sizes lie, count does not
    with pytest.raises(CanonicalizeError):
        canonicalize.validate_architecture(model, spec)


# ---------------------------------------------------------------------------
# 2. THE LAYOUT CONTRACT
# ---------------------------------------------------------------------------


def test_qkv_offsets_partition_the_fused_tensor_without_overlap():
    assert qkv_offset("q", TINY) == 0
    assert qkv_offset("k", TINY) == TINY.n_embd
    assert qkv_offset("v", TINY) == 2 * TINY.n_embd
    with pytest.raises(CanonicalizeError):
        qkv_offset("o", TINY)


def test_head_columns_never_cross_a_qkv_boundary():
    """The fault that produces a running, lying model: a head slice that
    starts inside Q and ends inside K."""
    for which in ("q", "k", "v"):
        base = qkv_offset(which, TINY)
        for h in range(TINY.n_head):
            s = head_columns(which, h, TINY)
            assert s.stop - s.start == TINY.head_dim
            assert base <= s.start < s.stop <= base + TINY.n_embd, (
                f"{which} head {h} slice {s} leaves its own third of c_attn")


def test_head_columns_tile_each_third_exactly_once():
    for which in ("q", "k", "v"):
        covered = []
        for h in range(TINY.n_head):
            s = head_columns(which, h, TINY)
            covered.extend(range(s.start, s.stop))
        expected = list(range(qkv_offset(which, TINY),
                              qkv_offset(which, TINY) + TINY.n_embd))
        assert covered == expected, f"{which} heads do not tile their third"


def test_head_rows_of_out_proj_tile_the_input_axis_exactly_once():
    covered = []
    for h in range(TINY.n_head):
        s = head_rows_of_out_proj(h, TINY)
        covered.extend(range(s.start, s.stop))
    assert covered == list(range(TINY.n_embd))


def test_head_indices_out_of_range_are_rejected():
    for bad in (-1, TINY.n_head):
        with pytest.raises(CanonicalizeError):
            head_columns("q", bad, TINY)
        with pytest.raises(CanonicalizeError):
            head_rows_of_out_proj(bad, TINY)


def test_archspec_parameter_count_matches_the_real_gpt2_figure():
    """Arithmetic only -- no torch needed, so this runs in the base venv."""
    assert GPT2_124M.n_params == 124_439_808
    assert GPT2_124M.head_dim == 64


def test_every_candidate_has_a_pre_registered_float64_tolerance():
    """A candidate with no registered tolerance could not be judged."""
    for cls in ALL_SYMMETRIES:
        assert cls().name in TOL_F64, (
            f"{cls().name} has no pre-registered float64 tolerance; it cannot "
            "be measured against anything")
        assert cls().name in F32_DIAGNOSTIC


def test_the_float64_tolerances_and_collapse_factor_are_as_pre_registered():
    """These three numbers were fixed in the approved plan before anything was
    measured, and are not adjustable. Pinned here so a later edit is a test
    failure rather than a silent loosening of the study's ruler."""
    assert COLLAPSE_FACTOR == 1e4
    assert TOL_F64["head_permutation"] == 1e-12
    assert TOL_F64["layernorm_gain_rescale"] == 1e-12
    assert TOL_F64["head_internal_transform"] == 1e-10


def test_the_float32_noise_floor_is_recorded_as_a_measurement_not_a_threshold():
    """S43. The pre-registered float32 bound of 1e-06 lies INSIDE this
    architecture's own measured float32 noise floor, so it never discriminated
    anything and was demoted rather than moved. This test pins that
    relationship: if the recorded floor ever stops straddling the old bound,
    the reason for the demotion has changed and needs revisiting."""
    low, high = MEASURED_F32_NOISE_FLOOR
    assert low < 1e-6 < high, (
        "the demotion of the float32 bound rests on 1e-06 sitting inside the "
        f"measured noise floor {MEASURED_F32_NOISE_FLOOR}")


def test_symmetry_lookup_by_name_round_trips():
    for cls in ALL_SYMMETRIES:
        assert symmetry_by_name(cls().name) is cls
    with pytest.raises(CanonicalizeError):
        symmetry_by_name("no_such_symmetry")


def test_apply_before_sample_is_refused():
    """Silently doing nothing would make every downstream test vacuous."""
    for cls in ALL_SYMMETRIES:
        with pytest.raises(CanonicalizeError) as exc:
            cls().apply(object(), TINY)
        assert "sample()" in str(exc.value)


# ---------------------------------------------------------------------------
# 3. EQUIVALENCE, PER CANDIDATE, IN ISOLATION
# ---------------------------------------------------------------------------


@requires_torch
def test_the_harness_itself_reports_exactly_zero_for_an_untouched_copy(tiny):
    """The null control. If a deepcopy differs at all, every number below is
    measuring the harness rather than the symmetry."""
    d = canonicalize.logit_difference(tiny, copy.deepcopy(tiny),
                                      probe_tokens(TINY))
    assert d.max_abs == 0.0, (
        f"an untouched copy differs by {d.max_abs}; the measurement harness "
        "is adding error and nothing below can be trusted")


@requires_torch
@pytest.mark.parametrize("name", CONFIRMED_SYMMETRIES)
def test_confirmed_symmetry_leaves_the_logits_unchanged(name):
    """Equivalence, in isolation, against the float64 tolerance registered in
    the approved plan before any measurement. That bound is untouched.

    The float32 number is NOT asserted against a bound here -- only against the
    architecture's measured noise floor, as a sanity check that the error looks
    like ordinary reassociation rather than something structural.
    """
    result = canonicalize.measure_equivalence(
        tiny_builder, TINY, symmetry_by_name(name), seed=12345)
    assert result.f64.max_rel <= result.tol_f64, (
        f"{name} exceeds its pre-registered float64 tolerance: "
        f"{result.f64.max_rel:.3e} > {result.tol_f64:.0e}")
    assert result.f32.max_rel <= MEASURED_F32_NOISE_FLOOR[1], (
        f"{name} float32 error {result.f32.max_rel:.3e} is above the measured "
        f"noise floor {MEASURED_F32_NOISE_FLOOR[1]:.3e}; that is a diagnostic "
        "signal, not a tolerance failure, but it wants explaining")


@requires_torch
@pytest.mark.parametrize("name", CONFIRMED_SYMMETRIES)
def test_confirmed_symmetry_error_collapses_in_float64(name):
    """THE DISCRIMINATOR. Exact maths shows float32-epsilon error in float32
    and many orders less in float64. This is the criterion that cannot be
    passed by loosening a tolerance."""
    result = canonicalize.measure_equivalence(
        tiny_builder, TINY, symmetry_by_name(name), seed=12345)
    assert result.collapse >= COLLAPSE_FACTOR, (
        f"{name} error did not collapse in float64 (factor "
        f"{result.collapse:.2e} < {COLLAPSE_FACTOR:.0e}); that signature means "
        "the maths is wrong, not that the floats are noisy")


@requires_torch
@pytest.mark.parametrize("name", CONFIRMED_NOT_SYMMETRIES)
def test_dropped_candidate_is_not_a_symmetry_and_does_not_collapse(name):
    """The three drops, asserted as drops.

    Each fails by five to six orders of magnitude AND shows no float64
    collapse. If one of these ever starts passing, the architecture changed
    underneath this module and the recipe is no longer valid.
    """
    result = canonicalize.measure_equivalence(
        tiny_builder, TINY, symmetry_by_name(name), seed=12345)
    assert not result.passes, (
        f"{name} is recorded as NOT a symmetry but passed the criterion; "
        "the drop list is stale and the recipe may be wrong")
    assert result.f32.max_rel > 1e3 * MEASURED_F32_NOISE_FLOOR[1], (
        f"{name} float32 error {result.f32.max_rel:.3e} is close to the noise "
        "floor; a drop this marginal needs re-examining rather than trusting")
    assert result.collapse < COLLAPSE_FACTOR, (
        f"{name} error collapsed by {result.collapse:.2e} in float64, which "
        "is the signature of exact maths -- this candidate may have been "
        "dropped in error")


@requires_torch
def test_every_candidate_is_classified_exactly_once():
    """No candidate may sit outside both lists and go unjudged."""
    names = {cls().name for cls in ALL_SYMMETRIES}
    classified = set(CONFIRMED_SYMMETRIES) | set(CONFIRMED_NOT_SYMMETRIES)
    assert names == classified, (
        f"unclassified candidates: {names ^ classified}")
    assert not set(CONFIRMED_SYMMETRIES) & set(CONFIRMED_NOT_SYMMETRIES)


@requires_torch
def test_residual_rotation_fails_because_of_layernorm_not_because_of_tying():
    """The reasoning, pinned as a test rather than left in a comment.

    If tying were the obstruction, residual PERMUTATION would fail too -- a
    permutation is orthogonal, and tying treats all orthogonal maps alike.
    Permutation passes and rotation fails, which is only consistent with the
    obstruction being LayerNorm's per-channel gain.
    """
    rot = canonicalize.measure_equivalence(
        tiny_builder, TINY, symmetry_by_name("residual_rotation"), seed=12345)
    perm = canonicalize.measure_equivalence(
        tiny_builder, TINY, symmetry_by_name("residual_permutation"), seed=12345)
    assert not rot.passes and rot.collapse < COLLAPSE_FACTOR
    assert perm.collapse >= COLLAPSE_FACTOR, (
        "residual permutation must survive: it is orthogonal, so tying "
        "supplies its inverse, and a permutation conjugates LayerNorm's "
        "diagonal gain to another diagonal")


@requires_torch
def test_a_symmetry_applied_twice_from_one_draw_is_reproducible():
    """sample() and apply() are split so the SAME transform can be applied to
    two models. If apply() re-drew, every round-trip test would silently
    compare two different transforms."""
    import torch

    a, b = tiny_builder(), tiny_builder()
    sym = symmetry_by_name("head_permutation")().sample(a, TINY, 999)
    sym.apply(a, TINY)
    sym.apply(b, TINY)          # same object, same params, second model
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), f"{n} differs between two applications"


# ---------------------------------------------------------------------------
# the zero-gradient argument, measured rather than asserted
# ---------------------------------------------------------------------------


@requires_torch
@pytest.mark.parametrize("name", [
    "key_bias_shift", "value_bias_shift", "layernorm_gain_rescale",
    "head_internal_transform",
])
def test_the_loss_is_flat_along_a_confirmed_gauge_direction(name):
    """If the loss is exactly invariant along a direction, its gradient there
    is exactly zero -- so Adam's second moment is zero, the coordinate never
    updates, and it stays at its initialization value for the whole run.

    Measured as the cosine between the loss gradient and the symmetry's
    tangent direction. The bias shifts are LINEAR in the parameters so the
    cosine is at machine zero; the curved ones show a small second-order
    residual from the finite-difference tangent, not from inexactness.
    """
    result = canonicalize.gauge_gradient_alignment(
        tiny_builder, TINY, symmetry_by_name(name), seed=4242)
    assert result["continuous"] is True
    assert abs(result["cosine"]) < 1e-5, (
        f"{name} gauge direction has gradient cosine "
        f"{result['cosine']:.3e}; the loss is not flat along it, so the "
        "zero-gradient argument does not hold for it")


@requires_torch
@pytest.mark.parametrize("name", CONFIRMED_NOT_SYMMETRIES)
def test_the_loss_is_not_flat_along_a_dropped_candidates_direction(name):
    """The converse. A direction that is not a symmetry has real gradient
    along it, which is why the three drops matter: those coordinates DO move
    during training."""
    result = canonicalize.gauge_gradient_alignment(
        tiny_builder, TINY, symmetry_by_name(name), seed=4242)
    assert abs(result["cosine"]) > 1e-5, (
        f"{name} is recorded as not a symmetry, yet the loss appears flat "
        f"along it (cosine {result['cosine']:.3e})")


@requires_torch
def test_discrete_symmetries_report_no_gauge_direction():
    """A permutation has no tangent direction, so the zero-gradient argument
    does not apply to it and must not be silently claimed."""
    for name in ("residual_permutation", "head_permutation",
                 "ffn_neuron_permutation"):
        result = canonicalize.gauge_gradient_alignment(
            tiny_builder, TINY, symmetry_by_name(name), seed=4242)
        assert result["continuous"] is False
        assert result["cosine"] is None
        assert "does not apply" in result["note"]


# ---------------------------------------------------------------------------
# 4. THE ORDERING CONTRACT
#
# burst_match.gradient_parameters() is positional and filters requires_grad;
# this module is name-keyed. Every cosine in 8b-iv rests on those two agreeing.
# ---------------------------------------------------------------------------


@requires_torch
def test_named_parameters_matches_gradient_parameters_by_identity_on_tiny(tiny):
    from burst_match import gradient_parameters

    named = list(tiny.named_parameters())
    positional = gradient_parameters(tiny)
    assert len(named) == len(positional)
    for i, ((name, p), q) in enumerate(zip(named, positional)):
        assert p is q, (
            f"index {i} ({name}): named_parameters() and "
            "gradient_parameters() disagree by identity")


@requires_torch
def test_named_parameters_matches_gradient_parameters_by_identity_on_gpt2(real_gpt2):
    """The one that matters. If this fails, every cosine in
    docs/measurements/8b-iv-gradient-direction.json is computed over a
    permutation of one of the vectors and is silently meaningless."""
    from burst_match import gradient_parameters

    named = list(real_gpt2.named_parameters())
    positional = gradient_parameters(real_gpt2)
    assert len(named) == 148, (
        f"GPT-2 should expose 148 parameter tensors, got {len(named)}")
    assert len(positional) == len(named)
    for i, ((name, p), q) in enumerate(zip(named, positional)):
        assert p is q, (
            f"index {i} ({name}): the name-keyed and positional orderings "
            "disagree; 8b-iv's cosines are invalid")
    assert sum(p.numel() for p in positional) == GPT2_124M.n_params


@requires_torch
def test_the_tie_is_what_makes_the_two_orderings_agree(real_gpt2):
    """Both orderings drop lm_head.weight, and they drop it for the same
    reason: it is the same tensor object as wte.weight."""
    assert real_gpt2.lm_head.weight is real_gpt2.transformer.wte.weight
    deduped = [n for n, _ in real_gpt2.named_parameters()]
    raw = [n for n, _ in real_gpt2.named_parameters(remove_duplicate=False)]
    assert "lm_head.weight" not in deduped
    assert "lm_head.weight" in raw
    assert len(raw) - len(deduped) == 1


@requires_torch
def test_the_committed_8b_iv_profile_length_matches_the_live_ordering(real_gpt2):
    """The committed results file records one norm per parameter tensor. If
    that count ever stops matching, the file describes a different model."""
    import json

    from burst_match import gradient_parameters

    path = REPO_ROOT / "docs" / "measurements" / "8b-iv-gradient-direction.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    profile = data["per_arm"]["fluent-fabricated"]["per_tensor_norm_profile"]
    assert len(profile) == len(gradient_parameters(real_gpt2))
    assert data["parameters"] == GPT2_124M.n_params


# ---------------------------------------------------------------------------
# real GPT-2 equivalence
#
# The float32 numbers on the real model are NOT asserted against the
# pre-registered tolerance here, and that is deliberate rather than a
# concession. Phase 2 measured the float32 noise floor for this architecture at
# 4.6e-07 to 1.4e-06 across seeds, which means the registered 1e-06 bound sits
# INSIDE the noise band and discriminates random seeds rather than symmetries.
# No tolerance has been changed. The bound is frozen pending a decision by the
# plan owner, and what is asserted here is the collapse criterion, which
# separates the two groups by six orders of magnitude with no overlap.
# ---------------------------------------------------------------------------


@requires_torch
@pytest.mark.parametrize("name", CONFIRMED_SYMMETRIES)
def test_confirmed_symmetry_collapses_in_float64_on_real_gpt2(name, real_gpt2):
    def build():
        return copy.deepcopy(real_gpt2)

    result = canonicalize.measure_equivalence(
        build, GPT2_124M, symmetry_by_name(name), seed=12345)
    assert result.f64.max_rel <= result.tol_f64, (
        f"{name} exceeds its float64 tolerance on the real model: "
        f"{result.f64.max_rel:.3e} > {result.tol_f64:.0e}")
    assert result.collapse >= COLLAPSE_FACTOR, (
        f"{name} did not collapse on the real model (factor "
        f"{result.collapse:.2e})")


@requires_torch
@pytest.mark.parametrize("name", CONFIRMED_NOT_SYMMETRIES)
def test_dropped_candidate_stays_dropped_on_real_gpt2(name, real_gpt2):
    def build():
        return copy.deepcopy(real_gpt2)

    result = canonicalize.measure_equivalence(
        build, GPT2_124M, symmetry_by_name(name), seed=12345)
    assert result.collapse < COLLAPSE_FACTOR
    assert result.f32.max_rel > 1e-3, (
        f"{name} error on the real model is {result.f32.max_rel:.3e}, which is "
        "close to the float32 noise floor; a drop this marginal needs "
        "re-examining rather than trusting")
