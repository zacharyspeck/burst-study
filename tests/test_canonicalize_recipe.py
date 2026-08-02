"""Tests for the canonicalization recipe -- step 9, phase 3.

Five things this file guards.

1. FUNCTION PRESERVATION. Canonicalization rewrites weights; it must not change
   what the model computes. Every axis fault -- permuting c_attn's rows instead
   of its columns, crossing the Q/K/V boundary, transposing a factor -- shows up
   here and nowhere else.

2. ROUND TRIP, THE PRIMARY TEST. canonicalize(M) and canonicalize(sigma(M))
   must agree tensor by tensor. Function preservation alone is passed by a
   canonicalizer that does nothing at all; this one is not.

3. COMPOSITION. Each symmetry surviving in isolation does not mean the recipe
   handles them jointly. The failure mode is a later step undoing an earlier
   step's canonical form. Tested by sampling every recipe symmetry at once,
   across several seeds, reporting the worst per-tensor disagreement.

4. RECIPE ORDER. The order is part of the definition of canonical form, not a
   convenience. Permuted orders must break the round trip. Where one does NOT
   break it, that is recorded explicitly with the reason, because "the steps
   are more independent than assumed" and "the test is weaker than it looks"
   are very different findings.

5. THE FROZEN AXES. The vocabulary axis and the position axis are the model's
   output and input indexing. Reordering either would relabel what every logit
   refers to.

Mutation faults live in tests/test_canonicalize_mutations.py.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import canonicalize  # noqa: E402
from canonicalize import (  # noqa: E402
    DEFAULT_RECIPE,
    FROZEN_AXES,
    TINY,
    AbsorbLayerNormGains,
    CanonicalizeError,
    CanonicalizeHeadInternal,
    SortFFNNeurons,
    SortHeads,
    ZeroKeyBiasGauge,
    ZeroValueBiasGauge,
    assert_embedding_tie_preserved,
    assert_frozen_axes_unchanged,
    canonical_state_dict,
    canonicalize as run_canonicalize,
    probe_tokens,
    state_dict_difference,
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

#: The six symmetries DEFAULT_RECIPE actually quotients out.
#: residual_permutation is a CONFIRMED symmetry but is deliberately NOT here --
#: see test_residual_permutation_is_deliberately_not_quotiented.
RECIPE_SYMMETRIES = (
    "layernorm_gain_rescale",
    "head_permutation",
    "head_internal_transform",
    "ffn_neuron_permutation",
    "key_bias_shift",
    "value_bias_shift",
)

#: Round-trip agreement is exact arithmetic executed in float64. Anything above
#: this is a real disagreement, not rounding. Measured baseline is ~1.5e-15.
ROUND_TRIP_TOL = 1e-10


def fresh():
    """A generic tiny model in float64. float64 so a round-trip disagreement
    cannot be confused with float32 reassociation noise."""
    import torch
    return canonicalize.build_tiny_model(seed=0).to(dtype=torch.float64)


def scrambled(names, seed):
    model = fresh()
    for name in names:
        symmetry_by_name(name)().sample(model, TINY, seed).apply(model, TINY)
    return model


def round_trip_worst(names, seed, recipe=None):
    """Worst per-tensor disagreement between canon(M) and canon(sigma(M))."""
    a, b = fresh(), scrambled(names, seed)
    run_canonicalize(a, TINY, recipe=recipe)
    run_canonicalize(b, TINY, recipe=recipe)
    return state_dict_difference(canonical_state_dict(a), canonical_state_dict(b))


# ---------------------------------------------------------------------------
# 1. FUNCTION PRESERVATION
# ---------------------------------------------------------------------------


@requires_torch
def test_canonicalization_does_not_change_what_the_model_computes():
    """The recipe rewrites weights. If it changes the function, it is not a
    re-gauging at all and every distance computed after it is meaningless."""
    model = fresh()
    before = copy.deepcopy(model)
    run_canonicalize(model, TINY)
    diff = canonicalize.logit_difference(before, model, probe_tokens(TINY))
    assert diff.max_rel < 1e-12, (
        f"canonicalization changed the logits by {diff.max_rel:.3e}; it is "
        "rewriting the model rather than re-gauging it")


@requires_torch
def test_canonicalization_is_idempotent():
    """Canonicalizing an already-canonical model must be a no-op. If it is not,
    the 'canonical form' is not a fixed point and the round trip only appears
    to work because both sides move the same way."""
    model = fresh()
    run_canonicalize(model, TINY)
    once = canonical_state_dict(model)
    run_canonicalize(model, TINY)
    worst, name, _ = state_dict_difference(once, canonical_state_dict(model))
    assert worst < ROUND_TRIP_TOL, (
        f"a second canonicalization moved {name} by {worst:.3e}; canonical "
        "form is not a fixed point")


# ---------------------------------------------------------------------------
# 2. ROUND TRIP -- THE PRIMARY TEST
# ---------------------------------------------------------------------------


@requires_torch
@pytest.mark.parametrize("name", RECIPE_SYMMETRIES)
def test_round_trip_agrees_for_each_recipe_symmetry(name):
    worst, tensor, _ = round_trip_worst([name], seed=777)
    assert worst < ROUND_TRIP_TOL, (
        f"canon(M) and canon({name}(M)) disagree by {worst:.3e} at {tensor}; "
        "the recipe does not quotient this symmetry out")


@requires_torch
def test_residual_permutation_is_deliberately_not_quotiented():
    """A confirmed symmetry that the recipe deliberately leaves alone.

    Recorded as a test rather than a comment so the cost is visible. Same-seed
    twins diverging mid-training do not spontaneously permute residual
    channels, so quotienting it buys this study nothing -- but two
    INDEPENDENTLY initialised models would still show this gauge difference,
    and that limit should fail loudly if anyone assumes otherwise.
    """
    worst, _, _ = round_trip_worst(["residual_permutation"], seed=777)
    assert worst > 1e-3, (
        "residual permutation now round-trips, which means something added it "
        "to the recipe. That is a change to the definition of canonical form "
        "and the frozen-axis contract has to be revisited with it")


# ---------------------------------------------------------------------------
# 3. COMPOSITION -- all recipe symmetries at once, several seeds
# ---------------------------------------------------------------------------


@requires_torch
@pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
def test_round_trip_survives_all_recipe_symmetries_composed(seed):
    """Surviving in isolation does not mean surviving jointly. The failure this
    catches is a later step undoing an earlier step's canonical form."""
    worst, tensor, per_tensor = round_trip_worst(RECIPE_SYMMETRIES, seed)
    assert worst < ROUND_TRIP_TOL, (
        f"composed scramble at seed {seed}: worst disagreement {worst:.3e} at "
        f"{tensor}. Per-tensor top 3: "
        f"{sorted(per_tensor.items(), key=lambda kv: -kv[1])[:3]}")


@requires_torch
def test_composition_order_of_the_scramble_does_not_matter():
    """The recipe's order is load-bearing; the SCRAMBLE's order must not be.
    A canonical form that depended on how the model got out of gauge would not
    be a canonical form."""
    forward = scrambled(RECIPE_SYMMETRIES, 909)
    backward = scrambled(tuple(reversed(RECIPE_SYMMETRIES)), 909)
    run_canonicalize(forward, TINY)
    run_canonicalize(backward, TINY)
    worst, tensor, _ = state_dict_difference(
        canonical_state_dict(forward), canonical_state_dict(backward))
    assert worst < ROUND_TRIP_TOL, (
        f"canonical form depends on the order the scramble was applied in "
        f"({worst:.3e} at {tensor})")


# ---------------------------------------------------------------------------
# 4. NON-TRIVIALITY -- the identity function must not pass
# ---------------------------------------------------------------------------


@requires_torch
def test_canonicalization_actually_changes_a_non_canonical_model():
    """Test 2 alone is passed by a canonicalizer that does nothing."""
    model = fresh()
    before = canonical_state_dict(model)
    run_canonicalize(model, TINY)
    worst, _, _ = state_dict_difference(before, canonical_state_dict(model))
    assert worst > 1e-3, (
        f"canonicalization moved nothing (worst {worst:.3e}); it may be the "
        "identity function, which would pass the round trip vacuously")


@requires_torch
def test_each_step_postcondition_actually_holds():
    """Every step must reach the state it claims, not merely run."""
    import torch

    model = fresh()
    run_canonicalize(model, TINY)
    for i, block in enumerate(model.transformer.h):
        for ln_name in ("ln_1", "ln_2"):
            gain = getattr(block, ln_name).weight
            assert torch.allclose(gain, torch.ones_like(gain)), (
                f"block {i} {ln_name} gain is not all-ones after absorption")
        bias = block.attn.c_attn.bias
        for h in range(TINY.n_head):
            k = canonicalize.head_columns("k", h, TINY)
            v = canonicalize.head_columns("v", h, TINY)
            assert torch.all(bias[k] == 0), f"block {i} head {h}: b_K not zeroed"
            assert torch.all(bias[v] == 0), f"block {i} head {h}: b_V not zeroed"


@requires_torch
def test_ln_f_gain_is_deliberately_not_absorbed():
    """ln_f feeds lm_head, which is the tied embedding. Absorbing its gain
    would corrupt the input embedding -- and it is exactly why residual
    rotation is not a symmetry here (D17)."""
    import torch

    model = fresh()
    before = model.transformer.ln_f.weight.detach().clone()
    run_canonicalize(model, TINY)
    assert torch.equal(before, model.transformer.ln_f.weight), (
        "ln_f's gain was modified; that can only have been done by folding it "
        "into lm_head, which is the tied embedding")


@requires_torch
def test_head_singular_values_come_out_sorted_descending():
    """The head-internal step's canonical form.

    The Q factor is U sqrt(Sigma) over the AUGMENTED space, so the singular
    values are the squared column norms of the weight rows PLUS the squared
    bias entry. Leaving the bias term out gives a sequence that is not even
    monotonic -- measured (0.0010, 0.0099, 0.0082, 0.0057) against a true
    spectrum of (0.0509, 0.0099, 0.0082, 0.0057) -- because most of the largest
    singular value's mass sits in the bias row.
    """
    model = fresh()
    run_canonicalize(model, TINY)
    for block in model.transformer.h:
        c_attn = block.attn.c_attn
        for h in range(TINY.n_head):
            q = canonicalize.head_columns("q", h, TINY)
            s = ((c_attn.weight[:, q].detach() ** 2).sum(dim=0)
                 + c_attn.bias[q].detach() ** 2)
            values = s.tolist()
            assert values == sorted(values, reverse=True), (
                f"head {h} Q/K singular values are not descending: {values}")
            v = canonicalize.head_columns("v", h, TINY)
            s2 = (c_attn.weight[:, v].detach() ** 2).sum(dim=0).tolist()
            assert s2 == sorted(s2, reverse=True), (
                f"head {h} V/O singular values are not descending: {s2}")


# ---------------------------------------------------------------------------
# 5. RECIPE ORDER IS PART OF THE DEFINITION
# ---------------------------------------------------------------------------


def _reordered(indices):
    return tuple(DEFAULT_RECIPE[i] for i in indices)


@requires_torch
@pytest.mark.parametrize("label,order", [
    ("head_internal before gain absorption", [1, 2, 3, 0, 4, 5]),
    ("sort_heads before head_internal", [0, 1, 2, 4, 3, 5]),
    ("zero b_V after head_internal", [0, 1, 3, 2, 4, 5]),
    ("gain absorption last", [1, 2, 3, 4, 5, 0]),
    ("fully reversed", [5, 4, 3, 2, 1, 0]),
])
def test_a_permuted_recipe_order_breaks_the_round_trip(label, order):
    """Order is part of the definition of canonical form. Gain absorption
    rewrites c_attn's rows and the head-internal step reads c_attn; the sort
    steps compute keys from tensors the earlier steps rewrite. Running them in
    the wrong order produces a different -- and not canonical -- form."""
    worst, _, _ = round_trip_worst(RECIPE_SYMMETRIES, seed=777,
                                   recipe=_reordered(order))
    assert worst > ROUND_TRIP_TOL, (
        f"recipe order '{label}' still round-trips at {worst:.3e}. Either "
        "those steps are genuinely independent, or the round-trip test is "
        "weaker than it looks -- and which one it is needs establishing "
        "before the order is treated as free")


@requires_torch
def test_zeroing_the_key_bias_commutes_with_the_head_internal_step():
    """MEASURED INDEPENDENCE, recorded rather than glossed.

    This is the one permutation of DEFAULT_RECIPE that does NOT break the round
    trip, and the reason is specific: the Q/K invariant this module forms is
    [W_Q ; b_Q] W_K^T, which never reads b_K at all. So zeroing b_K before or
    after the head-internal step gives the identical canonical form.

    The step is still REQUIRED -- without it b_K differs between a model and
    its key-shifted twin, which the round trip does catch. Only its POSITION
    relative to the head-internal step is free. It is kept early because
    removing the gauge first measurably improves the worst-case
    singular-value gap (3.570e-05 to 1.068e-04 across 144 heads of real GPT-2),
    which is a numerical-quality argument, not a correctness one.
    """
    worst, _, _ = round_trip_worst(RECIPE_SYMMETRIES, seed=777,
                                   recipe=_reordered([0, 2, 3, 1, 4, 5]))
    assert worst < ROUND_TRIP_TOL, (
        "zeroing b_K after the head-internal step now breaks the round trip. "
        "That means the Q/K invariant has started depending on b_K, and the "
        "recorded reason this ordering is free no longer holds")


@requires_torch
def test_dropping_the_key_bias_step_entirely_does_break_the_round_trip():
    """The companion to the test above: free position, not free omission."""
    recipe = tuple(s for s in DEFAULT_RECIPE
                   if not isinstance(s, ZeroKeyBiasGauge))
    worst, _, _ = round_trip_worst(["key_bias_shift"], seed=777, recipe=recipe)
    assert worst > ROUND_TRIP_TOL, (
        "removing the key-bias step entirely still round-trips, so the step "
        "is doing nothing and the b_K gauge is not being removed")


@requires_torch
def test_the_default_recipe_is_the_expected_six_steps_in_order():
    """Pins the recipe itself. Changing it changes the definition of canonical
    form for the whole study, and should be a deliberate, visible act."""
    assert [s.name for s in DEFAULT_RECIPE] == [
        "absorb_layernorm_gains",
        "zero_key_bias_gauge",
        "zero_value_bias_gauge",
        "canonicalize_head_internal",
        "sort_heads",
        "sort_ffn_neurons",
    ]
    assert [type(s) for s in DEFAULT_RECIPE] == [
        AbsorbLayerNormGains, ZeroKeyBiasGauge, ZeroValueBiasGauge,
        CanonicalizeHeadInternal, SortHeads, SortFFNNeurons,
    ]


# ---------------------------------------------------------------------------
# 6. THE FROZEN AXES
# ---------------------------------------------------------------------------


def test_the_frozen_axes_are_the_output_and_position_indexings():
    """Arithmetic only -- runs without torch."""
    named = {(f.tensor, f.axis) for f in FROZEN_AXES}
    assert named == {("transformer.wte.weight", 0),
                     ("transformer.wpe.weight", 0)}
    for frozen in FROZEN_AXES:
        assert frozen.why, "a frozen axis must record why it is frozen"


@requires_torch
def test_canonicalization_leaves_the_vocabulary_and_position_axes_alone():
    model = fresh()
    before = canonical_state_dict(model)
    run_canonicalize(model, TINY)
    assert_frozen_axes_unchanged(before, canonical_state_dict(model))


@requires_torch
def test_the_current_recipe_leaves_the_embeddings_byte_identical():
    """Stronger than the per-axis contract, and true only because the current
    recipe touches neither embedding at all.

    Asserted separately from assert_frozen_axes_unchanged on purpose: that
    function states the weaker per-axis property so it stays correct if
    residual permutation is ever admitted. This test is what would have to
    change at that point, which makes the change visible.
    """
    import torch

    model = fresh()
    before = canonical_state_dict(model)
    run_canonicalize(model, TINY)
    after = canonical_state_dict(model)
    for name in ("transformer.wte.weight", "transformer.wpe.weight"):
        assert torch.equal(before[name], after[name]), (
            f"{name} was modified by canonicalization; the recipe is not "
            "supposed to touch the embeddings at all")


@requires_torch
def test_canonicalization_preserves_the_embedding_tie():
    model = fresh()
    run_canonicalize(model, TINY)
    assert_embedding_tie_preserved(model)


@requires_torch
def test_the_frozen_axis_check_actually_catches_a_reordered_vocabulary():
    """A contract nothing can violate is not a contract. Reorder the vocab
    axis by hand and confirm the check fires."""
    import torch

    model = fresh()
    before = canonical_state_dict(model)
    after = canonical_state_dict(model)
    perm = torch.randperm(TINY.vocab_size)
    after["transformer.wte.weight"] = after["transformer.wte.weight"][perm]
    with pytest.raises(CanonicalizeError) as exc:
        assert_frozen_axes_unchanged(before, after)
    assert "vocabulary axis" in str(exc.value)


@requires_torch
def test_the_frozen_axis_check_tolerates_a_residual_regauging():
    """The contract is per-AXIS. A permutation of the residual channels moves
    values inside each row but does not reorder the rows, so it must pass --
    otherwise the contract would block residual permutation from ever being
    admitted to the recipe for the wrong reason."""
    import torch

    model = fresh()
    before = canonical_state_dict(model)
    symmetry_by_name("residual_permutation")().sample(model, TINY, 5).apply(
        model, TINY)
    assert_frozen_axes_unchanged(before, canonical_state_dict(model))


@requires_torch
def test_the_tie_check_catches_an_untied_head():
    import torch

    model = fresh()
    model.lm_head.weight = torch.nn.Parameter(
        model.transformer.wte.weight.detach().clone())
    with pytest.raises(CanonicalizeError) as exc:
        assert_embedding_tie_preserved(model)
    assert "tie" in str(exc.value)


# ---------------------------------------------------------------------------
# 7. ALIGNMENT -- the other route, not a fallback
# ---------------------------------------------------------------------------


@requires_torch
@requires_scipy
def test_alignment_recovers_a_head_permutation_exactly():
    """Matching on the whole feature vector, solved with the Hungarian
    algorithm, must undo a head permutation exactly."""
    reference = fresh()
    model = copy.deepcopy(reference)
    symmetry_by_name("head_permutation")().sample(model, TINY, 31).apply(
        model, TINY)
    canonicalize.align_permutations_to(model, reference, TINY)
    worst, tensor, _ = state_dict_difference(
        canonical_state_dict(reference), canonical_state_dict(model))
    assert worst < ROUND_TRIP_TOL, (
        f"alignment did not recover the permutation: {worst:.3e} at {tensor}")


@requires_torch
@requires_scipy
def test_alignment_recovers_an_ffn_permutation_exactly():
    reference = fresh()
    model = copy.deepcopy(reference)
    symmetry_by_name("ffn_neuron_permutation")().sample(model, TINY, 31).apply(
        model, TINY)
    canonicalize.align_permutations_to(model, reference, TINY)
    worst, tensor, _ = state_dict_difference(
        canonical_state_dict(reference), canonical_state_dict(model))
    assert worst < ROUND_TRIP_TOL, (
        f"alignment did not recover the permutation: {worst:.3e} at {tensor}")


# ---------------------------------------------------------------------------
# 8. REAL GPT-2 -- the only model that actually exists in this repo
#
# Everything above runs on a 2-layer, 16-wide fixture. These confirm the recipe
# on the 12-layer, 768-wide model with trained weights, where the condition
# numbers are three orders larger and the round-trip residual is correspondingly
# looser (1e-12 rather than 1e-15).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_gpt2_f64():
    import torch
    try:
        return canonicalize.build_real_gpt2().to(dtype=torch.float64)
    except CanonicalizeError as exc:
        pytest.skip(f"GPT-2 unavailable: {exc}")


@requires_torch
def test_the_recipe_preserves_the_function_on_real_gpt2(real_gpt2_f64):
    model = copy.deepcopy(real_gpt2_f64)
    report = run_canonicalize(model, canonicalize.GPT2_124M)
    diff = canonicalize.logit_difference(
        real_gpt2_f64, model, probe_tokens(canonicalize.GPT2_124M))
    assert diff.max_rel < 1e-12, (
        f"canonicalization changed real GPT-2's logits by {diff.max_rel:.3e}")
    assert report.min_layernorm_gain > 0
    assert report.min_singular_gap > 0


@requires_torch
def test_the_recipe_actually_moves_real_gpt2(real_gpt2_f64):
    model = copy.deepcopy(real_gpt2_f64)
    before = canonical_state_dict(model)
    run_canonicalize(model, canonicalize.GPT2_124M)
    worst, _, _ = state_dict_difference(before, canonical_state_dict(model))
    assert worst > 1.0, (
        f"canonicalization barely moved real GPT-2 (worst {worst:.3e}); on a "
        "trained model with non-unit gains it should move a great deal")


@requires_torch
def test_composition_round_trip_holds_on_real_gpt2(real_gpt2_f64):
    """The headline check on the real model: every recipe symmetry at once."""
    arch = canonicalize.GPT2_124M
    a = copy.deepcopy(real_gpt2_f64)
    b = copy.deepcopy(real_gpt2_f64)
    for name in RECIPE_SYMMETRIES:
        symmetry_by_name(name)().sample(b, arch, 101).apply(b, arch)
    run_canonicalize(a, arch)
    run_canonicalize(b, arch)
    worst, tensor, _ = state_dict_difference(
        canonical_state_dict(a), canonical_state_dict(b))
    assert worst < 1e-9, (
        f"composed round trip on real GPT-2 disagrees by {worst:.3e} at "
        f"{tensor}")


@requires_torch
def test_the_frozen_axes_and_tie_survive_on_real_gpt2(real_gpt2_f64):
    model = copy.deepcopy(real_gpt2_f64)
    before = canonical_state_dict(model)
    run_canonicalize(model, canonicalize.GPT2_124M)
    assert_frozen_axes_unchanged(before, canonical_state_dict(model))
    assert_embedding_tie_preserved(model)


@requires_torch
@requires_scipy
def test_alignment_is_a_no_op_when_the_models_already_agree():
    reference = fresh()
    model = copy.deepcopy(reference)
    report = canonicalize.align_permutations_to(model, reference, TINY)
    for perm in report.head_permutations:
        assert list(perm) == sorted(perm) == list(range(TINY.n_head))
    worst, _, _ = state_dict_difference(
        canonical_state_dict(reference), canonical_state_dict(model))
    assert worst == 0.0
