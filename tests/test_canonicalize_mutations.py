"""Mutation tests for the canonicalization recipe -- step 9, phase 4.

A test suite that passes on broken code is worse than no test suite. This file
breaks the recipe on purpose, six different ways, and asserts that the other
tests catch each break -- AND records which kind of test catches it.

That last part is the point. If every fault were caught by the same check, the
taxonomy would be doing no work and five of the six checks could be deleted.
The table this file pins:

    fault                              caught by
    1  permute c_attn's row axis       function preservation
    2  permute across the QKV boundary function preservation
    3  swap the V and O factors        function preservation
    4  skip gain absorption            ROUND TRIP ONLY
    5  drop the SVD sign convention    ROUND TRIP ONLY
    6  drop the bias row               ROUND TRIP ONLY

Faults 4, 5 and 6 pass function preservation. They are the reason round trip is
the primary test: a canonicalizer can leave the model's behaviour untouched and
still fail to produce a canonical form, and only the round trip sees it.

The faults are defined HERE, not in the shipped module. `canonicalize` exposes
two override points -- CanonicalizeHeadInternal._query_factor and ._svd, plus
SortHeads._apply -- so a fault can be injected by subclassing rather than by
adding a broken-mode flag to production code.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import canonicalize  # noqa: E402
from canonicalize import (  # noqa: E402
    DEFAULT_RECIPE,
    TINY,
    AbsorbLayerNormGains,
    CanonicalizeError,
    CanonicalizeHeadInternal,
    SortHeads,
    _paired_svd,
    canonical_state_dict,
    canonicalize as run_canonicalize,
    head_columns,
    head_rows_of_out_proj,
    probe_tokens,
    state_dict_difference,
    symmetry_by_name,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None,
    reason="torch and transformers are optional; install .[measure]",
)

RECIPE_SYMMETRIES = (
    "layernorm_gain_rescale",
    "head_permutation",
    "head_internal_transform",
    "ffn_neuron_permutation",
    "key_bias_shift",
    "value_bias_shift",
)

#: Anything above this is a real disagreement. Baseline is ~1.5e-15.
DETECTION_FLOOR = 1e-10


# ---------------------------------------------------------------------------
# the six faults
# ---------------------------------------------------------------------------


@dataclass
class PermutesRowAxis(SortHeads):
    """FAULT 1. c_attn's rows are the RESIDUAL axis, not the head axis.
    Conv1D is (in, out), so the heads live in the columns."""

    name: str = "FAULT1_sort_heads_permutes_the_row_axis"

    def _apply(self, block, arch, order):
        weight = block.attn.c_attn.weight
        new = weight.detach().clone()
        for new_head, old_head in enumerate(order):
            new[head_rows_of_out_proj(new_head, arch), :] = \
                weight[head_rows_of_out_proj(old_head, arch), :]
        weight.copy_(new)


@dataclass
class CrossesQKVBoundary(SortHeads):
    """FAULT 2. Blocks of head_dim taken over the whole 3*n_embd output axis
    instead of within each third, so a 'head' straddles Q and K."""

    name: str = "FAULT2_sort_heads_crosses_the_qkv_boundary"

    def _apply(self, block, arch, order):
        c_attn = block.attn.c_attn
        dh = arch.head_dim
        weight, bias = c_attn.weight, c_attn.bias
        new_w, new_b = weight.detach().clone(), bias.detach().clone()
        for new_head, old_head in enumerate(order):
            new_w[:, new_head * dh:(new_head + 1) * dh] = \
                weight[:, old_head * dh:(old_head + 1) * dh]
            new_b[new_head * dh:(new_head + 1) * dh] = \
                bias[old_head * dh:(old_head + 1) * dh]
        weight.copy_(new_w)
        bias.copy_(new_b)


@dataclass
class SwapsValueAndOutputFactors(CanonicalizeHeadInternal):
    """FAULT 3. U and V exchanged when reconstructing the V/O pair. Both are
    (n_embd, head_dim) there, so it is shape-valid and silently wrong."""

    name: str = "FAULT3_head_internal_swaps_the_V_and_O_factors"

    def _svd(self, F, G):
        U, s, V = _paired_svd(F, G, sign_fix=True)
        return (V, s, U) if F.shape == G.shape else (U, s, V)


@dataclass
class DropsSignConvention(CanonicalizeHeadInternal):
    """FAULT 5. Each singular-vector pair may be negated together without
    changing the product, so with no convention the form is canonical only up
    to 2^head_dim sign choices."""

    name: str = "FAULT5_head_internal_drops_the_sign_convention"

    def _svd(self, F, G):
        return _paired_svd(F, G, sign_fix=False)


@dataclass
class DropsBiasRow(CanonicalizeHeadInternal):
    """FAULT 6. The weights-only Q/K invariant, discarding b_Q's row."""

    name: str = "FAULT6_head_internal_drops_the_bias_row"

    def _query_factor(self, w_q, b_q):
        return w_q


def _substitute(recipe, step_type, replacement):
    return tuple(replacement if type(step) is step_type else step
                 for step in recipe)


FAULTY_RECIPES = {
    "FAULT1_row_axis": _substitute(DEFAULT_RECIPE, SortHeads, PermutesRowAxis()),
    "FAULT2_across_qkv": _substitute(DEFAULT_RECIPE, SortHeads,
                                     CrossesQKVBoundary()),
    "FAULT3_swap_v_o": _substitute(DEFAULT_RECIPE, CanonicalizeHeadInternal,
                                   SwapsValueAndOutputFactors()),
    "FAULT4_skip_gain_absorption": tuple(
        s for s in DEFAULT_RECIPE if not isinstance(s, AbsorbLayerNormGains)),
    "FAULT5_no_sign_fix": _substitute(DEFAULT_RECIPE, CanonicalizeHeadInternal,
                                      DropsSignConvention()),
}

#: Faults that change what the model computes.
FUNCTION_BREAKING = ("FAULT1_row_axis", "FAULT2_across_qkv", "FAULT3_swap_v_o")
#: Faults that leave the model's behaviour untouched and only break canonicity.
CANONICITY_BREAKING = ("FAULT4_skip_gain_absorption", "FAULT5_no_sign_fix")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fresh(build=None):
    import torch
    build = canonicalize.build_tiny_model if build is None else build
    return build().to(dtype=torch.float64)


def function_preservation(recipe, build=None):
    """Relative logit change caused by canonicalizing. Should be ~1e-16."""
    model = fresh(build)
    before = copy.deepcopy(model)
    try:
        run_canonicalize(model, TINY, recipe=recipe)
    except CanonicalizeError:
        return float("inf")
    return canonicalize.logit_difference(
        before, model, probe_tokens(TINY)).max_rel


def round_trip(recipe, names=RECIPE_SYMMETRIES, seed=777, build=None):
    """Worst round-trip disagreement over `names`. inf if the recipe raised."""
    worst = 0.0
    for name in names:
        a, b = fresh(build), fresh(build)
        symmetry_by_name(name)().sample(b, TINY, seed).apply(b, TINY)
        try:
            run_canonicalize(a, TINY, recipe=recipe)
            run_canonicalize(b, TINY, recipe=recipe)
        except CanonicalizeError:
            return float("inf")
        value, _, _ = state_dict_difference(canonical_state_dict(a),
                                            canonical_state_dict(b))
        worst = max(worst, value)
    return worst


# ---------------------------------------------------------------------------
# the baseline, without which no fault result means anything
# ---------------------------------------------------------------------------


@requires_torch
def test_the_correct_recipe_passes_both_checks():
    """If the baseline did not pass, every 'fault detected' below could just be
    the suite failing on everything."""
    assert function_preservation(DEFAULT_RECIPE) < 1e-12
    assert round_trip(DEFAULT_RECIPE) < DETECTION_FLOOR


# ---------------------------------------------------------------------------
# every fault must be caught by something
# ---------------------------------------------------------------------------


@requires_torch
@pytest.mark.parametrize("label", sorted(FAULTY_RECIPES))
def test_every_injected_fault_is_caught_by_at_least_one_check(label):
    recipe = FAULTY_RECIPES[label]
    caught_by_function = function_preservation(recipe) > 1e-10
    caught_by_round_trip = round_trip(recipe) > DETECTION_FLOOR
    assert caught_by_function or caught_by_round_trip, (
        f"{label} was not caught by ANY check. A test suite that passes on "
        "broken code is worse than none")


@requires_torch
@pytest.mark.parametrize("label", FUNCTION_BREAKING)
def test_axis_faults_are_caught_by_function_preservation(label):
    """Faults 1-3 change what the model computes, so canonicalizing with them
    moves the logits. These are the 'runs and lies' faults."""
    moved = function_preservation(FAULTY_RECIPES[label])
    assert moved > 1e-10, (
        f"{label} left the logits unchanged at {moved:.3e}; an axis fault that "
        "does not move the output is not being exercised")


@requires_torch
@pytest.mark.parametrize("label", CANONICITY_BREAKING)
def test_canonicity_faults_pass_function_preservation_and_fail_round_trip(label):
    """THE TESTS THAT EARN ROUND TRIP ITS 'PRIMARY' LABEL.

    Fault 4 skips gain absorption and fault 5 drops the sign convention.
    Neither changes what the model computes -- both leave function
    preservation at float-noise level. Both destroy canonicity, and only the
    round trip sees it. If either of these were ever caught by function
    preservation instead, the fault would not be testing what it claims to.
    """
    recipe = FAULTY_RECIPES[label]
    moved = function_preservation(recipe)
    broke = round_trip(recipe)
    assert moved < 1e-12, (
        f"{label} changed the logits by {moved:.3e}. It is supposed to be "
        "function-preserving, so this fault is no longer isolating what only "
        "the round trip can see")
    assert broke > DETECTION_FLOOR, (
        f"{label} still round-trips at {broke:.3e}. This is the fault that "
        "proves the round trip earns its 'primary' label -- if it passes, the "
        "round-trip test is weaker than it looks and nothing else in phase 3 "
        "can be trusted")


@requires_torch
def test_skipping_gain_absorption_is_caught_specifically_by_the_gain_symmetry():
    """Fault 4, narrowed. It must fail on the LayerNorm gain symmetry in
    particular -- that is the gauge the skipped step exists to remove."""
    recipe = FAULTY_RECIPES["FAULT4_skip_gain_absorption"]
    broke = round_trip(recipe, names=["layernorm_gain_rescale"])
    assert broke > DETECTION_FLOOR, (
        f"skipping gain absorption still round-trips under a gain rescale "
        f"({broke:.3e}); the step is doing nothing")


# ---------------------------------------------------------------------------
# fault 6 -- the one that makes Amendment 3 load-bearing
# ---------------------------------------------------------------------------


@requires_torch
def test_dropping_the_bias_row_fails_the_round_trip_on_a_generic_model():
    """FAULT 6, and it fails for a more basic reason than degeneracy.

    The augmented Q/K invariant is [W_Q ; b_Q] W_K^T. Dropping the bias row
    does not merely weaken tie-breaking between near-equal singular values --
    it means b_Q is never transformed at all, so the query bias is left in
    whatever gauge it arrived in. The canonical form is incomplete, and it is
    incomplete on ANY model with a non-zero b_Q.

    The disagreement is therefore worst at c_attn.bias, which is the signature
    identifying this specific fault. It does not stop there: the head sort key
    is the invariant's spectrum, and the Q-side spectrum includes b_Q's
    contribution, so a stranded b_Q also gives the two models different sort
    keys and hence different head orderings. Measured, the damage reaches the
    weights at 1.8e-01 as well. Recorded because the propagation is the
    behaviour, not an accident to be asserted away.
    """
    recipe = _substitute(DEFAULT_RECIPE, CanonicalizeHeadInternal,
                         DropsBiasRow())
    a, b = fresh(), fresh()
    symmetry_by_name("head_internal_transform")().sample(b, TINY, 777).apply(
        b, TINY)
    run_canonicalize(a, TINY, recipe=recipe)
    run_canonicalize(b, TINY, recipe=recipe)
    worst, tensor, per_tensor = state_dict_difference(
        canonical_state_dict(a), canonical_state_dict(b))

    assert worst > DETECTION_FLOOR, (
        f"dropping the bias row still round-trips at {worst:.3e}; either "
        "Amendment 3 is not load-bearing or b_Q is zero in this fixture")
    assert "c_attn.bias" in tensor, (
        f"the worst disagreement is at {tensor}, expected a c_attn.bias. The "
        "signature of this fault is that b_Q is left un-canonicalized")
    bias_worst = max(v for k, v in per_tensor.items()
                     if k.endswith("c_attn.bias"))
    other_worst = max(v for k, v in per_tensor.items()
                      if not k.endswith("c_attn.bias"))
    assert bias_worst > other_worst, (
        f"c_attn.bias disagrees by {bias_worst:.3e} but something else "
        f"disagrees by {other_worst:.3e}; the stranded query bias should be "
        "the largest single disagreement this fault produces")


@requires_torch
def test_dropping_the_bias_row_fails_worse_on_a_near_degenerate_model():
    """The degeneracy argument, measured on the fixture built to expose it.

    build_degenerate_model gives each head a Q/K spectrum with two singular
    values a hair apart, so the weights-only SVD basis is nearly arbitrary in
    that subspace. The correct augmented recipe still round-trips there; the
    weights-only fault fails, and fails harder than on a generic model.
    """
    recipe = _substitute(DEFAULT_RECIPE, CanonicalizeHeadInternal,
                         DropsBiasRow())
    build = canonicalize.build_degenerate_model
    correct = round_trip(DEFAULT_RECIPE, names=["head_internal_transform"],
                         build=build)
    faulty = round_trip(recipe, names=["head_internal_transform"], build=build)
    assert correct < DETECTION_FLOOR, (
        f"the CORRECT recipe fails on the near-degenerate fixture "
        f"({correct:.3e}); the augmented invariant is supposed to split the "
        "near-degenerate pair, so this says it does not")
    assert faulty > DETECTION_FLOOR, (
        f"the weights-only fault survives the near-degenerate fixture "
        f"({faulty:.3e}); the degenerate case is not severe enough to "
        "demonstrate what Amendment 3 buys")


@requires_torch
def test_the_degenerate_fixture_really_is_near_degenerate():
    """A fixture that is not actually degenerate would make the test above
    pass for the wrong reason."""
    import torch

    model = canonicalize.build_degenerate_model().to(dtype=torch.float64)
    gaps = []
    for block in model.transformer.h:
        for h in range(TINY.n_head):
            q = head_columns("q", h, TINY)
            k = head_columns("k", h, TINY)
            w_q = block.attn.c_attn.weight[:, q].detach()
            w_k = block.attn.c_attn.weight[:, k].detach()
            _, s, _ = _paired_svd(w_q, w_k)
            gaps.append(float(canonicalize._relative_gaps(s).min()))
    assert max(gaps) < 1e-4, (
        f"the weights-only spectrum is not near-degenerate (smallest gap "
        f"across heads is {max(gaps):.3e}); the fixture is not exercising the "
        "case it was built for")


@requires_torch
def test_the_head_internal_step_refuses_a_spectrum_below_the_gap_floor():
    """Below the floor the SVD basis is arbitrary and there IS no canonical
    form. Refusing is the honest response; proceeding would emit plausible
    numbers forever.

    Exercised on the guard directly rather than through a model, because no
    model-level construction actually reaches the floor: asking
    build_degenerate_model for gap=0.0 still produces a COMPUTED relative gap
    of 4.077e-08 in float64, since forming W_Q from an orthogonal factor and
    then decomposing it does not preserve an exact tie. That is recorded rather
    than worked around -- the floor is a backstop for a genuinely rank-starved
    invariant, and the realistic hazard is the near-degenerate case above,
    which proceeds and silently disagrees.
    """
    import torch

    step = CanonicalizeHeadInternal()
    fine = torch.tensor([2.0, 1.0, 0.5, 0.25], dtype=torch.float64)
    step._check_spectrum(fine, "Q/K")          # must not raise

    too_close = torch.tensor([1.0, 1.0 - 1e-12, 0.5, 0.25],
                             dtype=torch.float64)
    with pytest.raises(CanonicalizeError) as exc:
        step._check_spectrum(too_close, "Q/K")
    message = str(exc.value)
    assert "singular values" in message
    assert "arbitrary" in message

    rank_starved = torch.tensor([1.0, 0.5, 0.25, 0.0], dtype=torch.float64)
    with pytest.raises(CanonicalizeError) as exc:
        step._check_spectrum(rank_starved, "Q/K")
    assert "rank deficient" in str(exc.value)
