"""S55 hardening: every named diagnostic recomputed by an independent route.

THE RISK THIS FILE EXISTS FOR.

Three times in this build a quantity was NAMED for what it was supposed to
measure and COMPUTED as something else, and each time the wrong number was
plausible, nothing crashed, and the number was then used to argue that
something was safe:

  S49  the head sort key was documented as the invariant's spectrum and
       computed as weight-row column norms, which omit the bias row's
       contribution. Not merely inaccurate -- not even monotonic.
  S53  the FFN sort margin was documented as the margin that decides a
       comparison and computed as the largest difference anywhere in the key
       tuple. Overstated by 59,000x, and that number appeared in a phase 3
       report as evidence the sort was comfortable.
  S48  the test fixture was documented as generic and was at exactly the
       structured values a fresh init produces, which made an entire amendment
       a no-op that still passed every test.

S55 records that nothing guarded against a fourth instance: all three were
caught by measuring something adjacent and being surprised. This file is that
guard.

TWO KINDS OF CHECK, and the second is the one that would have caught S53.

1. INDEPENDENT RECOMPUTATION. Each diagnostic is recomputed by a deliberately
   different route -- an explicit full SVD rather than the QR-plus-core-SVD
   shortcut, a direct scan rather than an accumulated minimum -- and the two
   must agree. This catches an implementation that drifted from its definition.

2. OPERATIONAL MEANING. A margin is not just a number, it is a THRESHOLD: the
   amount a key must move before the sort order flips. So it is tested by
   actually moving a key by slightly less and slightly more, and asserting the
   order holds and then breaks. This is what distinguishes a margin from an
   arbitrary difference, and it is exactly the check S53's metric would have
   failed while its recomputation-by-the-same-idea would have passed.

COMPLETENESS IS ENFORCED. Every field of CanonReport must appear either in
INDEPENDENT_CHECKS or in EXEMPT_FIELDS with a stated reason. Adding a new
diagnostic without an independent check fails
test_every_canonreport_field_is_either_checked_or_explicitly_exempt.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import canonicalize  # noqa: E402
from canonicalize import (  # noqa: E402
    SORT_ONLY_RECIPE,
    TINY,
    CanonReport,
    canonicalize as run_canonicalize,
    head_columns,
    head_rows_of_out_proj,
)

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None,
    reason="torch and transformers are optional; install .[measure]",
)

#: Agreement between two routes to the same quantity. Both run in float64 on
#: the same tensors, so a real disagreement is a definition mismatch, not
#: rounding.
AGREEMENT_RTOL = 1e-9


def fresh():
    import torch
    return canonicalize.build_tiny_model(seed=0).to(dtype=torch.float64)


# ---------------------------------------------------------------------------
# independent recomputations -- deliberately different routes
# ---------------------------------------------------------------------------


def _explicit_spectra(model, arch):
    """Every head's two invariant spectra, via an EXPLICIT full SVD.

    The module computes these by QR-ing each factor and taking the SVD of the
    small core, which is the cheap route. This forms the invariant matrix
    outright and decomposes it, which is the expensive one. They must agree;
    if they do not, the shortcut has drifted from what it claims to compute.
    """
    import torch

    out = []
    for block in model.transformer.h:
        c_attn, c_proj = block.attn.c_attn, block.attn.c_proj
        for h in range(arch.n_head):
            q, k, v = (head_columns(w, h, arch) for w in canonicalize.QKV)
            w_q = c_attn.weight[:, q].detach().double()
            b_q = c_attn.bias[q].detach().double()
            w_k = c_attn.weight[:, k].detach().double()
            w_v = c_attn.weight[:, v].detach().double()
            w_o = c_proj.weight[head_rows_of_out_proj(h, arch), :].detach().double()

            f_q = torch.cat([w_q, b_q.unsqueeze(0)], dim=0)
            qk = torch.linalg.svdvals(f_q @ w_k.transpose(0, 1))[:arch.head_dim]
            vo = torch.linalg.svdvals(w_v @ w_o)[:arch.head_dim]
            out.append((qk, vo))
    return out


def recompute_min_singular_gap(pre, post, arch) -> float:
    model = post          # invariants survive canonicalization unchanged
    worst = math.inf
    for qk, vo in _explicit_spectra(model, arch):
        for s in (qk, vo):
            gaps = (s[:-1] - s[1:]) / s[0]
            worst = min(worst, float(gaps.min()))
    return worst


def recompute_max_head_condition(pre, post, arch) -> float:
    model = post          # same: the spectra are invariants
    worst = 0.0
    for qk, vo in _explicit_spectra(model, arch):
        for s in (qk, vo):
            worst = max(worst, float(s[0] / s[-1]))
    return worst


def recompute_min_layernorm_gain(pre, post, arch) -> float:
    """Scanned off the PRE-canonicalization gains.

    Unlike the spectra, this one is destroyed by the step that measures it:
    absorption sets every ln_1/ln_2 gain to exactly 1.0, so the canonicalized
    model reports 1.0 and says nothing about what was absorbed. Which state a
    diagnostic describes is itself a thing that gets muddled, so every
    recomputation here takes both and picks explicitly.

    ln_f is excluded because it is deliberately not absorbed.
    """
    model = pre
    worst = math.inf
    for block in model.transformer.h:
        for ln_name in ("ln_1", "ln_2"):
            gain = getattr(block, ln_name).weight.detach()
            worst = min(worst, float(gain.abs().min()))
    return worst


def _ffn_keys(block):
    bias = block.mlp.c_fc.bias.detach()
    in_norm = block.mlp.c_fc.weight.detach().norm(dim=0)
    out_norm = block.mlp.c_proj.weight.detach().norm(dim=1)
    return list(zip(bias.tolist(), in_norm.tolist(), out_norm.tolist()))


def _head_keys(block, arch):
    return [canonicalize.SortHeads._key(block, arch, h)
            for h in range(arch.n_head)]


def _deciding_margin(keys) -> float:
    """Scan adjacent pairs of the SORTED keys for the first differing
    component. Written as a plain scan rather than reusing the module's helper,
    so the two are genuinely independent implementations of one definition."""
    ordered = sorted(keys)
    worst = math.inf
    for a, b in zip(ordered, ordered[1:]):
        for x, y in zip(a, b):
            if x != y:
                worst = min(worst, abs(x - y))
                break
        else:
            worst = 0.0
    return worst


def recompute_min_ffn_sort_margin(pre, post, arch) -> float:
    model = post          # the sort reads the post-head-internal state
    return min(_deciding_margin(_ffn_keys(b)) for b in model.transformer.h)


def recompute_min_head_sort_margin(pre, post, arch) -> float:
    model = post          # likewise
    return min(_deciding_margin(_head_keys(b, arch))
               for b in model.transformer.h)


#: diagnostic name -> independent recomputation
INDEPENDENT_CHECKS = {
    "min_layernorm_gain": recompute_min_layernorm_gain,
    "min_singular_gap": recompute_min_singular_gap,
    "max_head_condition": recompute_max_head_condition,
    "min_head_sort_margin": recompute_min_head_sort_margin,
    "min_ffn_sort_margin": recompute_min_ffn_sort_margin,
}

#: Fields that are records rather than measurements, with the reason. A field
#: parked here is a deliberate statement that there is nothing to cross-check,
#: not an omission.
EXEMPT_FIELDS = {
    "steps": "a list of names, not a measurement",
    "head_conditions": "the per-head vector max_head_condition reduces; its "
                       "reduction is checked instead",
    "head_orders": "an ordering the step chose, not a derived quantity",
    "ffn_orders": "an ordering the step chose, not a derived quantity",
    "aligned_to_reference": "a boolean record of which protocol half ran",
}


# ---------------------------------------------------------------------------
# completeness -- the structural guard against a fourth instance
# ---------------------------------------------------------------------------


def test_every_canonreport_field_is_either_checked_or_explicitly_exempt():
    """THE GUARD. Adding a diagnostic without an independent recomputation
    fails here, which is the only thing standing between this build and a
    fourth instance of the S55 pattern."""
    fields = {f.name for f in dataclasses.fields(CanonReport)}
    covered = set(INDEPENDENT_CHECKS) | set(EXEMPT_FIELDS)
    missing = fields - covered
    stale = covered - fields
    assert not missing, (
        f"CanonReport fields with no independent check and no exemption: "
        f"{sorted(missing)}. Every named diagnostic must be recomputable by a "
        "second route, or explicitly recorded as not a measurement -- see S55")
    assert not stale, (
        f"checks or exemptions naming fields that no longer exist: "
        f"{sorted(stale)}")


def test_every_exemption_states_a_reason():
    for name, reason in EXEMPT_FIELDS.items():
        assert reason and len(reason) > 15, (
            f"{name} is exempt without a stated reason; an exemption with no "
            "reason is indistinguishable from an oversight")


# ---------------------------------------------------------------------------
# 1. independent recomputation must agree
# ---------------------------------------------------------------------------


@requires_torch
@pytest.mark.parametrize("field", sorted(INDEPENDENT_CHECKS))
def test_reported_diagnostic_agrees_with_an_independent_recomputation(field):
    """Each diagnostic, recomputed by a deliberately different route.

    Run against SORT_ONLY_RECIPE because that is the recipe that populates
    every diagnostic -- the shipped recipe replaces the FFN sort with matching
    and so never computes an FFN sort margin. The margins remain properties of
    the MODEL, which is why they are still worth cross-checking.
    """
    import copy

    model = fresh()
    pre = copy.deepcopy(model)
    report = run_canonicalize(model, TINY, recipe=SORT_ONLY_RECIPE)
    reported = getattr(report, field)

    # Both states are handed over and each recomputation picks explicitly.
    # Spectra survive canonicalization; LayerNorm gains do not -- absorption
    # sets them all to 1.0 -- so "which model does this describe" is part of
    # each diagnostic's definition rather than an implementation detail.
    recomputed = INDEPENDENT_CHECKS[field](pre, model, TINY)

    assert math.isfinite(reported), f"{field} was never populated"
    assert math.isfinite(recomputed), f"{field} recomputation produced no value"
    scale = max(abs(reported), abs(recomputed), 1e-300)
    assert abs(reported - recomputed) / scale < AGREEMENT_RTOL, (
        f"{field} reported {reported:.6e} but an independent route computes "
        f"{recomputed:.6e}. The implementation has drifted from the quantity "
        "the field is named for -- see S49, S53, S55")


# ---------------------------------------------------------------------------
# 2. OPERATIONAL MEANING -- the check that would have caught S53
# ---------------------------------------------------------------------------


def _min_margin_layer_and_pair(model, arch):
    """Which layer holds the tightest FFN pair, and the two neuron indices."""
    best = (math.inf, None, None, None)
    for layer, block in enumerate(model.transformer.h):
        keys = _ffn_keys(block)
        order = sorted(range(len(keys)), key=lambda j: keys[j])
        for lo, hi in zip(order, order[1:]):
            a, b = keys[lo], keys[hi]
            for x, y in zip(a, b):
                if x != y:
                    if abs(x - y) < best[0]:
                        best = (abs(x - y), layer, lo, hi)
                    break
    return best


@requires_torch
def test_the_ffn_sort_margin_is_the_threshold_at_which_the_order_flips():
    """A MARGIN IS A THRESHOLD, NOT A DIFFERENCE, and this asserts that.

    The reported margin claims to be the amount a key must move before the sort
    order changes. So move it: by 0.4x the margin the order must hold, and by
    2x it must flip. An implementation that reports some other difference --
    S53 reported the largest difference anywhere in the key tuple, 59,000x too
    large -- passes a recomputation that shares its misunderstanding and fails
    this.
    """
    import torch

    model = fresh()
    report = run_canonicalize(model, TINY, recipe=SORT_ONLY_RECIPE)
    margin, layer, lo, hi = _min_margin_layer_and_pair(model, TINY)
    assert layer is not None, "no adjacent FFN pair found"
    assert margin == pytest.approx(report.min_ffn_sort_margin, rel=1e-9), (
        "the tightest pair found here disagrees with the reported margin")

    block = model.transformer.h[layer]
    baseline = sorted(range(TINY.n_inner), key=lambda j: _ffn_keys(block)[j])

    def order_after(delta):
        original = float(block.mlp.c_fc.bias[lo].detach())
        with torch.no_grad():
            block.mlp.c_fc.bias[lo] = original + delta
        try:
            return sorted(range(TINY.n_inner),
                          key=lambda j: _ffn_keys(block)[j])
        finally:
            with torch.no_grad():
                block.mlp.c_fc.bias[lo] = original

    assert order_after(0.4 * margin) == baseline, (
        f"the order changed at 0.4x the reported margin ({margin:.3e}), so the "
        "reported value is LARGER than the true threshold -- exactly the S53 "
        "failure, where the metric overstated the margin by 59,000x")
    assert order_after(2.0 * margin) != baseline, (
        f"the order did NOT change at 2x the reported margin ({margin:.3e}), "
        "so the reported value is smaller than the true threshold and is not "
        "the quantity it is named for")


@requires_torch
def test_the_old_max_over_tuple_metric_would_fail_the_operational_check():
    """S53 reconstructed, so the guard is shown to catch the thing it exists
    for rather than merely asserted to.

    The retired metric took the largest difference anywhere in the key tuple.
    Recomputed here on the same model, it must come out strictly larger than
    the true deciding margin -- which is what makes it fail the threshold test
    above, and what made it 59,000x too optimistic on real GPT-2.
    """
    model = fresh()
    run_canonicalize(model, TINY, recipe=SORT_ONLY_RECIPE)

    def retired_metric(keys):
        ordered = sorted(keys)
        return min(max(abs(x - y) for x, y in zip(a, b))
                   for a, b in zip(ordered, ordered[1:]))

    for block in model.transformer.h:
        keys = _ffn_keys(block)
        assert retired_metric(keys) > _deciding_margin(keys), (
            "the retired max-over-tuple metric did not overstate the margin on "
            "this model, so this reconstruction is not exercising S53")


@requires_torch
def test_the_head_sort_key_really_is_the_invariant_spectrum():
    """S49 reconstructed. The key is documented as the head's two spectra; the
    first implementation used weight-row norms, which omit the bias row and are
    not even monotonic. Checked against an explicit SVD."""
    model = fresh()
    run_canonicalize(model, TINY, recipe=SORT_ONLY_RECIPE)
    spectra = _explicit_spectra(model, TINY)

    index = 0
    for block in model.transformer.h:
        for h in range(TINY.n_head):
            key = canonicalize.SortHeads._key(block, TINY, h)
            qk, vo = spectra[index]
            index += 1
            expected = tuple(qk.tolist()) + tuple(vo.tolist())
            assert len(key) == len(expected)
            for got, want in zip(key, expected):
                assert got == pytest.approx(want, rel=1e-8), (
                    "the head sort key is not the invariant spectrum it is "
                    "documented as -- see S49")
