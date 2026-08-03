#!/usr/bin/env python
"""Step 9's own error bar. Phase 5.

    python scripts/canonicalization_error.py

Canonicalization is float arithmetic and is not exact. This script measures how
much error it introduces, so the ruler's own noise floor is a number rather
than an assumption. Five measurements, none of them interpreted here:

  A  SYMMETRY RESIDUAL. Apply a symmetry, canonicalize both sides, and report
     the residual distance as a fraction of the raw distance. This is the
     ruler tested against itself, on two models that are secretly identical.

  B  EPSILON SWEEP. The use case is NOT two secretly-identical models -- it is
     two models that genuinely differ a little, an arm and its seed-matched
     twin. So perturb by a small random epsilon and ask whether canonicalizing
     INFLATES the distance. A ratio near 1 means canonicalization is neutral;
     much above 1 means step 9 is a noise source however correct its algebra.
     Swept over seven decades, two perturbation shapes, five seeds, with
     per-head attribution by condition number and with counters for the
     discrete events -- sort flips, sign flips -- that would explain a spike.

  C  DISPERSION SWEEP. Every number in A and B is measured on fully trained
     public GPT-2. The study injects at step 200, roughly 52M tokens into a
     from-scratch run, where LayerNorm gains have barely moved from exactly 1.0
     and Conv1D biases from exactly 0.0. That configuration hid two independent
     defects during this build (S42, S48). This sweeps both gains and biases
     from their initialization values toward GPT-2's observed spread and
     reports how the conditioning diagnostics vary along the way.

  D  ATTRIBUTION. A pooled ratio that comes out large says something is wrong;
     it does not say what. This re-runs the same perturbations with individual
     recipe steps removed, and with the Hungarian-matching route substituted
     for sorting, so the inflation is attributed to a step rather than guessed
     at. It also reports the distribution of margins that actually DECIDE the
     sort comparisons, which is the quantity governing whether sorting is a
     safe way to fix a permutation gauge at all.

  E  PERMUTED-MODEL RECOVERY. Measurement D compares variants on models whose
     correct correspondence is already the identity, because an epsilon
     perturbation reorders nothing. Alignment scoring the same as having no
     permutation step there is NOT evidence that alignment works -- it is
     evidence there was nothing to recover. This presents models that genuinely
     differ by an FFN neuron permutation before perturbing, which is the only
     case that separates "drop the permutation step" from "align instead of
     sorting".

NOTHING HERE CONCLUDES WHETHER THE RULER IS USABLE. Every measurement carries a
LIMITATION field, and the range of the distortion factor is recorded as an
OPEN QUESTION -- a stated weakness of the study -- rather than as a caveat. The
comparison target, the study's seed-only noise floor, requires trained models
that do not exist, so no epsilon here is calibrated to anything real.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import canonicalize as C  # noqa: E402

DEFAULT_REPORT_DIR = REPO_ROOT / "docs" / "measurements"
REPORT_STEM = "9-canonicalization-error"

#: Seven decades, bracketing from below float32 epsilon up to a large change.
EPSILONS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
#: Widened from five. The sort's failure was seed-dependent, which makes seed
#: stability a live question for the alignment path too rather than an
#: assumption -- so the sweep reports spread across seeds, not just a median.
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
SHAPES = ("isotropic", "per_tensor")

#: Interpolation from initialization (0.0) to public GPT-2 (1.0).
DISPERSION_LEVELS = (0.0, 0.01, 0.03, 0.1, 0.2, 0.4, 0.7, 1.0)

LIMITATION = (
    "EVERY NUMBER IN THIS FILE IS MEASURED ON FULLY-TRAINED PUBLIC GPT-2. The "
    "study's own model does not exist -- there is no training loop and no "
    "step-200 checkpoint. Two consequences. First, the comparison target for "
    "the epsilon sweep does not exist either: the study's seed-only noise "
    "floor requires trained models, so no epsilon here is calibrated to "
    "anything real and the ratio curve must be read as a shape, not against a "
    "threshold. Second, and more specific to this ruler, the study injects at "
    "step 200 -- roughly 52M tokens (256 x 1024 x 200) into a from-scratch "
    "run -- where LayerNorm gains have barely moved from exactly 1.0 and "
    "Conv1D biases from exactly 0.0. Public GPT-2 is nowhere near that state. "
    "Measurement C is the quantification of that gap and is the one to read "
    "before trusting A or B at the study's actual injection point. THESE "
    "NUMBERS MUST BE RE-MEASURED AGAINST A REAL STEP-200 CHECKPOINT ONCE "
    "TRAINING INFRASTRUCTURE EXISTS."
)

#: A stated weakness of the study, not a caveat on a measurement.
OPEN_QUESTION_DISTORTION_FACTOR = (
    "THE RULER'S DISTORTION FACTOR IS NOT A NUMBER, IT IS A RANGE, AND WHICH "
    "END APPLIES CANNOT BE KNOWN YET. Even with the FFN sort removed -- the "
    "step measurement D attributes essentially all of the inflation to -- this "
    "canonicalization is not distance-neutral. It scores 3.05 at eps=1e-8 and "
    "eps=1e-6, and 84.4 at eps=1e-4. So it roughly TRIPLES a small difference "
    "at the low end and inflates by more than eightyfold three decades up. "
    "3.05 is not 1, and 84.4 is not 3.05. Which of them is operative depends "
    "on how far a burst arm actually sits from its seed-matched twin after "
    "training, expressed as a fraction of the parameter norm -- and that "
    "quantity does not exist until models are trained. A ruler whose "
    "distortion factor ranges over more than an order of magnitude depending "
    "on an unmeasured quantity is a WEAKNESS OF THE STUDY and is recorded "
    "here as an open question rather than as a footnote. Resolving it requires "
    "measuring the twin-vs-twin distance on real checkpoints and reading the "
    "curve in this file at that epsilon."
)

RECIPE_SYMMETRIES = (
    "layernorm_gain_rescale",
    "head_permutation",
    "head_internal_transform",
    "ffn_neuron_permutation",
    "key_bias_shift",
    "value_bias_shift",
)


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------


def l2_distance(sd_a: dict, sd_b: dict) -> float:
    """L2 over the whole flat parameter vector, accumulated in float64.

    Computed per tensor rather than by materialising a 124-million-element
    vector, which would be 1 GB per copy for a number that does not need it.
    """
    total = 0.0
    for name in sd_a:
        total += float((sd_a[name].double() - sd_b[name].double()).pow(2).sum())
    return math.sqrt(total)


def l2_norm(sd: dict) -> float:
    return math.sqrt(sum(float(t.double().pow(2).sum()) for t in sd.values()))


def head_slice_distance(sd_a: dict, sd_b: dict, layer: int, head: int,
                        arch) -> float:
    """L2 restricted to one head's own parameters.

    A head owns its three column blocks of c_attn (and their bias entries) plus
    its row block of attn.c_proj. Restricting the distance to those is what
    lets a spike be attributed to a particular head's conditioning instead of
    being averaged away across 124 million parameters.
    """
    total = 0.0
    w = f"transformer.h.{layer}.attn.c_attn.weight"
    b = f"transformer.h.{layer}.attn.c_attn.bias"
    o = f"transformer.h.{layer}.attn.c_proj.weight"
    for which in C.QKV:
        sl = C.head_columns(which, head, arch)
        total += float((sd_a[w][:, sl].double()
                        - sd_b[w][:, sl].double()).pow(2).sum())
        total += float((sd_a[b][sl].double() - sd_b[b][sl].double()).pow(2).sum())
    rows = C.head_rows_of_out_proj(head, arch)
    total += float((sd_a[o][rows, :].double()
                    - sd_b[o][rows, :].double()).pow(2).sum())
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# perturbation
# ---------------------------------------------------------------------------


def perturb(model, epsilon: float, shape: str, seed: int) -> float:
    """Move the model by a relative distance of `epsilon`. Returns d_raw.

    Both shapes are normalised so that d_raw is exactly epsilon * ||theta||,
    known by construction rather than measured -- which means the ratio in the
    sweep has an exact denominator.

    TWO SHAPES, NOT ONE, and the difference matters. GPT-2's tensors have
    wildly unequal sizes: wte alone is 38.6M of 124.4M parameters, while an
    entire LayerNorm gain is 768. An ISOTROPIC direction therefore puts nearly
    all of its energy into the embedding and the big projections and almost
    none into the LayerNorm gains or the attention biases -- which is exactly
    where canonicalization is most sensitive. PER_TENSOR moves every tensor by
    the same relative amount instead. A sweep that ran only the first could
    look clean while missing the actual risk.
    """
    import torch

    gen = torch.Generator().manual_seed(seed)
    params = [p for _, p in model.named_parameters()]
    theta_norm = math.sqrt(sum(float(p.detach().double().pow(2).sum())
                               for p in params))
    target = epsilon * theta_norm

    with torch.no_grad():
        if shape == "isotropic":
            draws = [torch.randn(p.shape, generator=gen, dtype=torch.float64)
                     for p in params]
            norm = math.sqrt(sum(float(d.pow(2).sum()) for d in draws))
            for p, d in zip(params, draws):
                p.add_((d * (target / norm)).to(dtype=p.dtype))
        elif shape == "per_tensor":
            for p in params:
                d = torch.randn(p.shape, generator=gen, dtype=torch.float64)
                dn = float(d.pow(2).sum()) ** 0.5
                pn = float(p.detach().double().pow(2).sum()) ** 0.5
                if dn == 0 or pn == 0:
                    continue
                p.add_((d * (epsilon * pn / dn)).to(dtype=p.dtype))
        else:
            raise C.CanonicalizeError(f"unknown perturbation shape {shape!r}")
    return target


# ---------------------------------------------------------------------------
# the gauge subspace, by dimension counting
# ---------------------------------------------------------------------------


def gauge_dimensions(arch) -> dict:
    """How many of the model's coordinates are pure gauge.

    Counted exactly rather than estimated. An isotropic random perturbation
    puts a fraction dim_gauge / D of its ENERGY into the gauge subspace, so
    this is what says how much of a generic small difference between two models
    canonicalization could possibly be removing.

    The head-internal freedom is GL(head_dim) on each of the Q/K and V/O
    pairings, per head, per layer. The LayerNorm gain freedom is one scalar per
    channel per absorbed norm (ln_1 and ln_2 only; ln_f is not absorbed). The
    bias shifts are head_dim per head per layer, twice.
    """
    per_layer_heads = arch.n_head
    head_internal = 2 * arch.head_dim ** 2 * per_layer_heads * arch.n_layer
    gains = 2 * arch.n_embd * arch.n_layer
    bias_shifts = 2 * arch.head_dim * per_layer_heads * arch.n_layer
    total_params = arch.n_params
    dims = {
        "head_internal_GL": head_internal,
        "layernorm_gain": gains,
        "key_and_value_bias_shift": bias_shifts,
        "continuous_total": head_internal + gains + bias_shifts,
        "model_parameters": total_params,
    }
    dims["continuous_fraction_of_dimensions"] = (
        dims["continuous_total"] / total_params)
    # Discrete symmetries contribute no dimensions at all -- a permutation has
    # no tangent direction -- so they are counted separately, as group sizes.
    dims["discrete_group_log10_size"] = (
        arch.n_layer * (math.log10(math.factorial(arch.n_head))
                        + sum(math.log10(i) for i in range(1, arch.n_inner + 1)))
    )
    return dims


# ---------------------------------------------------------------------------
# A. symmetry residual
# ---------------------------------------------------------------------------


def measure_symmetry_residual(build, arch, seeds, stream, recipe=None) -> dict:
    import copy
    import torch

    # The twin defines the frame; both sides are then measured against it.
    # Canonical form is pairwise-relative since the FFN permutation is fixed by
    # matching rather than sorting, so a reference is required for the shipped
    # recipe to do anything at all.
    frame = build()
    C.canonicalize(frame, arch, recipe=recipe)
    theta_norm = l2_norm(C.canonical_state_dict(frame))

    rows = {}
    for name in RECIPE_SYMMETRIES + ("__composed__",):
        ratios, raws, canons = [], [], []
        for seed in seeds:
            plain = build()
            moved = copy.deepcopy(plain)
            names = RECIPE_SYMMETRIES if name == "__composed__" else (name,)
            for sym_name in names:
                C.symmetry_by_name(sym_name)().sample(moved, arch, seed).apply(
                    moved, arch)
            d_raw = l2_distance(C.canonical_state_dict(plain),
                                C.canonical_state_dict(moved))
            C.canonicalize(plain, arch, recipe=recipe, reference=frame)
            C.canonicalize(moved, arch, recipe=recipe, reference=frame)
            d_canon = l2_distance(C.canonical_state_dict(plain),
                                  C.canonical_state_dict(moved))
            raws.append(d_raw)
            canons.append(d_canon)
            ratios.append(d_canon / d_raw if d_raw else float("nan"))
            del plain, moved
        rows[name] = {
            "d_raw_median": statistics.median(raws),
            "d_canonical_median": statistics.median(canons),
            "d_canonical_max": max(canons),
            "ratio_median": statistics.median(ratios),
            "ratio_max": max(ratios),
        }
        print(f"    {name:<28} d_raw={rows[name]['d_raw_median']:.4e}  "
              f"d_canon={rows[name]['d_canonical_median']:.4e}  "
              f"ratio={rows[name]['ratio_median']:.3e}", file=stream, flush=True)
    return {
        "note": ("residual distance left after canonicalizing two models that "
                 "are secretly the same model in different gauges. This is the "
                 "ruler tested against ITSELF; measurement B is the one that "
                 "tests the use case."),
        "parameter_norm": theta_norm,
        "per_symmetry": rows,
    }


# ---------------------------------------------------------------------------
# B. epsilon sweep
# ---------------------------------------------------------------------------


def _order_flips(a, b) -> int:
    return sum(1 for x, y in zip(a, b) if tuple(x) != tuple(y))


def _basis_drift(sd_a, sd_b, arch) -> float:
    """Smallest |cos| between corresponding canonical Q-factor columns.

    1.0 means the head-internal SVD picked the same basis in both models; below
    1.0 means it ROTATED. This exists because _sign_flips only sees discrete
    +/-1 flips, and the failure that actually bites is a CONTINUOUS rotation of
    the basis inside a near-degenerate subspace. On real GPT-2 the smallest
    singular gap is 5.479e-06, so a perturbation of that order rotates the
    basis substantially while flipping no signs at all -- which is exactly the
    case the sign counter reported as 0/0/0 while the distance inflated 2000x.
    """
    worst = 1.0
    for layer in range(arch.n_layer):
        w = sd_a[f"transformer.h.{layer}.attn.c_attn.weight"]
        w2 = sd_b[f"transformer.h.{layer}.attn.c_attn.weight"]
        for head in range(arch.n_head):
            sl = head_columns_local(head, arch)
            a, b = w[:, sl].double(), w2[:, sl].double()
            na = a.norm(dim=0).clamp_min(1e-300)
            nb = b.norm(dim=0).clamp_min(1e-300)
            cos = ((a * b).sum(dim=0) / (na * nb)).abs()
            worst = min(worst, float(cos.min()))
    return worst


def head_columns_local(head, arch):
    return C.head_columns("q", head, arch)


def _sign_flips(sd_a, sd_b, arch) -> int:
    """Columns of the canonical Q factor that came out with opposite signs."""
    import torch

    flips = 0
    for layer in range(arch.n_layer):
        w = sd_a[f"transformer.h.{layer}.attn.c_attn.weight"]
        w2 = sd_b[f"transformer.h.{layer}.attn.c_attn.weight"]
        for head in range(arch.n_head):
            sl = C.head_columns("q", head, arch)
            a, b = w[:, sl].double(), w2[:, sl].double()
            same = (a - b).pow(2).sum(dim=0)
            opposite = (a + b).pow(2).sum(dim=0)
            flips += int((opposite < same).sum())
    return flips


def measure_epsilon_sweep(build, arch, epsilons, shapes, seeds, stream,
                          recipe=None) -> dict:
    """Does canonicalization inflate the distance between two NEARLY IDENTICAL
    models? That is the question the study actually asks of this ruler."""
    import copy

    reference = build()
    ref_plain_sd = C.canonical_state_dict(reference)
    ref_report = C.canonicalize(reference, arch, recipe=recipe)
    ref_sd = C.canonical_state_dict(reference)
    conditions = list(ref_report.head_conditions)

    # Split the heads by conditioning so a spike can be attributed. The
    # head-internal step inverts through this spectrum, so the worst-
    # conditioned heads are where inflation should appear first if it appears.
    order = sorted(range(len(conditions)), key=lambda i: conditions[i])
    n = len(order)
    # Top decile by condition number, and a band of the same size straddling
    # the median. max(1, ...) so both buckets are non-empty on a small model --
    # an empty median bucket silently reported nan on the first run.
    width = max(1, n // 10)
    worst_heads = order[-width:]
    lo = max(0, n // 2 - max(1, width // 2))
    median_heads = order[lo:lo + width]

    def head_of(index):
        return divmod(index, arch.n_head)

    results = {}
    for shape in shapes:
        for epsilon in epsilons:
            key = f"{shape}|{epsilon:g}"
            ratios, worst_ratios, median_ratios = [], [], []
            head_flips, ffn_flips, sign_flips, drifts = [], [], [], []
            gaps, margins = [], []
            for seed in seeds:
                moved = build()
                d_raw = perturb(moved, epsilon, shape, seed)
                moved_plain_sd = C.canonical_state_dict(moved)
                report = C.canonicalize(moved, arch, recipe=recipe,
                                        reference=reference)
                moved_sd = C.canonical_state_dict(moved)

                d_canon = l2_distance(ref_sd, moved_sd)
                ratios.append(d_canon / d_raw)

                for bucket, store in ((worst_heads, worst_ratios),
                                      (median_heads, median_ratios)):
                    num = den = 0.0
                    for idx in bucket:
                        layer, head = head_of(idx)
                        den += head_slice_distance(ref_plain_sd, moved_plain_sd,
                                                   layer, head, arch) ** 2
                        num += head_slice_distance(ref_sd, moved_sd,
                                                   layer, head, arch) ** 2
                    store.append(math.sqrt(num / den) if den else float("nan"))

                head_flips.append(_order_flips(ref_report.head_orders,
                                               report.head_orders))
                ffn_flips.append(_order_flips(ref_report.ffn_orders,
                                              report.ffn_orders))
                sign_flips.append(_sign_flips(ref_sd, moved_sd, arch))
                drifts.append(_basis_drift(ref_sd, moved_sd, arch))
                gaps.append(report.min_singular_gap)
                margins.append(min(report.min_head_sort_margin,
                                   report.min_ffn_sort_margin))
                del moved, moved_sd, moved_plain_sd

            spread = (max(ratios) / min(ratios)) if min(ratios) > 0 else float("inf")
            results[key] = {
                "shape": shape, "epsilon": epsilon,
                "n_seeds": len(ratios),
                # Seed stability. The retired sort's inflation was
                # seed-dependent, which is what disqualified it, so this is
                # measured for the shipped recipe rather than assumed away.
                "ratio_stdev": statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
                "ratio_max_over_min": spread,
                "ratio_min": min(ratios),
                "ratio_median": statistics.median(ratios),
                "ratio_max": max(ratios),
                "ratio_worst_conditioned_heads_median":
                    statistics.median(worst_ratios),
                "ratio_median_conditioned_heads_median":
                    statistics.median(median_ratios),
                "head_order_flips_total": sum(head_flips),
                "ffn_order_flips_total": sum(ffn_flips),
                "sign_flips_total": sum(sign_flips),
                # 1.0 = the head-internal basis is identical in both models.
                # Below 1.0 = it rotated, which sign_flips cannot see.
                "min_basis_alignment_cos": min(drifts),
                "min_singular_gap_min": min(gaps),
                "min_sort_margin_min": min(margins),
            }
            r = results[key]
            print(f"    {shape:<11} eps={epsilon:<8g} ratio med={r['ratio_median']:.4f} "
                  f"[{r['ratio_min']:.4f},{r['ratio_max']:.4f}]  "
                  f"worst-cond={r['ratio_worst_conditioned_heads_median']:.4f} "
                  f"med-cond={r['ratio_median_conditioned_heads_median']:.4f}  "
                  f"flips h/f/s={r['head_order_flips_total']}/"
                  f"{r['ffn_order_flips_total']}/{r['sign_flips_total']} "
                  f"basis_cos={r['min_basis_alignment_cos']:.4f}",
                  file=stream, flush=True)

    return {
        "note": ("ratio = ||canon(M) - canon(M+eps)|| / ||M - (M+eps)||. "
                 "Near 1 means canonicalization is distance-neutral on nearly "
                 "identical models. Much above 1 means it INFLATES a small "
                 "real difference, which would make step 9 a noise source "
                 "whatever its algebra. Much below 1 means it is collapsing a "
                 "genuine difference. All three are reported, none judged."),
        "head_conditioning": {
            "n_heads": n,
            "min": min(conditions), "median": statistics.median(conditions),
            "max": max(conditions),
            "worst_decile_indices": worst_heads,
            "note": ("heads split by the condition number of their Q/K "
                     "invariant so a spike in the ratio can be attributed to "
                     "conditioning rather than guessed at"),
        },
        "cells": results,
    }


# ---------------------------------------------------------------------------
# D. attribution -- which step inflates, and does matching fix it
# ---------------------------------------------------------------------------


def measure_step_attribution(build, arch, epsilons, seeds, stream) -> dict:
    """Which recipe step is responsible for the inflation in measurement B.

    Measurement B reports a pooled ratio. A pooled number that is large says
    something is wrong; it does not say what. This re-runs the same
    perturbation against recipes with individual steps removed, and against the
    Hungarian-matching route, so the inflation can be attributed to a step
    rather than guessed at.
    """
    import copy

    # Variants are defined against the SHIPPED recipe. An earlier version
    # filtered for SortFFNNeurons, which DEFAULT_RECIPE no longer contains, so
    # four of the five removals were no-ops and the table reported them all as
    # identical. The live attribution question is now the head-internal step.
    variants = {
        "shipped_recipe": C.DEFAULT_RECIPE,
        "without_ffn_permutation": tuple(
            s for s in C.DEFAULT_RECIPE
            if not isinstance(s, (C.AlignFFNNeurons, C.SortFFNNeurons))),
        "without_head_sort": tuple(
            s for s in C.DEFAULT_RECIPE if not isinstance(s, C.SortHeads)),
        "without_head_internal": tuple(
            s for s in C.DEFAULT_RECIPE
            if not isinstance(s, C.CanonicalizeHeadInternal)),
        "RETIRED_sort_recipe": C.SORT_ONLY_RECIPE,
    }
    rows = {}
    for label, recipe in variants.items():
        reference = build()
        C.canonicalize(reference, arch, recipe=recipe)
        ref_sd = C.canonical_state_dict(reference)
        cells = {}
        for epsilon in epsilons:
            ratios = []
            for seed in seeds:
                moved = build()
                d_raw = perturb(moved, epsilon, "isotropic", seed)
                # reference= is REQUIRED: without it AlignFFNNeurons is a no-op
                # and every variant collapses to "no permutation step", which
                # is what made an earlier version of this table report all five
                # variants as identical.
                C.canonicalize(moved, arch, recipe=recipe, reference=reference)
                ratios.append(
                    l2_distance(ref_sd, C.canonical_state_dict(moved)) / d_raw)
                del moved
            cells[f"{epsilon:g}"] = statistics.median(ratios)
        rows[label] = cells
        print(f"    {label:<22} " + "  ".join(
            f"eps={k}:{v:.4g}" for k, v in cells.items()),
            file=stream, flush=True)
        del reference, ref_sd

    # The matching route, on the same perturbations.
    no_sort = variants["without_ffn_permutation"]
    reference = build()
    C.canonicalize(reference, arch, recipe=no_sort)
    ref_sd = C.canonical_state_dict(reference)
    cells = {}
    for epsilon in epsilons:
        ratios = []
        for seed in seeds:
            moved = build()
            d_raw = perturb(moved, epsilon, "isotropic", seed)
            C.canonicalize(moved, arch, recipe=no_sort)
            C.align_permutations_to(moved, reference, arch)
            ratios.append(
                l2_distance(ref_sd, C.canonical_state_dict(moved)) / d_raw)
            del moved
        cells[f"{epsilon:g}"] = statistics.median(ratios)
    rows["hungarian_alignment"] = cells
    print(f"    {'hungarian_alignment':<22} " + "  ".join(
        f"eps={k}:{v:.4g}" for k, v in cells.items()), file=stream, flush=True)
    del reference, ref_sd

    return {
        "note": ("median ratio under an isotropic perturbation, with recipe "
                 "steps removed one at a time, plus the Hungarian-matching "
                 "route in place of sorting. Attributes the pooled ratio in "
                 "measurement B to a specific step."),
        "variants": rows,
    }


def measure_permuted_recovery(build, arch, epsilons, seeds, stream) -> dict:
    """The case that separates 'drop the step' from 'align instead of sorting'.

    Measurement D compared the variants on models whose correct correspondence
    was already the IDENTITY -- an epsilon perturbation does not reorder
    anything. Alignment scoring the same as having no permutation step there is
    therefore not evidence that alignment works. It is evidence that there was
    nothing to recover.

    Here the models genuinely differ by a permutation: apply a real FFN neuron
    permutation, THEN perturb by epsilon. The honest difference between the two
    models is still only the epsilon -- the permutation is gauge and carries no
    information -- so a canonicalizer that works should return a ratio near
    whatever it scores without the permutation, and one that cannot remove the
    permutation should blow up by the full size of the permutation.
    """
    import copy

    # AlignFFNNeurons MUST be in this exclusion list. It was not, and because
    # DEFAULT_RECIPE no longer contains SortFFNNeurons the filter removed
    # nothing -- so the variant labelled "no_permutation_step" still aligned,
    # and reported that dropping the step costs nothing. It costs 6.9e+07.
    no_perm = tuple(s for s in C.DEFAULT_RECIPE
                    if not isinstance(s, (C.SortHeads, C.SortFFNNeurons,
                                          C.AlignFFNNeurons)))
    no_sort = no_perm

    def canon_none(model, reference):
        C.canonicalize(model, arch, recipe=no_sort, reference=reference)

    def canon_align(model, reference):
        C.canonicalize(model, arch, recipe=no_sort)
        C.align_permutations_to(model, reference, arch)

    def canon_sort(model, reference):
        C.canonicalize(model, arch, recipe=C.SORT_ONLY_RECIPE)

    def canon_shipped(model, reference):
        C.canonicalize(model, arch, recipe=C.DEFAULT_RECIPE, reference=reference)

    variants = {
        "no_permutation_step": (no_sort, canon_none),
        "hungarian_alignment": (no_sort, canon_align),
        "ffn_sort_RETIRED": (C.SORT_ONLY_RECIPE, canon_sort),
        "shipped_recipe": (C.DEFAULT_RECIPE, canon_shipped),
    }

    rows = {}
    for label, (ref_recipe, canon_fn) in variants.items():
        reference = build()
        C.canonicalize(reference, arch, recipe=ref_recipe)
        ref_sd = C.canonical_state_dict(reference)
        cells = {}
        for epsilon in epsilons:
            ratios = []
            for seed in seeds:
                moved = build()
                # A GENUINE permutation, not a perturbation. This is the whole
                # point: the correct correspondence is no longer the identity.
                C.symmetry_by_name("ffn_neuron_permutation")().sample(
                    moved, arch, seed).apply(moved, arch)
                # d_raw is the epsilon alone. The permutation is gauge and
                # contributes nothing a correct ruler should report.
                d_raw = perturb(moved, epsilon, "isotropic", seed)
                canon_fn(moved, reference)   # each variant threads reference
                ratios.append(
                    l2_distance(ref_sd, C.canonical_state_dict(moved)) / d_raw)
                del moved
            cells[f"{epsilon:g}"] = statistics.median(ratios)
        rows[label] = cells
        print(f"    {label:<22} " + "  ".join(
            f"eps={k}:{v:.4g}" for k, v in cells.items()),
            file=stream, flush=True)
        del reference, ref_sd

    return {
        "note": ("models that GENUINELY differ by an FFN neuron permutation, "
                 "then perturbed by epsilon. d_raw is the epsilon alone, since "
                 "the permutation is gauge. Measurement D could not separate "
                 "'no permutation step' from 'alignment' because the correct "
                 "correspondence there was always the identity; here it is "
                 "not, so a variant that cannot recover a permutation must "
                 "blow up by the full size of one."),
        "variants": rows,
    }


def measure_sort_margins(build, arch, stream) -> dict:
    """The distribution of margins that actually decide the sort comparisons.

    A lexicographic sort is settled by the first component where two keys
    differ. That difference is what a perturbation must exceed to flip the
    pair, and it is the quantity that governs whether sorting is a safe way to
    fix a permutation gauge on this model.
    """
    model = build()
    C.canonicalize(model, arch)
    margins = []
    for block in model.transformer.h:
        bias = block.mlp.c_fc.bias.detach()
        in_norm = block.mlp.c_fc.weight.detach().norm(dim=0)
        out_norm = block.mlp.c_proj.weight.detach().norm(dim=1)
        keys = sorted(zip(bias.tolist(), in_norm.tolist(), out_norm.tolist()))
        for a, b in zip(keys, keys[1:]):
            for x, y in zip(a, b):
                if x != y:
                    margins.append(abs(x - y))
                    break
            else:
                margins.append(0.0)
    margins.sort()

    def pct(q):
        return margins[max(0, int(len(margins) * q / 100) - 1) if q else 0]

    out = {
        "note": ("FFN sort: the margin that DECIDES each adjacent comparison, "
                 "i.e. the difference at the first differing key component. "
                 "This is the quantity a perturbation must exceed to flip a "
                 "pair. CanonReport previously reported the largest difference "
                 "anywhere in the key tuple instead, which overstated it by a "
                 "factor of about 59,000 -- see S53."),
        "n_adjacent_pairs": len(margins),
        "percentiles": {str(q): pct(q) for q in (0, 0.1, 1, 5, 25, 50)},
    }
    print(f"    FFN deciding margin: min {out['percentiles']['0']:.3e}  "
          f"1st pct {out['percentiles']['1']:.3e}  "
          f"median {out['percentiles']['50']:.3e}", file=stream, flush=True)
    del model
    return out


# ---------------------------------------------------------------------------
# C. dispersion sweep -- toward initialization
# ---------------------------------------------------------------------------


def observed_distributions(model) -> dict:
    """Public GPT-2's actual gain and bias spreads -- the far end of the sweep."""
    import torch

    gains = torch.cat([ln.weight.detach().double().reshape(-1)
                       for ln in C._all_layernorms(model)])
    biases = torch.cat([b.detach().double().reshape(-1)
                        for b in C._conv1d_biases(model)])

    def describe(t, near):
        return {
            "min": float(t.min()), "median": float(t.median()),
            "max": float(t.max()), "mean": float(t.mean()),
            "std": float(t.std()),
            "fraction_within_1pct_of_init":
                float((t - near).abs().lt(0.01).double().mean()),
        }

    return {
        "layernorm_gain": describe(gains, 1.0),
        "conv1d_bias": describe(biases, 0.0),
        "note": ("initialization sets every gain to exactly 1.0 and every "
                 "Conv1D bias to exactly 0.0; these are what a fully trained "
                 "model looks like instead. The study injects at step 200, "
                 "much nearer the initialization end."),
    }


def measure_dispersion_sweep(build, arch, levels, stream) -> dict:
    """Interpolate gains and biases from initialization toward GPT-2.

    t = 0 is exactly the initialization configuration -- every gain 1.0, every
    Conv1D bias 0.0 -- which is the state that hid two independent defects
    during this build (S42, S48) and is close to where the study injects.
    t = 1 is public GPT-2 unchanged. Weight matrices are GPT-2's throughout, so
    only the two families being swept vary.
    """
    import torch

    source = build()
    gains = [ln.weight.detach().clone() for ln in C._all_layernorms(source)]
    biases = [b.detach().clone() for b in C._conv1d_biases(source)]

    rows = {}
    for t in levels:
        model = build()
        with torch.no_grad():
            for ln, g in zip(C._all_layernorms(model), gains):
                ln.weight.copy_(1.0 + t * (g - 1.0))
            for b, original in zip(C._conv1d_biases(model), biases):
                b.copy_(t * original)
        try:
            # SORT_ONLY_RECIPE deliberately: the FFN sort margin is a property
            # of the MODEL's keys, not of the shipped recipe, and it is what
            # says why the sort was retired and how much worse it gets toward
            # initialization. The shipped recipe does not compute it, so it is
            # measured through the retired one and labelled as such.
            report = C.canonicalize(model, arch, recipe=C.SORT_ONLY_RECIPE)
            row = {
                "min_layernorm_gain": report.min_layernorm_gain,
                "min_singular_gap": report.min_singular_gap,
                "max_head_condition": report.max_head_condition,
                "median_head_condition":
                    statistics.median(report.head_conditions),
                "min_head_sort_margin": report.min_head_sort_margin,
                "min_ffn_sort_margin": report.min_ffn_sort_margin,
                "refused": None,
            }
        except C.CanonicalizeError as exc:
            row = {"refused": str(exc).splitlines()[0]}
        rows[f"{t:g}"] = row
        if row["refused"]:
            print(f"    t={t:<6g} REFUSED: {row['refused'][:60]}",
                  file=stream, flush=True)
        else:
            print(f"    t={t:<6g} gap={row['min_singular_gap']:.3e} "
                  f"cond_max={row['max_head_condition']:.3e} "
                  f"cond_med={row['median_head_condition']:.3e} "
                  f"head_margin={row['min_head_sort_margin']:.3e} "
                  f"ffn_margin={row['min_ffn_sort_margin']:.3e}",
                  file=stream, flush=True)
        del model

    return {
        "note": ("t = 0 is exactly the initialization configuration (all gains "
                 "1.0, all Conv1D biases 0.0); t = 1 is public GPT-2. Weight "
                 "matrices are GPT-2's at every level, so only the two swept "
                 "families vary. THE STUDY INJECTS AT STEP 200, near the t = 0 "
                 "end. Every conditioning number reported elsewhere in this "
                 "file was measured at t = 1."),
        "levels": rows,
        "observed_at_t_equals_1": observed_distributions(source),
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def format_report(payload) -> str:
    rule = "=" * 78
    thin = "-" * 78
    lines = ["", rule, "STEP 9 -- CANONICALIZATION ERROR BAR", rule,
             "PROXY MODEL. See the LIMITATION field in the JSON.", ""]

    lines += [rule, "A. SYMMETRY RESIDUAL -- the ruler against itself", rule,
              "Two models that are secretly identical, in different gauges.", "",
              f"{'symmetry':<28}{'d_raw':>13}{'d_canonical':>14}{'ratio':>12}"]
    for name, row in payload["symmetry_residual"]["per_symmetry"].items():
        lines.append(f"{name:<28}{row['d_raw_median']:>13.4e}"
                     f"{row['d_canonical_median']:>14.4e}"
                     f"{row['ratio_median']:>12.3e}")

    sweep = payload["epsilon_sweep"]
    lines += ["", rule,
              "B. EPSILON SWEEP -- does canonicalizing INFLATE a real difference?",
              rule,
              "ratio = ||canon(M) - canon(M+eps)|| / ||M - (M+eps)||.",
              "1.0 means neutral. Above means inflation. Below means collapse.",
              ""]
    hc = sweep["head_conditioning"]
    lines.append(f"head condition number over {hc['n_heads']} heads: "
                 f"min {hc['min']:.3g}, median {hc['median']:.3g}, "
                 f"max {hc['max']:.3g}")
    lines += ["",
              f"{'shape':<12}{'epsilon':>9}{'ratio med':>11}{'ratio max':>11}"
              f"{'worst-cond':>12}{'med-cond':>10}{'flips h/f/s':>14}"]
    for key, cell in sweep["cells"].items():
        lines.append(
            f"{cell['shape']:<12}{cell['epsilon']:>9.0e}"
            f"{cell['ratio_median']:>11.4f}{cell['ratio_max']:>11.4f}"
            f"{cell['ratio_worst_conditioned_heads_median']:>12.4f}"
            f"{cell['ratio_median_conditioned_heads_median']:>10.4f}"
            f"{cell['head_order_flips_total']:>5}"
            f"/{cell['ffn_order_flips_total']}"
            f"/{cell['sign_flips_total']}")

    retired = payload.get("epsilon_sweep_RETIRED_sort_recipe")
    if retired:
        lines += ["", thin,
                  "the RETIRED sort-based recipe, same sweep, for comparison",
                  thin,
                  "NOT the study's ruler. Kept so the measurement that retired",
                  "it stays reproducible. sort_ffn_neurons in place of",
                  "align_ffn_neurons; every other step identical.",
                  "",
                  f"{'shape':<12}{'epsilon':>9}{'ratio med':>13}{'ratio min':>12}"
                  f"{'ratio max':>13}{'ffn flips':>11}"]
        for cell in retired["cells"].values():
            lines.append(
                f"{cell['shape']:<12}{cell['epsilon']:>9.0e}"
                f"{cell['ratio_median']:>13.4g}{cell['ratio_min']:>12.4g}"
                f"{cell['ratio_max']:>13.4g}"
                f"{cell['ffn_order_flips_total']:>11}")
        lines.append("")
        lines.append("  The shipped recipe's rows are in section B above.")

    gauge = payload["gauge_dimensions"]
    lines += ["", thin,
              "gauge subspace, counted exactly", thin,
              f"  head-internal GL freedom   {gauge['head_internal_GL']:>12,}",
              f"  LayerNorm gain freedom     {gauge['layernorm_gain']:>12,}",
              f"  key/value bias shifts      {gauge['key_and_value_bias_shift']:>12,}",
              f"  continuous total           {gauge['continuous_total']:>12,}",
              f"  model parameters           {gauge['model_parameters']:>12,}",
              f"  fraction of dimensions     "
              f"{gauge['continuous_fraction_of_dimensions']:>12.4%}",
              "  An isotropic perturbation puts this fraction of its ENERGY in",
              "  gauge directions; the rest is physical and canonicalization",
              "  cannot remove it."]

    disp = payload["dispersion_sweep"]
    lines += ["", rule,
              "C. DISPERSION SWEEP -- toward the state the study injects at",
              rule,
              "t=0 is exactly initialization (gains 1.0, Conv1D biases 0.0).",
              "t=1 is public GPT-2. THE STUDY INJECTS NEAR t=0.", "",
              f"{'t':>6}{'min sv gap':>13}{'cond max':>12}{'cond med':>12}"
              f"{'head margin':>14}{'ffn margin':>13}"]
    for t, row in disp["levels"].items():
        if row.get("refused"):
            lines.append(f"{t:>6}   REFUSED: {row['refused'][:52]}")
        else:
            lines.append(f"{t:>6}{row['min_singular_gap']:>13.3e}"
                         f"{row['max_head_condition']:>12.3e}"
                         f"{row['median_head_condition']:>12.3e}"
                         f"{row['min_head_sort_margin']:>14.3e}"
                         f"{row['min_ffn_sort_margin']:>13.3e}")

    obs = disp["observed_at_t_equals_1"]
    lines += ["", thin, "public GPT-2's actual spread (the t=1 reference)", thin]
    for field, label in (("layernorm_gain", "LayerNorm gain (init 1.0)"),
                         ("conv1d_bias", "Conv1D bias (init 0.0)")):
        d = obs[field]
        lines.append(f"  {label:<28} min {d['min']:>10.4g}  "
                     f"med {d['median']:>9.4g}  max {d['max']:>9.4g}")
        lines.append(f"  {'':<28} within 1% of init: "
                     f"{d['fraction_within_1pct_of_init']:.4%}")

    attr = payload["step_attribution"]
    lines += ["", rule,
              "D. ATTRIBUTION -- which step inflates, and does matching fix it",
              rule,
              "Median ratio, isotropic perturbation, recipe steps removed one",
              "at a time. A pooled ratio says something is wrong; this says what.",
              ""]
    eps_keys = list(next(iter(attr["variants"].values())).keys())
    lines.append(f"{'recipe variant':<24}" + "".join(f"{'eps=' + k:>16}"
                                                     for k in eps_keys))
    for label, cells in attr["variants"].items():
        lines.append(f"{label:<24}" + "".join(f"{cells[k]:>16.4f}"
                                              for k in eps_keys))

    margins = payload["ffn_sort_margins"]
    lines += ["", thin,
              "FFN sort: the margin that DECIDES each adjacent comparison",
              thin,
              f"  over {margins['n_adjacent_pairs']:,} adjacent pairs"]
    for q, value in margins["percentiles"].items():
        lines.append(f"    {q:>5}th percentile   {value:.3e}")

    perm = payload["permuted_model_recovery"]
    lines += ["", rule,
              "E. PERMUTED-MODEL RECOVERY -- the case D could not separate",
              rule,
              "Models that GENUINELY differ by an FFN neuron permutation, then",
              "perturbed by epsilon. d_raw is the epsilon alone; the",
              "permutation is gauge and a correct ruler should not report it.",
              ""]
    pk = list(next(iter(perm["variants"].values())).keys())
    lines.append(f"{'variant':<24}" + "".join(f"{'eps=' + k:>16}" for k in pk))
    for label, cells in perm["variants"].items():
        lines.append(f"{label:<24}" + "".join(f"{cells[k]:>16.4g}" for k in pk))

    lines += ["", rule, "OPEN QUESTION -- the distortion factor is a range", rule]
    for chunk in payload["OPEN_QUESTION_distortion_factor"].split(". "):
        if chunk.strip():
            lines.append("  " + chunk.strip() + ".")

    lines += ["", rule, "LIMITATION", rule]
    for chunk in payload["LIMITATION"].split(". "):
        if chunk.strip():
            lines.append("  " + chunk.strip() + ".")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/canonicalization_error.py",
        description="Measure step 9's own numerical error. Changes no arm and "
                    "trains nothing.")
    parser.add_argument("--reportdir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--tiny", action="store_true",
                        help="run against the in-process tiny model, for a "
                             "fast smoke test rather than a real measurement")
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument(
        "--sections", default="ABCDE",
        help=("which measurements to run, e.g. DE. A subset MERGES into the "
              "existing results file rather than replacing it, so an "
              "expensive section does not have to be recomputed to correct a "
              "cheap one. The merged file records which sections were "
              "refreshed and when."))
    return parser


def main(argv=None) -> int:
    import copy
    import torch

    args = _build_parser().parse_args(argv)
    stream = sys.stdout
    rule = "=" * 78
    print(rule, file=stream)
    print("canonicalization_error -- step 9 phase 5", file=stream)
    print(rule, file=stream)

    if args.tiny:
        arch = C.TINY
        def build():
            return C.build_tiny_model(seed=0).to(dtype=torch.float64)
    else:
        arch = C.GPT2_124M
        source = C.build_real_gpt2().to(dtype=torch.float64)
        def build():
            return copy.deepcopy(source)

    seeds = SEEDS[:args.seeds]
    want = set(args.sections.upper())
    stem = REPORT_STEM + ("-tiny" if args.tiny else "")
    existing = {}
    prior = Path(args.reportdir) / f"{stem}.json"
    if want != set("ABCDE") and prior.is_file():
        existing = json.loads(prior.read_text(encoding="utf-8"))
        print(f"merging sections {sorted(want)} into {prior}", file=stream)
    residual = sweep = retired_sweep = dispersion = None
    attribution = sort_margins = permuted = None

    print(f"model: {'TINY (smoke test)' if args.tiny else 'gpt2 124M'} | "
          f"{arch.n_layer}L {arch.n_head}H {arch.n_embd}D | "
          f"{arch.n_params:,} params | seeds {list(seeds)}", file=stream)

    if "A" in want:
        print("\nA. symmetry residual", file=stream)
        residual = measure_symmetry_residual(build, arch, seeds, stream,
                                             recipe=C.DEFAULT_RECIPE)

    if "B" in want:
        print("\nB. epsilon sweep", file=stream)
        sweep = measure_epsilon_sweep(build, arch, EPSILONS, SHAPES, seeds, stream,
                                      recipe=C.DEFAULT_RECIPE)

        print(chr(10) + 'B-retired. epsilon sweep  [RETIRED sort recipe]', file=stream)
        retired_sweep = measure_epsilon_sweep(
            build, arch, EPSILONS, ('isotropic',), seeds, stream,
            recipe=C.SORT_ONLY_RECIPE)

    if "C" in want:
        print("\nC. dispersion sweep", file=stream)
        dispersion = measure_dispersion_sweep(build, arch, DISPERSION_LEVELS, stream)

    if "D" in want:
        print("\nD. step attribution", file=stream)
        attribution = measure_step_attribution(
            build, arch, (1e-8, 1e-6, 1e-4), seeds[:3], stream)
        sort_margins = measure_sort_margins(build, arch, stream)

    if "E" in want:
        print("\nE. permuted-model recovery", file=stream)
        permuted = measure_permuted_recovery(
            build, arch, (1e-8, 1e-6, 1e-4), seeds[:3], stream)

    payload = {
        "task": "step 9 phase 5",
        "LIMITATION": LIMITATION,
        "OPEN_QUESTION_distortion_factor": OPEN_QUESTION_DISTORTION_FACTOR,
        "model": ("gpt2 124M (public, fully trained) -- A PROXY, see LIMITATION"
                  if not args.tiny else "TINY in-process smoke test"),
        "architecture": {"n_layer": arch.n_layer, "n_head": arch.n_head,
                         "n_embd": arch.n_embd, "n_inner": arch.n_inner,
                         "parameters": arch.n_params},
        "recipe": [s.name for s in C.DEFAULT_RECIPE],
        "retired_recipe": [s.name for s in C.SORT_ONLY_RECIPE],
        "recipe_note": ("The SHIPPED recipe ends align_ffn_neurons and every "
                        "headline number here describes it. Anything under a "
                        "RETIRED key describes the superseded sort_ffn_neurons "
                        "recipe, kept so the measurement that retired it stays "
                        "reproducible. Those are NOT the study's ruler."),
        "seeds": list(seeds),
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform(),
                        "torch": torch.__version__},
        "epsilon_sweep_RETIRED_sort_recipe": retired_sweep or existing.get("epsilon_sweep_RETIRED_sort_recipe"),
        "symmetry_residual": residual or existing.get("symmetry_residual"),
        "epsilon_sweep": sweep or existing.get("epsilon_sweep"),
        "gauge_dimensions": gauge_dimensions(arch),
        "dispersion_sweep": dispersion or existing.get("dispersion_sweep"),
        "step_attribution": attribution or existing.get("step_attribution"),
        "ffn_sort_margins": sort_margins or existing.get("ffn_sort_margins"),
        "permuted_model_recovery": permuted or existing.get("permuted_model_recovery"),
        "sections_refreshed_this_run": sorted(want),
    }

    reportdir = Path(args.reportdir)
    reportdir.mkdir(parents=True, exist_ok=True)
    (reportdir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = format_report(payload)
    (reportdir / f"{stem}.md").write_text(text + "\n", encoding="utf-8",
                                          newline="\n")
    print(text, file=stream)
    print(f"\nwrote {reportdir / (stem + '.json')}", file=stream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
