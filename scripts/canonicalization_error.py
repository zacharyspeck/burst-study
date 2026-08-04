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
def open_question_distortion_factor(payload) -> str:
    """The distortion-factor open question, with its numbers READ FROM THE DATA.

    This text was a hardcoded string until 2026-08-03, and it was wrong: it
    still quoted 3.05 and 84.4, which are the RETIRED six-step recipe's
    figures. The shipped recipe reads 1.0004. So the report's own open-question
    section was asserting that the shipped ruler triples a small difference
    when the measurement one screen above said it does not -- the fifth time in
    this build that stale prose outlived the number it described.

    Deriving the figures from the payload is the structural fix. Prose that
    quotes a measurement cannot go stale if it reads the measurement.
    """
    iso = {c["epsilon"]: c for c in payload["epsilon_sweep"]["cells"].values()
           if c["shape"] == "isotropic"}
    lo = min(iso)
    cliff = _worst_seed_row(payload)
    flat = [e for e in sorted(iso)
            if cliff is None or e < cliff["epsilon"]]

    text = (
        "THE RULER'S DISTORTION FACTOR IS NOT A NUMBER, IT IS A RANGE, AND "
        "WHICH END APPLIES CANNOT BE KNOWN YET. With the FFN sort and the "
        f"head-internal step both retired, at eps={lo:g} the shipped recipe "
        f"reads {iso[lo]['ratio_median']:.4f} median with a per-seed spread of "
        f"[{iso[lo]['ratio_min']:.4f}, {iso[lo]['ratio_max']:.4f}]")
    text += (f", and it holds that through eps={max(flat):g}. "
             if flat else ". ")

    if cliff is not None:
        factor = cliff["ratio_max"] / cliff["ratio_median"]
        text += (
            f"IT DOES NOT HOLD EVERYWHERE. At eps={cliff['epsilon']:g} the "
            f"median is still {cliff['ratio_median']:.4f} while the worst of "
            f"{cliff['n_seeds']} seeds reads {cliff['ratio_max']:.4g} -- a "
            f"factor of {factor:.4g} between the median and the worst seed, "
            f"with {cliff['head_order_flips_total']} head-order flip(s) "
            f"recorded in that row. Logged as D-3. So the range is not a "
            f"smooth curve that can be read off at whatever epsilon turns out "
            f"to be real; it is near-neutral behaviour with a cliff in it, and "
            f"THE CLIFF IS INVISIBLE IN THE MEDIAN. ")
    else:
        text += (
            "No cliff appears anywhere in the swept range, so on this data the "
            "factor is a narrow band rather than a range. That is a statement "
            "about the epsilons and seeds actually swept and about nothing "
            "else. ")

    return text + (
        "Which regime is operative depends on how far a burst arm actually "
        "sits from its seed-matched twin after training, expressed as a "
        "fraction of the parameter norm -- and that quantity does not exist "
        "until models are trained. A ruler whose distortion depends on an "
        "unmeasured quantity is a WEAKNESS OF THE STUDY and is recorded here "
        "as an open question rather than as a footnote. Resolving it requires "
        "measuring the twin-vs-twin distance on real checkpoints and reading "
        "the curve in this file at that epsilon."
    )

#: What the SHIPPED recipe quotients. head_internal_transform is deliberately
#: absent -- D-1 removed that step -- and is measured separately so its
#: un-quotiented residual is visible rather than buried in the composed row.
RECIPE_SYMMETRIES = (
    "head_permutation",
    "ffn_neuron_permutation",
    "key_bias_shift",
    "value_bias_shift",
)

#: Confirmed symmetries the shipped recipe deliberately does NOT remove. Their
#: residual is expected to be large; reporting it is the point. Three of the
#: four candidate symmetries that survived the mathematics were retired for
#: instability or direction-dependence -- see S62, S65.
NOT_QUOTIENTED = ("head_internal_transform", "layernorm_gain_rescale",
                  "residual_permutation")


# ---------------------------------------------------------------------------
# distances
# ---------------------------------------------------------------------------


def _merge_seed_cell(prior, label, epsilon, seeds, ratios) -> dict:
    """Merge this chunk's per-seed values into whatever is already stored.

    Ten seeds is mandatory and does not fit the observed ~600s task-duration
    cap for most sections, so a section is measured in seed windows and merged.
    Per-seed values are kept so the merge is exact and so min/median/max are
    recomputed over the union rather than over one chunk.
    """
    cell = (prior or {}).get("variants", {}).get(label, {})
    cell = cell.get(f"{epsilon:g}") if isinstance(cell, dict) else None
    merged = dict(cell.get("by_seed", {}) if isinstance(cell, dict) else {})
    merged.update(dict(zip([str(s) for s in seeds], ratios)))
    vals = [merged[k] for k in sorted(merged, key=int)]
    return {"by_seed": merged, "n_seeds": len(vals), "min": min(vals),
            "median": statistics.median(vals), "max": max(vals)}


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


#: Cache of unit perturbation directions, keyed by (shape, seed).
#:
#: EXACT, NOT AN APPROXIMATION. The direction is drawn from a generator seeded
#: only by `seed`, so it is identical for every epsilon -- epsilon only scales
#: it. Redrawing per epsilon was 3.77s of an 8.80s measurement cell, 43% of the
#: total, spent recomputing a tensor that was bit-identical to the last one.
_DIRECTION_CACHE: dict = {}
_THETA_NORM_CACHE: dict = {}


def _unit_direction(model, shape: str, seed: int):
    """A unit-norm perturbation direction, cached per (shape, seed)."""
    import torch

    key = (shape, seed)
    if key in _DIRECTION_CACHE:
        return _DIRECTION_CACHE[key]

    gen = torch.Generator().manual_seed(seed)
    params = [p for _, p in model.named_parameters()]
    if shape == "isotropic":
        draws = [torch.randn(p.shape, generator=gen, dtype=torch.float64)
                 for p in params]
        norm = math.sqrt(sum(float(d.pow(2).sum()) for d in draws))
        direction = [d / norm for d in draws]
    elif shape == "per_tensor":
        direction = []
        for p in params:
            d = torch.randn(p.shape, generator=gen, dtype=torch.float64)
            dn = float(d.pow(2).sum()) ** 0.5
            pn = float(p.detach().double().pow(2).sum()) ** 0.5
            direction.append(d * (pn / dn) if dn and pn else d * 0.0)
        norm = math.sqrt(sum(float(d.pow(2).sum()) for d in direction))
        direction = [d / norm for d in direction]
    else:
        raise C.CanonicalizeError(f"unknown perturbation shape {shape!r}")
    _DIRECTION_CACHE[key] = direction
    return direction


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

    if "theta" not in _THETA_NORM_CACHE:
        _THETA_NORM_CACHE["theta"] = math.sqrt(
            sum(float(p.detach().double().pow(2).sum())
                for _, p in model.named_parameters()))
    target = epsilon * _THETA_NORM_CACHE["theta"]

    direction = _unit_direction(model, shape, seed)
    with torch.no_grad():
        for p, d in zip((q for _, q in model.named_parameters()), direction):
            p.add_((d * target).to(dtype=p.dtype))
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
    for name in RECIPE_SYMMETRIES + NOT_QUOTIENTED + ("__composed__",):
        ratios, raws, canons = [], [], []
        for seed in seeds:
            plain = build()
            moved = copy.deepcopy(plain)
            names = RECIPE_SYMMETRIES if name == "__composed__" else (name,)
            # NOT_QUOTIENTED entries are measured alone, and a large ratio
            # there is the expected, documented cost of D-1 rather than a
            # failure. The composed row uses only what the recipe removes.
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
            # n_seeds is recorded PER ROW, not just for the section, so the
            # file states its own coverage. Section A carried no seed count at
            # all until 2026-08-03: its ten seeds were asserted only by a
            # commit message, which is not somewhere a reader of the JSON can
            # look. Every other section stores per-seed values; A reduces to
            # medians as it goes, so the count is what it can honestly keep.
            "n_seeds": len(seeds),
            "seeds": list(seeds),
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
        "n_seeds": len(seeds),
        "seeds": list(seeds),
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
                          recipe=None, prior=None) -> dict:
    """Does canonicalization inflate the distance between two NEARLY IDENTICAL
    models? That is the question the study actually asks of this ruler."""
    import copy

    prior_cells = (prior or {}).get("cells")
    reference = build()
    ref_plain_sd = C.canonical_state_dict(reference)
    ref_report = C.canonicalize(reference, arch, recipe=recipe)
    ref_sd = C.canonical_state_dict(reference)
    # Computed independently rather than read off the report: the shipped
    # recipe no longer runs the head-internal step and therefore no longer
    # produces these, but they are a property of the MODEL and are still the
    # right thing to bucket heads by. Reading the report gave nan.
    conditions = list(C.head_condition_numbers(reference, arch))

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

            by_seed = dict(zip([str(s) for s in seeds], ratios))
            prior_cell = (prior_cells or {}).get(key, {})
            merged = dict(prior_cell.get("ratio_by_seed", {})
                          if isinstance(prior_cell, dict) else {})
            merged.update(by_seed)
            ratios = [merged[k] for k in sorted(merged, key=int)]
            spread = (max(ratios) / min(ratios)) if min(ratios) > 0 else float("inf")
            results[key] = {
                # Per-seed values are stored so a section can be measured in
                # chunks and merged. Ten seeds is mandatory (S55: four times a
                # low-seed number was overturned), and ten seeds does not fit
                # the observed task-duration cap for every section.
                "ratio_by_seed": merged,
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


def measure_step_attribution(build, arch, epsilons, seeds, stream,
                             prior=None) -> dict:
    """Which recipe step is responsible for the inflation in measurement B.

    Measurement B reports a pooled ratio. A pooled number that is large says
    something is wrong; it does not say what. This re-runs the same
    perturbation against recipes with individual steps removed, and against the
    Hungarian-matching route, so the inflation can be attributed to a step
    rather than guessed at.
    """
    import copy

    # Variants are defined against the SHIPPED recipe, and this list has gone
    # stale THREE times now as steps were retired -- each time producing a
    # table whose rows all agreed, which reads as a clean result rather than a
    # broken harness. See S61 and S69.
    #
    # Per-step attribution moved to measurement F, which has an EMPTY control.
    # What D is for now is the comparison against the three RETIRED recipes,
    # which is the thing F cannot show.
    variants = {
        "shipped_recipe": C.DEFAULT_RECIPE,
        "without_ffn_permutation": tuple(
            s for s in C.DEFAULT_RECIPE
            if not isinstance(s, (C.AlignFFNNeurons, C.SortFFNNeurons))),
        "without_head_sort": tuple(
            s for s in C.DEFAULT_RECIPE if not isinstance(s, C.SortHeads)),
        "RETIRED_gain_absorption": C.RETIRED_GAIN_ABSORPTION_RECIPE,
        "RETIRED_head_internal": C.RETIRED_HEAD_INTERNAL_RECIPE,
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
            cells[f"{epsilon:g}"] = _merge_seed_cell(
                prior, label, epsilon, seeds, ratios)
        rows[label] = cells
        print(f"    {label:<24} " + "  ".join(
            f"eps={k}: n={v['n_seeds']} [{v['min']:.4g}, {v['median']:.4g}, "
            f"{v['max']:.4g}]" for k, v in cells.items()),
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
        cells[f"{epsilon:g}"] = _merge_seed_cell(
            prior, "hungarian_alignment", epsilon, seeds, ratios)
    rows["hungarian_alignment"] = cells
    print(f"    {'hungarian_alignment':<24} " + "  ".join(
        f"eps={k}: n={v['n_seeds']} [{v['min']:.4g}, {v['median']:.4g}, "
        f"{v['max']:.4g}]" for k, v in cells.items()), file=stream, flush=True)
    del reference, ref_sd

    return {
        "note": ("median ratio under an isotropic perturbation, with recipe "
                 "steps removed one at a time, plus the Hungarian-matching "
                 "route in place of sorting. Attributes the pooled ratio in "
                 "measurement B to a specific step."),
        "variants": rows,
    }


def measure_permuted_recovery(build, arch, epsilons, seeds, stream,
                              prior=None) -> dict:
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
            cells[f"{epsilon:g}"] = _merge_seed_cell(
                prior, label, epsilon, seeds, ratios)
        rows[label] = cells
        print(f"    {label:<24} " + "  ".join(
            f"eps={k}: n={v['n_seeds']} [{v['min']:.4g}, {v['median']:.4g}, "
            f"{v['max']:.4g}]" for k, v in cells.items()),
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


def measure_step_contributions(build, arch, epsilons, seeds, stream,
                               prior=None) -> dict:
    """Where the shipped ruler's flat 0.907 comes from.

    0.907 is a 9% systematic CONTRACTION -- it makes two models look CLOSER
    than they are. It is flat across seven decades and stable across every
    seed, so it is not noise; it is a mechanism, and an unexplained systematic
    factor in a ruler is not shippable.

    Each remaining step is removed in turn. The EMPTY recipe is the control and
    must return exactly 1.0: with no steps, canon(M) is M and the ratio is
    ||M - (M+eps)|| / d_raw by construction. If the empty recipe does NOT
    return 1.0, the factor is in the measurement harness rather than in the
    ruler, which is a different and more serious problem.
    """
    import copy

    variants = {"shipped_recipe": C.DEFAULT_RECIPE, "EMPTY_control": ()}
    for step in C.DEFAULT_RECIPE:
        variants[f"without_{step.name}"] = tuple(
            s for s in C.DEFAULT_RECIPE if s is not step)

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
                C.canonicalize(moved, arch, recipe=recipe, reference=reference)
                ratios.append(
                    l2_distance(ref_sd, C.canonical_state_dict(moved)) / d_raw)
                del moved
            by_seed = dict(zip([str(s) for s in seeds], ratios))
            prior_cell = (prior or {}).get("variants", {}).get(label, {})
            prior_cell = prior_cell.get(f"{epsilon:g}") if isinstance(
                prior_cell, dict) else None
            # Tolerate the pre-chunking format, where a cell was a bare float.
            merged = dict(prior_cell.get("by_seed", {})
                          if isinstance(prior_cell, dict) else {})
            merged.update(by_seed)
            vals = [merged[k] for k in sorted(merged, key=int)]
            cells[f"{epsilon:g}"] = {
                "by_seed": merged, "n_seeds": len(vals),
                "min": min(vals), "median": statistics.median(vals),
                "max": max(vals)}
        rows[label] = cells
        print(f"    {label:<34} " + "  ".join(
            f"eps={k}: n={v['n_seeds']} "
            f"[{v['min']:.5f}, {v['median']:.5f}, {v['max']:.5f}]"
            for k, v in cells.items()),
            file=stream, flush=True)
        del reference, ref_sd

    return {
        "note": ("attribution of the shipped ruler's flat systematic factor. "
                 "EMPTY_control must be exactly 1.0 -- with no steps the ratio "
                 "is 1 by construction, so any deviation there would put the "
                 "factor in the harness rather than the ruler."),
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


#: Column header for the D and E tables. Shared so the header and the row
#: renderer cannot drift apart into a table whose columns are mislabelled.
_CELL_HEADER = (f"{'recipe variant':<26}{'eps':>9}{'n':>4}"
                f"{'min':>14}{'median':>14}{'max':>14}")


def _cell_row(label, eps, cell, width=26):
    """Render a measurement cell, tolerating the pre-chunking float format.

    Sections are refreshed independently, so a file can legitimately hold new
    per-seed cells for one section and old bare floats for another. Rendering
    has to survive that rather than crash on the mix -- the banner is what
    tells the reader which is which.

    THE EPSILON IS PRINTED. It was not until 2026-08-03: the caller looped over
    epsilons and dropped the key, which was harmless only because D and E had
    been cut to a single epsilon to afford ten seeds. Restore a second epsilon
    and the table would have silently grown duplicate rows with the same label
    and no way to tell them apart -- the same defect class as S61 and S69,
    where a table reads clean because the harness is broken. Fixed while it
    costs nothing rather than after it costs a conclusion.
    """
    if isinstance(cell, dict):
        return (f"{label:<{width}}{eps:>9}{cell['n_seeds']:>4}"
                f"{cell['min']:>14.5g}{cell['median']:>14.5g}"
                f"{cell['max']:>14.5g}")
    return (f"{label:<{width}}{eps:>9}{'?':>4}{'':>14}{cell:>14.5g}"
            f"{'':>14}   [stale float]")


#: Ten seeds is the floor for every seed-bearing section. The FFN sort's
#: failure was seed-dependent and three low-seed results have since been
#: overturned by widening, so a section below this is not reportable.
MIN_SEEDS = 10

#: Sections whose EPSILON coverage was deliberately narrowed to afford ten
#: seeds, and what they swept before. D and E swept these three at three seeds
#: through c155f08; on 2026-08-03 they were cut to one epsilon at ten seeds.
#:
#: F is deliberately NOT listed. F has always run at a single epsilon, so
#: describing it as narrowed would attribute a trade that never happened -- the
#: banner derives which sections are thin from the data, but "down from three"
#: is a historical claim and only the sections that actually paid it get it.
EPSILON_NARROWED = {
    "D": (1e-8, 1e-6, 1e-4),
    "E": (1e-8, 1e-6, 1e-4),
}


def _variant_cells(section):
    """Yield (epsilon_key, cell) over a D/E/F-shaped section."""
    for cells in (section or {}).get("variants", {}).values():
        for eps, cell in cells.items():
            yield eps, cell


def seed_coverage(payload) -> dict:
    """Per-section seed count and epsilon list, READ FROM THE CELLS.

    The top-level "seeds" key records the last chunk's WINDOW, not the
    coverage, because sections are measured in seed windows and merged. Anyone
    reading it as coverage gets the wrong answer -- so coverage is derived from
    the cells themselves, which is the only place it is actually true.
    """
    out = {}

    a = payload.get("symmetry_residual") or {}
    out["A"] = {"n_seeds": a.get("n_seeds"), "epsilons": []}
    out["C"] = {"n_seeds": None, "epsilons": []}

    for letter, key in (("B", "epsilon_sweep"),
                        ("R", "epsilon_sweep_RETIRED_sort_recipe")):
        cells = (payload.get(key) or {}).get("cells", {})
        counts = [c.get("n_seeds") for c in cells.values()]
        out[letter] = {
            "n_seeds": min(counts) if counts and None not in counts else None,
            "epsilons": sorted({c["epsilon"] for c in cells.values()}),
        }

    for letter, key in (("D", "step_attribution"),
                        ("E", "permuted_model_recovery"),
                        ("F", "step_contributions")):
        pairs = list(_variant_cells(payload.get(key)))
        counts = [c["n_seeds"] for _, c in pairs if isinstance(c, dict)]
        out[letter] = {
            "n_seeds": min(counts) if counts else None,
            "epsilons": sorted({float(e) for e, _ in pairs}),
        }
    return out


#: How far the worst seed must exceed the median before the spread is a
#: finding rather than float noise. A row at this gap is one a median-only
#: reader would miss entirely, which is the whole reason ranges are reported.
SEED_SPREAD_ALERT = 1.5


def _worst_seed_row(payload):
    """The isotropic B cell whose worst seed most exceeds its median.

    Returns None when NO row exceeds SEED_SPREAD_ALERT. That branch matters:
    the read-the-range warning must be earned by the data. Asserting a cliff
    unconditionally is how prose comes to describe a measurement it is no
    longer attached to -- the exact failure this function is part of fixing.
    """
    iso = [c for c in payload["epsilon_sweep"]["cells"].values()
           if c["shape"] == "isotropic" and c.get("ratio_median")]
    if not iso:
        return None
    worst = max(iso, key=lambda c: c["ratio_max"] / c["ratio_median"])
    if worst["ratio_max"] / worst["ratio_median"] < SEED_SPREAD_ALERT:
        return None
    return worst


def build_provenance(payload) -> str:
    """The PROVENANCE record, DERIVED from the payload rather than hand-written.

    This key was hand-added to the JSON on 2026-08-02 and the script neither
    wrote nor read it, so the next run would have silently deleted it -- a
    provenance record that disappears is worse than none, because its absence
    looks like the file was always machine-clean. Same for the markdown banner.
    Both are now generated here, from the data they describe.
    """
    cov = seed_coverage(payload)
    worst = _worst_seed_row(payload)
    measured = [k for k in ("A", "B", "C", "D", "E", "F")
                if payload.get({"A": "symmetry_residual",
                                "B": "epsilon_sweep",
                                "C": "dispersion_sweep",
                                "D": "step_attribution",
                                "E": "permuted_model_recovery",
                                "F": "step_contributions"}[k])]
    seeded = {k: cov[k]["n_seeds"] for k in measured if cov[k]["n_seeds"]}
    thin_eps = sorted(k for k in ("B", "D", "E", "F")
                      if len(cov[k]["epsilons"]) == 1)

    parts = [
        f"Sections {', '.join(measured)} are all measured against the SHIPPED "
        f"recipe ({' -> '.join(payload['recipe'])}).",
        "Seed coverage is read from the cells, not from the top-level 'seeds' "
        "key, which records only the LAST CHUNK'S WINDOW: "
        + "; ".join(f"{k} at {n} seeds" for k, n in sorted(seeded.items()))
        + ". Section C is a sweep over interpolation level and carries no seed "
          "dimension.",
        f"The floor is {MIN_SEEDS} seeds and every seeded section meets it."
        if all(n >= MIN_SEEDS for n in seeded.values())
        else "AT LEAST ONE SECTION IS BELOW THE TEN-SEED FLOOR: "
             + ", ".join(f"{k}={n}" for k, n in sorted(seeded.items())
                         if n < MIN_SEEDS) + ".",
    ]

    if thin_eps:
        narrowed = [k for k in thin_eps if k in EPSILON_NARROWED]
        detail = "; ".join(
            f"{k} at eps={cov[k]['epsilons'][0]:g} only" for k in thin_eps)
        note = (
            f"COVERAGE TRADED FOR SEEDS, AND IT CUTS BOTH WAYS: {detail}. ")
        if narrowed:
            was = ", ".join(
                f"{k} was {', '.join(f'{e:g}' for e in EPSILON_NARROWED[k])}"
                for k in narrowed)
            note += (
                f"{was} -- three epsilons at three seeds, until ten seeds at "
                f"three epsilons proved not to fit the ~600s task-duration "
                f"cap. Epsilon breadth was spent to buy seed breadth. ")
        note += (
            "The reason ten seeds is mandatory is that low-seed numbers in "
            "this build kept being overturned by widening. A single epsilon "
            "has the MIRROR-IMAGE weakness")
        note += (
            f", and section B demonstrates it directly here: the ruler holds "
            f"flat across the low decades and then steps off a cliff at "
            f"eps={worst['epsilon']:g}, which a section measured at one "
            f"epsilon could not have seen. "
            if worst is not None else
            ", since a single epsilon cannot show where the ruler stops "
            "behaving, only that it behaves at one point. ")
        note += "Both limits are live and neither substitutes for the other."
        parts.append(note)

    parts.append(
        "Big sections were measured in seed windows and merged per seed -- a "
        "cell costs about 4.8s -- and every merged cell stores its per-seed "
        "values, so min/median/max are computed over the union of chunks "
        "rather than over one chunk.")
    if worst is not None:
        parts.append(
            f"READ THE RANGE, NOT THE MEDIAN: at eps={worst['epsilon']:g} "
            f"section B's median is {worst['ratio_median']:.4f} and its worst "
            f"of {worst['n_seeds']} seeds is {worst['ratio_max']:.4g}, with "
            f"{worst['head_order_flips_total']} head-order flip(s). Logged as "
            f"D-3. A median-only report would have shown a flawless ruler.")
    else:
        parts.append(
            "No row in section B shows a median-to-worst-seed gap above "
            f"{SEED_SPREAD_ALERT}x, so no seed-spread alert is raised for this "
            "data. Ranges are still reported everywhere; the absence of a "
            "flagged row is a result, not a reason to read medians.")

    retired = payload.get("epsilon_sweep_RETIRED_sort_recipe")
    if retired:
        parts.append(
            "Section R is the retired sort-based recipe and is deliberately "
            "NOT re-measured: SORT_ONLY_RECIPE has not changed, so its numbers "
            "remain valid for the recipe they describe. Its cells predate "
            "per-seed storage and cannot be re-merged.")
    return " ".join(parts)


def report_banner(payload) -> list:
    """The markdown report's header banner, generated from the same facts.

    Hand-editing this banner into the generated file is what made it fragile:
    every previous banner was destroyed by the next run of this script. It is
    generated so that it cannot be.
    """
    cov = seed_coverage(payload)
    worst = _worst_seed_row(payload)
    seeded = {k: v["n_seeds"] for k, v in cov.items() if v["n_seeds"]}
    thin_eps = sorted(k for k in ("B", "D", "E", "F")
                      if len(cov[k]["epsilons"]) == 1)
    floor_ok = seeded and all(n >= MIN_SEEDS for n in seeded.values())

    lines = ["ALL SECTIONS MEASURED AGAINST THE SHIPPED RECIPE"
             if floor_ok else "*** A SECTION IS BELOW THE TEN-SEED FLOOR ***",
             "",
             "Shipped recipe: " + ", ".join(payload["recipe"]) + ".",
             "",
             "Seed coverage, read from the cells and not from the top-level",
             "'seeds' key (which is only the last chunk's window):",
             "  " + "   ".join(f"{k}={n}" for k, n in sorted(seeded.items()))
             + "   (C is a sweep over t, no seed dimension)"]

    if thin_eps:
        narrowed = [k for k in thin_eps if k in EPSILON_NARROWED]
        lines += [
            "",
            "TWO LIMITATIONS, AND THEY ARE MIRROR IMAGES. Read them together.",
            "",
            "  1. TEN SEEDS IS THE FLOOR because low-seed numbers in this",
            "     build kept being overturned when the seed count widened.",
            f"  2. {', '.join(thin_eps)} run at a SINGLE EPSILON "
            f"(eps={cov[thin_eps[0]]['epsilons'][0]:g})."]
        if narrowed:
            lines += [
                f"     {', '.join(narrowed)} previously swept "
                f"{', '.join(f'{e:g}' for e in EPSILON_NARROWED[narrowed[0]])}"
                " at three seeds.",
                "     Ten seeds did not fit the ~600s task cap at three",
                "     epsilons, so epsilon breadth was spent to buy seed",
                "     breadth."]
        lines += (
            ["     Section B shows the ruler holding flat across the low",
             f"     decades and then stepping off a cliff at "
             f"eps={worst['epsilon']:g}",
             "     -- a section at a single epsilon CANNOT SEE THAT CLIFF."]
            if worst is not None else
            ["     A single epsilon shows only that the ruler behaves at one",
             "     point, never where it stops behaving."])
        lines += [
            "",
            "  Neither limit substitutes for the other. A wide-seed,",
            "  one-epsilon result and a one-seed, wide-epsilon result are",
            "  both partial, in opposite directions."]

    if worst is not None:
        lines += [
            "",
            "READ THE RANGE, NOT THE MEDIAN. At "
            f"eps={worst['epsilon']:g} section B's median is",
            f"{worst['ratio_median']:.4f} and its worst of "
            f"{worst['n_seeds']} seeds is {worst['ratio_max']:.4g}, with "
            f"{worst['head_order_flips_total']} head-order flip(s).",
            "A median-only report would have shown nothing. Logged as D-3."]
    else:
        lines += [
            "",
            "No section B row shows a median-to-worst-seed gap above "
            f"{SEED_SPREAD_ALERT}x.",
            "Ranges are reported regardless; read them, not the medians."]

    if payload.get("epsilon_sweep_RETIRED_sort_recipe"):
        lines += [
            "",
            "Section R is the retired sort-based recipe and is deliberately",
            "not re-measured -- SORT_ONLY_RECIPE is unchanged, so its numbers",
            "remain valid for the recipe they describe."]
    return lines


def format_report(payload) -> str:
    rule = "=" * 78
    thin = "-" * 78
    lines = ["", rule] + report_banner(payload) + [rule, ""]
    lines += [rule, "STEP 9 -- CANONICALIZATION ERROR BAR", rule,
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
    lines.append(_CELL_HEADER)
    for label, cells in attr["variants"].items():
        for eps, c in cells.items():
            lines.append(_cell_row(label, eps, c))

    contrib = payload.get("step_contributions")
    if contrib:
        lines += ["", rule,
                  "F. STEP CONTRIBUTIONS -- where the systematic factor comes from",
                  rule,
                  "Each remaining step removed in turn. The EMPTY control must",
                  "return exactly 1.0: with no steps the ratio is 1 by",
                  "construction, so any deviation there would put the factor in",
                  "the harness rather than the ruler.",
                  "",
                  f"{'variant':<36}{'eps':>9}{'n':>4}{'min':>11}"
                  f"{'median':>11}{'max':>11}"]
        # Prints eps for the same reason D and E do: this loop had the key in
        # hand and dropped it, which is invisible at one epsilon and ambiguous
        # at two. F has only ever run at one, which is exactly how a latent
        # defect stays latent.
        for label, cells in contrib["variants"].items():
            for eps, c in cells.items():
                lines.append(f"{label:<36}{eps:>9}{c['n_seeds']:>4}"
                             f"{c['min']:>11.5f}{c['median']:>11.5f}"
                             f"{c['max']:>11.5f}")

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
    lines.append(_CELL_HEADER)
    for label, cells in perm["variants"].items():
        for eps, c in cells.items():
            lines.append(_cell_row(label, eps, c))

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
        "--seed-window", default=None, metavar="A:B",
        help=("run only seeds[A:B] and MERGE the per-seed results into what is "
              "already in the file. Ten seeds is mandatory and ten seeds does "
              "not fit the observed task-duration cap for every section, so "
              "chunking by seed is how a section gets to ten without a single "
              "run long enough to be killed."))
    parser.add_argument(
        "--render-only", action="store_true",
        help=("re-render the .md and the DERIVED json fields from the "
              "measurements already in the .json, measuring nothing. The "
              "report's formatting and its provenance text are pure functions "
              "of the payload, so a change to either must not cost a "
              "re-measurement -- that cost is what tempted hand-editing the "
              "generated file in the first place."))
    parser.add_argument(
        "--sections", default="ABCDEFR",
        help=("which measurements to run, e.g. DE. A subset MERGES into the "
              "existing results file rather than replacing it, so an "
              "expensive section does not have to be recomputed to correct a "
              "cheap one. The merged file records which sections were "
              "refreshed and when."))
    return parser


def _write_report(payload, reportdir: Path, stem: str, stream) -> int:
    """Recompute every DERIVED field, then write both files.

    The derived fields are computed here, last, from the assembled payload --
    so they describe what is actually in the file rather than what the caller
    believed was going into it. PROVENANCE and the report banner were both
    hand-maintained until 2026-08-03 and both were one regeneration away from
    being silently deleted. See S70.
    """
    payload["OPEN_QUESTION_distortion_factor"] = (
        open_question_distortion_factor(payload))
    payload["seed_coverage"] = seed_coverage(payload)
    payload["PROVENANCE"] = build_provenance(payload)

    reportdir.mkdir(parents=True, exist_ok=True)
    (reportdir / f"{stem}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = format_report(payload)
    (reportdir / f"{stem}.md").write_text(text + "\n", encoding="utf-8",
                                          newline="\n")
    print(text, file=stream)
    print(f"\nwrote {reportdir / (stem + '.json')}", file=stream)
    return 0


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    stream = sys.stdout
    rule = "=" * 78
    print(rule, file=stream)
    print("canonicalization_error -- step 9 phase 5", file=stream)
    print(rule, file=stream)

    stem = REPORT_STEM + ("-tiny" if args.tiny else "")
    if args.render_only:
        # Deliberately BEFORE torch is imported: re-rendering measurements that
        # already exist is not an ML operation and must not require an ML
        # stack. Same reason burst/ stays importable without torch.
        source = Path(args.reportdir) / f"{stem}.json"
        if not source.is_file():
            print(f"nothing to render: {source} does not exist", file=stream)
            return 1
        print(f"render-only: re-rendering {source}, measuring nothing",
              file=stream)
        payload = json.loads(source.read_text(encoding="utf-8"))
        return _write_report(payload, Path(args.reportdir), stem, stream)

    import copy
    import torch

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
    if args.seed_window:
        a, b = (int(x) if x else None for x in args.seed_window.split(":"))
        seeds = seeds[a:b]
    want = set(args.sections.upper())
    existing = {}
    prior = Path(args.reportdir) / f"{stem}.json"
    if want != set("ABCDEFR") and prior.is_file():
        existing = json.loads(prior.read_text(encoding="utf-8"))
        print(f"merging sections {sorted(want)} into {prior}", file=stream)
    residual = sweep = retired_sweep = dispersion = None
    attribution = sort_margins = permuted = contributions = None

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
                                      recipe=C.DEFAULT_RECIPE,
                                      prior=existing.get("epsilon_sweep"))

    if "R" in want:
        # Separate letter on purpose: SORT_ONLY_RECIPE has not changed, so its
        # numbers stay valid and re-measuring them costs ~22 minutes for no new
        # information. Only run R if that recipe is edited.
        print(chr(10) + 'R. epsilon sweep  [RETIRED sort recipe]', file=stream)
        retired_sweep = measure_epsilon_sweep(
            build, arch, EPSILONS, ('isotropic',), seeds, stream,
            recipe=C.SORT_ONLY_RECIPE,
            prior=existing.get("epsilon_sweep_RETIRED_sort_recipe"))

    if "C" in want:
        print("\nC. dispersion sweep", file=stream)
        dispersion = measure_dispersion_sweep(build, arch, DISPERSION_LEVELS, stream)

    if "D" in want:
        print("\nD. step attribution", file=stream)
        attribution = measure_step_attribution(
            build, arch, (1e-6,), seeds, stream,
            prior=existing.get("step_attribution"))
        sort_margins = measure_sort_margins(build, arch, stream)

    if "F" in want:
        print(chr(10) + "F. step contributions  [attributing the flat factor]",
              file=stream)
        contributions = measure_step_contributions(
            build, arch, (1e-6,), seeds, stream,
            prior=existing.get("step_contributions"))

    if "E" in want:
        print("\nE. permuted-model recovery", file=stream)
        permuted = measure_permuted_recovery(
            build, arch, (1e-6,), seeds, stream,
            prior=existing.get("permuted_model_recovery"))

    payload = {
        "task": "step 9 phase 5",
        "LIMITATION": LIMITATION,
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
        "step_contributions": contributions or existing.get("step_contributions"),
        "sections_refreshed_this_run": sorted(want),
    }

    return _write_report(payload, Path(args.reportdir), stem, stream)


if __name__ == "__main__":
    raise SystemExit(main())
