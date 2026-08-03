#!/usr/bin/env python
"""Measure the step 10 metrics on junk checkpoints. Step 10, first half.

WHAT THIS REPORT IS, AND WHAT IT IS NOT

It is a PLUMBING result. There are no trained checkpoints in this study and
there will not be for weeks, so every number here is measured on throwaway
models built in process from a pinned seed. Junk models sit at chance loss, so
the barrier in particular has nothing to climb over -- a max_excess of zero
here is the expected outcome and says nothing whatever about whether the metric
would find a barrier between two real checkpoints.

What the report DOES establish is that each metric responds: that identical
models score identically, that different models do not, and that the two
independent routes to layer activations agree. That is the whole claim.

Every derived number in this report is computed from the payload. Nothing is
hardcoded prose. Five separate times in step 9 a hardcoded interpretation
outlived the number it described, and the fifth was shipping a claim
contradicted by a table one screen below it -- so the S70 pattern is built in
here from the first commit rather than retrofitted after the sixth.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import metrics as MX  # noqa: E402

DEFAULT_REPORT_DIR = REPO_ROOT / "docs" / "measurements"
REPORT_STEM = "10-metrics"

#: Widened for the same reason step 9 widened: low-seed numbers in this build
#: were overturned three separate times when the seed count grew.
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)
MIN_SEEDS = 10

#: Which junk pairs are measured, and the DIRECTION each one is expected to
#: move. Stated here so the report can check its own results against the
#: expectation rather than presenting whatever came out as the finding.
PAIR_EXPECTATION = {
    "identical": "every metric at its identical value; any deviation is a bug",
    "zeroed_block": "layers before the zeroed block unchanged, layers at or "
                    "after it changed",
    "noise": "strictly between identical and independent on every metric",
    "independent": "far apart on L2, CKA and cosine",
}

def limitation(payload) -> str:
    """The limitations, with the batch-dependent ones READ FROM THE BATCH.

    Recomputed on every write rather than frozen into the payload at
    measurement time: it is static text owned by this script plus facts derived
    from the batch identity, so a stored copy could only ever go stale. The
    same reasoning as PROVENANCE.
    """
    batch = payload.get("batch") or {}
    n_tokens = batch.get("n_tokens")
    source = batch.get("source", "an unrecorded batch")

    text = (
        "EVERY NUMBER IN THIS FILE IS MEASURED ON JUNK CHECKPOINTS. No trained "
        "model exists in this study and none will for weeks, so these are "
        "throwaway models built in process from pinned seeds and never written "
        "to disk. The consequence is not a caveat, it is the frame: this is a "
        "PLUMBING result showing that each metric responds to a difference, "
        "and it is not a measurement of anything about training. No number "
        "here should be quoted as a property of the study's models. "
    )

    if n_tokens:
        text += (
            f"THE ACTIVATION METRICS REST ON A THIN BASIS. CKA and activation "
            f"similarity are computed over {n_tokens} token positions drawn "
            f"from a SINGLE DOCUMENT ({source}), and for CKA those positions "
            f"are the entire sample -- it is a statement about a sample of "
            f"activations, and one document is a narrow one. This is a "
            f"limitation of the numbers in this file as they stand, not merely "
            f"of a decision left untaken, and it is independent of which text "
            f"is used. The held-out corpus slice offers 10,240 sequences "
            f"against this one, and would be the better basis. The batch is "
            f"deliberately NOT being changed: every number here is keyed to a "
            f"batch identity, so a swap would make old and new numbers "
            f"incomparable, and that is a change to make once against trained "
            f"checkpoints rather than twice. "
        )
        text += (
            "That text is ALSO corpus-derived -- bursts/context.txt is "
            "openwebtext document 73 -- so it is safe from a memorisation "
            "confound only because the held-out slice is taken from the FRONT "
            "of the corpus. See cross-module obligation 1."
        )
    return text


def cannot_answer(payload) -> str:
    """What this half of step 10 cannot tell you. DERIVED, so it cannot rot.

    Someone reading this file in two months must not be able to mistake it for
    the metrics module being finished. The absent metrics are listed by asking
    the module which ones raise, rather than by a hand-written list that would
    quietly become wrong the moment one of them is implemented.
    """
    blocked = []
    for name in ("aligned_barrier", "aligned_l2", "rsf_subspace_probe"):
        fn = getattr(MX, name, None)
        if fn is None:
            continue
        try:
            fn()
        except NotImplementedError:
            blocked.append(name)
        except Exception:  # pragma: no cover - defensive
            pass
    built = ["barrier", "l2_distance_raw", "activation_similarity",
             "per_layer_cka"]
    return (
        "THIS IS HALF OF STEP 10 AND THE METRICS MODULE IS NOT DONE. "
        f"Built and measured here: {', '.join(built)}. "
        f"NOT built, and still raising NotImplementedError: "
        f"{', '.join(blocked) if blocked else 'none'} -- all of which route "
        "through scripts/canonicalize.py, which is Conv1D-only, while which "
        "layout the study trains is undecided in this repo. "
        "So this file cannot tell you: how far two checkpoints are once "
        "permutation gauge is removed, whether a barrier survives alignment, "
        "or anything about the RSF subspace. "
        "It also cannot tell you anything about TRAINED models, because none "
        "exist. See docs/layout-cost.md and open question 3 in "
        "docs/step9-summary.md."
    )


# ---------------------------------------------------------------------------
# Cells: per-seed values retained, min/median/max over the union of chunks
# ---------------------------------------------------------------------------


def _merge_cell(prior_cell, seeds, values) -> dict:
    """Merge this chunk's per-seed values into whatever is already stored.

    Per-seed values are kept so a merge is exact and so min/median/max are
    recomputed over the UNION of chunks rather than over one chunk. Same
    machinery as step 9, and for the same reason: ten seeds does not always fit
    a single run under the observed task-duration cap.
    """
    merged = dict((prior_cell or {}).get("by_seed", {}))
    merged.update({str(s): float(v) for s, v in zip(seeds, values)})
    ordered = [merged[k] for k in sorted(merged, key=int)]
    return {
        "by_seed": merged,
        "n_seeds": len(ordered),
        "min": min(ordered),
        "median": statistics.median(ordered),
        "max": max(ordered),
    }


def _prior_cell(existing, *path):
    node = existing
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node if isinstance(node, dict) else None


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


def measure_pair(kind, seeds, batch, alphas, tiny, stream, existing=None):
    """One junk-pair kind across the seed window."""
    existing = existing or {}
    l2_vals, bar_max, bar_min, rose = [], [], [], []
    cka_by_layer, cos_by_layer, norm_by_layer = {}, {}, {}

    for seed in seeds:
        a, b = MX.build_junk_pair(kind, seed=seed, tiny=tiny)
        l2_vals.append(MX.l2_distance_raw(a, b))

        result = MX.barrier(a, b, batch, alphas=alphas)
        bar_max.append(result["max_excess"])
        bar_min.append(result["min_excess"])
        rose.append(bool(result["rose_above_chord"]))

        acts_a = MX.layer_activations(a, batch)
        acts_b = MX.layer_activations(b, batch)
        for row in MX.per_layer_cka(acts_a, acts_b):
            cka_by_layer.setdefault(row["layer"], []).append(row["cka"])
        for row in MX.per_layer_activation_similarity(acts_a, acts_b):
            cos_by_layer.setdefault(row["layer"], []).append(
                row["cosine_median"])
            norm_by_layer.setdefault(row["layer"], []).append(
                row["norm_ratio_median"])
        del a, b, acts_a, acts_b
        print(f"    {kind:<14} seed={seed:<5} l2={l2_vals[-1]:.5g} "
              f"barrier_max={bar_max[-1]:.4g}", file=stream, flush=True)

    return {
        "expectation": PAIR_EXPECTATION[kind],
        "l2_raw": _merge_cell(
            _prior_cell(existing, "l2_raw"), seeds, l2_vals),
        "barrier_max_excess": _merge_cell(
            _prior_cell(existing, "barrier_max_excess"), seeds, bar_max),
        "barrier_min_excess": _merge_cell(
            _prior_cell(existing, "barrier_min_excess"), seeds, bar_min),
        # NOT stored as a count. A scalar summed inside one call keeps only
        # this chunk's seeds and silently reads as a total; the count is
        # derived from barrier_max_excess.by_seed in barriers_above_noise.
        "barrier_rose_above_chord_this_chunk": sum(rose),
        "cka": {str(k): _merge_cell(
            _prior_cell(existing, "cka", str(k)), seeds, v)
            for k, v in sorted(cka_by_layer.items())},
        "cosine": {str(k): _merge_cell(
            _prior_cell(existing, "cosine", str(k)), seeds, v)
            for k, v in sorted(cos_by_layer.items())},
        "norm_ratio": {str(k): _merge_cell(
            _prior_cell(existing, "norm_ratio", str(k)), seeds, v)
            for k, v in sorted(norm_by_layer.items())},
    }


def measure_route_cross_check(seeds, batch, tiny, stream, prior=None):
    """The hooks-vs-native agreement, run at every seed. FIRST-CLASS.

    Not a one-off sanity check at import time: it runs on every model this
    report measures, because a tap list that is right for one model and wrong
    for another is exactly the failure it exists to catch.

    MERGED ACROSS CHUNKS, per seed, for the same reason the metric cells are:
    a run measures one seed window, and recording only the last window would
    understate how many models the check actually cleared.
    """
    by_seed = dict((prior or {}).get("by_seed", {}))
    for seed in seeds:
        model = MX.build_junk_model(seed=seed, tiny=tiny)
        report = MX.cross_check_activation_routes(model, batch)
        by_seed[str(seed)] = report["worst_abs_gap"]
        del model
    worst = list(by_seed.values())
    print(f"    cross-check    {len(by_seed)} seeds total, worst gap "
          f"{max(worst):.3e}", file=stream, flush=True)
    return {
        "by_seed": by_seed,
        "n_seeds": len(by_seed),
        "worst_abs_gap": max(worst),
        "all_agree": True,
        "note": ("hooks and output_hidden_states must reach identical tensors. "
                 "This is the S55 independent-route pattern applied before a "
                 "bug exists rather than after one was found. On disagreement "
                 "the run STOPS rather than picking a route -- which route is "
                 "correct cannot be determined from the disagreement itself."),
    }


# ---------------------------------------------------------------------------
# Derived records -- S70 pattern, present from the first commit
# ---------------------------------------------------------------------------


def seed_coverage(payload) -> dict:
    """Per-pair seed count, READ FROM THE CELLS.

    The top-level "seeds" key records the last chunk's WINDOW, not the
    coverage, because sections are measured in seed windows and merged.
    """
    out = {}
    for kind, section in (payload.get("junk_pairs") or {}).items():
        counts = []
        for key in ("l2_raw", "barrier_max_excess"):
            cell = section.get(key)
            if isinstance(cell, dict) and "n_seeds" in cell:
                counts.append(cell["n_seeds"])
        for group in ("cka", "cosine"):
            for cell in (section.get(group) or {}).values():
                if isinstance(cell, dict) and "n_seeds" in cell:
                    counts.append(cell["n_seeds"])
        out[kind] = min(counts) if counts else None
    return out


def _identical_pair_deviation(payload):
    """How far the identical pair strays from its required values.

    Two quantities, kept apart on purpose because they mean different things:

    `weight_deviation` covers the metrics that compare two models directly --
    L2, CKA, cosine. Two copies of one model must be indistinguishable to all
    three, and any deviation at all is a defect.

    `barrier_noise` is the identical pair's max_excess, and it is NOT expected
    to be exactly zero. The barrier evaluates interpolated weights, and
    (1 - a) * w + a * w is not bitwise w in floating point for most a, so the
    interior alphas run a model a few ulps away from the endpoints. The
    resulting excess is float noise in the loss, not a barrier. Reporting it
    under the same heading as the weight metrics would either look like a
    defect or, worse, train a reader to ignore a number that should be zero.
    """
    section = (payload.get("junk_pairs") or {}).get("identical")
    if not section:
        return None
    worst = abs(section["l2_raw"]["max"])
    for group in ("cka", "cosine"):
        for cell in (section.get(group) or {}).values():
            worst = max(worst, abs(cell["min"] - 1.0), abs(cell["max"] - 1.0))
    return {
        "weight_deviation": worst,
        "barrier_noise": abs(section["barrier_max_excess"]["max"]),
    }


def barrier_noise_floor(payload):
    """The identical pair's worst max_excess: the floor below which nothing counts.

    MEASURED, NOT ASSUMED. The barrier evaluates interpolated weights, and
    (1 - a) * w + a * w is not bitwise w in float, so even two copies of one
    model show a small positive excess -- 2.7e-08 median at GPT-2 shapes, and
    the curve technically "rose above the chord" on half the seeds.

    Counting those as barriers would put a defect in the report where there is
    only arithmetic. So every rise is counted against THIS number rather than
    against zero.
    """
    section = (payload.get("junk_pairs") or {}).get("identical")
    if not section:
        return None
    return abs(section["barrier_max_excess"]["max"])


def barriers_above_noise(payload) -> dict:
    """Per pair, how many seeds rose at all and how many cleared the floor.

    BOTH COUNTS ARE DERIVED FROM by_seed, never stored. An earlier version
    stored `rose_above_chord` as a scalar summed inside one measurement call,
    which meant a chunked run kept only the LAST chunk's count: the report
    printed "rose above chord: 4 of 10 seeds (8 of them above the noise
    floor)", which is arithmetically impossible and was the only reason the
    bug was visible at all. Anything that summarizes across seeds has to be
    computed from the per-seed values, exactly like min/median/max.

    A seed rose iff its max_excess is above zero: the endpoints contribute
    excess exactly 0 by construction, so a positive maximum can only come from
    an interior alpha.
    """
    floor = barrier_noise_floor(payload)
    out = {}
    for kind, section in (payload.get("junk_pairs") or {}).items():
        by_seed = section.get("barrier_max_excess", {}).get("by_seed", {})
        out[kind] = {
            "n_rose_above_chord": sum(1 for v in by_seed.values() if v > 0.0),
            "n_above_noise_floor": sum(
                1 for v in by_seed.values()
                if floor is not None and v > floor),
            "n_seeds": len(by_seed),
            "max_excess": section["barrier_max_excess"]["max"],
        }
    return out


def _responsiveness(payload):
    """Did the metrics actually separate identical from different?

    The whole claim of this report, computed from the cells.
    """
    pairs = payload.get("junk_pairs") or {}
    out = {}
    for kind in ("zeroed_block", "noise", "independent"):
        section = pairs.get(kind)
        if not section:
            continue
        ckas = [c["median"] for c in (section.get("cka") or {}).values()]
        out[kind] = {
            "l2_median": section["l2_raw"]["median"],
            "lowest_layer_cka_median": min(ckas) if ckas else None,
            "separated_from_identical": bool(section["l2_raw"]["min"] > 0.0),
        }
    return out


def build_provenance(payload) -> str:
    coverage = seed_coverage(payload)
    seeded = {k: v for k, v in coverage.items() if v}
    floor_ok = seeded and all(v >= MIN_SEEDS for v in seeded.values())
    dev = _identical_pair_deviation(payload)
    resp = _responsiveness(payload)

    parts = [
        f"Metrics measured: barrier, raw L2, activation similarity and "
        f"per-layer CKA ({payload['cka_variant']}), on "
        f"{len(coverage)} junk checkpoint pairs.",
        "Seed coverage is read from the cells, not from the top-level 'seeds' "
        "key, which records only the LAST CHUNK'S WINDOW: "
        + "; ".join(f"{k} at {v} seeds" for k, v in sorted(seeded.items()))
        + ".",
        f"The floor is {MIN_SEEDS} seeds and every pair meets it."
        if floor_ok else
        "AT LEAST ONE PAIR IS BELOW THE TEN-SEED FLOOR: "
        + ", ".join(f"{k}={v}" for k, v in sorted(seeded.items())
                    if v < MIN_SEEDS) + ".",
        f"The measurement input is {payload['batch']['source']} at "
        f"{payload['batch']['n_tokens']} tokens, file sha256 "
        f"{payload['batch']['file_sha256'][:12]}, token sha256 "
        f"{payload['batch']['token_sha256'][:12]}. The token hash is derived "
        "by re-tokenizing the committed text every run: a tokenizer version "
        "shift leaves the file hash alone and moves the token hash, so drift "
        "is loud rather than silent.",
    ]

    if dev is not None:
        parts.append(
            f"Two copies of one model deviate on L2, CKA and cosine by at most "
            f"{dev['weight_deviation']:.3e}."
            if dev["weight_deviation"] < 1e-6 else
            f"THE IDENTICAL PAIR DOES NOT SCORE IDENTICALLY: worst deviation "
            f"{dev['weight_deviation']:.3e} on L2/CKA/cosine. Two copies of one "
            f"model must be indistinguishable to all three; this is a defect, "
            f"not a result.")
        parts.append(
            f"Its barrier reads {dev['barrier_noise']:.3e} rather than exactly "
            "zero, and that is arithmetic rather than a barrier: the "
            "interpolation evaluates (1 - a) * w + a * w, which is not bitwise "
            "w in floating point for most a, so interior alphas run a model a "
            "few ulps from the endpoints. It is the floor below which no "
            "barrier measured on this grid is meaningful.")

    floor = barrier_noise_floor(payload)
    above = barriers_above_noise(payload)
    cleared = {k: v for k, v in above.items()
               if k != "identical" and v["n_above_noise_floor"]}
    parts.append(
        "THE BARRIER IS NOT DEMONSTRATED BY THIS FILE, AND CANNOT BE. Junk "
        "checkpoints sit at chance loss along the entire interpolation, so "
        "there is mostly nothing to climb over: the curve sags below the chord "
        "instead of rising above it, and max_excess is 0 by definition when "
        "the maximum falls at an endpoint. "
        + (f"Rises are counted against the identical pair's floor of "
           f"{floor:.3e}, not against zero, because interpolation float noise "
           f"alone lifts two copies of one model above the chord on some "
           f"seeds. " if floor is not None else "")
        + (("Above that floor: "
            + "; ".join(f"{k} on {v['n_above_noise_floor']} of {v['n_seeds']} "
                        f"seeds, largest {v['max_excess']:.3g}"
                        for k, v in sorted(cleared.items())) + ".")
           if cleared else
           "No pair cleared that floor on any seed.")
        + " That is the expected outcome for untrained weights and says "
          "nothing about whether the metric would find a real barrier between "
          "two trained checkpoints. What is established is that the "
          "interpolation is evaluated and that the arithmetic finds a peak when "
          "one is present -- the latter tested against a synthetic curve in "
          "tests/test_metrics.py, not by any number in this file.")

    if resp:
        worst = min(
            (v["lowest_layer_cka_median"] for v in resp.values()
             if v["lowest_layer_cka_median"] is not None), default=None)
        if worst is not None:
            parts.append(
                f"Responsiveness: the most similar any differing pair gets at "
                f"its least-affected layer is CKA {worst:.4f}, against 1.0 for "
                f"two copies of one model.")

    parts.append(
        "READ THE RANGE, NOT THE MEDIAN. Every cell stores its per-seed values "
        "and reports min/median/max over the union of chunks.")
    return " ".join(parts)


def report_banner(payload) -> list:
    coverage = seed_coverage(payload)
    seeded = {k: v for k, v in coverage.items() if v}
    floor_ok = seeded and all(v >= MIN_SEEDS for v in seeded.values())
    dev = _identical_pair_deviation(payload)
    batch = payload["batch"]

    lines = [
        "JUNK CHECKPOINTS. THIS IS A PLUMBING RESULT, NOT A MEASUREMENT."
        if floor_ok else "*** A PAIR IS BELOW THE TEN-SEED FLOOR ***",
        "",
        "No trained model exists in this study. Every number below comes",
        "from throwaway models built in process from a pinned seed. What is",
        "established is that each metric RESPONDS -- not anything about",
        "training.",
        "",
        f"CKA variant: {payload['cka_variant']}",
        f"Input batch: {batch['source']}, {batch['n_tokens']} tokens, "
        f"file {batch['file_sha256'][:12]}, tokens {batch['token_sha256'][:12]}",
        "",
        "Seed coverage, read from the cells and not from the top-level",
        "'seeds' key (which is only the last chunk's window):",
        "  " + "   ".join(f"{k}={v}" for k, v in sorted(seeded.items())),
    ]

    if dev is not None:
        lines += [
            "",
            f"Two copies of one model deviate by at most "
            f"{dev['weight_deviation']:.3e} on L2, CKA and cosine."
            if dev["weight_deviation"] < 1e-6 else
            f"*** THE IDENTICAL PAIR DEVIATES BY "
            f"{dev['weight_deviation']:.3e} -- A DEFECT ***",
            f"Its barrier reads {dev['barrier_noise']:.3e}: interpolation float",
            "noise, not a barrier. No barrier below this floor is meaningful."]

    lines += [
        "",
        "THE BARRIER IS NOT DEMONSTRATED HERE AND CANNOT BE. Junk weights sit",
        "at chance loss along the whole interpolation, so the curve sags below",
        "the chord rather than rising above it and max_excess is 0. That is",
        "the expected outcome for untrained weights. Whether the metric would",
        "find a real barrier is tested against a synthetic curve with a peak",
        "in it, in tests/test_metrics.py -- not by this file.",
        "",
        "NOT BUILT: aligned barrier, aligned L2, RSF subspace probe. All three",
        "need canonicalize, which is Conv1D-only while the study's layout is",
        "undecided. This is HALF of step 10.",
    ]
    return lines


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _cell(label, cell, width=24):
    return (f"{label:<{width}}{cell['n_seeds']:>4}{cell['min']:>15.6g}"
            f"{cell['median']:>15.6g}{cell['max']:>15.6g}")


_CELL_HEADER = (f"{'quantity':<24}{'n':>4}{'min':>15}{'median':>15}{'max':>15}")


def format_report(payload) -> str:
    rule = "=" * 78
    thin = "-" * 78
    lines = ["", rule] + report_banner(payload) + [rule, ""]
    lines += [rule, "STEP 10 (FIRST HALF) -- LAYOUT-INDEPENDENT METRICS", rule,
              f"model: {payload['model']}", ""]
    floor = barrier_noise_floor(payload)
    above = barriers_above_noise(payload)

    for kind, section in payload["junk_pairs"].items():
        lines += [rule, f"PAIR: {kind}", rule,
                  f"  expected: {section['expectation']}", "",
                  _CELL_HEADER,
                  _cell("l2_raw", section["l2_raw"]),
                  _cell("barrier_max_excess", section["barrier_max_excess"]),
                  _cell("barrier_min_excess", section["barrier_min_excess"]),
                  f"  rose above chord: "
                  f"{above.get(kind, {}).get('n_rose_above_chord', 0)} of "
                  f"{above.get(kind, {}).get('n_seeds', 0)} seeds"
                  + (f", of which "
                     f"{above.get(kind, {}).get('n_above_noise_floor', 0)} "
                     f"cleared the {floor:.2e} float-noise floor"
                     if floor is not None else ""),
                  "", thin, "  per-layer CKA and cosine", thin,
                  f"{'layer':>6}{'cka min':>13}{'cka med':>13}{'cka max':>13}"
                  f"{'cos med':>13}{'norm ratio':>13}"]
        for layer in sorted(section["cka"], key=int):
            cka = section["cka"][layer]
            cos = section["cosine"][layer]
            norm = section["norm_ratio"][layer]
            lines.append(
                f"{layer:>6}{cka['min']:>13.6f}{cka['median']:>13.6f}"
                f"{cka['max']:>13.6f}{cos['median']:>13.6f}"
                f"{norm['median']:>13.6f}")
        outside = [l for l, c in section["cka"].items()
                   if c["min"] < 0.0 or c["max"] > 1.0]
        if outside:
            lines += ["",
                      f"  CKA outside [0,1] at layers {sorted(outside, key=int)}.",
                      "  NOT CLIPPED, on purpose. The unbiased HSIC estimator "
                      "can return a slightly",
                      "  negative value when the true dependence is at or below "
                      "its noise floor.",
                      "  Clipping to 1.0 would turn a diagnostic into a "
                      "plausible float that looks",
                      "  like a working metric."]
        lines.append("")

    xc = payload["route_cross_check"]
    lines += [rule, "ACTIVATION ROUTE CROSS-CHECK", rule,
              f"  hooks vs output_hidden_states, {xc['n_seeds']} seeds",
              f"  worst absolute gap: {xc['worst_abs_gap']:.3e}",
              f"  all agree: {xc['all_agree']}", ""]
    for chunk in xc["note"].split(". "):
        if chunk.strip():
            lines.append("  " + chunk.strip() + ".")

    timing = payload.get("timing")
    if timing:
        lines += ["", rule, "TIMING -- measured before the run, not estimated",
                  rule]
        for key, value in timing.items():
            lines.append(f"  {key:<34}{value}")

    for title, key in (("WHAT THIS FILE CANNOT ANSWER", "CANNOT_ANSWER"),
                       ("LIMITATION", "LIMITATION"),
                       ("PROVENANCE", "PROVENANCE")):
        lines += ["", rule, title, rule]
        for chunk in payload[key].split(". "):
            if chunk.strip():
                lines.append("  " + chunk.strip().rstrip(".") + ".")
    return "\n".join(lines)


def _write_report(payload, reportdir: Path, stem: str, stream) -> int:
    """Recompute every DERIVED field, then write both files."""
    payload["seed_coverage"] = seed_coverage(payload)
    payload["responsiveness"] = _responsiveness(payload)
    payload["barrier_noise_floor"] = barrier_noise_floor(payload)
    payload["barriers_above_noise"] = barriers_above_noise(payload)
    payload["CANNOT_ANSWER"] = cannot_answer(payload)
    payload["LIMITATION"] = limitation(payload)
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/metrics_report.py",
        description="Measure the step 10 metrics on junk checkpoints. Trains "
                    "nothing and touches no arm.")
    parser.add_argument("--reportdir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--tiny", action="store_true",
                        help="run against the in-process tiny model, for a "
                             "fast smoke test rather than a real measurement")
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    parser.add_argument(
        "--seed-window", default=None, metavar="A:B",
        help=("run only seeds[A:B] and MERGE the per-seed results into what is "
              "already in the file. Ten seeds is mandatory and may not fit a "
              "single run under the observed task-duration cap."))
    parser.add_argument(
        "--pairs", default=",".join(MX.JUNK_PAIRS),
        help="which junk pairs to measure. A subset MERGES into the existing "
             "file rather than replacing it.")
    parser.add_argument(
        "--alphas", type=int, default=len(MX.ALPHA_GRID),
        help="number of interpolation points, endpoints included")
    parser.add_argument(
        "--render-only", action="store_true",
        help=("re-render the .md and the DERIVED json fields from the "
              "measurements already on disk, measuring nothing"))
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    stream = sys.stdout
    rule = "=" * 78
    print(rule, file=stream)
    print("metrics_report -- step 10, first half", file=stream)
    print(rule, file=stream)

    stem = REPORT_STEM + ("-tiny" if args.tiny else "")
    reportdir = Path(args.reportdir)

    if args.render_only:
        source = reportdir / f"{stem}.json"
        if not source.is_file():
            print(f"nothing to render: {source} does not exist", file=stream)
            return 1
        print(f"render-only: re-rendering {source}, measuring nothing",
              file=stream)
        return _write_report(
            json.loads(source.read_text(encoding="utf-8")), reportdir, stem,
            stream)

    import time

    seeds = SEEDS[:args.seeds]
    if args.seed_window:
        a, b = (int(x) if x else None for x in args.seed_window.split(":"))
        seeds = seeds[a:b]
    wanted = [p.strip() for p in args.pairs.split(",") if p.strip()]
    for kind in wanted:
        if kind not in MX.JUNK_PAIRS:
            raise SystemExit(f"unknown pair {kind!r}; expected {MX.JUNK_PAIRS}")

    if args.alphas < 2:
        raise SystemExit("--alphas must be at least 2 (both endpoints)")
    alphas = tuple(round(i / (args.alphas - 1), 6) for i in range(args.alphas))

    existing = {}
    prior = reportdir / f"{stem}.json"
    if prior.is_file():
        existing = json.loads(prior.read_text(encoding="utf-8"))
        print(f"merging {wanted} at seeds {list(seeds)} into {prior}",
              file=stream)

    if args.tiny:
        # The tiny model has a 64-token vocabulary and 32 positions; the real
        # batch is 1024 GPT-2 tokens and does not fit it. The batch is part of
        # the measurement, so the smoke path gets its own and labels itself.
        print("\nSMOKE TEST: using the byte-derived tiny batch, not the "
              "committed measurement input", file=stream)
        batch = MX.tiny_smoke_batch()
    else:
        print("\nloading the committed measurement batch", file=stream)
        batch = MX.load_context_batch(stream=stream)
    print(f"  {batch.source}: {batch.n_tokens} tokens, "
          f"token sha256 {batch.token_sha256[:12]}", file=stream)

    started = time.perf_counter()
    print("\nactivation route cross-check", file=stream)
    cross = measure_route_cross_check(
        seeds, batch, args.tiny, stream,
        prior=existing.get("route_cross_check"))

    print("\nmeasuring junk pairs", file=stream)
    pairs = dict(existing.get("junk_pairs") or {})
    for kind in wanted:
        pairs[kind] = measure_pair(
            kind, seeds, batch, alphas, args.tiny, stream,
            existing=(existing.get("junk_pairs") or {}).get(kind))
    elapsed = time.perf_counter() - started

    payload = {
        "task": "step 10 first half -- layout-independent metrics",
        "model": ("TINY in-process smoke test" if args.tiny
                  else "GPT-2 124M shapes, JUNK weights (never trained)"),
        "cka_variant": MX.CKA_VARIANT,
        "batch": batch.identity(),
        "seeds": list(seeds),
        "alpha_grid": list(alphas),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "junk_pairs": pairs,
        "route_cross_check": cross,
        "timing": {
            "seconds_this_run": round(elapsed, 2),
            "seeds_this_run": len(seeds),
            "pairs_this_run": len(wanted),
            "alpha_grid_points": len(alphas),
            "seconds_per_pair_seed": round(
                elapsed / max(1, len(seeds) * len(wanted)), 2),
        },
        "pairs_refreshed_this_run": sorted(wanted),
    }
    return _write_report(payload, reportdir, stem, stream)


if __name__ == "__main__":
    raise SystemExit(main())
