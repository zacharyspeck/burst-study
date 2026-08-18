#!/usr/bin/env python
"""Sweep burst position and diagnose which quantities discriminate. 8b-ii.

    python scripts/position_sweep.py
    python scripts/position_sweep.py --positions 1 100 200 400 600 830

WHY THIS EXISTS. 8b-i measured every arm at position 400, chosen arbitrarily,
and found that burst-region loss spread 2.32x while the gradient norm taken
from the full-sequence loss spread only 1.09x -- against a filler-only floor
that one arm fell below. A quantity that cannot tell a burst from no burst
cannot serve as a matching criterion. Two things were untested: whether the
gradient compresses because it is taken from the wrong loss, and whether
position mattered at all.

So this sweeps both. FOUR NUMBERS PER ARM PER POSITION, named so they cannot
be confused:

    loss_burst_region                  mean NLL over the 194 burst predictions
    loss_full_sequence                 mean NLL over all 1023 predictions
    gradnorm_from_burst_region_loss    L2 norm of the gradient of the first
    gradnorm_from_full_sequence_loss   L2 norm of the gradient of the second

THE CONTROL. At every position a no-burst row measures the SAME 194-token
window holding filler instead of a burst. It is not an arm. Without it a
burst-region number has no position-matched floor to sit against.

A precision point: the burst-region gradient is NOT filler-free. The burst's
predictions are conditioned on preceding filler, so gradients still flow
through filler activations. What it excludes is the filler's own prediction
errors from the differentiated quantity.

This script reports. It sets no tolerance, recommends no position, and draws
no conclusions.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from burst_match import (  # noqa: E402
    DEFAULT_BASE_CONFIG,
    RULE,
    THIN,
    BurstMatchError,
    load_model,
    measure_filler_region,
    measure_in_context,
    resolve_batch_size,
    resolve_seq_len,
)
from make_bursts import (  # noqa: E402
    ARM_SPECS,
    CONTEXT_NAME,
    DEFAULT_OUTDIR,
    PROVENANCE_NAME,
    MakeBurstsError,
    arm_by_name,
    sha256_file,
)

DEFAULT_REPORT_DIR = REPO_ROOT / "docs" / "measurements"
REPORT_STEM = "8b-ii-position-sweep"

#: Both ends are edge cases on purpose. At 1 the burst has a single token of
#: context; at 830 it runs to the end with no filler after it.
DEFAULT_POSITIONS: tuple[int, ...] = (1, 100, 200, 400, 600, 830)

#: The four-cell grid: structure (fluent / scrambled) x truth (false / true).
GRID = (("fluent-fabricated", "fluent-attested"), ("scrambled-false", "scrambled-true"))

QUANTITIES = (
    ("loss_burst_region", "burst-region loss", "region_loss"),
    ("loss_full_sequence", "full-sequence loss", "full_sequence_loss"),
    ("gradnorm_from_burst_region_loss", "gradnorm from burst-region loss",
     "gradnorm_from_region_loss"),
    ("gradnorm_from_full_sequence_loss", "gradnorm from full-sequence loss",
     "gradnorm_from_full_sequence_loss"),
)


def spread(values: dict) -> dict:
    lo_name = min(values, key=values.get)
    hi_name = max(values, key=values.get)
    lo, hi = values[lo_name], values[hi_name]
    return {"min": lo, "min_arm": lo_name, "max": hi, "max_arm": hi_name,
            "absolute": hi - lo, "ratio": (hi / lo) if lo else None}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/position_sweep.py",
        description="Sweep burst position, reporting both loss figures and "
                    "both gradient variants per arm. Recommends nothing.")
    parser.add_argument("--positions", type=int, nargs="+",
                        default=list(DEFAULT_POSITIONS), metavar="N")
    parser.add_argument("--burstdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--reportdir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(RULE)
    print("position_sweep -- 8b-ii gradient diagnostic")
    print(RULE)
    try:
        return _run(args)
    except (BurstMatchError, MakeBurstsError) as exc:
        print(f"\nERROR\n{exc}\n", file=sys.stderr)
        return 1


def _run(args) -> int:
    burstdir = Path(args.burstdir)
    batch = resolve_batch_size(args.batch_size, args.base_config)
    seq = resolve_seq_len(args.seq_len, args.base_config)
    context_path = burstdir / CONTEXT_NAME
    if not context_path.is_file():
        raise MakeBurstsError(
            f"{context_path} does not exist. Run scripts/make_bursts.py first.")

    tokenizer, model = load_model()
    print()

    context_ids = tokenizer(context_path.read_text(encoding="utf-8"),
                            add_special_tokens=False)["input_ids"]
    arm_ids = {}
    for spec in ARM_SPECS:
        path = burstdir / spec.filename
        if not path.is_file():
            raise MakeBurstsError(f"{path} does not exist.")
        arm_ids[spec.name] = tokenizer(path.read_text(encoding="utf-8"),
                                       add_special_tokens=False)["input_ids"]

    lengths = {n: len(i) for n, i in arm_ids.items()}
    if len(set(lengths.values())) != 1:
        raise BurstMatchError(f"arms differ in length: {lengths}")
    n_burst = next(iter(lengths.values()))
    filler_ids = context_ids[:seq.value - n_burst]

    positions = sorted(set(args.positions))
    print(f"arms:      {len(ARM_SPECS)} at {n_burst} tokens each")
    print(f"positions: {positions}  (valid range 1..{len(filler_ids)})")
    print(f"gradients: 2 per row -- from the burst-region loss and from the "
          "full-sequence loss")
    print(f"rows:      {(len(ARM_SPECS) + 1) * len(positions)} measurements, "
          f"{2 * (len(ARM_SPECS) + 1) * len(positions)} backward passes")
    print()

    results: dict = {}
    for position in positions:
        print(f"position {position:>4}:", end=" ", flush=True)
        per_arm = {}
        for spec in ARM_SPECS:
            print(".", end="", flush=True)
            per_arm[spec.name] = measure_in_context(
                arm_ids[spec.name], filler_ids, position, tokenizer, model,
                spec.name, batch_size=batch.value, train_seq_len=seq.value)
        print("+", end="", flush=True)
        control = measure_filler_region(
            context_ids, position, n_burst, tokenizer, model,
            batch_size=batch.value, train_seq_len=seq.value)
        results[position] = (per_arm, control)
        print(" done", flush=True)

    text = format_report(results, positions, n_burst, batch, seq, context_path)
    print(text)

    if args.no_write:
        return 0

    reportdir = Path(args.reportdir)
    reportdir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(results, positions, n_burst, batch, seq,
                            context_path, burstdir)
    (reportdir / f"{REPORT_STEM}.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    (reportdir / f"{REPORT_STEM}.md").write_text(
        text + "\n", encoding="utf-8", newline="\n")
    print()
    print(f"wrote {reportdir / (REPORT_STEM + '.json')}")
    print(f"wrote {reportdir / (REPORT_STEM + '.md')}")
    return 0


def _get(measurement, attr):
    return getattr(measurement, attr)


def source_deltas(per_arm) -> dict:
    """Burst-region loss of each derived arm minus that of its source.

    The derives_from pairs are the only place in the study where the SAME
    WORDS appear at two structure levels, so this delta isolates what
    scrambling alone costs -- topic, vocabulary and length are all held
    constant by construction.
    """
    out = {}
    for spec in ARM_SPECS:
        if not spec.derives_from:
            continue
        arm, src = per_arm[spec.name], per_arm[spec.derives_from]
        out[spec.name] = {
            "source": spec.derives_from,
            "arm_loss": arm.region_loss,
            "source_loss": src.region_loss,
            "delta": arm.region_loss - src.region_loss,
            "arm_gradnorm_from_region": arm.gradnorm_from_region_loss,
            "source_gradnorm_from_region": src.gradnorm_from_region_loss,
            "gradnorm_delta": (arm.gradnorm_from_region_loss
                               - src.gradnorm_from_region_loss),
        }
    return out


def grid_cells(per_arm) -> dict:
    """The four-cell structure x truth grid, as raw differences."""
    ff, ft = per_arm["fluent-fabricated"], per_arm["fluent-attested"]
    sf, st = per_arm["scrambled-false"], per_arm["scrambled-true"]
    return {
        "cells": {n: per_arm[n].region_loss
                  for n in ("fluent-fabricated", "fluent-attested",
                            "scrambled-false", "scrambled-true")},
        "truth_gap_fluent": ft.region_loss - ff.region_loss,
        "truth_gap_scrambled": st.region_loss - sf.region_loss,
        "structure_gap_false": sf.region_loss - ff.region_loss,
        "structure_gap_true": st.region_loss - ft.region_loss,
        "truth_gap_difference": ((st.region_loss - sf.region_loss)
                                 - (ft.region_loss - ff.region_loss)),
    }


def format_report(results, positions, n_burst, batch, seq, context_path) -> str:
    lines = [
        "",
        RULE,
        "8b-ii POSITION SWEEP AND GRADIENT DIAGNOSTIC",
        RULE,
        f"{len(ARM_SPECS)} arms at {n_burst} tokens | sequence {seq.value} | "
        f"batch {batch.value}",
        f"filler identical in every arm and at every position "
        f"({context_path.name})",
        "",
        "loss[b194]  mean NLL over the 194 burst-token predictions",
        "loss[s1024] mean NLL over all 1023 predictions in the sequence",
        "grad[b194]  L2 norm of the gradient OF loss[b194]",
        "grad[s1024] L2 norm of the gradient OF loss[s1024]",
    ]

    for position in positions:
        per_arm, control = results[position]
        lines += [
            "",
            RULE,
            f"POSITION {position}",
            RULE,
            f"{'arm':<18}{'loss[b194]':>12}{'loss[s1024]':>13}"
            f"{'grad[b194]':>12}{'grad[s1024]':>13}",
            THIN,
        ]
        for spec in ARM_SPECS:
            m = per_arm[spec.name]
            lines.append(
                f"{spec.name:<18}{m.region_loss:>12.4f}"
                f"{m.full_sequence_loss:>13.4f}"
                f"{m.gradnorm_from_region_loss:>12.4f}"
                f"{m.gradnorm_from_full_sequence_loss:>13.4f}")
        lines += [
            THIN,
            f"{'no-burst [ctrl]':<18}{control.region_loss:>12.4f}"
            f"{control.full_sequence_loss:>13.4f}"
            f"{control.gradnorm_from_region_loss:>12.4f}"
            f"{control.gradnorm_from_full_sequence_loss:>13.4f}",
            "  ^ same window, same position, filler instead of a burst. "
            "NOT an arm.",
            "",
            f"{'quantity':<34}{'max/min':>10}{'min vs ctrl':>13}"
            f"{'max vs ctrl':>13}",
        ]
        for _, label, attr in QUANTITIES:
            vals = {s.name: _get(per_arm[s.name], attr) for s in ARM_SPECS}
            sp = spread(vals)
            floor = _get(control, attr)
            lines.append(
                f"{label:<34}{sp['ratio']:>10.4f}"
                f"{(sp['min'] / floor - 1) * 100:>+12.1f}%"
                f"{(sp['max'] / floor - 1) * 100:>+12.1f}%")

        # --- derived-arm deltas -------------------------------------------
        deltas = source_deltas(per_arm)
        lines += ["", "  scrambling cost, same words at two structure levels:"]
        for arm, d in deltas.items():
            lines.append(
                f"    {arm:<17} - {d['source']:<14} "
                f"loss {d['delta']:>+8.4f}   grad[b194] {d['gradnorm_delta']:>+8.4f}")

        # --- the four-cell grid -------------------------------------------
        g = grid_cells(per_arm)
        lines += [
            "",
            "  four-cell grid, burst-region loss:",
            f"{'':>16}{'false':>12}{'true':>12}{'truth gap':>12}",
            f"    {'fluent':<12}{g['cells']['fluent-fabricated']:>12.4f}"
            f"{g['cells']['fluent-attested']:>12.4f}"
            f"{g['truth_gap_fluent']:>+12.4f}",
            f"    {'scrambled':<12}{g['cells']['scrambled-false']:>12.4f}"
            f"{g['cells']['scrambled-true']:>12.4f}"
            f"{g['truth_gap_scrambled']:>+12.4f}",
            f"    {'structure gap':<12}{g['structure_gap_false']:>+12.4f}"
            f"{g['structure_gap_true']:>+12.4f}"
            f"{g['truth_gap_difference']:>+12.4f}",
            "    (bottom-right is truth_gap_scrambled - truth_gap_fluent)",
        ]

    # --- summary across positions -----------------------------------------
    lines += ["", RULE, "SUMMARY: max/min across arms, by quantity and position",
              RULE,
              f"{'quantity':<34}" + "".join(f"{p:>9}" for p in positions)]
    for _, label, attr in QUANTITIES:
        row = []
        for position in positions:
            per_arm, _ = results[position]
            vals = {s.name: _get(per_arm[s.name], attr) for s in ARM_SPECS}
            row.append(f"{spread(vals)['ratio']:>9.3f}")
        lines.append(f"{label:<34}" + "".join(row))

    lines += ["",
              f"{'quantity':<34}" + "".join(f"{p:>9}" for p in positions),
              "  arm range as % of the no-burst control at that position:"]
    for _, label, attr in QUANTITIES:
        row = []
        for position in positions:
            per_arm, control = results[position]
            vals = {s.name: _get(per_arm[s.name], attr) for s in ARM_SPECS}
            sp, floor = spread(vals), _get(control, attr)
            row.append(f"{(sp['max'] - sp['min']) / floor * 100:>8.1f}%")
        lines.append(f"{label:<34}" + "".join(row))

    lines += [
        "",
        RULE,
        "WHAT THIS DOES NOT SAY",
        RULE,
        "1. No tolerance is applied, no position is recommended, and no",
        "   quantity is declared usable. Those are not this script's calls.",
        "2. The burst-region gradient is NOT filler-free. The burst's",
        "   predictions are conditioned on preceding filler, so gradients",
        "   still flow through filler activations. What it excludes is the",
        "   filler's own prediction errors from the differentiated quantity.",
        "3. Fully-trained public GPT-2. The burst is injected into a",
        "   from-scratch model at step 200 with ~52M tokens of training and",
        "   only crude statistical structure. Arms matched here are NOT",
        "   guaranteed to be matched there, and there is where the weights",
        "   actually move. A proxy until re-verified on a real checkpoint.",
    ]
    return "\n".join(lines)


def build_payload(results, positions, n_burst, batch, seq, context_path,
                  burstdir) -> dict:
    per_position = {}
    for position in positions:
        per_arm, control = results[position]
        quantities = {}
        for key, _, attr in QUANTITIES:
            vals = {s.name: _get(per_arm[s.name], attr) for s in ARM_SPECS}
            sp = spread(vals)
            floor = _get(control, attr)
            quantities[key] = {
                "per_arm": vals,
                "control": floor,
                "spread_ratio": sp["ratio"],
                "spread_absolute": sp["absolute"],
                "min_arm": sp["min_arm"], "max_arm": sp["max_arm"],
                "min_vs_control_pct": (sp["min"] / floor - 1) * 100,
                "max_vs_control_pct": (sp["max"] / floor - 1) * 100,
            }
        per_position[str(position)] = {
            "quantities": quantities,
            "source_relative": source_deltas(per_arm),
            "grid": grid_cells(per_arm),
        }
    return {
        "spec": "v4", "task": "8b-ii",
        "what_this_measures": {
            "loss_burst_region": "mean NLL over the 194 burst-token predictions",
            "loss_full_sequence": "mean NLL over all 1023 predictions",
            "gradnorm_from_burst_region_loss":
                "L2 norm of the gradient of loss_burst_region",
            "gradnorm_from_full_sequence_loss":
                "L2 norm of the gradient of loss_full_sequence",
            "control": ("same 194-token window at the same position holding "
                        "filler instead of a burst; NOT an arm"),
        },
        "arms": [s.name for s in ARM_SPECS],
        "burst_tokens": n_burst,
        "sequence_tokens": seq.value,
        "batch_size": batch.value,
        "positions": positions,
        "model": "gpt2 (public, fully trained) -- a PROXY, see limitations",
        "context_file": {"name": CONTEXT_NAME,
                         "sha256": sha256_file(context_path)},
        "provenance_sha256": sha256_file(burstdir / PROVENANCE_NAME),
        "environment": {"python": sys.version.split()[0],
                        "platform": platform.platform()},
        "by_position": per_position,
        "caveats": [
            "No tolerance applied; no position or quantity recommended.",
            "The burst-region gradient is not filler-free: burst predictions "
            "are conditioned on preceding filler, so gradients flow through "
            "filler activations. It excludes the filler's own prediction "
            "errors from the differentiated quantity.",
            "Fully-trained GPT-2 is a proxy for a step-200 from-scratch "
            "model; the match must be re-verified on a real checkpoint.",
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
