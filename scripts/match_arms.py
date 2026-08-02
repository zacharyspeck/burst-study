#!/usr/bin/env python
"""Measure all five spec-v4 arms in context and report the spread. 8b-i.

    python scripts/match_arms.py --position 400

Each arm's 194-token burst is spliced into ONE fixed 1024-token sequence.
Every arm uses byte-identical filler, at the same offset, so the arms differ
only in the burst region.

TWO LOSS FIGURES, AND THEY ARE NOT INTERCHANGEABLE

    burst-region loss    mean over the burst's own 194 token predictions.
                         *** THIS IS THE MATCHED QUANTITY ***
    full-sequence loss   mean over all 1023 predictions in the sequence.
                         Reported as context. NOT matched.

830 of the 1024 tokens are filler shared byte-identically by every arm, so
full-sequence loss is ~81% the same text in all five. Matching on it would let
arms with wildly different burst-level surprise appear matched, which defeats
the purpose of matching at all.

The GRADIENT is taken from the full-sequence loss, because that is what a
training step applies. So the two matched quantities are measured over
different scopes -- loss over 194 tokens, gradient over 1024. That asymmetry
is accepted deliberately and recorded in implementation-notes.md: the quantity
matched on is not the quantity that moves the weights, and both are recorded
rather than pretending they are the same.

THE DIAGNOSTIC ROW. A no-burst row measures the filler alone. It is NOT a
sixth arm. It exists so you can see how much of each arm's full-sequence
number is filler rather than burst -- the floor every arm's gradient norm is
being pulled toward.

This script reports the spread. It does not decide whether the spread passes.
The tolerance is not set here and applying it is not this script's job.
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
    batch_divisor,
    load_model,
    measure_in_context,
    measure_sequence_only,
    resolve_batch_size,
    resolve_seq_len,
)
from make_bursts import (  # noqa: E402
    ARM_SPECS,
    CONTEXT_NAME,
    DEFAULT_OUTDIR,
    PROVENANCE_NAME,
    MakeBurstsError,
    sha256_file,
    token_count,
)

DEFAULT_REPORT_DIR = REPO_ROOT / "docs" / "measurements"
REPORT_STEM = "8b-i-in-context-match"

#: The arm every gap is quoted against. Not a claim that it is correct -- it
#: is the fixed hand-written passage the others were built to match.
REFERENCE_ARM = "fluent-false"


def spread(values: dict) -> dict:
    """Max-min and max/min across arms. The deliverable number."""
    lo_name = min(values, key=values.get)
    hi_name = max(values, key=values.get)
    lo, hi = values[lo_name], values[hi_name]
    return {
        "min": lo, "min_arm": lo_name,
        "max": hi, "max_arm": hi_name,
        "absolute": hi - lo,
        "ratio": (hi / lo) if lo else None,
        "percent_of_min": ((hi - lo) / lo * 100.0) if lo else None,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/match_arms.py",
        description="Measure the five spec-v4 arms in a shared 1024-token "
                    "sequence and report the spread. Applies no tolerance.")
    parser.add_argument(
        "--position", type=int, required=True, metavar="N",
        help=("REQUIRED, no default: 0-indexed token offset where the burst "
              "region starts. Identical for every arm in a run, and recorded "
              "in the report. Must be between 1 and (sequence - burst)"),
    )
    parser.add_argument("--burstdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--reportdir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--batch-size", type=int, default=None, metavar="N")
    parser.add_argument("--seq-len", type=int, default=None, metavar="N")
    parser.add_argument("--no-write", action="store_true",
                        help="print the report but write no files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(RULE)
    print("match_arms -- five arms measured in a shared 1024-token sequence")
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
    print(f"batch:   {batch.value} sequences  ({batch.source})")
    print(f"seq_len: {seq.value} tokens  ({seq.source})")

    context_path = burstdir / CONTEXT_NAME
    if not context_path.is_file():
        raise MakeBurstsError(
            f"{context_path} does not exist. Run scripts/make_bursts.py first.")
    for spec in ARM_SPECS:
        if not (burstdir / spec.filename).is_file():
            raise MakeBurstsError(
                f"{burstdir / spec.filename} does not exist. Run "
                "scripts/make_bursts.py first.")

    tokenizer, model = load_model()
    print()

    # context.txt holds a WHOLE sequence. The arms splice into its leading
    # (seq_len - N) tokens; the no-burst diagnostic uses all of it, so that
    # row is the same length as every arm and its gradient norm is comparable.
    context_ids = tokenizer(context_path.read_text(encoding="utf-8"),
                            add_special_tokens=False)["input_ids"]
    arm_ids = {}
    for spec in ARM_SPECS:
        text = (burstdir / spec.filename).read_text(encoding="utf-8")
        arm_ids[spec.name] = tokenizer(text, add_special_tokens=False)["input_ids"]

    lengths = {name: len(ids) for name, ids in arm_ids.items()}
    if len(set(lengths.values())) != 1:
        raise BurstMatchError(
            f"arms are not all the same token length: {lengths}. Matching is "
            "meaningless across different lengths -- loss is a mean.")
    n_burst = next(iter(lengths.values()))

    if len(context_ids) != seq.value:
        raise BurstMatchError(
            f"{context_path.name} is {len(context_ids)} tokens but "
            f"training.seq_len is {seq.value}. Regenerate with "
            "scripts/make_bursts.py.")
    filler_ids = context_ids[:seq.value - n_burst]

    position = args.position
    print(f"burst:   {n_burst} tokens at position {position} of {seq.value}")
    print(f"filler:  {len(filler_ids)} tokens, identical in every arm")
    print(f"         {context_path.name} sha256 {sha256_file(context_path)[:16]}...")
    print()

    print("measuring:", end=" ", flush=True)
    measurements = {}
    for spec in ARM_SPECS:
        print(spec.name, end=" ", flush=True)
        measurements[spec.name] = measure_in_context(
            arm_ids[spec.name], filler_ids, position, tokenizer, model,
            spec.name, batch_size=batch.value, train_seq_len=seq.value)

    # Diagnostic row: filler only, no burst. Not an arm.
    print("[no-burst]", end=" ", flush=True)
    baseline = measure_sequence_only(
        context_ids, tokenizer, model, "no-burst (diagnostic)",
        batch_size=batch.value, train_seq_len=seq.value)
    print()

    burst_losses = {n: m.burst_region_loss for n, m in measurements.items()}
    grad_norms = {n: m.full_sequence_grad_norm for n, m in measurements.items()}
    spreads = {
        "burst_region_loss": spread(burst_losses),
        "full_sequence_grad_norm": spread(grad_norms),
    }

    text = format_report(measurements, baseline, spreads, batch, seq, position,
                         context_path)
    print(text)

    if args.no_write:
        return 0

    reportdir = Path(args.reportdir)
    reportdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "spec": "v4",
        "task": "8b-i",
        "matched_on": ["burst_region_loss", "full_sequence_grad_norm"],
        "not_matched_context_only": ["full_sequence_loss"],
        "burst_position": position,
        "burst_tokens": n_burst,
        "sequence_tokens": seq.value,
        "filler_tokens": len(filler_ids),
        "batch_size": batch.value,
        "batch_divisor": batch_divisor(batch.value, seq.value, seq.value),
        "model": "gpt2 (public, fully trained) -- a PROXY, see limitations",
        "context_file": {"name": CONTEXT_NAME,
                         "sha256": sha256_file(context_path)},
        "provenance_sha256": sha256_file(burstdir / PROVENANCE_NAME),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "arms": {
            name: {
                "burst_region_loss_MATCHED": m.burst_region_loss,
                "full_sequence_loss_context_only": m.full_sequence_loss,
                "full_sequence_grad_norm_MATCHED": m.full_sequence_grad_norm,
                "full_sequence_grad_norm_batch_scaled":
                    m.full_sequence_grad_norm_batch_scaled,
                "burst_tokens": m.n_burst_tokens,
                "sequence_tokens": m.n_sequence_tokens,
                "position": m.position,
                "file_sha256": sha256_file(burstdir / spec.filename),
                "derived_seed": _derived_seed_of(burstdir, name),
            }
            for spec, (name, m) in zip(ARM_SPECS, measurements.items())
        },
        "diagnostic_no_burst": {
            "note": "filler only, no burst spliced in. NOT a sixth arm.",
            "full_sequence_loss": baseline.full_sequence_loss,
            "full_sequence_grad_norm": baseline.full_sequence_grad_norm,
        },
        "spread": spreads,
    }
    json_path = reportdir / f"{REPORT_STEM}.json"
    md_path = reportdir / f"{REPORT_STEM}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n",
                         encoding="utf-8", newline="\n")
    md_path.write_text(text + "\n", encoding="utf-8", newline="\n")
    print()
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


def _derived_seed_of(burstdir: Path, arm: str):
    data = json.loads((burstdir / PROVENANCE_NAME).read_text(encoding="utf-8"))
    return data["arms"][arm]["derived_seed"]


def format_report(measurements, baseline, spreads, batch, seq, position,
                  context_path) -> str:
    lines = [
        "",
        RULE,
        "8b-i IN-CONTEXT MATCH REPORT",
        RULE,
        f"burst position {position} | sequence {seq.value} tokens | "
        f"batch {batch.value}",
        f"filler identical in every arm ({context_path.name})",
        "",
        "MATCHED: burst-region loss (194 burst tokens) and gradient norm.",
        "CONTEXT ONLY, NOT MATCHED: full-sequence loss -- 81% shared filler.",
        "",
        f"{'arm':<18}{'burst loss':>12}{'full-seq loss':>15}"
        f"{'grad norm':>13}{'grad/batch':>13}",
        f"{'':<18}{'[MATCHED]':>12}{'[context]':>15}{'[MATCHED]':>13}{'':>13}",
        THIN,
    ]
    for name, m in measurements.items():
        lines.append(
            f"{name:<18}{m.burst_region_loss:>12.6f}"
            f"{m.full_sequence_loss:>15.6f}"
            f"{m.full_sequence_grad_norm:>13.6f}"
            f"{m.full_sequence_grad_norm_batch_scaled:>13.6f}")
    lines += [
        THIN,
        f"{'no-burst [diag]':<18}{'--':>12}{baseline.full_sequence_loss:>15.6f}"
        f"{baseline.full_sequence_grad_norm:>13.6f}"
        f"{baseline.full_sequence_grad_norm_batch_scaled:>13.6f}",
        "  ^ filler alone, no burst. NOT an arm. Shows how much of each",
        "    full-sequence figure is filler rather than burst.",
        "",
        RULE,
        "SPREAD ACROSS THE FIVE ARMS",
        RULE,
    ]
    for key, label in (("burst_region_loss", "burst-region loss  [MATCHED]"),
                       ("full_sequence_grad_norm", "gradient norm      [MATCHED]")):
        s = spreads[key]
        lines += [
            f"{label}",
            f"   min {s['min']:.6f}  ({s['min_arm']})",
            f"   max {s['max']:.6f}  ({s['max_arm']})",
            f"   max - min = {s['absolute']:.6f}",
            f"   max / min = {s['ratio']:.4f}   "
            f"({s['percent_of_min']:+.1f}% of min)",
            "",
        ]
    lines += [
        RULE,
        "WHAT THIS DOES NOT SAY",
        RULE,
        "1. No tolerance is applied here and no row is recommended. Whether",
        "   this spread counts as matched is not this script's decision.",
        "2. These numbers come from fully-trained public GPT-2. The burst is",
        "   injected into a from-scratch model at step 200, which has seen",
        "   ~52M tokens and has only crude statistical structure. Arms matched",
        "   here are NOT guaranteed to be matched on that model, and that",
        "   model is the one whose weights actually move. This is a proxy",
        "   until it can be re-verified against a real step-200 checkpoint.",
        "3. The matched loss covers 194 tokens; the matched gradient covers",
        "   all 1024. They are deliberately different scopes.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
