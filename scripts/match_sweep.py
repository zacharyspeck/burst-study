#!/usr/bin/env python
"""Sweep the scrambled arm's window size k, measured in context.

    python scripts/match_sweep.py --position 400
    python scripts/match_sweep.py --position 400 --k 2 3 5 8 15 30

Under spec v4 the scrambled arm is one of five, and k is its own private
parameter. This sweeps it so you can see how the dial behaves; every other
arm is fixed and is not swept here. To measure all five arms against each
other, use scripts/match_arms.py -- that is the 8b-i deliverable, this is a
tuning tool for one arm.

Measured IN CONTEXT, like everything else in v4: each candidate scrambling is
spliced into the same 1024-token sequence at the same offset. Numbers taken
standing alone are void and are not comparable to these.

Reports burst-region loss (the matched quantity) and gradient norm. Picks no
winner: the tolerance is not set here.
"""

from __future__ import annotations

import argparse
import random
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
    measure_in_context,
    percent_change,
    resolve_batch_size,
    resolve_seq_len,
)
from make_bursts import (  # noqa: E402
    CONTEXT_NAME,
    DEFAULT_CACHE,
    DEFAULT_DOCS,
    DEFAULT_OUTDIR,
    MakeBurstsError,
    arm_by_name,
    arm_spans,
    derived_seed,
    load_corpus_slice,
    span_words_for,
    token_count,
    trim_to_tokens,
    window_shuffle,
)

DEFAULT_KS: tuple[int, ...] = (2, 3, 5, 8, 15, 30)
FULL_SPAN = "full"
SCRAMBLED = "scrambled"
REFERENCE = "fluent-false"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/match_sweep.py",
        description="Sweep the scrambled arm's window size k in context, "
                    "against fluent-false. Picks no winner.")
    parser.add_argument("--position", type=int, required=True, metavar="N",
                        help="REQUIRED: token offset where the burst starts")
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS),
                        metavar="N")
    parser.add_argument("--seed", type=int, default=0, metavar="N")
    parser.add_argument("--burstdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--docs", type=int, default=DEFAULT_DOCS)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    print(RULE)
    print("match_sweep -- scrambled window size k, measured in context")
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
    print(f"batch:   {batch.value} | seq_len: {seq.value}")

    reference_spec = arm_by_name(REFERENCE)
    ref_path = burstdir / reference_spec.filename
    context_path = burstdir / CONTEXT_NAME
    for path in (ref_path, context_path):
        if not path.is_file():
            raise MakeBurstsError(
                f"{path} does not exist. Run scripts/make_bursts.py first.")

    tokenizer, model = load_model()
    print()

    context_ids = tokenizer(context_path.read_text(encoding="utf-8"),
                            add_special_tokens=False)["input_ids"]
    ref_ids = tokenizer(ref_path.read_text(encoding="utf-8"),
                        add_special_tokens=False)["input_ids"]
    n_target = len(ref_ids)
    filler_ids = context_ids[:seq.value - n_target]
    print(f"target:  N = {n_target} tokens at position {args.position}")

    docs = load_corpus_slice(Path(args.cache), args.docs)
    spec = arm_by_name(SCRAMBLED)
    span = arm_spans(docs, args.seed, n_target)[SCRAMBLED]
    raw = span_words_for(spec, span, n_target)
    n_words = len(raw.split())
    print(f"span:    document {span.doc_index}, {n_words} words")

    def measure(ids, label):
        return measure_in_context(ids, filler_ids, args.position, tokenizer,
                                  model, label, batch_size=batch.value,
                                  train_seq_len=seq.value)

    print()
    print("measuring:", end=" ", flush=True)
    print(REFERENCE, end=" ", flush=True)
    rows = [(REFERENCE, "-", measure(ref_ids, REFERENCE), True)]

    for k in sorted(set(args.k)) + [FULL_SPAN]:
        window = n_words if k == FULL_SPAN else k
        print(f"k={k}", end=" ", flush=True)
        # A fresh rng per k, seeded the same way make_bursts seeds this arm,
        # so the rows are independent measurements of one dial rather than a
        # chain where each depends on the draws the previous one consumed.
        rng = random.Random(derived_seed(args.seed, SCRAMBLED))
        text, _ = trim_to_tokens(tokenizer, window_shuffle(raw, window, rng),
                                 n_target, label=f"scrambled k={k}")
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        rows.append((SCRAMBLED, str(k), measure(ids, f"scrambled k={k}"), False))
    print()

    ref = next(m for _, _, m, is_ref in rows if is_ref)
    lines = [
        "",
        RULE,
        "scrambled k sweep, in context",
        RULE,
        f"reference: {REFERENCE}, burst-region loss {ref.burst_region_loss:.6f}, "
        f"grad norm {ref.full_sequence_grad_norm:.6f}",
        "",
        f"{'arm':<12}{'k':>6}{'burst loss':>13}{'d':>11}{'d %':>9}"
        f"{'grad norm':>12}{'d':>11}{'d %':>9}",
        f"{'':<12}{'':>6}{'[MATCHED]':>13}{'':>11}{'':>9}{'[MATCHED]':>12}",
        THIN,
    ]
    for label, k, m, is_ref in rows:
        if is_ref:
            cells = f"{'-':>11}{'-':>9}" + f"{m.full_sequence_grad_norm:>12.6f}" \
                    + f"{'-':>11}{'-':>9}"
            lines.append(f"{label:<12}{k:>6}{m.burst_region_loss:>13.6f}{cells}")
            continue
        dl = m.burst_region_loss - ref.burst_region_loss
        dg = m.full_sequence_grad_norm - ref.full_sequence_grad_norm
        pl = percent_change(ref.burst_region_loss, m.burst_region_loss)
        pg = percent_change(ref.full_sequence_grad_norm, m.full_sequence_grad_norm)
        lines.append(
            f"{label:<12}{k:>6}{m.burst_region_loss:>13.6f}{dl:>+11.6f}"
            f"{pl:>+8.1f}%{m.full_sequence_grad_norm:>12.6f}{dg:>+11.6f}"
            f"{pg:>+8.1f}%")
    lines += [
        THIN,
        "Every row is the same token length and the same filler, so these are",
        "directly comparable. No k is recommended -- the tolerance is not set",
        "and applying it is not this script's job.",
    ]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
