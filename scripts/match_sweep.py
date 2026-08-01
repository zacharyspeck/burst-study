#!/usr/bin/env python
"""Sweep the noise window size k and report the gap from coherent.

The noise arm is meant to be structurally destroyed but otherwise comparable
to the coherent arm. How destroyed is a dial: shuffling word order inside
windows of k=2 barely disturbs the text, and shuffling the whole span at once
destroys every phrase in it. Somewhere along that dial the noise passage may
deliver the same size shove to the weights as the coherent one. This script
measures the dial. It does not read anything off it.

    python scripts/match_sweep.py
    python scripts/match_sweep.py --k 2 3 5 8 15 30 --seed 0

Each row is one measurement from scripts/burst_match.py -- imported, not
reimplemented, so the sweep cannot drift from the single-passage numbers.

WHAT THIS SCRIPT WILL NOT DO
It will not pick a winner, rank the rows, or say that any k matches. The
tolerance is not set yet, and applying it is not this script's job. It prints
the table and stops. Anything that looked like a recommendation here would
become the decision by default.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Imported, never copied. If the measurement changes, this table changes with
# it, and the brief was explicit that it must not be reimplemented here.
from burst_match import (  # noqa: E402
    RULE,
    THIN,
    BurstMatchError,
    batch_scaled,
    load_model,
    measure_text,
    percent_change,
    resolve_batch_size,
)
from make_bursts import (  # noqa: E402
    COHERENT_NAME,
    DEFAULT_CACHE,
    DEFAULT_DOCS,
    DEFAULT_OUTDIR,
    ORDINARY_NAME,
    SPAN_WORD_MULTIPLIER,
    MakeBurstsError,
    load_corpus_slice,
    select_spans,
    token_count,
    trim_to_tokens,
    window_shuffle,
)

#: The window sizes swept by default. The brief's list, plus a full-span
#: shuffle appended separately as the k = every-word end of the dial.
DEFAULT_KS: tuple[int, ...] = (2, 3, 5, 8, 15, 30)

#: Label for the full-span shuffle, which is not a number.
FULL_SPAN = "full"


def _fmt_gap(value: float | None, width: int, spec: str) -> str:
    """A gap column, or a dash for the row that is the reference."""
    if value is None:
        return "-".rjust(width)
    return format(value, spec).rjust(width)


def format_table(rows, batch) -> str:
    """One table: k, tokens, loss, gradient norm, and the gaps from coherent.

    `rows` is a list of (label, k_display, Measurement, is_reference).
    """
    reference = next((m for _, _, m, is_ref in rows if is_ref), None)
    if reference is None:
        raise BurstMatchError("no coherent row to compare against")

    lines = [
        "",
        RULE,
        "match sweep -- gap from coherent",
        RULE,
        f"reference: coherent, {reference.n_tokens} tokens, "
        f"mean loss {reference.mean_loss:.6f}, "
        f"grad norm {reference.grad_norm:.6f}",
        f"batch:     {batch.value} sequences  ({batch.source})",
        "",
        f"{'passage':<12}{'k':>6}{'tokens':>8}{'loss':>11}{'d loss':>11}"
        f"{'d loss %':>10}{'grad norm':>12}{'d grad':>11}{'d grad %':>10}"
        f"{'grad/batch':>12}",
        THIN,
    ]

    for label, k_display, m, is_ref in rows:
        if is_ref:
            d_loss = d_loss_pct = d_grad = d_grad_pct = None
        else:
            d_loss = m.mean_loss - reference.mean_loss
            d_grad = m.grad_norm - reference.grad_norm
            d_loss_pct = percent_change(reference.mean_loss, m.mean_loss)
            d_grad_pct = percent_change(reference.grad_norm, m.grad_norm)

        lines.append(
            f"{label:<12}{k_display:>6}{m.n_tokens:>8}"
            f"{m.mean_loss:>11.6f}"
            f"{_fmt_gap(d_loss, 11, '+.6f')}"
            f"{_fmt_gap(d_loss_pct, 10, '+.1f')}"
            f"{m.grad_norm:>12.6f}"
            f"{_fmt_gap(d_grad, 11, '+.6f')}"
            f"{_fmt_gap(d_grad_pct, 10, '+.1f')}"
            f"{batch_scaled(m.grad_norm, batch.value):>12.6f}"
        )

    lines += [
        THIN,
        "d columns are (row - coherent). Percentages are of the coherent "
        "value.",
        "grad/batch is the standalone norm divided by the batch size: what one",
        "sequence in a batch actually contributes.",
        "",
        "Every row is the same token length, so the gradient norms are "
        "directly",
        "comparable. No row is recommended here -- the tolerance is not set "
        "and",
        "applying it is not this script's job.",
    ]
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/match_sweep.py",
        description=(
            "Measure coherent, ordinary, and noise at a range of shuffle "
            "window sizes, and print the gap from coherent. Picks no winner."
        ),
    )
    parser.add_argument("--k", type=int, nargs="+", default=list(DEFAULT_KS),
                        metavar="N", help=f"window sizes (default: "
                                          f"{' '.join(map(str, DEFAULT_KS))})")
    parser.add_argument("--seed", type=int, default=0, metavar="N",
                        help="seed for span selection and shuffling (default: 0)")
    parser.add_argument("--burstdir", type=Path, default=DEFAULT_OUTDIR,
                        metavar="PATH",
                        help=f"where the burst files live (default: {DEFAULT_OUTDIR})")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE,
                        metavar="PATH", help="cached corpus slice")
    parser.add_argument("--docs", type=int, default=DEFAULT_DOCS, metavar="N",
                        help=f"documents to read from the cache (default: "
                             f"{DEFAULT_DOCS})")
    parser.add_argument("--batch-size", type=int, default=None, metavar="N",
                        help="override the batch size read from the config")
    parser.add_argument("--base-config", type=Path, default=None, metavar="PATH",
                        help="config to read training.batch_size from")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    print(RULE)
    print("match_sweep -- can noise be matched to coherent, and at what k?")
    print(RULE)

    try:
        return _run(args)
    except (BurstMatchError, MakeBurstsError) as exc:
        print(f"\nERROR\n{exc}\n", file=sys.stderr)
        return 1


def _run(args) -> int:
    from burst_match import DEFAULT_BASE_CONFIG

    base_config = args.base_config or DEFAULT_BASE_CONFIG
    batch = resolve_batch_size(args.batch_size, base_config)
    print(f"batch:   {batch.value} sequences  ({batch.source})")

    burstdir = Path(args.burstdir)
    coherent_path = burstdir / COHERENT_NAME
    ordinary_path = burstdir / ORDINARY_NAME
    for path in (coherent_path, ordinary_path):
        if not path.is_file():
            raise MakeBurstsError(
                f"{path} does not exist. Run scripts/make_bursts.py first."
            )

    tokenizer, model = load_model()
    print()

    coherent_text = coherent_path.read_text(encoding="utf-8")
    n_target = token_count(tokenizer, coherent_text)
    print(f"target:  N = {n_target} tokens")

    # The noise span is selected exactly as make_bursts.py selects it -- same
    # cached documents, same seed, same filters, same call -- so the k values
    # swept here describe the passage that script would produce, not a
    # different one that happens to look similar.
    span_words = int(n_target * SPAN_WORD_MULTIPLIER)
    docs = load_corpus_slice(Path(args.cache), args.docs)
    selection_rng = random.Random(args.seed)
    _, noise_span = select_spans(docs, 2, span_words, selection_rng)
    print(f"noise span: document {noise_span.doc_index}, "
          f"{noise_span.word_count} words")

    ks: list[int | str] = sorted(set(args.k))
    n_words = len(noise_span.text.split())
    ks.append(FULL_SPAN)

    rows = []
    print()
    print("measuring:", end=" ", flush=True)

    print("coherent", end=" ", flush=True)
    rows.append(("coherent", "-", measure_text(
        coherent_text, "coherent", tokenizer, model), True))

    print("ordinary", end=" ", flush=True)
    rows.append(("ordinary", "-", measure_text(
        ordinary_path.read_text(encoding="utf-8"), "ordinary", tokenizer,
        model), False))

    for k in ks:
        window = n_words if k == FULL_SPAN else k
        print(f"noise/{k}", end=" ", flush=True)
        # A fresh rng per k, seeded identically. Without this, the shuffle at
        # k=5 would depend on how many draws k=3 happened to consume, and the
        # rows would not be independent measurements of the same dial.
        shuffled = window_shuffle(noise_span.text, window,
                                  random.Random(args.seed))
        text, _ = trim_to_tokens(tokenizer, shuffled, n_target)
        rows.append(("noise", str(k), measure_text(
            text, f"noise k={k}", tokenizer, model), False))
    print()

    counts = {m.n_tokens for _, _, m, _ in rows}
    if len(counts) != 1:
        raise BurstMatchError(
            f"rows are not all the same token length: {sorted(counts)}. "
            "Gradient norms across different lengths are not comparable."
        )

    print(format_table(rows, batch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
